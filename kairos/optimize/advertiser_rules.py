"""Scoped, conditional advertiser rules for per-spot pricing and placement.

Background. The channel keeps a baseline rule per advertiser in
``data/advertiser_rules.csv`` (default premium, allowed positions, allowed
genres, prime-time-only flag). Until now that file was written by the CRUD in
``kairos_api/advertisers.py`` and never read by the engine. This module makes it
real and adds a second, finer store: ``data/advertiser_conditions.csv`` holds
scoped conditional rules (premium multipliers, requirements, forbids) keyed by
advertiser and scoped by position, genre and daypart.

Where the rules take effect. The weekly break-count optimizer
(:mod:`kairos.optimize.optimizer`) decides how many breaks a programme segment
carries; it never attributes a break to an advertiser or a position, so it
cannot consume per-advertiser rules without a larger redesign and this module
does not pretend it can. The daily Wally spot file
(:func:`kairos.data.loaders.load_daily_input`) is where an advertiser, a
campaign and a position genuinely attach to an individual spot. The per-spot
pricing path in :mod:`kairos.export.spots` is the one that honors these rules:
it multiplies a spot's revenue by :meth:`AdvertiserRuleEngine.effective_premium`
and drops or flags any spot that fails :meth:`AdvertiserRuleEngine.is_allowed`.
The weekly break-count optimizer does NOT yet honor advertiser rules.

Honesty rules:

  * An unknown advertiser yields a premium of 1.0 (never zero) and is allowed,
    so a spot for an advertiser with no rule is priced and placed unchanged.
  * Scopes are token sets per dimension. An empty scope or the literal ``ANY``
    matches everything in that dimension. Two scopes intersect on a dimension
    when both are ANY, or their token sets share at least one token.
  * Nothing is invented to fill an empty conditions file: with no conditional
    rules every advertiser keeps only its baseline behaviour.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = ROOT / "data" / "advertiser_rules.csv"
DEFAULT_CONDITIONS_PATH = ROOT / "data" / "advertiser_conditions.csv"

# Effects a conditional rule can carry.
PREMIUM = "premium"
REQUIRE = "require"
FORBID = "forbid"
# PRESSURE is a placement-only lever: it steers WHERE the optimizer wants to place a
# spot (it raises the slot's apparent value) but is NEVER charged, so the real
# revenue total is unchanged. Its value is a percent uplift (10 means +10%). This is
# how an operator says "prefer placing here" without inventing money that is not paid.
PRESSURE = "pressure"
_EFFECTS = (PREMIUM, REQUIRE, FORBID, PRESSURE)

ANY = "ANY"

# How the optimizer/pricing path describes positions inside a break: a 1-based
# integer. A baseline allow_positions list is matched against the string of that
# integer, so "1,2" allows positions 1 and 2 only. The special token GOLD names the
# premium "gold break" (Hebrew: ברייק זהב) so it can be scoped like any position.
GOLD_POSITION = "gold"

# How a PREMIUM rule's value is interpreted. A premium rule turns into a single
# multiplier on the spot's real revenue; the mode decides how its raw ``value``
# becomes that multiplier:
#   * MULTIPLIER (default, the original behaviour): value IS the multiplier, so 1.15
#     means +15%. Every legacy row with no mode column keeps this meaning exactly.
#   * PERCENT: value is a signed percent, so +15 -> 1.15 and -15 -> 0.85.
#   * CPP_ABSOLUTE / CPP_ADD / CPP_DISCOUNT: value is a cost-per-point AMOUNT in the
#     same units as the engine's configured point price (base_cpp). ABSOLUTE sets the
#     spot's CPP to value; ADD raises it by value; DISCOUNT lowers it by value (never
#     below zero). These need base_cpp to convert a price delta into a multiplier; with
#     no base_cpp known they leave the premium unchanged rather than guess.
MULTIPLIER = "multiplier"
PERCENT = "percent"
CPP_ABSOLUTE = "cpp_absolute"
CPP_ADD = "cpp_add"
CPP_DISCOUNT = "cpp_discount"
_PREMIUM_MODES = (MULTIPLIER, PERCENT, CPP_ABSOLUTE, CPP_ADD, CPP_DISCOUNT)
_CPP_MODES = (CPP_ABSOLUTE, CPP_ADD, CPP_DISCOUNT)


def _tokens(raw: object) -> frozenset[str]:
    """Split a comma-joined scope string into a token set.

    An empty value or the literal ``ANY`` (case-insensitive) becomes the empty
    set, which this module reads as "matches everything in this dimension".
    """
    text = str(raw or "").strip()
    if not text or text.upper() == ANY:
        return frozenset()
    return frozenset(part.strip() for part in text.split(",") if part.strip())


def _dimension_matches(scope: frozenset[str], value: Optional[str]) -> bool:
    """True when a single observed ``value`` falls inside a scope token set.

    An empty scope (ANY) matches any value, including a missing one. A non-empty
    scope matches only when ``value`` is one of its tokens.
    """
    if not scope:
        return True
    if value is None:
        return False
    return str(value) in scope


def _scopes_intersect(a: frozenset[str], b: frozenset[str]) -> bool:
    """True when two scope token sets can describe the same value.

    ANY (empty set) intersects everything; otherwise they intersect when they
    share at least one token.
    """
    if not a or not b:
        return True
    return bool(a & b)


def _to_float(raw: object, default: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _to_bool(raw: object) -> bool:
    return str(raw).strip().lower() in {"true", "1", "yes", "y"}


def _normalize_mode(raw: object) -> str:
    """Read a premium ``mode`` cell into one of :data:`_PREMIUM_MODES`.

    An empty or unknown mode falls back to :data:`MULTIPLIER`, the original
    behaviour where ``value`` is the multiplier itself, so every legacy
    conditions row is priced exactly as before this column existed.
    """
    text = str(raw or "").strip().lower()
    return text if text in _PREMIUM_MODES else MULTIPLIER


def _premium_factor(value: float, mode: str, base_cpp: Optional[float]) -> float:
    """Convert one premium rule's (value, mode) into a single revenue multiplier.

    For the price-delta modes the multiplier is ``effective_cpp / base_cpp``, the
    only honest way to express a cost-per-point change as a factor on the engine's
    CPP math. With no positive ``base_cpp`` to convert against, a CPP-mode rule
    returns 1.0 (leaves revenue unchanged) rather than invent a conversion.
    """
    if mode == PERCENT:
        return 1.0 + value / 100.0
    if mode in _CPP_MODES:
        if base_cpp is None or base_cpp <= 0:
            return 1.0
        if mode == CPP_ABSOLUTE:
            effective_cpp = value
        elif mode == CPP_ADD:
            effective_cpp = base_cpp + value
        else:  # CPP_DISCOUNT
            effective_cpp = base_cpp - value
        return max(0.0, effective_cpp) / base_cpp
    # MULTIPLIER (and any unknown mode normalized to it): value is the multiplier.
    return value


@dataclass(frozen=True)
class Baseline:
    """The baseline rule for one advertiser, from advertiser_rules.csv."""

    advertiser_id: str
    default_premium: float = 1.0
    allow_positions: frozenset[str] = frozenset()
    allow_genres: frozenset[str] = frozenset()
    prime_time_only: bool = False

    def allows(self, *, position: Optional[int], genre: Optional[str], daypart: Optional[str]) -> bool:
        """True when a spot passes this advertiser's baseline constraints.

        ANY (empty set) on allow_positions or allow_genres means no limit. When
        ``prime_time_only`` is set, a spot is allowed only in the prime-time
        daypart; a missing daypart cannot satisfy a prime-time-only baseline.
        """
        position_token = None if position is None else str(position)
        if not _dimension_matches(self.allow_positions, position_token):
            return False
        if not _dimension_matches(self.allow_genres, genre):
            return False
        if self.prime_time_only:
            if daypart is None or str(daypart).strip().lower() != "prime":
                return False
        return True


@dataclass(frozen=True)
class Condition:
    """One scoped conditional rule for an advertiser.

    ``effect`` is :data:`PREMIUM`, :data:`REQUIRE`, :data:`FORBID` or
    :data:`PRESSURE`. ``value`` is the premium amount (a multiplier, a percent or a
    cost-per-point amount depending on ``mode``) or the pressure percent, and is
    ignored for require/forbid. ``mode`` is one of :data:`_PREMIUM_MODES` and only
    matters for a premium rule (it says how to read ``value``); it defaults to
    :data:`MULTIPLIER` so a legacy row behaves unchanged. The four scope sets are
    token sets (empty = ANY = matches everything).
    """

    advertiser_id: str
    rule_id: str
    effect: str
    value: float = 1.0
    mode: str = MULTIPLIER
    scope_positions: frozenset[str] = frozenset()
    scope_genres: frozenset[str] = frozenset()
    scope_dayparts: frozenset[str] = frozenset()
    scope_programmes: frozenset[str] = frozenset()
    notes: str = ""

    def matches(
        self,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
    ) -> bool:
        """True when an observed (position, genre, daypart, programme) is in scope.

        Each dimension is an independent token set with ANY/empty meaning "no limit"
        on that dimension. ``programme`` matches a specific show title, so a rule can
        target one programme as precisely as it targets a genre or a daypart.
        """
        position_token = None if position is None else str(position)
        return (
            _dimension_matches(self.scope_positions, position_token)
            and _dimension_matches(self.scope_genres, genre)
            and _dimension_matches(self.scope_dayparts, daypart)
            and _dimension_matches(self.scope_programmes, programme)
        )

    def scope_intersects(self, other: "Condition") -> bool:
        """True when this rule's scope can describe the same spot as ``other``."""
        return (
            _scopes_intersect(self.scope_positions, other.scope_positions)
            and _scopes_intersect(self.scope_genres, other.scope_genres)
            and _scopes_intersect(self.scope_dayparts, other.scope_dayparts)
            and _scopes_intersect(self.scope_programmes, other.scope_programmes)
        )


@dataclass(frozen=True)
class AllowDecision:
    """Whether a spot is allowed, with a human-readable reason for diagnostics."""

    allowed: bool
    reason: str


@dataclass(frozen=True)
class OverlapFinding:
    """One overlap or conflict between two of an advertiser's conditional rules.

    ``kind`` is ``conflict`` for a require/forbid pair whose scopes intersect,
    ``stacked_premium`` for two premium rules whose scopes intersect, and
    ``overlap`` for any other same-effect pair whose scopes intersect.
    """

    advertiser_id: str
    kind: str
    rule_id_a: str
    rule_id_b: str
    detail: str


@dataclass
class AdvertiserRuleEngine:
    """Pure rule engine over the baseline CSV plus the conditions store.

    Built with :meth:`from_files` (the real data) or directly from parsed
    :class:`Baseline` and :class:`Condition` objects (tests). Every public method
    is deterministic and has no hidden constants.
    """

    baselines: dict[str, Baseline] = field(default_factory=dict)
    conditions: dict[str, list[Condition]] = field(default_factory=dict)

    @classmethod
    def from_files(
        cls,
        *,
        rules_path: str | Path | None = None,
        conditions_path: str | Path | None = None,
    ) -> "AdvertiserRuleEngine":
        baselines = _load_baselines(Path(rules_path) if rules_path else DEFAULT_RULES_PATH)
        conditions = _load_conditions(
            Path(conditions_path) if conditions_path else DEFAULT_CONDITIONS_PATH
        )
        return cls(baselines=baselines, conditions=conditions)

    def _conditions_for(self, advertiser_id: str) -> list[Condition]:
        return self.conditions.get(advertiser_id, [])

    def effective_premium(
        self,
        advertiser_id: str,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
        base_cpp: Optional[float] = None,
    ) -> float:
        """The premium multiplier to apply to a spot's REAL revenue.

        It is the advertiser's baseline ``default_premium`` times the product of
        every premium-effect rule whose scope matches the spot, each turned into a
        multiplier by its ``mode`` (percent, plain multiplier, or a cost-per-point
        amount, see :func:`_premium_factor`). An advertiser with no baseline is
        treated as default_premium 1.0 (never zero), so an unknown advertiser
        leaves revenue unchanged. ``base_cpp`` is the engine's configured point
        price, needed only for the cpp_* modes; without it a cpp_* rule leaves the
        premium unchanged rather than guess. Placement-pressure rules are
        deliberately excluded here: they steer placement but are never charged, so
        they must not touch the real revenue (see :meth:`placement_multiplier`).
        """
        baseline = self.baselines.get(advertiser_id)
        premium = baseline.default_premium if baseline is not None else 1.0
        for condition in self._conditions_for(advertiser_id):
            if condition.effect == PREMIUM and condition.matches(
                position=position, genre=genre, daypart=daypart, programme=programme
            ):
                premium *= _premium_factor(condition.value, condition.mode, base_cpp)
        return premium

    def pressure_multiplier(
        self,
        advertiser_id: str,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
    ) -> float:
        """The placement-only multiplier from matching pressure rules (never charged).

        Each matching :data:`PRESSURE` rule contributes ``(1 + value/100)`` (its value
        is a percent uplift). The product is >= 0 and is 1.0 when no pressure rule
        matches. It expresses "prefer placing here" as a virtual premium that the
        optimizer's ranking can see but that never appears in the real revenue.
        """
        multiplier = 1.0
        for condition in self._conditions_for(advertiser_id):
            if condition.effect == PRESSURE and condition.matches(
                position=position, genre=genre, daypart=daypart, programme=programme
            ):
                multiplier *= max(0.0, 1.0 + condition.value / 100.0)
        return multiplier

    def placement_multiplier(
        self,
        advertiser_id: str,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
        base_cpp: Optional[float] = None,
    ) -> float:
        """The value the optimizer should RANK on: real premium times pressure.

        This is ``effective_premium x pressure_multiplier``. The optimizer uses it to
        decide where a spot wants to go (a +10% pressure makes the slot rank as if it
        paid 10% more), while the revenue total uses only :meth:`effective_premium`.
        ``base_cpp`` is forwarded so cpp_* premium modes resolve the same way they do
        when the spot is actually priced. Honest money, biased placement.
        """
        return self.effective_premium(
            advertiser_id, position=position, genre=genre, daypart=daypart,
            programme=programme, base_cpp=base_cpp,
        ) * self.pressure_multiplier(
            advertiser_id, position=position, genre=genre, daypart=daypart, programme=programme
        )

    def allow_decision(
        self,
        advertiser_id: str,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
    ) -> AllowDecision:
        """Whether a spot is allowed, plus a reason string for diagnostics.

        The logic, in order:
          1. The baseline allow_positions / allow_genres / prime_time_only must
             pass (ANY means no limit). An unknown advertiser has no baseline and
             therefore no baseline limit.
          2. No forbid-rule may match the spot. A matching forbid always blocks,
             and overrides any require.
          3. If the advertiser has any require-rules at all, at least one of them
             must match the spot. With no require-rules this step is skipped.
        """
        baseline = self.baselines.get(advertiser_id)
        if baseline is not None and not baseline.allows(
            position=position, genre=genre, daypart=daypart
        ):
            return AllowDecision(False, "blocked by baseline allow_positions/allow_genres/prime_time_only")

        rules = self._conditions_for(advertiser_id)
        for condition in rules:
            if condition.effect == FORBID and condition.matches(
                position=position, genre=genre, daypart=daypart, programme=programme
            ):
                return AllowDecision(False, f"forbidden by rule {condition.rule_id}")

        requires = [c for c in rules if c.effect == REQUIRE]
        if requires:
            matched = next(
                (c for c in requires if c.matches(
                    position=position, genre=genre, daypart=daypart, programme=programme)),
                None,
            )
            if matched is None:
                return AllowDecision(False, "no require-rule scope matches this spot")

        return AllowDecision(True, "allowed")

    def is_allowed(
        self,
        advertiser_id: str,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
    ) -> bool:
        """Boolean shorthand for :meth:`allow_decision`."""
        return self.allow_decision(
            advertiser_id, position=position, genre=genre, daypart=daypart, programme=programme
        ).allowed

    def overlaps(self, advertiser_id: str) -> list[OverlapFinding]:
        """Find conflicting or overlapping conditional rules for an advertiser.

        Powers an operator "what covers what" view. For every unordered pair of
        the advertiser's conditions whose scopes intersect:
          * a require/forbid pair is a ``conflict`` (forbid overrides require, so
            the require can never fire on the intersection),
          * two premium rules are a ``stacked_premium`` (their multipliers stack
            on the intersection),
          * any other same-effect pair is an ``overlap``.
        """
        rules = self._conditions_for(advertiser_id)
        findings: list[OverlapFinding] = []
        for i in range(len(rules)):
            for j in range(i + 1, len(rules)):
                a, b = rules[i], rules[j]
                if not a.scope_intersects(b):
                    continue
                effects = {a.effect, b.effect}
                if effects == {REQUIRE, FORBID}:
                    kind = "conflict"
                    detail = "a require and a forbid cover the same scope; forbid overrides require"
                elif a.effect == PREMIUM and b.effect == PREMIUM:
                    kind = "stacked_premium"
                    detail = "two premium multipliers stack on the same scope"
                elif a.effect == PRESSURE and b.effect == PRESSURE:
                    kind = "stacked_pressure"
                    detail = "two placement-pressure levers stack on the same scope (placement only)"
                else:
                    kind = "overlap"
                    detail = f"two {a.effect} rules cover the same scope"
                findings.append(OverlapFinding(
                    advertiser_id=advertiser_id,
                    kind=kind,
                    rule_id_a=a.rule_id,
                    rule_id_b=b.rule_id,
                    detail=detail,
                ))
        return findings


def _load_baselines(path: Path) -> dict[str, Baseline]:
    """Read advertiser_rules.csv into Baseline objects, keyed by advertiser_id."""
    if not path.exists():
        return {}
    out: dict[str, Baseline] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            advertiser_id = str(row.get("advertiser_id", "")).strip()
            if not advertiser_id:
                continue
            out[advertiser_id] = Baseline(
                advertiser_id=advertiser_id,
                default_premium=_to_float(row.get("default_premium"), 1.0),
                allow_positions=_tokens(row.get("allow_positions")),
                allow_genres=_tokens(row.get("allow_genres")),
                prime_time_only=_to_bool(row.get("prime_time_only")),
            )
    return out


def _load_conditions(path: Path) -> dict[str, list[Condition]]:
    """Read advertiser_conditions.csv into Condition lists, keyed by advertiser.

    Rows whose effect is not premium/require/forbid are skipped, so a malformed
    line never silently changes pricing. An empty file (header only) yields no
    conditions, the honest answer for the seeded state.
    """
    if not path.exists():
        return {}
    out: dict[str, list[Condition]] = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            condition = _condition_from_row(row)
            if condition is not None:
                out.setdefault(condition.advertiser_id, []).append(condition)
    return out


def _condition_from_row(row: dict[str, str]) -> Optional[Condition]:
    advertiser_id = str(row.get("advertiser_id", "")).strip()
    rule_id = str(row.get("rule_id", "")).strip()
    effect = str(row.get("effect", "")).strip().lower()
    if not advertiser_id or not rule_id or effect not in _EFFECTS:
        return None
    return Condition(
        advertiser_id=advertiser_id,
        rule_id=rule_id,
        effect=effect,
        value=_to_float(row.get("value"), 1.0),
        mode=_normalize_mode(row.get("mode")),
        scope_positions=_tokens(row.get("scope_positions")),
        scope_genres=_tokens(row.get("scope_genres")),
        scope_dayparts=_tokens(row.get("scope_dayparts")),
        scope_programmes=_tokens(row.get("scope_programmes")),
        notes=str(row.get("notes", "")),
    )


def normalize_scope(raw: object) -> str:
    """Serialize a scope value back to the canonical comma-or-ANY string.

    Used by the API when persisting a rule, so what is read back matches the
    token semantics here: an empty scope is stored as ``ANY``.
    """
    tokens = _tokens(raw)
    if not tokens:
        return ANY
    return ",".join(sorted(tokens))
