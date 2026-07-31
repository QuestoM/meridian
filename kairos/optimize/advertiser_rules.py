"""Scoped, conditional advertiser rules for per-spot pricing and placement.

Background. The channel keeps a baseline rule per advertiser in
``data/advertiser_rules.csv`` (default premium, allowed positions, allowed
genres, prime-time-only flag). Until now that file was written by the CRUD in
``kairos_api/advertisers.py`` and never read by the engine. This module makes it
real and adds a second, finer store: ``data/advertiser_conditions.csv`` holds
scoped conditional rules (premium multipliers, requirements, forbids) keyed by
advertiser and scoped by position, genre, daypart, programme, campaign and
weekday (ISO 1..7, so a rule can price Saturdays, שבת = 6, differently).

Where the rules take effect. The weekly break-count optimizer
(:mod:`kairos.optimize.optimizer`) decides how many breaks a programme segment
carries; it never attributes a break to an advertiser or a position, so it
cannot consume per-advertiser rules without a larger redesign. The daily Wally
spot file (:func:`kairos.data.loaders.load_daily_input`) is where an advertiser,
a campaign and a position genuinely attach to an individual spot. The per-spot
pricing path in :mod:`kairos.export.spots` is the one that honors these rules:
it multiplies a spot's revenue by :meth:`AdvertiserRuleEngine.effective_premium`
and drops or flags any spot that fails :meth:`AdvertiserRuleEngine.is_allowed`.

Segment-level placement demand. :meth:`AdvertiserRuleEngine.segment_demand`
aggregates pressure and above-baseline premium rules across ALL advertisers for a
segment scope (genre, daypart, programme) into a placement-preference weight
>= 1.0, supplied to :func:`~kairos.optimize.optimizer.optimize_breaks` as
``demand_weights`` to bias WHERE breaks go without changing revenue. Off by default.

Who a rule is about (:meth:`AdvertiserRuleEngine.key_for`). A lookup takes the
advertiser string its caller holds, which on the daily path is the real name the
file carries. A stored key is used as it comes; anything else resolves through
the store's ``name``, ``display_name`` and ``aliases`` columns
(:mod:`kairos.optimize.advertiser_rules_identity`), and a name bound to no row
resolves to nothing and keeps the unknown-advertiser outcome below.

Honesty rules: an unknown advertiser yields a premium of 1.0 (never zero) and is
allowed; scopes are token sets per dimension where an empty scope or ``ANY``
matches everything; a weekday-scoped rule never matches a caller that has no
date (``weekday=None``), so nothing is guessed; nothing is invented to fill an
empty conditions file.

Pure math helpers and CSV loaders live in :mod:`kairos.optimize._rule_helpers`,
and the Baseline/Condition dataclasses in :mod:`kairos.optimize._rule_models`,
to keep this file under the project line limit. Both are re-exported here so
import paths are stable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from kairos.optimize._rule_helpers import (
    CPP_MODES,
    apply_surcharge_discount,
    compute_premium_factor,
    condition_from_row as _condition_from_row_helper,
    dimension_matches,
    load_baselines as _load_baselines,
    load_conditions as _load_conditions,
    parse_bool,
    parse_float,
    parse_mode,
    scope_tokens,
    scopes_intersect,
)
from kairos.optimize.advertiser_rules_identity import NameIndex, load_name_index
from kairos.optimize._rule_models import (  # noqa: F401 - re-exported names
    ANY,
    FORBID,
    GOLD_POSITION,
    PREMIUM,
    PRESSURE,
    REQUIRE,
    _EFFECTS,
    AllowDecision,
    Baseline,
    Condition,
    OverlapFinding,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = ROOT / "data" / "advertiser_rules.csv"
DEFAULT_CONDITIONS_PATH = ROOT / "data" / "advertiser_conditions.csv"

# How a PREMIUM rule's value is interpreted.
#   * MULTIPLIER (default): value IS the multiplier, e.g. 1.15 means +15%.
#   * PERCENT: value is a signed percent, e.g. +15 -> 1.15, -15 -> 0.85.
#   * CPP_ABSOLUTE / CPP_ADD / CPP_DISCOUNT: value is a cost-per-point AMOUNT.
#   * PREMIUM_DISCOUNT: value is a percent 0..100 taken off the premium
#     SURCHARGE only, applied AFTER every other premium mode; it never pushes
#     the premium below 1.0 or above its pre-discount value.
MULTIPLIER = "multiplier"
PERCENT = "percent"
CPP_ABSOLUTE = "cpp_absolute"
CPP_ADD = "cpp_add"
CPP_DISCOUNT = "cpp_discount"
PREMIUM_DISCOUNT = "premium_discount"
_PREMIUM_MODES = (MULTIPLIER, PERCENT, CPP_ABSOLUTE, CPP_ADD, CPP_DISCOUNT, PREMIUM_DISCOUNT)
_CPP_MODES = CPP_MODES

# Private aliases matching the underscore style used in the rest of this file.
_tokens = scope_tokens
_dimension_matches = dimension_matches
_scopes_intersect = scopes_intersect
_to_float = parse_float
_to_bool = parse_bool
_normalize_mode = parse_mode
_premium_factor = compute_premium_factor


@dataclass
class AdvertiserRuleEngine:
    """Pure rule engine over the baseline CSV plus the conditions store.

    Built with :meth:`from_files` (the real data) or directly from parsed
    :class:`Baseline` and :class:`Condition` objects (tests). Every public method
    is deterministic and has no hidden constants.
    """

    baselines: dict[str, Baseline] = field(default_factory=dict)
    conditions: dict[str, list[Condition]] = field(default_factory=dict)
    # Names and aliases the store binds to its rows; empty means keys only.
    names: NameIndex = field(default_factory=NameIndex)

    @classmethod
    def from_files(
        cls,
        *,
        rules_path: str | Path | None = None,
        conditions_path: str | Path | None = None,
    ) -> "AdvertiserRuleEngine":
        rules = Path(rules_path) if rules_path else DEFAULT_RULES_PATH
        baselines = _load_baselines(rules)
        conditions = _load_conditions(
            Path(conditions_path) if conditions_path else DEFAULT_CONDITIONS_PATH
        )
        return cls(baselines=baselines, conditions=conditions, names=load_name_index(rules))

    def key_for(self, advertiser: str) -> str:
        """The stored key an advertiser string addresses (see the module note)."""
        if advertiser in self.baselines or advertiser in self.conditions:
            return advertiser
        return self.names.get(advertiser) or advertiser

    def _conditions_for(self, advertiser_id: str) -> list[Condition]:
        return self.conditions.get(advertiser_id, [])

    def pacing_overrides(self) -> dict[str, tuple[Optional[float], Optional[float]]]:
        """Per-advertiser pacing-strength defaults, keyed by advertiser_id.

        Returns ``{advertiser_id: (urgency_k, ahead_k)}`` for every advertiser whose
        baseline sets at least one of the two; either element may be ``None`` to mean
        "defer to the global default" for that one strength. Advertisers with neither
        set are omitted entirely, so the map is empty (and the pacing signal stays a
        pure identity no-op) until an advertiser is actually given a custom strength.
        This is the middle tier consumed by
        :func:`kairos.optimize.pacing.build_pacing_weights`.
        """
        out: dict[str, tuple[Optional[float], Optional[float]]] = {}
        for advertiser_id, baseline in self.baselines.items():
            if baseline.urgency_k is not None or baseline.ahead_k is not None:
                out[advertiser_id] = (baseline.urgency_k, baseline.ahead_k)
        return out

    def effective_premium(
        self,
        advertiser_id: str,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
        weekday: Optional[int] = None,
        base_cpp: Optional[float] = None,
    ) -> float:
        """The premium multiplier to apply to a spot's REAL revenue.

        Placement-pressure rules are deliberately excluded: they steer placement
        but are never charged, so they must not touch real revenue.

        Modes compose differently. PERCENT, MULTIPLIER, CPP_ADD and CPP_DISCOUNT
        are relative: each matching rule stacks on the running premium. CPP_ABSOLUTE
        is authoritative: it SETS the effective cost-per-point to ``value`` (per the
        contract in docs/advertiser-rules-upgrade.md), so a matching absolute rule
        REPLACES the running premium with ``value / base_cpp`` rather than multiplying
        the baseline by it. Multiple matching absolutes are last-wins (CSV row order);
        a relative rule after an absolute still composes on the absolute's result. An
        absolute with no usable ``base_cpp`` resolves to a 1.0 factor and is therefore
        a no-op, leaving the running premium unchanged rather than collapsing it.

        PREMIUM_DISCOUNT rules compose LAST, whatever their row order: each one
        takes its percent off the composed premium's surcharge (the part above
        1.0), so discounts stack multiplicatively on the surcharge, never push
        the premium below 1.0, and are a no-op when there is no surcharge.

        ``weekday`` is the spot date's ISO weekday (1..7, Monday=1, Saturday=6);
        pass ``None`` when the caller has no date and weekday-scoped rules will
        simply not match.
        """
        advertiser_id = self.key_for(advertiser_id)
        baseline = self.baselines.get(advertiser_id)
        premium = baseline.default_premium if baseline is not None else 1.0
        discounts: list[Condition] = []
        for condition in self._conditions_for(advertiser_id):
            if condition.target_layer:
                continue  # targeted layer/final override: handled by the layered path
            if condition.effect != PREMIUM or not condition.matches(
                position=position, genre=genre, daypart=daypart,
                programme=programme, weekday=weekday,
            ):
                continue
            if condition.mode == PREMIUM_DISCOUNT:
                discounts.append(condition)
                continue
            factor = _premium_factor(condition.value, condition.mode, base_cpp)
            if condition.mode == CPP_ABSOLUTE and not (base_cpp is None or base_cpp <= 0):
                premium = factor  # authoritative: SET the CPP, override prior factors
            else:
                premium *= factor
        for condition in discounts:
            premium = apply_surcharge_discount(premium, condition.value)
        return premium

    def pressure_multiplier(
        self,
        advertiser_id: str,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
        weekday: Optional[int] = None,
    ) -> float:
        """The placement-only multiplier from matching pressure rules (never charged).

        Each matching PRESSURE rule contributes ``(1 + value/100)``. The product
        is >= 0 and is 1.0 when no pressure rule matches.
        """
        multiplier = 1.0
        for condition in self._conditions_for(self.key_for(advertiser_id)):
            if condition.effect == PRESSURE and condition.matches(
                position=position, genre=genre, daypart=daypart,
                programme=programme, weekday=weekday,
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
        weekday: Optional[int] = None,
        base_cpp: Optional[float] = None,
    ) -> float:
        """The value the optimizer should RANK on: real premium times pressure.

        This is ``effective_premium x pressure_multiplier``. Honest money, biased
        placement: revenue uses only effective_premium; ranking sees both.
        """
        return self.effective_premium(
            advertiser_id, position=position, genre=genre, daypart=daypart,
            programme=programme, weekday=weekday, base_cpp=base_cpp,
        ) * self.pressure_multiplier(
            advertiser_id, position=position, genre=genre, daypart=daypart,
            programme=programme, weekday=weekday,
        )

    def segment_demand(
        self,
        *,
        channel: Optional[str] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
        weekday: Optional[int] = None,
    ) -> float:
        """Placement-preference weight for a programme segment, >= 1.0.

        Aggregates, across ALL advertisers in the engine, the placement-only
        signals that express interest in inventory matching this segment's scope.
        ``channel`` is accepted for convenience but has no effect, because channel
        is not a rule dimension. Specifically, for each advertiser the method
        collects:

          * every PRESSURE rule whose scope intersects (genre, daypart, programme),
            contributing its ``(1 + value/100)`` factor,
          * every PREMIUM rule whose scope intersects the same dimensions AND whose
            effective factor is above 1.0 (expressing genuine demand for this
            inventory), contributing that factor.

        Position-scoped rules are excluded: positions belong to individual spots,
        not to the programme segment as a whole. CPP-mode premium rules are
        skipped because no ``base_cpp`` is available at segment scope, and
        premium_discount rules never contribute (a discount is not demand; its
        standalone factor is 1.0). A weekday-scoped rule participates only when
        the caller supplies the segment day's ISO ``weekday``; with no weekday
        it never matches, so nothing is guessed.

        The per-advertiser contributions are multiplied together; then all
        advertisers' products are multiplied into a single weight. A weight of
        1.0 means no advertiser's rules express a demand bias for this scope.

        This weight is ONLY used in the greedy break-count ranking step of the
        optimizer, to steer WHERE breaks go. It is NEVER added to reported revenue
        and is NEVER charged. Reported revenue is identical whether the signal is
        supplied or not.
        """
        weekday_token = None if weekday is None else str(int(weekday))
        weight = 1.0
        for advertiser_id, conditions in self.conditions.items():
            adv_factor = 1.0
            for condition in conditions:
                if condition.scope_positions:
                    continue  # position-scoped: spot-level only
                if not _dimension_matches(condition.scope_genres, genre):
                    continue
                if not _dimension_matches(condition.scope_dayparts, daypart):
                    continue
                if not _dimension_matches(condition.scope_programmes, programme):
                    continue
                if not _dimension_matches(condition.scope_weekdays, weekday_token):
                    continue
                if condition.effect == PRESSURE:
                    adv_factor *= max(0.0, 1.0 + condition.value / 100.0)
                elif condition.effect == PREMIUM:
                    if condition.mode in _CPP_MODES:
                        continue  # skip: no base_cpp to resolve
                    factor = _premium_factor(condition.value, condition.mode, base_cpp=None)
                    if factor > 1.0:
                        adv_factor *= factor
            weight *= adv_factor
        return max(1.0, weight)

    def allow_decision(
        self,
        advertiser_id: str,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
        weekday: Optional[int] = None,
    ) -> AllowDecision:
        """Whether a spot is allowed, plus a reason string for diagnostics.

        Precedence: baseline limits first, then forbid rules (always win over
        require), then require rules (at least one must match when any exist).
        """
        advertiser_id = self.key_for(advertiser_id)
        baseline = self.baselines.get(advertiser_id)
        if baseline is not None and not baseline.allows(
            position=position, genre=genre, daypart=daypart
        ):
            return AllowDecision(False, "blocked by baseline allow_positions/allow_genres/prime_time_only")

        rules = self._conditions_for(advertiser_id)
        for condition in rules:
            if condition.effect == FORBID and condition.matches(
                position=position, genre=genre, daypart=daypart,
                programme=programme, weekday=weekday,
            ):
                return AllowDecision(False, f"forbidden by rule {condition.rule_id}")

        requires = [c for c in rules if c.effect == REQUIRE]
        if requires:
            matched = next(
                (c for c in requires if c.matches(
                    position=position, genre=genre, daypart=daypart,
                    programme=programme, weekday=weekday)),
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
        weekday: Optional[int] = None,
    ) -> bool:
        """Boolean shorthand for :meth:`allow_decision`."""
        return self.allow_decision(
            advertiser_id, position=position, genre=genre, daypart=daypart,
            programme=programme, weekday=weekday,
        ).allowed

    def overlaps(self, advertiser_id: str) -> list[OverlapFinding]:
        """Find conflicting or overlapping conditional rules for an advertiser.

        For every unordered pair whose scopes intersect: a require/forbid pair is
        a ``conflict``, two premium rules are ``stacked_premium``, two pressure
        rules are ``stacked_pressure``, and any other same-effect pair is
        ``overlap``. Scope intersection understands weekday scopes, so two rules
        on disjoint weekdays (Saturday-only versus Sunday-only) do not overlap.
        """
        rules = self._conditions_for(self.key_for(advertiser_id))
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


# Thin wrappers so the rest of the codebase and tests keep their import paths.
# The real logic lives in _rule_helpers to stay within the line limit.


def _condition_from_row(row: dict[str, str]) -> Optional[Condition]:
    return _condition_from_row_helper(row)  # type: ignore[return-value]


def normalize_scope(raw: object) -> str:
    """Serialize a scope value back to the canonical comma-or-ANY string."""
    tokens = _tokens(raw)
    if not tokens:
        return ANY
    return ",".join(sorted(tokens))
