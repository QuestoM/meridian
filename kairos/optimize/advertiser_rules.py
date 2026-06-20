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

Honesty rules: an unknown advertiser yields a premium of 1.0 (never zero) and is
allowed; scopes are token sets per dimension where an empty scope or ``ANY``
matches everything; nothing is invented to fill an empty conditions file.

Pure math helpers and CSV loaders live in
:mod:`kairos.optimize._rule_helpers` to keep this file under 450 lines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from kairos.optimize._rule_helpers import (
    CPP_MODES,
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

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RULES_PATH = ROOT / "data" / "advertiser_rules.csv"
DEFAULT_CONDITIONS_PATH = ROOT / "data" / "advertiser_conditions.csv"

# Effects a conditional rule can carry.
PREMIUM = "premium"
REQUIRE = "require"
FORBID = "forbid"
# PRESSURE is a placement-only lever: steers WHERE the optimizer wants to place
# a spot (raises the slot's apparent value) but is NEVER charged, so the real
# revenue total is unchanged. Its value is a percent uplift (10 means +10%).
PRESSURE = "pressure"
_EFFECTS = (PREMIUM, REQUIRE, FORBID, PRESSURE)

ANY = "ANY"

# How the optimizer/pricing path describes positions inside a break.
GOLD_POSITION = "gold"

# How a PREMIUM rule's value is interpreted.
#   * MULTIPLIER (default): value IS the multiplier, e.g. 1.15 means +15%.
#   * PERCENT: value is a signed percent, e.g. +15 -> 1.15, -15 -> 0.85.
#   * CPP_ABSOLUTE / CPP_ADD / CPP_DISCOUNT: value is a cost-per-point AMOUNT.
MULTIPLIER = "multiplier"
PERCENT = "percent"
CPP_ABSOLUTE = "cpp_absolute"
CPP_ADD = "cpp_add"
CPP_DISCOUNT = "cpp_discount"
_PREMIUM_MODES = (MULTIPLIER, PERCENT, CPP_ABSOLUTE, CPP_ADD, CPP_DISCOUNT)
_CPP_MODES = CPP_MODES

# Private aliases matching the underscore style used in the rest of this file.
_tokens = scope_tokens
_dimension_matches = dimension_matches
_scopes_intersect = scopes_intersect
_to_float = parse_float
_to_bool = parse_bool
_normalize_mode = parse_mode
_premium_factor = compute_premium_factor


@dataclass(frozen=True)
class Baseline:
    """The baseline rule for one advertiser, from advertiser_rules.csv."""

    advertiser_id: str
    default_premium: float = 1.0
    allow_positions: frozenset[str] = frozenset()
    allow_genres: frozenset[str] = frozenset()
    prime_time_only: bool = False
    # Optional per-advertiser delivery-pacing defaults, read from the urgency_k /
    # ahead_k columns of advertiser_rules.csv. ``None`` means "use the channel-wide
    # default". They steer how aggressively this advertiser's campaigns are leaned
    # toward (behind pace) or away from (over-delivered) when no per-campaign
    # override is set, the same layering the premium rules use. They touch only the
    # optimizer's placement ranking, never charged revenue.
    urgency_k: Optional[float] = None
    ahead_k: Optional[float] = None

    def allows(self, *, position: Optional[int], genre: Optional[str], daypart: Optional[str]) -> bool:
        """True when a spot passes this advertiser's baseline constraints."""
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

    ``effect`` is one of PREMIUM, REQUIRE, FORBID or PRESSURE. ``value`` is the
    premium amount (a multiplier, a percent or a cost-per-point amount depending
    on ``mode``) or the pressure percent, ignored for require/forbid. ``mode``
    defaults to MULTIPLIER so legacy rows behave unchanged. The four scope sets
    are token sets (empty = ANY = matches everything).
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
    # S6: a campaign always belongs to one advertiser, so a campaign-scoped rule
    # narrows this advertiser's rule to specific campaigns. Empty = ANY campaign.
    scope_campaigns: frozenset[str] = frozenset()
    # S5: which named rate-card layer a PREMIUM rule REPLACES, instead of stacking
    # on the running premium. "" (the default) keeps the legacy whole-stack
    # behavior so every existing rule is byte-identical. A non-empty target_layer
    # (one of program/prime/day/show/position/ad_type, or "final" for an
    # adjust-the-whole-price rule) makes the rule a per-layer or final override,
    # consumed by kairos.optimize.layer_overrides; the legacy effective_premium
    # path ignores targeted rules so charged revenue is unchanged until the
    # layered spot-pricing path is switched on.
    target_layer: str = ""
    priority: int = 0
    notes: str = ""

    def matches(
        self,
        *,
        position: Optional[int] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
        campaign: Optional[str] = None,
    ) -> bool:
        """True when an observed (position, genre, daypart, programme, campaign) is in scope."""
        position_token = None if position is None else str(position)
        return (
            _dimension_matches(self.scope_positions, position_token)
            and _dimension_matches(self.scope_genres, genre)
            and _dimension_matches(self.scope_dayparts, daypart)
            and _dimension_matches(self.scope_programmes, programme)
            and _dimension_matches(self.scope_campaigns, campaign)
        )

    def specificity(self) -> int:
        """How many scope dimensions this rule constrains (more = more specific).

        Used by the most-specific-wins resolver (S7): a rule scoped to advertiser +
        campaign + position is more specific than one scoped to advertiser + position,
        so it wins per layer. An empty scope set counts as unconstrained (ANY).
        """
        return sum(1 for s in (
            self.scope_positions, self.scope_genres, self.scope_dayparts,
            self.scope_programmes, self.scope_campaigns,
        ) if s)

    def scope_intersects(self, other: "Condition") -> bool:
        """True when this rule's scope can describe the same spot as ``other``."""
        return (
            _scopes_intersect(self.scope_positions, other.scope_positions)
            and _scopes_intersect(self.scope_genres, other.scope_genres)
            and _scopes_intersect(self.scope_dayparts, other.scope_dayparts)
            and _scopes_intersect(self.scope_programmes, other.scope_programmes)
            and _scopes_intersect(self.scope_campaigns, other.scope_campaigns)
        )


@dataclass(frozen=True)
class AllowDecision:
    """Whether a spot is allowed, with a human-readable reason for diagnostics."""

    allowed: bool
    reason: str


@dataclass(frozen=True)
class OverlapFinding:
    """One overlap or conflict between two of an advertiser's conditional rules."""

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
        """
        baseline = self.baselines.get(advertiser_id)
        premium = baseline.default_premium if baseline is not None else 1.0
        for condition in self._conditions_for(advertiser_id):
            if condition.target_layer:
                continue  # targeted layer/final override: handled by the layered path
            if condition.effect == PREMIUM and condition.matches(
                position=position, genre=genre, daypart=daypart, programme=programme
            ):
                factor = _premium_factor(condition.value, condition.mode, base_cpp)
                if condition.mode == CPP_ABSOLUTE and not (base_cpp is None or base_cpp <= 0):
                    premium = factor  # authoritative: SET the CPP, override prior factors
                else:
                    premium *= factor
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

        Each matching PRESSURE rule contributes ``(1 + value/100)``. The product
        is >= 0 and is 1.0 when no pressure rule matches.
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

        This is ``effective_premium x pressure_multiplier``. Honest money, biased
        placement: revenue uses only effective_premium; ranking sees both.
        """
        return self.effective_premium(
            advertiser_id, position=position, genre=genre, daypart=daypart,
            programme=programme, base_cpp=base_cpp,
        ) * self.pressure_multiplier(
            advertiser_id, position=position, genre=genre, daypart=daypart, programme=programme
        )

    def segment_demand(
        self,
        *,
        channel: Optional[str] = None,
        genre: Optional[str] = None,
        daypart: Optional[str] = None,
        programme: Optional[str] = None,
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
        skipped because no ``base_cpp`` is available at segment scope.

        The per-advertiser contributions are multiplied together; then all
        advertisers' products are multiplied into a single weight. A weight of
        1.0 means no advertiser's rules express a demand bias for this scope.

        This weight is ONLY used in the greedy break-count ranking step of the
        optimizer, to steer WHERE breaks go. It is NEVER added to reported revenue
        and is NEVER charged. Reported revenue is identical whether the signal is
        supplied or not.
        """
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
    ) -> AllowDecision:
        """Whether a spot is allowed, plus a reason string for diagnostics.

        Precedence: baseline limits first, then forbid rules (always win over
        require), then require rules (at least one must match when any exist).
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

        For every unordered pair whose scopes intersect: a require/forbid pair is
        a ``conflict``, two premium rules are ``stacked_premium``, two pressure
        rules are ``stacked_pressure``, and any other same-effect pair is
        ``overlap``.
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


# Thin wrappers so the rest of the codebase and tests keep their import paths.
# The real logic lives in _rule_helpers to stay within the 450-line limit.


def _condition_from_row(row: dict[str, str]) -> Optional[Condition]:
    return _condition_from_row_helper(row)  # type: ignore[return-value]


def normalize_scope(raw: object) -> str:
    """Serialize a scope value back to the canonical comma-or-ANY string."""
    tokens = _tokens(raw)
    if not tokens:
        return ANY
    return ",".join(sorted(tokens))
