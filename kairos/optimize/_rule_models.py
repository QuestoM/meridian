"""Dataclasses and effect constants for the advertiser/agency rule engine.

Split out of :mod:`kairos.optimize.advertiser_rules` to keep that module under
the project line limit. Everything here is re-exported by advertiser_rules, so
the rest of the codebase keeps its import paths; nothing else should import
from this module directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from kairos.optimize._rule_helpers import dimension_matches, scopes_intersect
from kairos.optimize.positions import GOLD_POSITION, canonical_token  # noqa: F401

ANY = "ANY"

# GOLD_POSITION is re-exported above from kairos.optimize.positions, which owns
# the one position vocabulary (ordinals 1 to 5, L for last, and the gold break).

# Effects a conditional rule can carry.
PREMIUM = "premium"
REQUIRE = "require"
FORBID = "forbid"
# PRESSURE is a placement-only lever: steers WHERE the optimizer wants to place
# a spot (raises the slot's apparent value) but is NEVER charged, so the real
# revenue total is unchanged. Its value is a percent uplift (10 means +10%).
PRESSURE = "pressure"
_EFFECTS = (PREMIUM, REQUIRE, FORBID, PRESSURE)


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
        position_token = canonical_token(position)
        if not dimension_matches(self.allow_positions, position_token):
            return False
        if not dimension_matches(self.allow_genres, genre):
            return False
        if self.prime_time_only:
            if daypart is None or str(daypart).strip().lower() != "prime":
                return False
        return True


@dataclass(frozen=True)
class Condition:
    """One scoped conditional rule for an advertiser (or, keyed by id, an agency).

    ``effect`` is one of PREMIUM, REQUIRE, FORBID or PRESSURE. ``value`` is the
    premium amount (a multiplier, a percent, a cost-per-point amount or a
    surcharge-discount percent depending on ``mode``) or the pressure percent,
    ignored for require/forbid. ``mode`` defaults to the multiplier so legacy
    rows behave unchanged. The scope sets are token sets (empty = ANY = matches
    everything).
    """

    advertiser_id: str
    rule_id: str
    effect: str
    value: float = 1.0
    mode: str = "multiplier"
    scope_positions: frozenset[str] = frozenset()
    scope_genres: frozenset[str] = frozenset()
    scope_dayparts: frozenset[str] = frozenset()
    scope_programmes: frozenset[str] = frozenset()
    # A campaign always belongs to one advertiser, so a campaign-scoped rule
    # narrows this advertiser's rule to specific campaigns. Empty = ANY campaign.
    scope_campaigns: frozenset[str] = frozenset()
    # ISO weekday tokens "1".."7" (Monday=1 .. Sunday=7; Saturday, Hebrew שבת,
    # is 6). Empty = ANY weekday. A weekday-scoped rule matches a spot only when
    # the spot's date resolves to a weekday in this set; a consumer that has NO
    # date passes ``weekday=None`` and the rule then never matches (no guessing).
    scope_weekdays: frozenset[str] = frozenset()
    # Which named rate-card layer a PREMIUM rule REPLACES, instead of stacking
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
        weekday: Optional[int] = None,
    ) -> bool:
        """True when an observed spot scope is inside this rule's scope.

        ``weekday`` is the spot date's ISO weekday (1..7) or ``None`` when the
        caller has no date; against a non-empty ``scope_weekdays`` a ``None``
        weekday never matches, so a weekday-scoped rule cannot bite a dateless
        path by accident.
        """
        position_token = canonical_token(position)
        weekday_token = None if weekday is None else str(int(weekday))
        return (
            dimension_matches(self.scope_positions, position_token)
            and dimension_matches(self.scope_genres, genre)
            and dimension_matches(self.scope_dayparts, daypart)
            and dimension_matches(self.scope_programmes, programme)
            and dimension_matches(self.scope_campaigns, campaign)
            and dimension_matches(self.scope_weekdays, weekday_token)
        )

    def specificity(self) -> int:
        """How many scope dimensions this rule constrains (more = more specific).

        Used by the most-specific-wins resolver: a rule scoped to advertiser +
        campaign + position is more specific than one scoped to advertiser + position,
        so it wins per layer. An empty scope set counts as unconstrained (ANY).
        """
        return sum(1 for s in (
            self.scope_positions, self.scope_genres, self.scope_dayparts,
            self.scope_programmes, self.scope_campaigns, self.scope_weekdays,
        ) if s)

    def scope_intersects(self, other: "Condition") -> bool:
        """True when this rule's scope can describe the same spot as ``other``.

        Two rules on disjoint weekday sets (for example Saturday-only versus
        Sunday-only) cannot describe the same spot, so they do not overlap.
        """
        return (
            scopes_intersect(self.scope_positions, other.scope_positions)
            and scopes_intersect(self.scope_genres, other.scope_genres)
            and scopes_intersect(self.scope_dayparts, other.scope_dayparts)
            and scopes_intersect(self.scope_programmes, other.scope_programmes)
            and scopes_intersect(self.scope_campaigns, other.scope_campaigns)
            and scopes_intersect(self.scope_weekdays, other.scope_weekdays)
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
