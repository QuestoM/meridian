"""Deterministic ad-frequency and competitive-separation enforcement.

WHERE THIS LIVES (the honest constraint). The WEEKLY break-count optimizer
decides how many breaks a programme segment carries and has NO advertiser
attribution, so it cannot enforce a per-advertiser frequency rule. Real spot
attribution (advertiser, campaign, ad/creative, break_start, position) exists
only on the DAILY spot-pricing path (:mod:`kairos.export.spots`), so frequency
and competitive separation are enforced HERE, over the already-priced, already-
ordered spots, and nowhere else. The dashboard status text says this plainly.

WHAT THIS DOES. Given the ordered priced spots from
:func:`kairos.export.spots.price_daily_spots`, this pass walks them in their
priced order and, for each spot, asks whether keeping it would violate the
MOST-SPECIFIC effective limit for its identity (ad > campaign > advertiser >
default) or a competitive-separation rule between two competing advertisers. A
spot that would violate is dropped and recorded with an explicit reason
(mirroring :class:`kairos.export.spots.DroppedSpot`), so the operator sees
exactly what was removed and why; a spot that is fine is kept in place.

LIMIT TYPES (see :mod:`kairos.optimize._frequency_rules` for authoring):

  * MAX_PER_BREAK         -> at most N spots of one target in a single break,
  * MAX_CONSECUTIVE       -> no same target in N adjacent positions of a break,
  * MIN_SEPARATION        -> a minimum gap (minutes or positions) between two
                             spots of the same target,
  * MAX_PER_DAY           -> at most N spots of one target across the whole day,
  * COMPETITIVE_SEPARATION-> keep two DIFFERENT but competing advertisers apart
                             (not in the same break, or >= N minutes/positions).

IMPORTANT BOUNDARY. Competitive separation is advertiser-vs-advertiser WITHIN the
client's OWN channel (e.g. two banks in the same news break). It is NOT the
competitor-CHANNEL boundary, which is a separate mechanism and is never touched
here; this pass only ever reorders/drops spots that already belong to the owned
channel's own log.

HONESTY. The pass is pure: no datetime.now / no random. With no enabled rule the
output spot list is identical to the input (the identity case). Every removed
spot carries a reason. A spot is never both kept and dropped. A conservative,
EXPLICIT default (max 1 spot per advertiser per break) ships in the CSV but is
overridable and never hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import pandas as pd

from kairos.optimize._frequency_rules import (
    COMPETITIVE_SEPARATION,
    MAX_CONSECUTIVE,
    MAX_PER_BREAK,
    MAX_PER_DAY,
    MIN_SEPARATION,
    MINUTES,
    POSITIONS,
    FrequencyRule,
    FrequencyRuleSet,
    competitive_groups,
    load_frequency_rules,
    resolve_effective,
)


@dataclass(frozen=True)
class SpotView:
    """The minimal, attribution-bearing view of one spot the pass needs.

    ``key`` is whatever the caller uses to identify the spot in its own list
    (an index into the priced list). ``break_id`` groups spots into one break
    (the break_start clock); ``minute`` is the spot's air-time in minutes from an
    arbitrary day origin (for MIN_SEPARATION by minutes); ``position`` is the
    position-in-break. A None minute disables minute-based checks for that spot
    (recorded, never guessed).
    """

    key: Any
    advertiser: str
    campaign: str
    ad: str
    break_id: str
    position: Optional[int]
    minute: Optional[float]


@dataclass(frozen=True)
class FrequencyDrop:
    """One spot dropped by a frequency / separation rule, with the reason."""

    key: Any
    advertiser: str
    campaign: str
    ad: str
    break_id: str
    rule_id: str
    limit_type: str
    reason: str


@dataclass
class EnforcementResult:
    """Kept spot keys (in order) and the dropped spots with reasons."""

    kept: list[Any] = field(default_factory=list)
    dropped: list[FrequencyDrop] = field(default_factory=list)

    @property
    def status_text(self) -> str:
        """One honest sentence for the dashboard about where this enforces."""
        return (
            "Ad-frequency and competitive-separation rules are enforced on the "
            "per-spot daily path, over the owned channel's spots only "
            f"({len(self.kept)} kept, {len(self.dropped)} dropped by a rule); the "
            "weekly break-count optimizer has no advertiser attribution and does "
            "not enforce them."
        )


def _target_identity(spot: SpotView) -> tuple[str, str, str]:
    return spot.advertiser, spot.campaign, spot.ad


def _identity_key(rule: FrequencyRule, spot: SpotView) -> tuple[str, ...]:
    """The grouping key a same-target rule counts over, by its scope.

    A default/advertiser rule counts per advertiser; a campaign rule per
    (advertiser, campaign); an ad rule per (advertiser, campaign, ad). This is
    what makes "max 1 per advertiser per break" mean the advertiser, not the ad.
    """
    if rule.scope == "ad":
        return (spot.advertiser, spot.campaign, spot.ad)
    if rule.scope == "campaign":
        return (spot.advertiser, spot.campaign)
    return (spot.advertiser,)


def _check_same_target(
    spot: SpotView,
    rule: FrequencyRule,
    kept: list[SpotView],
) -> Optional[str]:
    """Return a reason string if keeping ``spot`` violates ``rule``, else None.

    ``kept`` are the already-accepted spots (the deterministic prefix). The check
    is forward-only: a spot is judged against what was kept before it, so the pass
    is order-stable and never revisits a decision.
    """
    ident = _identity_key(rule, spot)
    same = [s for s in kept if _identity_key(rule, s) == ident]

    if rule.limit_type == MAX_PER_BREAK:
        # A spot with no break attribution cannot be grouped into a break; we do
        # not collapse all unknown-break spots into one pseudo-break (that would
        # fabricate co-location), so the per-break cap simply does not apply.
        if not spot.break_id:
            return None
        in_break = [s for s in same if s.break_id and s.break_id == spot.break_id]
        if len(in_break) >= rule.value:
            return (
                f"max_per_break={int(rule.value)} reached for "
                f"{'/'.join(ident)} in break {spot.break_id}"
            )
        return None

    if rule.limit_type == MAX_PER_DAY:
        if len(same) >= rule.value:
            return f"max_per_day={int(rule.value)} reached for {'/'.join(ident)}"
        return None

    if rule.limit_type == MAX_CONSECUTIVE:
        if not spot.break_id or spot.position is None:
            return None
        in_break = [
            s for s in same
            if s.break_id == spot.break_id and s.position is not None
        ]
        run = _consecutive_run(spot.position, [s.position for s in in_break])
        if run > rule.value:
            return (
                f"max_consecutive={int(rule.value)} would be exceeded for "
                f"{'/'.join(ident)} near position {spot.position}"
            )
        return None

    if rule.limit_type == MIN_SEPARATION:
        for prior in same:
            gap = _separation(prior, spot, rule.unit)
            if gap is not None and gap < rule.value:
                return (
                    f"min_separation {rule.value} {rule.unit} not met vs prior "
                    f"{'/'.join(ident)} spot (gap {round(gap, 2)})"
                )
        return None

    return None


def _consecutive_run(candidate: int, kept_positions: list[Optional[int]]) -> int:
    """Length of the contiguous run of same-target positions including ``candidate``.

    Positions form a run when each is exactly one apart from the next (gap == 1),
    which is what "consecutive ad placements" means inside a break. Returns the
    size of the run the candidate would join. With no adjacent kept position the
    run is 1 (just the candidate).
    """
    present = {p for p in kept_positions if p is not None} | {candidate}
    length = 1
    lower = candidate - 1
    while lower in present:
        length += 1
        lower -= 1
    upper = candidate + 1
    while upper in present:
        length += 1
        upper += 1
    return length


def _separation(a: SpotView, b: SpotView, unit: str) -> Optional[float]:
    """The gap between two spots in the requested unit, or None if unknowable."""
    if unit == MINUTES:
        if a.minute is None or b.minute is None:
            return None
        return abs(b.minute - a.minute)
    if unit == POSITIONS:
        if a.break_id != b.break_id:
            # Different breaks are trivially separated in position terms.
            return float("inf")
        if a.position is None or b.position is None:
            return None
        return float(abs(b.position - a.position))
    return None


def _check_competitive(
    spot: SpotView,
    rule: FrequencyRule,
    kept: list[SpotView],
) -> Optional[str]:
    """Return a reason if ``spot`` lands too close to a competing advertiser.

    Only fires when ``spot``'s advertiser is a member AND a DIFFERENT member was
    already kept too close. Same-advertiser pairs are ignored here (that is the
    frequency rules' job), so this is strictly advertiser-vs-competitor.
    """
    if spot.advertiser not in rule.members:
        return None
    for prior in kept:
        if prior.advertiser == spot.advertiser:
            continue
        if prior.advertiser not in rule.members:
            continue
        same_break = bool(spot.break_id) and prior.break_id == spot.break_id
        gap = _separation(prior, spot, rule.unit)
        if rule.unit == POSITIONS and rule.value <= 0:
            # "not in the same break"; unknown breaks cannot be proven co-located.
            too_close = same_break
        elif gap is None:
            # gap unknowable (no minutes / no positions): fall back to same-break,
            # and only when the break is actually known.
            too_close = same_break
        else:
            too_close = gap < rule.value
        if too_close:
            return (
                f"competitive_separation '{rule.competing_group}': "
                f"{spot.advertiser} too close to {prior.advertiser} "
                f"(need {rule.value} {rule.unit})"
            )
    return None


def enforce_spots(
    spots: list[SpotView],
    ruleset: FrequencyRuleSet,
) -> EnforcementResult:
    """Walk priced spots in order and drop any that violate an effective rule.

    Deterministic and pure. For each spot the single most-specific frequency rule
    of each limit type is resolved, then every competitive-separation group is
    checked. The first violating rule found drops the spot with its reason; an
    accepted spot joins ``kept`` and constrains later spots.
    """
    result = EnforcementResult()
    kept_views: list[SpotView] = []

    by_limit = {
        MAX_PER_BREAK: ruleset.by_limit(MAX_PER_BREAK),
        MAX_PER_DAY: ruleset.by_limit(MAX_PER_DAY),
        MAX_CONSECUTIVE: ruleset.by_limit(MAX_CONSECUTIVE),
        MIN_SEPARATION: ruleset.by_limit(MIN_SEPARATION),
    }
    comp_rules = competitive_groups(ruleset.rules)

    for spot in spots:
        advertiser, campaign, ad = _target_identity(spot)
        drop: Optional[tuple[FrequencyRule, str]] = None

        for limit_type, rules in by_limit.items():
            if not rules:
                continue
            effective = resolve_effective(rules, advertiser, campaign, ad)
            if effective is None:
                continue
            reason = _check_same_target(spot, effective, kept_views)
            if reason is not None:
                drop = (effective, reason)
                break

        if drop is None:
            for rule in comp_rules:
                reason = _check_competitive(spot, rule, kept_views)
                if reason is not None:
                    drop = (rule, reason)
                    break

        if drop is None:
            kept_views.append(spot)
            result.kept.append(spot.key)
        else:
            rule, reason = drop
            result.dropped.append(FrequencyDrop(
                key=spot.key,
                advertiser=spot.advertiser,
                campaign=spot.campaign,
                ad=spot.ad,
                break_id=spot.break_id,
                rule_id=rule.rule_id,
                limit_type=rule.limit_type,
                reason=reason,
            ))

    return result


def _minute_of_day(value: Any) -> Optional[float]:
    """Parse a clock string (HH:MM:SS) to minutes-from-midnight, or None.

    Pure: uses pandas string parsing only, no current time. A post-midnight spot
    keeps its small minute value; within a single daily file all spots share a
    date so minute order is the air order, which is all MIN_SEPARATION needs.
    """
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.hour * 60.0 + parsed.minute + parsed.second / 60.0


def view_from_priced(key: Any, priced: Any, *, break_id: str, minute: Optional[float]) -> SpotView:
    """Build a :class:`SpotView` from a :class:`kairos.export.spots.PricedSpot`.

    ``ad`` is taken from the priced spot's campaign when no finer creative id is
    carried; the caller supplies ``break_id`` and ``minute`` from the raw row,
    because the PricedSpot intentionally does not carry the clock.
    """
    return SpotView(
        key=key,
        advertiser=getattr(priced, "advertiser", ""),
        campaign=getattr(priced, "campaign", ""),
        ad=getattr(priced, "ad", "") or getattr(priced, "campaign", ""),
        break_id=break_id,
        position=getattr(priced, "position", None),
        minute=minute,
    )


def load_default_ruleset(path: Any = None) -> FrequencyRuleSet:
    """Convenience: load the shipped frequency-rules CSV (or a given path)."""
    return load_frequency_rules(path)
