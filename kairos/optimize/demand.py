"""Build per-segment advertiser demand weights for the optimizer.

Every call to :func:`kairos.optimize.optimizer.optimize_breaks` should receive
demand weights: they steer WHERE breaks go when two segments have similar
economics, without changing reported revenue. This module provides the single
place that computes them.

The weights are always computed (never gated behind a flag), self-neutralizing
by construction: when the advertiser CSVs contain no rules that match the
segments, every weight is exactly 1.0 and the optimizer output is byte-identical
to a run with no weights at all. The safety argument requires no flag; the
identity case is the mathematical proof.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional

from kairos.data.dayparts import daypart_for_hour
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.optimizer import ProgramSegment

_SECONDS_PER_HOUR = 3600.0

# Global bounds on the combined placement weight. Advertiser demand and inventory
# awareness are boost-only (>= 1.0) by their data semantics, but the pacing signal
# is TWO-SIDED: it may push a slot below 1.0 to de-prioritize an over-delivered
# campaign. WEIGHT_FLOOR keeps that penalty bounded so a slot is never effectively
# forbidden (which would indirectly suppress revenue); WEIGHT_CAP keeps one
# desperate campaign from dominating the whole schedule. With no data every signal
# is exactly 1.0 and clamp(1.0, FLOOR, CAP) == 1.0, so the identity no-op holds.
WEIGHT_FLOOR = 0.25
WEIGHT_CAP = 4.0


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _hour_from_start_seconds(start_seconds: float) -> Optional[int]:
    """Clock hour (0..23) from seconds-since-midnight, or None when invalid."""
    try:
        h = int(start_seconds // _SECONDS_PER_HOUR) % 24
        return h if 0 <= h <= 23 else None
    except (TypeError, ValueError):
        return None


def build_demand_weights(
    segments: Iterable[ProgramSegment],
    engine: AdvertiserRuleEngine,
    *,
    inventory_weights: Optional[Mapping[str, float]] = None,
    pacing_weights: Optional[Mapping[str, float]] = None,
) -> dict[str, float]:
    """Compute per-segment placement-preference weights for the optimizer.

    For each segment the daypart is derived from ``segment.start_seconds`` via
    the canonical Israeli taxonomy in :mod:`kairos.data.dayparts`. When the
    start time is missing or out of range the daypart is ``None``, and
    :meth:`~AdvertiserRuleEngine.segment_demand` treats an absent scope as
    ``ANY`` (matches every rule in that dimension), which is the honest
    conservative choice.

    Three independent placement signals are folded together MULTIPLICATIVELY,
    each off-by-identity until its data lands:

      * advertiser demand (the rule engine, always computed; boost-only, >= 1.0);
      * inventory awareness (``inventory_weights``, from
        :func:`kairos.optimize.inventory.build_inventory_weights`; boost-only);
      * delivery-pacing urgency (``pacing_weights``, from
        :func:`kairos.optimize.pacing.build_pacing_weights`; TWO-SIDED, may be
        below 1.0 to de-prioritize an over-delivered campaign).

    The combined weight is ``clamp(advertiser * inventory * pacing, WEIGHT_FLOOR,
    WEIGHT_CAP)``. Each extra map defaults to ``None`` (treated as 1.0 everywhere),
    so omitting both reproduces the advertiser-only weight exactly, and an
    advertiser engine with no rules plus no extra maps yields every weight at 1.0,
    the optimizer's identity case, byte-identical to a run with no weights at all.

    The returned dict maps segment_id -> weight. These weights touch only the
    optimizer's ranking comparison, never charged revenue.
    """
    weights: dict[str, float] = {}
    for segment in segments:
        hour = _hour_from_start_seconds(segment.start_seconds)
        daypart = daypart_for_hour(hour) if hour is not None else None
        weight = engine.segment_demand(
            channel=segment.channel,
            genre=segment.program_type,
            daypart=daypart,
            programme=segment.program_title if segment.program_title else None,
        )
        if inventory_weights is not None:
            # Inventory awareness is boost-only by construction.
            weight *= max(1.0, inventory_weights.get(segment.segment_id, 1.0))
        if pacing_weights is not None:
            # Pacing is two-sided: it may pull the product below 1.0 to
            # de-prioritize an over-delivered campaign. Do NOT floor it here.
            weight *= pacing_weights.get(segment.segment_id, 1.0)
        # One global floor/cap so penalties and boosts compose honestly: a sub-1.0
        # pacing penalty survives (down to WEIGHT_FLOOR) while boosts stay bounded.
        weights[segment.segment_id] = _clamp(weight, WEIGHT_FLOOR, WEIGHT_CAP)
    return weights
