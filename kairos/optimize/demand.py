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
    each ``>= 1.0`` and each off-by-identity until its data lands:

      * advertiser demand (the rule engine, always computed);
      * inventory awareness (``inventory_weights``, from
        :func:`kairos.optimize.inventory.build_inventory_weights`);
      * delivery-pacing urgency (``pacing_weights``, from
        :func:`kairos.optimize.pacing.build_pacing_weights`).

    The combined weight is ``max(1.0, advertiser * inventory * pacing)``. Each
    extra map defaults to ``None`` (treated as 1.0 everywhere), so omitting both
    reproduces the advertiser-only weight exactly, and an advertiser engine with
    no rules plus no extra maps yields every weight at 1.0 - the optimizer's
    identity case, byte-identical to a run with no weights at all.

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
            weight *= max(1.0, inventory_weights.get(segment.segment_id, 1.0))
        if pacing_weights is not None:
            weight *= max(1.0, pacing_weights.get(segment.segment_id, 1.0))
        weights[segment.segment_id] = max(1.0, weight)
    return weights
