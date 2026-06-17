"""Per-segment and per-group math primitives for the Kairos optimizer.

These are the building blocks the greedy allocator, refiner, and result
builder all share: retention, revenue, break geometry, and the group-level
objective contribution.  They are pure functions with no side-effects.

The objective the optimizer maximises is a CONVEX BLEND, not a subtraction:

    objective = revenue_weight * clamp(revenue / revenue_scale, 0, 1)
              + (1 - revenue_weight) * clamp(aggregate_retention, 0, 1)

The reported aggregate objective has this convex-blend form.  The per-group
contribution returned by :func:`_group_objective_contribution` is the ADDITIVE
SHARE of that same blend (revenue is a sum, the retention term is a
tvr-weighted sum, both divided by the same global constants), so summing every
group's contribution reproduces the global convex-blend objective.  The greedy
loop values individual BREAKS by the marginal change in this same expression,
which is separable and monotone in each segment independently.
"""

from __future__ import annotations

from typing import Optional, Sequence

from kairos.optimize.guardrails import Break
from kairos.optimize.objective import (
    break_revenue,
    conservative_impact,
    predicted_retention,
)
from kairos.optimize._types import (
    DEFAULT_BREAK_LENGTH_SECONDS,
    PlacementPin,
    ProgramSegment,
)

_EPSILON = 1e-9
_SECONDS_PER_HOUR = 3600.0


def _risk_adjusted_coefficient(segment: ProgramSegment, risk_lambda: float) -> float:
    """The per-break retention coefficient the optimizer should decide with.

    When the impact model supplies a credible interval on the coefficient, the
    decision is made against a (possibly more pessimistic) value via
    :func:`~kairos.optimize.objective.conservative_impact`, so an uncertain cost
    is not undervalued. With no interval, or with ``risk_lambda == 0``, this is
    exactly the point coefficient, so the default behavior is unchanged.
    """
    if segment.impact_ci_low is None or segment.impact_ci_high is None or risk_lambda <= 0.0:
        return segment.impact_coefficient
    return conservative_impact(
        segment.impact_coefficient,
        segment.impact_ci_low,
        segment.impact_ci_high,
        risk_lambda=risk_lambda,
    )


def _segment_retention(segment: ProgramSegment, k: int) -> float:
    """Retention of the segment once it carries ``k`` breaks.

    The base model charges ``impact_coefficient`` once per break. When the
    measured first-break adjustment is active (``first_break_multiplier > 1.0``),
    the show's FIRST interruption sheds more audience than later ones, so its
    coefficient is scaled by the multiplier while every later break keeps the base
    coefficient. The extra cost is purely additive: ``coefficient * (multiplier -
    1)`` is applied once when ``k >= 1``, so with a multiplier of 1.0 this is
    exactly the original model and reported revenue is unchanged. This is the only
    seam where the per-break coefficient enters retention, so the whole optimizer
    (marginal value, guardrails, plan totals) sees the adjustment consistently.
    """
    retention = predicted_retention(segment.retention_baseline, segment.impact_coefficient, k)
    if k >= 1 and segment.first_break_multiplier != 1.0:
        extra = segment.impact_coefficient * (segment.first_break_multiplier - 1.0)
        retention = predicted_retention(retention, extra, 1)
    return retention


def _marginal_revenue(
    segment: ProgramSegment,
    k: int,
    pins: Optional[Sequence[PlacementPin]] = None,
) -> float:
    """Revenue gained by the k-th break (the segment goes from k-1 to k breaks).

    The break is valued at the retention that holds once it is present, so each
    successive break earns less, which is what gives the greedy search its
    diminishing returns. With explicit ``pins`` the k-th break is valued at the
    k-th pin's own duration rather than the segment's fixed break length, so a
    variable-length pinned break earns exactly its own revenue.
    """
    if k <= 0:
        return 0.0
    retention = _segment_retention(segment, k)
    effective_tvr = segment.baseline_tvr * retention
    length = segment.break_length_seconds
    if pins is not None and 1 <= k <= len(pins):
        length = pins[k - 1].duration_seconds
    return break_revenue(
        effective_tvr,
        length,
        segment.cpp,
        unit_seconds=segment.unit_seconds,
        premium=segment.premium,
    )


def _segment_revenue(
    segment: ProgramSegment,
    k: int,
    pins: Optional[Sequence[PlacementPin]] = None,
) -> float:
    return sum(_marginal_revenue(segment, j, pins) for j in range(1, k + 1))


def _segment_break_objects(
    segment: ProgramSegment,
    k: int,
    *,
    is_gold: bool = False,
    pins: Optional[Sequence[PlacementPin]] = None,
) -> list[Break]:
    """Lay k breaks through the segment for guardrail evaluation.

    Without pins, breaks are spaced evenly at ``duration / (k + 1)``: a short
    programme cannot then hold many breaks without breaching the spacing
    guardrail, which is the real constraint. With explicit ``pins`` each break is
    placed at ``segment.start_seconds + pin.offset_seconds`` and carries the
    pin's own duration and gold flag, so an operator's exact geometry is what the
    guardrails are checked against. Every break carries the segment's realised
    (final) retention, the value the retention floor must be checked against.
    ``is_gold`` lets a caller mark the breaks gold without mutating the frozen
    segment, which is how a gold override is honored.
    """
    if k <= 0:
        return []
    retention = _segment_retention(segment, k)
    gold = segment.is_gold or is_gold
    breaks: list[Break] = []
    if pins is not None:
        for pin in pins:
            start = segment.start_seconds + pin.offset_seconds
            breaks.append(Break(
                channel=segment.channel,
                day=segment.day,
                hour=int(start // _SECONDS_PER_HOUR),
                start_seconds=start,
                duration_seconds=pin.duration_seconds,
                program_type=segment.program_type,
                retention=retention,
                is_gold=gold or pin.is_gold,
            ))
        return breaks
    spacing = segment.duration_seconds / (k + 1)
    for j in range(1, k + 1):
        start = segment.start_seconds + spacing * j - segment.break_length_seconds / 2.0
        start = max(segment.start_seconds, start)
        breaks.append(Break(
            channel=segment.channel,
            day=segment.day,
            hour=int(start // _SECONDS_PER_HOUR),
            start_seconds=start,
            duration_seconds=segment.break_length_seconds,
            program_type=segment.program_type,
            retention=retention,
            is_gold=gold,
        ))
    return breaks


def _group_breaks(
    group: list[ProgramSegment],
    state: dict[str, int],
    gold_by_id: Optional[dict[str, bool]] = None,
    placements: Optional[dict[str, Sequence[PlacementPin]]] = None,
) -> list[Break]:
    gold_by_id = gold_by_id or {}
    placements = placements or {}
    breaks: list[Break] = []
    for segment in group:
        breaks.extend(_segment_break_objects(
            segment, state[segment.segment_id],
            is_gold=gold_by_id.get(segment.segment_id, False),
            pins=placements.get(segment.segment_id),
        ))
    return breaks


def _group_objective_contribution(
    group: list[ProgramSegment],
    counts: dict[str, int],
    *,
    revenue_weight: float,
    revenue_scale: float,
    total_tvr: float,
    placements: Optional[dict[str, Sequence[PlacementPin]]] = None,
) -> tuple[float, float, float]:
    """A channel-day's additive share of the global convex-blend objective.

    The global objective is a convex blend of normalised revenue and
    aggregate retention (see module docstring). It is separable across
    segments: revenue is a sum of per-segment revenue and the retention
    term is a tvr-weighted sum of per-segment retention, both divided by
    the same global constants. So the objective of the whole schedule
    equals the sum of every group's contribution here, and the global
    optimum is reached by maximising each group on its own (groups share
    no guardrail: every check in :mod:`kairos.optimize.guardrails` is
    scoped to one channel-day or finer). Returns
    ``(contribution, revenue, retention_weighted)`` so the caller can also
    roll the totals back up.
    """
    placements = placements or {}
    revenue = 0.0
    retention_weighted = 0.0
    for segment in group:
        k = counts[segment.segment_id]
        revenue += _segment_revenue(segment, k, placements.get(segment.segment_id))
        retention_weighted += segment.baseline_tvr * _segment_retention(segment, k)
    revenue_term = revenue_weight * (revenue / revenue_scale)
    retention_share = retention_weighted / total_tvr if total_tvr > _EPSILON else 1.0
    retention_term = (1.0 - revenue_weight) * retention_share
    return revenue_term + retention_term, revenue, retention_weighted
