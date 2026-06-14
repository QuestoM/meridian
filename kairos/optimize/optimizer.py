"""Greedy, guardrail-respecting break optimizer for Kairos.

The optimizer answers the daily question the channel faces: for each programme
segment, how many commercial breaks should it carry, so that the schedule earns
the most ad revenue without losing the audience or breaching policy.

It is deliberately a transparent greedy allocator rather than an opaque solver:

  * Start from zero breaks everywhere.
  * Repeatedly add the single break that gains the most on the weighted
    revenue-versus-retention objective, but only if it keeps the schedule
    inside every guardrail.
  * Stop when no further break improves the objective, or every remaining
    break would breach a guardrail.

Greedy fits this problem because the objective has diminishing returns: each
extra break in a segment earns less (the audience that stays is smaller) and
costs more retention, so the marginal value falls monotonically. Every decision
is recorded with the gain that justified it, which is the story a programme
manager and a marketing lead can both read.

The economics come from :mod:`kairos.optimize.objective` and the limits from
:mod:`kairos.optimize.guardrails`; this module only sequences the decisions.
It needs no trained model, so it runs anywhere, and a fitted impact coefficient
can be supplied per segment once the Meridian model is available.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from kairos.optimize.guardrails import Break, Guardrails, Violation, evaluate, is_compliant
from kairos.optimize.objective import (
    STANDARD_UNIT_SECONDS,
    break_revenue,
    predicted_retention,
    weighted_objective,
)

SECONDS_PER_HOUR = 3600.0
DEFAULT_BREAK_LENGTH_SECONDS = 120.0  # a two-minute break, a common unit
_EPSILON = 1e-9


@dataclass(frozen=True)
class ProgramSegment:
    """One programme the optimizer may load with breaks.

    ``start_seconds`` is measured from midnight, so a break's clock hour is
    ``start_seconds // 3600``. ``impact_coefficient`` is the retention change per
    break (normally negative) and defaults to zero so a caller without a fitted
    impact model still gets a revenue-only allocation.
    """

    segment_id: str
    channel: str
    day: str
    start_seconds: float
    duration_seconds: float
    program_type: str
    baseline_tvr: float                 # rating points with no breaks
    cpp: float                          # cost per rating point (daypart-adjusted)
    impact_coefficient: float = 0.0     # retention delta per break, usually <= 0
    retention_baseline: float = 1.0
    premium: float = 1.0                # position / daypart premium
    is_gold: bool = False
    max_breaks: int = 4
    break_length_seconds: float = DEFAULT_BREAK_LENGTH_SECONDS
    unit_seconds: float = STANDARD_UNIT_SECONDS   # the duration ``cpp`` is quoted per

    @property
    def hour(self) -> int:
        return int(self.start_seconds // SECONDS_PER_HOUR)

    def validate(self) -> None:
        if self.duration_seconds <= 0:
            raise ValueError(f"segment {self.segment_id}: duration_seconds must be positive")
        if self.baseline_tvr < 0:
            raise ValueError(f"segment {self.segment_id}: baseline_tvr must be non-negative")
        if self.cpp < 0:
            raise ValueError(f"segment {self.segment_id}: cpp must be non-negative")
        if self.premium < 0:
            raise ValueError(f"segment {self.segment_id}: premium must be non-negative")
        if self.max_breaks < 0:
            raise ValueError(f"segment {self.segment_id}: max_breaks must be non-negative")
        if self.break_length_seconds <= 0:
            raise ValueError(f"segment {self.segment_id}: break_length_seconds must be positive")
        if self.unit_seconds <= 0:
            raise ValueError(f"segment {self.segment_id}: unit_seconds must be positive")
        if not 0.0 <= self.retention_baseline <= 1.0:
            raise ValueError(f"segment {self.segment_id}: retention_baseline must be in [0, 1]")


@dataclass(frozen=True)
class BreakPlacement:
    """A single break the optimizer placed, with the value credited to it."""

    segment_id: str
    channel: str
    day: str
    hour: int
    start_seconds: float
    duration_seconds: float
    program_type: str
    position_in_segment: int       # 1-based order within the segment
    retention: float               # realised retention of the segment it sits in
    revenue: float                 # marginal revenue credited at insertion
    is_gold: bool


@dataclass(frozen=True)
class SegmentPlan:
    """The optimizer's decision for one segment."""

    segment_id: str
    num_breaks: int
    retention: float
    revenue: float
    placements: tuple[BreakPlacement, ...]


@dataclass(frozen=True)
class Decision:
    """One greedy step, kept so the schedule can explain itself."""

    segment_id: str
    break_index: int                   # the break number added (1-based)
    marginal_objective_gain: float
    marginal_revenue: float
    retention_after: float


@dataclass(frozen=True)
class OptimizationResult:
    segments: tuple[SegmentPlan, ...]
    placements: tuple[BreakPlacement, ...]     # every break, flat
    total_revenue: float
    aggregate_retention: float
    objective: float
    violations: tuple[Violation, ...]          # empty when compliant
    revenue_weight: float
    revenue_scale: float
    decisions: tuple[Decision, ...]

    @property
    def total_breaks(self) -> int:
        return len(self.placements)

    @property
    def is_compliant(self) -> bool:
        return not self.violations


def _segment_retention(segment: ProgramSegment, k: int) -> float:
    return predicted_retention(segment.retention_baseline, segment.impact_coefficient, k)


def _marginal_revenue(segment: ProgramSegment, k: int) -> float:
    """Revenue gained by the k-th break (the segment goes from k-1 to k breaks).

    The break is valued at the retention that holds once it is present, so each
    successive break earns less, which is what gives the greedy search its
    diminishing returns.
    """
    if k <= 0:
        return 0.0
    retention = _segment_retention(segment, k)
    effective_tvr = segment.baseline_tvr * retention
    return break_revenue(
        effective_tvr,
        segment.break_length_seconds,
        segment.cpp,
        unit_seconds=segment.unit_seconds,
        premium=segment.premium,
    )


def _segment_revenue(segment: ProgramSegment, k: int) -> float:
    return sum(_marginal_revenue(segment, j) for j in range(1, k + 1))


def _segment_break_objects(segment: ProgramSegment, k: int) -> list[Break]:
    """Lay k breaks evenly through the segment for guardrail evaluation.

    Even spacing of ``duration / (k + 1)`` means a short programme cannot hold
    many breaks without breaching the spacing guardrail, which is the real
    constraint. Every break carries the segment's realised (final) retention,
    the value the retention floor must be checked against.
    """
    if k <= 0:
        return []
    retention = _segment_retention(segment, k)
    spacing = segment.duration_seconds / (k + 1)
    breaks: list[Break] = []
    for j in range(1, k + 1):
        start = segment.start_seconds + spacing * j - segment.break_length_seconds / 2.0
        start = max(segment.start_seconds, start)
        breaks.append(Break(
            channel=segment.channel,
            day=segment.day,
            hour=int(start // SECONDS_PER_HOUR),
            start_seconds=start,
            duration_seconds=segment.break_length_seconds,
            program_type=segment.program_type,
            retention=retention,
            is_gold=segment.is_gold,
        ))
    return breaks


def _group_breaks(group: list[ProgramSegment], state: dict[str, int]) -> list[Break]:
    breaks: list[Break] = []
    for segment in group:
        breaks.extend(_segment_break_objects(segment, state[segment.segment_id]))
    return breaks


def optimize_breaks(
    segments: Iterable[ProgramSegment],
    guardrails: Optional[Guardrails] = None,
    *,
    revenue_weight: float = 0.5,
    revenue_scale: Optional[float] = None,
) -> OptimizationResult:
    """Allocate breaks across ``segments`` to maximise the weighted objective.

    ``revenue_weight`` in [0, 1] sets the balance: 1.0 chases revenue only and
    fills every segment up to the guardrails, 0.0 protects retention only and
    places no breaks. ``revenue_scale`` normalises revenue so it is comparable to
    retention; when omitted it defaults to the revenue of loading every segment
    to ``max_breaks`` (the marketing-maximal reference), floored above zero.

    The returned schedule is always compliant: ``violations`` is empty unless a
    guardrail interaction the greedy step could not localise slipped through, in
    which case it is reported rather than hidden.
    """
    guardrails = guardrails or Guardrails()
    if not 0.0 <= revenue_weight <= 1.0:
        raise ValueError("revenue_weight must be in [0, 1]")

    # Sort by id so the search is deterministic regardless of input order.
    segs = sorted(segments, key=lambda s: s.segment_id)
    for segment in segs:
        segment.validate()
    by_id = {s.segment_id: s for s in segs}
    if len(by_id) != len(segs):
        raise ValueError("segment_id values must be unique")

    groups: dict[tuple[str, str], list[ProgramSegment]] = defaultdict(list)
    for segment in segs:
        groups[(segment.channel, segment.day)].append(segment)

    if revenue_scale is None:
        full = sum(_segment_revenue(s, s.max_breaks) for s in segs)
        revenue_scale = max(full, _EPSILON)
    elif revenue_scale <= 0:
        raise ValueError("revenue_scale must be positive")

    total_tvr = sum(s.baseline_tvr for s in segs)
    state: dict[str, int] = {s.segment_id: 0 for s in segs}
    total_revenue = 0.0
    retention_weighted = sum(s.baseline_tvr * _segment_retention(s, 0) for s in segs)

    def aggregate_retention() -> float:
        if total_tvr <= _EPSILON:
            return 1.0
        return retention_weighted / total_tvr

    def objective_of(revenue: float, retention: float) -> float:
        return weighted_objective(
            revenue,
            retention,
            revenue_weight=revenue_weight,
            revenue_scale=revenue_scale,
        )

    decisions: list[Decision] = []
    while True:
        base_objective = objective_of(total_revenue, aggregate_retention())
        best_gain = _EPSILON
        best_id: Optional[str] = None
        for segment in segs:
            k = state[segment.segment_id]
            if k >= segment.max_breaks:
                continue
            marginal_rev = _marginal_revenue(segment, k + 1)
            delta_retention = segment.baseline_tvr * (
                _segment_retention(segment, k + 1) - _segment_retention(segment, k)
            )
            candidate_revenue = total_revenue + marginal_rev
            candidate_retention = (
                (retention_weighted + delta_retention) / total_tvr
                if total_tvr > _EPSILON else 1.0
            )
            gain = objective_of(candidate_revenue, candidate_retention) - base_objective
            if gain <= best_gain:
                continue
            group = groups[(segment.channel, segment.day)]
            state[segment.segment_id] = k + 1
            feasible = is_compliant(_group_breaks(group, state), guardrails)
            state[segment.segment_id] = k
            if feasible:
                best_gain = gain
                best_id = segment.segment_id

        if best_id is None:
            break
        segment = by_id[best_id]
        k = state[best_id]
        marginal_rev = _marginal_revenue(segment, k + 1)
        state[best_id] = k + 1
        total_revenue += marginal_rev
        retention_weighted += segment.baseline_tvr * (
            _segment_retention(segment, k + 1) - _segment_retention(segment, k)
        )
        decisions.append(Decision(
            segment_id=best_id,
            break_index=k + 1,
            marginal_objective_gain=best_gain,
            marginal_revenue=marginal_rev,
            retention_after=_segment_retention(segment, k + 1),
        ))

    return _build_result(
        segs, state, total_revenue, aggregate_retention(),
        objective_of(total_revenue, aggregate_retention()),
        guardrails, revenue_weight, revenue_scale, decisions,
    )


def _build_result(
    segs: list[ProgramSegment],
    state: dict[str, int],
    total_revenue: float,
    aggregate_retention: float,
    objective: float,
    guardrails: Guardrails,
    revenue_weight: float,
    revenue_scale: float,
    decisions: list[Decision],
) -> OptimizationResult:
    placements: list[BreakPlacement] = []
    segment_plans: list[SegmentPlan] = []
    for segment in segs:
        k = state[segment.segment_id]
        retention = _segment_retention(segment, k)
        breaks = _segment_break_objects(segment, k)
        segment_placements: list[BreakPlacement] = []
        for index, brk in enumerate(breaks, start=1):
            segment_placements.append(BreakPlacement(
                segment_id=segment.segment_id,
                channel=segment.channel,
                day=segment.day,
                hour=brk.hour,
                start_seconds=brk.start_seconds,
                duration_seconds=brk.duration_seconds,
                program_type=segment.program_type,
                position_in_segment=index,
                retention=retention,
                revenue=_marginal_revenue(segment, index),
                is_gold=segment.is_gold,
            ))
        placements.extend(segment_placements)
        segment_plans.append(SegmentPlan(
            segment_id=segment.segment_id,
            num_breaks=k,
            retention=retention,
            revenue=_segment_revenue(segment, k),
            placements=tuple(segment_placements),
        ))

    all_breaks = [
        Break(
            channel=p.channel, day=p.day, hour=p.hour,
            start_seconds=p.start_seconds, duration_seconds=p.duration_seconds,
            program_type=p.program_type, retention=p.retention, is_gold=p.is_gold,
        )
        for p in placements
    ]
    return OptimizationResult(
        segments=tuple(segment_plans),
        placements=tuple(placements),
        total_revenue=total_revenue,
        aggregate_retention=aggregate_retention,
        objective=objective,
        violations=tuple(evaluate(all_breaks, guardrails)),
        revenue_weight=revenue_weight,
        revenue_scale=revenue_scale,
        decisions=tuple(decisions),
    )
