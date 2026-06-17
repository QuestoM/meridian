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
from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Optional, Sequence

from kairos.optimize.guardrails import Break, Guardrails, Violation, evaluate, is_compliant
from kairos.optimize.objective import (
    STANDARD_UNIT_SECONDS,
    break_revenue,
    conservative_impact,
    predicted_retention,
    weighted_objective,
)
from kairos.optimize.overrides import OverrideSet

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

    ``impact_ci_low`` / ``impact_ci_high`` are the credible interval on that
    per-break coefficient when the impact model supplies one (both ``None`` means
    only the point is known, so the optimizer treats the cost as certain).
    ``impact_n`` is how many real breaks the estimate rests on and
    ``impact_confidence`` is its high / medium / low label, both carried purely so
    the plan can report how trustworthy each segment's retention cost is.
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
    impact_ci_low: Optional[float] = None         # credible interval on the coefficient
    impact_ci_high: Optional[float] = None
    impact_n: int = 0                             # real breaks behind the estimate
    impact_confidence: str = "low"                # high / medium / low label
    program_title: str = ""                       # programme Title, for cross-date matching
    first_break_multiplier: float = 1.0           # extra retention cost on the show's first break

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
        if self.first_break_multiplier < 1.0:
            raise ValueError(f"segment {self.segment_id}: first_break_multiplier must be >= 1.0")


@dataclass(frozen=True)
class PlacementPin:
    """One explicit break the operator pinned onto a segment.

    ``offset_seconds`` is measured from the segment's start, so the break's
    absolute clock position is ``segment.start_seconds + offset_seconds``.
    ``duration_seconds`` is this break's own length (breaks in one segment may
    differ in length). ``is_gold`` marks just this break gold, on top of any
    segment-level or override gold flag. Pinned breaks are honored as a HARD
    constraint by every optimizer tier: the segment's count is fixed at the
    number of pins and the breaks are emitted at exactly these positions.
    """

    offset_seconds: float
    duration_seconds: float
    is_gold: bool = False


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
    """The optimizer's decision for one segment.

    The ``retention_cost_*`` fields make the retention side of the decision
    auditable: ``retention_cost_point`` is the impact model's point estimate of the
    per-break retention drop, ``retention_cost_used`` is the (possibly more
    conservative) value the optimizer actually decided with after applying
    ``risk_lambda``, ``retention_cost_ci_low`` / ``retention_cost_ci_high`` is the
    credible interval (``None`` when only a point is known), ``retention_cost_n`` is
    the number of real breaks behind it, and ``retention_confidence`` is its
    high / medium / low label. They let the dashboard show not just how many breaks
    a segment carries but how trustworthy the cost driving that count was.
    """

    segment_id: str
    num_breaks: int
    retention: float
    revenue: float
    placements: tuple[BreakPlacement, ...]
    retention_cost_point: float = 0.0
    retention_cost_used: float = 0.0
    retention_cost_ci_low: Optional[float] = None
    retention_cost_ci_high: Optional[float] = None
    retention_cost_n: int = 0
    retention_confidence: str = "low"


@dataclass(frozen=True)
class Decision:
    """One greedy step, kept so the schedule can explain itself."""

    segment_id: str
    break_index: int                   # the break number added (1-based)
    marginal_objective_gain: float
    marginal_revenue: float
    retention_after: float


@dataclass(frozen=True)
class RejectedOverride:
    """One operator override the optimizer could not honor, with why.

    Honesty surface: an override is rejected (and kept OUT of the plan) when
    obeying it would breach a hard guardrail, for example a force that exceeds the
    segment's ``max_breaks`` or a pin that breaks the spacing guardrail. The
    operator sees exactly which override was dropped and the reason, so nothing is
    silently bent or silently ignored.
    """

    segment_id: str
    kind: str                          # pin / force (forbid and gold cannot be infeasible)
    requested: int                     # the break count the override asked for
    reason: str


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
    rejected_overrides: tuple[RejectedOverride, ...] = ()
    risk_lambda: float = 0.0                    # uncertainty preference applied to costs

    @property
    def total_breaks(self) -> int:
        return len(self.placements)

    @property
    def is_compliant(self) -> bool:
        return not self.violations


def _risk_adjusted_coefficient(segment: ProgramSegment, risk_lambda: float) -> float:
    """The per-break retention coefficient the optimizer should decide with.

    When the impact model supplies a credible interval on the coefficient, the
    decision is made against a (possibly more pessimistic) value via
    :func:`conservative_impact`, so an uncertain cost is not undervalued. With no
    interval, or with ``risk_lambda == 0``, this is exactly the point coefficient,
    so the default behavior is unchanged.
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
                hour=int(start // SECONDS_PER_HOUR),
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
            hour=int(start // SECONDS_PER_HOUR),
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
    """A channel-day's additive share of the global weighted objective.

    The global objective is separable across segments: revenue is a sum of
    per-segment revenue and the retention term is a tvr-weighted sum of
    per-segment retention, both divided by the same global constants. So the
    objective of the whole schedule equals the sum of every group's contribution
    here, and the global optimum is reached by maximising each group on its own
    (groups share no guardrail: every check in :mod:`kairos.optimize.guardrails`
    is scoped to one channel-day or finer). Returns ``(contribution, revenue,
    retention_weighted)`` so the caller can also roll the totals back up.
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


def optimize_breaks(
    segments: Iterable[ProgramSegment],
    guardrails: Optional[Guardrails] = None,
    *,
    revenue_weight: float = 0.5,
    revenue_scale: Optional[float] = None,
    overrides: Optional[OverrideSet] = None,
    risk_lambda: float = 0.0,
    placement_pins: Optional[Mapping[str, Sequence[PlacementPin]]] = None,
    demand_weights: Optional[Mapping[str, float]] = None,
    refine: bool = True,
) -> OptimizationResult:
    """Allocate breaks across ``segments`` to maximise the weighted objective.

    ``revenue_weight`` in [0, 1] sets the balance: 1.0 chases revenue only and
    fills every segment up to the guardrails, 0.0 protects retention only and
    places no breaks. ``revenue_scale`` normalises revenue so it is comparable to
    retention; when omitted it defaults to the revenue of loading every segment
    to ``max_breaks`` (the marketing-maximal reference), floored above zero.

    ``overrides`` is an optional :class:`~kairos.optimize.overrides.OverrideSet`
    of operator overrides honored as HARD constraints at the level the engine
    genuinely supports (break COUNTS per segment): a pinned segment is fixed at
    its count, a forced segment is floored at its minimum, a forbidden segment is
    held at 0, and a gold segment emits is_gold placements. An override that would
    breach a hard guardrail (for example a force above ``max_breaks`` or a pin
    that breaks spacing) is kept OUT of the plan and reported in
    ``result.rejected_overrides`` with a reason, never silently applied.

    ``risk_lambda`` in [0, 1] is the uncertainty preference applied to each
    segment's retention cost when the impact model gave that cost a credible
    interval: 0.0 (the default) decides with the point estimate and changes nothing,
    1.0 decides with the worst plausible cost in the interval, and values in between
    apply a partial variance penalty (see
    :func:`~kairos.optimize.objective.conservative_impact`). A segment with only a
    point coefficient is unaffected at any ``risk_lambda``.

    ``placement_pins`` maps a segment id to an explicit list of
    :class:`PlacementPin` (absolute offset-from-start, per-break duration, gold
    flag). A pinned segment is fixed at exactly those breaks: its count is forced
    to ``len(pins)`` and every tier emits the breaks at the pinned positions and
    durations, with revenue summed over the per-break durations. Pins are validated
    first (in-bounds and non-overlapping, then the spacing / load guardrails on the
    pinned geometry); a segment whose pins are invalid or breach a guardrail is
    dropped to 0 breaks and reported in ``result.rejected_overrides`` with
    ``kind="placement"``, never silently bent.

    ``demand_weights`` is an optional mapping from segment id to a placement-
    preference weight >= 1.0. When supplied, the greedy ranking step multiplies
    each segment's apparent objective gain by its weight before comparing segments,
    so a higher-demand segment is preferred when two segments have similar gains.
    This biases WHERE breaks go without changing reported revenue: weights touch
    only the ranking comparison, never ``total_revenue`` or any ``SegmentPlan``
    revenue field. A missing or 1.0 weight leaves a segment's ranking unchanged.
    Omitting the argument entirely (``None``) gives byte-identical output to
    today's optimizer. Produced by
    :meth:`~kairos.optimize.advertiser_rules.AdvertiserRuleEngine.segment_demand`.

    ``refine`` (default ``True``) runs the per-channel-day F1 refiner
    (:mod:`kairos.optimize.refiner`) after greedy converges: greedy is the
    warm-start and decision trace, then each channel-day's break counts are
    refined (exact for tiny groups, guardrail-aware local search for real ones)
    and adopted only where they STRICTLY beat greedy, recovering the revenue
    greedy leaves on the table because re-spacing makes feasibility non-monotone
    in break count. The refiner respects every guardrail and override greedy
    respects and never regresses a group, so the result is always at least as
    good as pure greedy. Set ``refine=False`` for pure-greedy output (A/B
    comparison, or the fast tests that assume the greedy allocation).

    The returned schedule is always compliant: ``violations`` is empty unless a
    guardrail interaction the greedy step could not localise slipped through, in
    which case it is reported rather than hidden.
    """
    guardrails = guardrails or Guardrails()
    if not 0.0 <= revenue_weight <= 1.0:
        raise ValueError("revenue_weight must be in [0, 1]")
    if not 0.0 <= risk_lambda <= 1.0:
        raise ValueError("risk_lambda must be in [0, 1]")

    # Sort by id so the search is deterministic regardless of input order.
    originals = sorted(segments, key=lambda s: s.segment_id)
    for segment in originals:
        segment.validate()
    original_by_id = {s.segment_id: s for s in originals}
    if len(original_by_id) != len(originals):
        raise ValueError("segment_id values must be unique")

    # Decide against the risk-adjusted coefficient: a more conservative (more
    # negative) retention cost where the estimate is uncertain, the point estimate
    # otherwise. The originals are kept so the plan can still report the point, the
    # interval and the confidence behind each segment's decision.
    segs = [replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda)) for s in originals]
    by_id = {s.segment_id: s for s in segs}

    groups: dict[tuple[str, str], list[ProgramSegment]] = defaultdict(list)
    for segment in segs:
        groups[(segment.channel, segment.day)].append(segment)

    if revenue_scale is None:
        full = sum(_segment_revenue(s, s.max_breaks) for s in segs)
        revenue_scale = max(full, _EPSILON)
    elif revenue_scale <= 0:
        raise ValueError("revenue_scale must be positive")

    constraints = overrides.segment_constraints() if overrides is not None else {}
    floors, caps, gold_by_id, rejected = _apply_segment_overrides(
        segs, groups, guardrails, constraints,
    )

    # Explicit placement pins fix a segment at exactly the supplied breaks. Valid
    # pins force floor == cap == len(pins) and feed the side map every tier reads;
    # invalid or guardrail-breaching pins are dropped to 0 and reported.
    placements = _apply_placement_pins(
        segs, groups, guardrails, placement_pins, floors, caps, gold_by_id, rejected,
    )

    total_tvr = sum(s.baseline_tvr for s in segs)
    state: dict[str, int] = dict(floors)
    total_revenue = sum(
        _segment_revenue(by_id[sid], k, placements.get(sid)) for sid, k in state.items()
    )
    retention_weighted = sum(s.baseline_tvr * _segment_retention(s, state[s.segment_id]) for s in segs)

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
            if k >= caps[segment.segment_id]:
                continue
            marginal_rev = _marginal_revenue(segment, k + 1, placements.get(segment.segment_id))
            delta_retention = segment.baseline_tvr * (
                _segment_retention(segment, k + 1) - _segment_retention(segment, k)
            )
            candidate_revenue = total_revenue + marginal_rev
            candidate_retention = (
                (retention_weighted + delta_retention) / total_tvr
                if total_tvr > _EPSILON else 1.0
            )
            gain = objective_of(candidate_revenue, candidate_retention) - base_objective
            # Demand weight scales the apparent gain used for ranking only. It
            # steers which segment gets the next break (placement bias) without
            # touching candidate_revenue or total_revenue, so reported revenue is
            # always real. A weight of 1.0 (or None) leaves ranking unchanged.
            if demand_weights is not None:
                gain = gain * max(1.0, demand_weights.get(segment.segment_id, 1.0))
            if gain <= best_gain:
                continue
            group = groups[(segment.channel, segment.day)]
            state[segment.segment_id] = k + 1
            feasible = is_compliant(_group_breaks(group, state, gold_by_id, placements), guardrails)
            state[segment.segment_id] = k
            if feasible:
                best_gain = gain
                best_id = segment.segment_id

        if best_id is None:
            break
        segment = by_id[best_id]
        k = state[best_id]
        marginal_rev = _marginal_revenue(segment, k + 1, placements.get(best_id))
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

    # Greedy gives a fast, compliant warm start, but because breaks are re-spaced
    # at duration/(k+1) the feasible region is not monotone in break count, so
    # greedy can stop short of a better compliant allocation it could only reach
    # through an infeasible intermediate. The objective is separable across
    # channel-days (groups share no guardrail: every check is scoped to one
    # channel-day or finer), so the true optimum is the sum of each group's own
    # optimum. The F1 refiner (kairos.optimize.refiner) climbs each group from the
    # greedy warm start (exact for tiny groups, guardrail-aware local search for
    # real ones); adopt the refined counts wherever they STRICTLY beat greedy, and
    # rebuild that group's decision trace so the explanation matches the shipped
    # plan. A group greedy already optimised keeps its greedy result. Skipped
    # entirely when refine is False, giving pure-greedy output for A/B and tests.
    decisions_by_group: dict[tuple[str, str], list[Decision]] = defaultdict(list)
    for decision in decisions:
        seg = by_id[decision.segment_id]
        decisions_by_group[(seg.channel, seg.day)].append(decision)

    if refine:
        # Imported here (not at module top) so refiner can import this module's
        # primitives without a circular import at load time.
        from kairos.optimize.refiner import optimize_group, replay_group_decisions
        for key, group in groups.items():
            greedy_counts = {s.segment_id: state[s.segment_id] for s in group}
            greedy_contribution, _, _ = _group_objective_contribution(
                group, greedy_counts,
                revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
                placements=placements,
            )
            refined_counts = optimize_group(
                group, greedy_counts, floors, caps, gold_by_id, guardrails,
                revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
                placements=placements,
            )
            refined_contribution, _, _ = _group_objective_contribution(
                group, refined_counts,
                revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
                placements=placements,
            )
            # Pure improvement guarantee: never adopt a worse-or-equal group.
            if refined_contribution <= greedy_contribution + _EPSILON:
                continue  # greedy already reached this group's optimum
            assert refined_contribution >= greedy_contribution, (
                "refiner regressed a group below greedy"
            )
            for segment in group:
                state[segment.segment_id] = refined_counts[segment.segment_id]
            decisions_by_group[key] = replay_group_decisions(
                group, refined_counts, floors,
                revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
                placements=placements,
            )

    # Roll the (possibly corrected) per-segment counts back up into the totals the
    # result reports. ``retention_weighted`` is a free variable of the
    # ``aggregate_retention`` closure, so re-binding it here updates that result too.
    total_revenue = sum(
        _segment_revenue(by_id[sid], k, placements.get(sid)) for sid, k in state.items()
    )
    retention_weighted = sum(
        s.baseline_tvr * _segment_retention(s, state[s.segment_id]) for s in segs
    )
    decisions = [d for key in groups for d in decisions_by_group[key]]

    return _build_result(
        segs, state, total_revenue, aggregate_retention(),
        objective_of(total_revenue, aggregate_retention()),
        guardrails, revenue_weight, revenue_scale, decisions, gold_by_id, rejected,
        original_by_id=original_by_id, risk_lambda=risk_lambda, placements=placements,
    )


def _apply_segment_overrides(
    segs: list[ProgramSegment],
    groups: dict[tuple[str, str], list[ProgramSegment]],
    guardrails: Guardrails,
    constraints: dict[str, dict[str, object]],
) -> tuple[dict[str, int], dict[str, int], dict[str, bool], list[RejectedOverride]]:
    """Turn segment constraints into per-segment floors, caps, gold flags.

    Returns ``(floors, caps, gold_by_id, rejected)``. A forbid pins the cap to 0.
    A pin sets floor == cap == the pinned count. A force lifts the floor. Gold
    tags the segment's breaks. Any pin or force that exceeds ``max_breaks`` or
    that makes its channel-day infeasible at the requested floor is rejected and
    left out (the segment falls back to a 0 floor and its normal cap), so an
    infeasible override never breaches a hard guardrail.
    """
    floors: dict[str, int] = {s.segment_id: 0 for s in segs}
    caps: dict[str, int] = {s.segment_id: s.max_breaks for s in segs}
    gold_by_id: dict[str, bool] = {}
    rejected: list[RejectedOverride] = []

    for segment in segs:
        entry = constraints.get(segment.segment_id)
        if not entry:
            continue
        if entry.get("gold"):
            gold_by_id[segment.segment_id] = True
        if entry.get("forbid"):
            floors[segment.segment_id] = 0
            caps[segment.segment_id] = 0
            continue
        pin = entry.get("pin")
        if pin is not None:
            requested = int(pin)
            if requested > segment.max_breaks:
                rejected.append(RejectedOverride(
                    segment_id=segment.segment_id, kind="pin", requested=requested,
                    reason=f"pinned count {requested} exceeds max_breaks {segment.max_breaks}",
                ))
            else:
                floors[segment.segment_id] = requested
                caps[segment.segment_id] = requested
            continue
        minimum = entry.get("min")
        if minimum is not None:
            requested = int(minimum)
            if requested > segment.max_breaks:
                rejected.append(RejectedOverride(
                    segment_id=segment.segment_id, kind="force", requested=requested,
                    reason=f"forced minimum {requested} exceeds max_breaks {segment.max_breaks}",
                ))
            else:
                floors[segment.segment_id] = requested

    # Verify each channel-day is compliant at its floors. If a pinned or forced
    # floor makes the group breach a guardrail (for example spacing), back that
    # override out and report it, rather than ship an out-of-policy plan.
    for group in groups.values():
        _reject_infeasible_floors(group, floors, caps, gold_by_id, guardrails, constraints, rejected)

    return floors, caps, gold_by_id, rejected


def _placements_in_bounds(segment: ProgramSegment, pins: Sequence[PlacementPin]) -> Optional[str]:
    """Reason the pins are invalid for the segment, or None when they are valid.

    Each break must sit inside the segment (``0 <= offset`` and
    ``offset + duration <= segment.duration_seconds``) and, ordered by offset, no
    two breaks may overlap (``prev.offset + prev.duration <= next.offset``).
    """
    for pin in pins:
        if pin.duration_seconds <= 0:
            return f"placement duration {pin.duration_seconds} must be positive"
        if pin.offset_seconds < 0:
            return f"placement offset {pin.offset_seconds} is before the segment start"
        if pin.offset_seconds + pin.duration_seconds > segment.duration_seconds + _EPSILON:
            return (
                f"placement at {pin.offset_seconds}s + {pin.duration_seconds}s exceeds "
                f"segment duration {segment.duration_seconds}s"
            )
    ordered = sorted(pins, key=lambda p: p.offset_seconds)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.offset_seconds + previous.duration_seconds > current.offset_seconds + _EPSILON:
            return "placements overlap within the segment"
    return None


def _apply_placement_pins(
    segs: list[ProgramSegment],
    groups: dict[tuple[str, str], list[ProgramSegment]],
    guardrails: Guardrails,
    placement_pins: Optional[Mapping[str, Sequence[PlacementPin]]],
    floors: dict[str, int],
    caps: dict[str, int],
    gold_by_id: dict[str, bool],
    rejected: list[RejectedOverride],
) -> dict[str, Sequence[PlacementPin]]:
    """Validate explicit placement pins and force the segments that carry them.

    For each pinned segment: validate the pins are in-bounds and non-overlapping,
    then run the guardrail checks on the pinned geometry for its channel-day. If
    either fails, the segment's pins are dropped (it falls back to a 0 floor and
    its normal cap) and a ``RejectedOverride(kind="placement")`` is recorded.
    A valid pin set forces ``floor == cap == len(pins)`` so every tier leaves the
    segment fixed, and is returned in the side map every emit / revenue path reads.
    """
    placements: dict[str, Sequence[PlacementPin]] = {}
    if not placement_pins:
        return placements

    seg_by_id = {s.segment_id: s for s in segs}
    for segment_id, pins in placement_pins.items():
        segment = seg_by_id.get(segment_id)
        if segment is None or not pins:
            continue
        reason = _placements_in_bounds(segment, pins)
        if reason is None:
            # Check the pinned geometry against the spacing / load guardrails in
            # isolation (per-segment); the channel-day check below catches breaches
            # that only show up once the whole group's pinned breaks are combined.
            probe = _segment_break_objects(segment, len(pins), pins=pins)
            if not is_compliant(probe, guardrails):
                reason = "pinned breaks breach a guardrail (spacing/load) for the segment"
        if reason is not None:
            rejected.append(RejectedOverride(
                segment_id=segment_id, kind="placement", requested=len(pins), reason=reason,
            ))
            # A rejected placement drops the segment to 0 breaks (the operator asked
            # for explicit geometry that cannot be honored; falling back to free
            # optimization would silently substitute different breaks).
            floors[segment_id] = 0
            caps[segment_id] = 0
            continue
        # Per-break gold lives on the individual PlacementPin (honored in the emit
        # path); it is NOT promoted to a segment-level gold flag, so a single gold
        # pin does not gild every break in the segment.
        placements[segment_id] = pins
        floors[segment_id] = len(pins)
        caps[segment_id] = len(pins)

    # Combined channel-day guardrail check: a group whose pinned floors breach a
    # guardrail has the largest pinned segment backed out one at a time until the
    # group's floor state is compliant, mirroring _reject_infeasible_floors.
    for group in groups.values():
        _reject_infeasible_placements(group, floors, caps, gold_by_id, guardrails, placements, rejected)
    return placements


def _reject_infeasible_placements(
    group: list[ProgramSegment],
    floors: dict[str, int],
    caps: dict[str, int],
    gold_by_id: dict[str, bool],
    guardrails: Guardrails,
    placements: dict[str, Sequence[PlacementPin]],
    rejected: list[RejectedOverride],
) -> None:
    """Back out pinned segments in a group until its floor geometry is compliant.

    Only placement-pinned segments are candidates (a non-pinned floor is left for
    the override path to handle), removed largest first, each recorded as a
    ``placement`` rejection.
    """
    state = {s.segment_id: floors[s.segment_id] for s in group}
    while not is_compliant(_group_breaks(group, state, gold_by_id, placements), guardrails):
        candidates = [s for s in group if s.segment_id in placements]
        if not candidates:
            break  # the infeasibility is not from a placement; leave it for reporting
        worst = max(candidates, key=lambda s: len(placements[s.segment_id]))
        rejected.append(RejectedOverride(
            segment_id=worst.segment_id, kind="placement", requested=len(placements[worst.segment_id]),
            reason="pinned breaks breach a guardrail for the channel-day (spacing/load)",
        ))
        del placements[worst.segment_id]
        state[worst.segment_id] = 0
        floors[worst.segment_id] = 0
        caps[worst.segment_id] = worst.max_breaks


def _reject_infeasible_floors(
    group: list[ProgramSegment],
    floors: dict[str, int],
    caps: dict[str, int],
    gold_by_id: dict[str, bool],
    guardrails: Guardrails,
    constraints: dict[str, dict[str, object]],
    rejected: list[RejectedOverride],
) -> None:
    """Back out pin/force floors in a group until its floor state is compliant.

    Removes the largest offending override-floored segment one at a time until
    the group's floors are guardrail-compliant, recording each as rejected. A
    forbid (cap 0) is never the cause and never backed out.
    """
    state = {s.segment_id: floors[s.segment_id] for s in group}
    while not is_compliant(_group_breaks(group, state, gold_by_id), guardrails):
        candidates = [
            s for s in group
            if state[s.segment_id] > 0 and _is_override_floored(s.segment_id, constraints)
        ]
        if not candidates:
            break  # the infeasibility is not from an override; leave it for reporting
        worst = max(candidates, key=lambda s: state[s.segment_id])
        entry = constraints.get(worst.segment_id, {})
        kind = "pin" if entry.get("pin") is not None else "force"
        rejected.append(RejectedOverride(
            segment_id=worst.segment_id, kind=kind, requested=state[worst.segment_id],
            reason="override floor breaks a guardrail for its channel-day (spacing/load)",
        ))
        state[worst.segment_id] = 0
        floors[worst.segment_id] = 0
        if kind == "pin":
            caps[worst.segment_id] = worst.max_breaks


def _is_override_floored(segment_id: str, constraints: dict[str, dict[str, object]]) -> bool:
    entry = constraints.get(segment_id, {})
    return entry.get("pin") is not None or entry.get("min") is not None


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
    gold_by_id: Optional[dict[str, bool]] = None,
    rejected: Optional[list[RejectedOverride]] = None,
    original_by_id: Optional[dict[str, ProgramSegment]] = None,
    risk_lambda: float = 0.0,
    placements: Optional[dict[str, Sequence[PlacementPin]]] = None,
) -> OptimizationResult:
    gold_by_id = gold_by_id or {}
    original_by_id = original_by_id or {}
    pin_placements = placements or {}
    flat_placements: list[BreakPlacement] = []
    segment_plans: list[SegmentPlan] = []
    for segment in segs:
        k = state[segment.segment_id]
        retention = _segment_retention(segment, k)
        gold = segment.is_gold or gold_by_id.get(segment.segment_id, False)
        pins = pin_placements.get(segment.segment_id)
        breaks = _segment_break_objects(segment, k, is_gold=gold, pins=pins)
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
                revenue=_marginal_revenue(segment, index, pins),
                is_gold=brk.is_gold,
            ))
        flat_placements.extend(segment_placements)
        # ``segment`` carries the risk-adjusted coefficient the decision used; the
        # original carries the point estimate and the interval behind it.
        original = original_by_id.get(segment.segment_id, segment)
        segment_plans.append(SegmentPlan(
            segment_id=segment.segment_id,
            num_breaks=k,
            retention=retention,
            revenue=_segment_revenue(segment, k, pins),
            placements=tuple(segment_placements),
            retention_cost_point=original.impact_coefficient,
            retention_cost_used=segment.impact_coefficient,
            retention_cost_ci_low=original.impact_ci_low,
            retention_cost_ci_high=original.impact_ci_high,
            retention_cost_n=original.impact_n,
            retention_confidence=original.impact_confidence,
        ))

    all_breaks = [
        Break(
            channel=p.channel, day=p.day, hour=p.hour,
            start_seconds=p.start_seconds, duration_seconds=p.duration_seconds,
            program_type=p.program_type, retention=p.retention, is_gold=p.is_gold,
        )
        for p in flat_placements
    ]
    return OptimizationResult(
        segments=tuple(segment_plans),
        placements=tuple(flat_placements),
        total_revenue=total_revenue,
        aggregate_retention=aggregate_retention,
        objective=objective,
        violations=tuple(evaluate(all_breaks, guardrails)),
        revenue_weight=revenue_weight,
        revenue_scale=revenue_scale,
        decisions=tuple(decisions),
        rejected_overrides=tuple(rejected or ()),
        risk_lambda=risk_lambda,
    )
