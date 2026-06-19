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

Objective form (convex blend, not a subtraction)
-------------------------------------------------
The scalar this optimizer maximises is:

    objective = revenue_weight * clamp(revenue / revenue_scale, 0, 1)
              + (1 - revenue_weight) * clamp(aggregate_retention, 0, 1)

Both terms are clamped into [0, 1], so the result is also in [0, 1]. This is a
CONVEX BLEND of the two objectives, not "revenue minus retention cost." The
per-group contribution returned by
:func:`~kairos.optimize._segment_math._group_objective_contribution` is the
additive share of this same blend (revenue is a sum, retention is a
tvr-weighted sum, both divided by the same global constants), so summing every
group's contribution exactly reproduces the global convex-blend objective. Any
description that calls the objective "revenue minus retention" is imprecise:
the retention term is the audience-weighted RETENTION SHARE (a number in
[0, 1]), not a monetary retention cost subtracted from revenue.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Iterable, Mapping, Optional, Sequence

from kairos.optimize.guardrails import Break, Guardrails, Violation, evaluate, is_compliant
from kairos.optimize.objective import STANDARD_UNIT_SECONDS, weighted_objective
from kairos.optimize.overrides import OverrideSet

# Re-export all public types from _types so callers can keep importing from here.
from kairos.optimize._types import (  # noqa: F401
    BreakPlacement,
    Decision,
    DEFAULT_BREAK_LENGTH_SECONDS,
    OptimizationResult,
    PlacementPin,
    ProgramSegment,
    RejectedOverride,
    SegmentPlan,
)

# Re-export the math primitives refiner.py and other internal callers use.
from kairos.optimize._segment_math import (  # noqa: F401
    _EPSILON,
    _group_breaks,
    _group_objective_contribution,
    _marginal_revenue,
    _risk_adjusted_coefficient,
    _segment_break_objects,
    _segment_retention,
    _segment_revenue,
)

from kairos.optimize._override_logic import (
    _apply_placement_pins,
    _apply_segment_overrides,
)

SECONDS_PER_HOUR = 3600.0


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

    The objective is a convex blend:

        revenue_weight * clamp(revenue / revenue_scale, 0, 1)
        + (1 - revenue_weight) * clamp(aggregate_retention, 0, 1)

    Both terms live in [0, 1]. ``revenue_weight`` in [0, 1] sets the balance:
    1.0 chases revenue only and fills every segment up to the guardrails, 0.0
    protects retention only and places no breaks. ``revenue_scale`` normalises
    revenue so it is comparable to retention; when omitted it defaults to the
    revenue of loading every segment to ``max_breaks`` (the marketing-maximal
    reference), floored above zero.

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
    interval: 0.0 (the default) decides with the point estimate and changes
    nothing, 1.0 decides with the worst plausible cost in the interval, and
    values in between apply a partial variance penalty (see
    :func:`~kairos.optimize.objective.conservative_impact`). A segment with
    only a point coefficient is unaffected at any ``risk_lambda``.

    ``placement_pins`` maps a segment id to an explicit list of
    :class:`PlacementPin` (absolute offset-from-start, per-break duration, gold
    flag). A pinned segment is fixed at exactly those breaks: its count is forced
    to ``len(pins)`` and every tier emits the breaks at the pinned positions and
    durations, with revenue summed over the per-break durations. Pins are
    validated first (in-bounds and non-overlapping, then the spacing / load
    guardrails on the pinned geometry); a segment whose pins are invalid or breach
    a guardrail is dropped to 0 breaks and reported in
    ``result.rejected_overrides`` with ``kind="placement"``, never silently bent.

    ``demand_weights`` is an optional mapping from segment id to a placement-
    preference weight >= 1.0. When supplied, the greedy ranking step multiplies
    each segment's apparent objective gain by its weight before comparing
    segments, so a higher-demand segment is preferred when two segments have
    similar gains. This biases WHERE breaks go without changing reported revenue:
    weights touch only the ranking comparison, never ``total_revenue`` or any
    ``SegmentPlan`` revenue field. A missing or 1.0 weight leaves a segment's
    ranking unchanged. Omitting the argument entirely (``None``) gives
    byte-identical output to today's optimizer. Produced by
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
    # otherwise. The originals are kept so the plan can still report the point,
    # the interval and the confidence behind each segment's decision.
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
            # always real. A weight of 1.0 (or None) leaves ranking unchanged; a
            # weight above 1.0 boosts a segment, below 1.0 de-prioritizes it
            # (over-delivered campaign). The bias is applied only to a positive
            # gain: a non-positive gain never wins the greedy step anyway (best_gain
            # starts at _EPSILON > 0), so gating on gain > 0 keeps the sign safe and
            # lets a sub-1.0 weight genuinely lower a candidate's rank.
            if demand_weights is not None and gain > 0.0:
                weight = demand_weights.get(segment.segment_id, 1.0)
                gain = gain * (weight if weight > 0.0 else _EPSILON)
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
