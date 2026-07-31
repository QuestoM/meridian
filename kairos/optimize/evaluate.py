"""What a placement is worth, without allocating one.

The optimizer answers "where should the breaks go". A person dragging a break
across a timeline is asking something much smaller: "what is THIS arrangement
worth". Those are not the same question, and answering the small one by running
the large one is why a drag cannot move a money figure inside a person's
attention span. Measured on one real channel-day of 82 segments,
``_optimize_one_day`` is 0.87 s per leg while
:func:`~kairos.optimize._segment_math._group_objective_contribution` over the
same segments is 60 microseconds, a ratio of about 15,000 to one.

So this module is the small question's own seam. Given the built segments, a
break count per segment and an optional placement map, it returns
``(objective, revenue, retention)`` and allocates nothing.

It is a thin wrapper, deliberately. ``_group_objective_contribution`` is already
pure and already additive by its own docstring, and the reported objective is
already :func:`~kairos.optimize.objective.weighted_objective`. What this module
adds is the three details that make the cheap answer equal the expensive one
rather than merely resemble it:

1. **The same risk adjustment.** :func:`~kairos.optimize.optimizer.optimize_breaks`
   decides and reports against segments whose ``impact_coefficient`` has been
   replaced by the risk-adjusted value. Scoring the raw segments would quietly
   report a different retention cost whenever ``risk_lambda`` is above zero.
2. **The same normalising constants.** ``revenue_scale`` defaults to the revenue
   of every segment at its ``max_breaks``, and ``total_tvr`` to the sum of the
   baseline ratings, both over the same risk-adjusted segments the optimizer
   used. A different scale is a different objective.
3. **The same order.** The optimizer sorts by ``segment_id`` before it sums, and
   floating-point addition is not associative, so this sums in that order too.

The result is the number the plan reports, not an approximation of it: on the
real channel-day the difference from the optimizer's own totals is exactly zero
in revenue, retention and objective. Two honest limits. The score is the engine
basis, so with the owner-gated quarter-hour settlement switched on the plan's
restated revenue differs by construction (see
:mod:`kairos.optimize.qh_billing`). And a score is not a compliance check: a
placement that breaks a guardrail still scores, so a caller that lets a person
move a break must also run :func:`kairos.optimize.guardrails.evaluate`.

**What the basis has to be, and what happens when it is not.** Both normalising
constants divide totals summed over the basis, so a basis narrower than the
arrangement being scored reports a confidently wrong number rather than a
missing one. The first round of this module let a caller override
``revenue_scale`` and gave no way to supply ``total_tvr``, which made the
dangerous state the easy one: measured on the real 82-segment channel-day
``רשת 13 / 2024-11-01`` with one break in every segment, scoring its first 41
segments with the whole day's ``revenue_scale`` and their own derived
``total_tvr`` reports objective 0.445411 where the day reports 0.542990, because
the blend mixed a whole-day revenue term with a half-day retention term. So:

1. ``total_tvr`` is now an explicit argument, exactly as ``revenue_scale`` is.
   The two constants travel together and supplying one without the other is
   refused, because that combination cannot describe any single scope.
2. :func:`score` refuses an arrangement wider than its basis. ``counts`` or
   ``placements`` naming a segment the basis does not carry used to be dropped
   in silence, which is a whole day's counts scored against part of a day and
   reported as the day.
3. Supplying neither constant is the declaration that these segments **are** the
   whole scope, and the constants are derived from them. The seam cannot check
   that claim, and says so rather than implying it verified it. A caller holding
   a genuine subset supplies both constants from the whole and gets that
   subset's honest contribution to the whole.

A basis may span more than one channel-day. That is not an error and is not
refused: ``_group_objective_contribution`` is additive over groups against
global constants, so a multi-day basis reports the global blend, which is what
the optimizer itself computes when it is handed several days.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence

from kairos.optimize._segment_math import (
    _EPSILON,
    _group_objective_contribution,
    _risk_adjusted_coefficient,
    _segment_revenue,
)
from kairos.optimize._types import PlacementPin, ProgramSegment
from kairos.optimize.objective import weighted_objective


class Evaluation(NamedTuple):
    """What one arrangement is worth.

    ``objective`` is the reported convex blend, ``revenue`` is ILS on the engine
    basis, and ``retention`` is the tvr-weighted share the plan keeps, the same
    three numbers :class:`~kairos.optimize._types.OptimizationResult` reports.
    """

    objective: float
    revenue: float
    retention: float


class Basis(NamedTuple):
    """The constants one scope, normally one channel-day, is scored against.

    Built once and reused across many scores, which is what makes re-scoring a
    drag cost microseconds: ``segments`` are risk-adjusted and sorted exactly as
    the optimizer sorts them, and the two constants normalise the blend.

    ``derived`` records where the constants came from: true when they were
    summed over these very segments, which is the claim that this basis is its
    own whole, and false when the caller supplied both from a wider scope, which
    is the claim that these segments are a part of it.
    """

    segments: tuple[ProgramSegment, ...]
    revenue_scale: float
    total_tvr: float
    derived: bool = True


def evaluation_basis(
    segments: Iterable[ProgramSegment],
    *,
    risk_lambda: float = 0.0,
    revenue_scale: Optional[float] = None,
    total_tvr: Optional[float] = None,
) -> Basis:
    """The scoring constants for one scope, derived as the optimizer derives them.

    Supply neither constant and both are summed over ``segments``, which is the
    engine default and the declaration that these segments are the whole scope:
    ``revenue_scale`` is the revenue of every segment loaded to its
    ``max_breaks`` and ``total_tvr`` is the sum of the baseline ratings.

    Supply both and this basis is scored as a part of the scope those constants
    describe, which is what a caller holding a window of a day needs.

    Supplying exactly one is refused. The two constants divide the two halves of
    one convex blend, so one from the whole and one from a part is a number with
    no scope at all, and it is the state a caller falls into by accident: it is
    what this seam produced before ``total_tvr`` could be passed.
    """
    ordered = tuple(sorted(segments, key=lambda segment: segment.segment_id))
    duplicates = len(ordered) - len({segment.segment_id for segment in ordered})
    if duplicates:
        raise ValueError(
            f"the basis carries {duplicates} duplicate segment_id(s); every segment "
            "would be counted once per copy, so the totals would not be the day's"
        )
    if risk_lambda > 0.0:
        ordered = tuple(
            replace(segment, impact_coefficient=_risk_adjusted_coefficient(segment, risk_lambda))
            for segment in ordered
        )
    derived = revenue_scale is None and total_tvr is None
    if (revenue_scale is None) != (total_tvr is None):
        supplied, missing = (
            ("revenue_scale", "total_tvr") if total_tvr is None
            else ("total_tvr", "revenue_scale")
        )
        raise ValueError(
            f"{supplied} was supplied and {missing} was not; both constants "
            "normalise one blend, so they must describe the same scope. Pass both "
            "from the whole scope, or neither to derive both from these segments"
        )
    if revenue_scale is None:
        full = sum(_segment_revenue(segment, segment.max_breaks) for segment in ordered)
        revenue_scale = max(full, _EPSILON)
    elif revenue_scale <= 0:
        raise ValueError("revenue_scale must be positive")
    if total_tvr is None:
        total_tvr = sum(segment.baseline_tvr for segment in ordered)
    elif total_tvr < 0:
        raise ValueError("total_tvr must not be negative")
    return Basis(
        segments=ordered, revenue_scale=revenue_scale, total_tvr=total_tvr, derived=derived,
    )


def score(
    basis: Basis,
    counts: Mapping[str, int],
    *,
    revenue_weight: float,
    placements: Optional[Mapping[str, Sequence[PlacementPin]]] = None,
) -> Evaluation:
    """Score one arrangement against a prepared basis. This is the cheap call.

    The arrangement and the basis must describe the same segments, exactly, in
    both directions. A missing count would score a plan that is not the plan. A
    count or a pin for a segment the basis does not carry is the more dangerous
    half: it used to be dropped in silence, so a caller holding a whole day's
    counts and a partial basis got the part's revenue reported as the day's,
    with nothing to notice.
    """
    known = {segment.segment_id for segment in basis.segments}
    missing = sorted(known - set(counts))
    if missing:
        raise ValueError(
            f"counts is missing {len(missing)} segment(s), first {missing[0]!r}; "
            "every segment in the basis must carry a break count"
        )
    unknown = sorted(set(counts) - known) + sorted(set(placements or {}) - known)
    if unknown:
        raise ValueError(
            f"the arrangement names {len(unknown)} segment(s) the basis does not "
            f"carry, first {unknown[0]!r}; the basis is narrower than the plan "
            "being scored, so its totals would be a part reported as the whole"
        )
    _, revenue, retention_weighted = _group_objective_contribution(
        list(basis.segments), dict(counts),
        revenue_weight=revenue_weight,
        revenue_scale=basis.revenue_scale,
        total_tvr=basis.total_tvr,
        placements=dict(placements) if placements else None,
    )
    retention = (
        retention_weighted / basis.total_tvr if basis.total_tvr > _EPSILON else 1.0
    )
    objective = weighted_objective(
        revenue, retention, revenue_weight=revenue_weight, revenue_scale=basis.revenue_scale,
    )
    return Evaluation(objective=objective, revenue=revenue, retention=retention)


def evaluate_day(
    segments: Iterable[ProgramSegment],
    counts: Mapping[str, int],
    *,
    revenue_weight: float,
    risk_lambda: float = 0.0,
    placements: Optional[Mapping[str, Sequence[PlacementPin]]] = None,
    revenue_scale: Optional[float] = None,
    total_tvr: Optional[float] = None,
) -> Evaluation:
    """Score one arrangement from raw segments, basis included.

    The convenience form for a caller that scores once. A surface that scores
    the same day repeatedly (a drag, a slider) should build the
    :func:`evaluation_basis` once and call :func:`score` per arrangement. The
    two constants carry the meaning and the refusals they carry there.
    """
    basis = evaluation_basis(
        segments, risk_lambda=risk_lambda, revenue_scale=revenue_scale, total_tvr=total_tvr,
    )
    return score(basis, counts, revenue_weight=revenue_weight, placements=placements)


def counts_from_result(result: Any) -> dict[str, int]:
    """The break count per segment from a finished optimization result."""
    return {plan.segment_id: plan.num_breaks for plan in result.segments}


def pins_from_result(
    result: Any, segments: Iterable[ProgramSegment],
) -> dict[str, tuple[PlacementPin, ...]]:
    """The placed breaks as pins, offsets measured from each segment's start.

    For a plan the operator has pinned, this recovers the pin map to re-score
    after an edit. For a plan the optimizer spaced itself, the recovered pins
    describe the same geometry, so scoring with them or with ``None`` gives the
    same revenue; ``None`` is the honest argument in that case, because it says
    nothing was pinned.
    """
    start_of = {segment.segment_id: segment.start_seconds for segment in segments}
    pins: dict[str, list[PlacementPin]] = {}
    for placement in result.placements:
        start = start_of.get(placement.segment_id)
        if start is None:
            continue
        pins.setdefault(placement.segment_id, []).append(PlacementPin(
            offset_seconds=placement.start_seconds - start,
            duration_seconds=placement.duration_seconds,
            is_gold=placement.is_gold,
        ))
    return {segment_id: tuple(items) for segment_id, items in pins.items()}
