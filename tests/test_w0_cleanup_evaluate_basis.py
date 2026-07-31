"""A basis narrower than the plan it scores is refused, not scored.

W0-5's scoring seam normalises both halves of the convex blend by constants
summed over the basis. It shipped letting a caller override ``revenue_scale``
with no way to supply ``total_tvr``, and :func:`score` dropped a count for a
segment the basis did not carry. Both make the same failure: a confidently wrong
number from a partial basis, with nothing on the surface to notice.

The measurement that says this is worth a refusal rather than a warning, taken
on the real 82-segment channel-day רשת 13 / 2024-11-01:

  whole day, both constants derived      objective 0.542990, revenue 1,067,797
  half the day, both constants derived   objective 0.542990, revenue   426,777
  half the day, the whole day's scale    objective 0.445411, revenue   426,777

The middle row is the seam working as documented on a scope of its own. The last
row is the accident the shipped signature made easy: a caller doing the
responsible thing with the one constant it could pass got a blend of a whole-day
revenue term and a half-day retention term, a number describing no scope at all.

These tests use synthetic segments, because the defect is about the shape of the
call and not about the data.
"""

from __future__ import annotations

import pytest

from kairos.optimize._types import PlacementPin
from kairos.optimize.evaluate import (
    counts_from_result,
    evaluate_day,
    evaluation_basis,
    score,
)
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks

GR = Guardrails()


def make_segment(**overrides) -> ProgramSegment:
    base = dict(
        segment_id="s1",
        channel="רשת 13",
        day="2024-11-01",
        start_seconds=21 * 3600.0,
        duration_seconds=3600.0,
        program_type="Drama",
        baseline_tvr=10.0,
        cpp=1000.0,
        impact_coefficient=-0.02,
        max_breaks=4,
        break_length_seconds=120.0,
    )
    base.update(overrides)
    return ProgramSegment(**base)


def a_day(count: int = 6) -> list[ProgramSegment]:
    return [
        make_segment(
            segment_id=f"s{index}",
            start_seconds=(8 + index) * 3600.0,
            baseline_tvr=4.0 + index,
            cpp=800.0 + 50.0 * index,
            impact_coefficient=-0.01 - 0.004 * index,
        )
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------

def test_a_basis_narrower_than_the_arrangement_is_refused() -> None:
    """The gap itself: the whole day's counts against part of the day's basis.

    Before the fix this returned the half day's revenue with no complaint, and
    a surface would have shown it as the day's.
    """
    day = a_day(6)
    counts = {segment.segment_id: 1 for segment in day}
    partial = evaluation_basis(day[:3])

    with pytest.raises(ValueError, match="the basis does not carry"):
        score(partial, counts, revenue_weight=0.6)

    # The control: the same counts against the basis they belong to score fine.
    whole = evaluation_basis(day)
    assert score(whole, counts, revenue_weight=0.6).revenue > 0.0


def test_a_pin_for_a_segment_outside_the_basis_is_refused() -> None:
    """A pin map is an arrangement too, and it was dropped just as silently."""
    day = a_day(4)
    basis = evaluation_basis(day[:2])
    counts = {segment.segment_id: 1 for segment in day[:2]}
    pins = {"s3": (PlacementPin(offset_seconds=600.0, duration_seconds=90.0),)}

    with pytest.raises(ValueError, match="the basis does not carry"):
        score(basis, counts, revenue_weight=0.6, placements=pins)


def test_one_constant_without_the_other_is_refused() -> None:
    """The mixed basis, which is the state the shipped signature made easy."""
    day = a_day(6)
    whole = evaluation_basis(day)

    with pytest.raises(ValueError, match="total_tvr"):
        evaluation_basis(day[:3], revenue_scale=whole.revenue_scale)
    with pytest.raises(ValueError, match="revenue_scale"):
        evaluation_basis(day[:3], total_tvr=whole.total_tvr)
    with pytest.raises(ValueError, match="total_tvr"):
        evaluate_day(
            day[:3], {s.segment_id: 1 for s in day[:3]},
            revenue_weight=0.6, revenue_scale=whole.revenue_scale,
        )


def test_a_duplicated_segment_is_refused_rather_than_counted_twice() -> None:
    day = a_day(3)
    with pytest.raises(ValueError, match="duplicate segment_id"):
        evaluation_basis(day + day[:1])


def test_a_missing_count_is_still_refused() -> None:
    """The direction that already worked keeps working, and keeps its message."""
    basis = evaluation_basis(a_day(3))
    with pytest.raises(ValueError, match="missing"):
        score(basis, {"s0": 1, "s1": 1}, revenue_weight=0.6)


# ---------------------------------------------------------------------------
# What the constants now let a caller say
# ---------------------------------------------------------------------------

def test_a_subset_scored_against_the_whole_is_that_subsets_contribution() -> None:
    """Both constants supplied: the part's honest share of the whole's blend.

    The three parts of a partitioned day sum to the whole day's objective and
    the whole day's revenue, which is the additivity
    ``_group_objective_contribution`` documents. That is the property a partial
    basis has to have to be worth anything, and it is only reachable now that
    both constants can be passed.
    """
    day = a_day(6)
    counts = {segment.segment_id: 1 for segment in day}
    whole = evaluation_basis(day)
    whole_score = score(whole, counts, revenue_weight=0.6)

    parts = [day[0:2], day[2:4], day[4:6]]
    scores = [
        score(
            evaluation_basis(
                part, revenue_scale=whole.revenue_scale, total_tvr=whole.total_tvr,
            ),
            {segment.segment_id: 1 for segment in part},
            revenue_weight=0.6,
        )
        for part in parts
    ]

    assert sum(part.revenue for part in scores) == pytest.approx(whole_score.revenue, rel=1e-12)
    # The blend is additive over the parts once both constants are the whole's,
    # because each part's retention term is its own share of the whole's tvr
    # rather than a share of its own. With the shipped signature the parts could
    # not be made to add up at all: only one of the two constants could be
    # passed, so each part's retention was normalised by its own tvr.
    assert sum(part.objective for part in scores) == pytest.approx(
        whole_score.objective, rel=1e-12,
    )


def test_derived_records_which_claim_the_basis_is_making() -> None:
    day = a_day(4)
    whole = evaluation_basis(day)
    assert whole.derived is True
    part = evaluation_basis(
        day[:2], revenue_scale=whole.revenue_scale, total_tvr=whole.total_tvr,
    )
    assert part.derived is False
    assert part.revenue_scale == whole.revenue_scale
    assert part.total_tvr == whole.total_tvr


def test_the_engine_default_is_unchanged_by_all_of_this() -> None:
    """Bar 3: the seam still returns the optimizer's own totals, to the bit."""
    day = a_day(6)
    for revenue_weight in (0.0, 0.35, 0.6, 1.0):
        result = optimize_breaks(day, GR, revenue_weight=revenue_weight)
        evaluation = evaluate_day(
            day, counts_from_result(result), revenue_weight=revenue_weight,
        )
        assert evaluation.revenue == pytest.approx(result.total_revenue, abs=1e-9)
        assert evaluation.retention == pytest.approx(result.aggregate_retention, abs=1e-12)
        assert evaluation.objective == pytest.approx(result.objective, abs=1e-12)


def test_an_empty_basis_still_scores_as_the_optimizer_reports_it() -> None:
    """The empty day is a real state and none of the new refusals may take it."""
    result = optimize_breaks([], GR)
    evaluation = evaluate_day([], {}, revenue_weight=0.6)
    assert evaluation.revenue == result.total_revenue == 0.0
    assert evaluation.retention == result.aggregate_retention == 1.0
