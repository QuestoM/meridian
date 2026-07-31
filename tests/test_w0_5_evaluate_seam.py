"""The cheap scoring seam returns the expensive path's own numbers.

:mod:`kairos.optimize.evaluate` exists so a person moving a break can see the
money move without paying for a re-allocation. That is only worth anything if
the cheap number IS the plan's number, so these tests hold it to the optimizer's
own reported totals: on synthetic segments across the weight, risk and pin
dimensions, and on one real channel-day to the shekel.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from kairos.optimize._types import PlacementPin
from kairos.optimize.evaluate import (
    Evaluation,
    counts_from_result,
    evaluate_day,
    evaluation_basis,
    pins_from_result,
    score,
)
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks

ROOT = Path(__file__).resolve().parents[1]
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
        retention_baseline=1.0,
        premium=1.0,
        is_gold=False,
        max_breaks=4,
        break_length_seconds=120.0,
    )
    base.update(overrides)
    return ProgramSegment(**base)


def a_day(count: int = 6) -> list[ProgramSegment]:
    """A synthetic channel-day whose segments differ in every scored dimension."""
    return [
        make_segment(
            segment_id=f"s{index}",
            start_seconds=(8 + index) * 3600.0,
            baseline_tvr=4.0 + index,
            cpp=800.0 + 50.0 * index,
            impact_coefficient=-0.01 - 0.004 * index,
            premium=1.0 + 0.1 * index,
        )
        for index in range(count)
    ]


@pytest.mark.parametrize("revenue_weight", [0.0, 0.35, 0.6, 1.0])
def test_evaluation_equals_the_optimizer_totals(revenue_weight: float) -> None:
    segments = a_day()
    result = optimize_breaks(segments, GR, revenue_weight=revenue_weight)
    evaluation = evaluate_day(
        segments, counts_from_result(result), revenue_weight=revenue_weight,
    )
    assert evaluation.revenue == pytest.approx(result.total_revenue, abs=1e-9)
    assert evaluation.retention == pytest.approx(result.aggregate_retention, abs=1e-12)
    assert evaluation.objective == pytest.approx(result.objective, abs=1e-12)


def test_evaluation_uses_the_optimizers_own_revenue_scale() -> None:
    segments = a_day()
    result = optimize_breaks(segments, GR, revenue_weight=0.6)
    basis = evaluation_basis(segments)
    assert basis.revenue_scale == pytest.approx(result.revenue_scale, abs=1e-9)
    assert basis.total_tvr == pytest.approx(sum(s.baseline_tvr for s in segments), abs=1e-12)


def test_risk_adjustment_is_applied_exactly_as_the_optimizer_does() -> None:
    """With an interval and a positive lambda the decision coefficient moves.

    Scoring the raw segments would report a different retention cost, so this is
    the test that a caller cannot silently get an unadjusted number.
    """
    segments = [
        make_segment(
            segment_id=f"s{index}",
            start_seconds=(10 + index) * 3600.0,
            impact_coefficient=-0.02,
            impact_ci_low=-0.05,
            impact_ci_high=-0.005,
        )
        for index in range(4)
    ]
    result = optimize_breaks(segments, GR, revenue_weight=0.6, risk_lambda=0.8)
    adjusted = evaluate_day(
        segments, counts_from_result(result), revenue_weight=0.6, risk_lambda=0.8,
    )
    assert adjusted.revenue == pytest.approx(result.total_revenue, abs=1e-9)
    assert adjusted.retention == pytest.approx(result.aggregate_retention, abs=1e-12)
    unadjusted = evaluate_day(segments, counts_from_result(result), revenue_weight=0.6)
    assert unadjusted.retention != pytest.approx(result.aggregate_retention, abs=1e-9)


def test_pinned_placements_score_at_their_own_durations() -> None:
    """A pin map is honoured, and it is what makes a drag scorable."""
    segments = a_day(3)
    pins = {
        "s0": (PlacementPin(offset_seconds=600.0, duration_seconds=90.0),
               PlacementPin(offset_seconds=2400.0, duration_seconds=150.0)),
    }
    result = optimize_breaks(segments, GR, revenue_weight=0.6, placement_pins=pins)
    evaluation = evaluate_day(
        segments, counts_from_result(result), revenue_weight=0.6, placements=pins,
    )
    assert evaluation.revenue == pytest.approx(result.total_revenue, abs=1e-9)
    assert evaluation.objective == pytest.approx(result.objective, abs=1e-12)
    # The same counts scored without the pins value those two breaks at the
    # segment's own break length, so the number is different and the pin map is
    # doing real work rather than riding along.
    unpinned = evaluate_day(segments, counts_from_result(result), revenue_weight=0.6)
    assert unpinned.revenue != pytest.approx(result.total_revenue, abs=1.0)


def test_recovered_pins_reproduce_the_plan_geometry() -> None:
    segments = a_day(3)
    result = optimize_breaks(segments, GR, revenue_weight=0.6)
    recovered = pins_from_result(result, segments)
    counts = counts_from_result(result)
    for segment_id, count in counts.items():
        assert len(recovered.get(segment_id, ())) == count
    evaluation = score(
        evaluation_basis(segments), counts, revenue_weight=0.6, placements=recovered,
    )
    assert evaluation.revenue == pytest.approx(result.total_revenue, abs=1e-9)


def test_a_move_that_earns_more_scores_higher() -> None:
    """The seam has to be able to say a placement is better, not just equal."""
    segments = a_day(3)
    basis = evaluation_basis(segments)
    counts = {segment.segment_id: 1 for segment in segments}
    one = score(basis, counts, revenue_weight=1.0)
    counts["s2"] = 2
    two = score(basis, counts, revenue_weight=1.0)
    assert two.revenue > one.revenue
    assert two.objective > one.objective
    assert two.retention < one.retention


def test_missing_count_is_refused_rather_than_scored() -> None:
    segments = a_day(3)
    basis = evaluation_basis(segments)
    with pytest.raises(ValueError, match="missing"):
        score(basis, {"s0": 1, "s1": 1}, revenue_weight=0.6)


def test_empty_day_scores_as_the_optimizer_reports_it() -> None:
    result = optimize_breaks([], GR)
    evaluation = evaluate_day([], {}, revenue_weight=0.6)
    assert evaluation.revenue == result.total_revenue == 0.0
    assert evaluation.retention == result.aggregate_retention == 1.0


def test_evaluation_is_the_three_number_tuple_the_contract_promises() -> None:
    objective, revenue, retention = evaluate_day(
        a_day(2), {"s0": 1, "s1": 1}, revenue_weight=0.6,
    )
    assert isinstance(Evaluation(objective, revenue, retention), tuple)
    assert 0.0 <= objective <= 1.0
    assert revenue > 0.0
    assert 0.0 <= retention <= 1.0


@pytest.fixture(scope="module")
def real_day():
    """One real channel-day through the commit path's own seams."""
    from kairos_api.preview_inputs import preview_inputs

    try:
        from kairos.data.loaders import load_programmes

        programmes = load_programmes()
    except Exception as exc:  # pragma: no cover - environment without reference data
        pytest.skip(f"no programmes reference data: {exc}")
    valid = programmes[programmes["start_dt"].notna()]
    if valid.empty:  # pragma: no cover - environment without reference data
        pytest.skip("programmes reference has no parseable rows")
    channel = str(valid["Channel"].iloc[0])
    mine = valid[valid["Channel"].astype(str) == channel]
    day = mine["start_dt"].dt.strftime("%Y-%m-%d").min()
    segments, engine_kwargs = preview_inputs(channel, day, None)
    if not segments:  # pragma: no cover - environment without reference data
        pytest.skip(f"no segments for {channel} {day}")
    return segments, engine_kwargs


def test_real_day_matches_the_expensive_path_to_the_shekel(real_day) -> None:
    """The bar, on real data: the cheap answer is the plan's own answer."""
    from kairos.optimize.day_core import _optimize_one_day

    segments, engine_kwargs = real_day
    result = _optimize_one_day(segments, **engine_kwargs)
    evaluation = evaluate_day(
        segments,
        counts_from_result(result),
        revenue_weight=engine_kwargs["revenue_weight"],
        risk_lambda=engine_kwargs["risk_lambda"],
    )
    assert abs(evaluation.revenue - result.total_revenue) < 1.0
    assert evaluation.revenue == pytest.approx(result.total_revenue, rel=1e-12)
    assert evaluation.retention == pytest.approx(result.aggregate_retention, rel=1e-12)
    assert evaluation.objective == pytest.approx(result.objective, rel=1e-12)


def test_real_day_scores_far_faster_than_it_optimizes(real_day) -> None:
    """The whole point: a score is orders of magnitude cheaper than a plan."""
    from kairos.optimize.day_core import _optimize_one_day

    segments, engine_kwargs = real_day
    start = time.perf_counter()
    result = _optimize_one_day(segments, **engine_kwargs)
    optimize_seconds = time.perf_counter() - start

    counts = counts_from_result(result)
    basis = evaluation_basis(segments, risk_lambda=engine_kwargs["risk_lambda"])
    start = time.perf_counter()
    for _ in range(10):
        score(basis, counts, revenue_weight=engine_kwargs["revenue_weight"])
    score_seconds = (time.perf_counter() - start) / 10

    assert score_seconds < 0.005
    assert optimize_seconds / score_seconds > 100
