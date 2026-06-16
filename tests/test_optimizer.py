"""Tests for the greedy break optimizer.

These prove the two properties that matter: the schedule it returns is always
guardrail-compliant, and it spends the revenue-versus-retention trade-off the
way the weight asks it to.
"""

from __future__ import annotations

import pytest

from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks

GR = Guardrails()


def make_segment(**overrides) -> ProgramSegment:
    base = dict(
        segment_id="s1",
        channel="קשת 12",
        day="Mon",
        start_seconds=21 * 3600.0,   # 21:00
        duration_seconds=3600.0,     # one hour
        program_type="Drama",
        baseline_tvr=10.0,
        cpp=1000.0,
        impact_coefficient=0.0,
        retention_baseline=1.0,
        premium=1.0,
        is_gold=False,
        max_breaks=4,
        break_length_seconds=120.0,
    )
    base.update(overrides)
    return ProgramSegment(**base)


def test_empty_schedule_is_well_formed() -> None:
    result = optimize_breaks([], GR)
    assert result.segments == ()
    assert result.placements == ()
    assert result.total_breaks == 0
    assert result.total_revenue == 0.0
    assert result.aggregate_retention == 1.0
    assert result.is_compliant


def test_retention_only_places_no_breaks() -> None:
    # Any break lowers retention and revenue does not count, so the best move
    # is to place nothing.
    result = optimize_breaks([make_segment(impact_coefficient=-0.1)], GR, revenue_weight=0.0)
    assert result.total_breaks == 0
    assert result.segments[0].num_breaks == 0
    assert result.aggregate_retention == 1.0
    assert result.is_compliant


def test_revenue_only_fills_to_breaks_per_hour_limit() -> None:
    # No retention penalty and revenue-only, so it fills until a guardrail bites.
    guardrails = Guardrails(max_breaks_per_hour=2)
    result = optimize_breaks([make_segment()], guardrails, revenue_weight=1.0)
    assert result.segments[0].num_breaks == 2
    assert result.is_compliant


def test_revenue_only_reaches_segment_max_when_guardrails_allow() -> None:
    result = optimize_breaks([make_segment(max_breaks=4)], GR, revenue_weight=1.0)
    assert result.segments[0].num_breaks == 4
    assert result.is_compliant


def test_retention_floor_caps_allocation() -> None:
    # impact -0.2: retention is 0.8 after one break, 0.6 after two. The floor is
    # 0.72, so the second break is refused even though revenue wants it.
    result = optimize_breaks(
        [make_segment(impact_coefficient=-0.2)], GR, revenue_weight=1.0,
    )
    assert result.segments[0].num_breaks == 1
    assert result.segments[0].retention >= GR.min_retention_floor
    assert result.is_compliant


def test_protected_programme_gets_tighter_ad_load() -> None:
    # 150s breaks: a News hour caps at 480s (3 breaks), Drama at 720s (4 breaks).
    news = optimize_breaks(
        [make_segment(program_type="News", break_length_seconds=150.0)],
        GR, revenue_weight=1.0,
    )
    drama = optimize_breaks(
        [make_segment(program_type="Drama", break_length_seconds=150.0)],
        GR, revenue_weight=1.0,
    )
    assert news.segments[0].num_breaks == 3
    assert drama.segments[0].num_breaks == 4
    assert news.is_compliant and drama.is_compliant


def test_short_segment_is_spacing_limited() -> None:
    # A ten-minute programme cannot hold two breaks 7 minutes apart.
    result = optimize_breaks(
        [make_segment(duration_seconds=600.0)], GR, revenue_weight=1.0,
    )
    assert result.segments[0].num_breaks == 1
    assert result.is_compliant


def test_weight_trades_revenue_against_retention() -> None:
    segment = make_segment(impact_coefficient=-0.1)
    low = optimize_breaks([segment], GR, revenue_weight=0.2)
    high = optimize_breaks([segment], GR, revenue_weight=0.95)
    assert low.total_breaks < high.total_breaks
    assert high.total_revenue > low.total_revenue
    assert low.is_compliant and high.is_compliant


def test_decisions_show_diminishing_returns() -> None:
    # Each added break earns less than the previous one, because the audience
    # that stays is smaller.
    result = optimize_breaks(
        [make_segment(impact_coefficient=-0.05, max_breaks=3)], GR, revenue_weight=1.0,
    )
    revenues = [d.marginal_revenue for d in result.decisions]
    assert [d.break_index for d in result.decisions] == [1, 2, 3]
    assert revenues == sorted(revenues, reverse=True)
    assert revenues[0] > revenues[-1]


def test_placement_revenue_sums_to_total() -> None:
    result = optimize_breaks(
        [make_segment(impact_coefficient=-0.05, max_breaks=3)], GR, revenue_weight=1.0,
    )
    assert sum(p.revenue for p in result.placements) == pytest.approx(result.total_revenue)
    assert len(result.placements) == result.total_breaks
    assert len(result.decisions) == result.total_breaks


def test_result_is_deterministic_regardless_of_input_order() -> None:
    segments = [
        make_segment(segment_id="a", channel="קשת 12"),
        make_segment(segment_id="b", channel="רשת 13"),
        make_segment(segment_id="c", channel="כאן 11"),
    ]
    forward = optimize_breaks(segments, GR, revenue_weight=0.7)
    reversed_ = optimize_breaks(list(reversed(segments)), GR, revenue_weight=0.7)
    assert forward == reversed_


def test_separate_channels_are_allocated_independently() -> None:
    # A breaks-per-hour cap on one channel must not constrain another.
    guardrails = Guardrails(max_breaks_per_hour=2)
    segments = [
        make_segment(segment_id="a", channel="קשת 12"),
        make_segment(segment_id="b", channel="רשת 13"),
    ]
    result = optimize_breaks(segments, guardrails, revenue_weight=1.0)
    per_segment = {s.segment_id: s.num_breaks for s in result.segments}
    assert per_segment == {"a": 2, "b": 2}
    assert result.is_compliant


def test_unit_seconds_scales_revenue_per_second() -> None:
    # With unit_seconds=1 and cpp as a per-second price, one 120s break at TVR 10
    # earns cpp * tvr * duration: 60 * 10 * 120 = 72000.
    segment = make_segment(
        max_breaks=1, cpp=60.0, unit_seconds=1.0, baseline_tvr=10.0,
        break_length_seconds=120.0, impact_coefficient=0.0, premium=1.0,
    )
    result = optimize_breaks([segment], GR, revenue_weight=1.0)
    assert result.segments[0].num_breaks == 1
    assert result.total_revenue == pytest.approx(60.0 * 10.0 * 120.0)


def test_invalid_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        optimize_breaks([make_segment()], GR, revenue_weight=1.5)


def test_invalid_segment_is_rejected() -> None:
    with pytest.raises(ValueError):
        optimize_breaks([make_segment(duration_seconds=0.0)], GR)


def test_duplicate_segment_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        optimize_breaks([make_segment(segment_id="x"), make_segment(segment_id="x")], GR)


# ---------------------------------------------------------------------------
# risk_lambda: the optimizer decides against the uncertainty-adjusted cost.
# ---------------------------------------------------------------------------


def _breaks_at(risk_lambda: float, **seg_overrides) -> int:
    result = optimize_breaks(
        [make_segment(**seg_overrides)], GR, revenue_weight=0.5, risk_lambda=risk_lambda,
    )
    return result.total_breaks


def test_risk_lambda_zero_matches_point_decision() -> None:
    # With risk_lambda 0 the interval is ignored: the decision and the reported
    # cost are exactly the point coefficient, identical to a segment with no
    # interval at all.
    with_interval = optimize_breaks(
        [make_segment(impact_coefficient=-0.02, impact_ci_low=-0.30, impact_ci_high=-0.01)],
        GR, revenue_weight=0.5, risk_lambda=0.0,
    )
    point_only = optimize_breaks(
        [make_segment(impact_coefficient=-0.02)], GR, revenue_weight=0.5, risk_lambda=0.0,
    )
    assert with_interval.total_breaks == point_only.total_breaks
    plan = with_interval.segments[0]
    assert plan.retention_cost_point == pytest.approx(-0.02)
    assert plan.retention_cost_used == pytest.approx(-0.02)
    assert with_interval.risk_lambda == 0.0


def test_risk_lambda_makes_uncertain_break_more_expensive() -> None:
    # A nearly-free point cost (-0.005) but a wide, damaging interval down to
    # -0.30. Risk-neutral fills breaks; full risk aversion decides against the
    # worst plausible cost and places strictly fewer.
    seg = dict(impact_coefficient=-0.005, impact_ci_low=-0.30, impact_ci_high=-0.005)
    neutral = _breaks_at(0.0, **seg)
    partial = _breaks_at(0.5, **seg)
    averse = _breaks_at(1.0, **seg)
    assert neutral >= partial >= averse
    assert averse < neutral


def test_risk_lambda_surfaces_cost_provenance_on_plan() -> None:
    # The plan reports the point, the conservative value actually used, the
    # interval, the sample size and the confidence, so the dashboard can show how
    # trustworthy the cost behind the break count was.
    result = optimize_breaks(
        [make_segment(
            impact_coefficient=-0.03, impact_ci_low=-0.08, impact_ci_high=-0.01,
            impact_n=40, impact_confidence="medium",
        )],
        GR, revenue_weight=0.5, risk_lambda=1.0,
    )
    plan = result.segments[0]
    assert plan.retention_cost_point == pytest.approx(-0.03)
    # At full risk aversion the used cost is the worst bound of the interval.
    assert plan.retention_cost_used == pytest.approx(-0.08)
    assert plan.retention_cost_ci_low == pytest.approx(-0.08)
    assert plan.retention_cost_ci_high == pytest.approx(-0.01)
    assert plan.retention_cost_n == 40
    assert plan.retention_confidence == "medium"
    assert result.risk_lambda == 1.0


def test_risk_lambda_out_of_range_is_rejected() -> None:
    with pytest.raises(ValueError):
        optimize_breaks([make_segment()], GR, risk_lambda=1.5)
    with pytest.raises(ValueError):
        optimize_breaks([make_segment()], GR, risk_lambda=-0.1)
