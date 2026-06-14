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


def test_invalid_weight_is_rejected() -> None:
    with pytest.raises(ValueError):
        optimize_breaks([make_segment()], GR, revenue_weight=1.5)


def test_invalid_segment_is_rejected() -> None:
    with pytest.raises(ValueError):
        optimize_breaks([make_segment(duration_seconds=0.0)], GR)


def test_duplicate_segment_ids_are_rejected() -> None:
    with pytest.raises(ValueError):
        optimize_breaks([make_segment(segment_id="x"), make_segment(segment_id="x")], GR)
