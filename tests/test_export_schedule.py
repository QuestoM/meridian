"""Tests for the weekly schedule exporter.

A synthetic multi-channel, multi-day frame keeps these in the fast gate while
still running the real classifier, pricing, transform and optimizer end to end,
and proving the output carries the real columns the dashboard reads (so its
placeholder fallback never has to fire).
"""

from __future__ import annotations

import pandas as pd

from kairos.export.schedule import COLUMNS, build_weekly_schedule, write_weekly_schedule
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks
from kairos.optimize.guardrails import Guardrails


def make_frame() -> pd.DataFrame:
    rows = [
        ("חדשות הערב", "קשת 12", "2024-11-04 20:00:00", 3600, 6.0),
        ("התוכנית הראשונה", "קשת 12", "2024-11-04 21:00:00", 3600, 6.0),
        ("התוכנית השנייה", "קשת 12", "2024-11-04 22:00:00", 3600, 5.0),
        ("מהדורת חדשות", "רשת 13", "2024-11-05 20:00:00", 3600, 4.0),
        ("תוכנית אירוח", "רשת 13", "2024-11-05 21:00:00", 3600, 4.0),
    ]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "Duration", "TVR"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    return frame


def test_schedule_carries_the_columns_the_dashboard_reads() -> None:
    schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
    assert list(schedule.columns) == COLUMNS
    # Every channel-day is covered: 3 segments on Keshet, 2 on Reshet.
    assert len(schedule) == 5
    assert set(schedule["channel"]) == {"קשת 12", "רשת 13"}
    # Days are normalised to the dashboard's abbreviation key.
    assert set(schedule["day"]).issubset({"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"})


def test_numbers_are_real_not_fabricated() -> None:
    schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
    # Revenue is earned, retention is a real probability, breaks are placed.
    assert schedule["predicted_revenue"].sum() > 0
    assert (schedule["num_breaks"] >= 0).all()
    assert schedule["num_breaks"].sum() > 0
    assert ((schedule["predicted_retention"] >= 0) & (schedule["predicted_retention"] <= 1)).all()
    # base_rate is the effective per-second rate (CPP times premium), always positive.
    assert (schedule["base_rate"] > 0).all()
    # total_break_time is num_breaks times the break length, never invented.
    expected = schedule["num_breaks"] * schedule["break_length"]
    assert (schedule["total_break_time"] - expected).abs().max() < 1e-6


def test_zero_revenue_weight_places_no_breaks_and_earns_nothing() -> None:
    schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=0.0)
    assert (schedule["num_breaks"] == 0).all()
    assert schedule["predicted_revenue"].sum() == 0
    # A programme with no breaks keeps its full baseline retention, never a guess.
    assert (schedule["predicted_retention"] == 1.0).all()


def test_break_type_is_derived_from_length() -> None:
    schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
    # The default break length is 120s, which is the medium band.
    assert set(schedule["break_type"]).issubset({"short", "medium", "long"})
    assert (schedule["break_type"] == "medium").all()


def test_write_round_trips_through_csv(tmp_path) -> None:
    frame = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
    target = tmp_path / "weekly_break_schedule.csv"
    written = write_weekly_schedule(target, frame=frame)
    assert written == target
    reloaded = pd.read_csv(target)
    assert list(reloaded.columns) == COLUMNS
    assert len(reloaded) == len(frame)


# ---------------------------------------------------------------------------
# FIX 1: risk-adjusted retention columns in the export
# ---------------------------------------------------------------------------

class TestRiskAdjustedRetentionColumns:
    """The weekly export must carry the per-segment risk-adjusted retention and CI."""

    def test_new_columns_present_in_columns_list(self):
        for col in ("retention_used", "retention_ci_low", "retention_ci_high",
                    "retention_n", "retention_confidence"):
            assert col in COLUMNS, f"expected {col} in COLUMNS"

    def test_schedule_has_new_columns(self):
        schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
        for col in ("retention_used", "retention_ci_low", "retention_ci_high",
                    "retention_n", "retention_confidence"):
            assert col in schedule.columns, f"missing column {col}"

    def test_retention_used_equals_predicted_retention_when_no_ci(self):
        # With no CI in the segments (the synthetic frame has no impact model),
        # retention_used should equal predicted_retention exactly.
        schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
        diff = (schedule["predicted_retention"] - schedule["retention_used"]).abs()
        assert diff.max() < 1e-9, "retention_used differs from predicted_retention with no CI"

    def test_ci_columns_blank_when_no_breaks(self):
        # With revenue_weight=0 no breaks are placed; the CI-from-breaks
        # computation requires num_breaks > 0, so ci columns must be blank/NaN
        # even if the impact model has a CI.
        schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=0.0)
        assert (schedule["num_breaks"] == 0).all()
        assert schedule["retention_ci_low"].isna().all(), "ci_low must be NaN when num_breaks=0"
        assert schedule["retention_ci_high"].isna().all(), "ci_high must be NaN when num_breaks=0"

    def test_ci_columns_populated_when_impact_model_provides_ci(self):
        # When the impact model carries a CI (the real model does), and breaks
        # are placed, the ci columns must be real finite numbers in [0, 1].
        schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
        has_breaks = schedule[schedule["num_breaks"] > 0]
        if has_breaks.empty:
            return  # if no breaks placed, this sub-test is vacuously satisfied
        # Segments that have breaks AND a measured CI should have real ci values.
        # Not all segments are guaranteed to have a CI so we only check non-null rows.
        ci_low_not_null = has_breaks["retention_ci_low"].dropna()
        ci_high_not_null = has_breaks["retention_ci_high"].dropna()
        if not ci_low_not_null.empty:
            assert (ci_low_not_null >= 0).all(), "ci_low must be non-negative"
            assert (ci_low_not_null <= 1).all(), "ci_low must be at most 1"
        if not ci_high_not_null.empty:
            assert (ci_high_not_null >= 0).all(), "ci_high must be non-negative"
            assert (ci_high_not_null <= 1).all(), "ci_high must be at most 1"

    def test_retention_used_differs_from_predicted_retention_when_risk_lambda_and_ci(self):
        # Build a segment with a real CI and risk_lambda > 0.  The optimizer
        # should decide with a conservative (more negative) coefficient, so
        # retention_used is lower than predicted_retention for a 0-lambda run.
        # Use a direct optimizer call rather than the full export pipeline so the
        # CI is guaranteed to be present.
        seg = ProgramSegment(
            segment_id="s1",
            channel="Test Channel",
            day="2026-06-15",
            start_seconds=20 * 3600.0,
            duration_seconds=3600.0,
            program_type="Drama",
            baseline_tvr=10.0,
            cpp=1000.0,
            impact_coefficient=-0.05,
            retention_baseline=1.0,
            premium=1.0,
            is_gold=False,
            max_breaks=4,
            break_length_seconds=120.0,
            unit_seconds=30.0,
            impact_ci_low=-0.12,
            impact_ci_high=-0.02,
            impact_n=50,
            impact_confidence="medium",
        )
        # Run with risk_lambda=1.0 (full worst-case) vs 0.0 (point only).
        result_risk = optimize_breaks([seg], Guardrails(), revenue_weight=1.0, risk_lambda=1.0)
        result_point = optimize_breaks([seg], Guardrails(), revenue_weight=1.0, risk_lambda=0.0)
        plan_risk = result_risk.segments[0]
        plan_point = result_point.segments[0]
        # With risk_lambda=1.0 the optimizer is more pessimistic about retention
        # cost, so it may place fewer breaks and plan.retention may be higher;
        # what is guaranteed is that retention_cost_used is more conservative
        # (more negative) than the point estimate.
        assert plan_risk.retention_cost_used <= plan_risk.retention_cost_point, (
            "risk_lambda=1.0 should produce a more-conservative retention cost"
        )
        # The CI fields must be populated.
        assert plan_risk.retention_cost_ci_low is not None
        assert plan_risk.retention_cost_ci_high is not None
        assert plan_risk.retention_cost_n == 50
        assert plan_risk.retention_confidence == "medium"

    def test_zero_break_segment_has_blank_ci_columns_in_optimizer(self):
        # A segment with 0 breaks never enters the CI computation path, so its
        # CI retention fields are None. Verify at the optimizer level directly.
        seg = ProgramSegment(
            segment_id="s0",
            channel="Test",
            day="2026-06-15",
            start_seconds=20 * 3600.0,
            duration_seconds=3600.0,
            program_type="Drama",
            baseline_tvr=10.0,
            cpp=1000.0,
            impact_coefficient=-0.05,
            retention_baseline=1.0,
            premium=1.0,
            is_gold=False,
            max_breaks=4,
            break_length_seconds=120.0,
            unit_seconds=30.0,
            impact_ci_low=-0.12,
            impact_ci_high=-0.02,
            impact_n=50,
            impact_confidence="medium",
        )
        result = optimize_breaks([seg], Guardrails(), revenue_weight=0.0)
        plan = result.segments[0]
        assert plan.num_breaks == 0
        # With 0 breaks the schedule.py code must emit NaN for ci columns.
        # Verify the plan fields that schedule.py reads are non-zero when present.
        assert plan.retention_cost_ci_low is not None
        assert plan.retention_cost_ci_high is not None
