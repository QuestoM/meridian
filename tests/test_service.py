"""Tests for the service layer that the API calls.

Synthetic frames keep these fast while still exercising the real classifier,
pricing, transform and optimizer end to end.
"""

from __future__ import annotations

import pandas as pd

from kairos.optimize.guardrails import Guardrails
from kairos.service import guardrails_from_settings, optimize_day_plan, run_scenario


def make_frame() -> pd.DataFrame:
    rows = [
        ("חדשות הערב", "קשת 12", "20:00:00", 3600, 6.0),
        ("התוכנית הראשונה", "קשת 12", "21:00:00", 3600, 6.0),
        ("התוכנית השנייה", "קשת 12", "22:00:00", 3600, 6.0),
        ("תוכנית ערוץ אחר", "רשת 13", "21:00:00", 3600, 4.0),
    ]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "time", "Duration", "TVR"])
    frame["start_dt"] = pd.to_datetime("2024-11-04 " + frame["time"])
    return frame


def test_settings_minutes_map_to_guardrail_seconds() -> None:
    settings = {
        "max_ad_minutes_per_hour": 12,
        "max_breaks_per_hour": 4,
        "min_break_spacing_minutes": 7,
        "min_retention_floor": 0.72,
        "max_daily_ad_minutes": 160,
        "protected_program_types": ["News", "Kids", "Children"],
        "protected_program_max_ad_minutes_per_hour": 8,
        "gold_breaks_max_per_day": 3,
    }
    guardrails = guardrails_from_settings(settings)
    assert guardrails.max_ad_seconds_per_hour == 720
    assert guardrails.min_break_spacing_seconds == 420
    assert guardrails.max_daily_ad_seconds == 9600
    assert guardrails.protected_max_ad_seconds_per_hour == 480
    assert guardrails.protected_program_types == ("News", "Kids", "Children")
    assert guardrails.gold_breaks_max_per_day == 3


def test_partial_settings_fall_back_to_defaults() -> None:
    base = Guardrails()
    guardrails = guardrails_from_settings({"max_breaks_per_hour": 2})
    assert guardrails.max_breaks_per_hour == 2
    assert guardrails.max_ad_seconds_per_hour == base.max_ad_seconds_per_hour
    assert guardrails_from_settings({}) == base


def test_optimize_day_plan_returns_real_serialisable_plan() -> None:
    plan = optimize_day_plan(programmes=make_frame(), channel="קשת 12", revenue_weight=1.0)
    assert plan["summary"]["compliant"] is True
    assert plan["summary"]["total_breaks"] > 0
    assert plan["summary"]["projected_revenue"] > 0
    assert plan["weights"]["revenue_weight"] == 1.0
    assert plan["segment_count"] == 3
    # Every placement is JSON-friendly with a clock time and a percent retention.
    first = plan["placements"][0]
    assert len(first["start_time"]) == 5 and first["start_time"][2] == ":"
    assert 0 <= first["retention_percent"] <= 100
    # The guardrails and assumptions used are echoed for the dashboard.
    assert "guardrails" in plan and "assumptions" in plan


def test_optimize_day_plan_honours_settings_guardrails() -> None:
    settings = {"max_breaks_per_hour": 1}
    plan = optimize_day_plan(
        programmes=make_frame(), channel="קשת 12", revenue_weight=1.0, settings=settings,
    )
    assert plan["guardrails"]["max_breaks_per_hour"] == 1
    assert plan["summary"]["compliant"] is True


def test_run_scenario_trades_off_with_the_weight() -> None:
    frame = make_frame()
    low = run_scenario(revenue_weight=0, retention_floor=0.72, max_breaks_per_hour=4, programmes=frame)
    high = run_scenario(revenue_weight=100, retention_floor=0.72, max_breaks_per_hour=4, programmes=frame)
    assert low["summary"]["total_breaks"] == 0
    assert high["summary"]["total_breaks"] > 0
    assert high["controls"] == {
        "revenue_weight": 100, "retention_floor": 0.72, "max_breaks_per_hour": 4, "risk_lambda": 0.0,
    }
    assert high["guardrails"]["max_breaks_per_hour"] == 4
    assert high["summary"]["compliant"] is True


def test_optimize_day_plan_surfaces_retention_cost_provenance() -> None:
    # Every serialised segment carries the retention-cost block the dashboard
    # needs to show how trustworthy the cost behind its break count is: the point
    # estimate, the value actually used in the decision, the interval, the sample
    # size and the confidence label. The effective risk preference is echoed too.
    plan = optimize_day_plan(programmes=make_frame(), channel="קשת 12", revenue_weight=1.0)
    assert plan["weights"]["risk_lambda"] == 0.0
    assert plan["segments"]
    for segment in plan["segments"]:
        cost = segment["retention_cost"]
        assert set(cost) == {"point", "used", "ci_low", "ci_high", "n", "confidence"}
        # Risk-neutral by default, so the decision uses the point estimate exactly.
        assert cost["used"] == cost["point"]
        # A retention cost never raises retention, so it is non-positive.
        assert cost["point"] <= 0
        assert isinstance(cost["n"], int) and cost["n"] >= 0
        assert cost["confidence"] in {"low", "medium", "high"}


def test_run_scenario_threads_risk_lambda() -> None:
    # The scenario slider's risk preference is echoed in both the controls and the
    # weights, proving it reaches optimize_breaks rather than being silently dropped.
    frame = make_frame()
    plan = run_scenario(
        revenue_weight=60, retention_floor=0.72, max_breaks_per_hour=4,
        risk_lambda=1.0, programmes=frame,
    )
    assert plan["controls"]["risk_lambda"] == 1.0
    assert plan["weights"]["risk_lambda"] == 1.0
