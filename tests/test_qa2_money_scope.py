"""Headline money is scoped to the operator's channel (competitor boundary).

The saved weekly CSV carries every channel because the retention model needs
competitor rows, but _summarize_schedule summed all of them (about 5.5x the
owned plan on the reference data) and /api/forecasts grouped the whole month
times four channels into seven weekday rows. These tests pin the scoped
contract: owned-channel money, per-real-date forecast rows, TVR-weighted
retention, and explicit basis fields, with the whole-frame fallback reserved
for the not-yet-configured state and labeled as such.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"

pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists(), reason="no committed weekly plan on disk"
)


@pytest.fixture()
def plan_and_owned():
    from kairos_api.core import _load_break_schedule, _load_settings

    plan = _load_break_schedule()
    owned = str(_load_settings().operator_channel or "").strip()
    if plan.empty:
        pytest.skip("no plan rows")
    if not owned:
        pytest.skip("no operator channel configured")
    if "channel" not in plan.columns or not (plan["channel"] == owned).any():
        pytest.skip("owned channel absent from the plan")
    return plan, owned


def test_summary_money_is_the_owned_channel_plan(plan_and_owned):
    from kairos_api.core import _summarize_schedule

    plan, owned = plan_and_owned
    summary = _summarize_schedule(plan)
    scoped = plan[plan["channel"].astype(str).str.strip() == owned]

    assert summary["scope_channel"] == owned
    assert summary["n_channels_total"] == int(plan["channel"].nunique())
    assert summary["n_dates"] == int(scoped["date"].nunique())
    assert summary["projected_revenue"] == pytest.approx(
        scoped["predicted_revenue"].sum(), abs=0.05
    )
    assert summary["total_breaks"] == int(scoped["num_breaks"].sum())
    assert summary["total_ad_seconds"] == int(scoped["total_break_time"].sum())
    # The whole-frame figure must NOT leak into the headline when a channel is set.
    whole = float(plan["predicted_revenue"].sum())
    if abs(whole - float(scoped["predicted_revenue"].sum())) > 1.0:
        assert summary["projected_revenue"] != pytest.approx(whole, abs=1.0)


def test_summary_retention_is_tvr_weighted(plan_and_owned):
    from kairos_api.core import _summarize_schedule

    plan, owned = plan_and_owned
    if "baseline_tvr" not in plan.columns:
        pytest.skip("plan carries no baseline_tvr column")
    summary = _summarize_schedule(plan)
    scoped = plan[plan["channel"].astype(str).str.strip() == owned]
    valid = scoped[pd.to_numeric(scoped["predicted_retention"], errors="coerce") > 0]
    weights = pd.to_numeric(valid["baseline_tvr"], errors="coerce").where(lambda s: s > 0)
    expected = float(
        (pd.to_numeric(valid["predicted_retention"], errors="coerce") * weights).sum()
        / weights.sum()
    )
    assert summary["retention_basis"] == "tvr_weighted"
    assert summary["average_retention"] == pytest.approx(round(expected * 100, 1), abs=0.05)


def test_summary_falls_back_to_whole_frame_only_when_channel_unset(monkeypatch, plan_and_owned):
    import kairos_api.core as core

    plan, _owned = plan_and_owned
    monkeypatch.setattr(core, "_load_settings", lambda: core.KairosSettings(operator_channel=""))
    summary = core._summarize_schedule(plan)
    assert summary["scope_channel"] is None, "the fallback must be labeled, never silent"
    assert summary["projected_revenue"] == pytest.approx(
        plan["predicted_revenue"].sum(), abs=0.05
    )


def test_summary_empty_frame_keeps_honest_nulls_with_basis():
    from kairos_api.core import _summarize_schedule

    summary = _summarize_schedule(pd.DataFrame())
    assert summary["projected_revenue"] is None
    assert summary["average_retention"] is None
    assert summary["risk_score"] is None
    assert summary["total_breaks"] == 0
    for key in ("scope_channel", "n_dates", "n_channels_total", "retention_basis"):
        assert key in summary


def test_forecast_rows_are_per_real_date_on_the_owned_channel(plan_and_owned):
    from kairos_api.catalog_api import _build_forecasts
    from kairos_api.core import _load_settings

    plan, owned = plan_and_owned
    settings = _load_settings()
    payload = _build_forecasts(plan, settings)
    scoped = plan[plan["channel"].astype(str).str.strip() == owned]
    expected_dates = sorted(scoped["date"].astype(str).unique())

    basis = payload["by_day_basis"]
    assert basis["scope_channel"] == owned
    assert basis["grouped_by"] == "date"
    assert basis["n_dates"] == len(expected_dates)

    rows = payload["by_day"]
    assert [row["date"] for row in rows] == expected_dates
    per_date_revenue = scoped.groupby("date")["predicted_revenue"].sum()
    per_date_breaks = scoped.groupby("date")["num_breaks"].sum()
    weekday_map = scoped.groupby("date")["day"].first()
    for row in rows:
        assert row["revenue"] == pytest.approx(per_date_revenue[row["date"]], abs=0.05)
        assert row["breaks"] == pytest.approx(per_date_breaks[row["date"]])
        assert row["day"] == weekday_map[row["date"]], "weekday must ride beside the date"
        assert 0.0 <= float(row["retention"]) <= 1.0


def test_forecast_retention_is_weighted_not_a_raw_row_mean(plan_and_owned):
    from kairos_api.catalog_api import _build_forecasts
    from kairos_api.core import _load_settings

    plan, owned = plan_and_owned
    if "baseline_tvr" not in plan.columns:
        pytest.skip("plan carries no baseline_tvr column")
    payload = _build_forecasts(plan, _load_settings())
    assert payload["by_day_basis"]["retention_basis"] == "tvr_weighted"
    scoped = plan[plan["channel"].astype(str).str.strip() == owned]
    first = payload["by_day"][0]
    day_rows = scoped[scoped["date"].astype(str) == first["date"]]
    weights = pd.to_numeric(day_rows["baseline_tvr"], errors="coerce").where(lambda s: s > 0)
    expected = float(
        (pd.to_numeric(day_rows["predicted_retention"], errors="coerce") * weights).sum()
        / weights.sum()
    )
    assert first["retention"] == pytest.approx(expected, rel=1e-9)


def test_forecast_empty_and_unmatched_scopes_stay_honest(monkeypatch):
    import kairos_api.catalog_api as catalog
    from kairos_api.core import KairosSettings

    empty = catalog._build_forecasts(pd.DataFrame(), KairosSettings(operator_channel=""))
    assert empty["by_day"] == []
    assert empty["by_day_basis"]["n_dates"] == 0

    # An owned channel with no plan rows yields empty rows, never another
    # channel's money. Scenario re-runs are stubbed out (engine-heavy).
    monkeypatch.setattr(catalog, "_build_forecast_scenarios", lambda settings: [])
    frame = pd.DataFrame(
        [{"channel": "someone-else", "date": "2024-11-01", "day": "Fri",
          "predicted_revenue": 5.0, "predicted_retention": 0.9, "num_breaks": 1}]
    )
    unmatched = catalog._build_forecasts(frame, KairosSettings(operator_channel="עכשיו 14"))
    assert unmatched["by_day"] == []
    assert unmatched["by_day_basis"]["scope_channel"] == "עכשיו 14"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
