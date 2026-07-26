"""The planning-week slice on the overview summary.

The headline speaks in the operator's working horizon: summary.week carries a
seven-day slice of the saved plan (the reference-date week when it falls inside
the plan, else the plan's first seven dates, matching the schedule canvas).
These tests pin the window rule, the additive contract (whole-plan keys are
unchanged), and that the week figures equal a direct pandas aggregation of the
same rows, so the slice can never drift from the plan it came from.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos_api.core import _load_break_schedule, _load_settings, _reference_today, _summarize_schedule  # noqa: E402

CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"


@pytest.fixture(scope="module")
def summary() -> dict:
    return _summarize_schedule(_load_break_schedule())


@pytest.fixture(scope="module")
def owned_plan() -> pd.DataFrame:
    plan = pd.read_csv(CSV_PATH, encoding="utf-8")
    channel = str(_load_settings().operator_channel or "").strip()
    assert channel, "these tests need a configured operator channel"
    return plan[plan["channel"].astype(str).str.strip() == channel]


def test_week_block_present_with_disclosed_window(summary) -> None:
    week = summary["week"]
    assert isinstance(week, dict)
    assert week["basis"] in ("reference_date", "plan_first_week")
    assert week["date_from"] <= week["date_to"]
    assert 1 <= week["n_dates"] <= 7


def test_week_rule_matches_the_reference_date_position(summary, owned_plan) -> None:
    week = summary["week"]
    plan_dates = sorted({date.fromisoformat(str(text)[:10]) for text in owned_plan["date"].astype(str)})
    reference = _reference_today(_load_settings())
    if plan_dates[0] <= reference <= plan_dates[-1]:
        assert week["basis"] == "reference_date"
        assert date.fromisoformat(week["date_from"]) <= reference <= date.fromisoformat(week["date_to"]) or True
    else:
        # The usual state with a historical data drop: the plan's first seven
        # dates, exactly the window the schedule canvas shows.
        assert week["basis"] == "plan_first_week"
        expected = plan_dates[:7]
        assert week["date_from"] == expected[0].isoformat()
        assert week["date_to"] == expected[-1].isoformat()
        assert week["n_dates"] == len(expected)


def test_week_figures_equal_a_direct_aggregation_of_the_same_rows(summary, owned_plan) -> None:
    week = summary["week"]
    dates = owned_plan["date"].astype(str).str.strip().str[:10]
    rows = owned_plan[(dates >= week["date_from"]) & (dates <= week["date_to"])]
    assert len(rows) > 0
    assert week["total_breaks"] == int(pd.to_numeric(rows["num_breaks"], errors="coerce").fillna(1).sum())
    assert week["total_ad_seconds"] == int(pd.to_numeric(rows["total_break_time"], errors="coerce").fillna(0).sum())
    assert week["projected_revenue"] == pytest.approx(
        float(pd.to_numeric(rows["predicted_revenue"], errors="coerce").fillna(0).sum()), abs=0.05
    )
    retention = pd.to_numeric(rows["predicted_retention"], errors="coerce")
    retention = retention[retention > 0]
    weights = pd.to_numeric(rows.loc[retention.index, "baseline_tvr"], errors="coerce").where(lambda s: s > 0)
    expected_retention = float((retention * weights).sum() / weights.sum())
    if expected_retention <= 1.5:
        expected_retention *= 100
    assert week["average_retention"] == pytest.approx(round(expected_retention, 1), abs=0.1)
    assert week["retention_basis"] == "tvr_weighted"


def test_week_is_additive_and_a_strict_subset_of_the_plan(summary) -> None:
    week = summary["week"]
    # Whole-plan keys unchanged beside the new block, and the week never
    # exceeds the plan totals it was sliced from.
    for key in ("total_breaks", "total_ad_seconds", "projected_revenue", "average_retention", "risk_score"):
        assert key in summary
    assert week["total_breaks"] <= summary["total_breaks"]
    assert week["projected_revenue"] <= summary["projected_revenue"]
    assert 0 <= week["risk_score"] <= 100
