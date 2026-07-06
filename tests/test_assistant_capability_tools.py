"""Contract tests for the four capability read tools.

Everything below the tool seam is real: get_recommendations runs the genuine
overview builder, get_plan_days prices the repository's committed weekly CSV,
get_audience_stability reads the real coefficients artifact, and get_frontier
exercises the real (non-blocking) frontier machinery. Each tool must return
owned-channel-scoped data stamped with a non-empty provenance source, never a
competitor figure, and honest unavailable states instead of fabrication.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

import kairos_api.assistant_context as assistant_context
import kairos_api.assistant_tools as tools

NEW_TOOLS = ("get_recommendations", "get_frontier", "get_audience_stability", "get_plan_days")


def _plan_facts() -> tuple[Any, str]:
    server = assistant_context._server()
    frame = pd.read_csv(server.OUTPUT_DIR / "weekly_break_schedule.csv")
    owned = str(server._load_settings().operator_channel or "").strip()
    assert not frame.empty and owned
    return frame, owned


# --- registry wiring --------------------------------------------------------------
def test_new_tools_are_registered_read_tools_with_sources() -> None:
    for name in NEW_TOOLS:
        assert name in tools.READ_TOOL_NAMES
        assert name in tools._READ_EXECUTORS
        assert tools.SOURCE_BY_TOOL[name]
    read_only = {schema["name"] for schema in tools.anthropic_tools(include_propose=False)}
    assert set(NEW_TOOLS) <= read_only
    assert not read_only & tools.PROPOSE_TOOL_NAMES
    # Prescriptive descriptions: each tells the model WHEN to call it.
    by_name = {schema["name"]: schema for schema in tools.READ_TOOL_SCHEMAS}
    for name in NEW_TOOLS:
        assert "Call this when" in by_name[name]["description"]


# --- get_plan_days: the committed plan, owned channel, real per-day money ---------
def test_get_plan_days_matches_the_committed_csv() -> None:
    frame, owned = _plan_facts()
    own = frame[frame["channel"].astype(str).str.strip() == owned]

    payload = tools.execute_read_tool("get_plan_days", {})
    assert "error" not in payload
    assert payload["source"] == "saved weekly plan, owned channel"
    assert payload["channel"] == owned
    assert payload["days_total"] == own["date"].astype(str).str.strip().nunique()
    assert len(payload["days"]) == min(payload["days_total"], 31)

    sample = payload["days"][0]
    assert {"date", "weekday", "breaks", "revenue_ils", "retention_cost_ils"} <= set(sample)
    day_rows = own[own["date"].astype(str).str.strip() == sample["date"]]
    assert sample["breaks"] == int(pd.to_numeric(day_rows["num_breaks"], errors="coerce").fillna(0).sum())
    expected_revenue = int(round(float(
        pd.to_numeric(day_rows["predicted_revenue"], errors="coerce").fillna(0).sum()
    )))
    assert sample["revenue_ils"] == expected_revenue
    # The repository CSV persists baseline_tvr, so the per-day retention cost is
    # derivable here: a real modeled figure, and net stays gross minus cost.
    assert sample["retention_cost_ils"] is not None
    assert sample["retention_cost_ils"] >= 0
    assert sample["revenue_net_ils"] <= sample["revenue_ils"]


# --- get_recommendations: the real overview builder, owned scope ------------------
def test_get_recommendations_returns_the_overview_builders_rows() -> None:
    _, owned = _plan_facts()
    payload = tools.execute_read_tool("get_recommendations", {})
    assert "error" not in payload
    assert payload["source"] == "overview recommendations, owned channel"
    rows = payload["recommendations"]
    assert payload["count"] == len(rows)
    assert 0 < len(rows) <= 5
    for row in rows:
        assert row["channel"] == owned
        assert row["risk"] in {"High", "Medium", "Low"}
        assert {"title", "program_type", "date", "impact_ils", "retention_pct",
                "proposed_kind", "actionable"} <= set(row)


# --- get_audience_stability: measured drift or honest unavailable ------------------
def test_get_audience_stability_is_measured_or_honestly_unavailable() -> None:
    payload = tools.execute_read_tool("get_audience_stability", {})
    assert "error" not in payload
    assert payload["source"] == "measured coefficients artifact, level-drift monitor"
    assert payload.get("status") in {"measured", "unavailable"}
    if payload["status"] == "measured":
        assert payload["n_weeks"] >= 1
        assert isinstance(payload.get("weekly_levels"), list)
        assert len(payload["weekly_levels"]) <= 12
    else:
        assert payload["reason"]


# --- get_frontier: honest states from the real non-blocking machinery --------------
def test_get_frontier_reports_an_honest_state_with_owned_scope() -> None:
    payload = tools.execute_read_tool("get_frontier", {})
    assert "error" not in payload
    assert payload["source"] == "owned-channel frontier sweep"
    # The machinery never blocks: a cold cache is an honest 'computing' with no
    # points, a warm one is 'ready' with the sweep. Both are legitimate here.
    assert payload["status"] in {"ready", "computing"}
    assert "current_plan_point" in payload and "net_focused_point" in payload
    if payload["status"] == "computing":
        assert payload["points"] == []
        assert "computing" in payload["reason"]
    else:
        assert payload["points"]
        for point in payload["points"]:
            assert {"retention", "revenue", "retention_floor", "num_breaks"} <= set(point)


def test_get_frontier_shapes_ready_points_current_and_net(monkeypatch: pytest.MonkeyPatch) -> None:
    from kairos_api import dashboard_api

    points = [
        {"retention": 80.0, "revenue": 100.0, "retention_floor": 0.72, "num_breaks": 9, "selected": False},
        {"retention": 90.0, "revenue": 90.0, "retention_floor": 0.85, "num_breaks": 7, "selected": True},
    ]
    bundle = {
        "net_point": {"retention": 88.0, "revenue": 95.0, "retention_floor": 0.85,
                      "num_breaks": 8, "selected": False, "id": "net_focused"},
        "comparison_available": True,
    }
    monkeypatch.setattr(
        dashboard_api, "_frontier_state", lambda settings, scope=None: (points, bundle, "ready")
    )
    payload = tools.execute_read_tool("get_frontier", {})
    assert payload["status"] == "ready"
    assert payload["current_plan_point"] == points[1]
    assert payload["net_focused_point"]["id"] == "net_focused"
    assert len(payload["points"]) == 2


# --- competitor boundary across all four tools -------------------------------------
def test_no_competitor_channel_name_leaks_from_the_new_tools() -> None:
    frame, owned = _plan_facts()
    channels = {text for text in frame["channel"].astype(str).str.strip().unique() if text}
    competitors = sorted(channels - {owned})
    assert competitors, "the saved plan must carry competitor channels for this test to bite"

    for name in NEW_TOOLS:
        serialized = json.dumps(tools.execute_read_tool(name, {}), ensure_ascii=False, default=str)
        for competitor in competitors:
            assert competitor not in serialized, f"{name} leaked {competitor}"
