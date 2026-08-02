"""P2: the comparison JS-2 is defined on, and the three defects it had.

The panel printed "Net after retention cost: Not exposed" and a delta of zero on
every operational figure, and it compared one representative broadcast day while
every other figure on the destination was the week. That is three failures at
once. The quantity the job story is defined on was withheld, the only lever
offered could not move the plan (measured on רשת 13 / 2024-11-11, revenue weight
60 and 85 return the identical plan and only the blended score differs), and the
window was not the one the story names.

All three are closed here. The money and week helpers are exercised on recorded
payloads and synthetic day records so they run without the engine, and the shape
of the route is checked against the real app so an old caller's request keeps
working.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from kairos_api import scenario_compare_api as compare_api
from kairos_api import scenario_compare_api_week as week_api
from kairos_api.scenario_compare_api_money import (
    _identical_note,
    _priced,
    _resolve_levers,
    _scenario_summary,
    compare_body,
)
from kairos_api.scenario_compare_levers import (
    LEVER_FIELDS,
    ScenarioCompareRequest,
    ScenarioLevers,
)
from kairos_api.server import app


SHARED = {
    "retention_floor": 0.72,
    "max_breaks_per_hour": 4,
    "risk_lambda": 0.0,
    "objective_mode": "blend",
}

# A recorded run_scenario payload, trimmed to the fields the summary reads. The
# figures are the ones the live route returned for רשת 13 / 2024-11-11.
PAYLOAD = {
    "controls": {"revenue_weight": 60},
    "channel": "רשת 13",
    "day": "2024-11-11",
    "segments": [{"segment_id": "s1", "num_breaks": 2, "revenue": 1414695.2}],
    "summary": {
        "projected_revenue": 1414695.2,
        "average_retention": 95.0,
        "total_breaks": 80,
        "total_ad_seconds": 9600,
        "objective": 0.5404,
        "compliant": True,
    },
}


def test_a_request_that_names_only_the_two_weights_still_resolves_every_lever():
    """The fallback chain is what keeps the shape an old caller sends working."""
    levers = _resolve_levers(None, 60, SHARED)
    assert levers == {**SHARED, "revenue_weight": 60}


def test_a_per_leg_value_wins_over_the_shared_one():
    leg = ScenarioLevers(retention_floor=0.9, objective_mode="revenue_net")
    levers = _resolve_levers(leg, 60, SHARED)
    assert levers["retention_floor"] == 0.9
    assert levers["objective_mode"] == "revenue_net"
    # Everything it did not name still comes from the shared level.
    assert levers["max_breaks_per_hour"] == SHARED["max_breaks_per_hour"]
    assert levers["risk_lambda"] == SHARED["risk_lambda"]


def test_every_lever_the_runner_accepts_is_offered_on_a_leg():
    assert set(ScenarioLevers.model_fields) == set(LEVER_FIELDS)


def test_the_net_is_gross_minus_the_retention_cost_and_never_the_objective(monkeypatch):
    monkeypatch.setattr(
        "kairos_api.plan_read_frontier.scenario_plan_money",
        lambda payload, segments, risk: {
            "available": True, "gross": 1414695.2, "retention_cost": 141224.8,
            "net": 1273470.4, "breaks": 80,
        },
    )
    summary = _priced(_scenario_summary(PAYLOAD, {**SHARED, "revenue_weight": 60}), PAYLOAD, ["segment"], 0.0)
    assert summary["money_available"] is True
    assert round(summary["gross"] - summary["retention_cost"], 2) == summary["revenue_net"]
    # The convex-blend score keeps its own name and is not the net.
    assert summary["objective"] == 0.5404
    assert summary["objective"] != summary["revenue_net"]


def test_a_refusal_from_the_pricer_is_carried_verbatim_and_never_becomes_a_zero(monkeypatch):
    monkeypatch.setattr(
        "kairos_api.plan_read_frontier.scenario_plan_money",
        lambda payload, segments, risk: {"available": False, "reason": "segments no longer join"},
    )
    summary = _priced(_scenario_summary(PAYLOAD, SHARED), PAYLOAD, ["segment"], 0.0)
    assert summary["money_available"] is False
    assert summary["money_reason"] == "segments no longer join"
    assert "revenue_net" not in summary
    assert "gross" not in summary


def test_no_segments_means_no_money_and_a_stated_reason():
    summary = _priced(_scenario_summary(PAYLOAD, SHARED), PAYLOAD, [], 0.0)
    assert summary["money_available"] is False
    assert "retention cost cannot be priced" in summary["money_reason"]


def test_two_legs_that_produced_the_same_plan_say_so_and_name_the_lever():
    a = _scenario_summary(PAYLOAD, {**SHARED, "revenue_weight": 60})
    b = _scenario_summary(PAYLOAD, {**SHARED, "revenue_weight": 85})
    note = _identical_note(a, b)
    assert note["identical"] is True
    assert note["levers_that_differ"] == ["revenue_weight"]
    assert "retention_floor" in note["levers_that_match"]


def test_two_legs_that_produced_different_plans_carry_no_sameness_note():
    a = _scenario_summary(PAYLOAD, SHARED)
    moved = {**PAYLOAD, "summary": {**PAYLOAD["summary"], "projected_revenue": 1071648.55}}
    b = _scenario_summary(moved, {**SHARED, "retention_floor": 0.9})
    assert _identical_note(a, b) is None


def test_the_engine_unavailable_answer_stays_honest(monkeypatch):
    class _Server:
        _ENGINE_AVAILABLE = False

    monkeypatch.setattr(compare_api, "_server", lambda: _Server())
    body = compare_api._build_scenario_compare(
        compare_api.ScenarioCompareRequest(weight_a=60, weight_b=85)
    )
    assert body == {"available": False, "reason": "Optimization engine unavailable."}


def test_the_route_still_accepts_the_request_shape_it_always_accepted():
    """Bar 3: the two-weight body is the one the shipped client sends."""
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    body = schema["paths"]["/api/scenario-compare"]["post"]["requestBody"]
    ref = body["content"]["application/json"]["schema"]["$ref"].rsplit("/", 1)[-1]
    model = schema["components"]["schemas"][ref]
    assert set(model["required"]) == {"weight_a", "weight_b"}
    assert {"a", "b"} <= set(model["properties"])


# The week ---------------------------------------------------------------------
#
# JS-2's comparison is of next week. Running it on one representative broadcast
# day put two different quantities under one label on one destination: the goal
# strip's week beside a single day's money. These pin the window.

def _day(date, net_a, net_b, tvr=100.0, retention=95.0, objective=0.5, breaks=80):
    """One day of one leg, in the shape the runner hands the accumulator."""
    return {
        "date": date, "available": True, "reason": None,
        "projected_revenue": net_a, "average_retention": retention,
        "total_breaks": breaks, "total_ad_seconds": breaks * 120,
        "objective": objective, "compliant": True, "money_available": True,
        "money_reason": None, "gross": net_a, "retention_cost": net_a - net_b,
        "revenue_net": net_b, "total_tvr": tvr, "segments": 80,
    }


def test_the_comparison_asks_for_the_week_unless_a_caller_says_otherwise():
    assert ScenarioCompareRequest(weight_a=60, weight_b=85).scope == "week"
    assert ScenarioCompareRequest(weight_a=60, weight_b=85, scope="day").scope == "day"
    with pytest.raises(ValueError):
        ScenarioCompareRequest(weight_a=60, weight_b=85, scope="month")


def test_the_week_is_the_sum_of_its_days_and_the_retention_is_weighted():
    days = [
        _day("2024-11-01", 1000.0, 900.0, tvr=100.0, retention=94.0, objective=0.50),
        _day("2024-11-02", 2000.0, 1800.0, tvr=300.0, retention=96.0, objective=0.60),
    ]
    total = week_api._leg_total(days, {"revenue_weight": 60}, "רשת 13")
    assert total["projected_revenue"] == 3000.0
    assert total["gross"] == 3000.0
    assert total["revenue_net"] == 2700.0
    assert total["retention_cost"] == 300.0
    assert total["total_breaks"] == 160
    # 94 at weight 100 and 96 at weight 300 is 95.5, not the unweighted 95.0.
    assert total["average_retention"] == 95.5
    # The blended score is normalised inside a day, so a week reports the mean
    # of its days and says so rather than summing scores that cannot be summed.
    assert total["objective"] == 0.55
    assert total["objective_basis"] == "mean_of_days"
    assert total["days"] == 2
    assert total["channel"] == "רשת 13"
    assert total["day"] is None


def test_one_day_that_cannot_be_priced_refuses_the_whole_week_and_names_it():
    """A total that silently drops a day is worse than no total."""
    broken = {**_day("2024-11-02", 0.0, 0.0), "available": False, "money_available": False}
    broken["reason"] = "the programme source carries no segments for 2024-11-02"
    total = week_api._leg_total([_day("2024-11-01", 1000.0, 900.0), broken], {"revenue_weight": 60}, "רשת 13")
    assert total["money_available"] is False
    assert "2024-11-02" in total["money_reason"]
    assert "gross" not in total and "revenue_net" not in total


def test_the_week_payload_discloses_its_window_its_runs_and_its_clock(monkeypatch):
    window = {
        "available": True, "channel": "רשת 13",
        "dates": ["2024-11-01", "2024-11-02"], "date_from": "2024-11-01",
        "date_to": "2024-11-02", "n_dates": 2, "basis": "plan_first_week",
    }
    calls = []

    def fake_leg(channel, day, levers):
        calls.append((channel, day, levers["revenue_weight"]))
        net = 900.0 if levers["revenue_weight"] == 60 else 950.0
        return {**_day(day, 1000.0, net), "date": day}, True

    monkeypatch.setattr(week_api, "day_leg", fake_leg)
    body = compare_body(
        week_api.run_week(window, {"revenue_weight": 60}, {"revenue_weight": 85}),
        {"retention_floor": 0.72, "max_breaks_per_hour": 4, "risk_lambda": 0.0},
    )
    scope = body["scope"]
    assert scope["mode"] == "week"
    assert scope["dates"] == ["2024-11-01", "2024-11-02"]
    assert scope["n_dates"] == 2 and scope["days_priced"] == 2
    assert scope["basis"] == "plan_first_week"
    # Four runs for two days and two legs, and the payload says how many were
    # really computed rather than letting a cached answer read as fresh work.
    assert scope["runs"] == {"total": 4, "computed": 4, "reused": 0}
    assert isinstance(scope["elapsed_ms"], int)
    assert len(calls) == 4
    # Every day is on the table, with the difference the planner is choosing on.
    assert [row["date"] for row in body["by_day"]] == ["2024-11-01", "2024-11-02"]
    assert body["by_day"][0]["delta_revenue_net"] == 50.0
    assert body["delta"]["revenue_net"] == 100.0
    assert "every broadcast day in the plan's own week" in body["money_basis"]


def test_a_single_day_run_says_so_and_never_borrows_the_week_s_words():
    day_scope = {
        "a": {"money_available": True, "gross": 1.0, "retention_cost": 0.0, "revenue_net": 1.0,
              "levers": {"revenue_weight": 60}},
        "b": {"money_available": True, "gross": 2.0, "retention_cost": 0.0, "revenue_net": 2.0,
              "levers": {"revenue_weight": 85}},
        "by_day": None,
        "scope": {"mode": "day", "day": "2024-11-11", "n_dates": 1, "day_reason": "no week"},
    }
    body = compare_body(day_scope, {})
    assert body["scope"]["mode"] == "day"
    assert "one representative broadcast day, not the week" in body["money_basis"]
    assert body["by_day"] is None


def test_the_stream_emits_the_window_then_every_day_then_one_final_body(monkeypatch):
    window = {
        "available": True, "channel": "רשת 13",
        "dates": ["2024-11-01", "2024-11-02"], "date_from": "2024-11-01",
        "date_to": "2024-11-02", "n_dates": 2, "basis": "plan_first_week",
    }
    monkeypatch.setattr(
        compare_api, "prepare_week",
        lambda request: {
            "available": True, "window": window,
            "levers_a": {"revenue_weight": 60}, "levers_b": {"revenue_weight": 85},
            "guardrails": {"retention_floor": 0.72},
        },
    )
    monkeypatch.setattr(
        week_api, "day_leg",
        lambda channel, day, levers: ({**_day(day, 1000.0, 900.0), "date": day}, True),
    )
    client = TestClient(app)
    with client.stream("POST", "/api/scenario-compare/stream", json={"weight_a": 60, "weight_b": 85}) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        frames = []
        for block in "".join(response.iter_text()).split("\n\n"):
            if not block.strip():
                continue
            event = block.split("\n")[0].split("event: ", 1)[1]
            data = json.loads(block.split("data: ", 1)[1])
            frames.append((event, data))
    assert [event for event, _ in frames] == ["window", "day", "day", "final"]
    assert frames[0][1]["runs_total"] == 4
    assert [data["index"] for event, data in frames if event == "day"] == [1, 2]
    # The terminal frame is the whole comparison, not a summary of it, so a
    # client that only reads the last frame has exactly what the plain route
    # would have returned.
    final = frames[-1][1]
    assert final["available"] is True
    assert final["scope"]["mode"] == "week"
    assert len(final["by_day"]) == 2
    assert final["a"]["revenue_net"] == 1800.0


def test_a_reused_day_and_a_computed_day_are_counted_apart(monkeypatch):
    """The panel prints how many of its runs really ran, so the count has to be
    exact rather than a cache statistic another request can move."""
    week_api._day_leg_cached.cache_clear()
    ran = []

    def fake_run(**kwargs):
        ran.append(kwargs["day"])
        return {"summary": {"projected_revenue": 1.0, "average_retention": 90.0,
                            "total_breaks": 1, "total_ad_seconds": 120,
                            "objective": 0.5, "compliant": True},
                "segments": [{"segment_id": "s1", "num_breaks": 1, "revenue": 1.0}],
                "controls": {"revenue_weight": 60}, "channel": "רשת 13", "day": kwargs["day"]}

    monkeypatch.setattr("kairos.service.run_scenario", fake_run)
    monkeypatch.setattr(week_api, "_plan_segment_index", lambda pairs, settings: {})
    levers = {"revenue_weight": 60, "retention_floor": 0.72, "max_breaks_per_hour": 4,
              "risk_lambda": 0.0, "objective_mode": "blend"}
    _, first = week_api.day_leg("רשת 13", "2024-11-01", levers)
    _, second = week_api.day_leg("רשת 13", "2024-11-01", levers)
    assert first is True, "the first ask for a day computes it"
    assert second is False, "the second ask for the same day and levers reuses it"
    assert ran == ["2024-11-01"], "the optimizer ran exactly once"
    week_api._day_leg_cached.cache_clear()


def test_the_comparison_and_the_goal_strip_read_the_same_week():
    """One destination, one week. The measured failure this closes is one label,
    הכנסה צפויה, carrying the goal strip's week and the comparison's single day
    on the same screen. Both windows now come from the saved plan's own week, so
    they cannot drift; this reads them from the two modules and compares them."""
    from kairos_api.core import _load_break_schedule, _load_settings, _summarize_schedule
    from kairos_api.week_api_progress import build_progress

    settings = _load_settings()
    week = _summarize_schedule(_load_break_schedule()).get("week")
    if not isinstance(week, dict) or not week.get("date_from"):
        pytest.skip("no saved plan on this tree, so there is no week to compare")
    window = week_api.plan_week_window(settings)
    if not window["available"]:
        pytest.skip(window["reason"])
    goal = build_progress()["window"]
    assert window["date_from"] == goal["date_from"]
    assert window["date_to"] == goal["date_to"]
    assert window["n_dates"] == goal["n_dates"]
    assert window["basis"] == goal["basis"]
    # Every date the comparison runs is a date the operator's own plan carries.
    assert window["dates"][0] == window["date_from"]
    assert window["dates"][-1] == window["date_to"]
    assert window["channel"] == settings.operator_channel


def test_the_panel_says_which_window_it_is_showing_and_names_the_missing_one():
    """The day-scope sentence the baseline recorded, restored as a control: the
    panel must print it, gate it on the run really being one day, and offer the
    path to the weekly comparison rather than leaving a dead end."""
    from pathlib import Path

    panel = (Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src" / "plan" / "week"
             / "ComparePanel.jsx").read_text(encoding="utf-8")
    assert "One representative broadcast day, not the week." in panel
    assert "יום שידור מייצג אחד, ולא השבוע." in panel
    assert "{!week && (" in panel, "the sentence is gated on the run being a single day"
    assert "scope.day_reason" in panel, "the reason the week was unavailable travels to the reader"
    assert "Run the plan on step 2" in panel, "the path forward is named"
    # And the window travels with the money rather than sitting in a tooltip.
    assert "windowText" in panel and "plan-money-window" in panel
    assert "runCostLine" in panel, "the run's real cost is printed, not hidden"


def test_a_week_that_cannot_be_resolved_streams_the_reason_and_no_figure(monkeypatch):
    monkeypatch.setattr(
        compare_api, "prepare_week",
        lambda request: {"available": False, "reason": "the saved plan carries no week for your channel"},
    )
    client = TestClient(app)
    with client.stream("POST", "/api/scenario-compare/stream", json={"weight_a": 60, "weight_b": 85}) as response:
        text = "".join(response.iter_text())
    assert "event: error" in text
    assert "no week for your channel" in text
    assert "revenue_net" not in text
