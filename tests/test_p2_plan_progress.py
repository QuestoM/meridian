"""P2: the goal and the progress against it.

The one question this destination could never answer is whether the week is on
plan, because no target exists in the data. These tests pin the two halves of
the honest answer: the projection is the plan's own operator-scoped week sum and
never something recomputed here, and the verdict is the target store's own rule
so there is exactly one threshold in the product.

Every write in this file goes to an isolated store through
``KAIROS_PLAN_TARGETS_PATH``. The operator's own ``data/plan_targets.csv`` is
never touched, and one test asserts that.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kairos_api import target_store
from kairos_api.core import _load_break_schedule, _summarize_schedule
from kairos_api.server import app
from kairos_api.week_api_progress import build_progress

ROOT = Path(__file__).resolve().parents[1]
REAL_STORE = ROOT / "data" / "plan_targets.csv"
CHANNEL = "רשת 13"


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    path = tmp_path / "plan_targets.csv"
    monkeypatch.setenv(target_store.PATH_ENV, str(path))
    return path


@pytest.fixture()
def client():
    return TestClient(app)


def _plan_week():
    week = _summarize_schedule(_load_break_schedule()).get("week")
    if not isinstance(week, dict) or not week.get("date_from"):
        pytest.skip("no saved plan on this tree, so there is no window to measure")
    return week


def test_the_window_is_the_plans_own_week_not_a_window_of_its_own(isolated_store):
    week = _plan_week()
    body = build_progress()
    assert body["window"]["date_from"] == week["date_from"]
    assert body["window"]["date_to"] == week["date_to"]
    assert body["window"]["is_plan_week"] is True
    assert body["window"]["basis"] == week["basis"]


def test_the_projection_is_the_engines_own_figure_to_the_agora(isolated_store):
    """One number, one implementation: the strip and the headline cannot drift."""
    week = _plan_week()
    body = build_progress()
    assert body["projected"]["revenue"] == week["projected_revenue"]
    assert body["projected"]["breaks"] == week["total_breaks"]
    assert body["projected"]["ad_seconds"] == week["total_ad_seconds"]


def test_the_projection_is_scoped_to_the_operators_own_channel(isolated_store):
    body = build_progress()
    assert body["scope"]["scope_channel"] == CHANNEL
    assert body["scope"]["n_channels_total"] >= 1
    # No rival channel name may reach this payload in any form.
    text = json.dumps(body, ensure_ascii=False)
    for rival in ("קשת 12", "כאן 11", "עכשיו 14"):
        assert rival not in text, rival


def test_no_target_is_an_honest_unavailable_and_never_a_zero(isolated_store):
    body = build_progress()
    assert body["target"]["state"] == "unset"
    assert body["target"]["amount_ils"] is None
    assert body["verdict"]["state"] == "unavailable"
    assert body["verdict"]["reason"] == "no_target"
    assert body["verdict"]["variance_ils"] is None
    assert body["verdict"]["variance_percent"] is None
    # And the path to supply one is named rather than described.
    assert body["supply"]["route"] == "PUT /api/plan-target"
    assert body["supply"]["door"] == "today"


@pytest.mark.parametrize(
    "fraction,expected",
    [(0.95, "on_plan"), (1.03, "at_risk"), (1.30, "behind")],
)
def test_the_three_states_come_from_the_stores_own_rule(isolated_store, fraction, expected):
    week = _plan_week()
    projected = float(week["projected_revenue"])
    target_store.save_target(
        channel=CHANNEL,
        period_start=week["date_from"],
        period_end=week["date_to"],
        amount_ils=round(projected * fraction, 2),
        at_risk_band_percent=5.0,
        note="isolated store, test only",
    )
    body = build_progress()
    assert body["target"]["state"] == "set"
    assert body["verdict"]["state"] == expected
    assert body["verdict"]["threshold_en"].startswith("On plan at or above the target")
    assert body["verdict"]["threshold_he"].startswith("על התוכנית ביעד")
    # The variance is the subtraction the store made, never one made here.
    assert body["verdict"]["variance_ils"] == pytest.approx(
        projected - float(body["target"]["amount_ils"]), abs=0.01
    )


def test_a_target_on_another_window_is_disclosed_and_never_borrowed(isolated_store):
    week = _plan_week()
    target_store.save_target(
        channel=CHANNEL,
        period_start="2024-11-08",
        period_end="2024-11-14",
        amount_ils=1_000_000.0,
        at_risk_band_percent=5.0,
        note="a different span",
    )
    body = build_progress()
    assert body["target"]["state"] == "unset", "a different span must not read as this window's"
    assert body["verdict"]["state"] == "unavailable"
    assert len(body["other_windows"]) == 1
    assert body["other_windows"][0]["period_start"] == "2024-11-08"
    assert week["date_from"] != "2024-11-08"


def test_the_route_answers_and_stamps_can_edit(isolated_store, client):
    response = client.get("/api/plan-progress")
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert "can_edit" in body
    assert body["metric"] == "projected_revenue"
    assert body["currency"] == "ILS"


def test_the_route_accepts_an_explicit_window_and_says_it_is_not_the_plan_week(
    isolated_store, client
):
    response = client.get(
        "/api/plan-progress", params={"period_start": "2024-11-08", "period_end": "2024-11-14"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["window"]["date_from"] == "2024-11-08"
    assert body["window"]["is_plan_week"] is False
    assert body["window"]["basis"] == "requested"


def test_nothing_in_this_module_writes_the_operators_own_target_store(isolated_store):
    """The store on disk is the owner's. A test that writes it would be a lie in
    the product's own data, so this asserts the isolation rather than trusting it."""
    before = REAL_STORE.read_text(encoding="utf-8") if REAL_STORE.exists() else ""
    target_store.save_target(
        channel=CHANNEL,
        period_start="2024-11-01",
        period_end="2024-11-07",
        amount_ils=1.0,
        at_risk_band_percent=1.0,
        note="isolated",
    )
    after = REAL_STORE.read_text(encoding="utf-8") if REAL_STORE.exists() else ""
    assert before == after
    assert isolated_store.exists()
