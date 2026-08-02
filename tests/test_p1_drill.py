"""The money drill's second level: one day, and the rows that produced it.

A figure that opens its parts once is a summary. A figure that opens its parts
twice, and whose parts add back up at both levels, is auditable. These tests
hold the second level to the same rules the first one already passes: the same
arithmetic, the operator's channel only, no training word, and an absence named
rather than filled in where the level below does not exist.
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

TRAINING_LEXICON = (
    "gate",
    "held_out",
    "tau",
    "drift",
    "coefficient",
    "pooling",
    "p_value",
    "training_window",
    "wartime",
)


@pytest.fixture(scope="module")
def owned_channel():
    """Pin the operator's channel, for the reason test_p1_today.py states."""
    from kairos_api import channel_scope, core, overview_api, overview_api_target
    from kairos_api.core import _load_break_schedule, _load_settings

    plan_channels = sorted(
        {str(name).strip() for name in _load_break_schedule()["channel"].astype(str).unique() if str(name).strip()}
    )
    assert len(plan_channels) > 1, "the reference plan must carry a lineup for the boundary to be testable"
    chosen = "רשת 13" if "רשת 13" in plan_channels else plan_channels[0]
    pinned = _load_settings().model_copy(update={"operator_channel": chosen})

    saved = (channel_scope.operator_channel, core._load_settings, overview_api._load_settings, overview_api_target._load_settings)
    channel_scope.operator_channel = lambda settings=None: chosen
    core._load_settings = lambda: pinned
    overview_api._load_settings = lambda: pinned
    overview_api_target._load_settings = lambda: pinned
    overview_api._overview_cached.cache_clear()
    yield chosen
    channel_scope.operator_channel, core._load_settings, overview_api._load_settings, overview_api_target._load_settings = saved
    overview_api._overview_cached.cache_clear()


@pytest.fixture(scope="module")
def client(owned_channel) -> TestClient:
    from kairos_api import overview_api

    app = FastAPI()
    app.include_router(overview_api.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def today(client) -> dict:
    response = client.get("/api/today")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture(scope="module")
def middle_day(client, today) -> dict:
    days = today["money"]["days"]
    assert len(days) >= 3, "the window needs three days for the walk to be testable"
    response = client.get(f"/api/today/day/{days[2]['date']}")
    assert response.status_code == 200, response.text
    return response.json()


def test_the_rows_behind_a_day_sum_back_to_that_day(middle_day):
    assert middle_day["available"] is True
    assert middle_day["row_count"] == len(middle_day["rows"])
    assert middle_day["rows_total_ils"] == pytest.approx(middle_day["projected_revenue"], abs=0.5)
    assert middle_day["reconciled"] is True
    assert abs(middle_day["residual_ils"]) < 0.5


def test_the_day_in_the_drill_is_the_same_day_the_window_figure_listed(today, middle_day):
    """Two levels of one figure, never two computations of it."""
    listed = today["money"]["days"][2]
    assert middle_day["date"] == listed["date"]
    assert middle_day["projected_revenue"] == listed["projected_revenue"]
    assert middle_day["total_breaks"] == listed["total_breaks"]
    assert middle_day["total_ad_seconds"] == listed["total_ad_seconds"]


def test_every_row_carries_the_facts_the_figure_was_made_of(middle_day):
    rows = middle_day["rows"]
    assert rows, "a day with money must have rows behind it"
    for row in rows:
        assert row["segment_id"]
        assert row["start_clock"]
        assert row["projected_revenue"] is not None
        assert row["breaks"] >= 0
    shares = [row["share_percent"] for row in rows if row["share_percent"] is not None]
    assert sum(shares) == pytest.approx(100.0, abs=1.0)
    assert rows == sorted(rows, key=lambda row: row["projected_revenue"], reverse=True)


def test_the_reader_keeps_their_place_in_the_set_they_came_from(client, today):
    """Linear's device: a record says where it sits and walks the set from inside."""
    days = [day["date"] for day in today["money"]["days"]]
    first = client.get(f"/api/today/day/{days[0]}").json()["position"]
    middle = client.get(f"/api/today/day/{days[2]}").json()["position"]
    last = client.get(f"/api/today/day/{days[-1]}").json()["position"]
    assert first == {"index": 1, "total": len(days), "previous": None, "next": days[1]}
    assert middle == {"index": 3, "total": len(days), "previous": days[1], "next": days[3]}
    assert last["index"] == len(days)
    assert last["next"] is None


def test_no_rival_channel_reaches_the_second_level(client, today, middle_day):
    from kairos_api.core import _load_break_schedule

    owned = today["money"]["scope"]["channel"]
    rivals = {
        str(name).strip()
        for name in _load_break_schedule()["channel"].astype(str).unique()
        if str(name).strip() and str(name).strip() != owned
    }
    assert rivals, "the reference data carries rival channels, so this test can bite"
    body = client.get(f"/api/today/day/{middle_day['date']}").text
    for rival in rivals:
        assert rival not in body
    assert middle_day["boundary"]["scope_channel"] == owned
    assert middle_day["boundary"]["competitor_rows_excluded"] > 0
    assert all(row["segment_id"].split("|")[1] == owned for row in middle_day["rows"] if "|" in row["segment_id"])


def test_no_training_word_reaches_the_second_level(client, middle_day):
    body = client.get(f"/api/today/day/{middle_day['date']}").text
    assert not {word: 1 for word in TRAINING_LEXICON if re.search(word, body, re.I)}


def test_the_level_below_is_named_and_never_filled_in(middle_day):
    """Delivered money for this window does not exist, so it is an absence.

    And the absence is measured through the route, not asserted in it: the
    coverage the sentence names is the coverage the ledger on disk actually has,
    read by the same reader the rest of the product reads it with.
    """
    from kairos.export.spots_coverage import daily_input_days

    covered = sorted(daily_input_days())
    delivered = middle_day["delivered"]
    assert delivered["available"] is False
    assert delivered["state"] == "unavailable"
    assert delivered["covers"] == covered
    assert middle_day["date"] not in covered, "this day is outside the ledger, which is why it is an absence"
    assert delivered["reason_en"] and delivered["reason_he"]
    assert delivered["needs_en"] and delivered["needs_he"]
    for date in covered:
        assert date in delivered["reason_en"], "the sentence names the coverage it was derived from"
    assert delivered["opens"] == "sources"
    assert not any(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in delivered.values()
    )


def test_a_date_the_plan_does_not_cover_is_an_absence_and_not_a_zero(client):
    body = client.get("/api/today/day/2030-01-01").json()
    assert body["available"] is False
    assert body["reason"]
    assert body["projected_revenue"] is None
    assert body["rows"] == []
    assert body["position"]["index"] is None


def test_the_money_figure_carries_the_basis_a_reader_needs_to_check_it(today):
    """Stripe's four: whose, over what range, on which edges, in which zone."""
    scope = today["money"]["scope"]
    assert scope["channel"]
    assert scope["date_from"] and scope["date_to"]
    assert scope["inclusive"] is True
    assert scope["timezone"] == "Asia/Jerusalem"
    assert scope["currency"] == "ILS"
    assert scope["source"] == "saved_plan"
