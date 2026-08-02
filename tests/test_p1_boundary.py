"""What Today reports before the operator has declared which channel is theirs.

The saved weekly plan carries every channel in the market, because the retention
model is measured against the lineup. Until settings name one of them as the
operator's, ``channel_scope`` cannot filter and hands the caller the whole
market plus the reason it could not scope. What the caller does with that is the
whole of this file.

The wrong answer is to serve it. Measured on the reference plan with the
operator channel blank, the unscoped window total is 54,651,396.48 against the
operator's own 10,123,070.80, so serving it would put a five-times-too-large
figure and three rival broadcasters' programmes on an operator's home screen at
once. The right answer is one absence, one cause and one control.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def no_channel():
    from kairos_api import channel_scope, core, overview_api, overview_api_target
    from kairos_api.core import _load_settings

    blank = _load_settings().model_copy(update={"operator_channel": ""})
    saved = (channel_scope.operator_channel, core._load_settings, overview_api._load_settings, overview_api_target._load_settings)
    channel_scope.operator_channel = lambda settings=None: ""
    core._load_settings = lambda: blank
    overview_api._load_settings = lambda: blank
    overview_api_target._load_settings = lambda: blank
    overview_api._overview_cached.cache_clear()
    yield
    channel_scope.operator_channel, core._load_settings, overview_api._load_settings, overview_api_target._load_settings = saved
    overview_api._overview_cached.cache_clear()


@pytest.fixture(scope="module")
def client(no_channel) -> TestClient:
    from kairos_api import overview_api

    app = FastAPI()
    app.include_router(overview_api.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def today(client) -> dict:
    response = client.get("/api/today")
    assert response.status_code == 200, response.text
    return response.json()


def test_no_channel_name_at_all_reaches_the_payload(client, today):
    """Not one of the four, because none of them has been claimed."""
    from kairos_api.core import _load_break_schedule

    everyone = {
        str(name).strip()
        for name in _load_break_schedule()["channel"].astype(str).unique()
        if str(name).strip()
    }
    assert len(everyone) > 1
    body = client.get("/api/today").text
    assert not [name for name in everyone if name in body]
    assert today["channel"] is None


def test_the_market_total_is_never_served_as_the_operators_money(today):
    money = today["money"]
    assert money["available"] is False
    assert money["amount_ils"] is None
    assert money["total_breaks"] is None
    assert money["total_ad_seconds"] is None
    assert money["average_retention"] is None
    assert money["days"] == []
    assert money["scope"]["channel"] is None
    assert money["unavailable"]["reason"] == "no_operator_channel"


def test_the_absence_names_its_cause_and_the_control_that_ends_it(today):
    withheld = today["money"]["unavailable"]
    assert withheld["reason_en"] and withheld["reason_he"]
    assert withheld["needs_en"] and withheld["needs_he"]
    assert withheld["opens"] == "settings"
    lead = today["health"]["checks"][0]
    assert lead["id"] == "operator_channel_unset"
    assert lead["status"] == "attention"
    assert lead["opens"] == "settings"


def test_rival_segments_are_not_offered_as_the_operators_priorities(today):
    decisions = today["decisions"]
    assert decisions["items"] == []
    assert decisions["count"] == 0
    assert decisions["unavailable"]["reason"] == "no_operator_channel"


def test_the_drill_refuses_rather_than_listing_the_whole_market(client, today):
    from kairos_api.core import _load_break_schedule

    date = str(_load_break_schedule()["date"].astype(str).min())
    body = client.get(f"/api/today/day/{date}")
    assert body.status_code == 200
    detail = body.json()
    assert detail["available"] is False
    assert detail["rows"] == []
    assert detail["projected_revenue"] is None
    assert "settings" in detail["reason"]


def test_the_verdict_is_unavailable_rather_than_measured_against_the_market(today):
    assert today["verdict"]["state"] == "unavailable"
    assert today["verdict"]["variance_ils"] is None
