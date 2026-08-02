"""P2: the competitor boundary on the week's own reads.

Section 8.3 of the specification names P2 as the piece that scopes
``/api/schedule``, and W0-4 measured the breach it has to close: with
``operator_channel = רשת 13`` the 200-row ``break_schedule`` slice carried 96
rows of קשת 12, 73 of כאן 11, 28 of עכשיו 14 and 3 of the operator's own, and the
``rows`` canvas carried 1,852 programmes of which 1,328 were competitors'.

These tests drive the real route through the real app, because a boundary that
holds in a unit and leaks through a cache key is not a boundary. The saved plan
on disk carries four channels, so the assertions below are meaningful on the
shipped data rather than on a fixture built to pass them.
"""

from __future__ import annotations

import collections

import pytest
from fastapi.testclient import TestClient

from kairos_api import week_api
from kairos_api.server import app


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def schedule(client):
    week_api._schedule_cached.cache_clear()
    response = client.get("/api/schedule")
    assert response.status_code == 200, response.text
    return response.json()


def _owned(client) -> str:
    settings = client.get("/api/settings")
    assert settings.status_code == 200, settings.text
    return str(settings.json().get("operator_channel") or "").strip()


def test_the_payload_carries_the_scope_it_was_summed_on(schedule):
    scope = schedule["scope"]
    for side in ("plan", "programmes"):
        note = scope[side]
        assert set(note) >= {
            "scope_channel", "scoped", "rows_in", "rows_out",
            "competitor_rows_excluded", "competitor_channels_excluded", "reason",
        }


def test_no_competitor_row_reaches_the_plan_slice(client, schedule):
    owned = _owned(client)
    if not owned:
        pytest.skip("no operator channel is configured, so there is no boundary to enforce")
    channels = collections.Counter(row.get("channel") for row in schedule["break_schedule"])
    assert set(channels) == {owned}, channels


def test_no_competitor_row_reaches_the_week_canvas(client, schedule):
    owned = _owned(client)
    if not owned:
        pytest.skip("no operator channel is configured, so there is no boundary to enforce")
    assert [row["channel"] for row in schedule["rows"]] == [owned]


def test_no_competitor_programme_reaches_the_embedded_break_board(client, schedule):
    owned = _owned(client)
    if not owned:
        pytest.skip("no operator channel is configured, so there is no boundary to enforce")
    programmes = schedule["break_operations"].get("programs", [])
    assert programmes, "the embedded break board is empty, so this test would prove nothing"
    assert {programme.get("channel") for programme in programmes} == {owned}


def test_the_row_count_is_the_operators_own_and_the_scope_says_what_it_dropped(client, schedule):
    owned = _owned(client)
    if not owned:
        pytest.skip("no operator channel is configured, so there is no boundary to enforce")
    note = schedule["scope"]["plan"]
    assert note["scoped"] is True
    assert note["scope_channel"] == owned
    # The count the client prints as "200 of N" is the operator's N, not the
    # market's, so the two figures on that line share one scope.
    assert schedule["break_schedule_total_rows"] == note["rows_out"]
    assert note["rows_out"] < note["rows_in"]
    assert note["competitor_rows_excluded"] == note["rows_in"] - note["rows_out"]
    assert note["competitor_channels_excluded"] >= 1


def test_an_unconfigured_channel_says_why_rather_than_serving_the_market(monkeypatch):
    """With no owned channel the rows pass through and the note names the reason."""
    from kairos_api import channel_scope

    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: "")
    week_api._schedule_cached.cache_clear()
    try:
        payload = week_api._schedule_cached((("unscoped", 0, 0),))
        note = payload["scope"]["plan"]
        assert note["scoped"] is False
        assert note["reason"] == channel_scope.NO_OPERATOR_CHANNEL_REASON
        assert note["competitor_rows_excluded"] == 0
    finally:
        week_api._schedule_cached.cache_clear()
