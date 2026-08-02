"""The day board's routes, and the honesty of the answer they give a drag.

The measurement these tests protect is the one that decides what the surface is
allowed to show: moving a break inside its programme changes revenue by exactly
zero, because a break is priced on its programme's rating, rate, premium and
length and not on the minute it starts at. Changing the length moves real money.
A surface that animates a revenue figure on a horizontal drag would be showing a
number that did not move, so the difference is asserted here rather than trusted.
"""

from __future__ import annotations

import time
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import break_store

# The settings file is shared and mutable, so the channel these tests need can be
# blanked underneath them by anything else that writes it. The helper next door
# declares one when that happens, with the reasoning written where it is defined.
from test_p3_break_store import declare_operator_channel

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module", autouse=True)
def owned_channel():
    patch = declare_operator_channel()
    yield
    if patch is not None:
        patch.undo()


@pytest.fixture(scope="module")
def client():
    from kairos_api.break_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture(scope="module")
def day(client):
    payload = client.get("/api/plan/days").json()
    if not payload["available"]:
        pytest.skip(payload["reason"])
    return payload["days"][0]


@pytest.fixture(scope="module")
def board(client, day):
    response = client.get("/api/plan/day", params={"day": day})
    assert response.status_code == 200
    return response.json()


def test_the_days_route_names_the_operator_channel_and_offers_only_its_days(client):
    payload = client.get("/api/plan/days").json()
    assert payload["operator_channel"] == break_store.operator_channel()
    assert payload["count"] == len(payload["days"])
    if not payload["available"]:
        assert payload["reason"], "an unavailable day list must say what is missing"


def test_every_figure_on_the_board_carries_the_scope_it_was_computed_on(board):
    basis = board["basis"]
    assert basis["channel"] == board["operator_channel"]
    assert basis["day"] == board["day"]
    assert basis["currency"] == "ILS"
    assert basis["segments"] == len(board["programmes"])
    assert set(board["totals"]) == {
        "objective", "revenue", "retention", "breaks", "ad_seconds", "gold_breaks",
    }


def test_the_board_carries_only_the_operator_channel(board):
    owned = break_store.operator_channel()
    assert {row["channel"] for row in board["breaks"]} == {owned}
    assert {row["channel"] for row in board["programmes"]} == {owned}


def test_delivered_money_is_an_honest_state_and_never_a_figure(board):
    for row in board["breaks"]:
        delivered = row["delivered"]
        assert delivered["state"] in {"real", "unavailable", "unknown"}
        assert delivered["amount"] is None
        if delivered["state"] != "real":
            assert delivered["reason"]
            assert delivered["path_forward"]


def test_the_compliance_verdict_names_how_many_checks_were_run(board):
    compliance = board["compliance"]
    assert compliance["checks_run"] == 6
    assert len(compliance["checks"]) == 6
    assert compliance["compliant"] == (not compliance["violations"])


def test_moving_a_break_inside_its_programme_moves_no_money(client, board, day):
    item = board["breaks"][0]
    response = client.post("/api/plan/day/score", json={
        "day": day,
        "moves": [{"break_id": item["break_id"], "offset_seconds": item["offset_seconds"] + 60}],
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["changed_inputs"] == {"placement": True, "duration": False, "gold": False}
    assert payload["delta"]["revenue"] == 0.0
    assert payload["delta"]["retention"] == 0.0
    assert payload["delta"]["objective"] == 0.0
    assert payload["revenue_ignores"] == ["offset_seconds"]


def test_changing_the_length_of_a_break_moves_real_money(client, board, day):
    item = board["breaks"][0]
    response = client.post("/api/plan/day/score", json={
        "day": day,
        "moves": [{"break_id": item["break_id"], "duration_seconds": item["duration_seconds"] + 60}],
    })
    payload = response.json()
    assert payload["changed_inputs"]["duration"] is True
    assert payload["delta"]["revenue"] > 0.0
    assert payload["delta"]["ad_seconds"] == pytest.approx(60.0)


def test_the_score_of_an_untouched_day_is_the_day_itself(client, board, day):
    payload = client.post("/api/plan/day/score", json={"day": day, "moves": []}).json()
    assert payload["current"] == payload["saved"]
    assert payload["current"]["revenue"] == board["totals"]["revenue"]
    assert payload["delta"]["revenue"] == 0.0


def test_the_score_answers_well_inside_the_five_hundred_millisecond_bar(client, board, day):
    item = board["breaks"][0]
    body = {"day": day, "moves": [{"break_id": item["break_id"], "offset_seconds": 30}]}
    client.post("/api/plan/day/score", json=body)
    samples = []
    for _ in range(5):
        started = time.perf_counter()
        response = client.post("/api/plan/day/score", json=body)
        samples.append((time.perf_counter() - started) * 1000.0)
        assert response.status_code == 200
    assert max(samples) < 500.0, f"slowest score was {max(samples):.1f} ms"


def test_a_move_that_breaches_a_guardrail_is_refused_in_words_not_in_silence(client, board, day):
    """Two breaks pushed together must fail the spacing check with the numbers."""
    grouped: dict[str, list[dict]] = {}
    for row in board["breaks"]:
        grouped.setdefault(row["segment_id"], []).append(row)
    pair = next((rows for rows in grouped.values() if len(rows) >= 2), None)
    if pair is None:
        pytest.skip("no programme on this day carries two breaks to push together")
    first, second = sorted(pair, key=lambda row: row["start_seconds"])[:2]
    target = first["offset_seconds"] + first["duration_seconds"] + 10
    payload = client.post("/api/plan/day/score", json={
        "day": day,
        "moves": [{"break_id": second["break_id"], "offset_seconds": target}],
    }).json()
    codes = {violation["code"] for violation in payload["compliance"]["violations"]}
    assert "break_spacing" in codes
    breach = next(v for v in payload["compliance"]["violations"] if v["code"] == "break_spacing")
    assert breach["observed"] < breach["limit"]
    assert breach["detail"]


def test_an_edit_naming_a_break_that_is_not_in_the_day_is_refused(client, day):
    """Dropping an unknown edit in silence would score an arrangement nobody asked for."""
    response = client.post("/api/plan/day/score", json={
        "day": day, "moves": [{"break_id": "not-a-real-segment~1", "offset_seconds": 10}],
    })
    assert response.status_code == 404
    malformed = client.post("/api/plan/day/score", json={
        "day": day, "moves": [{"break_id": "no-ordinal-at-all", "offset_seconds": 10}],
    })
    assert malformed.status_code == 422


def test_one_break_opens_with_its_money_its_basis_and_its_uncertainty(client, board):
    item = board["breaks"][0]
    response = client.get(f"/api/breaks/{quote(item['break_id'], safe='')}")
    assert response.status_code == 200
    detail = response.json()
    assert detail["break_id"] == item["break_id"]
    assert detail["money"]["projected"]["amount"] == item["projected_revenue"]
    assert detail["money"]["projected"]["basis"]
    assert detail["money"]["delivered"]["state"] == "unavailable"
    assert detail["money"]["delivered"]["path_forward"]
    assert detail["retention"]["sample_breaks"] >= 0
    assert detail["retention"]["confidence"] in {"high", "medium", "low"}
    assert detail["contents"]["state"] == "unavailable"
    assert detail["contents"]["path_forward"]
    assert detail["programme"]["title"]


def test_the_stated_basis_multiplies_back_to_the_break_s_own_credit(client, board):
    """A basis that cannot reproduce the figure it explains is a wrong basis.

    Measured on רשת 13 / 2024-11-01 before this closed. The four breaks of
    ``2024-11-01|רשת 13|001`` are credited 10,711.71, 10,162.61, 9,613.52 and
    9,064.43, while the stated basis, rate per point 60 times the baseline rating
    1.7 times 120 s over a 1 s unit times a premium of 0.92, gives 11,260.80 for
    every one of them: over by 549.09 on the first break and by 2,196.37 on the
    fourth. The engine prices a break on the rating that survives once the break
    is present, so the four ratios are 1 minus k times the 0.048762 retention
    cost, and the basis now says so and carries that rating.

    Every break in the day is checked, not one. Across the 80 breaks the worst
    disagreement was 6,072.33 ILS before and 0.0065 ILS after, the latter being
    the six decimal places the rating is published to.
    """
    assert board["breaks"], "a day with no breaks proves nothing here"
    worst = 0.0
    ladders: dict[str, list[float]] = {}
    for row in board["breaks"]:
        detail = client.get(f"/api/breaks/{quote(row['break_id'], safe='')}").json()
        programme = detail["programme"]
        projected = detail["money"]["projected"]
        assert projected["rating_at_this_break"] > 0
        assert 0.0 < projected["retention_at_this_break"] <= 1.0
        units = detail["placement"]["duration_seconds"] / programme["rate_unit_seconds"]
        rebuilt = programme["rate_per_point"] * projected["rating_at_this_break"] * units * programme["premium"]
        worst = max(worst, abs(rebuilt - projected["amount"]))
        ladders.setdefault(row["segment_id"], []).append(projected["amount"])
    assert worst <= 0.01, f"the stated basis missed the credit by {worst}"

    # The property the old basis denied: a later break in a programme earns less
    # than an earlier one, because the audience it is priced on is smaller.
    laddered = [amounts for amounts in ladders.values() if len(amounts) > 1]
    assert laddered, "this day carries no programme with two breaks, so the ladder is untested"
    for amounts in laddered:
        assert amounts == sorted(amounts, reverse=True), amounts


def test_the_retention_the_break_is_priced_at_is_the_engine_s_own_and_not_a_restatement(client, board):
    """The published rating comes from the optimizer's own seam, so it agrees.

    The alternative was to restate four lines of engine arithmetic in the API,
    which is a second price. This asserts the two are the same object: the rating
    published for the k-th break equals the segment's baseline rating times the
    engine's own retention at k breaks.
    """
    from kairos.optimize._segment_math import _segment_retention

    plan = break_store.day_plan(board["day"])
    plans = {row.segment_id: row for row in plan.result.segments}
    checked = 0
    for row in board["breaks"][:12]:
        detail = client.get(f"/api/breaks/{quote(row['break_id'], safe='')}").json()
        segment = plan.segment(row["segment_id"])
        assert segment is not None
        from dataclasses import replace

        priced = replace(segment, impact_coefficient=float(plans[row["segment_id"]].retention_cost_used))
        expected = _segment_retention(priced, int(row["ordinal"]))
        assert detail["money"]["projected"]["retention_at_this_break"] == pytest.approx(expected, abs=5e-7)
        rating = float(segment.baseline_tvr) * expected
        assert detail["money"]["projected"]["rating_at_this_break"] == pytest.approx(rating, abs=5e-7)
        checked += 1
    assert checked == 12


def test_a_break_on_a_day_with_no_plan_is_a_404_and_not_an_invented_break(client):
    response = client.get(f"/api/breaks/{quote('1999-01-01|nowhere|000~1', safe='')}")
    assert response.status_code == 404


def test_the_hour_a_break_sits_in_opens_the_breaks_that_make_its_load(client, board):
    """A load is the sum of objects, and every one of those objects is addressable.

    The hour row states ad seconds against a licence limit. Without the breaks
    behind it the figure is a dead end: a person reading that an hour carries
    480 s of a 720 s allowance has no way from the number to the four breaks that
    make it, which is the one thing they would do next.
    """
    item = board["breaks"][0]
    detail = client.get(f"/api/breaks/{quote(item['break_id'], safe='')}").json()
    hour_row = detail["guardrails"]["hour"]
    rows = detail["guardrails"]["hour_breaks"]
    assert rows, "the hour holding this break holds at least this break"
    assert len(rows) == hour_row["breaks"]
    assert sum(row["duration_seconds"] for row in rows) == pytest.approx(hour_row["ad_seconds"])
    on_board = {row["break_id"] for row in board["breaks"] if row["hour"] == hour_row["hour"]}
    assert {row["break_id"] for row in rows} == on_board, "the hour and the board disagree about the same hour"
    assert item["break_id"] in on_board
    assert rows == sorted(rows, key=lambda row: row["start_seconds"])
    for row in rows[:3]:
        opened = client.get(f"/api/breaks/{quote(row['break_id'], safe='')}")
        assert opened.status_code == 200, f"{row['break_id']} is offered as a link and does not resolve"


def test_the_programme_a_break_names_is_addressable_and_not_only_a_title(client, board):
    """The drawer's title opens a record, so the payload has to carry its id."""
    item = board["breaks"][0]
    detail = client.get(f"/api/breaks/{quote(item['break_id'], safe='')}").json()
    assert detail["programme"]["segment_id"] == item["segment_id"]
    assert detail["identity"]["channel"] == board["operator_channel"]
    assert detail["identity"]["day"] == board["day"]
    named = {row["segment_id"] for row in board["programmes"]}
    assert detail["programme"]["segment_id"] in named, "the id names a programme the board actually drew"


def test_the_hour_strip_reports_the_load_the_licence_is_measured_against(board):
    assert board["hours"], "a day with breaks has at least one loaded hour"
    for row in board["hours"]:
        assert row["max_ad_seconds"] == board["guardrails"]["max_ad_seconds_per_hour"]
        assert row["max_breaks"] == board["guardrails"]["max_breaks_per_hour"]
        assert row["over_ad_seconds"] == (row["ad_seconds"] > row["max_ad_seconds"])
        assert row["over_breaks"] == (row["breaks"] > row["max_breaks"])
