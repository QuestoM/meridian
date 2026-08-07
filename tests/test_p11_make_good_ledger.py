"""P11: the make-good ledger, through the real router over redirected stores.

Every store this file touches is pointed at a temporary directory, so the suite
never writes a make-good into the operator's real data. The campaign store and the
delivery ledger are written here as CSV in exactly the columns the shipped readers
expect, so the assertions are about those readers.

The bar the assertions encode is the one the spec names for a make-good: a store
keyed on the flight, holding the measured shortfall, an approval state and a
back-link, with nothing derived that the owner never supplied.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import makegood_store as ledger

CHANNEL = "רשת 13"

CAMPAIGN_HEADER = (
    "record_type,campaign_id,flight_id,name,advertiser,agency_id,status,starts_on,ends_on,"
    "goal_kind,goal_value,rebate_percent,surcharge_discount_percent,surcharge_weekdays,notes,"
    "created_at,created_by,data_source,channel,brand,category,budget_ils,bonus_ils,"
    "rating_goal_points,rating_goal_audience,price_model,priority,pacing_mode,is_demo,demo_note\n"
)
DELIVERY_HEADER = (
    "campaign_id,broadcast_date,air_state,channel,spots,seconds,rating_points_planned,spend_ils,"
    "spots_dropped_by_rule,dropped_rule_id,figures_basis,source_file,counted_as_of,"
    "counted_as_of_basis,is_demo,note\n"
)


def _campaign_row(campaign_id: str, *, goal: str, channel: str = CHANNEL,
                  starts: str = "2025-04-27", ends: str = "2025-04-29") -> str:
    return (
        f"campaign,{campaign_id},,{campaign_id} name,מפרסם,AGY_01,active,{starts},{ends},"
        f",,,,,,2026-08-07T00:00:00+00:00,,manual,{channel},,,70000,,{goal},all_viewers,cpp,,,false,\n"
    )


def _delivery_row(campaign_id: str, when: str, state: str, rating: str, spend: str = "1000") -> str:
    figures = "" if state == "unknown" else "traffic file"
    source = "" if state == "unknown" else "Wally_Prime_Reshet_Example_2025-04-27.csv"
    value = "" if state == "unknown" else rating
    money = "" if state == "unknown" else spend
    return (
        f"{campaign_id},{when},{state},{CHANNEL},1,30,{value},{money},0,,{figures},{source},"
        f"2025-04-29T23:00:00,the last programme booked on the newest sourced day,false,\n"
    )


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client over only P11's router, with every store inside tmp_path."""
    from kairos_api import campaigns_api_store, campaigns_delivery, channel_scope, pacing_alerts_api

    campaigns = tmp_path / "campaigns.csv"
    campaigns.write_text(
        CAMPAIGN_HEADER
        # Behind, and every elapsed day sourced: a real gap to date.
        + _campaign_row("CMP_BEHIND", goal="90")
        # On pace, so there is nothing to raise against.
        + _campaign_row("CMP_OK", goal="30")
        # A competitor's campaign, which must never reach a row.
        + _campaign_row("CMP_RIVAL", goal="90", channel="קשת 12"),
        encoding="utf-8",
    )
    delivery = tmp_path / "campaign_delivery.csv"
    delivery.write_text(
        DELIVERY_HEADER
        + _delivery_row("CMP_BEHIND", "2025-04-27", "aired", "10")
        + _delivery_row("CMP_BEHIND", "2025-04-28", "aired", "10")
        + _delivery_row("CMP_BEHIND", "2025-04-29", "aired", "10")
        + _delivery_row("CMP_OK", "2025-04-27", "aired", "10")
        + _delivery_row("CMP_OK", "2025-04-28", "aired", "10")
        + _delivery_row("CMP_OK", "2025-04-29", "aired", "10")
        + _delivery_row("CMP_RIVAL", "2025-04-27", "aired", "1"),
        encoding="utf-8",
    )
    monkeypatch.setattr(campaigns_api_store, "CAMPAIGNS_PATH", campaigns)
    monkeypatch.setattr(campaigns_delivery, "DELIVERY_PATH", delivery)
    monkeypatch.setattr(ledger, "MAKE_GOODS_PATH", tmp_path / "make_goods.csv")
    monkeypatch.setattr(ledger, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: CHANNEL)

    app = FastAPI()
    app.include_router(pacing_alerts_api.router)
    return TestClient(app)


def test_the_board_serves_only_the_operators_own_channel(client) -> None:
    body = client.get("/api/pacing").json()
    ids = [row["campaign_id"] for row in body["rows"]]
    assert "CMP_RIVAL" not in ids
    assert set(ids) == {"CMP_BEHIND", "CMP_OK"}
    assert body["scope"]["scope_channel"] == CHANNEL
    assert body["scope"]["competitor_rows_excluded"] == 1
    assert "קשת 12" not in client.get("/api/pacing").text


def test_a_raise_measures_the_shortfall_and_the_caller_cannot_supply_one(client) -> None:
    # The request model carries a campaign and a note and no figure at all.
    from kairos_api.pacing_alerts_api import RaiseMakeGood

    assert set(RaiseMakeGood.model_fields) == {"campaign_id", "note"}

    response = client.post("/api/make-goods", json={"campaign_id": "CMP_BEHIND", "note": "client called"})
    assert response.status_code == 201, response.text
    record = response.json()["make_good"]
    # Goal 90 over 3 days, 30 counted through day 3, so the gap to the reference is 60.
    assert record["shortfall"]["goal_value"] == 90.0
    assert record["shortfall"]["counted_value"] == 30.0
    assert record["shortfall"]["deficit_value"] == 60.0
    assert record["shortfall"]["unit"] == "rating_points"
    assert record["shortfall"]["deficit_kind"] == ledger.MEASURED_CLOSED
    assert record["shortfall"]["counted_as_of"] == "2025-04-29T23:00:00"
    assert record["campaign_id"] == "CMP_BEHIND"
    assert record["state"] == ledger.RAISED
    assert record["offer"]["value"] is None


def test_a_campaign_with_no_measured_shortfall_cannot_carry_a_make_good(client) -> None:
    response = client.post("/api/make-goods", json={"campaign_id": "CMP_OK"})
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["message_en"] and detail["message_he"]
    assert detail["opens"] == {"kind": "campaign", "id": "CMP_OK"}


def test_a_campaign_on_another_channel_is_not_a_campaign_here(client) -> None:
    response = client.post("/api/make-goods", json={"campaign_id": "CMP_RIVAL"})
    assert response.status_code == 404
    assert "קשת 12" not in response.text


def test_a_second_raise_against_an_open_make_good_is_refused_by_name(client) -> None:
    first = client.post("/api/make-goods", json={"campaign_id": "CMP_BEHIND"}).json()["make_good"]
    again = client.post("/api/make-goods", json={"campaign_id": "CMP_BEHIND"})
    assert again.status_code == 409
    assert again.json()["detail"]["opens"] == {"kind": "make_good", "id": first["make_good_id"]}


def test_the_state_machine_refuses_a_move_it_does_not_hold_and_names_what_it_does(client) -> None:
    record = client.post("/api/make-goods", json={"campaign_id": "CMP_BEHIND"}).json()["make_good"]
    make_good_id = record["make_good_id"]
    assert record["next_states"] == ["offered", "withdrawn"]

    straight_to_settled = client.post(f"/api/make-goods/{make_good_id}/state", json={"state": "settled"})
    assert straight_to_settled.status_code == 409
    detail = straight_to_settled.json()["detail"]
    # The refusal names the states a person may move to, in the words the ledger
    # publishes for them, and never in the keys the rows are stored under.
    labels = {entry["value"]: entry for entry in ledger.STATE_VOCABULARY}
    assert labels["offered"]["label_en"] in detail["message_en"]
    assert labels["offered"]["label_he"] in detail["message_he"]
    assert labels["settled"]["label_en"] in detail["message_en"]
    for key in ledger.TRANSITIONS:
        assert key not in detail["message_en"], key
        assert key not in detail["message_he"], key


def test_an_offer_is_a_persons_number_and_settling_needs_one(client) -> None:
    record = client.post("/api/make-goods", json={"campaign_id": "CMP_BEHIND"}).json()["make_good"]
    make_good_id = record["make_good_id"]

    no_value = client.post(f"/api/make-goods/{make_good_id}/state", json={"state": "offered"})
    assert no_value.status_code == 400

    backwards = client.post(f"/api/make-goods/{make_good_id}/state", json={
        "state": "offered", "offer_value": 60,
        "offer_window_start": "2025-05-10", "offer_window_end": "2025-05-04",
    })
    assert backwards.status_code == 400

    offered = client.post(f"/api/make-goods/{make_good_id}/state", json={
        "state": "offered", "offer_value": 60,
        "offer_window_start": "2025-05-04", "offer_window_end": "2025-05-10",
        "note": "three days in the same programme",
    })
    assert offered.status_code == 200, offered.text
    body = offered.json()["make_good"]
    assert body["state"] == ledger.OFFERED
    assert body["offer"]["value"] == 60.0
    assert body["offer"]["window_start"] == "2025-05-04"
    assert body["offer"]["offered_at"]
    # The measured shortfall is untouched by the offer.
    assert body["shortfall"]["deficit_value"] == 60.0

    settled = client.post(f"/api/make-goods/{make_good_id}/state", json={"state": "settled"})
    assert settled.status_code == 200
    assert settled.json()["make_good"]["state"] == ledger.SETTLED
    assert settled.json()["make_good"]["closed_at"]
    assert settled.json()["make_good"]["next_states"] == []


def test_a_settled_make_good_frees_the_campaign_and_a_withdrawn_one_is_not_deleted(client) -> None:
    first = client.post("/api/make-goods", json={"campaign_id": "CMP_BEHIND"}).json()["make_good"]
    client.post(f"/api/make-goods/{first['make_good_id']}/state", json={"state": "withdrawn", "note": "raised in error"})

    ledger_body = client.get("/api/make-goods").json()
    assert ledger_body["count"] == 1
    assert ledger_body["open_count"] == 0
    assert ledger_body["make_goods"][0]["state"] == ledger.WITHDRAWN
    assert ledger_body["make_goods"][0]["close_note"] == "raised in error"

    second = client.post("/api/make-goods", json={"campaign_id": "CMP_BEHIND"})
    assert second.status_code == 201
    assert client.get("/api/make-goods").json()["count"] == 2


def test_the_ledger_publishes_what_it_does_not_decide(client) -> None:
    body = client.get("/api/make-goods").json()
    assert body["sign_off"]["configured"] is False
    assert body["sign_off"]["reason_he"] and body["sign_off"]["path_forward_he"]
    assert body["sign_off"]["offer_reserves_nothing_en"]
    states = {entry["value"] for entry in body["vocabulary"]["states"]}
    assert states == {"raised", "offered", "settled", "declined", "withdrawn"}
    for entry in body["vocabulary"]["states"]:
        assert entry["label_he"] and entry["meaning_he"]


def test_the_board_names_the_campaigns_that_already_carry_an_open_make_good(client) -> None:
    record = client.post("/api/make-goods", json={"campaign_id": "CMP_BEHIND"}).json()["make_good"]
    board_body = client.get("/api/pacing").json()
    assert board_body["make_goods"] == {"CMP_BEHIND": [record["make_good_id"]]}


def test_the_ledger_row_survives_a_reload_from_disk_exactly(client, tmp_path) -> None:
    written = client.post("/api/make-goods", json={"campaign_id": "CMP_BEHIND", "note": "n"}).json()["make_good"]
    reread = client.get("/api/make-goods").json()["make_goods"][0]
    assert reread == written
    assert (tmp_path / "make_goods.csv").exists()
