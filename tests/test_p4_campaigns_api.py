"""P4: the campaign and flight entity, and the one flow that creates a client.

Every store this file touches is redirected into a temporary directory, so the
suite never writes an agency, a link, a campaign or a version into the
operator's real data. That is deliberate and it is the defect section 5.6 of the
specification records: 186 of the 200 manifests on disk were written by pytest.

The bars measured here are JS-5's, stated as assertions rather than as prose:
all three entities exist and are linked after one request, a repeat of the same
insertion order creates no duplicate, and a booked goal never becomes a
delivered figure.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client over only P4's routers, with every store pointed at tmp_path."""
    from kairos_api import agencies, agency_conditions, campaigns_api, campaigns_api_store, version_store

    monkeypatch.setattr(agencies, "AGENCIES_PATH", tmp_path / "agencies.csv")
    monkeypatch.setattr(agencies, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agency_conditions, "LINKS_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(agency_conditions, "CONDITIONS_PATH", tmp_path / "agency_conditions.csv")
    monkeypatch.setattr(agency_conditions, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(campaigns_api_store, "CAMPAIGNS_PATH", tmp_path / "campaigns.csv")
    monkeypatch.setattr(campaigns_api_store, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(version_store, "snapshot_manual_edit", lambda request, logical: None)
    # No daily file resolves inside the temp tree, so the observed link layer is
    # empty and every link a test sees is one a test made.
    monkeypatch.setattr(agency_conditions, "_latest_daily_pairs", lambda: ([], None))

    app = FastAPI()
    app.include_router(agencies.router)
    app.include_router(agency_conditions.router)
    app.include_router(campaigns_api.router)
    return TestClient(app)


def _agency(client, agency_id="AGY_01", name="OMD"):
    response = client.post("/api/agencies", json={
        "agency_id": agency_id,
        "name": name,
        "rebate_percent": 4.0,
        "commission_percent": 15.0,
        "payment_terms_days": 60,
        "credit_limit_ils": 3000000,
        "vat_id": "513200001",
        "contact_name": "מיכל ברקוביץ",
    })
    assert response.status_code == 201, response.text
    return response.json()


ORDER = {
    "advertiser": "בנק הפועלים",
    "campaign_name": "מתחתנים 2026",
    "campaign_starts_on": "2026-09-01",
    "campaign_ends_on": "2026-09-30",
    "rebate_percent": 4.0,
    "surcharge_discount_percent": 20.0,
    "surcharge_weekdays": "6",
    "flights": [
        {"starts_on": "2026-09-01", "ends_on": "2026-09-15", "goal_kind": "spots", "goal_value": 40},
        {"starts_on": "2026-09-16", "ends_on": "2026-09-30", "goal_kind": "grp", "goal_value": 120},
    ],
}


def test_the_store_starts_empty_and_says_so(client):
    """No campaign is not an error, and the delivery limit is on the payload."""
    payload = client.get("/api/clients/campaigns").json()
    assert payload["campaigns"] == []
    assert payload["count"] == 0
    assert payload["delivery"]["available"] is False
    assert "delivery" in payload["delivery"]["reason_en"].lower()
    assert payload["delivery"]["reason_he"]
    assert payload["delivery"]["path_forward_en"]
    assert payload["delivery"]["path_forward_he"]
    assert payload["terms"]["priced_by_engine"] is False
    assert payload["terms"]["reason_en"] and payload["terms"]["reason_he"]


def test_one_request_creates_and_links_all_three_entities(client):
    """JS-5's done condition: three entities exist, linked, visible after."""
    _agency(client)
    response = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    assert response.status_code == 201, response.text
    result = response.json()

    assert result["agency"]["agency_id"] == "AGY_01"
    assert result["agency"]["outcome"] == "reused"
    assert result["advertiser"]["outcome"] == "linked"
    assert result["campaign"]["advertiser"] == ORDER["advertiser"]
    assert result["campaign"]["agency_id"] == "AGY_01"
    assert len(result["flights"]) == 2

    links = client.get("/api/agencies/AGY_01/advertisers").json()
    assert ORDER["advertiser"] in links["effective"]

    campaigns = client.get("/api/clients/campaigns").json()["campaigns"]
    assert len(campaigns) == 1
    assert campaigns[0]["campaign_id"] == result["campaign"]["campaign_id"]
    assert [flight["goal_kind"] for flight in campaigns[0]["flights"]] == ["spots", "grp"]


def test_a_new_agency_is_created_inside_the_same_flow(client):
    """The account manager never leaves to make the agency first."""
    response = client.post("/api/clients/onboarding", json={
        "agency": {"name": "אגנסי חדשה", "rebate_percent": 3.5, "payment_terms_days": 45},
        **ORDER,
    })
    assert response.status_code == 201, response.text
    result = response.json()
    assert result["agency"]["outcome"] == "created"
    assert result["created"]["agency"] is True
    stored = client.get(f"/api/agencies/{result['agency']['agency_id']}").json()
    assert stored["name"] == "אגנסי חדשה"
    assert stored["rebate_percent"] == 3.5


def test_running_the_same_order_twice_creates_no_duplicate(client):
    """Zero duplicates, enforced: the second run reuses and then refuses."""
    _agency(client)
    first = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    assert first.status_code == 201

    second = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    assert second.status_code == 409
    refusal = second.json()["detail"]
    assert ORDER["campaign_name"] in refusal["message_en"]
    assert ORDER["campaign_name"] in refusal["message_he"]

    assert len(client.get("/api/agencies").json()["agencies"]) == 1
    assert len(client.get("/api/clients/campaigns").json()["campaigns"]) == 1
    links = client.get("/api/agencies/AGY_01/advertisers").json()
    assert links["manual"].count(ORDER["advertiser"]) == 1


def test_a_second_campaign_for_the_same_client_is_not_a_duplicate(client):
    """The refusal is on name plus client, so a real second campaign lands."""
    _agency(client)
    client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    again = client.post("/api/clients/onboarding", json={
        "agency": {"agency_id": "AGY_01"},
        **{**ORDER, "campaign_name": "חורף 2026"},
    })
    assert again.status_code == 201
    assert again.json()["advertiser"]["outcome"] == "already_linked"
    assert len(client.get("/api/clients/campaigns").json()["campaigns"]) == 2


def test_the_weekday_discount_prices_only_when_it_is_asked_to(client):
    """A campaign term prices nothing until it is written where it prices."""
    _agency(client)
    stored = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER}).json()
    assert stored["discount"]["outcome"] == "stored_on_the_campaign"
    assert stored["discount"]["note_en"] and stored["discount"]["note_he"]
    assert client.get("/api/agencies/AGY_01/conditions").json()["conditions"] == []

    applied = client.post("/api/clients/onboarding", json={
        "agency": {"agency_id": "AGY_01"},
        **{**ORDER, "campaign_name": "אביב 2026", "apply_surcharge_as_agency_rule": True},
    }).json()
    assert applied["discount"]["outcome"] == "priced_as_agency_condition"
    conditions = client.get("/api/agencies/AGY_01/conditions").json()["conditions"]
    assert len(conditions) == 1
    assert conditions[0]["scope_weekdays"] == "6"
    assert conditions[0]["mode"] == "premium_discount"
    assert conditions[0]["value"] == 20.0
    assert "every campaign" in applied["discount"]["covers_en"]
    assert applied["discount"]["covers_he"]


def test_amending_a_campaign_to_no_weekday_is_refused_the_same_as_booking_it(client):
    """The blind critic's exact repro: amend, not onboard, and no agency rule.

    ``PUT /api/clients/campaigns/{id}`` never writes an agency condition, so an
    empty scope here can only mean the campaign term would carry a percent with
    no day it covers, never the onboarding flow's ANY-widening. The refusal
    fires all the same, and the campaign is left holding its original scope.
    """
    _agency(client)
    created = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER}).json()
    campaign_id = created["campaign"]["campaign_id"]

    refused = client.put(f"/api/clients/campaigns/{campaign_id}", json={
        "surcharge_discount_percent": 12.0, "surcharge_weekdays": "",
    })
    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]
    assert "no day it covers" in detail["message_en"]
    assert "every day" not in detail["message_en"], "the amend path never becomes an agency condition"

    stored = client.get("/api/clients/campaigns").json()["campaigns"][0]
    assert stored["surcharge_weekdays"] == "6", "the refused write left the campaign row untouched"


def test_amending_only_the_weekdays_to_empty_is_refused_when_a_percent_is_on_file(client):
    """A partial PUT that clears the days must see the percent already on file."""
    _agency(client)
    created = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER}).json()
    campaign_id = created["campaign"]["campaign_id"]

    refused = client.put(f"/api/clients/campaigns/{campaign_id}", json={"surcharge_weekdays": ""})
    assert refused.status_code == 400, refused.text


def test_a_new_campaign_with_a_percent_and_no_weekday_is_refused_at_creation(client):
    """The plain create endpoint, not just onboarding, enforces the same rule."""
    _agency(client)
    refused = client.post("/api/clients/campaigns", json={
        "name": "ללא יום", "advertiser": "בנק הפועלים", "agency_id": "AGY_01",
        "starts_on": "2026-09-01", "ends_on": "2026-09-30",
        "surcharge_discount_percent": 12.0, "surcharge_weekdays": "",
    })
    assert refused.status_code == 400
    assert "no day it covers" in refused.json()["detail"]["message_en"]
    assert client.get("/api/clients/campaigns").json()["campaigns"] == []


def test_a_campaign_is_ended_and_never_deleted(client):
    """Deactivate beats delete, exactly as it does for an agency."""
    _agency(client)
    created = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER}).json()
    campaign_id = created["campaign"]["campaign_id"]
    ended = client.post(f"/api/clients/campaigns/{campaign_id}/deactivate")
    assert ended.status_code == 200
    assert ended.json()["status"] == "ended"
    campaigns = client.get("/api/clients/campaigns").json()["campaigns"]
    assert len(campaigns) == 1
    assert campaigns[0]["status"] == "ended"
    assert len(campaigns[0]["flights"]) == 2


def test_bad_input_is_refused_at_the_door_with_a_reason(client):
    """Nothing is silently coerced: a wrong date, unit or percent is a 400."""
    _agency(client)
    base = {"agency": {"agency_id": "AGY_01"}, **ORDER}

    bad_date = client.post("/api/clients/onboarding", json={**base, "campaign_starts_on": "01/09/2026"})
    assert bad_date.status_code == 400
    assert "ISO date" in bad_date.json()["detail"]["message_en"]

    backwards = client.post("/api/clients/onboarding", json={
        **base, "campaign_starts_on": "2026-09-30", "campaign_ends_on": "2026-09-01",
    })
    assert backwards.status_code == 400

    bad_percent = client.post("/api/clients/onboarding", json={**base, "rebate_percent": 140})
    assert bad_percent.status_code == 400

    bad_unit = client.post("/api/clients/onboarding", json={
        **base,
        "flights": [{"starts_on": "2026-09-01", "ends_on": "2026-09-02", "goal_kind": "bananas", "goal_value": 1}],
    })
    assert bad_unit.status_code == 400


def test_a_campaign_cannot_be_booked_through_an_agency_that_does_not_exist(client):
    """No orphan: an unknown agency id is refused rather than stored."""
    response = client.post("/api/clients/campaigns", json={
        "name": "קמפיין יתום",
        "advertiser": "מגדל",
        "agency_id": "AGY_99",
        "starts_on": "2026-09-01",
        "ends_on": "2026-09-30",
    })
    assert response.status_code == 400
    assert "AGY_99" in response.json()["detail"]["message_en"]
    assert "AGY_99" in response.json()["detail"]["message_he"]


def test_a_flight_carries_a_booked_goal_and_no_delivered_figure(client):
    """The store has no delivery column, so no surface can invent one."""
    from kairos_api import campaigns_api_store as store

    _agency(client)
    created = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER}).json()
    assert "delivered" not in " ".join(store.COLUMNS)
    flight = created["flights"][0]
    assert flight["goal_kind"] == "spots"
    assert flight["goal_value"] == 40.0
    assert "delivered" not in flight


def test_every_written_row_says_when_it_was_written(client):
    """A commercial record that cannot say when it was made is not a record.

    ``blank_row`` seeds every column, so the store's original ``setdefault``
    found ``created_at`` already present and stamped nothing. Every campaign and
    flight written before that fix reached disk with an empty stamp.
    """
    _agency(client)
    created = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER}).json()
    assert created["campaign"]["created_at"], created["campaign"]
    assert created["campaign"]["created_at"].startswith("20")
    for flight in created["flights"]:
        assert flight["created_at"], flight

    stored = client.get("/api/clients/campaigns").json()["campaigns"][0]
    assert stored["created_at"] == created["campaign"]["created_at"]


def test_the_store_is_one_file_with_two_declared_record_kinds(client, tmp_path):
    """One lock, one atomic write and one restore point for a campaign and its flights."""
    _agency(client)
    client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    frame = pd.read_csv(tmp_path / "campaigns.csv", encoding="utf-8-sig", dtype=str, keep_default_na=False)
    assert list(frame.columns) == __import__("kairos_api.campaigns_api_store", fromlist=["x"]).COLUMNS
    assert sorted(frame["record_type"].tolist()) == ["campaign", "flight", "flight"]


def test_a_flight_can_be_added_and_removed_without_touching_the_campaign(client):
    """A flight is a line of the campaign, so removing one leaves the campaign."""
    _agency(client)
    created = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER}).json()
    campaign_id = created["campaign"]["campaign_id"]
    added = client.post(f"/api/clients/campaigns/{campaign_id}/flights", json={
        "starts_on": "2026-10-01", "ends_on": "2026-10-10", "goal_kind": "seconds", "goal_value": 900,
    })
    assert added.status_code == 201
    flight_id = added.json()["flight_id"]
    assert len(client.get("/api/clients/campaigns").json()["campaigns"][0]["flights"]) == 3

    removed = client.delete(f"/api/clients/campaigns/{campaign_id}/flights/{flight_id}")
    assert removed.status_code == 200
    campaigns = client.get("/api/clients/campaigns").json()["campaigns"]
    assert len(campaigns) == 1
    assert len(campaigns[0]["flights"]) == 2


def test_the_form_offers_real_choices_and_never_asks_for_an_invented_id(client):
    """Every choice the flow offers comes from a store, including the next ids."""
    _agency(client)
    options = client.get("/api/clients/onboarding/options").json()
    assert [entry["agency_id"] for entry in options["agencies"]] == ["AGY_01"]
    assert options["next_agency_id"] == "AGY_02"
    assert options["next_campaign_id"] == "CMP_001"
    assert options["goal_kinds"][0] == "spots"
    assert [day["key"] for day in options["weekdays"]][0] == "7"
    assert [day["key"] for day in options["weekdays"]][-1] == "6"
    assert len(options["advertisers"]) == 41


# --------------------------------------------------------------------------
# Every refusal, in the language of the person it is addressed to
# --------------------------------------------------------------------------

HEBREW = set("אבגדהוזחטיכלמנסעפצקרשתךםןףץ")


def _refusals(client) -> list[tuple[str, int, dict]]:
    """One request per refusal this destination can produce, with its answer.

    Pydantic's own 422 for a missing body field is not in this list, because it
    is FastAPI's message and not this piece's to write. Everything below is a
    refusal one of P4's three write modules raises itself.
    """
    _agency(client)
    booked = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    assert booked.status_code == 201, booked.text
    campaign_id = booked.json()["campaign"]["campaign_id"]
    base = {"agency": {"agency_id": "AGY_01"}, **ORDER}
    attempts = [
        ("duplicate campaign", client.post("/api/clients/onboarding", json=base)),
        ("campaign with no name", client.post("/api/clients/campaigns", json={"name": " ", "advertiser": "מגדל"})),
        ("campaign with no client", client.post("/api/clients/campaigns", json={"name": "קמפיין", "advertiser": " "})),
        ("agency that does not exist", client.post("/api/clients/campaigns", json={
            "name": "קמפיין יתום", "advertiser": "מגדל", "agency_id": "AGY_99",
            "starts_on": "2026-09-01", "ends_on": "2026-09-30",
        })),
        ("campaign id already taken", client.post("/api/clients/campaigns", json={
            "name": "שם אחר", "advertiser": "מגדל", "campaign_id": campaign_id,
            "starts_on": "2026-09-01", "ends_on": "2026-09-30",
        })),
        ("date that is not ISO", client.post("/api/clients/onboarding", json={**base, "campaign_starts_on": "01/09/2026"})),
        ("window that runs backwards", client.post("/api/clients/onboarding", json={
            **base, "campaign_starts_on": "2026-09-30", "campaign_ends_on": "2026-09-01",
        })),
        ("percent outside the range", client.post("/api/clients/onboarding", json={**base, "rebate_percent": 140})),
        ("weekday scope that is not ISO", client.post("/api/clients/onboarding", json={**base, "surcharge_weekdays": "9"})),
        ("goal unit outside the vocabulary", client.post(f"/api/clients/campaigns/{campaign_id}/flights", json={
            "starts_on": "2026-09-01", "ends_on": "2026-09-02", "goal_kind": "bananas", "goal_value": 1,
        })),
        ("goal below zero", client.post(f"/api/clients/campaigns/{campaign_id}/flights", json={
            "starts_on": "2026-09-01", "ends_on": "2026-09-02", "goal_kind": "spots", "goal_value": -5,
        })),
        ("status outside the vocabulary", client.put(f"/api/clients/campaigns/{campaign_id}", json={"status": "paused"})),
        ("campaign that is not there", client.put("/api/clients/campaigns/CMP_404", json={"notes": "x"})),
        ("flight that is not there", client.delete(f"/api/clients/campaigns/{campaign_id}/flights/NOPE")),
        ("agency with no name", client.post("/api/clients/onboarding", json={
            **ORDER, "agency": {"name": " "}, "campaign_name": "קמפיין אחר",
        })),
    ]
    return [(label, response.status_code, response.json()) for label, response in attempts]


def test_every_write_refusal_arrives_in_both_languages(client):
    """The Hebrew flow refuses in Hebrew, which is the language it asked in.

    Measured rather than asserted once: fifteen distinct refusals, each one
    carrying an English half and a Hebrew half, with the Hebrew half really in
    Hebrew letters and the English half really not. A refusal a person cannot
    read is a dead end with a status code on it.
    """
    seen = _refusals(client)
    assert len(seen) == 15
    for label, status, body in seen:
        assert 400 <= status < 500, f"{label} was not refused"
        detail = body["detail"]
        assert isinstance(detail, dict), f"{label} refused with a bare string"
        english = detail["message_en"]
        hebrew = detail["message_he"]
        assert english and hebrew, label
        assert english != hebrew, f"{label} answers Hebrew with the English sentence"
        assert set(hebrew) & HEBREW, f"{label} has no Hebrew in its Hebrew half"
        # A quoted value is data the operator typed, and a Hebrew client name
        # belongs in both halves. Outside the quotes the English half is prose,
        # and prose in the wrong language is the defect this measures.
        prose = re.sub(r"'[^']*'", "", english)
        assert not set(prose) & HEBREW, f"{label} has Hebrew prose in its English half"


def test_a_refusal_names_the_field_as_a_person_names_it(client):
    """starts_on is a column. The refusal says the start date, in both languages."""
    _agency(client)
    bad = client.post("/api/clients/onboarding", json={
        "agency": {"agency_id": "AGY_01"}, **ORDER, "campaign_starts_on": "01/09/2026",
    })
    detail = bad.json()["detail"]
    assert "The start date" in detail["message_en"]
    assert "תאריך ההתחלה" in detail["message_he"]
    assert "starts_on" not in detail["message_en"]
    assert "starts_on" not in detail["message_he"]
