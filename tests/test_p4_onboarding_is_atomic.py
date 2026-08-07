"""P4: a refused order writes nothing at all.

The measured defect. ``POST /api/clients/onboarding`` created the agency, then
the advertiser link, then the campaign, and validated the campaign's own fields
inside that third step. So an end date typed ``29/08/2026`` was refused with the
ISO sentence while the stores had already moved by one agency, one link and one
name row. An agency can only be suspended, never deleted, so one mistyped
character left a record the product cannot remove, on the flow whose whole
promise is that the account manager never has to leave and come back.

A second path had the same shape and was never named: a client already holding a
manual link to another agency is refused by the link store, which runs second,
so the agency created first stayed. That refusal also arrived as a bare English
string on a Hebrew flow.

Both are measured here as row counts across the four stores the flow writes,
before and after every refusal this flow can raise. The last test removes the
check and asserts the rows come back, so a pass here can never be vacuous.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ORDER = {
    "advertiser": "טורנדו מוצרי צריכה",
    "campaign_name": "קיץ 2026",
    "campaign_starts_on": "2026-08-02",
    "campaign_ends_on": "2026-08-29",
    "rebate_percent": 4.0,
    "surcharge_discount_percent": 20.0,
    "surcharge_weekdays": "6",
    "flights": [
        {"starts_on": "2026-08-02", "ends_on": "2026-08-15", "goal_kind": "spots", "goal_value": 40},
    ],
}
NEW_AGENCY = {"name": "סוכנות חדשה", "rebate_percent": 4.0}


@pytest.fixture
def stores(tmp_path, monkeypatch) -> Path:
    """Every store this flow writes, redirected into a temporary directory.

    The name space is found beside the link store by the code under test, so
    redirecting the link store redirects it too and the operator's own
    ``data/advertiser_names.csv`` is never written by this suite.
    """
    from kairos_api import agencies, agency_conditions, campaigns_api_store, version_store

    monkeypatch.setattr(agencies, "AGENCIES_PATH", tmp_path / "agencies.csv")
    monkeypatch.setattr(agencies, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agency_conditions, "LINKS_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(agency_conditions, "CONDITIONS_PATH", tmp_path / "agency_conditions.csv")
    monkeypatch.setattr(agency_conditions, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(campaigns_api_store, "CAMPAIGNS_PATH", tmp_path / "campaigns.csv")
    monkeypatch.setattr(campaigns_api_store, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(version_store, "snapshot_manual_edit", lambda request, logical: None)
    monkeypatch.setattr(agency_conditions, "_latest_daily_pairs", lambda: ([], None))
    return tmp_path


@pytest.fixture
def client(stores) -> TestClient:
    from kairos_api import agencies, agency_conditions, campaigns_api

    app = FastAPI()
    app.include_router(agencies.router)
    app.include_router(agency_conditions.router)
    app.include_router(campaigns_api.router)
    return TestClient(app)


def rows(store: Path) -> dict[str, int]:
    """How many records each of the four stores holds, header excluded."""
    def count(name: str) -> int:
        path = store / name
        if not path.exists():
            return 0
        return max(0, len(path.read_text(encoding="utf-8-sig").strip().splitlines()) - 1)
    return {
        "agencies": count("agencies.csv"),
        "links": count("agency_advertisers.csv"),
        "names": count("advertiser_names.csv"),
        "campaigns": count("campaigns.csv"),
        "conditions": count("agency_conditions.csv"),
    }


# Every refusal this flow can raise, as the request that raises it. Each one was
# reachable only after at least one store had been written.
REFUSALS = [
    ("end date that is not ISO", {"campaign_ends_on": "29/08/2026"}),
    ("start date that is not ISO", {"campaign_starts_on": "01/09/2026"}),
    ("window that runs backwards", {"campaign_starts_on": "2026-09-30", "campaign_ends_on": "2026-09-01"}),
    ("rebate outside the range", {"rebate_percent": 140}),
    ("weekday scope that is not ISO", {"surcharge_weekdays": "9"}),
    ("weekday scope left empty though a discount is set", {"surcharge_weekdays": ""}),
    ("goal unit outside the vocabulary", {"flights": [
        {"starts_on": "2026-08-02", "ends_on": "2026-08-15", "goal_kind": "bananas", "goal_value": 40},
    ]}),
    ("goal below zero", {"flights": [
        {"starts_on": "2026-08-02", "ends_on": "2026-08-15", "goal_kind": "spots", "goal_value": -5},
    ]}),
    ("flight window that runs backwards", {"flights": [
        {"starts_on": "2026-08-15", "ends_on": "2026-08-02", "goal_kind": "spots", "goal_value": 40},
    ]}),
    ("campaign with no name", {"campaign_name": " "}),
    ("campaign with no client", {"advertiser": " "}),
]


@pytest.mark.parametrize("label,change", REFUSALS, ids=[label for label, _ in REFUSALS])
def test_a_refused_order_leaves_every_store_exactly_as_it_was(client, stores, label, change):
    """The bar: refused means nothing was written, on every one of the four."""
    before = rows(stores)
    refused = client.post("/api/clients/onboarding", json={"agency": NEW_AGENCY, **ORDER, **change})
    assert 400 <= refused.status_code < 500, f"{label} was not refused"
    detail = refused.json()["detail"]
    assert detail["message_en"] and detail["message_he"], f"{label} refused in one language"
    assert rows(stores) == before, f"{label} left rows behind: {rows(stores)} against {before}"


def test_the_end_date_that_was_measured_keeps_its_sentence(client, stores):
    """The exact request from the finding, with the sentence it already said."""
    refused = client.post("/api/clients/onboarding", json={
        "agency": NEW_AGENCY, **ORDER, "campaign_ends_on": "29/08/2026",
    })
    assert refused.status_code == 400
    detail = refused.json()["detail"]
    assert detail["message_en"] == "The end date must be an ISO date, YYYY-MM-DD"
    assert detail["message_he"] == "יש להזין את תאריך הסיום כתאריך ISO, YYYY-MM-DD"
    assert rows(stores) == {"agencies": 0, "links": 0, "names": 0, "campaigns": 0, "conditions": 0}


def test_a_client_linked_elsewhere_is_refused_before_an_agency_is_created(client, stores):
    """The second path, and its refusal now reaches a Hebrew reader in Hebrew."""
    assert client.post("/api/agencies", json={"agency_id": "AGY_09", "name": "סוכנות אחרת"}).status_code == 201
    assert client.post("/api/agencies/AGY_09/advertisers", json={"advertiser": ORDER["advertiser"]}).status_code in {200, 201}
    before = rows(stores)
    refused = client.post("/api/clients/onboarding", json={"agency": NEW_AGENCY, **ORDER})
    assert refused.status_code == 409
    detail = refused.json()["detail"]
    assert "AGY_09" in detail["message_en"] and "AGY_09" in detail["message_he"]
    assert set(detail["message_he"]) & set("אבגדהוזחטיכלמנסעפצקרשת"), "the Hebrew half has no Hebrew in it"
    assert rows(stores) == before, "the agency the order named was created before the link refused it"


def test_a_duplicate_campaign_is_refused_without_touching_anything(client, stores):
    """Running the same signed order twice stays one agency, one link, one campaign."""
    first = client.post("/api/clients/onboarding", json={"agency": NEW_AGENCY, **ORDER})
    assert first.status_code == 201, first.text
    after_first = rows(stores)
    assert after_first["agencies"] == 1
    assert after_first["links"] == 1
    assert after_first["names"] == 1
    assert after_first["campaigns"] == 2, "one campaign row and one flight row"
    repeat = client.post("/api/clients/onboarding", json={"agency": NEW_AGENCY, **ORDER})
    assert repeat.status_code == 409
    assert rows(stores) == after_first


def test_an_order_that_is_correct_still_creates_all_of_it(client, stores):
    """The check refuses orders, never delays them: the flow still does its job."""
    created = client.post("/api/clients/onboarding", json={
        "agency": NEW_AGENCY, **ORDER, "apply_surcharge_as_agency_rule": True,
    })
    assert created.status_code == 201, created.text
    body = created.json()
    assert body["created"] == {"agency": True, "advertiser_link": True, "campaign": True, "flights": 1}
    assert body["discount"]["outcome"] == "priced_as_agency_condition"
    assert rows(stores) == {"agencies": 1, "links": 1, "names": 1, "campaigns": 2, "conditions": 1}


def test_without_the_check_the_same_order_leaves_the_rows_behind(client, stores, monkeypatch):
    """Proof the tests above bite. Remove the check and the defect comes back."""
    from kairos_api import campaigns_api_onboarding

    monkeypatch.setattr(campaigns_api_onboarding, "precheck", lambda payload: None)
    refused = client.post("/api/clients/onboarding", json={
        "agency": NEW_AGENCY, **ORDER, "campaign_ends_on": "29/08/2026",
    })
    assert refused.status_code == 400
    assert rows(stores) == {"agencies": 1, "links": 1, "names": 1, "campaigns": 0, "conditions": 0}
