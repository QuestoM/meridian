"""P4: a client the product creates is a named client, in the same act.

Two stores describe a client. ``data/agency_advertisers.csv`` says which agency
it buys through, and ``data/advertiser_names.csv`` says who it is. Only the
first of the two had a creation path, so the flagship flow made clients the
name space never learned about.

Measured on the working tree that carried the defect: the link store held 45
advertisers and the name space held 41, and the four in the difference were
exactly the clients the onboarding flow had created. The identity suite asserts
the two are equal, so the tree shipped red, and on screen the created client's
own record read "source unknown" for ever, because nothing resolved it.

Every store here is redirected into a temporary directory, including the name
space, which follows the link store's own directory rather than a second
constant. The suite therefore never writes a name into the operator's real
data, which is the defect section 5.6 of the specification records.

Nothing here passes vacuously: the last two tests put the defect back, once by
neutralising the registration inside the flow and once inside the standalone
link route, and assert the divergence returns.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "tv-break-dashboard" / "src" / "clients" / "clients-money-helpers.js"

NEW_CLIENT = "לקוח חדש לגמרי"
OTHER_CLIENT = "לקוח שני לגמרי"

ORDER = {
    "advertiser": NEW_CLIENT,
    "campaign_name": "השקה 2026",
    "campaign_starts_on": "2026-09-01",
    "campaign_ends_on": "2026-09-30",
    "flights": [
        {"starts_on": "2026-09-01", "ends_on": "2026-09-15", "goal_kind": "spots", "goal_value": 40},
    ],
}


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Every P4 store in one temporary directory, name space included."""
    from kairos_api import agencies, agency_conditions, campaigns_api, campaigns_api_store, version_store

    monkeypatch.setattr(agencies, "AGENCIES_PATH", tmp_path / "agencies.csv")
    monkeypatch.setattr(agencies, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agency_conditions, "LINKS_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(agency_conditions, "CONDITIONS_PATH", tmp_path / "agency_conditions.csv")
    monkeypatch.setattr(agency_conditions, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(campaigns_api_store, "CAMPAIGNS_PATH", tmp_path / "campaigns.csv")
    monkeypatch.setattr(campaigns_api_store, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(version_store, "snapshot_manual_edit", lambda request, logical: None)
    # No daily file resolves inside the temp tree, so every link a test sees is
    # one a test made and no observed link arrives from outside it.
    monkeypatch.setattr(agency_conditions, "_latest_daily_pairs", lambda: ([], None))

    app = FastAPI()
    app.include_router(agencies.router)
    app.include_router(agency_conditions.router)
    app.include_router(campaigns_api.router)
    return tmp_path, TestClient(app)


def _rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _linked(root: Path) -> set[str]:
    from kairos.optimize.advertiser_rules_identity import normalize_name

    return {
        normalize_name(row.get("advertiser", ""))
        for row in _rows(root / "agency_advertisers.csv")
        if str(row.get("advertiser", "")).strip()
    }


def _named(root: Path) -> set[str]:
    from kairos.optimize.advertiser_rules_identity import normalize_name

    return {
        normalize_name(row.get("name", ""))
        for row in _rows(root / "advertiser_names.csv")
        if str(row.get("name", "")).strip()
    }


def _agency(client, agency_id="AGY_01", name="OMD"):
    response = client.post("/api/agencies", json={
        "agency_id": agency_id, "name": name, "rebate_percent": 4.0,
    })
    assert response.status_code == 201, response.text
    return response.json()


# --- the bar ------------------------------------------------------------------


def test_onboarding_keeps_the_link_set_and_the_name_set_equal(store):
    """The invariant the identity suite asserts, held across a real creation."""
    root, client = store
    _agency(client)
    assert _linked(root) == _named(root) == set()

    created = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    assert created.status_code == 201, created.text

    assert _linked(root) == _named(root)
    assert len(_named(root)) == 1


def test_the_created_client_is_named_manual_with_the_date_it_was_created(store):
    """Named, and honestly named: created, not observed, and dated."""
    root, client = store
    _agency(client)
    client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})

    rows = _rows(root / "advertiser_names.csv")
    assert [row["name"] for row in rows] == [NEW_CLIENT]
    assert rows[0]["source"] == "manual"
    assert rows[0]["first_seen"]
    # Nothing is invented beside the name the operator typed.
    assert rows[0]["display_name"] == ""
    assert rows[0]["aliases"] == ""


def test_the_response_says_the_client_was_named(store):
    """The flow reports it, so the confirmation can state it rather than imply it."""
    root, client = store
    _agency(client)
    payload = client.post(
        "/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER},
    ).json()

    assert payload["advertiser"]["identity"]["outcome"] == "registered"
    assert payload["advertiser"]["identity"]["source"] == "manual"
    assert payload["advertiser"]["identity"]["first_seen"]
    assert payload["advertiser"]["identity"]["reason_en"]
    assert payload["advertiser"]["identity"]["reason_he"]


def test_the_standalone_link_route_names_the_client_too(store):
    """The chokepoint is the link, not the flow, so both writers name a client."""
    root, client = store
    _agency(client)

    response = client.post("/api/agencies/AGY_01/advertisers", json={"advertiser": OTHER_CLIENT})
    assert response.status_code == 201, response.text
    assert response.json()["identity"]["outcome"] == "registered"
    assert _linked(root) == _named(root)


def test_booking_the_same_client_twice_writes_one_name(store):
    """Zero duplicates covers the name space too, and the second act says so."""
    root, client = store
    _agency(client)
    client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    again = client.post("/api/clients/onboarding", json={
        "agency": {"agency_id": "AGY_01"}, **{**ORDER, "campaign_name": "השקה 2027"},
    })

    assert again.status_code == 201, again.text
    assert again.json()["advertiser"]["identity"]["outcome"] == "already_named"
    assert [row["name"] for row in _rows(root / "advertiser_names.csv")] == [NEW_CLIENT]
    assert _linked(root) == _named(root)


def test_the_created_client_resolves_on_the_read_its_own_record_uses(store, monkeypatch):
    """Consequence two: the record reads its real source instead of unknown."""
    from kairos_api import advertisers_identity

    root, client = store
    _agency(client)
    client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    monkeypatch.setattr(advertisers_identity, "_names_path", lambda: root / "advertiser_names.csv")

    resolved = advertisers_identity.resolve_one(NEW_CLIENT)
    assert resolved["resolved"] is True
    assert resolved["source"] == "manual"
    assert resolved["shown_name"] == NEW_CLIENT
    # And the surface has a word for that source, so no record prints "unknown".
    assert "'manual'" in HELPERS.read_text(encoding="utf-8")
    assert "הוזן ידנית" in HELPERS.read_text(encoding="utf-8")


# --- the defect, put back ------------------------------------------------------


def test_without_the_registration_the_flow_diverges_again(store, monkeypatch):
    """Neutralise the naming inside the flow and the two sets separate."""
    from kairos_api import agency_conditions

    root, client = store
    _agency(client)
    monkeypatch.setattr(agency_conditions, "register_advertiser_name", lambda *a, **k: {})

    client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_01"}, **ORDER})
    assert _linked(root) == {_normalized(NEW_CLIENT)}
    assert _named(root) == set()
    assert _linked(root) != _named(root)


def test_without_the_registration_the_link_route_diverges_again(store, monkeypatch):
    """The same mutation on the standalone route, so neither pass is vacuous."""
    from kairos_api import agency_conditions

    root, client = store
    _agency(client)
    monkeypatch.setattr(agency_conditions, "register_advertiser_name", lambda *a, **k: {})

    client.post("/api/agencies/AGY_01/advertisers", json={"advertiser": OTHER_CLIENT})
    assert _linked(root) == {_normalized(OTHER_CLIENT)}
    assert _named(root) == set()


def _normalized(name: str) -> str:
    from kairos.optimize.advertiser_rules_identity import normalize_name

    return normalize_name(name)
