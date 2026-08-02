"""P4: the Clients destination tells the truth before the click.

The specification's section 4.5 states the rule as one sentence: a refusal is
legible before the click, never a 403 after it. This destination carries four
write controls (onboard a client, book a campaign, end a campaign, add a
flight), and before this file every one of them rendered enabled to a read-only
account and answered 403 once it was pressed.

So these tests measure two things that have to agree. The read payload carries
``can_edit`` and, when it is false, the same string the refusal would use; and
the write actually refuses that account. A payload that said false while the
write let the account through, or the reverse, would be worse than either.

Affiliation is deliberately not a gate here. Booking a client is a run-side
commercial act, so a channel-affiliated operator may do it and a company viewer
may not, which is the section 4.5 sentence applied without an exception:
affiliation decides which side of the line you see, role decides what you change
on your side.

Annotations are postponed for the load-bearing reason
``tests/test_w0_cleanup_wall_reads.py`` records: 79 of the 80 modules in
``kairos_api`` postpone theirs, so a route's parameters reach FastAPI as strings,
and a test module that did not postpone them would pass while a real adopter
broke.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import auth, auth_store

ADMIN_PASSWORD = "rootpass-1234"
VIEWER_PASSWORD = "viewerpass-123"
CHANNEL_PASSWORD = "channelpass-123"


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    yield tmp_path
    auth_store.reset_runtime_state()


@pytest.fixture()
def app(auth_env, monkeypatch):
    """P4's own routers plus auth, with every store inside the temp tree."""
    from kairos_api import agencies, agency_conditions, campaigns_api, campaigns_api_store, version_store

    monkeypatch.setattr(agencies, "AGENCIES_PATH", auth_env / "agencies.csv")
    monkeypatch.setattr(agencies, "BACKUP_DIR", auth_env / "_backups")
    monkeypatch.setattr(agency_conditions, "LINKS_PATH", auth_env / "agency_advertisers.csv")
    monkeypatch.setattr(agency_conditions, "CONDITIONS_PATH", auth_env / "agency_conditions.csv")
    monkeypatch.setattr(agency_conditions, "BACKUP_DIR", auth_env / "_backups")
    monkeypatch.setattr(campaigns_api_store, "CAMPAIGNS_PATH", auth_env / "campaigns.csv")
    monkeypatch.setattr(campaigns_api_store, "BACKUP_DIR", auth_env / "_backups")
    monkeypatch.setattr(version_store, "snapshot_manual_edit", lambda request, logical: None)

    application = FastAPI()
    application.include_router(auth.router)
    application.include_router(campaigns_api.router)

    @application.middleware("http")
    async def enforce(request, call_next):
        denial = auth.enforce_request(request)
        if denial is not None:
            return denial
        return await call_next(request)

    return application


def _clients(app: FastAPI) -> dict[str, TestClient]:
    """An admin, a company viewer and a channel operator, each signed in."""
    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
    admin = TestClient(app)
    assert admin.post("/api/auth/login", json={
        "username": "admin", "password": ADMIN_PASSWORD,
    }).status_code == 200
    for username, password, role, affiliation in (
        ("view1", VIEWER_PASSWORD, "viewer", "company"),
        ("chan1", CHANNEL_PASSWORD, "operator", "channel"),
    ):
        created = admin.post("/api/auth/users", json={
            "username": username, "password": password, "role": role,
            "display_name": username, "must_change_password": False,
            "affiliation": affiliation,
        })
        assert created.status_code == 201, created.text
    signed = {"admin": admin}
    for username, password in (("view1", VIEWER_PASSWORD), ("chan1", CHANNEL_PASSWORD)):
        client = TestClient(app)
        assert client.post("/api/auth/login", json={
            "username": username, "password": password,
        }).status_code == 200
        signed[username] = client
    return signed


ORDER = {
    "agency": {"name": "סוכנות הרשאות", "rebate_percent": 3.0},
    "advertiser": "בנק הפועלים",
    "campaign_name": "בדיקת הרשאות",
    "campaign_starts_on": "2026-09-01",
    "campaign_ends_on": "2026-09-30",
    "flights": [],
}


def test_a_viewer_is_told_it_cannot_book_before_it_tries(app):
    """The read says no, in the words the refusal itself would use."""
    from kairos_api.campaigns_api import CLIENTS_WALL

    signed = _clients(app)
    payload = signed["view1"].get("/api/clients/campaigns").json()
    assert payload["can_edit"] is False
    assert payload["can_edit_reason"] == CLIENTS_WALL.role_detail

    options = signed["view1"].get("/api/clients/onboarding/options").json()
    assert options["can_edit"] is False
    assert options["can_edit_reason"] == CLIENTS_WALL.role_detail


def test_the_write_refuses_the_same_viewer_the_read_warned(app):
    """What the payload promised and what the server does cannot drift."""
    signed = _clients(app)
    refused = signed["view1"].post("/api/clients/onboarding", json=ORDER)
    assert refused.status_code == 403
    assert signed["view1"].get("/api/clients/campaigns").json()["count"] == 0


def test_an_operator_may_book_whichever_side_of_the_line_it_sits_on(app):
    """Affiliation is not a gate on a commercial act. Role is."""
    signed = _clients(app)
    for username in ("admin", "chan1"):
        assert signed[username].get("/api/clients/campaigns").json()["can_edit"] is True

    created = signed["chan1"].post("/api/clients/onboarding", json={
        **ORDER, "campaign_name": f"בדיקה {'chan1'}",
    })
    assert created.status_code == 201, created.text
    assert signed["chan1"].get("/api/clients/campaigns").json()["count"] == 1


def test_the_status_and_goal_words_travel_with_the_payload(app):
    """A closed value set arrives as words, so no surface holds a second copy."""
    signed = _clients(app)
    payload = signed["admin"].get("/api/clients/campaigns").json()

    statuses = {entry["value"]: entry for entry in payload["status_vocabulary"]}
    assert set(statuses) == set(payload["statuses"])
    for entry in statuses.values():
        assert entry["label_en"] and entry["label_he"]
        assert entry["what_to_do_en"] and entry["what_to_do_he"]

    goals = {entry["value"]: entry for entry in payload["goal_kind_vocabulary"]}
    assert set(goals) == set(payload["goal_kinds"])
    for entry in goals.values():
        assert entry["label_en"] and entry["label_he"]
