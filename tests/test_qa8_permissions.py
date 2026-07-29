"""Company/channel affiliation permissions for the event-management surface.

Event management is company-staff only: accounts carry an ``affiliation`` of
company or channel (missing reads company, so every legacy account keeps full
access). A channel-affiliated session reads everything but gets 403 on event
writes and on the event pricing activation switch. These tests run the FULL
server app (middleware included), with the auth store, the events store, the
version store and the settings file all relocated to tmp so nothing under
data/ is ever touched.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import kairos_api.core as core
import kairos_api.events_api as events_api
import kairos_api.version_store as vs
from kairos_api import auth_store
from kairos_api.server import app

ROOT = Path(__file__).resolve().parents[1]

ADMIN_PASSWORD = "rootpass-1234"
COMPANY_PASSWORD = "companypass-123"
CHANNEL_PASSWORD = "channelpass-123"

EVENTS_403_DETAIL = "עריכת אירועים שמורה לצוות החברה"
PRICING_403_DETAIL = "הפעלת תמחור אירועים שמורה לצוות החברה"

EVENT_PAYLOAD = {"name": "גמר גביע המדינה", "type": "sport", "start_date": "2024-11-20",
                 "end_date": "2024-11-20", "intensity": 4, "notes": "", "active": True}


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Relocate every store the permission surface touches into tmp."""
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    monkeypatch.setattr(events_api, "EVENTS_PATH", tmp_path / "calendar_events.csv")
    settings_copy = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", settings_copy)
    monkeypatch.setattr(core, "SETTINGS_PATH", settings_copy)
    auth_store.reset_runtime_state()
    yield tmp_path
    auth_store.reset_runtime_state()


def signed_in(username: str, password: str) -> TestClient:
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return client


def seeded_admin() -> TestClient:
    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
    return signed_in("admin", ADMIN_PASSWORD)


def create_operator(admin: TestClient, username: str, password: str, affiliation: str) -> None:
    response = admin.post("/api/auth/users", json={
        "username": username, "password": password, "role": "operator",
        "display_name": username, "must_change_password": False,
        "affiliation": affiliation,
    })
    assert response.status_code == 201, response.text
    assert response.json()["affiliation"] == affiliation


# ---------------------------------------------------------------------------
# Store semantics: persistence, the legacy default, the helper
# ---------------------------------------------------------------------------

def test_affiliation_persists_and_legacy_records_read_company(env):
    admin = seeded_admin()
    create_operator(admin, "chan1", CHANNEL_PASSWORD, "channel")

    assert auth_store.get_user("chan1")["affiliation"] == "channel"
    on_disk = json.loads((env / "auth" / "users.json").read_text(encoding="utf-8"))
    stored = {user["username"]: user for user in on_disk["users"]}
    assert stored["chan1"]["affiliation"] == "channel"

    # A legacy record without the field (pre-affiliation store) reads company.
    users = auth_store.load_users()
    for user in users:
        user.pop("affiliation", None)
    auth_store.save_users(users)
    assert auth_store.is_company_user("chan1") is True
    listing = admin.get("/api/auth/users")
    assert listing.status_code == 200
    assert all(user["affiliation"] == "company" for user in listing.json()["users"])

    # The seeded admin itself carries the company default.
    assert auth_store.is_company_user("admin") is True


def test_is_company_user_when_auth_is_off(env, monkeypatch):
    # Uninitialized store: everyone reads company (auth is off).
    assert auth_store.is_company_user("nobody") is True
    # Seeded store, unknown username: never company.
    seeded_admin()
    assert auth_store.is_company_user("nobody") is False
    # The explicit bypass reads company again, even for unknown names.
    monkeypatch.setenv("KAIROS_AUTH_DISABLED", "1")
    assert auth_store.is_company_user("nobody") is True


def test_add_user_rejects_an_unknown_affiliation(env):
    seeded_admin()
    with pytest.raises(ValueError, match="affiliation"):
        auth_store.add_user("weird1", "longenough-123", "operator", affiliation="agency")


# ---------------------------------------------------------------------------
# /api/auth/me and account management carry the affiliation
# ---------------------------------------------------------------------------

def test_me_carries_the_affiliation(env):
    admin = seeded_admin()
    create_operator(admin, "chan1", CHANNEL_PASSWORD, "channel")
    assert admin.get("/api/auth/me").json()["affiliation"] == "company"
    channel = signed_in("chan1", CHANNEL_PASSWORD)
    body = channel.get("/api/auth/me").json()
    assert body["affiliation"] == "channel"
    assert body["auth_disabled"] is False


def test_admin_flips_affiliation_and_it_takes_effect_live(env):
    admin = seeded_admin()
    create_operator(admin, "chan1", CHANNEL_PASSWORD, "channel")
    channel = signed_in("chan1", CHANNEL_PASSWORD)
    assert channel.post("/api/events", json=EVENT_PAYLOAD).status_code == 403

    flipped = admin.put("/api/auth/users/chan1/affiliation", json={"affiliation": "company"})
    assert flipped.status_code == 200 and flipped.json()["affiliation"] == "company"
    # The SAME live session gains the surface without signing in again.
    created = channel.post("/api/events", json=EVENT_PAYLOAD)
    assert created.status_code == 201, created.text

    back = admin.put("/api/auth/users/chan1/affiliation", json={"affiliation": "channel"})
    assert back.status_code == 200 and back.json()["affiliation"] == "channel"
    denied = channel.delete(f"/api/events/{created.json()['event_id']}")
    assert denied.status_code == 403

    assert admin.put("/api/auth/users/ghost/affiliation",
                     json={"affiliation": "channel"}).status_code == 404
    assert admin.put("/api/auth/users/chan1/affiliation",
                     json={"affiliation": "agency"}).status_code == 422
    # A non-admin cannot reach the affiliation endpoint at all.
    assert channel.put("/api/auth/users/chan1/affiliation",
                       json={"affiliation": "company"}).status_code == 403


# ---------------------------------------------------------------------------
# Events surface: reads open, writes company-only, can_edit honest
# ---------------------------------------------------------------------------

def test_channel_user_reads_events_but_every_write_is_403(env):
    admin = seeded_admin()
    create_operator(admin, "comp1", COMPANY_PASSWORD, "company")
    create_operator(admin, "chan1", CHANNEL_PASSWORD, "channel")

    company = signed_in("comp1", COMPANY_PASSWORD)
    created = company.post("/api/events", json=EVENT_PAYLOAD)
    assert created.status_code == 201, created.text
    event_id = created.json()["event_id"]
    company_view = company.get("/api/events")
    assert company_view.status_code == 200
    assert company_view.json()["can_edit"] is True

    channel = signed_in("chan1", CHANNEL_PASSWORD)
    view = channel.get("/api/events")
    assert view.status_code == 200, "reads stay open to channel accounts"
    body = view.json()
    assert body["can_edit"] is False
    assert [event["event_id"] for event in body["events"]] == [event_id]
    assert "model_context" in body and "holidays" in body

    post = channel.post("/api/events", json=EVENT_PAYLOAD)
    assert post.status_code == 403 and post.json()["detail"] == EVENTS_403_DETAIL
    put = channel.put(f"/api/events/{event_id}", json={"intensity": 1})
    assert put.status_code == 403 and put.json()["detail"] == EVENTS_403_DETAIL
    delete = channel.delete(f"/api/events/{event_id}")
    assert delete.status_code == 403 and delete.json()["detail"] == EVENTS_403_DETAIL

    # The store is untouched by the denied writes.
    after = company.get("/api/events").json()["events"]
    assert [event["event_id"] for event in after] == [event_id]
    assert after[0]["intensity"] == EVENT_PAYLOAD["intensity"]

    # The admin (company by default) also keeps can_edit true.
    assert admin.get("/api/events").json()["can_edit"] is True


# ---------------------------------------------------------------------------
# Pricing surface: only the events activation key is company-walled
# ---------------------------------------------------------------------------

def test_events_activation_put_is_company_only_other_pricing_edits_pass(env):
    admin = seeded_admin()
    create_operator(admin, "comp1", COMPANY_PASSWORD, "company")
    create_operator(admin, "chan1", CHANNEL_PASSWORD, "channel")
    channel = signed_in("chan1", CHANNEL_PASSWORD)

    denied = channel.put("/api/pricing", json={
        "overrides": {"pricing_activation": {"events": True}},
    })
    assert denied.status_code == 403 and denied.json()["detail"] == PRICING_403_DETAIL
    also_denied = channel.put("/api/pricing", json={
        "overrides": {"pricing_activation": {"events": False}},
    })
    assert also_denied.status_code == 403, "touching the key at all is walled, not just enabling"

    # Any other pricing edit stays open to a channel operator and leaves the
    # events activation exactly as it was (the deployed settings ship it on).
    state = channel.get("/api/pricing").json()
    enabled_before = state["events"]["enabled"]
    other = channel.put("/api/pricing", json={
        "overrides": {"base_price_per_second_per_tvr_point": float(state["base"]["value"])},
    })
    assert other.status_code == 200, other.text
    assert other.json()["events"]["enabled"] is enabled_before

    company = signed_in("comp1", COMPANY_PASSWORD)
    off = company.put("/api/pricing", json={
        "overrides": {"pricing_activation": {"events": False}},
    })
    assert off.status_code == 200, off.text
    assert off.json()["events"]["enabled"] is False
    on = company.put("/api/pricing", json={
        "overrides": {"pricing_activation": {"events": True}},
    })
    assert on.status_code == 200, on.text
    assert on.json()["events"]["enabled"] is True


# ---------------------------------------------------------------------------
# Auth off: everything reads company and nothing changes
# ---------------------------------------------------------------------------

def test_auth_off_keeps_the_events_surface_fully_open(env):
    client = TestClient(app)
    view = client.get("/api/events")
    assert view.status_code == 200
    assert view.json()["can_edit"] is True
    created = client.post("/api/events", json=EVENT_PAYLOAD)
    assert created.status_code == 201, created.text
    assert client.delete(f"/api/events/{created.json()['event_id']}").status_code == 200


# ---------------------------------------------------------------------------
# Apply-time mirror of the propose-time gate
# ---------------------------------------------------------------------------

def test_apply_time_block_mirrors_the_propose_gate(env):
    """A channel account must not be able to APPLY a company-only pending item
    it could never have proposed. The block is the pure helper the apply route
    calls, so it is pinned directly for every company-only kind plus the
    events-activation pricing shape, and proven silent for company actors and
    for kinds that are not company-only."""
    from kairos_api.events_access import assistant_apply_block

    admin = seeded_admin()
    create_operator(admin, "chan", "chanpass-123", "channel")
    create_operator(admin, "comp", "comppass-123", "company")

    event_item = {"kind": "event_change", "payload": {"action": "create"}}
    agency_item = {"kind": "agency_change", "payload": {"action": "update_record"}}
    activation_item = {"kind": "pricing_change",
                       "payload": {"changes": {"pricing_activation": {"events": True}}}}
    plain_pricing = {"kind": "pricing_change",
                     "payload": {"changes": {"base_cpp": 61.0}}}
    settings_item = {"kind": "settings", "payload": {"changes": {"revenue_weight": 70}}}

    for item in (event_item, agency_item, activation_item):
        assert assistant_apply_block("chan", [item]), f"channel must be blocked on {item['kind']}"
    assert assistant_apply_block("chan", [settings_item, event_item])
    assert assistant_apply_block("chan", [plain_pricing]) is None
    assert assistant_apply_block("chan", [settings_item]) is None
    for item in (event_item, agency_item, activation_item, plain_pricing, settings_item):
        assert assistant_apply_block("comp", [item]) is None
