"""The job field: a landing screen, never a permission.

Role says what an account may write and affiliation says which side of the line
it sees. Job says what this person's work is, which decides where they land and
nothing else. These tests pin that separation, the safe default that made the
field addable at all, and the exact payload the account record grew by.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from kairos_api import auth_store
from kairos_api.server import app

ADMIN_PASSWORD = "rootpass-1234"
VIEWER_PASSWORD = "viewerpass-123"
CHANNEL_PASSWORD = "channelpass-123"

# The account payload before this piece, plus the one key it added.
BASE_ACCOUNT_KEYS = {
    "username",
    "display_name",
    "role",
    "created_at",
    "must_change_password",
    "affiliation",
}


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    yield tmp_path / "auth"
    auth_store.reset_runtime_state()


def _signed_in(username: str, password: str) -> TestClient:
    client = TestClient(app)
    assert client.post("/api/auth/login", json={
        "username": username, "password": password}).status_code == 200
    return client


def _admin() -> TestClient:
    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
    return _signed_in("admin", ADMIN_PASSWORD)


# ---------------------------------------------------------------------------
# The safe default
# ---------------------------------------------------------------------------

def test_normalize_job_defaults_safely():
    assert auth_store.normalize_job(None) == "unset"
    assert auth_store.normalize_job("") == "unset"
    assert auth_store.normalize_job("chief of everything") == "unset"
    assert auth_store.normalize_job("PLANNER") == "planner"
    for job in auth_store.JOBS:
        assert auth_store.normalize_job(job) == job
    assert len(auth_store.JOBS) == 13
    assert len(set(auth_store.JOBS)) == 13


def test_a_record_that_predates_the_field_reads_unset(auth_env):
    admin = _admin()
    users = auth_store.load_users()
    for user in users:
        user.pop("job", None)
    auth_store.save_users(users)
    body = admin.get("/api/auth/me").json()
    assert body["job"] == "unset"
    assert all(user["job"] == "unset" for user in admin.get("/api/auth/users").json()["users"])


def test_the_account_payload_grew_by_exactly_one_key(auth_env):
    admin = _admin()
    body = admin.get("/api/auth/me").json()
    assert set(body) == BASE_ACCOUNT_KEYS | {"job", "auth_disabled"}
    listed = admin.get("/api/auth/users").json()["users"][0]
    assert set(listed) == BASE_ACCOUNT_KEYS | {"job"}
    assert "password_scrypt" not in body


# ---------------------------------------------------------------------------
# Self-service, because a job is not a permission
# ---------------------------------------------------------------------------

def test_a_viewer_sets_their_own_job_and_stays_read_only(auth_env):
    admin = _admin()
    assert admin.post("/api/auth/users", json={
        "username": "view1", "password": VIEWER_PASSWORD, "role": "viewer",
        "display_name": "view1", "must_change_password": False,
    }).status_code == 201

    viewer = _signed_in("view1", VIEWER_PASSWORD)
    assert viewer.get("/api/auth/me").json()["job"] == "unset"
    saved = viewer.put("/api/auth/job", json={"job": "traffic_operator"})
    assert saved.status_code == 200, saved.text
    assert saved.json()["job"] == "traffic_operator"
    assert viewer.get("/api/auth/me").json()["job"] == "traffic_operator"

    # The choice reached disk and changed nothing else about the account.
    stored = json.loads((auth_env / "users.json").read_text(encoding="utf-8"))
    record = {user["username"]: user for user in stored["users"]}["view1"]
    assert record["job"] == "traffic_operator"
    assert record["role"] == "viewer"
    assert record["affiliation"] == "company"

    # Still read-only: the job moved nobody's permission.
    assert viewer.post("/api/definitely-not-a-route").status_code == 403
    assert viewer.get("/api/auth/users").status_code == 403


def test_an_unknown_job_is_refused_and_unset_is_accepted(auth_env):
    admin = _admin()
    refused = admin.put("/api/auth/job", json={"job": "chief of everything"})
    assert refused.status_code == 400
    assert admin.get("/api/auth/me").json()["job"] == "unset"
    assert admin.put("/api/auth/job", json={"job": "model_steward"}).json()["job"] == "model_steward"
    assert admin.put("/api/auth/job", json={"job": "unset"}).json()["job"] == "unset"


def test_an_administrator_may_set_the_job_at_creation(auth_env):
    admin = _admin()
    created = admin.post("/api/auth/users", json={
        "username": "plan1", "password": "plannerpass-123", "role": "operator",
        "display_name": "plan1", "must_change_password": False, "job": "planner",
    })
    assert created.status_code == 201, created.text
    assert created.json()["job"] == "planner"
    # An account created without one is unset, so the picker asks.
    plain = admin.post("/api/auth/users", json={
        "username": "plan2", "password": "plannerpass-123", "role": "operator",
        "display_name": "plan2", "must_change_password": False,
    })
    assert plain.json()["job"] == "unset"


def test_the_job_does_not_open_the_company_side(auth_env):
    """A channel account may call itself the model steward and stay walled."""
    admin = _admin()
    assert admin.post("/api/auth/users", json={
        "username": "chan1", "password": CHANNEL_PASSWORD, "role": "operator",
        "display_name": "chan1", "must_change_password": False, "affiliation": "channel",
        "job": "model_steward",
    }).status_code == 201
    channel = _signed_in("chan1", CHANNEL_PASSWORD)
    assert channel.get("/api/auth/me").json()["job"] == "model_steward"
    assert auth_store.is_company_user("chan1") is False
    denied = channel.post("/api/events", json={
        "name": "בדיקה", "type": "sport", "start_date": "2024-11-20",
        "end_date": "2024-11-20", "intensity": 4, "notes": "", "active": True,
    })
    assert denied.status_code == 403
