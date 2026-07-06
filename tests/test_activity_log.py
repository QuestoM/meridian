"""Tests for the system-wide activity log (kairos_api/activity_log.py).

Every test points KAIROS_AUDIT_DIR at a per-test tmp directory so the real
data/audit store is never touched, and the auth-active tests additionally
point KAIROS_AUTH_DIR at a tmp store exactly like tests/test_auth.py. The
security-critical assertion here is on the RAW file bytes: a known password is
sent through the login flow and must never appear anywhere in the log file.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kairos_api import activity_log, auth_store
from kairos_api.server import app

ADMIN_PASSWORD = "rootpass-1234"

ENTRY_KEYS = {"ts", "user", "role", "event", "method", "path", "status", "duration_ms", "via"}


@pytest.fixture()
def audit_env(tmp_path, monkeypatch):
    """Relocated audit store with auth disabled (the conftest default)."""
    monkeypatch.setenv("KAIROS_AUDIT_DIR", str(tmp_path / "audit"))
    activity_log.reset_runtime_state()
    yield tmp_path / "audit"
    activity_log.reset_runtime_state()


@pytest.fixture()
def auth_env(audit_env, tmp_path, monkeypatch):
    """Relocated audit store plus a live (seeded-per-test) auth wall."""
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    yield audit_env
    auth_store.reset_runtime_state()


def read_log(audit_dir: Path) -> list[dict]:
    path = audit_dir / "activity.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def raw_log(audit_dir: Path) -> str:
    path = audit_dir / "activity.jsonl"
    return path.read_text(encoding="utf-8") if path.is_file() else ""


def seed_admin() -> None:
    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)


def signed_in_client(username: str, password: str) -> TestClient:
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return client


def create_account(admin: TestClient, username: str, password: str, role: str) -> None:
    response = admin.post(
        "/api/auth/users",
        json={
            "username": username,
            "password": password,
            "role": role,
            "display_name": username.title(),
            "must_change_password": False,
        },
    )
    assert response.status_code == 201, response.text


# ---------------------------------------------------------------------------
# Middleware recording: mutations logged with identity, reads not logged
# ---------------------------------------------------------------------------

def test_mutating_request_logged_with_identity_and_status(auth_env):
    seed_admin()
    admin = signed_in_client("admin", ADMIN_PASSWORD)
    response = admin.post("/api/pricing/price-slot", json={})
    assert response.status_code == 200, response.text

    entries = [e for e in read_log(auth_env) if e.get("path") == "/api/pricing/price-slot"]
    assert len(entries) == 1
    entry = entries[0]
    assert set(entry) == ENTRY_KEYS
    assert entry["user"] == "admin"
    assert entry["role"] == "admin"
    assert entry["event"] == "request"
    assert entry["method"] == "POST"
    assert entry["status"] == 200
    assert entry["via"] == "dashboard"
    assert isinstance(entry["duration_ms"], (int, float)) and entry["duration_ms"] >= 0
    assert datetime.fromisoformat(entry["ts"]).tzinfo is not None


def test_get_requests_are_not_logged(audit_env):
    client = TestClient(app)
    assert client.get("/api/settings").status_code == 200
    assert read_log(audit_env) == []
    # Differential control: a mutation in the same environment IS logged.
    assert client.post("/api/definitely-not-a-route").status_code in (404, 405)
    entries = read_log(audit_env)
    assert len(entries) == 1
    assert entries[0]["method"] == "POST"
    assert not any(e.get("method") == "GET" for e in entries)


# ---------------------------------------------------------------------------
# Auth events: dedicated hooks, and the password never touches the file
# ---------------------------------------------------------------------------

def test_auth_events_logged_and_password_never_on_disk(auth_env):
    seed_admin()
    client = TestClient(app)
    assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong-pass-123"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "ghost", "password": "whatever-pass-1"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD}).status_code == 200
    assert client.post("/api/auth/logout").status_code == 200

    raw = raw_log(auth_env)
    assert ADMIN_PASSWORD not in raw
    assert "wrong-pass-123" not in raw
    assert "whatever-pass-1" not in raw

    entries = read_log(auth_env)
    events = [(e["event"], e["user"]) for e in entries if e["event"] != "request"]
    assert events == [
        ("login_failed", "admin"),
        ("login_failed", "ghost"),
        ("login", "admin"),
        ("logout", "admin"),
    ]
    login_entry = next(e for e in entries if e["event"] == "login")
    assert login_entry["role"] == "admin"
    # Login and logout must never appear as middleware request entries: the
    # requests that carry credentials do not transit the recorder at all.
    assert not any(e.get("path") in ("/api/auth/login", "/api/auth/logout") for e in entries)


# ---------------------------------------------------------------------------
# Visibility: admin sees all, others see self, filter is admin-only
# ---------------------------------------------------------------------------

def test_role_scoping_matrix(auth_env):
    seed_admin()
    admin = signed_in_client("admin", ADMIN_PASSWORD)
    create_account(admin, "operator1", "operatorpass-123", "operator")
    create_account(admin, "viewer1", "viewerpass-123", "viewer")
    operator = signed_in_client("operator1", "operatorpass-123")
    viewer = signed_in_client("viewer1", "viewerpass-123")

    assert admin.post("/api/definitely-not-a-route").status_code in (404, 405)
    assert operator.post("/api/definitely-not-a-route").status_code in (404, 405)
    # The viewer's denied mutation (read-only role) is itself an audited event.
    assert viewer.post("/api/definitely-not-a-route").status_code == 403

    # Anonymous callers are walled by the auth middleware.
    assert TestClient(app).get("/api/activity-log").status_code == 401

    # Admin: scope "all", every actor visible.
    response = admin.get("/api/activity-log")
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "all"
    all_users = {entry["user"] for entry in body["entries"]}
    assert {"admin", "operator1", "viewer1"} <= all_users

    # Operator: scope "self", only its own entries, and it has some.
    response = operator.get("/api/activity-log")
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "self"
    assert body["entries"], "the operator's own login and mutation must be visible"
    assert {entry["user"] for entry in body["entries"]} == {"operator1"}

    # Viewer: scope "self", sees its own denied mutation with its real status.
    response = viewer.get("/api/activity-log")
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "self"
    assert {entry["user"] for entry in body["entries"]} == {"viewer1"}
    denied = [e for e in body["entries"] if e.get("path") == "/api/definitely-not-a-route"]
    assert denied and denied[0]["status"] == 403

    # The user filter is admin-only: everyone else is refused, even self-naming.
    assert operator.get("/api/activity-log", params={"user": "admin"}).status_code == 403
    assert viewer.get("/api/activity-log", params={"user": "viewer1"}).status_code == 403

    # Admin filter narrows to the named user.
    response = admin.get("/api/activity-log", params={"user": "operator1"})
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "all"
    assert body["entries"]
    assert {entry["user"] for entry in body["entries"]} == {"operator1"}

    # The reads above never landed in the log themselves.
    assert not any(entry.get("method") == "GET" for entry in read_log(auth_env))


def test_auth_disabled_scope_all_and_assistant_via(audit_env):
    client = TestClient(app)
    assert client.post("/api/definitely-not-a-route").status_code in (404, 405)
    assert client.post("/api/assistant/definitely-not-a-route").status_code in (404, 405)

    response = client.get("/api/activity-log")
    assert response.status_code == 200
    body = response.json()
    assert body["scope"] == "all"
    by_path = {entry["path"]: entry for entry in body["entries"]}
    assert by_path["/api/definitely-not-a-route"]["user"] == "auth-disabled"
    assert by_path["/api/definitely-not-a-route"]["via"] == "dashboard"
    assert by_path["/api/assistant/definitely-not-a-route"]["via"] == "assistant"
    # Newest first: the assistant call happened last, so it is listed first.
    assert body["entries"][0]["path"] == "/api/assistant/definitely-not-a-route"

    # Single-tenant dev mode: the narrowing filter still works (never widens).
    filtered = client.get("/api/activity-log", params={"user": "nobody"}).json()
    assert filtered["entries"] == []


# ---------------------------------------------------------------------------
# Store behavior: prune keeps the newest entries, failures never surface
# ---------------------------------------------------------------------------

def test_prune_keeps_newest_5000_entries(audit_env):
    for index in range(6001):
        activity_log.record_auth_event("login", user=f"user-{index}", role="viewer")
    entries = read_log(audit_env)
    assert len(entries) == 5000
    assert entries[0]["user"] == "user-1001"
    assert entries[-1]["user"] == "user-6000"


def test_append_failure_never_breaks_the_request(audit_env, monkeypatch):
    def boom(entry):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(activity_log, "_append_entry", boom)
    client = TestClient(app)
    response = client.post("/api/pricing/price-slot", json={})
    assert response.status_code == 200, response.text
    assert read_log(audit_env) == []
