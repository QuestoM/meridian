"""End-to-end tests for the Kairos login / user system (kairos_api.auth).

Every test points KAIROS_AUTH_DIR at a per-test tmp directory, so the real
data/auth store is never created or touched, and clears the in-process
session and rate-limit state on both sides of the test.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from kairos_api import auth_store
from kairos_api.server import app

ROOT = Path(__file__).resolve().parents[1]

ADMIN_PASSWORD = "rootpass-1234"


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    yield tmp_path / "auth"
    auth_store.reset_runtime_state()


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
# Uninitialized store: setup is required and the API stays closed
# ---------------------------------------------------------------------------

def test_uninitialized_store_reports_setup_required_and_api_closed(auth_env):
    client = TestClient(app)
    session = client.get("/api/auth/session")
    assert session.status_code == 503
    assert "setup is required" in session.json()["detail"].lower()
    me = client.get("/api/auth/me")
    assert me.status_code == 503
    assert client.get("/api/settings").status_code == 503
    assert client.post("/api/definitely-not-a-route").status_code == 503
    # Login stays public only so it can explain the setup requirement.
    response = client.post("/api/auth/login", json={"username": "admin", "password": "whatever123"})
    assert response.status_code == 503


def test_failed_password_seed_cannot_leave_the_api_running_open(auth_env, monkeypatch):
    from kairos_api import auth

    monkeypatch.setenv("KAIROS_ADMIN_PASSWORD", "configured-but-invalid")
    monkeypatch.setattr(auth_store, "seed_initial_admin", lambda: (_ for _ in ()).throw(ValueError("bad seed")))

    with pytest.raises(RuntimeError, match="Could not seed"):
        auth._announce_auth_state()


def test_generated_password_delivery_failure_keeps_setup_required(auth_env, monkeypatch):
    """A missing one-time password must never activate an unreachable admin."""
    marker = auth_env / "initial_admin_password.txt"
    real_replace = auth_store.os.replace

    def fail_delivery(source: str | Path, destination: str | Path) -> None:
        if Path(destination) == marker:
            raise OSError("injected one-time password delivery failure")
        real_replace(source, destination)

    monkeypatch.setattr(auth_store.os, "replace", fail_delivery)

    with pytest.raises(OSError, match="injected one-time password delivery failure"):
        auth_store.seed_initial_admin()

    assert not auth_store.store_initialized()
    assert auth_store.load_users() == []
    assert not auth_store.users_path().exists()
    assert not marker.exists()
    assert not marker.with_name(marker.name + ".tmp").exists()

    client = TestClient(app)
    assert client.get("/api/auth/session").status_code == 503
    assert client.get("/api/settings").status_code == 503


def test_seed_script_seeds_admin_and_writes_one_time_password(auth_env):
    env = {**os.environ, "KAIROS_AUTH_DIR": str(auth_env)}
    env.pop("KAIROS_ADMIN_PASSWORD", None)
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "init_auth.py")],
        capture_output=True, text=True, cwd=str(ROOT), env=env, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    password_lines = [line for line in result.stdout.splitlines() if "One-time admin password:" in line]
    assert len(password_lines) == 1
    one_time = password_lines[0].split(":", 1)[1].strip()

    marker = auth_env / "initial_admin_password.txt"
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").strip() == one_time
    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert stat.S_IMODE((auth_env / "users.json").stat().st_mode) == 0o600

    client = TestClient(app)
    login = client.post("/api/auth/login", json={"username": "admin", "password": one_time})
    assert login.status_code == 200
    body = login.json()
    assert body["must_change_password"] is True
    assert body["role"] == "admin"

    # A second run refuses to overwrite the seeded store.
    rerun = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "init_auth.py")],
        capture_output=True, text=True, cwd=str(ROOT), env=env, check=False,
    )
    assert rerun.returncode == 1
    assert "already has accounts" in rerun.stdout


def test_env_password_seed_does_not_force_change(auth_env, monkeypatch):
    monkeypatch.setenv("KAIROS_ADMIN_PASSWORD", "env-admin-pass-99")
    username, generated = auth_store.seed_initial_admin()
    assert username == "admin"
    assert generated is None
    assert not (auth_env / "initial_admin_password.txt").exists()
    client = signed_in_client("admin", "env-admin-pass-99")
    assert client.get("/api/auth/me").json()["must_change_password"] is False


# ---------------------------------------------------------------------------
# Login, rate limit, sessions
# ---------------------------------------------------------------------------

def test_public_session_probe_distinguishes_signed_out_and_signed_in(auth_env):
    seed_admin()
    anonymous = TestClient(app)
    signed_out = anonymous.get("/api/auth/session")
    assert signed_out.status_code == 200
    assert signed_out.json() == {"authenticated": False, "auth_disabled": False}

    signed_in = signed_in_client("admin", ADMIN_PASSWORD)
    active = signed_in.get("/api/auth/session")
    assert active.status_code == 200
    body = active.json()
    assert body["authenticated"] is True
    assert body["auth_disabled"] is False
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    assert "password_scrypt" not in body

def test_wrong_password_401_then_rate_limited_429(auth_env):
    seed_admin()
    client = TestClient(app)
    for _ in range(5):
        response = client.post("/api/auth/login", json={"username": "admin", "password": "wrong-pass-123"})
        assert response.status_code == 401
    blocked = client.post("/api/auth/login", json={"username": "admin", "password": "wrong-pass-123"})
    assert blocked.status_code == 429
    retry_after = int(blocked.headers["Retry-After"])
    assert 1 <= retry_after <= auth_store.RATE_LIMIT_WINDOW_SECONDS + 1
    # Even the correct password is blocked while the window is saturated.
    still_blocked = client.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD})
    assert still_blocked.status_code == 429


def test_success_before_limit_resets_failure_counter(auth_env):
    seed_admin()
    client = TestClient(app)
    for _ in range(4):
        assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong-pass-123"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD}).status_code == 200
    # The counter was cleared: four fresh failures are tolerated again.
    for _ in range(4):
        assert client.post("/api/auth/login", json={"username": "admin", "password": "wrong-pass-123"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD}).status_code == 200


def test_login_sets_cookie_and_me_returns_identity(auth_env):
    seed_admin()
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD})
    assert response.status_code == 200
    set_cookie = response.headers["set-cookie"]
    assert "kairos_session=" in set_cookie
    assert "httponly" in set_cookie.lower()
    assert "samesite=lax" in set_cookie.lower()
    assert "path=/" in set_cookie.lower()
    assert "secure" not in set_cookie.lower()  # localhost HTTP; production adds TLS + Secure
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    body = me.json()
    assert body["auth_disabled"] is False
    assert body["username"] == "admin"
    assert body["role"] == "admin"
    assert "password_scrypt" not in body


def test_session_expiry_slides_on_use(auth_env, monkeypatch):
    seed_admin()
    client = signed_in_client("admin", ADMIN_PASSWORD)
    base = auth_store._now()
    hour = 3600.0
    monkeypatch.setattr(auth_store, "_now", lambda: base + 11 * hour)
    assert client.get("/api/auth/me").status_code == 200  # renewed to base + 23h
    monkeypatch.setattr(auth_store, "_now", lambda: base + 22 * hour)
    assert client.get("/api/auth/me").status_code == 200  # renewed to base + 34h
    monkeypatch.setattr(auth_store, "_now", lambda: base + 22 * hour + 13 * hour)
    assert client.get("/api/auth/me").status_code == 401  # idle past 12h: expired


def test_logout_kills_the_session(auth_env):
    seed_admin()
    client = signed_in_client("admin", ADMIN_PASSWORD)
    assert client.get("/api/auth/me").status_code == 200
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/auth/me").status_code == 401


# ---------------------------------------------------------------------------
# Enforcement: sessions required, viewer read-only, admin surface
# ---------------------------------------------------------------------------

def test_seeded_store_walls_api_but_not_health_or_shell(auth_env):
    seed_admin()
    client = TestClient(app)
    assert client.get("/api/settings").status_code == 401
    assert client.post("/api/definitely-not-a-route").status_code == 401
    assert client.get("/api/health").status_code == 200
    # Non-API routes (the SPA shell) are never walled; 200 with a built dist,
    # 404 without one, but never a 401.
    assert client.get("/").status_code != 401


def test_viewer_is_read_only_operator_and_admin_can_mutate(auth_env):
    seed_admin()
    admin = signed_in_client("admin", ADMIN_PASSWORD)
    create_account(admin, "viewer1", "viewerpass-123", "viewer")
    create_account(admin, "operator1", "operatorpass-123", "operator")

    viewer = signed_in_client("viewer1", "viewerpass-123")
    assert viewer.get("/api/settings").status_code == 200
    denied = viewer.post("/api/definitely-not-a-route")
    assert denied.status_code == 403  # blocked by the guard before routing
    assert viewer.get("/api/auth/users").status_code == 403

    operator = signed_in_client("operator1", "operatorpass-123")
    # 404/405 (no such route / static mount) proves the mutation passed the guard.
    assert operator.post("/api/definitely-not-a-route").status_code in (404, 405)
    assert operator.get("/api/auth/users").status_code == 403
    assert operator.delete("/api/auth/users/viewer1").status_code == 403

    assert admin.post("/api/definitely-not-a-route").status_code in (404, 405)


def test_admin_manages_accounts(auth_env):
    seed_admin()
    admin = signed_in_client("admin", ADMIN_PASSWORD)
    create_account(admin, "viewer1", "viewerpass-123", "viewer")

    # Duplicates are rejected honestly.
    duplicate = admin.post(
        "/api/auth/users",
        json={"username": "viewer1", "password": "otherpass-123", "role": "viewer", "display_name": "V"},
    )
    assert duplicate.status_code == 409
    bad_name = admin.post(
        "/api/auth/users",
        json={"username": "no way", "password": "otherpass-123", "role": "viewer", "display_name": "V"},
    )
    assert bad_name.status_code == 400

    listing = admin.get("/api/auth/users")
    assert listing.status_code == 200
    users = listing.json()["users"]
    assert {user["username"] for user in users} == {"admin", "viewer1"}
    assert all("password_scrypt" not in user and "hash_hex" not in str(user) for user in users)

    # No plaintext password ever lands on disk.
    stored = (auth_env / "users.json").read_text(encoding="utf-8")
    assert "viewerpass-123" not in stored
    assert "password_scrypt" in stored

    # Reset: live sessions die, the temporary password works and forces a change.
    viewer = signed_in_client("viewer1", "viewerpass-123")
    reset = admin.post("/api/auth/users/viewer1/reset-password", json={"new_password": "temporary-pass-9"})
    assert reset.status_code == 200
    assert reset.json()["must_change_password"] is True
    assert viewer.get("/api/auth/me").status_code == 401
    assert TestClient(app).post(
        "/api/auth/login", json={"username": "viewer1", "password": "viewerpass-123"}
    ).status_code == 401
    reborn = signed_in_client("viewer1", "temporary-pass-9")
    assert reborn.get("/api/auth/me").json()["must_change_password"] is True

    # Delete: the account and its sessions are gone.
    assert admin.delete("/api/auth/users/viewer1").status_code == 200
    assert reborn.get("/api/auth/me").status_code == 401
    assert admin.delete("/api/auth/users/viewer1").status_code == 404


def test_cannot_delete_yourself_or_the_last_admin(auth_env):
    seed_admin()
    admin = signed_in_client("admin", ADMIN_PASSWORD)
    # Sole admin: deleting it (which is also self-deletion) hits the last-admin wall.
    response = admin.delete("/api/auth/users/admin")
    assert response.status_code == 400
    assert "last admin" in response.json()["detail"].lower()
    # With a second admin present, self-deletion is still refused.
    create_account(admin, "admin2", "secondadmin-123", "admin")
    admin2 = signed_in_client("admin2", "secondadmin-123")
    self_delete = admin2.delete("/api/auth/users/admin2")
    assert self_delete.status_code == 400
    assert "signed in" in self_delete.json()["detail"].lower()
    # Deleting the other admin is allowed while one remains.
    assert admin2.delete("/api/auth/users/admin").status_code == 200


def test_change_password_flow(auth_env):
    seed_admin()
    admin = signed_in_client("admin", ADMIN_PASSWORD)
    create_account(admin, "viewer1", "viewerpass-123", "viewer")
    viewer = signed_in_client("viewer1", "viewerpass-123")

    wrong_current = viewer.post(
        "/api/auth/change-password",
        json={"current_password": "not-the-pass-1", "new_password": "brand-new-pass-1"},
    )
    assert wrong_current.status_code == 403
    too_short = viewer.post(
        "/api/auth/change-password",
        json={"current_password": "viewerpass-123", "new_password": "short"},
    )
    assert too_short.status_code == 422

    # A viewer session may change its own password (self-service is not a
    # read-only-walled mutation).
    changed = viewer.post(
        "/api/auth/change-password",
        json={"current_password": "viewerpass-123", "new_password": "brand-new-pass-1"},
    )
    assert changed.status_code == 200
    assert changed.json()["must_change_password"] is False
    assert TestClient(app).post(
        "/api/auth/login", json={"username": "viewer1", "password": "viewerpass-123"}
    ).status_code == 401
    assert TestClient(app).post(
        "/api/auth/login", json={"username": "viewer1", "password": "brand-new-pass-1"}
    ).status_code == 200


def test_auth_disabled_env_bypasses_enforcement(auth_env, monkeypatch):
    seed_admin()
    monkeypatch.setenv("KAIROS_AUTH_DISABLED", "1")
    client = TestClient(app)
    me = client.get("/api/auth/me")
    assert me.status_code == 200
    assert me.json() == {"auth_disabled": True}
    assert client.get("/api/settings").status_code == 200
    assert client.post("/api/definitely-not-a-route").status_code in (404, 405)
