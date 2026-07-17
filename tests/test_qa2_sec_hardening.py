"""Security hardening: the cookie Secure flag and the login rate-limiter cap.

- KAIROS_COOKIE_SECURE=1 makes the session cookie Secure on login and logout;
  the default stays off so plain-HTTP localhost keeps working.
- The failed-logins map prunes its oldest usernames past the cap, so a spray of
  invented usernames cannot grow the in-process dict without bound, while the
  most recent failures (a live attack window) always survive.

The auth store is relocated to tmp_path via KAIROS_AUTH_DIR; nothing touches
real data/.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.auth as auth
import kairos_api.auth_store as auth_store

PASSWORD = "rootpass-1234"


@pytest.fixture()
def login_client(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    auth_store.reset_runtime_state()
    auth_store.seed_initial_admin(password=PASSWORD)
    app = FastAPI()
    app.include_router(auth.router)
    yield TestClient(app)
    auth_store.reset_runtime_state()


def _login(client: TestClient):
    response = client.post("/api/auth/login", json={"username": "admin", "password": PASSWORD})
    assert response.status_code == 200, response.text
    return response


def test_cookie_is_not_secure_by_default(login_client, monkeypatch) -> None:
    monkeypatch.delenv("KAIROS_COOKIE_SECURE", raising=False)
    header = _login(login_client).headers["set-cookie"]
    assert auth_store.COOKIE_NAME in header
    assert "httponly" in header.lower()
    assert "secure" not in header.lower(), "plain-HTTP localhost must keep working by default"


def test_cookie_secure_flag_applies_on_login_and_logout(login_client, monkeypatch) -> None:
    monkeypatch.setenv("KAIROS_COOKIE_SECURE", "1")
    login_header = _login(login_client).headers["set-cookie"]
    assert "secure" in login_header.lower()
    assert "httponly" in login_header.lower()

    logout = login_client.post("/api/auth/logout")
    assert logout.status_code == 200
    assert "secure" in logout.headers["set-cookie"].lower(), "the deletion cookie must match"


def test_failed_logins_map_prunes_oldest_usernames_past_the_cap(monkeypatch) -> None:
    auth_store.reset_runtime_state()
    monkeypatch.setattr(auth_store, "MAX_TRACKED_FAILED_USERNAMES", 5)
    # A deterministic clock so "oldest" is unambiguous.
    clock = {"now": 1_000_000.0}

    def tick() -> float:
        clock["now"] += 1.0
        return clock["now"]

    monkeypatch.setattr(auth_store, "_now", tick)
    try:
        for index in range(9):
            auth_store.record_login_failure(f"sprayed-{index}")
        assert len(auth_store._FAILED_LOGINS) == 5, "the map must not grow past the cap"
        # The newest five survive; the earliest four were pruned oldest-first.
        assert set(auth_store._FAILED_LOGINS) == {f"sprayed-{i}" for i in range(4, 9)}

        # A real username's recent failures still rate-limit after the spray.
        for _ in range(auth_store.RATE_LIMIT_MAX_FAILURES):
            auth_store.record_login_failure("admin")
        assert auth_store.login_retry_after("admin") > 0
    finally:
        auth_store.reset_runtime_state()
