"""The subscription bridge: order, freshness, and what may never cross it.

The bridge exists so Kai in the cloud runs on the operator's own Claude plan:
the machine pushes its short-lived access token, the cloud reads it at call
time. Three properties carry the whole design and each is pinned here:

- ORDER: a local keychain login always outranks the bridge, and the bridge
  outranks the console API key — the machine's own session is never shadowed
  by its own push, and credit is the fallback, not the default.
- FRESHNESS: one rule on both ends; a token near expiry is refused like an
  expired one, because a long answer can outlive it mid-call.
- CUSTODY: the payload is an allow-list. The refresh token — the only
  long-lived credential — can never ride along, even if the blob grows fields.
"""

from __future__ import annotations

import time

import pytest

from kairos_api import assistant_auth, assistant_bridge


NOW_MS = int(time.time() * 1000)
HOUR = 60 * 60 * 1000


def _blob(expires_in_ms: int = HOUR, **extra) -> dict:
    oauth = {
        "accessToken": "tok-live",
        "refreshToken": "tok-refresh-NEVER-CROSSES",
        "expiresAt": NOW_MS + expires_in_ms,
        "subscriptionType": "max",
    }
    oauth.update(extra)
    return {"claudeAiOauth": oauth}


def test_payload_is_an_allow_list_and_never_carries_the_refresh_token():
    payload = assistant_bridge.build_payload(_blob(unexpected="field"), "netanel")
    assert payload is not None
    assert set(payload) == set(assistant_bridge.PAYLOAD_FIELDS)
    assert "refresh" not in str(payload)
    assert payload["account"] == "netanel"
    assert payload["subscriptionType"] == "max"


def test_an_expired_or_near_expiry_token_is_not_pushed():
    assert assistant_bridge.build_payload(_blob(expires_in_ms=-1), None) is None
    near = assistant_bridge.FRESHNESS_MARGIN_MS - 1000
    assert assistant_bridge.build_payload(_blob(expires_in_ms=near), None) is None


def test_freshness_rule_is_shared_and_margin_guarded():
    fresh = {"expiresAt": NOW_MS + HOUR}
    stale = {"expiresAt": NOW_MS + assistant_bridge.FRESHNESS_MARGIN_MS - 1}
    assert assistant_bridge.is_fresh(fresh, now_ms=NOW_MS)
    assert not assistant_bridge.is_fresh(stale, now_ms=NOW_MS)
    assert not assistant_bridge.is_fresh({}, now_ms=NOW_MS)


def test_without_the_arn_the_bridge_is_silent_and_costs_nothing(monkeypatch):
    monkeypatch.delenv(assistant_bridge.BRIDGE_SECRET_ARN_ENV, raising=False)

    def explode(_arn):
        raise AssertionError("the bridge must not read anything without an ARN")

    monkeypatch.setattr(assistant_bridge, "_read_secret", explode)
    assert assistant_bridge.fetch_bridged_payload() is None
    assert assistant_auth._bridge_auth() is None
    assert assistant_bridge.bridge_state() is None


def test_bridge_outranks_the_console_key_and_reports_its_source(monkeypatch):
    monkeypatch.setenv(assistant_bridge.BRIDGE_SECRET_ARN_ENV, "arn:test:bridge")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-console")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    payload = assistant_bridge.build_payload(_blob(), "netanel")
    monkeypatch.setattr(assistant_bridge, "_read_secret", lambda arn: dict(payload))
    monkeypatch.setattr(assistant_bridge, "_cache", {"at": 0.0, "payload": None})
    # conftest turns keychain OAuth off, which is exactly the cloud shape.
    auth = assistant_auth.resolve_auth()
    assert auth is not None
    assert auth.mode == "oauth"
    assert auth.source == "bridge:netanel"
    assert auth.token == "tok-live"
    assert auth.subscription_type == "max"


def test_a_stale_bridge_falls_back_to_the_console_key(monkeypatch):
    monkeypatch.setenv(assistant_bridge.BRIDGE_SECRET_ARN_ENV, "arn:test:bridge")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-console")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    stale = {"accessToken": "tok-old", "expiresAt": NOW_MS - HOUR,
             "account": "netanel", "subscriptionType": "max", "pushedAt": NOW_MS - 2 * HOUR}
    monkeypatch.setattr(assistant_bridge, "_read_secret", lambda arn: dict(stale))
    monkeypatch.setattr(assistant_bridge, "_cache", {"at": 0.0, "payload": None})
    auth = assistant_auth.resolve_auth()
    assert auth is not None
    assert auth.mode == "api_key"
    state = assistant_bridge.bridge_state()
    assert state == {"configured": True, "fresh": False, "account": "netanel",
                     "expires_at_ms": stale["expiresAt"], "pushed_at_ms": stale["pushedAt"]}


def test_a_local_keychain_login_outranks_the_bridge(monkeypatch):
    monkeypatch.setenv(assistant_bridge.BRIDGE_SECRET_ARN_ENV, "arn:test:bridge")
    monkeypatch.setenv("KAIROS_ASSISTANT_USE_CLAUDE_CODE_OAUTH", "1")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(assistant_auth, "_load_credentials_blob", lambda: _blob())
    monkeypatch.setattr(assistant_auth, "_active_switch_label", lambda: "netanel")

    def explode(_arn):
        raise AssertionError("with a live keychain login the bridge must not be read")

    monkeypatch.setattr(assistant_bridge, "_read_secret", explode)
    monkeypatch.setattr(assistant_bridge, "_cache", {"at": 0.0, "payload": None})
    auth = assistant_auth.resolve_auth()
    assert auth is not None
    assert auth.source.startswith("claude_code_")


def test_a_broken_secret_read_is_an_absent_source_not_a_crash(monkeypatch):
    monkeypatch.setenv(assistant_bridge.BRIDGE_SECRET_ARN_ENV, "arn:test:bridge")

    def broken(_arn):
        raise RuntimeError("network down")

    monkeypatch.setattr(assistant_bridge, "_read_secret", broken)
    monkeypatch.setattr(assistant_bridge, "_cache", {"at": 0.0, "payload": None})
    assert assistant_bridge.fetch_bridged_payload() is None
