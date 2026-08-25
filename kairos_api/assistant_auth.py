"""Resolve credentials for the in-product assistant.

Two real paths exist on a developer machine:

1. **Claude Max / Claude Code OAuth**, the same subscription used by the
   local ``claude`` CLI (here: netanel@questo.media via ``claude-switch``).
   The live access token lives in the macOS Keychain item
   ``Claude Code-credentials`` (or ``~/.claude/.credentials.json`` on other
   platforms). Auth is ``Authorization: Bearer <accessToken>`` through the
   Anthropic SDK's ``auth_token=`` parameter, NOT ``x-api-key``.

2. **Console API key**, ``ANTHROPIC_API_KEY`` / ``KAIROS_ASSISTANT_API_KEY``
   pay-as-you-go keys. Separate product surface from Max; no credits on the
   key is independent of Max quota.

Default preference when ``KAIROS_ASSISTANT_USE_CLAUDE_CODE_OAUTH`` is not
disabled: try Claude Code OAuth first, then fall back to an API key. Tests
set the flag to ``0`` in ``tests/conftest.py`` so ambient Keychain state never
makes a suite look "configured".
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

KEY_ENVS = ("ANTHROPIC_API_KEY", "KAIROS_ASSISTANT_API_KEY")
AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
OAUTH_FLAG_ENV = "KAIROS_ASSISTANT_USE_CLAUDE_CODE_OAUTH"
# macOS Keychain service names Claude Code has used.
KEYCHAIN_SERVICES = ("Claude Code-credentials",)
CREDENTIALS_FILE = Path.home() / ".claude" / ".credentials.json"
ACTIVE_SWITCH_PATH = Path.home() / ".claude-switch" / "active.json"


@dataclass(frozen=True)
class AssistantAuth:
    """One resolved credential. ``token`` is never logged."""

    mode: str  # "oauth" | "api_key"
    token: str
    source: str
    account_hint: str | None = None
    subscription_type: str | None = None
    expires_at_ms: int | None = None

    def public_status(self) -> dict[str, Any]:
        """Fields safe to return from /api/assistant/status."""
        body: dict[str, Any] = {
            "mode": self.mode,
            "source": self.source,
        }
        if self.account_hint:
            body["account"] = self.account_hint
        if self.subscription_type:
            body["subscription_type"] = self.subscription_type
        if self.expires_at_ms:
            body["expires_at_ms"] = self.expires_at_ms
            body["expired"] = int(time.time() * 1000) >= self.expires_at_ms
        return body


def _flag_enabled(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _oauth_enabled() -> bool:
    # Default ON for real local use; tests force OFF via conftest.
    return _flag_enabled(OAUTH_FLAG_ENV, default=True)


def _active_switch_label() -> str | None:
    try:
        if not ACTIVE_SWITCH_PATH.is_file():
            return None
        data = json.loads(ACTIVE_SWITCH_PATH.read_text(encoding="utf-8"))
        label = data.get("label") or data.get("active") or data.get("account")
        return str(label).strip() if label else None
    except Exception:
        return None


def _load_credentials_blob() -> dict[str, Any] | None:
    """Load the live Claude Code credentials blob (never logs content)."""
    # 1) macOS Keychain, where Claude Code and claude-switch keep the active login.
    for service in KEYCHAIN_SERVICES:
        try:
            result = subprocess.run(
                ["security", "find-generic-password", "-s", service, "-w"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                return json.loads(result.stdout.strip())
        except FileNotFoundError:
            break  # not macOS / no security(1)
        except Exception as exc:
            logger.debug("keychain read %s failed: %s", service, exc)

    # 2) File path used on Linux/Windows and some Claude Code builds.
    try:
        if CREDENTIALS_FILE.is_file():
            return json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.debug("credentials file read failed: %s", exc)
    return None


def _oauth_from_blob(blob: dict[str, Any]) -> AssistantAuth | None:
    oauth = blob.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    token = str(oauth.get("accessToken") or "").strip()
    if not token:
        return None
    expires_at = oauth.get("expiresAt")
    expires_at_ms: int | None
    try:
        expires_at_ms = int(expires_at) if expires_at is not None else None
    except (TypeError, ValueError):
        expires_at_ms = None
    if expires_at_ms is not None and int(time.time() * 1000) >= expires_at_ms:
        # Do not hand the assistant an already-expired token; Claude Code will
        # refresh on its next launch. Fail over to API key or unavailable.
        logger.info("Claude Code OAuth access token is expired; not using it")
        return None

    label = _active_switch_label()
    # Best-effort account hint from claude-switch active file + oauth metadata.
    account_hint = label
    sub = oauth.get("subscriptionType")
    subscription = str(sub).strip() if sub else None
    source = "claude_code_keychain" if os.uname().sysname == "Darwin" else "claude_code_credentials_file"
    if label:
        source = f"{source}:{label}"
    return AssistantAuth(
        mode="oauth",
        token=token,
        source=source,
        account_hint=account_hint,
        subscription_type=subscription,
        expires_at_ms=expires_at_ms,
    )


def _api_key_from_env() -> AssistantAuth | None:
    for name in KEY_ENVS:
        value = os.environ.get(name, "").strip()
        if value:
            return AssistantAuth(mode="api_key", token=value, source=name)
    return None


def _auth_token_from_env() -> AssistantAuth | None:
    value = os.environ.get(AUTH_TOKEN_ENV, "").strip()
    if not value:
        return None
    return AssistantAuth(
        mode="oauth",
        token=value,
        source=AUTH_TOKEN_ENV,
        subscription_type=None,
    )


def _bridge_auth() -> Optional[AssistantAuth]:
    """The subscription bridge: a short-lived access token the operator's
    machine pushes to Secrets Manager, read here at call time. Configured by
    the bridge secret ARN env, so a machine without it pays nothing."""
    from kairos_api import assistant_bridge

    if not os.environ.get(assistant_bridge.BRIDGE_SECRET_ARN_ENV, "").strip():
        return None
    payload = assistant_bridge.fetch_bridged_payload()
    if payload is None:
        return None
    account = payload.get("account")
    return AssistantAuth(
        mode="oauth",
        token=str(payload["accessToken"]),
        source=f"bridge:{account}" if account else "bridge",
        account_hint=account,
        subscription_type=payload.get("subscriptionType"),
        expires_at_ms=payload.get("expiresAt"),
    )


def resolve_auth() -> Optional[AssistantAuth]:
    """Pick the credential the assistant should use, or None.

    Order:
    1. Explicit ``ANTHROPIC_AUTH_TOKEN`` (Bearer / Max-style).
    2. Claude Code OAuth from Keychain / credentials file (when enabled).
    3. The subscription bridge (cloud: the operator's machine pushes its live
       access token; the refresh token never leaves that machine).
    4. Console API key from env.
    """
    explicit = _auth_token_from_env()
    if explicit is not None:
        return explicit

    if _oauth_enabled():
        blob = _load_credentials_blob()
        if blob:
            oauth = _oauth_from_blob(blob)
            if oauth is not None:
                return oauth

    bridge = _bridge_auth()
    if bridge is not None:
        return bridge

    return _api_key_from_env()


def build_client(auth: AssistantAuth, *, timeout: float, max_retries: int = 1) -> Any:
    """Construct an Anthropic client for the resolved auth mode."""
    import anthropic

    if auth.mode == "oauth":
        # Max / Claude Code OAuth: Bearer token. Do not pass api_key, because an ambient
        # ANTHROPIC_API_KEY must not override the subscription path.
        return anthropic.Anthropic(
            auth_token=auth.token,
            timeout=timeout,
            max_retries=max_retries,
        )
    return anthropic.Anthropic(
        api_key=auth.token,
        timeout=timeout,
        max_retries=max_retries,
    )
