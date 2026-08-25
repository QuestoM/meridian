"""The subscription bridge: Kai in the cloud on the operator's own Claude plan.

The operator's machine holds the only long-lived credential (the Claude Code
refresh token, in the macOS Keychain). The cloud never sees it. What crosses
is the SHORT-LIVED access token: a launchd job on the machine pushes it to one
Secrets Manager secret every few minutes, and the cloud task reads that secret
at call time and speaks to Anthropic with ``Authorization: Bearer`` exactly as
the local product does. When the machine is off long enough for the last
pushed token to expire, the assistant reports that truthfully and falls back
to the console API key if one is configured.

Payload contract (one JSON object in the secret):
  accessToken       the live Claude Code access token
  expiresAt         epoch milliseconds, copied from the credentials blob
  account           the claude-switch label, for the status surface
  subscriptionType  as the blob reports it
  pushedAt          epoch milliseconds at push time

``refreshToken`` is NEVER part of the payload: :func:`build_payload` strips to
the allow-list above, so a future blob field cannot ride along unnoticed.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

BRIDGE_SECRET_ARN_ENV = "KAIROS_ASSISTANT_BRIDGE_SECRET_ARN"

# A token about to expire is as useless as an expired one: a long answer can
# outlive it mid-call. Refuse anything with less than this much runway.
FRESHNESS_MARGIN_MS = 5 * 60 * 1000

# The cloud side caches the secret read so a burst of questions costs one
# Secrets Manager call, not one per question.
FETCH_CACHE_SECONDS = 120

PAYLOAD_FIELDS = ("accessToken", "expiresAt", "account", "subscriptionType", "pushedAt")


def build_payload(blob: dict[str, Any], label: str | None) -> dict[str, Any] | None:
    """The push payload from a Claude Code credentials blob, or None.

    Returns None when the blob holds no usable (present, unexpired) access
    token — the pusher then pushes nothing, and the last pushed token ages out
    on its own schedule.
    """
    oauth = blob.get("claudeAiOauth")
    if not isinstance(oauth, dict):
        return None
    token = str(oauth.get("accessToken") or "").strip()
    if not token:
        return None
    try:
        expires_at = int(oauth.get("expiresAt"))
    except (TypeError, ValueError):
        return None
    payload = {
        "accessToken": token,
        "expiresAt": expires_at,
        "account": (label or "").strip() or None,
        "subscriptionType": (str(oauth.get("subscriptionType") or "").strip() or None),
        "pushedAt": int(time.time() * 1000),
    }
    if not is_fresh(payload):
        return None
    return {key: payload[key] for key in PAYLOAD_FIELDS}


def is_fresh(payload: dict[str, Any], now_ms: int | None = None) -> bool:
    """One freshness rule for both ends of the bridge."""
    try:
        expires_at = int(payload.get("expiresAt"))
    except (TypeError, ValueError):
        return False
    now = int(time.time() * 1000) if now_ms is None else now_ms
    return now + FRESHNESS_MARGIN_MS < expires_at


# --------------------------------------------------------------------- cloud
_cache: dict[str, Any] = {"at": 0.0, "payload": None}


def _read_secret(arn: str) -> dict[str, Any] | None:
    """One Secrets Manager read. boto3 is imported here, lazily, because the
    local product never sets the ARN and must not pay the import."""
    import boto3

    client = boto3.client("secretsmanager")
    value = client.get_secret_value(SecretId=arn)
    return json.loads(value["SecretString"])


def fetch_bridged_payload() -> Optional[dict[str, Any]]:
    """The current bridged payload, fresh or None. Never raises into the ask
    path: a broken bridge is an unavailable source, not a crashed question."""
    arn = os.environ.get(BRIDGE_SECRET_ARN_ENV, "").strip()
    if not arn:
        return None
    now = time.monotonic()
    if _cache["payload"] is not None and now - _cache["at"] < FETCH_CACHE_SECONDS:
        payload = _cache["payload"]
        return payload if is_fresh(payload) else None
    try:
        payload = _read_secret(arn)
    except Exception as exc:
        logger.warning("bridge secret read failed: %s", type(exc).__name__)
        return None
    if not isinstance(payload, dict):
        return None
    _cache["at"] = now
    _cache["payload"] = payload
    return payload if is_fresh(payload) else None


def bridge_state() -> dict[str, Any] | None:
    """For the status surface: configured or not, fresh or stale, and since
    when. Never includes the token."""
    arn = os.environ.get(BRIDGE_SECRET_ARN_ENV, "").strip()
    if not arn:
        return None
    payload = _cache["payload"]
    if payload is None:
        payload = fetch_bridged_payload() or _cache["payload"]
    if payload is None:
        return {"configured": True, "fresh": False}
    return {
        "configured": True,
        "fresh": is_fresh(payload),
        "account": payload.get("account"),
        "expires_at_ms": payload.get("expiresAt"),
        "pushed_at_ms": payload.get("pushedAt"),
    }


# --------------------------------------------------------------------- push
def push(secret_id: str, profile: str, region: str) -> int:
    """Read the live Claude Code token and push it to the bridge secret.

    The token travels keychain -> pipe -> AWS CLI stdin; it is never an
    argument (visible in ps) and never printed. Returns a process exit code.
    """
    from kairos_api.assistant_auth import _active_switch_label, _load_credentials_blob

    blob = _load_credentials_blob()
    if not blob:
        print("אין פרטי התחברות של Claude Code במחשב הזה; אין מה לדחוף.")
        return 3
    payload = build_payload(blob, _active_switch_label())
    if payload is None:
        print("הטוקן המקומי פג או חסר; Claude Code ירענן אותו בשימוש הבא ואז הדחיפה תתחדש.")
        return 3
    body = json.dumps(payload)
    base = ["aws", "secretsmanager", "--profile", profile, "--region", region]
    put = base + ["put-secret-value", "--secret-id", secret_id,
                  "--secret-string", "file:///dev/stdin",
                  "--query", "VersionId", "--output", "text"]
    result = subprocess.run(put, input=body, capture_output=True, text=True, timeout=60)
    if result.returncode != 0 and "ResourceNotFoundException" in result.stderr:
        create = base + ["create-secret", "--name", secret_id,
                         "--description", "Kai subscription bridge: short-lived access token pushed from the operator's machine",
                         "--secret-string", "file:///dev/stdin",
                         "--query", "ARN", "--output", "text"]
        result = subprocess.run(create, input=body, capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        print(f"הדחיפה נכשלה: {result.stderr.strip()[:200]}")
        return 1
    expires_in_min = max(0, (int(payload["expiresAt"]) - int(time.time() * 1000)) // 60000)
    account = payload.get("account") or "-"
    print(f"טוקן המנוי נדחף לגשר ({account}); תקף עוד כ-{expires_in_min} דקות. גרסה: {result.stdout.strip()}")
    return 0
