"""Persistent user store plus in-process session state for the Kairos login system.

This module is intentionally stdlib-only so the seed script can import it
without pulling the engine or FastAPI. The HTTP surface, the enforcement rule
and the full lifecycle documentation live in kairos_api/auth.py.

Storage: KAIROS_AUTH_DIR (default <repo>/data/auth) holds users.json with
scrypt password records. The file is written atomically with mode 600 and the
directory is gitignored, so credentials never leave the deployment. Sessions
and the login rate limiter are plain in-process dictionaries, which is correct
for the single-process uvicorn deployment model (one client per deployment);
a multi-worker deployment would need a shared session store instead.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("kairos.auth")

ROOT = Path(__file__).resolve().parents[1]

ROLES = ("admin", "operator", "viewer")

COOKIE_NAME = "kairos_session"

SCRYPT_N = 2**14
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_DKLEN = 32
SCRYPT_MAXMEM = 64 * 1024 * 1024

SESSION_TTL_SECONDS = 12 * 3600
RATE_LIMIT_MAX_FAILURES = 5
RATE_LIMIT_WINDOW_SECONDS = 10 * 60
# Cap on distinct usernames tracked by the login rate limiter, so a spray of
# invented usernames cannot grow the in-process map without bound. When the cap
# is exceeded the entries whose newest failure is oldest are dropped first;
# real attack windows (recent failures) always survive.
MAX_TRACKED_FAILED_USERNAMES = 10_000

MIN_PASSWORD_LENGTH = 10

USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{2,31}$")

_LOCK = threading.RLock()
_SESSIONS: dict[str, dict[str, Any]] = {}
_FAILED_LOGINS: dict[str, list[float]] = {}
_DUMMY_RECORD: dict[str, Any] | None = None


class DuplicateUserError(Exception):
    """Raised when creating an account whose username is already taken."""


class UnknownUserError(Exception):
    """Raised when the referenced account does not exist."""


def _now() -> float:
    """Single clock source so tests can control session and limiter time."""
    return time.time()


# ---------------------------------------------------------------------------
# Store location and file IO
# ---------------------------------------------------------------------------

def auth_dir() -> Path:
    """Resolve the auth directory, honouring the KAIROS_AUTH_DIR env knob."""
    value = os.getenv("KAIROS_AUTH_DIR", "").strip()
    if not value:
        return ROOT / "data" / "auth"
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def users_path() -> Path:
    return auth_dir() / "users.json"


def store_initialized() -> bool:
    """The store exists once users.json does; enforcement keys off this."""
    return users_path().is_file()


def load_users() -> list[dict[str, Any]]:
    path = users_path()
    if not path.is_file():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    users = raw.get("users") if isinstance(raw, dict) else None
    if not isinstance(users, list):
        return []
    return [user for user in users if isinstance(user, dict)]


def save_users(users: list[dict[str, Any]]) -> None:
    """Atomic write (tmp file + os.replace) with owner-only permissions."""
    path = users_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"users": users}, ensure_ascii=False, indent=2), encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Password hashing (stdlib scrypt, constant-time compare)
# ---------------------------------------------------------------------------

def hash_password(password: str) -> dict[str, Any]:
    salt = secrets.token_bytes(32)
    digest = hashlib.scrypt(
        password.encode("utf-8"), salt=salt,
        n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, maxmem=SCRYPT_MAXMEM, dklen=SCRYPT_DKLEN,
    )
    return {
        "salt_hex": salt.hex(),
        "hash_hex": digest.hex(),
        "n": SCRYPT_N,
        "r": SCRYPT_R,
        "p": SCRYPT_P,
    }


def verify_password(password: str, record: dict[str, Any]) -> bool:
    try:
        salt = bytes.fromhex(str(record["salt_hex"]))
        expected = bytes.fromhex(str(record["hash_hex"]))
        digest = hashlib.scrypt(
            password.encode("utf-8"), salt=salt,
            n=int(record["n"]), r=int(record["r"]), p=int(record["p"]),
            maxmem=SCRYPT_MAXMEM, dklen=len(expected),
        )
    except (KeyError, TypeError, ValueError):
        return False
    return hmac.compare_digest(digest, expected)


def burn_password_check(password: str) -> None:
    """Verify against a throwaway record so unknown usernames cost the same
    time as a real password check (no account enumeration via timing)."""
    global _DUMMY_RECORD
    if _DUMMY_RECORD is None:
        _DUMMY_RECORD = hash_password(secrets.token_urlsafe(16))
    verify_password(password, _DUMMY_RECORD)


# ---------------------------------------------------------------------------
# User records
# ---------------------------------------------------------------------------

def normalize_username(value: str) -> str:
    return str(value or "").strip().lower()


def get_user(username: str) -> dict[str, Any] | None:
    username = normalize_username(username)
    for user in load_users():
        if user.get("username") == username:
            return user
    return None


def _require_password(password: str) -> None:
    if not isinstance(password, str) or len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"The password must be at least {MIN_PASSWORD_LENGTH} characters long."
        )


def add_user(
    username: str,
    password: str,
    role: str,
    display_name: str = "",
    must_change_password: bool = False,
) -> dict[str, Any]:
    username = normalize_username(username)
    if not USERNAME_RE.match(username):
        raise ValueError(
            "The username must be 3 to 32 characters: lowercase letters, digits, dot, dash or underscore."
        )
    if role not in ROLES:
        raise ValueError(f"The role must be one of: {', '.join(ROLES)}.")
    _require_password(password)
    record = {
        "username": username,
        "password_scrypt": hash_password(password),
        "role": role,
        "display_name": str(display_name or "").strip() or username,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "must_change_password": bool(must_change_password),
    }
    with _LOCK:
        users = load_users()
        if any(user.get("username") == username for user in users):
            raise DuplicateUserError(f"The username {username} is already taken.")
        users.append(record)
        save_users(users)
    return record


def set_password(username: str, new_password: str, must_change_password: bool) -> dict[str, Any]:
    username = normalize_username(username)
    _require_password(new_password)
    hashed = hash_password(new_password)
    with _LOCK:
        users = load_users()
        for user in users:
            if user.get("username") == username:
                user["password_scrypt"] = hashed
                user["must_change_password"] = bool(must_change_password)
                save_users(users)
                return user
    raise UnknownUserError(f"No account named {username}.")


def remove_user(username: str) -> None:
    username = normalize_username(username)
    with _LOCK:
        users = load_users()
        remaining = [user for user in users if user.get("username") != username]
        if len(remaining) == len(users):
            raise UnknownUserError(f"No account named {username}.")
        save_users(remaining)


def admin_count() -> int:
    return sum(1 for user in load_users() if user.get("role") == "admin")


# ---------------------------------------------------------------------------
# Sessions (opaque tokens, 12h sliding expiry, in-process)
# ---------------------------------------------------------------------------

def create_session(username: str, role: str) -> str:
    token = secrets.token_urlsafe(32)
    with _LOCK:
        _SESSIONS[token] = {
            "username": username,
            "role": role,
            "expires": _now() + SESSION_TTL_SECONDS,
        }
    return token


def resolve_session(token: str | None) -> dict[str, Any] | None:
    """Return {username, role} for a live token and slide its expiry forward.
    Expired tokens are pruned lazily on every resolve."""
    if not token:
        return None
    with _LOCK:
        now = _now()
        for stale in [key for key, sess in _SESSIONS.items() if sess["expires"] <= now]:
            del _SESSIONS[stale]
        session = _SESSIONS.get(token)
        if session is None:
            return None
        session["expires"] = now + SESSION_TTL_SECONDS
        return {"username": session["username"], "role": session["role"]}


def drop_session(token: str | None) -> None:
    if not token:
        return
    with _LOCK:
        _SESSIONS.pop(token, None)


def drop_sessions_for(username: str) -> None:
    username = normalize_username(username)
    with _LOCK:
        for token in [t for t, sess in _SESSIONS.items() if sess["username"] == username]:
            del _SESSIONS[token]


# ---------------------------------------------------------------------------
# Login rate limiting (5 failures per username per 10 minutes)
# ---------------------------------------------------------------------------

def login_retry_after(username: str) -> int:
    """Seconds until another attempt is allowed, or 0 when not limited."""
    username = normalize_username(username)
    with _LOCK:
        now = _now()
        stamps = [s for s in _FAILED_LOGINS.get(username, []) if s > now - RATE_LIMIT_WINDOW_SECONDS]
        if stamps:
            _FAILED_LOGINS[username] = stamps
        else:
            _FAILED_LOGINS.pop(username, None)
        if len(stamps) < RATE_LIMIT_MAX_FAILURES:
            return 0
        freeing = sorted(stamps)[len(stamps) - RATE_LIMIT_MAX_FAILURES]
        return max(1, int(freeing + RATE_LIMIT_WINDOW_SECONDS - now) + 1)


def record_login_failure(username: str) -> None:
    username = normalize_username(username)
    with _LOCK:
        _FAILED_LOGINS.setdefault(username, []).append(_now())
        overflow = len(_FAILED_LOGINS) - MAX_TRACKED_FAILED_USERNAMES
        if overflow > 0:
            # Prune the usernames whose newest failure is oldest; the entry just
            # touched carries the newest stamp, so it always survives.
            for stale in sorted(_FAILED_LOGINS, key=lambda name: max(_FAILED_LOGINS[name]))[:overflow]:
                del _FAILED_LOGINS[stale]


def clear_login_failures(username: str) -> None:
    with _LOCK:
        _FAILED_LOGINS.pop(normalize_username(username), None)


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def seed_initial_admin(password: str | None = None) -> tuple[str, str | None]:
    """Create the first admin account. Returns (username, generated_password).

    Password precedence: explicit argument, then KAIROS_ADMIN_PASSWORD, then a
    generated one-time secrets.token_urlsafe(12). Only the generated path
    writes initial_admin_password.txt (mode 600) and marks the account
    must_change_password, so a deliberately configured password is never
    persisted in plaintext anywhere.
    """
    if load_users():
        raise RuntimeError(f"The login store at {users_path()} already has accounts.")
    chosen = password if password is not None else (os.getenv("KAIROS_ADMIN_PASSWORD") or None)
    generated: str | None = None
    if chosen is None:
        generated = secrets.token_urlsafe(12)
        chosen = generated
    add_user("admin", chosen, "admin", "Admin", must_change_password=bool(generated))
    if generated:
        marker = auth_dir() / "initial_admin_password.txt"
        marker.write_text(generated + "\n", encoding="utf-8")
        os.chmod(marker, 0o600)
        logger.warning(
            "Seeded the admin account with a generated one-time password, written once to %s. "
            "Sign in as admin and change it now.",
            marker,
        )
    else:
        logger.info("Seeded the admin account from the configured password.")
    return "admin", generated


def reset_runtime_state() -> None:
    """Clear in-process sessions and rate-limit counters (used by tests)."""
    with _LOCK:
        _SESSIONS.clear()
        _FAILED_LOGINS.clear()
