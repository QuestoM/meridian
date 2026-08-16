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

# Re-exported so every caller keeps reaching the password helpers through this
# module, exactly as before the split (auth.py calls store.verify_password and
# store.burn_password_check).
from kairos_api.auth_store_password import (  # noqa: F401
    SCRYPT_DKLEN,
    SCRYPT_MAXMEM,
    SCRYPT_N,
    SCRYPT_P,
    SCRYPT_R,
    burn_password_check,
    hash_password,
    verify_password,
)

logger = logging.getLogger("kairos.auth")

ROOT = Path(__file__).resolve().parents[1]

ROLES = ("admin", "operator", "viewer")

# Company staff manage everything; channel-affiliated accounts are walled off
# the event-management surface (calendar event writes and the event pricing
# activation switch). A missing or unrecognized value reads as ``unknown`` and
# therefore fails closed; an administrator can resolve that legacy record using
# the existing affiliation endpoint without resetting the account or session.
AFFILIATIONS = ("company", "channel")
UNKNOWN_AFFILIATION = "unknown"

# What this person's work is, which decides the view they land on and the order
# of their sidebar. It is not a permission: role and affiliation decide those,
# so a misconfigured job costs somebody a good first screen and never their
# access. Thirteen door-bearing roles plus the unset default that every
# existing record reads as, which is what makes the field safe to add. The
# same thirteen ids and the door each one opens live in
# tv-break-dashboard/src/session.js, and a test pins the two lists together.
JOBS = (
    "general_manager",
    "planner",
    "scheduler",
    "traffic_operator",
    "programming_representative",
    "compliance_owner",
    "yield_owner",
    "account_manager",
    "campaign_manager",
    "analyst",
    "data_steward",
    "account_administrator",
    "model_steward",
)
UNSET_JOB = "unset"

COOKIE_NAME = "kairos_session"

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
# User records
# ---------------------------------------------------------------------------

def normalize_username(value: str) -> str:
    return str(value or "").strip().lower()


def normalize_affiliation(value: Any) -> str:
    """Return a stored affiliation, failing closed for legacy/invalid records.

    ``unknown`` is a read-only migration state, not a value new or updated
    accounts may store.  This keeps old accounts usable for sign-in while
    withholding company-only data until an administrator makes the missing
    authorization decision explicitly.
    """
    text = str(value or "").strip().lower()
    return text if text in AFFILIATIONS else UNKNOWN_AFFILIATION


def normalize_job(value: Any) -> str:
    """Missing, empty or unrecognized values read as unset, which is the safe
    default: an unset job lands on Today with the picker, never on somebody
    else's first screen."""
    text = str(value or "").strip().lower()
    return text if text in JOBS else UNSET_JOB


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
    affiliation: str = "company",
    job: str = UNSET_JOB,
) -> dict[str, Any]:
    username = normalize_username(username)
    if not USERNAME_RE.match(username):
        raise ValueError(
            "The username must be 3 to 32 characters: lowercase letters, digits, dot, dash or underscore."
        )
    if role not in ROLES:
        raise ValueError(f"The role must be one of: {', '.join(ROLES)}.")
    affiliation = str(affiliation or "company").strip().lower()
    if affiliation not in AFFILIATIONS:
        raise ValueError(f"The affiliation must be one of: {', '.join(AFFILIATIONS)}.")
    _require_password(password)
    record = {
        "username": username,
        "password_scrypt": hash_password(password),
        "role": role,
        "display_name": str(display_name or "").strip() or username,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "must_change_password": bool(must_change_password),
        "affiliation": affiliation,
        "job": normalize_job(job),
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


def set_affiliation(username: str, affiliation: str) -> dict[str, Any]:
    affiliation = str(affiliation or "").strip().lower()
    if affiliation not in AFFILIATIONS:
        raise ValueError(f"The affiliation must be one of: {', '.join(AFFILIATIONS)}.")
    username = normalize_username(username)
    with _LOCK:
        users = load_users()
        for user in users:
            if user.get("username") == username:
                user["affiliation"] = affiliation
                save_users(users)
                return user
    raise UnknownUserError(f"No account named {username}.")


def set_job(username: str, job: str) -> dict[str, Any]:
    """Record which job this account does. Never touches role or affiliation."""
    normalized = normalize_job(job)
    if normalized == UNSET_JOB and str(job or "").strip().lower() != UNSET_JOB:
        raise ValueError(f"The job must be one of: {', '.join(JOBS)}, or {UNSET_JOB}.")
    username = normalize_username(username)
    with _LOCK:
        users = load_users()
        for user in users:
            if user.get("username") == username:
                user["job"] = normalized
                save_users(users)
                return user
    raise UnknownUserError(f"No account named {username}.")


def is_company_user(username: str) -> bool:
    """Whether this account may manage the company-only surfaces (calendar
    events and the event pricing activation switch).

    True only when the stored affiliation is explicitly company, and whenever
    auth is off (bypass env or an uninitialized store), so a deployment without
    login keeps full access. Missing, empty and malformed legacy values are an
    ``unknown`` migration state and fail closed.
    With auth on, an unknown username reads False: never grant the company
    surface to an identity the store cannot vouch for.
    """
    auth_off = (
        os.getenv("KAIROS_AUTH_DISABLED", "").strip().lower() in {"1", "true", "yes"}
        or not store_initialized()
    )
    if auth_off:
        return True
    user = get_user(username)
    if user is None:
        return False
    return normalize_affiliation(user.get("affiliation")) == "company"


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

def _write_initial_password_marker(password: str) -> Path:
    """Publish the generated bootstrap password atomically with mode 600.

    ``users.json`` is what turns authentication enforcement on.  The marker is
    therefore published first and the user store last: if password delivery
    fails, the deployment must remain in setup-required state rather than gain
    an administrator whose only password was never delivered.
    """
    marker = auth_dir() / "initial_admin_password.txt"
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_name(marker.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(password + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, marker)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return marker


def seed_initial_admin(password: str | None = None) -> tuple[str, str | None]:
    """Create the first admin account. Returns (username, generated_password).

    Password precedence: explicit argument, then KAIROS_ADMIN_PASSWORD, then a
    generated one-time secrets.token_urlsafe(12). Only the generated path
    writes initial_admin_password.txt (mode 600) and marks the account
    must_change_password, so a deliberately configured password is never
    persisted in plaintext anywhere.
    """
    # One lock covers existence check, password delivery and account creation.
    # Apart from closing the local interleaving, this matters when startup and a
    # setup action share the same process: a losing seed must not overwrite the
    # marker belonging to the account that won.
    with _LOCK:
        if load_users():
            raise RuntimeError(f"The login store at {users_path()} already has accounts.")
        chosen = password if password is not None else (os.getenv("KAIROS_ADMIN_PASSWORD") or None)
        generated: str | None = None
        marker: Path | None = None
        if chosen is None:
            generated = secrets.token_urlsafe(12)
            chosen = generated
            # Password delivery is the prepare step.  Only after it succeeds may
            # add_user atomically publish users.json and activate enforcement.
            marker = _write_initial_password_marker(generated)
        try:
            add_user("admin", chosen, "admin", "Admin", must_change_password=bool(generated))
        except BaseException:
            # Account publication failed, so the delivered password names no
            # account.  Remove it when possible; even if cleanup itself fails,
            # users.json is absent and the API remains setup-required.
            if marker is not None:
                try:
                    marker.unlink(missing_ok=True)
                except OSError:
                    logger.warning(
                        "Admin creation failed after password delivery; stale marker remains at %s.",
                        marker,
                        exc_info=True,
                    )
            raise
        if marker is not None:
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
