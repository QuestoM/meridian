"""Append-only, permission-scoped activity log for the Kairos API.

What lands here: every MUTATING /api request (POST/PUT/PATCH/DELETE) is
recorded by the middleware hook registered in server.py with the resolved
session identity, the final status code and the duration, plus the dedicated
login / login_failed / logout events appended by the hooks in
kairos_api.auth. GET traffic is deliberately not recorded: the owner's
question is "who changed what", and read noise would bury the answer.

Hard security rule: entries are METADATA ONLY. The logger never sees, let
alone stores, request or response bodies, query strings, headers, cookies,
tokens or passwords. The login and logout requests, whose bodies carry
credentials, are excluded from the middleware entirely and recorded as
body-free events from the auth module instead, so credentials never transit
this code path at all.

Storage is a single JSONL file at data/audit/activity.jsonl (gitignored
runtime state; KAIROS_AUDIT_DIR relocates it, which the tests use). Appends
happen under an in-process lock, correct for the single-process uvicorn
deployment, the same model as the auth session store. When the file grows
past PRUNE_TRIGGER lines it is atomically rewritten (tmp file + os.replace)
keeping only the newest MAX_KEPT_ENTRIES entries. That bound, the stamp on the
oldest surviving line and the number of lines held are all published (RETENTION,
first_entry_ts, entry_count), so a surface reading this record can say how far
back it reaches and how large it is instead of reporting a pruned day as a day
on which nothing happened, or a reader's own slice as the size of the store.

Visibility (GET /api/activity-log): the admin role sees every entry and may
narrow the view with ?user=<name>; any other signed-in role sees only its own
entries and may not filter, so no parameter can widen scope; anonymous
requests are refused (the auth middleware walls them first anyway). When auth
is disabled or not yet seeded there is no identity to scope by, so the
response says scope "all" honestly (single-tenant dev mode).
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from kairos_api import auth_store, history_api_actions
from kairos_api.auth import auth_active

logger = logging.getLogger("kairos.activity")

ROOT = Path(__file__).resolve().parents[1]

LOG_FILENAME = "activity.jsonl"

# Prune keeps the newest MAX_KEPT_ENTRIES once the file exceeds PRUNE_TRIGGER
# lines, so steady-state appends stay cheap and the file stays bounded.
MAX_KEPT_ENTRIES = 5000
PRUNE_TRIGGER = 6000

# The same two numbers, published as a fact a reader can be shown. A bounded file
# means the oldest days are dropped while nobody is looking, and a surface that
# reports "nothing was recorded on that day" without saying so is wrong on a
# destination whose whole job is evidence. Measured on this deployment on
# 2026-08-01: 5,227 lines held, the oldest stamped five hours before the read.
RETENTION = {"pruned": True, "keeps": MAX_KEPT_ENTRIES, "prune_at": PRUNE_TRIGGER, "unit": "lines"}

MUTATING_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# Login and logout bodies carry credentials, so those two requests are never
# routed through the request recorder; kairos_api.auth appends dedicated
# body-free events for them instead.
EXCLUDED_PATHS = frozenset({"/api/auth/login", "/api/auth/logout"})

_LOCK = threading.Lock()
# Cached line count so appends do not re-scan the file. Keyed by the resolved
# path because KAIROS_AUDIT_DIR changes between tests.
_line_count: int | None = None
_counted_path: Path | None = None


def audit_dir() -> Path:
    """Resolve the audit directory, honouring the KAIROS_AUDIT_DIR env knob."""
    value = os.getenv("KAIROS_AUDIT_DIR", "").strip()
    if not value:
        return ROOT / "data" / "audit"
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def log_path() -> Path:
    return audit_dir() / LOG_FILENAME


def reset_runtime_state() -> None:
    """Forget the cached line count (used by tests when the store moves)."""
    global _line_count, _counted_path
    with _LOCK:
        _line_count = None
        _counted_path = None


# ---------------------------------------------------------------------------
# Entry construction and the append/prune store
# ---------------------------------------------------------------------------

def _entry(
    user: str,
    role: str,
    event: str,
    method: str | None,
    path: str | None,
    status: int | None,
    duration_ms: float | None,
) -> dict[str, Any]:
    """One log entry. This is the complete schema: metadata only, and only
    these fields, so nothing sensitive can ride along by accident."""
    return {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "user": user,
        "role": role,
        "event": event,
        "method": method,
        "path": path,
        "status": status,
        "duration_ms": duration_ms,
        "via": "assistant" if (path or "").startswith("/api/assistant") else "dashboard",
    }


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _prune_locked(path: Path) -> int:
    """Rewrite the file atomically keeping the newest MAX_KEPT_ENTRIES lines.
    Callers hold _LOCK. os.replace keeps concurrent readers consistent."""
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = lines[-MAX_KEPT_ENTRIES:]
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)
    return len(kept)


def _append_entry(entry: dict[str, Any]) -> None:
    """Append one entry as a single JSON line under the store lock.

    The single write of one full line plus the in-process lock makes appends
    atomic for the single-process deployment. May raise on IO failure; every
    caller (the middleware and the auth hooks) treats that as log-and-continue
    so a logging problem can never fail the request being logged.
    """
    global _line_count, _counted_path
    line = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
    with _LOCK:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        existed = path.is_file()
        if _counted_path != path or _line_count is None:
            _line_count = _count_lines(path)
            _counted_path = path
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        if not existed:
            os.chmod(path, 0o600)
        _line_count += 1
        if _line_count > PRUNE_TRIGGER:
            _line_count = _prune_locked(path)


def first_entry_ts() -> str | None:
    """The stamp on the oldest line still in the file, or None when it holds none.

    How far back this record reaches is a fact about the store and not about the
    account reading it, so it is taken from the file itself rather than from any
    scoped slice of it. Pruning moves it: the file keeps the newest
    MAX_KEPT_ENTRIES lines and drops the rest, so a day older than this stamp
    carries no evidence here either way, and a screen that answers "nothing was
    recorded" for such a day has answered a question it cannot answer.

    Read line by line and stopped at the first parseable one, so the cost does
    not grow with the file. A malformed head is skipped rather than returned.
    """
    with _LOCK:
        path = log_path()
        if not path.is_file():
            return None
        with path.open("r", encoding="utf-8") as handle:
            for raw in handle:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(parsed, dict) and parsed.get("ts"):
                    return str(parsed["ts"])
    return None


def entry_count() -> int:
    """How many lines this store holds, whoever happens to be reading it.

    Published beside :func:`first_entry_ts` and for the same reason. How far the
    evidence goes and how much of it there is are both facts about the store, so
    both are read from the file rather than from any caller's scoped slice of it.

    Measured live on 2026-08-01 before this existed, against a file holding 5,261
    lines: an operator's read reported 2 and a viewer's reported 4, each printed
    under the store's own name and directly beside a sentence saying the recorder
    keeps the newest 5,000. A store that keeps 5,000 and holds 2 is a
    contradiction, and the figure that was wrong was the one whose whole job is
    to say how large the evidence is.

    A blank line is not counted, because a blank line is no record. A line that
    does not parse is counted, because it still occupies one of the
    MAX_KEPT_ENTRIES this figure is printed beside. The pruner's own counter
    counts physical lines instead, which is right for a trigger and would be
    wrong here.
    """
    with _LOCK:
        path = log_path()
        if not path.is_file():
            return 0
        with path.open("r", encoding="utf-8") as handle:
            return sum(1 for raw in handle if raw.strip())


def _read_entries() -> list[dict[str, Any]]:
    """All parseable entries in file order (oldest first). Malformed lines are
    skipped instead of failing the whole read."""
    with _LOCK:
        path = log_path()
        if not path.is_file():
            return []
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    entries: list[dict[str, Any]] = []
    for raw in raw_lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            parsed = json.loads(raw)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


# ---------------------------------------------------------------------------
# Recorders: the middleware path and the auth event hooks
# ---------------------------------------------------------------------------

def _request_identity(request: Request) -> tuple[str, str]:
    """Resolve who is acting, without ever enforcing anything here.

    "auth-disabled" when there is no login wall (bypassed or unseeded store),
    "anonymous" when auth is on but the request carries no live session.
    """
    if not auth_active():
        return "auth-disabled", ""
    session = auth_store.resolve_session(request.cookies.get(auth_store.COOKIE_NAME))
    if session is None:
        return "anonymous", ""
    return str(session.get("username") or ""), str(session.get("role") or "")


def record_request(request: Request, status: int, duration_ms: float) -> None:
    """Append the metadata of one mutating API request. May raise; the
    middleware wraps this in its own log-and-continue guard."""
    user, role = _request_identity(request)
    _append_entry(
        _entry(user, role, "request", request.method, request.url.path, int(status), round(duration_ms, 1))
    )


def record_auth_event(event: str, user: str, role: str = "") -> None:
    """Append a login / login_failed / logout event. Only the event name, the
    username and the role are stored, never any credential material. May
    raise; the auth hook wraps the call in its own guard."""
    _append_entry(_entry(str(user or ""), str(role or ""), str(event), None, None, None, None))


async def record_api_mutation(request: Request, call_next):
    """The middleware body registered in server.py.

    Only mutating /api requests are considered, login/logout are excluded (see
    EXCLUDED_PATHS), and a recording failure is logged and swallowed so it can
    never fail or slow the request it observes. A crash below us is recorded
    as a 500 and re-raised for the normal error handling.
    """
    path = request.url.path
    if (
        request.method not in MUTATING_METHODS
        or not path.startswith("/api/")
        or path in EXCLUDED_PATHS
    ):
        return await call_next(request)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        _record_request_safely(request, 500, started)
        raise
    _record_request_safely(request, response.status_code, started)
    return response


def _record_request_safely(request: Request, status: int, started: float) -> None:
    try:
        record_request(request, status, (time.perf_counter() - started) * 1000.0)
    except Exception:
        logger.exception("activity log append failed; the request itself is unaffected")


# ---------------------------------------------------------------------------
# Read API: permission-scoped visibility
# ---------------------------------------------------------------------------

router = APIRouter(tags=["activity-log"])


def visibility_scope(request: Request) -> tuple[str, str | None]:
    """Who this caller may see in the log: ``("all", None)`` or ``("self", name)``.

    The one implementation of the rule, so History and the log itself cannot
    drift: admin sees every entry, any other signed-in role sees only its own,
    an anonymous caller is refused, and a deployment without a login wall has no
    identity to scope by and honestly says "all".
    """
    if not auth_active():
        return "all", None
    session = auth_store.resolve_session(request.cookies.get(auth_store.COOKIE_NAME))
    if session is None:
        raise HTTPException(status_code=401, detail="A signed-in session is required.")
    if session.get("role") == "admin":
        return "all", None
    return "self", str(session.get("username") or "")


@router.get("/api/activity-log")
def get_activity_log(request: Request, limit: int = 100, user: str | None = None) -> dict[str, Any]:
    """Newest-first activity entries, scoped by the caller's role.

    Admin sees every entry and may narrow with ?user=<name>. Any other role
    sees only entries whose user matches its own session username, and the
    user filter is refused so no parameter can widen scope. Anonymous callers
    get 401. With auth disabled or unseeded the scope is honestly "all".
    """
    filter_user = (user or "").strip() or None
    scope, self_user = visibility_scope(request)

    entries = _read_entries()
    entries.reverse()  # newest first
    if scope == "self":
        if filter_user is not None:
            raise HTTPException(status_code=403, detail="Only the admin role can filter by user.")
        entries = [entry for entry in entries if entry.get("user") == self_user]
    elif filter_user is not None:
        entries = [entry for entry in entries if entry.get("user") == filter_user]

    capped = max(1, min(int(limit), 1000))
    return {"entries": [_with_action(entry) for entry in entries[:capped]], "scope": scope}


def _with_action(entry: dict[str, Any]) -> dict[str, Any]:
    """The stored entry plus the action code the surface renders a word for,
    whether that act saved anything, and whether it landed.

    All three are derived on the read and never stored, so the file schema is
    exactly the nine metadata fields it has always been. They exist so no
    surface has to match on an HTTP path to name what happened: the
    classification lives in one place and both History and the settings log read
    the same one.

    ``outcome`` is the newest of the three and it closes a measured defect this
    panel shared with History: the status code was on every line and nothing
    read it, so a refused write rendered under the word for the act it attempted.
    """
    read = dict(entry)
    action = history_api_actions.action_for(entry.get("method"), entry.get("path"))
    read["action"] = action
    read["saved"] = history_api_actions.kind_for(action) != "preview"
    read["outcome"] = history_api_actions.outcome_for(entry.get("status"))
    return read
