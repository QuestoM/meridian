"""Per-operator thread memory for the assistant.

Each operator's conversation with the assistant survives restarts: every
successful ask (streaming or not) appends one ``{question, answer, at,
batch_id}`` entry to a per-user JSON file under ``data/assistant/threads/``
(relocatable via KAIROS_ASSISTANT_DATA_DIR, like the rest of the action-plane
state). The identity keystone: the file is derived STRICTLY from the
authenticated session username (``auth-disabled`` when auth is off), never from
any client-supplied parameter, so no request can ever read or clear another
user's thread. The username is sanitized to a safe filename (a changed name is
suffixed with a hash of the original, so distinct users can never collide onto
one file and no path traversal or dotfile is possible). Writes are atomic
(temp file + os.replace) and each thread is pruned to the newest
``MAX_ENTRIES`` entries. GET returns the entries newest last for direct
rendering; DELETE clears only the caller's thread and audits the clear.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request

from kairos_api import assistant_actions

MAX_ENTRIES = 50
_ENTRY_KEYS = ("question", "answer", "at", "batch_id")
_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")

# No prefix: kairos_api.assistant includes this router under /api/assistant.
router = APIRouter(tags=["assistant"])

_LOCK = threading.Lock()
logger = logging.getLogger(__name__)


def _threads_dir() -> Path:
    return assistant_actions._data_dir() / "threads"


def _sanitize_username(username: str) -> str:
    """A filesystem-safe, collision-resistant name for one username.

    Characters outside [A-Za-z0-9._-] become underscores and leading or
    trailing dots are stripped (no traversal, no dotfiles). Whenever the
    sanitized form differs from the original, a short hash of the ORIGINAL is
    appended so two distinct usernames can never share a thread file.
    """
    raw = str(username or "").strip() or "anonymous"
    safe = _SAFE_CHARS_RE.sub("_", raw)
    safe = re.sub(r"\.{2,}", "_", safe).strip(".") or "user"
    if safe != raw:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:8]
        safe = f"{safe[:40]}-{digest}"
    return safe[:64]


def _path_for(username: str) -> Path:
    return _threads_dir() / f"{_sanitize_username(username)}.json"


def _load_entries(path: Path) -> list[dict[str, Any]]:
    """The stored entries of one thread file, oldest first. A missing file is
    an empty thread; an unreadable one is logged and treated as empty rather
    than crashing every ask that follows."""
    if not path.exists():
        return []
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception("assistant thread file %s is unreadable; starting fresh", path)
        return []
    entries = store.get("entries") if isinstance(store, dict) else None
    if not isinstance(entries, list):
        return []
    return [
        {key: entry.get(key) for key in _ENTRY_KEYS}
        for entry in entries
        if isinstance(entry, dict)
    ]


def _write_atomic(path: Path, username: str, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"user": username, "entries": entries}, ensure_ascii=False, indent=1, default=str
    )
    temp = path.with_suffix(".json.tmp")
    temp.write_text(payload, encoding="utf-8")
    os.replace(temp, path)


def append_entry(username: str, question: str, answer: str, batch_id: str | None = None) -> None:
    """Append one successful ask to the user's thread, pruned to the newest
    MAX_ENTRIES. Never raises: a memory failure must not fail the ask itself."""
    entry = {
        "question": question,
        "answer": answer,
        "at": datetime.now(timezone.utc).isoformat(),
        "batch_id": batch_id,
    }
    try:
        with _LOCK:
            path = _path_for(username)
            entries = _load_entries(path)
            entries.append(entry)
            _write_atomic(path, username, entries[-MAX_ENTRIES:])
    except Exception:  # noqa: BLE001 - memory is additive, the ask already succeeded
        logger.exception("assistant thread append failed for user %s", username)


@router.get("/thread")
def read_thread(request: Request) -> dict[str, Any]:
    """The caller's own thread, newest last, capped at the newest MAX_ENTRIES.

    The user is the authenticated session username (``auth-disabled`` when auth
    is off); no parameter can select a different thread.
    """
    username = assistant_actions._actor(request)
    with _LOCK:
        entries = _load_entries(_path_for(username))
    return {"entries": entries[-MAX_ENTRIES:], "user": username}


@router.delete("/thread")
def clear_thread(request: Request) -> dict[str, Any]:
    """Clear ONLY the caller's thread and audit the clear."""
    username = assistant_actions._actor(request)
    with _LOCK:
        path = _path_for(username)
        removed = len(_load_entries(path))
        if path.exists():
            path.unlink()
    assistant_actions.audit_append(
        "thread_clear", username, results={"entries_removed": removed}
    )
    return {"cleared": True, "entries_removed": removed, "user": username}
