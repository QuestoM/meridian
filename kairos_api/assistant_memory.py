"""Per-operator thread memory for the assistant.

Each operator's exchanges with the assistant survive restarts: every
successful ask (streaming or not) appends one ``{question, answer, at,
batch_id, conversation_id}`` entry to a per-conversation JSON file under
``data/assistant/threads/<user>/`` with a per-user index
(kairos_api.assistant_conversations; relocatable via KAIROS_ASSISTANT_DATA_DIR,
like the rest of the action-plane state). The identity keystone: every path is
derived STRICTLY from the authenticated session username (``auth-disabled``
when auth is off), never from any client-supplied parameter, so no request can
ever read or clear another user's thread. The username is sanitized to a safe
filename (a changed name is suffixed with a hash of the original, so distinct
users can never collide onto one file and no path traversal or dotfile is
possible). Writes are atomic (temp file + os.replace) and each conversation is
pruned to the newest ``MAX_ENTRIES`` entries. GET returns one conversation's
entries newest last for direct rendering (the newest conversation when no
``conversation_id`` is passed, keeping the old client working); DELETE clears
only the caller's conversations and audits the clear. A legacy flat
``threads/<user>.json`` is migrated on first access into one conversation
(``legacy-<user>``) by the conversation store.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from kairos_api import assistant_actions

MAX_ENTRIES = 50
_ENTRY_KEYS = (
    "question", "answer", "at", "batch_id", "conversation_id", "sources",
    "tool_trace", "coverage", "elapsed_seconds", "context_disclosure",
)
_SAFE_CHARS_RE = re.compile(r"[^A-Za-z0-9._-]+")

# No prefix: kairos_api.assistant includes this router under /api/assistant.
router = APIRouter(tags=["assistant"])

# Re-entrant: the conversation store nests its calls under this same lock, so
# every thread read and write across both modules shares one lock domain.
_LOCK = threading.RLock()
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


def append_entry(username: str, question: str, answer: str, batch_id: str | None = None,
                 conversation_id: str | None = None,
                 metadata: dict[str, Any] | None = None) -> None:
    """Append one successful ask to the user's thread, pruned to the newest
    MAX_ENTRIES per conversation (the newest conversation is used, or a fresh
    one minted, when no conversation_id is passed). Never raises: a memory
    failure must not fail the ask itself."""
    from kairos_api import assistant_conversations

    try:
        with _LOCK:
            assistant_conversations.append_exchange(
                username, conversation_id, question, answer, batch_id, metadata)
    except Exception:  # noqa: BLE001 - memory is additive, the ask already succeeded
        logger.exception("assistant thread append failed for user %s", username)


@router.get("/thread")
def read_thread(request: Request, conversation_id: str | None = None) -> dict[str, Any]:
    """One conversation of the caller's own thread, newest last, capped at the
    newest MAX_ENTRIES. Without ``conversation_id`` the newest conversation is
    returned, keeping the old no-param client working.

    The user is the authenticated session username (``auth-disabled`` when auth
    is off); no parameter can select a different USER's thread, only which of
    the caller's own conversations to read.
    """
    from kairos_api import assistant_conversations

    username = assistant_actions._actor(request)
    with _LOCK:
        if conversation_id:
            if not assistant_conversations.exists(username, conversation_id):
                raise HTTPException(status_code=404,
                                    detail=f"no conversation {conversation_id!r}")
            selected: str | None = conversation_id
        else:
            selected = assistant_conversations.newest_id(username)
        entries = assistant_conversations.entries_for(username, selected) if selected else []
    return {"entries": entries[-MAX_ENTRIES:], "user": username, "conversation_id": selected}


@router.delete("/thread")
def clear_thread(request: Request) -> dict[str, Any]:
    """Clear ALL of ONLY the caller's conversations and audit the clear."""
    from kairos_api import assistant_conversations

    username = assistant_actions._actor(request)
    with _LOCK:
        removed = assistant_conversations.clear_all(username)
    assistant_actions.audit_append(
        "thread_clear", username, results={"entries_removed": removed}
    )
    return {"cleared": True, "entries_removed": removed, "user": username}
