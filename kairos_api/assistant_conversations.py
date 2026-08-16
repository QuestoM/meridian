"""Per-conversation thread store for the assistant.

A conversation is a named sequence of exchanges with a server-minted id.
Storage is one JSON file per conversation under
``data/assistant/threads/<user>/<conversation_id>.json`` plus a small per-user
``index.json`` carrying id, title, created_at, updated_at and entry_count
(relocatable via KAIROS_ASSISTANT_DATA_DIR like the rest of the action-plane
state). Identity stays the keystone: every path is derived STRICTLY from the
authenticated session username through the same sanitizer the flat store used,
so no request can ever read or clear another user's conversations. All writes
are atomic (temp file + os.replace) under the shared thread lock
(``assistant_memory._LOCK``, re-entrant), entries per conversation are pruned
to the newest ``assistant_memory.MAX_ENTRIES`` and the index is pruned to the
newest ``MAX_CONVERSATIONS`` conversations, oldest removed with their files.

Migration: on first access of a user whose legacy flat ``threads/<user>.json``
still exists, its entries are wrapped as one conversation titled
``LEGACY_TITLE`` with id ``legacy-<user>``; the new files are written first and
the flat file is deleted only after they exist, so no crash can lose entries.
"""

from __future__ import annotations

import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kairos_api import assistant_memory

MAX_CONVERSATIONS = 30
TITLE_MAX_CHARS = 60
LEGACY_TITLE = "שיחה קודמת"
DEFAULT_TITLE = "שיחה חדשה"
# Minted ids are uuid4().hex[:12]; the migration mints legacy-<sanitized user>.
# Nothing outside this shape ever becomes a file name (no traversal, no dotfile).
_CONVERSATION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,79}$")

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def valid_id(conversation_id: str | None) -> bool:
    return bool(conversation_id) and bool(_CONVERSATION_ID_RE.fullmatch(str(conversation_id)))


def _user_dir(username: str) -> Path:
    return assistant_memory._threads_dir() / assistant_memory._sanitize_username(username)


def _index_path(username: str) -> Path:
    return _user_dir(username) / "index.json"


def _conversation_path(username: str, conversation_id: str) -> Path:
    if not valid_id(conversation_id):
        raise ValueError(f"invalid conversation id {conversation_id!r}")
    return _user_dir(username) / f"{conversation_id}.json"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str),
                   encoding="utf-8")
    os.replace(tmp, path)


def _read_index(username: str) -> list[dict[str, Any]]:
    """The user's conversation records, stored order. Runs the legacy migration
    first; an unreadable index is logged and treated as empty, never a crash."""
    _migrate_legacy(username)
    path = _index_path(username)
    if not path.exists():
        return []
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.exception("assistant conversation index %s is unreadable; starting fresh", path)
        return []
    records = store.get("conversations") if isinstance(store, dict) else None
    if not isinstance(records, list):
        return []
    return [dict(record) for record in records if isinstance(record, dict) and record.get("id")]


def _write_index(username: str, records: list[dict[str, Any]]) -> None:
    _write_json(_index_path(username), {"user": username, "conversations": records})


def _migrate_legacy(username: str) -> None:
    """Wrap a legacy flat thread file as one conversation, atomically.

    The conversation file and the index are written FIRST; the flat file is
    deleted only after both exist, so a crash mid-migration loses nothing and
    the next access simply retries. Never raises: a migration failure leaves
    the flat file in place and is logged.
    """
    flat = assistant_memory._path_for(username)
    if not flat.exists():
        return
    try:
        entries = assistant_memory._load_entries(flat)
        conversation_id = f"legacy-{assistant_memory._sanitize_username(username)}"
        for entry in entries:
            entry["conversation_id"] = conversation_id
        created = str(entries[0].get("at") or _now_iso()) if entries else _now_iso()
        updated = str(entries[-1].get("at") or _now_iso()) if entries else _now_iso()
        _write_json(_conversation_path(username, conversation_id),
                    {"user": username, "conversation_id": conversation_id, "entries": entries})
        records = _read_index_raw(username)
        if not any(record.get("id") == conversation_id for record in records):
            records.append({"id": conversation_id, "title": LEGACY_TITLE, "created_at": created,
                            "updated_at": updated, "entry_count": len(entries)})
            _write_index(username, records)
        flat.unlink()
    except Exception:  # noqa: BLE001 - keep the flat file rather than lose entries
        logger.exception("assistant legacy thread migration failed for user %s", username)


def _read_index_raw(username: str) -> list[dict[str, Any]]:
    """The index without triggering migration (used BY the migration)."""
    path = _index_path(username)
    if not path.exists():
        return []
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    records = store.get("conversations") if isinstance(store, dict) else None
    return [dict(r) for r in records if isinstance(r, dict) and r.get("id")] if isinstance(records, list) else []


def _public(record: dict[str, Any]) -> dict[str, Any]:
    return {key: record.get(key)
            for key in ("id", "title", "created_at", "updated_at", "entry_count")}


def list_records(username: str) -> list[dict[str, Any]]:
    """The user's conversations, newest updated first."""
    with assistant_memory._LOCK:
        records = _read_index(username)
    records.sort(key=lambda record: str(record.get("updated_at", "")), reverse=True)
    return [_public(record) for record in records]


def _find(records: list[dict[str, Any]], conversation_id: str) -> dict[str, Any] | None:
    return next((record for record in records if record.get("id") == conversation_id), None)


def exists(username: str, conversation_id: str) -> bool:
    if not valid_id(conversation_id):
        return False
    with assistant_memory._LOCK:
        return _find(_read_index(username), conversation_id) is not None


def newest_id(username: str) -> str | None:
    records = list_records(username)
    return str(records[0]["id"]) if records else None


def _prune(username: str, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the newest MAX_CONVERSATIONS by updated_at; delete pruned files."""
    if len(records) <= MAX_CONVERSATIONS:
        return records
    ordered = sorted(records, key=lambda record: str(record.get("updated_at", "")), reverse=True)
    kept, pruned = ordered[:MAX_CONVERSATIONS], ordered[MAX_CONVERSATIONS:]
    for record in pruned:
        try:
            _conversation_path(username, str(record.get("id"))).unlink(missing_ok=True)
        except ValueError:
            pass
    return [record for record in records if record in kept]


def create(username: str, title: str | None = None) -> dict[str, Any]:
    """Mint a new conversation; the index is pruned to MAX_CONVERSATIONS."""
    record = {"id": uuid.uuid4().hex[:12],
              "title": (str(title).strip()[:TITLE_MAX_CHARS] if title and str(title).strip()
                        else DEFAULT_TITLE),
              "created_at": _now_iso(), "updated_at": _now_iso(), "entry_count": 0}
    with assistant_memory._LOCK:
        records = _read_index(username)
        records.append(record)
        _write_index(username, _prune(username, records))
    return _public(record)


def rename(username: str, conversation_id: str, title: str) -> dict[str, Any] | None:
    """Retitle one conversation. None when the id is unknown."""
    with assistant_memory._LOCK:
        records = _read_index(username)
        record = _find(records, conversation_id)
        if record is None:
            return None
        record["title"] = str(title).strip()[:TITLE_MAX_CHARS]
        _write_index(username, records)
        return _public(record)


def delete(username: str, conversation_id: str) -> int | None:
    """Remove one conversation and its file. Returns the entry count removed,
    or None when the id is unknown for this user."""
    with assistant_memory._LOCK:
        records = _read_index(username)
        record = _find(records, conversation_id)
        if record is None:
            return None
        removed = len(entries_for(username, conversation_id))
        _write_index(username, [r for r in records if r is not record])
        _conversation_path(username, conversation_id).unlink(missing_ok=True)
    return removed


def entries_for(username: str, conversation_id: str | None) -> list[dict[str, Any]]:
    """The stored entries of one conversation, oldest first ([] when unknown)."""
    if not valid_id(conversation_id):
        return []
    with assistant_memory._LOCK:
        _migrate_legacy(username)
        return assistant_memory._load_entries(_conversation_path(username, str(conversation_id)))


def entry_batch_ids(username: str, conversation_id: str) -> set[str]:
    """The batch ids stored on this conversation's entries (legacy linkage)."""
    return {str(entry["batch_id"]) for entry in entries_for(username, conversation_id)
            if entry.get("batch_id")}


def resolve_for_ask(username: str, requested: str | None) -> str | None:
    """The conversation one ask lands in: the requested id when it exists, else
    the newest conversation when none was requested, else a fresh mint (an
    unknown or invalid requested id also mints). Never raises: a store failure
    yields None and the ask proceeds without thread memory."""
    try:
        with assistant_memory._LOCK:
            if requested and exists(username, requested):
                return str(requested)
            if not requested:
                newest = newest_id(username)
                if newest:
                    return newest
            return str(create(username)["id"])
    except Exception:  # noqa: BLE001 - memory is additive, the ask must proceed
        logger.exception("assistant conversation resolution failed for user %s", username)
        return None


def append_exchange(username: str, conversation_id: str | None, question: str,
                    answer: str, batch_id: str | None,
                    metadata: dict[str, Any] | None = None) -> None:
    """Append one successful ask to a conversation, pruned to the newest
    MAX_ENTRIES, updating the index (entry_count, updated_at, and the title on
    the first entry when it is still the default). May raise; the caller
    (assistant_memory.append_entry) swallows and logs."""
    with assistant_memory._LOCK:
        conversation_id = resolve_for_ask(username, conversation_id)
        if not conversation_id:
            raise RuntimeError("no conversation could be resolved or minted")
        path = _conversation_path(username, conversation_id)
        entries = assistant_memory._load_entries(path)
        entry = {"question": question, "answer": answer, "at": _now_iso(),
                 "batch_id": batch_id, "conversation_id": conversation_id}
        if isinstance(metadata, dict):
            entry.update({key: metadata.get(key) for key in assistant_memory._ENTRY_KEYS
                          if key not in entry and key in metadata})
        entries.append(entry)
        entries = entries[-assistant_memory.MAX_ENTRIES:]
        _write_json(path, {"user": username, "conversation_id": conversation_id,
                           "entries": entries})
        records = _read_index(username)
        record = _find(records, conversation_id)
        if record is None:
            record = {"id": conversation_id, "title": DEFAULT_TITLE,
                      "created_at": _now_iso(), "entry_count": 0}
            records.append(record)
        if not record.get("entry_count") and record.get("title") in (None, "", DEFAULT_TITLE):
            record["title"] = str(question).strip()[:TITLE_MAX_CHARS] or DEFAULT_TITLE
        record["entry_count"] = len(entries)
        record["updated_at"] = _now_iso()
        _write_index(username, records)


def clear_all(username: str) -> int:
    """Remove every conversation of one user. Returns total entries removed."""
    import shutil

    with assistant_memory._LOCK:
        _migrate_legacy(username)
        removed = 0
        for record in _read_index(username):
            removed += len(entries_for(username, str(record.get("id"))))
        directory = _user_dir(username)
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
    return removed
