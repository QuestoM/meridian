"""Conversation endpoints for the assistant: CRUD, applied changes, restore.

Every route derives the user STRICTLY from the authenticated session (the same
keystone as the thread store), so one user can never list, rename, delete or
restore another user's conversations. Deletes and restores are audited.

The conversation restore is a thin orchestration over shipped primitives and
invents no new snapshot mechanics: it collects the conversation's batches that
have any applied item, maps them to their ``assistant_apply`` versions in the
unified timeline (a batch elided by the byte-identical short-circuit maps to
the nearest OLDER assistant_apply version instead), picks per logical file the
OLDEST such version (its snapshot is the state BEFORE the conversation's first
mutation of that file), records one forced ``pre_restore`` safety version so
the whole operation is undoable, and puts the files back through the version
store's own restore primitive. Honest limits ride in the response: recomputes
run during the conversation cannot be un-run (inputs are restored, a recompute
is required for the plan to reflect them), and a whole-file restore also
reverts manual edits made to the same file after the conversation.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos_api import assistant_actions, assistant_conversations as conversations

# No prefix: kairos_api.assistant includes this router under /api/assistant.
router = APIRouter(tags=["assistant"])

RESTORE_NOTE = (
    "השחזור מחזיר את קבצי הקלט למצבם מלפני השינוי הראשון של השיחה; חישובים שהופעלו במהלך השיחה אינם ניתנים לביטול, ולכן נדרש חישוב מחדש כדי שהתוכנית תשקף את הקבצים המשוחזרים. שחזור קובץ שלם מבטל גם עריכות ידניות שנעשו בו אחרי השיחה."
)


class ConversationCreate(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class ConversationRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)


def _require_known(username: str, conversation_id: str) -> None:
    if not conversations.exists(username, conversation_id):
        raise HTTPException(status_code=404, detail=f"no conversation {conversation_id!r}")


@router.get("/conversations")
def list_conversations(request: Request) -> dict[str, Any]:
    """The caller's own conversations, newest updated first."""
    username = assistant_actions._actor(request)
    return {"conversations": conversations.list_records(username), "user": username}


@router.post("/conversations")
def create_conversation(request: Request, body: ConversationCreate | None = None) -> dict[str, Any]:
    record = conversations.create(assistant_actions._actor(request),
                                  body.title if body else None)
    return {"id": record["id"], "title": record["title"]}


@router.patch("/conversations/{conversation_id}")
def rename_conversation(conversation_id: str, body: ConversationRename,
                        request: Request) -> dict[str, Any]:
    username = assistant_actions._actor(request)
    record = conversations.rename(username, conversation_id, body.title)
    if record is None:
        raise HTTPException(status_code=404, detail=f"no conversation {conversation_id!r}")
    return {"id": record["id"], "title": record["title"]}


@router.delete("/conversations/{conversation_id}")
def delete_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
    """Delete ONLY this conversation of ONLY the caller, audited."""
    username = assistant_actions._actor(request)
    removed = conversations.delete(username, conversation_id)
    if removed is None:
        raise HTTPException(status_code=404, detail=f"no conversation {conversation_id!r}")
    assistant_actions.audit_append("conversation_delete", username,
                                   conversation_id=conversation_id,
                                   results={"entries_removed": removed})
    return {"deleted": True, "entries_removed": removed}


def _conversation_batches(username: str, conversation_id: str) -> list[dict[str, Any]]:
    """This conversation's proposal batches, oldest first: batches stamped with
    the conversation_id, plus (for the legacy conversation, whose batches
    predate the stamp) batches whose batch_id one of its entries carries."""
    entry_ids = conversations.entry_batch_ids(username, conversation_id)
    with assistant_actions._LOCK:
        store = assistant_actions._load_store()
    return [batch for batch in store["batches"]
            if batch.get("conversation_id") == conversation_id
            or (not batch.get("conversation_id") and batch.get("batch_id") in entry_ids)]


def _assistant_apply_versions() -> list[dict[str, Any]]:
    from kairos_api import version_store

    return [manifest for manifest in version_store._all_manifests()
            if manifest.get("source") == "assistant_apply"]


@router.get("/conversations/{conversation_id}/changes")
def conversation_changes(conversation_id: str, request: Request) -> dict[str, Any]:
    """The changes this conversation drove: its batches with per-item status
    and the assistant_apply version ids each batch recorded, newest first."""
    username = assistant_actions._actor(request)
    _require_known(username, conversation_id)
    versions_by_batch: dict[str, list[str]] = {}
    for manifest in _assistant_apply_versions():
        if manifest.get("batch_id"):
            versions_by_batch.setdefault(str(manifest["batch_id"]), []).append(
                str(manifest.get("version_id")))
    batches = [{
        "batch_id": batch.get("batch_id"),
        "question": batch.get("question"),
        "created_by": batch.get("created_by"),
        "created_at": batch.get("created_at"),
        "status": batch.get("status"),
        "items": [{key: item.get(key)
                   for key in ("id", "kind", "summary", "status", "resolved_by", "resolved_at")}
                  for item in batch.get("items", [])],
        "version_ids": versions_by_batch.get(str(batch.get("batch_id")), []),
    } for batch in _conversation_batches(username, conversation_id)]
    return {"batches": list(reversed(batches))}


def _applied_time(batch: dict[str, Any]) -> str:
    """The batch's earliest applied-item time (its apply snapshot is older)."""
    times = [str(item.get("resolved_at")) for item in batch.get("items", [])
             if item.get("status") == "applied" and item.get("resolved_at")]
    return min(times) if times else str(batch.get("created_at") or "")


def _restore_plan(batches: list[dict[str, Any]]) -> tuple[dict[str, str], list[str]]:
    """Per logical file, the assistant_apply version to restore it from.

    A batch with its own versions contributes them directly. A batch whose
    snapshot was elided as byte-identical maps each logical file its applied
    items touch to the nearest assistant_apply version OLDER than its apply
    time. Per file the OLDEST candidate wins: its snapshot is the state before
    the conversation's first mutation of that file.
    """
    from kairos_api.version_store import _LOGICAL_FOR_KIND

    versions = _assistant_apply_versions()  # newest first
    candidates: list[dict[str, Any]] = []
    elided: list[str] = []
    for batch in batches:
        own = [m for m in versions if str(m.get("batch_id")) == str(batch.get("batch_id"))]
        if own:
            candidates.extend(own)
            continue
        applied_kinds = {str(item.get("kind")) for item in batch.get("items", [])
                         if item.get("status") == "applied"}
        needed = {_LOGICAL_FOR_KIND[kind] for kind in applied_kinds if kind in _LOGICAL_FOR_KIND}
        if not needed:
            continue  # honest: recompute-only batches snapshot no state file
        elided.append(str(batch.get("batch_id")))
        cutoff = _applied_time(batch)
        for logical in needed:
            fallback = next(
                (m for m in versions if str(m.get("created_at", "")) <= cutoff
                 and any(f.get("logical") == logical for f in m.get("files", []))), None)
            if fallback is not None:
                candidates.append(fallback)
    chosen: dict[str, str] = {}
    order: dict[str, tuple[str, int]] = {}
    for manifest in candidates:
        key = (str(manifest.get("created_at", "")), int(manifest.get("seq", 0)))
        for entry in manifest.get("files", []):
            logical = str(entry.get("logical"))
            if logical not in chosen or key < order[logical]:
                chosen[logical] = str(manifest.get("version_id"))
                order[logical] = key
    return chosen, elided


@router.post("/conversations/{conversation_id}/restore")
def restore_conversation(conversation_id: str, request: Request) -> dict[str, Any]:
    """Put every state file this conversation mutated back to its pre-conversation
    state, through the version store's own primitives, undoably. 409 when the
    conversation applied nothing restorable."""
    from kairos_api import version_store

    user = assistant_actions._require_writer(request)
    _require_known(user, conversation_id)
    batches = [batch for batch in _conversation_batches(user, conversation_id)
               if any(item.get("status") == "applied" for item in batch.get("items", []))]
    if not batches:
        raise HTTPException(status_code=409, detail=(
            "this conversation applied no changes, so there is nothing to restore"))
    chosen, elided = _restore_plan(batches)
    if not chosen:
        raise HTTPException(status_code=409, detail=(
            "this conversation's applied changes left no restorable state snapshot "
            "(recomputes cannot be un-run; restore covers state files only)"))
    selected = [name for name in version_store._LOGICAL_ORDER if name in chosen]
    pre_restore = version_store.snapshot(source="pre_restore", actor=user, files=selected,
                                         label=f"לפני שחזור שיחה {conversation_id}", force=True)
    restored = [version_store._restore_logical(chosen[logical], logical) for logical in selected]
    version_ids_used = list(dict.fromkeys(chosen[logical] for logical in selected))
    version_store._audit("conversation_restore", user, conversation_id=conversation_id,
                         restored=restored, version_ids=version_ids_used,
                         safety_version_id=pre_restore)
    assistant_actions.audit_append(
        "conversation_restore", user, conversation_id=conversation_id,
        results={"restored_files": restored, "version_ids_used": version_ids_used,
                 "pre_restore_version_id": pre_restore,
                 "elided_batch_fallbacks": elided or None})
    return {
        "restored_files": restored,
        "version_ids_used": version_ids_used,
        "pre_restore_version_id": pre_restore,
        "recompute_required": True,
        "note": RESTORE_NOTE,
    }
