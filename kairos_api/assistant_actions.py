"""Proposal store, apply engine, restore points and audit log for the assistant.

The action plane is review-first: the model only CAPTURES proposals
(kairos_api.assistant_tools); nothing mutates until an operator or admin
approves items here, and the apply engine then replays each approved item
through the SAME validated code path the manual UI uses, never raw file writes.
Safety invariants, each covered by tests: a restore point is snapshotted BEFORE
the first mutation, copying exactly the state files the approved items touch and
restorable byte-for-byte, pruned beyond the newest 20; the same pre-apply state
is also recorded in the unified version timeline (kairos_api.version_store); a
failed item records {status: 'failed', error} and the rest continue; a plan run
goes through the async job registry; apply/reject/restore require the operator
or admin role via the auth seam (403 for a viewer, 'auth-disabled' when off);
every transition is audited. Runtime state lives under data/assistant/
(gitignored); tests relocate it with KAIROS_ASSISTANT_DATA_DIR.

The restore-point store itself, its routes and the preview of what a restore
would change live in kairos_api.assistant_restore, so both files stay under the
file-size cap. The names this module re-exports below are the ones its own call
sites and its callers already used, so nothing that imported them moved.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos_api import assistant_tools
from kairos_api.assistant_restore import (
    DATA_DIR_ENV,
    MAX_RESTORE_POINTS,
    ROOT,
    _data_dir,
    _ID_RE,
    _LOCK,
    _now_iso,
    _restore_root,
    manifests as _manifests,
    prune_restore_points as _prune_restore_points,
    snapshot as _snapshot,
    list_restore_points,
    preview as restore_preview,
    restore_state,
)

# No prefix: kairos_api.assistant includes this router under /api/assistant.
router = APIRouter(tags=["assistant"])

__all__ = [
    "DATA_DIR_ENV", "MAX_RESTORE_POINTS", "ROOT", "router", "audit_append",
    "create_batch", "apply_proposals", "reject_proposals",
    "list_restore_points", "restore_state", "restore_preview",
]


# ---------------------------------------------------------------------------
# Auth seam: who is acting, and who may mutate.
# ---------------------------------------------------------------------------
def _actor(request: Request | None) -> str:
    """The acting username for the audit log, or 'auth-disabled'."""
    from kairos_api import auth

    if not auth.auth_active():
        return "auth-disabled"
    session = auth._session_from_request(request) if request is not None else None
    return str(session["username"]) if session else "anonymous"


def _require_writer(request: Request) -> str:
    """403 unless the session role may mutate (operator or admin). With auth
    disabled the call is allowed and audited as 'auth-disabled'."""
    from kairos_api import auth

    if not auth.auth_active():
        return "auth-disabled"
    session = auth._session_from_request(request)
    if session is None:
        raise HTTPException(status_code=401, detail="A signed-in session is required.")
    if session["role"] not in auth.WRITE_ROLES:
        raise HTTPException(status_code=403, detail=(
            "The operator or admin role is required to apply, reject or restore assistant proposals."))
    return str(session["username"])


# ---------------------------------------------------------------------------
# Audit log: append-only JSONL, newest-first reads.
# ---------------------------------------------------------------------------
def audit_append(event: str, user: str, **fields: Any) -> None:
    entry = {"ts": _now_iso(), "user": user, "event": event}
    entry.update({key: value for key, value in fields.items() if value is not None})
    path = _data_dir() / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


@router.get("/audit")
def read_audit(limit: int = 50) -> dict[str, Any]:
    """The newest audit entries, newest first. A corrupt line is returned raw."""
    limit = max(1, min(int(limit), 500))
    path = _data_dir() / "audit.jsonl"
    if not path.exists():
        return {"entries": []}
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    entries: list[dict[str, Any]] = []
    for line in reversed(lines[-limit:]):
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            entries.append({"corrupt": True, "raw": line[:500]})
    return {"entries": entries}


# ---------------------------------------------------------------------------
# Proposal store.
# ---------------------------------------------------------------------------
def _load_store() -> dict[str, Any]:
    path = _data_dir() / "proposals.json"
    if not path.exists():
        return {"batches": []}
    try:
        store = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail=f"proposals store unreadable: {exc}") from exc
    if not isinstance(store, dict) or not isinstance(store.get("batches"), list):
        raise HTTPException(status_code=500, detail="proposals store has an unexpected shape")
    return store


def _save_store(store: dict[str, Any]) -> None:
    """Write the proposals store atomically (temp file + os.replace).

    Callers hold ``_LOCK`` across load-mutate-save; the atomic replace means a
    crash mid-write can never leave a truncated JSON behind for the next boot.
    """
    path = _data_dir() / "proposals.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(store, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def _batch_status(items: list[dict[str, Any]]) -> str:
    return "pending" if any(item.get("status") == "pending" for item in items) else "resolved"


def _find_batch(store: dict[str, Any], batch_id: str) -> dict[str, Any]:
    for batch in store["batches"]:
        if batch.get("batch_id") == batch_id:
            return batch
    raise HTTPException(status_code=404, detail=f"no proposal batch {batch_id!r}")


def create_batch(question: str, items: list[dict[str, Any]], user: str, model: str,
                 conversation_id: str | None = None) -> dict[str, Any]:
    """Persist one ask's captured proposal items as a batch and audit it. The
    conversation the ask ran in rides on the batch so the per-conversation
    changes view and restore can collect it without a join through entries.

    The batch also records WHICH PLAN it was reasoned against
    (kairos_api.assistant_proposal_freshness), so an item approved after a run
    can say that the figures its summary quotes were measured against a plan
    that has since moved. Nothing else about the batch depends on it, and a
    stamp that cannot be read is None rather than a guess.
    """
    from kairos_api.assistant_proposal_freshness import plan_stamp

    batch = {
        "batch_id": uuid.uuid4().hex[:12], "question": question, "created_at": _now_iso(),
        "created_by": user, "model": model, "status": _batch_status(items), "items": items,
        "conversation_id": conversation_id, "plan_stamp": plan_stamp(),
    }
    counts = {status: sum(1 for item in items if item["status"] == status)
              for status in ("pending", "rejected")}
    with _LOCK:
        store = _load_store()
        store["batches"].append(batch)
        _save_store(store)
        audit_append("proposal", user, batch_id=batch["batch_id"], question=question,
                     model=model, conversation_id=conversation_id,
                     item_ids=[item["id"] for item in items], results=counts)
    return batch


@router.get("/proposals")
def list_proposals(limit: int = 20) -> dict[str, Any]:
    """Recent proposal batches, newest first, each item carrying the terms its
    summary was built from so the surface can say it in the reader's language.

    Items written before those terms existed carry none on disk, so they are
    derived here from the payload the item already stores. The store itself is
    never touched: a read must not rewrite a file a gate has to diff.
    """
    from kairos_api.assistant_proposal_freshness import verdict as _plan_verdict
    from kairos_api.assistant_summary_terms import terms_for_item

    limit = max(1, min(int(limit), 200))
    with _LOCK:
        store = _load_store()
    batches = []
    for batch in list(reversed(store["batches"]))[:limit]:
        items = []
        for item in batch.get("items", []):
            terms = terms_for_item(item)
            items.append({**item, "summary_terms": terms} if terms else item)
        # Derived at read time, never written back, for the reason above: a read
        # must not rewrite a file a gate has to diff. It is also the only correct
        # place for it, since whether the plan has moved is a fact about NOW.
        batches.append({**batch, "items": items,
                        "plan_freshness": _plan_verdict(batch.get("plan_stamp"))})
    return {"batches": batches}


# ---------------------------------------------------------------------------
# Which state files each item kind touches, so the snapshot covers exactly them.
# ---------------------------------------------------------------------------
def _state_files_for(kinds: set[str]) -> list[Path]:
    """The state files these item kinds mutate, resolved from the owning
    modules AT CALL TIME so test monkeypatching and deployments both hold."""
    from kairos_api import constraints as constraints_api
    from kairos_api import core
    from kairos_api import overrides as overrides_api

    files: list[Path] = []
    if kinds & {"settings", "pricing"}:
        files.append(Path(core.SETTINGS_PATH))
    if "constraint" in kinds:
        files.append(Path(constraints_api.CONSTRAINTS_PATH))
    if "override" in kinds:
        files.append(Path(overrides_api.OVERRIDES_PATH))
    if "advertiser_change" in kinds:
        from kairos_api import advertisers as advertisers_api

        files.append(Path(advertisers_api.RULES_PATH))
    return files


# ---------------------------------------------------------------------------
# Apply engine: replay approved items through the real seams.
#
# The appliers themselves live in kairos_api.assistant_apply_kinds (size cap).
# The registry object is imported, not copied, so this module's _APPLIERS is
# that dict and assistant_propose_extra.register_action_plane still extends the
# same mapping. Each applier takes (payload, approving_account).
# ---------------------------------------------------------------------------
from kairos_api.assistant_apply_kinds import _APPLIERS, _expand_days  # noqa: E402,F401


class ItemIdsRequest(BaseModel):
    item_ids: list[str] = Field(min_length=1)


def _record_restore_point(batch: dict[str, Any], restore_id: str | None, user: str,
                          item_ids: list[str], version_id: str | None = None) -> None:
    """Keep the restore point ON the batch, not only in the apply response.

    Undo was reachable for exactly as long as the browser tab that applied the
    change: the id came back once in the apply body and was never stored, so a
    reload lost the only handle to the snapshot. The reference model is
    Cursor's, where a checkpoint sits on the request that created it and can be
    opened at any later time, so the batch now carries every restore point it
    produced, oldest first, each naming who applied what and when. A batch whose
    approved items touch no state file (a plan run) produces no snapshot and
    records nothing, which is why the id can be None here.

    The point also carries the unified version timeline's id for the same
    apply, when one was recorded: this is the id a "see it in the history"
    control has to address, since the History page is keyed on that timeline
    and the restore id it applies against is not an entry there.
    """
    if not restore_id:
        return
    points = batch.setdefault("restore_points", [])
    points.append({"restore_id": restore_id, "applied_at": _now_iso(),
                   "applied_by": user, "item_ids": list(item_ids), "version_id": version_id})


def _act_apply(item: dict[str, Any], user: str) -> None:
    """Apply one item through its real seam; per-item isolation lives here.

    ``user`` is the account that APPROVED the item. Every applier reaches its
    store programmatically (request=None), so the store's own actor lookup finds
    nobody; the stores that record who acted take this name instead of a blank.
    """
    applier = _APPLIERS.get(str(item.get("kind")))
    try:
        if applier is None:
            raise ValueError(f"no applier for kind {item.get('kind')!r}")
        result = applier(item.get("payload") or {}, user)
    except HTTPException as exc:
        item["status"] = "failed"
        item["error"] = str(exc.detail)
    except Exception as exc:  # noqa: BLE001 - per-item isolation, honest error
        item["status"] = "failed"
        item["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    else:
        item["status"] = "applied"
        item["result"] = result
        item.pop("error", None)


def _act_reject(item: dict[str, Any], user: str) -> None:
    del user  # a rejection mutates nothing, so it reaches no store that records an actor
    item["status"] = "rejected"


def _resolve_items(
    batch_id: str, body: ItemIdsRequest, request: Request, event: str, verb: str, act: Any
) -> dict[str, Any]:
    """Shared apply/reject skeleton: writer role, per-item resolution, audit.

    Only pending items are acted on; an unknown id or an already-resolved item
    lands in results as a failed entry and the rest continue. For apply, the
    restore point is snapshotted BEFORE the first mutation and covers exactly
    the state files the approved items touch.
    """
    user = _require_writer(request)
    requested = list(dict.fromkeys(str(item_id) for item_id in body.item_ids))
    with _LOCK:
        store = _load_store()
        batch = _find_batch(store, batch_id)
        by_id = {item["id"]: item for item in batch["items"]}
        approved = [item_id for item_id in requested
                    if item_id in by_id and by_id[item_id].get("status") == "pending"]
        extra: dict[str, Any] = {}
        if event == "apply":
            from kairos_api.events_access import assistant_apply_block

            blocked = assistant_apply_block(user, [by_id[item_id] for item_id in approved])
            if blocked:
                raise HTTPException(status_code=403, detail=blocked)
            kinds = {str(by_id[item_id].get("kind")) for item_id in approved}
            extra["restore_id"] = _snapshot(_state_files_for(kinds), batch_id, approved)
            extra["pruned_restore_points"] = _prune_restore_points() or None
            from kairos_api import version_store  # unified version timeline

            extra["version_id"] = version_store.snapshot_assistant_apply(kinds, batch, user)
            _record_restore_point(batch, extra["restore_id"], user, approved, extra["version_id"])
            # Which plan these items were reasoned against, answered at the moment
            # of approval. It rides the response AND the audit entry, so a change
            # approved against a plan that had already moved is legible afterwards
            # and not only to whoever happened to be looking at the card.
            from kairos_api.assistant_proposal_freshness import verdict as _plan_verdict

            extra["plan_freshness"] = _plan_verdict(batch.get("plan_stamp"))
        results: list[dict[str, Any]] = []
        for item_id in requested:
            item = by_id.get(item_id)
            if item is None:
                results.append({"id": item_id, "status": "failed", "error": "no such item in this batch"})
                continue
            if item_id not in approved:
                results.append({"id": item_id, "status": "failed", "error": (
                    f"item status is {item.get('status')!r}; only pending items can be {verb}")})
                continue
            act(item, user)
            item["resolved_at"] = _now_iso()
            item["resolved_by"] = user
            entry: dict[str, Any] = {"id": item_id, "status": item["status"]}
            if item["status"] == "applied":
                entry["result"] = item.get("result")
            elif item.get("error"):
                entry["error"] = item["error"]
            results.append(entry)
        batch["status"] = _batch_status(batch["items"])
        _save_store(store)
        audit_append(event, user, batch_id=batch_id, item_ids=requested, results=results, **extra)
    extra.pop("pruned_restore_points", None)
    return {"batch_id": batch_id, "status": batch["status"], "results": results, **extra}


@router.post("/proposals/{batch_id}/apply")
def apply_proposals(batch_id: str, body: ItemIdsRequest, request: Request) -> dict[str, Any]:
    """Apply approved pending items through the real seams, isolated per item."""
    return _resolve_items(batch_id, body, request, "apply", "applied", _act_apply)


@router.post("/proposals/{batch_id}/reject")
def reject_proposals(batch_id: str, body: ItemIdsRequest, request: Request) -> dict[str, Any]:
    """Mark pending items rejected by the operator. Nothing is mutated."""
    return _resolve_items(batch_id, body, request, "reject", "rejected", _act_reject)
