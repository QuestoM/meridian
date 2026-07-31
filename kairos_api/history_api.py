"""History: the version timeline, what a restore would change, and the restore.

The HTTP layer of the operation-state version store, moved verbatim from
version_store.py as part of the wave-zero router split. The store itself, which
eight other modules call to snapshot before they write, stays where it is and is
reached through the module here.

Behaviour is unchanged: the same role gates (a signed-in session to read, an
operator or admin role to snapshot, restore or rename), the same nine logical
files, and the same pre-restore safety point, so a restore is always itself
undoable.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from kairos_api import version_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/versions")

# Version ids are uuid4().hex[:12]; accept 8-32 lowercase hex so nothing else
# (a traversal path, a stray label) ever reaches the manifest reader.
_VERSION_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")


def _require_version_id(version_id: str) -> str:
    """404 on anything that is not a well-formed version id (hex, 8-32 chars)."""
    cleaned = str(version_id or "")
    if not _VERSION_ID_RE.fullmatch(cleaned):
        raise HTTPException(status_code=404, detail=f"no version {version_id!r}")
    return cleaned


# Auth seam: mirror the assistant action plane so roles behave identically.
def _require_session(request: Request, writer: bool = False) -> str:
    """401 without a signed-in session; 403 when ``writer`` and the role is read-only.
    With auth disabled every call is allowed and acts as 'auth-disabled'."""
    from kairos_api import auth
    if not auth.auth_active():
        return "auth-disabled"
    session = auth._session_from_request(request)
    if session is None:
        raise HTTPException(status_code=401, detail="A signed-in session is required.")
    if writer and session["role"] not in auth.WRITE_ROLES:
        raise HTTPException(status_code=403, detail=(
            "The operator or admin role is required to snapshot, restore or rename versions."))
    return str(session["username"])


def _require_writer(request: Request) -> str:
    return _require_session(request, writer=True)


_SCOPE_NOTE = ("Versions snapshot the operation-state files the operator edits: settings "
               "(with pricing overrides), placement constraints, manual overrides, "
               "advertiser rules, scoped advertiser conditions and calendar events. "
               "History is append-only; a restore first records the current state, "
               "so it is always undoable.")


def _public_entry(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "version_id": manifest.get("version_id"),
        "created_at": manifest.get("created_at"),
        "actor": manifest.get("actor"),
        "source": manifest.get("source"),
        "label": manifest.get("label"),
        "batch_id": manifest.get("batch_id"),
        "files": [f.get("logical") for f in manifest.get("files", [])],
    }


@router.get("", tags=["versions"])
def list_versions(request: Request, limit: int = 50) -> dict[str, Any]:
    """Recorded versions, newest first."""
    _require_session(request)
    limit = max(1, min(int(limit), version_store.MAX_VERSIONS))
    entries = [_public_entry(m) for m in version_store._all_manifests()[:limit]]
    return {"entries": entries, "note": _SCOPE_NOTE}


@router.get("/{version_id}/diff", tags=["versions"])
def version_diff(version_id: str, request: Request) -> dict[str, Any]:
    """Per logical file: what restoring this version would change from now."""
    _require_session(request)
    version_id = _require_version_id(version_id)
    manifest = version_store._read_manifest(version_id)
    diff = {entry["logical"]: version_store._diff_logical(version_id, entry["logical"])
            for entry in manifest.get("files", []) if entry.get("logical") in version_store._LOGICAL_ORDER}
    return {"version_id": version_id, "created_at": manifest.get("created_at"),
            "source": manifest.get("source"), "diff": diff}


class RestoreRequest(BaseModel):
    files: Optional[list[str]] = None


@router.post("/{version_id}/restore", tags=["versions"])
def restore_version(version_id: str, request: Request,
                    body: RestoreRequest | None = None) -> dict[str, Any]:
    """Put the selected files back. Snapshots the current state first (undoable)."""
    actor = _require_writer(request)
    version_id = _require_version_id(version_id)
    manifest = version_store._read_manifest(version_id)
    covered = [f["logical"] for f in manifest.get("files", []) if f.get("logical") in version_store._LOGICAL_ORDER]
    requested = body.files if body and body.files else covered
    selected = [name for name in version_store._LOGICAL_ORDER if name in set(requested) and name in covered]
    if not selected:
        raise HTTPException(status_code=400,
                            detail=f"no restorable files selected; this version covers {covered}")
    safety = version_store.snapshot(source="pre_restore", actor=actor, files=selected, force=True)
    restored = [version_store._restore_logical(version_id, logical) for logical in selected]
    version_store._audit("restore", actor, version_id=version_id, restored=restored, safety_version_id=safety)
    return {"restored": restored, "safety_version_id": safety}


class LabelRequest(BaseModel):
    label: Optional[str] = None


@router.post("/snapshot", tags=["versions"])
def create_snapshot(request: Request, body: LabelRequest | None = None) -> dict[str, Any]:
    """A named manual snapshot of the full operation state."""
    actor = _require_writer(request)
    label = body.label if body else None
    version_id = version_store.snapshot(source="manual_snapshot", actor=actor,
                                        files=list(version_store._LOGICAL_ORDER), label=label, force=True)
    version_store._audit("snapshot", actor, version_id=version_id, label=label)
    return _public_entry(version_store._read_manifest(str(version_id)))


@router.patch("/{version_id}", tags=["versions"])
def rename_version(version_id: str, body: LabelRequest, request: Request) -> dict[str, Any]:
    """Rename (relabel) a version. Writer roles only."""
    actor = _require_writer(request)
    version_id = _require_version_id(version_id)
    manifest = version_store._read_manifest(version_id)
    manifest["label"] = body.label
    version_store._atomic_write(version_store._manifest_path(version_id),
                                json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))
    version_store._audit("rename", actor, version_id=version_id, label=body.label)
    return _public_entry(manifest)
