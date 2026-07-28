"""Operation-state version history: immutable snapshots of the mutable state files.

A version is a point-in-time copy of the operation-state files the operator
edits: the settings JSON (with pricing overrides), the placement constraints store,
the manual overrides store, the advertiser rules, the scoped advertiser
conditions, and the calendar events store. Every mutation path snapshots the
touched file BEFORE it writes, so
the timeline is an append-only history:
restoring first snapshots the current state (a ``pre_restore`` point) and then puts
the selected files back, so a restore is always itself undoable.

The store is a directory of per-version folders holding byte copies plus a manifest.
It relocates for tests via KAIROS_VERSIONS_DIR (falling back beside the assistant
runtime root, then to data/versions), writes atomically, prunes to the newest 200,
and short-circuits an edit burst: a snapshot whose captured files are byte-identical
to the newest version is not re-recorded. Diffs and restores read the live files at
call time; a diff reports CURRENT state versus the chosen version, which is exactly
what restoring that version would change.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
VERSIONS_DIR_ENV = "KAIROS_VERSIONS_DIR"
ASSISTANT_DIR_ENV = "KAIROS_ASSISTANT_DATA_DIR"
MAX_VERSIONS = 200

router = APIRouter(prefix="/api/versions", tags=["versions"])

# The logical operation-state files. Paths resolve lazily (at call time) so a
# test that monkeypatches a store's PATH and the real deployment both hold.
_LOGICAL_ORDER = ("settings", "constraints", "overrides", "advertisers", "conditions",
                  "events", "agencies", "agency_links", "agency_conditions")

# Version ids are uuid4().hex[:12]; accept 8-32 lowercase hex so nothing else
# (a traversal path, a stray label) ever reaches the manifest reader.
_VERSION_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")


def _require_version_id(version_id: str) -> str:
    """404 on anything that is not a well-formed version id (hex, 8-32 chars)."""
    cleaned = str(version_id or "")
    if not _VERSION_ID_RE.fullmatch(cleaned):
        raise HTTPException(status_code=404, detail=f"no version {version_id!r}")
    return cleaned


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _versions_root() -> Path:
    raw = os.environ.get(VERSIONS_DIR_ENV, "").strip()
    if raw:
        return Path(raw)
    assistant = os.environ.get(ASSISTANT_DIR_ENV, "").strip()
    if assistant:
        return Path(assistant).parent / "versions"
    return ROOT / "data" / "versions"


def _logical_path(logical: str) -> Path:
    if logical == "settings":
        from kairos_api import core
        return Path(core.SETTINGS_PATH)
    if logical == "constraints":
        from kairos_api import constraints as constraints_api
        return Path(constraints_api.CONSTRAINTS_PATH)
    if logical == "overrides":
        from kairos_api import overrides as overrides_api
        return Path(overrides_api.OVERRIDES_PATH)
    if logical == "advertisers":
        from kairos_api import advertisers as advertisers_api
        return Path(advertisers_api.RULES_PATH)
    if logical == "conditions":
        from kairos_api import advertiser_conditions as conditions_api
        return Path(conditions_api.CONDITIONS_PATH)
    if logical == "events":
        from kairos_api import events_api
        return Path(events_api.EVENTS_PATH)
    if logical == "agencies":
        from kairos_api import agencies as agencies_api
        return Path(agencies_api.AGENCIES_PATH)
    if logical == "agency_links":
        from kairos_api import agency_conditions as agency_conditions_api
        return Path(agency_conditions_api.LINKS_PATH)
    if logical == "agency_conditions":
        from kairos_api import agency_conditions as agency_conditions_api
        return Path(agency_conditions_api.CONDITIONS_PATH)
    raise ValueError(f"unknown logical file {logical!r}")


_ID_COLUMN = {"constraints": "constraint_id", "overrides": "override_id",
              "advertisers": "advertiser_id", "conditions": "rule_id",
              "events": "event_id", "agencies": "agency_id",
              "agency_links": "agency_id", "agency_conditions": "rule_id"}


def _snapshot_name(logical: str) -> str:
    return f"{logical}{_logical_path(logical).suffix or '.dat'}"


# Auth seam: mirror the assistant action plane so roles behave identically.
def _actor(request: Request | None) -> str:
    from kairos_api import auth
    if not auth.auth_active():
        return "auth-disabled"
    session = auth._session_from_request(request) if request is not None else None
    return str(session["username"]) if session else "anonymous"


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


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _capture(logical: str) -> dict[str, Any]:
    """Read the current logical file's existence, bytes and hash."""
    path = _logical_path(logical)
    if path.exists():
        data = path.read_bytes()
        return {"logical": logical, "name": _snapshot_name(logical), "path": str(path),
                "existed": True, "sha256": _hash_bytes(data), "_bytes": data}
    return {"logical": logical, "name": _snapshot_name(logical), "path": str(path),
            "existed": False, "sha256": None, "_bytes": None}


def _manifest_path(version_id: str) -> Path:
    return _versions_root() / version_id / "manifest.json"


def _read_manifest(version_id: str) -> dict[str, Any]:
    path = _manifest_path(version_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"no version {version_id!r}")
    return json.loads(path.read_text(encoding="utf-8"))


def _all_manifests() -> list[dict[str, Any]]:
    root = _versions_root()
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for directory in root.iterdir():
        manifest = directory / "manifest.json"
        if not directory.is_dir() or not manifest.exists():
            continue
        try:
            found.append(json.loads(manifest.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    found.sort(key=lambda m: (str(m.get("created_at", "")), int(m.get("seq", 0))), reverse=True)
    return found


def _next_seq(manifests: list[dict[str, Any]]) -> int:
    return 1 + max((int(m.get("seq", 0)) for m in manifests), default=0)


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def _identical(newest: dict[str, Any], captured: list[dict[str, Any]]) -> bool:
    """True when the newest version covers exactly these logical files, each with
    the same existence and content hash, so re-recording would only spam the log."""
    prior = {f["logical"]: (f.get("existed"), f.get("sha256")) for f in newest.get("files", [])}
    now = {c["logical"]: (c["existed"], c["sha256"]) for c in captured}
    return prior == now


def _prune() -> list[str]:
    pruned: list[str] = []
    for manifest in _all_manifests()[MAX_VERSIONS:]:
        version_id = str(manifest.get("version_id", ""))
        directory = _versions_root() / version_id
        if version_id and directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            pruned.append(version_id)
    return pruned


def snapshot(source: str, actor: str, files: list[str], label: Optional[str] = None,
             batch_id: Optional[str] = None, force: bool = False) -> Optional[str]:
    """Record a version of the named logical files' current state.

    Returns the new version id, or the newest existing id when the capture is
    byte-identical to it (the edit-burst short-circuit), or None when no known
    logical file was named. ``force`` records unconditionally (used for the
    pre_restore safety point and the operator's named snapshot).
    """
    wanted = [name for name in _LOGICAL_ORDER if name in set(files)]
    if not wanted:
        return None
    captured = [_capture(name) for name in wanted]
    manifests = _all_manifests()
    if not force and manifests and _identical(manifests[0], captured):
        return str(manifests[0].get("version_id"))

    version_id = uuid.uuid4().hex[:12]
    directory = _versions_root() / version_id
    directory.mkdir(parents=True, exist_ok=True)
    file_entries: list[dict[str, Any]] = []
    for item in captured:
        if item["_bytes"] is not None:
            _atomic_write(directory / item["name"], item["_bytes"])
        file_entries.append({k: v for k, v in item.items() if k != "_bytes"})
    manifest = {
        "version_id": version_id, "created_at": _now_iso(), "seq": _next_seq(manifests),
        "actor": actor, "source": source, "label": label, "batch_id": batch_id,
        "files": file_entries,
    }
    _atomic_write(directory / "manifest.json",
                  json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))
    _prune()
    return version_id


# Assistant proposal-item kinds map to the logical files they mutate (pricing lives
# in the settings JSON; a recompute touches no snapshot-able state). An
# advertiser_change mutates the advertiser rules store, so it must be versioned
# too, or an assistant-driven agreement update would not be undoable.
_LOGICAL_FOR_KIND = {"settings": "settings", "pricing": "settings",
                     "constraint": "constraints", "override": "overrides",
                     "advertiser_change": "advertisers"}


def snapshot_assistant_apply(kinds: set[str], batch: dict[str, Any], actor: str) -> Optional[str]:
    """Record the pre-apply state in the unified version timeline (best-effort)."""
    logical = sorted({_LOGICAL_FOR_KIND[k] for k in kinds if k in _LOGICAL_FOR_KIND})
    if not logical:
        return None
    try:
        return snapshot(source="assistant_apply", actor=actor, files=logical,
                        label=(batch.get("question") or "").strip()[:60] or None,
                        batch_id=batch.get("batch_id"))
    except Exception:  # noqa: BLE001 - history is additive, never fail the apply
        return None


def _audit(event: str, actor: str, **fields: Any) -> None:
    entry = {"ts": _now_iso(), "actor": actor, "event": event}
    entry.update({k: v for k, v in fields.items() if v is not None})
    path = _versions_root() / "audit.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


# Diffs: current live state versus a chosen version (what restoring would change).
def _read_json(data: Optional[bytes]) -> dict[str, Any]:
    try:
        parsed = json.loads((data or b"{}").decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _read_rows(data: Optional[bytes], id_column: str) -> dict[str, dict[str, str]]:
    if not data:
        return {}
    text = data.decode("utf-8-sig")
    reader = csv.DictReader(text.splitlines())
    rows: dict[str, dict[str, str]] = {}
    for row in reader:
        key = str(row.get(id_column, "") or "").strip()
        if key:
            rows[key] = {k: ("" if v is None else str(v)) for k, v in row.items()}
    return rows


def _version_bytes(version_id: str, logical: str) -> Optional[bytes]:
    """The snapshotted bytes for one logical file in a version, or None if the
    file was absent at snapshot time."""
    manifest = _read_manifest(version_id)
    for entry in manifest.get("files", []):
        if entry.get("logical") == logical:
            if not entry.get("existed"):
                return None
            path = _versions_root() / version_id / str(entry.get("name"))
            return path.read_bytes() if path.exists() else None
    return None


def _current_bytes(logical: str) -> Optional[bytes]:
    path = _logical_path(logical)
    return path.read_bytes() if path.exists() else None


def _settings_diff(current: dict[str, Any], version: dict[str, Any]) -> dict[str, Any]:
    changed = []
    for field in sorted(set(current) | set(version)):
        cur = current.get(field)
        old = version.get(field)
        if cur != old:
            changed.append({"field": field, "from": cur, "to": old})
    return {"changed": changed}


def _rows_diff(current: dict[str, dict[str, str]], version: dict[str, dict[str, str]],
               id_key: str, names_only: bool) -> dict[str, Any]:
    added_ids = [k for k in version if k not in current]
    removed_ids = [k for k in current if k not in version]
    changed: list[dict[str, Any]] = []
    for key in current:
        if key not in version:
            continue
        cur_row, old_row = current[key], version[key]
        for field in sorted(set(cur_row) | set(old_row)):
            if str(cur_row.get(field, "")) != str(old_row.get(field, "")):
                changed.append({id_key: key, "field": field,
                                "from": cur_row.get(field, ""), "to": old_row.get(field, "")})
    if names_only:
        return {"added": sorted(added_ids), "removed": sorted(removed_ids), "changed": changed}
    return {
        "added": [version[k] for k in added_ids],
        "removed": [current[k] for k in removed_ids],
        "changed": changed,
    }


def _diff_logical(version_id: str, logical: str) -> dict[str, Any]:
    version_data = _version_bytes(version_id, logical)
    current_data = _current_bytes(logical)
    if logical == "settings":
        return _settings_diff(_read_json(current_data), _read_json(version_data))
    id_column = _ID_COLUMN[logical]
    id_key = "advertiser" if logical == "advertisers" else "id"
    return _rows_diff(_read_rows(current_data, id_column),
                      _read_rows(version_data, id_column),
                      id_key, names_only=(logical == "advertisers"))


# Restore: snapshot the current state first, then put the selected files back.
def _restore_logical(version_id: str, logical: str) -> str:
    manifest = _read_manifest(version_id)
    entry = next((f for f in manifest.get("files", []) if f.get("logical") == logical), None)
    if entry is None:
        raise HTTPException(status_code=404,
                            detail=f"version {version_id} does not cover {logical!r}")
    target = _logical_path(logical)
    if entry.get("existed"):
        snapshot_file = _versions_root() / version_id / str(entry.get("name"))
        if not snapshot_file.exists():
            raise HTTPException(status_code=500,
                                detail=f"version {version_id} is missing its {logical} snapshot")
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_write(target, snapshot_file.read_bytes())
    elif target.exists():
        target.unlink()
    return logical


def snapshot_manual_edit(request: Request | None, logical: str) -> None:
    """Hook for a manual dashboard edit: snapshot the touched file before it is
    written. A programmatic call (assistant apply, request None) is skipped, since
    that path snapshots the whole approved set itself. Never fails the request."""
    if request is None:
        return
    try:
        snapshot(source="manual_edit", actor=_actor(request), files=[logical])
    except Exception:  # noqa: BLE001 - a history hiccup must never fail the edit
        pass


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


@router.get("")
def list_versions(request: Request, limit: int = 50) -> dict[str, Any]:
    """Recorded versions, newest first."""
    _require_session(request)
    limit = max(1, min(int(limit), MAX_VERSIONS))
    entries = [_public_entry(m) for m in _all_manifests()[:limit]]
    return {"entries": entries, "note": _SCOPE_NOTE}


@router.get("/{version_id}/diff")
def version_diff(version_id: str, request: Request) -> dict[str, Any]:
    """Per logical file: what restoring this version would change from now."""
    _require_session(request)
    version_id = _require_version_id(version_id)
    manifest = _read_manifest(version_id)
    diff = {entry["logical"]: _diff_logical(version_id, entry["logical"])
            for entry in manifest.get("files", []) if entry.get("logical") in _LOGICAL_ORDER}
    return {"version_id": version_id, "created_at": manifest.get("created_at"),
            "source": manifest.get("source"), "diff": diff}


class RestoreRequest(BaseModel):
    files: Optional[list[str]] = None


@router.post("/{version_id}/restore")
def restore_version(version_id: str, request: Request,
                    body: RestoreRequest | None = None) -> dict[str, Any]:
    """Put the selected files back. Snapshots the current state first (undoable)."""
    actor = _require_writer(request)
    version_id = _require_version_id(version_id)
    manifest = _read_manifest(version_id)
    covered = [f["logical"] for f in manifest.get("files", []) if f.get("logical") in _LOGICAL_ORDER]
    requested = body.files if body and body.files else covered
    selected = [name for name in _LOGICAL_ORDER if name in set(requested) and name in covered]
    if not selected:
        raise HTTPException(status_code=400,
                            detail=f"no restorable files selected; this version covers {covered}")
    safety = snapshot(source="pre_restore", actor=actor, files=selected, force=True)
    restored = [_restore_logical(version_id, logical) for logical in selected]
    _audit("restore", actor, version_id=version_id, restored=restored, safety_version_id=safety)
    return {"restored": restored, "safety_version_id": safety}


class LabelRequest(BaseModel):
    label: Optional[str] = None


@router.post("/snapshot")
def create_snapshot(request: Request, body: LabelRequest | None = None) -> dict[str, Any]:
    """A named manual snapshot of the full operation state."""
    actor = _require_writer(request)
    label = body.label if body else None
    version_id = snapshot(source="manual_snapshot", actor=actor,
                          files=list(_LOGICAL_ORDER), label=label, force=True)
    _audit("snapshot", actor, version_id=version_id, label=label)
    return _public_entry(_read_manifest(str(version_id)))


@router.patch("/{version_id}")
def rename_version(version_id: str, body: LabelRequest, request: Request) -> dict[str, Any]:
    """Rename (relabel) a version. Writer roles only."""
    actor = _require_writer(request)
    version_id = _require_version_id(version_id)
    manifest = _read_manifest(version_id)
    manifest["label"] = body.label
    _atomic_write(_manifest_path(version_id),
                  json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))
    _audit("rename", actor, version_id=version_id, label=body.label)
    return _public_entry(manifest)
