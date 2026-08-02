"""Restore points for the assistant: the store, the reader and the preview.

A restore point is the state of exactly the files an approved batch was about to
touch, copied before the first mutation. Split out of kairos_api.assistant_actions
so that module stays under the file-size cap, and extended with the one thing
the product was missing: **a restore point you can open and read before you use
it.** Discovery measured the gap plainly, "there is no undo control in the
product", and a reversal nobody can inspect is not an undo, it is a second
change of unknown size.

``GET /api/assistant/restore/{restore_id}`` therefore answers what restoring
would actually do, computed by comparing the snapshot on disk with the file as
it stands now:

* a JSON store is diffed field by field on dotted keys, so a settings restore
  reads "min_retention_floor: 0.75 back to 0.72"
* a CSV store is diffed row by row on its first column, so an added, removed or
  edited row is named
* anything else reports its byte-size change, which is honest about being
  coarse rather than pretending to a field diff it cannot compute

Nothing here is estimated. A file whose snapshot is missing reads unavailable
with the reason, an unreadable file reads unavailable with the reason, and a
file that has not moved since the snapshot reads unchanged, which is the answer
that stops a person restoring for nothing.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR_ENV = "KAIROS_ASSISTANT_DATA_DIR"
MAX_RESTORE_POINTS = 20
_ID_RE = re.compile(r"^[0-9a-f]{8,32}$")

# One lock for the whole assistant runtime store: the proposals file and the
# restore points are written under the same critical sections.
_LOCK = threading.Lock()

# Field-level diffs are capped so a preview of a large store stays readable and
# the honest omitted count rides beside it.
MAX_DIFF_ROWS = 40

# Every reason and note code this preview can emit. The English sentence beside
# each code stays as the record an API reader gets; the operator surface says
# the same thing in the operator's language from the code, because English prose
# printed raw on a Hebrew screen is a defect, not a fallback. A test pins this
# tuple against the readings the surface holds, so a code added here without one
# fails there rather than on a screen.
PREVIEW_CODES = ("snapshot_missing", "snapshot_unreadable", "current_unreadable",
                 "absent_at_snapshot", "nothing_would_change")

# No prefix: kairos_api.assistant includes this router under /api/assistant.
router = APIRouter(tags=["assistant"])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _data_dir() -> Path:
    raw = os.environ.get(DATA_DIR_ENV, "").strip()
    return Path(raw) if raw else ROOT / "data" / "assistant"


def _restore_root() -> Path:
    return _data_dir() / "restore"


# ---------------------------------------------------------------------------
# The store: snapshot before the first mutation, restore byte-for-byte.
# ---------------------------------------------------------------------------
def snapshot(files: list[Path], batch_id: str, item_ids: list[str]) -> str | None:
    """Copy the touched state files into one restore point. None when empty."""
    if not files:
        return None
    restore_id = uuid.uuid4().hex[:12]
    directory = _restore_root() / restore_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest_files: list[dict[str, Any]] = []
    for source in files:
        existed = source.exists()
        if existed:
            shutil.copy2(source, directory / source.name)
        manifest_files.append({"path": str(source), "name": source.name, "existed": existed})
    manifest = {"restore_id": restore_id, "batch_id": batch_id, "item_ids": item_ids,
                "created_at": _now_iso(), "files": manifest_files}
    # Atomic manifest write: a restore point either exists with a complete
    # manifest or not at all, never with a torn one the lister reports corrupt.
    manifest_path = directory / "manifest.json"
    tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, manifest_path)
    return restore_id


def manifests() -> list[dict[str, Any]]:
    root = _restore_root()
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for directory in root.iterdir():
        manifest_path = directory / "manifest.json"
        if not directory.is_dir() or not manifest_path.exists():
            continue
        try:
            found.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            found.append({"restore_id": directory.name, "corrupt": True})
    found.sort(key=lambda manifest: str(manifest.get("created_at", "")), reverse=True)
    return found


def prune_restore_points() -> list[str]:
    """Delete restore points beyond the newest MAX_RESTORE_POINTS."""
    pruned: list[str] = []
    for manifest in manifests()[MAX_RESTORE_POINTS:]:
        restore_id = str(manifest.get("restore_id", ""))
        if _ID_RE.fullmatch(restore_id):
            shutil.rmtree(_restore_root() / restore_id, ignore_errors=True)
            pruned.append(restore_id)
    return pruned


# ---------------------------------------------------------------------------
# The preview: what restoring this point would actually change.
# ---------------------------------------------------------------------------
def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """A nested JSON object as dotted keys, so a diff reads as fields."""
    if isinstance(value, dict):
        flat: dict[str, Any] = {}
        for key, item in value.items():
            flat.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return flat
    return {prefix: value}


def _json_changes(current_text: str, restored_text: str) -> dict[str, Any] | None:
    """Field-level changes, or None when either side is not a JSON object."""
    try:
        current_obj = json.loads(current_text)
        restored_obj = json.loads(restored_text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(current_obj, dict) or not isinstance(restored_obj, dict):
        return None
    current = _flatten(current_obj)
    restored = _flatten(restored_obj)
    rows = [
        {"field": field,
         "current": current.get(field),
         "restored": restored.get(field),
         "state": "changed" if field in current and field in restored
                  else "removed" if field in current else "added"}
        for field in sorted(set(current) | set(restored))
        if current.get(field) != restored.get(field)
    ]
    payload: dict[str, Any] = {"kind": "fields", "change_count": len(rows),
                               "changes": rows[:MAX_DIFF_ROWS]}
    if len(rows) > MAX_DIFF_ROWS:
        payload["changes_omitted"] = len(rows) - MAX_DIFF_ROWS
    return payload


def _rows_by_key(text: str) -> tuple[dict[str, dict[str, str]], list[str]] | None:
    """A CSV keyed on its first column, or None when it has no header."""
    reader = csv.DictReader(io.StringIO(text))
    header = reader.fieldnames
    if not header:
        return None
    key = header[0]
    rows: dict[str, dict[str, str]] = {}
    for index, row in enumerate(reader):
        identifier = str(row.get(key) or f"row {index + 1}")
        rows[identifier] = {name: str(row.get(name) or "") for name in header}
    return rows, list(header)


def _csv_changes(current_text: str, restored_text: str) -> dict[str, Any] | None:
    """Row-level changes on the first column, or None when either side is not a
    readable CSV with a header."""
    current_parsed = _rows_by_key(current_text)
    restored_parsed = _rows_by_key(restored_text)
    if current_parsed is None or restored_parsed is None:
        return None
    current, header = current_parsed
    restored, _ = restored_parsed
    rows: list[dict[str, Any]] = []
    for identifier in sorted(set(current) | set(restored)):
        before = current.get(identifier)
        after = restored.get(identifier)
        if before == after:
            continue
        # State is written from the reader's side of the undo, not the file's:
        # a row absent now and present in the snapshot comes back.
        if before is None:
            rows.append({"row": identifier, "state": "added"})
        elif after is None:
            rows.append({"row": identifier, "state": "removed"})
        else:
            fields = [name for name in header if before.get(name) != after.get(name)]
            rows.append({"row": identifier, "state": "changed",
                         "fields": [{"field": name, "current": before.get(name),
                                     "restored": after.get(name)} for name in fields]})
    payload: dict[str, Any] = {
        "kind": "rows", "key_column": header[0], "change_count": len(rows),
        "rows_now": len(current), "rows_after_restore": len(restored),
        "changes": rows[:MAX_DIFF_ROWS],
    }
    if len(rows) > MAX_DIFF_ROWS:
        payload["changes_omitted"] = len(rows) - MAX_DIFF_ROWS
    return payload


def _file_preview(entry: dict[str, Any], directory: Path) -> dict[str, Any]:
    """What restoring one file in this point would do."""
    target = Path(str(entry.get("path") or ""))
    name = str(entry.get("name") or target.name)
    result: dict[str, Any] = {"file": name, "path": str(target)}
    if not entry.get("existed", True):
        result["effect"] = "delete" if target.exists() else "unchanged"
        result["kind"] = "absent_at_snapshot"
        result["note"] = ("the file did not exist when the restore point was taken, "
                          "so restoring removes it again")
        result["note_code"] = "absent_at_snapshot"
        return result
    source = directory / name
    if not source.exists():
        result["effect"] = "unavailable"
        result["reason"] = "the snapshot for this file is missing from the restore point"
        result["reason_code"] = "snapshot_missing"
        return result
    try:
        restored_bytes = source.read_bytes()
    except OSError as exc:
        result["effect"] = "unavailable"
        result["reason"] = f"the snapshot could not be read ({exc.__class__.__name__})"
        result["reason_code"] = "snapshot_unreadable"
        result["reason_detail"] = exc.__class__.__name__
        return result
    if not target.exists():
        result["effect"] = "recreate"
        result["kind"] = "absent_now"
        result["bytes_after_restore"] = len(restored_bytes)
        return result
    try:
        current_bytes = target.read_bytes()
    except OSError as exc:
        result["effect"] = "unavailable"
        result["reason"] = f"the current file could not be read ({exc.__class__.__name__})"
        result["reason_code"] = "current_unreadable"
        result["reason_detail"] = exc.__class__.__name__
        return result
    if current_bytes == restored_bytes:
        result["effect"] = "unchanged"
        result["kind"] = "identical"
        return result
    result["effect"] = "replace"
    try:
        current_text = current_bytes.decode("utf-8")
        restored_text = restored_bytes.decode("utf-8")
    except UnicodeDecodeError:
        result["kind"] = "bytes"
        result["bytes_now"] = len(current_bytes)
        result["bytes_after_restore"] = len(restored_bytes)
        return result
    detail = _json_changes(current_text, restored_text) or _csv_changes(current_text, restored_text)
    if detail is None:
        result["kind"] = "text"
        result["bytes_now"] = len(current_bytes)
        result["bytes_after_restore"] = len(restored_bytes)
        return result
    result.update(detail)
    return result


def preview(restore_id: str) -> dict[str, Any]:
    """The manifest plus, per file, what restoring it would change right now."""
    if not _ID_RE.fullmatch(restore_id):
        raise HTTPException(status_code=404, detail=f"no restore point {restore_id!r}")
    directory = _restore_root() / restore_id
    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        raise HTTPException(status_code=404, detail=f"no restore point {restore_id!r}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500,
                            detail=f"restore point {restore_id} has an unreadable manifest: {exc}") from exc
    files = [_file_preview(entry, directory) for entry in manifest.get("files", [])
             if isinstance(entry, dict)]
    changing = [row for row in files if row.get("effect") not in {"unchanged", "unavailable"}]
    unavailable = [row for row in files if row.get("effect") == "unavailable"]
    return {
        "restore_id": restore_id,
        "batch_id": manifest.get("batch_id"),
        "item_ids": manifest.get("item_ids") or [],
        "created_at": manifest.get("created_at"),
        "files": files,
        "files_changing": len(changing),
        "files_unavailable": len(unavailable),
        "restorable": bool(changing) and not unavailable,
        "reason": None if changing else "nothing would change: every file already matches the restore point",
        "reason_code": None if changing else "nothing_would_change",
    }


# ---------------------------------------------------------------------------
# Routes.
# ---------------------------------------------------------------------------
@router.get("/restore")
def list_restore_points() -> dict[str, Any]:
    """Available restore points with their manifests, newest first."""
    with _LOCK:
        return {"restore_points": manifests()}


@router.get("/restore/{restore_id}")
def read_restore_point(restore_id: str) -> dict[str, Any]:
    """What restoring this point would change, computed against the files as
    they stand now. Reading is open to anyone who may read the assistant; only
    applying the restore needs a write role, which is the same separation the
    reference products draw between viewing history and using it."""
    with _LOCK:
        return preview(restore_id)


@router.post("/restore/{restore_id}")
def restore_state(restore_id: str, request: Request) -> dict[str, Any]:
    """Put the snapshotted state files back byte-for-byte.

    A file that did not exist at snapshot time is removed again, so the
    restore reproduces the exact pre-apply state. Requires a writer role.
    """
    from kairos_api.assistant_actions import _require_writer, audit_append

    user = _require_writer(request)
    if not _ID_RE.fullmatch(restore_id):
        raise HTTPException(status_code=404, detail=f"no restore point {restore_id!r}")
    with _LOCK:
        directory = _restore_root() / restore_id
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail=f"no restore point {restore_id!r}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        restored: list[str] = []
        removed: list[str] = []
        for entry in manifest.get("files", []):
            target = Path(str(entry["path"]))
            if entry.get("existed", True):
                snapshot_path = directory / str(entry["name"])
                if not snapshot_path.exists():
                    raise HTTPException(status_code=500, detail=(
                        f"restore point {restore_id} is missing snapshot {entry['name']!r}"))
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(snapshot_path, target)
                restored.append(str(target))
            elif target.exists():
                target.unlink()
                removed.append(str(target))
        audit_append("restore", user, restore_id=restore_id, batch_id=manifest.get("batch_id"),
                     results={"restored": restored, "removed": removed})
    return {"restore_id": restore_id, "restored": restored, "removed": removed}
