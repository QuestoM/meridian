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

This module is the store. The five HTTP routes over it moved to
:mod:`kairos_api.history_api` in the wave-zero router split; the router and the
route callables still resolve from here, through the module ``__getattr__`` at the
foot of this file, so every existing import and mount keeps working against the
same objects.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, Request

ROOT = Path(__file__).resolve().parents[1]
VERSIONS_DIR_ENV = "KAIROS_VERSIONS_DIR"
ASSISTANT_DIR_ENV = "KAIROS_ASSISTANT_DATA_DIR"
MAX_VERSIONS = 200

# The logical operation-state files. Paths resolve lazily (at call time) so a
# test that monkeypatches a store's PATH and the real deployment both hold.
# TWO REGISTERS, because this was one tuple doing two jobs and the conflation is
# what made it dangerous to extend.
#
# ``_LOGICAL_ORDER`` is THE FULL RESTORE SET: what an operator's named snapshot
# captures and what a restore of that snapshot rolls back, atomically. Adding a
# name here changes what a restore UNDOES, which is why it stays at the nine it
# has always held.
#
# ``_KNOWN_LOGICAL`` is every name this store can capture at all. It is the
# vocabulary, not the restore set.
#
# The bug that forced the split: ``snapshot`` filtered its argument against the
# restore set and returned None for anything left over, so a caller naming a file
# the store had never heard of was indistinguishable from a caller whose file was
# untouched. Two callers had been doing exactly that. campaigns_api_store's own
# docstring said "Version the campaigns store before a manual edit writes it" and
# it versioned nothing; target_store did the same for the plan targets. Both are
# operator-editable through the dashboard, so a manual edit to either had no
# history to restore from and the timeline simply did not show it.
#
# Putting the two names in the restore set instead would have been the obvious
# move and it is the wrong one: restoring a settings version would then also
# revert campaign bookings, which is a far worse thing than the bug being fixed.
# A manual edit records a version holding ONLY the file it touched, and restoring
# that version restores only that file.
_LOGICAL_ORDER = ("settings", "constraints", "overrides", "advertisers", "conditions",
                  "events", "agencies", "agency_links", "agency_conditions")
_KNOWN_LOGICAL = _LOGICAL_ORDER + ("campaigns", "plan_targets", "make_goods")


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
    if logical == "campaigns":
        from kairos_api import campaigns_api_store
        return Path(campaigns_api_store.CAMPAIGNS_PATH)
    if logical == "plan_targets":
        from kairos_api import target_store
        return Path(target_store.TARGETS_PATH)
    if logical == "make_goods":
        from kairos_api import makegood_store
        return Path(makegood_store.MAKE_GOODS_PATH)
    raise ValueError(f"unknown logical file {logical!r}")


def _snapshot_name(logical: str) -> str:
    return f"{logical}{_logical_path(logical).suffix or '.dat'}"


# Auth seam: mirror the assistant action plane so roles behave identically.
def _actor(request: Request | None) -> str:
    from kairos_api import auth
    if not auth.auth_active():
        return "auth-disabled"
    session = auth._session_from_request(request) if request is not None else None
    return str(session["username"]) if session else "anonymous"


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
    byte-identical to it (the edit-burst short-circuit), or None when the caller
    named nothing at all. ``force`` records unconditionally (used for the
    pre_restore safety point and the operator's named snapshot).

    A name this store does not know RAISES. It used to be filtered out and the
    call returned None, which reads at every call site exactly like "nothing had
    changed", so two stores went unversioned without one line of evidence. The
    manual-edit hook still swallows the raise, because a history hiccup must
    never fail an operator's edit; what the raise buys is that a test, a script
    or the assistant sees it immediately.
    """
    named = set(files)
    unknown = sorted(named - set(_KNOWN_LOGICAL))
    if unknown:
        raise ValueError(
            f"version_store does not know {unknown!r}; add it to _KNOWN_LOGICAL and "
            "_logical_path, or the edit it guards is versioned nowhere")
    wanted = [name for name in _KNOWN_LOGICAL if name in named]
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


# Diffs live in kairos_api.version_store_diff (size cap). Every name the diff
# section used to define is re-exported here, against the SAME objects, so this
# split follows the rule the router split already set and
# tests/test_w0_1_module_seams.py holds: a module that splits keeps answering to
# every name it answered to before, or the split is a rename in disguise.
from kairos_api.version_store_diff import (  # noqa: E402,F401
    _ID_COLUMN,
    _current_bytes,
    _diff_logical,
    _read_json,
    _read_rows,
    _rows_diff,
    _settings_diff,
    _version_bytes,
)


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


# The HTTP layer over this store moved to :mod:`kairos_api.history_api` in the
# wave-zero router split. These names still resolve from here, against the SAME
# objects, so every existing import, mount and test fixture keeps working.
# Resolution is lazy: the store never imports its own router at module load,
# which would be an import cycle.
_ROUTE_LAYER_NAMES = (
    "router",
    "list_versions",
    "version_diff",
    "restore_version",
    "create_snapshot",
    "rename_version",
    "RestoreRequest",
    "LabelRequest",
    "_public_entry",
    "_SCOPE_NOTE",
    "_require_session",
    "_require_writer",
    "_require_version_id",
    "_VERSION_ID_RE",
)


def __getattr__(name: str) -> Any:
    if name in _ROUTE_LAYER_NAMES:
        from kairos_api import history_api

        return getattr(history_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_ROUTE_LAYER_NAMES))
