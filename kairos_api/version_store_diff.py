"""Diffs over the version timeline: live state versus a chosen version.

Split out of :mod:`kairos_api.version_store` under the file-size law. Nothing
here changed in the move; the store re-exports :func:`_diff_logical` under its
own name, so ``version_store._diff_logical`` is this function and both of its
callers (``history_api`` and ``history_api_files``) read exactly as they did.

A diff answers one question and it is worth stating precisely, because it is the
opposite of what the direction of the words suggests: it reports CURRENT state
versus the chosen version, so ``from`` is what is on disk now and ``to`` is what
restoring that version would put there. It is the change a restore would make,
not the change that produced the version.

Which shape a logical file diffs in depends on what it is. Settings is a JSON
document, so it diffs field by field. Everything else is a row store keyed by an
id column, so it diffs by row and reports added, removed and changed
separately. The advertisers store diffs names only, because its rows carry
commercial terms wide enough that a whole-row dump is unreadable.
"""

from __future__ import annotations

import csv
import json
from typing import Any, Optional

# The id column each row store is keyed by. plan_targets has none: a row there is
# keyed by channel, period and metric together, so a diff of it would be a
# whole-file diff rather than a row diff and it is deliberately absent.
_ID_COLUMN = {"constraints": "constraint_id", "overrides": "override_id",
              "advertisers": "advertiser_id", "conditions": "rule_id",
              "events": "event_id", "agencies": "agency_id",
              "agency_links": "agency_id", "agency_conditions": "rule_id",
              "campaigns": "campaign_id", "make_goods": "make_good_id"}


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
    from kairos_api import version_store

    manifest = version_store._read_manifest(version_id)
    for entry in manifest.get("files", []):
        if entry.get("logical") == logical:
            if not entry.get("existed"):
                return None
            path = version_store._versions_root() / version_id / str(entry.get("name"))
            return path.read_bytes() if path.exists() else None
    return None


def _current_bytes(logical: str) -> Optional[bytes]:
    from kairos_api import version_store

    path = version_store._logical_path(logical)
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
