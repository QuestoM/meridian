"""Binding writes: compiled agreement artifacts enter the live rule stores.

The one module that mutates existing stores on behalf of an agreement, and the
four rules that keep it safe:

- **Snapshot before write.** Every touched logical store is versioned through
  the unified version store first (source ``trade_approve`` /
  ``trade_decompile``), so an approval is as restorable as a manual edit.
- **Idempotent per agreement.** A write first removes every row whose rule_id
  belongs to this agreement (``TRD:<agreement>:``), then appends the new
  version's rows: re-approval replaces, supersession removes, and two
  agreements never touch each other's rows.
- **A row the target store cannot hold is REFUSED BY NAME.** The agency
  conditions store has no weekday column; writing a weekday-scoped row there
  would silently widen it to every weekday, which is worse than not binding.
  Refusals return to the approve response beside the compiler's own skips.
- **Byte identity when nothing is approved.** With no agreement rows, a
  bind-remove cycle leaves every store byte-identical; the test suite pins it.
"""

from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from kairos.trade.compile import CompiledArtifacts, RULE_PREFIX

_BACKUP_DIRNAME = "_backups"


def _store_targets() -> dict[str, dict[str, Any]]:
    """The three writable stores, resolved lazily so tests can relocate them."""
    from kairos.optimize._frequency_rules import DEFAULT_FREQUENCY_PATH
    from kairos_api import advertiser_conditions, agency_conditions

    return {
        "advertiser_conditions": {
            "path": Path(advertiser_conditions.CONDITIONS_PATH),
            "columns": list(advertiser_conditions.COLUMNS),
            "logical": "conditions",
            "key_column": "rule_id",
        },
        "agency_conditions": {
            "path": Path(agency_conditions.CONDITIONS_PATH),
            "columns": list(agency_conditions.CONDITION_COLUMNS),
            "logical": "agency_conditions",
            "key_column": "rule_id",
        },
        "frequency_rules": {
            "path": Path(DEFAULT_FREQUENCY_PATH),
            "columns": [
                "rule_id", "limit_type", "scope", "advertiser_id", "campaign",
                "ad", "pair_lead", "pair_closer", "competing_group", "members",
                "value", "value_max", "unit", "enabled", "notes",
            ],
            "logical": "frequency_rules",
            "key_column": "rule_id",
        },
    }


def _read_store(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in columns:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _write_store(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    if path.exists():
        backup_dir = path.parent / _BACKUP_DIRNAME
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        shutil.copy2(path, backup_dir / f"{path.stem}_{stamp}{path.suffix}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    frame[columns].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, path)


def _agreement_prefix(agreement_id: str) -> str:
    return f"{RULE_PREFIX}:{agreement_id}:"


def _refuse_unrepresentable(row: dict[str, Any], columns: list[str]) -> Optional[str]:
    """A scoped value the store has no column for must refuse, not widen."""
    for key, value in row.items():
        if key.startswith("_") or key in columns:
            continue
        text = str(value or "").strip()
        if key.startswith("scope_") and text and text != "ANY":
            return (
                f"המחסן היעד אינו שומר את המימד '{key}'; כתיבה בלעדיו הייתה "
                "מרחיבה את הכלל מעבר לחוזה"
            )
    return None


def _snapshot(logicals: list[str], actor: str, source: str, label: str) -> Optional[str]:
    from kairos_api import version_store

    try:
        return version_store.snapshot(
            source=source, actor=actor, files=sorted(set(logicals)), label=label,
        )
    except Exception:  # noqa: BLE001 - history is additive, never fail the bind
        return None


def bind(artifacts: CompiledArtifacts, actor: str) -> dict[str, Any]:
    """Write one agreement version's artifacts into the live stores.

    Returns {written: {store: n}, refused: [...], snapshot_version}. Rows of
    the SAME agreement from any earlier version are replaced wholesale.
    """
    targets = _store_targets()
    prefix = _agreement_prefix(artifacts.agreement_id)

    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in targets}
    refused: list[dict[str, Any]] = []
    for row in artifacts.conditions:
        store = str(row.get("_store", "advertiser_conditions"))
        reason = _refuse_unrepresentable(row, targets[store]["columns"])
        if reason is not None:
            refused.append({"rule_id": row.get("rule_id"), "store": store,
                            "reason_he": reason})
            continue
        grouped[store].append(row)
    for row in artifacts.frequency_rules:
        grouped["frequency_rules"].append(row)

    touched = [targets[name]["logical"] for name, rows in grouped.items() if rows]
    # Stores that might hold OLD rows of this agreement are touched too, even
    # when the new version writes nothing there — the removal is a write.
    for name, target in targets.items():
        frame = _read_store(target["path"], target["columns"])
        if frame["rule_id"].astype(str).str.startswith(prefix).any():
            touched.append(target["logical"])
    snapshot_version = _snapshot(
        touched, actor, "trade_approve",
        f"הסכם {artifacts.agreement_id} גרסה {artifacts.version_id}",
    ) if touched else None

    written: dict[str, int] = {}
    replaced: dict[str, int] = {}
    for name, target in targets.items():
        rows = grouped[name]
        frame = _read_store(target["path"], target["columns"])
        keep = ~frame["rule_id"].astype(str).str.startswith(prefix)
        removed = int((~keep).sum())
        if not rows and removed == 0:
            continue
        frame = frame[keep]
        if rows:
            addition = pd.DataFrame(rows)
            for column in target["columns"]:
                if column not in addition.columns:
                    addition[column] = ""
            frame = pd.concat([frame, addition[target["columns"]].astype(str)],
                              ignore_index=True)
        _write_store(target["path"], frame, target["columns"])
        if rows:
            written[name] = len(rows)
        if removed:
            replaced[name] = removed
    return {
        "written": written,
        "replaced": replaced,
        "refused": refused,
        "snapshot_version": snapshot_version,
    }


def unbind(agreement_id: str, actor: str) -> dict[str, Any]:
    """Remove every live rule row an agreement put in place (supersede/expire)."""
    targets = _store_targets()
    prefix = _agreement_prefix(agreement_id)
    removed: dict[str, int] = {}
    touched: list[str] = []
    for name, target in targets.items():
        frame = _read_store(target["path"], target["columns"])
        mask = frame["rule_id"].astype(str).str.startswith(prefix)
        if not mask.any():
            continue
        touched.append(target["logical"])
        removed[name] = int(mask.sum())
    snapshot_version = _snapshot(
        touched, actor, "trade_decompile", f"הסרת כללי הסכם {agreement_id}",
    ) if touched else None
    for name, target in targets.items():
        if name not in removed:
            continue
        frame = _read_store(target["path"], target["columns"])
        frame = frame[~frame["rule_id"].astype(str).str.startswith(prefix)]
        _write_store(target["path"], frame, target["columns"])
    return {"removed": removed, "snapshot_version": snapshot_version}


def bound_rules(agreement_id: str) -> dict[str, list[dict[str, Any]]]:
    """The live rows an agreement currently holds in each store."""
    targets = _store_targets()
    prefix = _agreement_prefix(agreement_id)
    out: dict[str, list[dict[str, Any]]] = {}
    for name, target in targets.items():
        frame = _read_store(target["path"], target["columns"])
        mask = frame["rule_id"].astype(str).str.startswith(prefix)
        if mask.any():
            out[name] = frame[mask].to_dict(orient="records")
    return out
