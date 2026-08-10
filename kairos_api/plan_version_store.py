"""Plan versions: the internal freeze of a saved weekly plan.

The weekly plan was the one operational artifact the product never versioned.
The nine logical files the operation-state store captures
(:mod:`kairos_api.version_store`) are the mutable stores; the plan itself,
``output/weekly_break_schedule.csv``, was overwritten in place on every run, so
"which plan is everyone downstream reading" had no answer and a run could not be
rolled back.

A plan version is an internal freeze, per the owner ruling of 2026-08-01: a
planner names it alone, it records who froze it and when, and it can be diffed
against the one before it and restored. Publishing is not a broadcast and not an
approval workflow; it is the moment a plan stops moving.

Two honesty rules hold this module.

- **The frozen bytes are the plan.** A version stores the CSV verbatim and its
  sha256, so a restore is byte-identical and a diff is computed from the frozen
  file rather than from a summary somebody wrote down.
- **A version's headline money is the operator's, and says so.** The saved plan
  carries every channel because the retention model is measured against the
  competitive lineup. The totals recorded here are scoped to
  ``settings.operator_channel`` through :mod:`kairos_api.channel_scope`, and the
  scope note travels with them. With no configured channel the totals are the
  whole file and the note says exactly that.

The store lives beside the operation-state versions, under ``data/plan_versions``
by default and wherever ``KAIROS_PLAN_VERSIONS_DIR`` points when it is set, which
is how a test relocates it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from kairos_api import channel_scope

ROOT = Path(__file__).resolve().parents[1]
PLAN_VERSIONS_DIR_ENV = "KAIROS_PLAN_VERSIONS_DIR"
MAX_PLAN_VERSIONS = 100
PLAN_FILENAME = "plan.csv"
MANIFEST_FILENAME = "manifest.json"
META_FILENAME = "plan.meta.json"

_NAME_MAX = 80


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def versions_root() -> Path:
    raw = os.environ.get(PLAN_VERSIONS_DIR_ENV, "").strip()
    return Path(raw) if raw else ROOT / "data" / "plan_versions"


def plan_path() -> Path:
    """The live saved plan, resolved at call time so a test can relocate it."""
    from kairos_api import core

    return Path(core.OUTPUT_DIR) / "weekly_break_schedule.csv"


def meta_path() -> Path:
    path = plan_path()
    return path.with_name(path.name + ".meta.json")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if number == number else default


def clean_name(value: Any) -> str:
    """A version name a person typed, trimmed and bounded. Never invented."""
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    return text[:_NAME_MAX]


def _totals(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"rows": 0, "breaks": 0, "ad_seconds": 0, "revenue": 0.0,
                "channels": 0, "days": 0, "date_from": None, "date_to": None}
    breaks = pd.to_numeric(frame.get("num_breaks", 0), errors="coerce").fillna(0)
    seconds = pd.to_numeric(frame.get("total_break_time", 0), errors="coerce").fillna(0)
    revenue = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    dates = frame["date"].astype(str) if "date" in frame.columns else pd.Series(dtype=str)
    return {
        "rows": int(len(frame)),
        "breaks": int(breaks.sum()),
        "ad_seconds": int(seconds.sum()),
        "revenue": round(float(revenue.sum()), 2),
        "channels": int(frame["channel"].nunique()) if "channel" in frame.columns else 0,
        "days": int(dates.nunique()) if len(dates) else 0,
        "date_from": str(dates.min()) if len(dates) else None,
        "date_to": str(dates.max()) if len(dates) else None,
    }


def _read_meta() -> dict[str, Any]:
    path = meta_path()
    if not path.exists():
        return {}
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _settings_basis() -> dict[str, Any]:
    """The operator decision the frozen plan was run under, named field by field."""
    from kairos_api.core import _load_settings

    settings = _load_settings()
    return {
        "revenue_weight": int(settings.revenue_weight),
        "min_retention_floor": float(settings.min_retention_floor),
        "max_breaks_per_hour": int(settings.max_breaks_per_hour),
        "risk_lambda": float(settings.risk_lambda),
        "objective_mode": str(getattr(settings, "objective_mode", "blend") or "blend"),
        "operator_channel": str(settings.operator_channel or "") or None,
    }


def _summarize(frame: pd.DataFrame) -> dict[str, Any]:
    """Whole-file totals plus the operator-scoped totals and the scope note."""
    owned, note = channel_scope.scope_frame(frame)
    return {
        "owned": _totals(owned),
        "all_channels": _totals(frame),
        "scope": note,
    }


def _owned_delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, Any]:
    """The operator totals that moved, current minus previous."""
    return {
        "rows": int(_number(current.get("rows"))) - int(_number(previous.get("rows"))),
        "breaks": int(_number(current.get("breaks"))) - int(_number(previous.get("breaks"))),
        "ad_seconds": int(_number(current.get("ad_seconds"))) - int(_number(previous.get("ad_seconds"))),
        "revenue": round(_number(current.get("revenue")) - _number(previous.get("revenue")), 2),
    }


def collapse_against_latest(
    live_summary: Optional[dict[str, Any]] = None,
    manifests: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
    """Whether the live operator plan fell to zero from the newest freeze.

    Breaks and money are checked separately. Either falling from a positive
    value to zero is a collapse that needs an explicit confirmation. With no
    prior freeze, an absolute zero still needs confirmation: the absence of a
    comparison must not make a zero plan safe to publish.
    """
    if live_summary is None:
        state = live_state()
        live_summary = state.get("summary")
    items = all_manifests() if manifests is None else manifests
    current = (live_summary or {}).get("owned") if isinstance(live_summary, dict) else None
    if not isinstance(current, dict):
        return {"available": False, "collapsed": False, "reason": "live operator totals are unavailable"}

    latest = items[0] if items else None
    previous = ((latest or {}).get("summary") or {}).get("owned") if latest else None
    current_breaks = int(_number(current.get("breaks")))
    current_revenue = _number(current.get("revenue"))
    previous_breaks = int(_number((previous or {}).get("breaks"))) if isinstance(previous, dict) else None
    previous_revenue = _number((previous or {}).get("revenue")) if isinstance(previous, dict) else None
    breaks_collapsed = current_breaks == 0 and (previous_breaks is None or previous_breaks > 0)
    revenue_collapsed = current_revenue == 0 and (previous_revenue is None or previous_revenue > 0)
    return {
        "available": True,
        "collapsed": breaks_collapsed or revenue_collapsed,
        "breaks_collapsed": breaks_collapsed,
        "revenue_collapsed": revenue_collapsed,
        "current": current,
        "previous": previous,
        "against_version_id": (latest or {}).get("version_id"),
        "against_name": (latest or {}).get("name"),
        "delta": _owned_delta(current, previous) if isinstance(previous, dict) else None,
    }


def live_state() -> dict[str, Any]:
    """What a freeze would capture right now, and whether it already is frozen.

    ``frozen_as`` is the version id whose frozen bytes are identical to the plan
    on disk, or None when the live plan has moved since the last freeze. It is a
    sha256 comparison, not a timestamp, so a re-run that produced the same plan
    reads as already frozen rather than as a change.
    """
    path = plan_path()
    state: dict[str, Any] = {"exists": path.exists(), "path": str(path)}
    if not path.exists():
        return state
    payload = path.read_bytes()
    digest = _sha256(payload)
    state["sha256"] = digest
    state["bytes"] = len(payload)
    state["computed_at"] = _read_meta().get("computed_at")
    manifests = all_manifests()
    state["frozen_as"] = next(
        (str(item.get("version_id")) for item in manifests if item.get("plan_sha256") == digest),
        None,
    )
    # WHAT A FREEZE WOULD CAPTURE, IN FIGURES, AND NOT ONLY WHETHER IT IS FROZEN.
    # Without this the publish panel could only compare AFTER the act, by opening
    # the diff, and a warning that arrives after the freeze is not a warning. A
    # blind critic measured a plan collapsed to zero breaks and zero shekels on
    # the operator's own channel being named and published with the version row
    # rendering in the same neutral type as its neighbours.
    try:
        state["summary"] = _summarize(pd.read_csv(path))
        state["collapse"] = collapse_against_latest(state["summary"], manifests)
    except Exception as exc:  # pragma: no cover - a plan that will not parse
        # Honest unknown rather than a fabricated zero: a plan whose totals
        # cannot be read must not present as a plan worth nothing.
        state["summary"] = None
        state["summary_reason"] = "the saved plan could not be read for totals: %s" % exc
    return state


def all_manifests() -> list[dict[str, Any]]:
    """Every frozen version, newest first. A directory without a manifest is skipped."""
    root = versions_root()
    if not root.exists():
        return []
    found: list[dict[str, Any]] = []
    for directory in root.iterdir():
        manifest = directory / MANIFEST_FILENAME
        if not directory.is_dir() or not manifest.exists():
            continue
        try:
            found.append(json.loads(manifest.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            continue
    found.sort(key=lambda item: (str(item.get("created_at", "")), int(item.get("seq", 0))), reverse=True)
    return found


def get(version_id: str) -> Optional[dict[str, Any]]:
    for manifest in all_manifests():
        if str(manifest.get("version_id")) == str(version_id):
            return manifest
    return None


def _next_seq(manifests: list[dict[str, Any]]) -> int:
    return max((int(item.get("seq", 0)) for item in manifests), default=0) + 1


def _prune() -> list[str]:
    pruned: list[str] = []
    for manifest in all_manifests()[MAX_PLAN_VERSIONS:]:
        version_id = str(manifest.get("version_id", ""))
        directory = versions_root() / version_id
        if version_id and directory.is_dir():
            shutil.rmtree(directory, ignore_errors=True)
            pruned.append(version_id)
    return pruned


def freeze(name: str, actor: str, note: str = "", source: str = "publish") -> dict[str, Any]:
    """Freeze the live saved plan as a named version and return its manifest.

    Raises ``FileNotFoundError`` when there is no saved plan to freeze and
    ``ValueError`` when the name is empty, because a version nobody can name is
    a version nobody can find again.
    """
    clean = clean_name(name)
    if not clean:
        raise ValueError("a plan version needs a name")
    path = plan_path()
    if not path.exists():
        raise FileNotFoundError(str(path))
    payload = path.read_bytes()
    frame = pd.read_csv(path)
    manifests = all_manifests()
    version_id = uuid.uuid4().hex[:12]
    directory = versions_root() / version_id
    directory.mkdir(parents=True, exist_ok=True)
    _atomic_write(directory / PLAN_FILENAME, payload)
    meta = _read_meta()
    if meta:
        _atomic_write(directory / META_FILENAME, json.dumps(meta, ensure_ascii=False, indent=1).encode("utf-8"))
    summary = _summarize(frame)
    previous_owned = (((manifests[0] if manifests else {}).get("summary") or {}).get("owned"))
    manifest = {
        "version_id": version_id,
        "seq": _next_seq(manifests),
        "name": clean,
        "note": clean_name(note) if note else "",
        "created_at": _now_iso(),
        "actor": actor,
        "source": source,
        "plan_sha256": _sha256(payload),
        "plan_bytes": len(payload),
        # The engine's own provenance for the frozen file: when it was run and
        # the fingerprints of every input it read. Empty when the sidecar is
        # missing, which reads downstream as an unknown rather than a guess.
        "computed_at": meta.get("computed_at"),
        "input_fingerprints": meta.get("fingerprints") or {},
        "settings_basis": _settings_basis(),
        "summary": summary,
        "previous_version_id": str(manifests[0].get("version_id")) if manifests else None,
        "owned_delta_from_previous": (
            _owned_delta(summary["owned"], previous_owned)
            if isinstance(previous_owned, dict) else None
        ),
    }
    _atomic_write(directory / MANIFEST_FILENAME,
                  json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))
    _prune()
    return manifest


def _frame_for(version_id: str) -> Optional[pd.DataFrame]:
    path = versions_root() / str(version_id) / PLAN_FILENAME
    if not path.exists():
        return None
    return pd.read_csv(path)


def _by_day(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    if frame.empty or "date" not in frame.columns:
        return {}
    work = frame.copy()
    work["_breaks"] = pd.to_numeric(work.get("num_breaks", 0), errors="coerce").fillna(0)
    work["_revenue"] = pd.to_numeric(work.get("predicted_revenue", 0), errors="coerce").fillna(0)
    grouped = work.groupby(work["date"].astype(str)).agg(breaks=("_breaks", "sum"), revenue=("_revenue", "sum"))
    return {
        str(index): {"breaks": int(row["breaks"]), "revenue": round(float(row["revenue"]), 2)}
        for index, row in grouped.iterrows()
    }


def diff(version_id: str, against: Optional[str] = None) -> dict[str, Any]:
    """What changed between one frozen version and another, or the live plan.

    ``against`` defaults to the version's own recorded predecessor. Passing
    ``"live"`` diffs the frozen version against the plan on disk right now,
    which is what a planner reads before restoring. Every figure is the
    operator-scoped one and carries the scope it was summed on.
    """
    manifest = get(version_id)
    if manifest is None:
        return {"available": False, "reason_code": "unknown_version", "reason": f"no plan version {version_id}"}
    frame = _frame_for(version_id)
    if frame is None:
        return {"available": False, "reason_code": "no_frozen_file", "reason": f"plan version {version_id} has no frozen file"}

    if against == "live":
        other_label = "live"
        other_manifest: dict[str, Any] = {"name": "live", "version_id": "live"}
        path = plan_path()
        other_frame = pd.read_csv(path) if path.exists() else None
    else:
        other_label = str(against or manifest.get("previous_version_id") or "")
        if not other_label:
            return {"available": False, "reason_code": "first_version", "reason": "this is the first plan version, so there is nothing before it"}
        found = get(other_label)
        if found is None:
            return {"available": False, "reason_code": "unknown_version", "reason": f"no plan version {other_label}"}
        other_manifest = found
        other_frame = _frame_for(other_label)
    if other_frame is None:
        return {"available": False, "reason_code": "no_frozen_file", "reason": "the other side of the comparison has no frozen file"}

    owned, note = channel_scope.scope_frame(frame)
    other_owned, _ = channel_scope.scope_frame(other_frame)
    totals = _totals(owned)
    other_totals = _totals(other_owned)
    days = _by_day(owned)
    other_days = _by_day(other_owned)
    changed_days = []
    for date in sorted(set(days) | set(other_days)):
        here = days.get(date, {"breaks": 0, "revenue": 0.0})
        there = other_days.get(date, {"breaks": 0, "revenue": 0.0})
        if here == there:
            continue
        changed_days.append({
            "date": date,
            "breaks": here["breaks"],
            "breaks_before": there["breaks"],
            "breaks_delta": here["breaks"] - there["breaks"],
            "revenue": here["revenue"],
            "revenue_before": there["revenue"],
            "revenue_delta": round(here["revenue"] - there["revenue"], 2),
        })
    return {
        "available": True,
        "version_id": str(version_id),
        "version_name": manifest.get("name"),
        "against": other_label,
        "against_name": other_manifest.get("name"),
        "scope": note,
        "totals": totals,
        "totals_before": other_totals,
        "delta": {
            "rows": totals["rows"] - other_totals["rows"],
            "breaks": totals["breaks"] - other_totals["breaks"],
            "ad_seconds": totals["ad_seconds"] - other_totals["ad_seconds"],
            "revenue": round(totals["revenue"] - other_totals["revenue"], 2),
        },
        "changed_days": changed_days,
        "identical": not changed_days and totals == other_totals,
    }


def restore(version_id: str, actor: str) -> dict[str, Any]:
    """Put a frozen plan back as the live saved plan, freezing the current one first.

    The rollback is byte-identical: the archived bytes are written back verbatim,
    and the archived freshness sidecar with them, so the plan and its provenance
    move together. The plan on disk before the restore is frozen first under an
    automatic name, so a rollback is itself reversible.
    """
    manifest = get(version_id)
    if manifest is None:
        raise KeyError(version_id)
    source = versions_root() / str(version_id) / PLAN_FILENAME
    if not source.exists():
        raise FileNotFoundError(str(source))

    safety: Optional[dict[str, Any]] = None
    if plan_path().exists():
        safety = freeze(
            name=f"before restoring {manifest.get('name')}",
            actor=actor,
            note=f"automatic freeze taken before restoring plan version {version_id}",
            source="pre_restore",
        )

    payload = source.read_bytes()
    _atomic_write(plan_path(), payload)
    archived_meta = versions_root() / str(version_id) / META_FILENAME
    if archived_meta.exists():
        _atomic_write(meta_path(), archived_meta.read_bytes())

    from kairos_api.core import _read_csv_cached

    _read_csv_cached.cache_clear()
    return {
        "ok": True,
        "restored": str(version_id),
        "name": manifest.get("name"),
        "plan_sha256": _sha256(payload),
        "safety_version_id": (safety or {}).get("version_id"),
        "bytes": len(payload),
    }
