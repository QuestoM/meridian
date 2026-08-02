"""Manual-overrides CRUD plus an honest WITH-vs-WITHOUT effect preview.

This is the operator-facing seam for the manual override layer
(:mod:`kairos.optimize.overrides`). It persists overrides to
``data/manual_overrides.csv`` with the same read-mutate-backup-write style as
:mod:`kairos_api.advertisers` (serialized under a module lock, written via a
temp file plus ``os.replace`` so readers never see a torn CSV), and it serves a
preview that runs the break-count optimizer with and without the overrides so
the operator can see exactly what changes and which overrides were rejected as
infeasible.

Honesty rules: an override kind is validated against its scope before it is
stored; the effect preview reports rejected overrides verbatim from the
optimizer (never hiding an infeasible one); and a preview that cannot build real
segments says so rather than inventing a delta. The preview runs through the
SAME engine seams the weekly recompute uses (saved settings, first-break fold,
wrapped classifier, stored constraints, demand fold), so its numbers are the
plan the commit would write, not a parallel engine.
"""

from __future__ import annotations

import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from kairos.optimize.overrides import (
    COLUMNS,
    DEFAULT_SOURCE,
    DEFAULT_STATUS,
    SEGMENT,
    SPOT,
    Override,
    OverrideSet,
    _SEGMENT_KINDS,
    _SPOT_KINDS,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
OVERRIDES_PATH = DATA_DIR / "manual_overrides.csv"

router = APIRouter(prefix="/api/overrides", tags=["overrides"])

# Serializes every load-mutate-write cycle on the overrides CSV so two
# concurrent edits cannot drop each other's rows (lost update).
_STORE_LOCK = threading.Lock()


class OverrideCreate(BaseModel):
    """A new operator override. scope and kind must agree.

    The trailing fields are the additive decision-plane extension: ``source``
    records where the override came from (lever, recommendation, or manual),
    ``rec_id`` links back to an approved recommendation, ``status`` tracks the
    lifecycle, and the ``anchor_*`` trio is the semantic anchor stored beside the
    build-order ``target_id`` so a later resolve can confirm the override still
    binds to the same real break after a re-ingest.
    """

    scope: str
    target_id: str
    kind: str
    value: str = ""
    gold: bool = False
    notes: str = ""
    source: str = DEFAULT_SOURCE
    rec_id: str = ""
    status: str = DEFAULT_STATUS
    anchor_date: str = ""
    anchor_start: str = ""
    anchor_title: str = ""


class OverrideUpdate(BaseModel):
    """Editable fields for an override. All optional for PATCH-style PUT."""

    scope: str | None = None
    target_id: str | None = None
    kind: str | None = None
    value: str | None = None
    gold: bool | None = None
    notes: str | None = None


def segment_clock(start_seconds: float) -> str:
    """Format a segment start (seconds past midnight) as the HH:MM the weekly CSV
    stores. Matches kairos.export.schedule._clock so a stored anchor built from the
    CSV start_time compares equal to a freshly built segment's clock.

    Public because every route that stores an anchor has to build the middle field
    with this exact formatter. A route that stored the date and the programme but
    left the clock blank wrote an anchor the guard below reads as a mismatch, so
    the override never bound and the act silently did nothing."""
    total_minutes = int(start_seconds // 60)
    return f"{(total_minutes // 60) % 24:02d}:{total_minutes % 60:02d}"


def _load_frame() -> pd.DataFrame:
    if not OVERRIDES_PATH.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    frame = pd.read_csv(OVERRIDES_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _backup() -> None:
    if not OVERRIDES_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(OVERRIDES_PATH, BACKUP_DIR / f"manual_overrides_{stamp}.csv")


def _write_frame(frame: pd.DataFrame) -> None:
    """Backup, then write atomically (temp file + os.replace, like auth_store).

    A reader that opens the CSV mid-write sees either the old or the new file,
    never a truncated one. Callers hold ``_STORE_LOCK`` across load-mutate-write.
    """
    _backup()
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OVERRIDES_PATH.with_name(OVERRIDES_PATH.name + ".tmp")
    frame[list(COLUMNS)].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, OVERRIDES_PATH)


def _snapshot_before_write(request: "Request | None") -> None:
    """Record a version of the overrides store before a manual edit writes it."""
    from kairos_api import version_store

    version_store.snapshot_manual_edit(request, "overrides")


def _record(row: "pd.Series[Any]") -> dict[str, Any]:
    return {column: str(row.get(column, "")) for column in COLUMNS}


def _validate(scope: str, kind: str) -> tuple[str, str]:
    scope_clean = str(scope or "").strip().lower()
    kind_clean = str(kind or "").strip().lower()
    if scope_clean == SEGMENT and kind_clean in _SEGMENT_KINDS:
        return scope_clean, kind_clean
    if scope_clean == SPOT and kind_clean in _SPOT_KINDS:
        return scope_clean, kind_clean
    raise HTTPException(
        status_code=400,
        detail=(
            f"kind '{kind}' is not valid for scope '{scope}'. "
            f"segment kinds: {sorted(_SEGMENT_KINDS)}; spot kinds: {sorted(_SPOT_KINDS)}"
        ),
    )


@router.get("")
def list_overrides() -> dict[str, Any]:
    """All overrides grouped by scope, plus the raw column order."""
    frame = _load_frame()
    grouped: dict[str, list[dict[str, Any]]] = {SEGMENT: [], SPOT: []}
    for _, row in frame.iterrows():
        record = _record(row)
        grouped.setdefault(record.get("scope", ""), []).append(record)
    return {"overrides": grouped, "columns": list(COLUMNS)}


@router.post("", status_code=201)
def create_override(payload: OverrideCreate, request: Request = None) -> dict[str, Any]:
    scope, kind = _validate(payload.scope, payload.kind)
    if not str(payload.target_id or "").strip():
        raise HTTPException(status_code=400, detail="target_id is required")
    new_row = {
        "override_id": uuid.uuid4().hex[:12],
        "scope": scope,
        "target_id": str(payload.target_id).strip(),
        "kind": kind,
        "value": str(payload.value or ""),
        "gold": str(bool(payload.gold)),
        "notes": str(payload.notes or ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(payload.source or "").strip().lower() or DEFAULT_SOURCE,
        "rec_id": str(payload.rec_id or "").strip(),
        "status": str(payload.status or "").strip().lower() or DEFAULT_STATUS,
        "anchor_date": str(payload.anchor_date or "").strip(),
        "anchor_start": str(payload.anchor_start or "").strip(),
        "anchor_title": str(payload.anchor_title or "").strip(),
    }
    with _STORE_LOCK:
        frame = _load_frame()
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        _snapshot_before_write(request)
        _write_frame(frame)
        return _record(frame.iloc[-1])


def _locate(frame: pd.DataFrame, override_id: str) -> int:
    mask = frame["override_id"].astype(str) == override_id
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"override '{override_id}' not found")
    return int(frame.index[mask][0])


@router.put("/{override_id}")
def update_override(override_id: str, payload: OverrideUpdate,
                    request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_frame()
        index = _locate(frame, override_id)
        scope = payload.scope if payload.scope is not None else str(frame.at[index, "scope"])
        kind = payload.kind if payload.kind is not None else str(frame.at[index, "kind"])
        if payload.scope is not None or payload.kind is not None:
            scope, kind = _validate(scope, kind)
            frame.at[index, "scope"] = scope
            frame.at[index, "kind"] = kind
        if payload.target_id is not None:
            frame.at[index, "target_id"] = str(payload.target_id).strip()
        if payload.value is not None:
            frame.at[index, "value"] = str(payload.value)
        if payload.gold is not None:
            frame.at[index, "gold"] = str(bool(payload.gold))
        if payload.notes is not None:
            frame.at[index, "notes"] = str(payload.notes)
        _snapshot_before_write(request)
        _write_frame(frame)
        return _record(frame.loc[index])


@router.delete("/{override_id}")
def delete_override(override_id: str, request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_frame()
        index = _locate(frame, override_id)
        frame = frame.drop(index=index).reset_index(drop=True)
        _snapshot_before_write(request)
        _write_frame(frame)
    return {"deleted": override_id}


def _stored_constraints() -> list:
    """The operator's stored placement constraints, from the same file the weekly
    recompute reads. Empty when the store was never created, so the preview stays
    byte-identical to a deployment without constraints. Resolved lazily so a
    test-relocated store path is honoured."""
    from kairos.optimize.constraints_store import load_constraints
    from kairos_api import constraints as constraints_api

    return load_constraints(constraints_api.CONSTRAINTS_PATH)


def _segment_anchors(segments: list) -> dict[str, tuple[str, str, str]]:
    """Each built segment's semantic anchor (date, start clock, program), the
    same triple the commit-time guard compares stored overrides against."""
    return {
        segment.segment_id: (
            str(segment.day),
            segment_clock(segment.start_seconds),
            str(segment.program_type),
        )
        for segment in segments
    }


def _resolved_store_overrides(segments: list) -> tuple[OverrideSet, list[dict[str, Any]]]:
    """The stored override set resolved against these segments' anchors: the
    active (binding) set plus the stale reports. The same re-ingest guard the
    commit runs, so a stale-anchored override can never silently rebind to a
    different-but-valid break; a blank-anchor (legacy) override still binds."""
    overrides = OverrideSet.from_csv(OVERRIDES_PATH)
    return overrides.resolve_against_segments(_segment_anchors(segments))


@router.get("/effect")
def override_effect(
    channel: str | None = None,
    day: str | None = None,
    daily_input: str | None = None,
    target_id: str | None = None,
    kind: str | None = None,
    value: str | None = None,
    gold: bool = False,
    scope: str | None = None,
) -> dict[str, Any]:
    """Preview the optimizer WITH vs WITHOUT the stored overrides.

    Runs the break-count optimizer twice on the same channel-day segments, once
    plain and once with the OverrideSet, and reports per-segment break-count
    deltas plus any rejected (infeasible) overrides. This is honest about where
    overrides bite: it only reflects segment-scope overrides, since those are the
    ones the weekly break-count optimizer consumes.

    Both legs run through :func:`kairos.optimize.day_core._optimize_one_day`, the
    commit path's own core, with the stored placement constraints and the demand
    fold applied on BOTH sides, so the delta isolates the overrides and the
    absolute numbers are the plan the weekly recompute would write.

    Candidate mode: with ``target_id`` (a ``day|channel|index`` segment id) and
    ``kind``, the preview scopes itself to that segment's channel-day and
    compares the plan WITH the stored overrides against the plan WITH stored
    overrides PLUS this one candidate, so the delta isolates the decision being
    considered before it is saved. Nothing is written.
    """
    from kairos.optimize.day_core import _optimize_one_day
    from kairos_api.preview_inputs import preview_inputs

    candidate: Override | None = None
    if target_id:
        if scope not in (None, "", "segment"):
            raise HTTPException(status_code=422, detail="Candidate preview supports scope=segment only")
        parts = str(target_id).split("|")
        if len(parts) != 3 or not parts[0] or not parts[1]:
            raise HTTPException(status_code=422, detail="target_id must be day|channel|index")
        day = parts[0]
        channel = parts[1]
        candidate_kind = str(kind or "").strip().lower()
        if candidate_kind not in _SEGMENT_KINDS:
            raise HTTPException(status_code=422, detail="kind must be pin, force, forbid or gold")
        candidate = Override(
            override_id="candidate-preview",
            scope=SEGMENT,
            target_id=str(target_id),
            kind=candidate_kind,
            value=str(value or ""),
            gold=bool(gold) or candidate_kind == "gold",
        )

    try:
        segments, engine_kwargs = preview_inputs(channel, day, daily_input)
    except Exception as exc:  # pragma: no cover - data/environment dependent
        raise HTTPException(status_code=503, detail=f"Could not build segments for preview: {exc}")
    if not segments:
        raise HTTPException(status_code=404, detail="No segments found for the requested channel-day")

    # Anchor guard (re-ingest safety): resolve the stored overrides against the
    # anchors of the segments we just built, exactly as the commit does.
    active_overrides, stale_overrides = _resolved_store_overrides(segments)
    # The stored placement constraints ride along on BOTH legs (the commit path
    # applies them beside the overrides), so toggling the overrides is the ONLY
    # difference between the two plans. The commit passes None when the override
    # store is empty; mirror that so the runs are argument-identical.
    constraints = _stored_constraints()
    stored = active_overrides if active_overrides.overrides else None
    if candidate is not None:
        # Candidate mode: the baseline is the CURRENT plan (stored overrides
        # applied), and the comparison adds only the candidate, so the delta is
        # exactly what this one decision would change.
        baseline = _optimize_one_day(
            segments, constraints=constraints, overrides=stored, **engine_kwargs,
        )
        with_candidate = OverrideSet(overrides=list(active_overrides.overrides) + [candidate])
        overridden = _optimize_one_day(
            segments, constraints=constraints, overrides=with_candidate, **engine_kwargs,
        )
    else:
        baseline = _optimize_one_day(segments, constraints=constraints, **engine_kwargs)
        overridden = _optimize_one_day(
            segments, constraints=constraints, overrides=stored, **engine_kwargs,
        )

    base_counts = {s.segment_id: s.num_breaks for s in baseline.segments}
    new_counts = {s.segment_id: s.num_breaks for s in overridden.segments}
    changed = [
        {
            "segment_id": segment_id,
            "before": base_counts.get(segment_id, 0),
            "after": new_counts.get(segment_id, 0),
        }
        for segment_id in sorted(new_counts)
        if base_counts.get(segment_id, 0) != new_counts.get(segment_id, 0)
    ]
    return {
        "channel": channel,
        "day": day,
        "candidate": (
            {"target_id": candidate.target_id, "kind": candidate.kind, "value": candidate.value}
            if candidate is not None
            else None
        ),
        "summary": {
            "before_total_breaks": baseline.total_breaks,
            "after_total_breaks": overridden.total_breaks,
            "before_revenue": round(baseline.total_revenue, 2),
            "after_revenue": round(overridden.total_revenue, 2),
            "changed_segments": len(changed),
        },
        "changed": changed,
        "rejected_overrides": [
            {"segment_id": r.segment_id, "kind": r.kind, "requested": r.requested, "reason": r.reason}
            for r in overridden.rejected_overrides
        ] + [
            {
                "segment_id": s["segment_id"],
                "kind": s["kind"],
                "requested": None,
                "reason": s["reason"],
                "anchor_stale": True,
                "override_id": s["override_id"],
                "expected": s["expected"],
                "found": s["found"],
            }
            for s in stale_overrides
        ],
    }
