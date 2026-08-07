"""Unified placement-constraint CRUD plus an honest WITH-vs-WITHOUT preview.

This is the operator-facing seam for the scoped placement-constraint store
(:mod:`kairos.optimize.constraints_store`). It persists constraints to
``data/kairos_constraints.csv`` with the same read-mutate-backup-write style as
:mod:`kairos_api.advertiser_conditions` (serialized under a module lock, written
via a temp file plus ``os.replace`` so readers never see a torn CSV), serves the
option lists the dashboard needs to build a scoped rule (the operator channel's
own programme titles, weekdays, effects, scope types), and a preview that runs the
weekly schedule with and without the constraints so the operator sees exactly
which segments change and which constraints were skipped.

Honesty rules: scope and effect are validated against the engine vocabularies
before a row is stored; the effect preview reports the resolver's skipped
constraints verbatim (never hiding one that could not be honored); and a preview
that cannot build real segments says so rather than inventing a delta. The
preview runs through the SAME engine seams the weekly recompute uses (saved
settings, first-break fold, wrapped classifier, stored overrides, demand fold),
so its numbers are the plan the commit would write, not a parallel engine.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from kairos.optimize.constraints_store import (
    COLUMNS,
    _SCOPES,
    _EFFECTS,
    load_constraints,
    resolve_constraints,
)
from kairos_api import constraints_sentence
from kairos_api.constraints_language import AUTHORING_COLUMNS
from kairos_api._constraint_options import (
    daypart_options_list as _daypart_options_list,
    load_operator_channel as _load_operator_channel,
    operator_scope_options as _operator_scope_options,
    predicate_field_schema as _predicate_field_schema,
    validate_where as _validate_where,
    weekday_options as _weekday_options,
    where_json_cell as _where_json_cell,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"

# Relocatable by environment, the same way the guardrail store is: a restriction
# changes what the optimizer does, so an instance that writes one for a
# measurement must be able to write it somewhere other than the shared file.
# Resolved at import, so importers and tests that patch the attribute are
# unaffected.
CONSTRAINTS_PATH_ENV = "KAIROS_CONSTRAINTS_PATH"
_STORE_OVERRIDE = os.getenv(CONSTRAINTS_PATH_ENV, "").strip()
CONSTRAINTS_PATH = (
    (Path(_STORE_OVERRIDE) if Path(_STORE_OVERRIDE).is_absolute() else ROOT / _STORE_OVERRIDE)
    if _STORE_OVERRIDE else DATA_DIR / "kairos_constraints.csv"
)

router = APIRouter(prefix="/api/constraints", tags=["constraints"])

# Serializes every load-mutate-write cycle on the constraints CSV so two
# concurrent edits cannot drop each other's rows (lost update).
_STORE_LOCK = threading.Lock()

# Numeric columns coerced to float-or-blank on write; the rest are plain strings.
_FLOAT_COLUMNS = (
    "offset_seconds",
    "offset_min_seconds",
    "offset_max_seconds",
    "duration_seconds",
    "duration_min_seconds",
    "duration_max_seconds",
)
_INT_COLUMNS = ("count", "order_index")


class ConstraintCreate(BaseModel):
    """A new scoped placement constraint. scope_type and effect must be valid.

    Field aliases keep the API tolerant of the dashboard's naming (for example
    ``offset_seconds_min`` and ``pin_count``) so a client and the store cannot drift
    apart silently; the canonical names still populate the same fields.

    ``where`` is the optional rich predicate tree (Group/Condition). When
    supplied it overrides the flat scope_type/scope_value matching; the flat
    fields are still persisted for back-compat. When absent the legacy flat
    matching is used. See docs/constraint-predicate-contract.md for the exact
    frozen JSON shape.
    """

    model_config = ConfigDict(populate_by_name=True)

    scope_type: str
    effect: str
    scope_value: str = ""
    channel: str = ""
    # Alias metadata rides in Annotated form: attaching Field(validation_alias=...)
    # to an Optional default is dropped when FastAPI re-generates the body schema
    # (pydantic UnsupportedFieldAttributeWarning); Annotated survives every context.
    offset_seconds: Optional[float] = None
    offset_min_seconds: Annotated[
        Optional[float], Field(validation_alias=AliasChoices("offset_min_seconds", "offset_seconds_min"))] = None
    offset_max_seconds: Annotated[
        Optional[float], Field(validation_alias=AliasChoices("offset_max_seconds", "offset_seconds_max"))] = None
    count: Annotated[
        Optional[int], Field(validation_alias=AliasChoices("count", "pin_count"))] = None
    duration_seconds: Optional[float] = None
    duration_min_seconds: Annotated[
        Optional[float], Field(validation_alias=AliasChoices("duration_min_seconds", "duration_seconds_min"))] = None
    duration_max_seconds: Annotated[
        Optional[float], Field(validation_alias=AliasChoices("duration_max_seconds", "duration_seconds_max"))] = None
    order_index: Optional[int] = None
    notes: str = ""
    where: Optional[dict[str, Any]] = None


class ConstraintUpdate(BaseModel):
    """Editable fields for a constraint. All optional for PATCH-style PUT."""

    scope_type: str | None = None
    effect: str | None = None
    scope_value: str | None = None
    channel: str | None = None
    offset_seconds: float | None = None
    offset_min_seconds: float | None = None
    offset_max_seconds: float | None = None
    count: int | None = None
    duration_seconds: float | None = None
    duration_min_seconds: float | None = None
    duration_max_seconds: float | None = None
    order_index: int | None = None
    notes: str | None = None
    where: Optional[dict[str, Any]] = None


# The frozen engine columns first, then the authoring columns a restriction
# adds. The engine loader reads by name, so a column it does not know is a
# column it ignores: attribution and expiry ride beside the compiled row
# without changing one thing the optimizer sees.
STORE_COLUMNS = tuple(COLUMNS) + AUTHORING_COLUMNS


def _load_frame() -> pd.DataFrame:
    if not CONSTRAINTS_PATH.exists():
        return pd.DataFrame(columns=list(STORE_COLUMNS))
    frame = pd.read_csv(CONSTRAINTS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in STORE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _backup() -> None:
    if not CONSTRAINTS_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(CONSTRAINTS_PATH, BACKUP_DIR / f"kairos_constraints_{stamp}.csv")


def _write_frame(frame: pd.DataFrame) -> None:
    """Backup, then write atomically (temp file + os.replace, like auth_store).

    A reader that opens the CSV mid-write sees either the old or the new file,
    never a truncated one. Callers hold ``_STORE_LOCK`` across load-mutate-write.
    """
    _backup()
    CONSTRAINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CONSTRAINTS_PATH.with_name(CONSTRAINTS_PATH.name + ".tmp")
    for column in STORE_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame[list(STORE_COLUMNS)].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, CONSTRAINTS_PATH)


def _snapshot_before_write(request: "Request | None") -> None:
    """Record a version of the constraints store before a manual edit writes it."""
    from kairos_api import version_store

    version_store.snapshot_manual_edit(request, "constraints")


def _record(row: "pd.Series[Any]") -> dict[str, Any]:
    result = {column: str(row.get(column, "")) for column in STORE_COLUMNS}
    # Also expose the parsed where predicate (convenience for API consumers).
    raw_json = str(row.get("where_json", "") or "")
    if raw_json.strip():
        try:
            result["where"] = json.loads(raw_json)
        except (json.JSONDecodeError, ValueError):
            result["where"] = None
    else:
        result["where"] = None
    return result


def _validate_scope(scope_type: str) -> str:
    cleaned = str(scope_type or "").strip().lower()
    if cleaned not in _SCOPES:
        raise constraints_sentence.refuse("bad_scope_type")
    return cleaned


def _validate_effect(effect: str) -> str:
    cleaned = str(effect or "").strip().lower()
    if cleaned not in _EFFECTS:
        raise constraints_sentence.refuse("bad_effect")
    return cleaned


def _num_cell(value: object) -> str:
    """Render a numeric payload value as a CSV cell (blank when None)."""
    return "" if value is None else str(value)


@router.get("")
def list_constraints() -> dict[str, Any]:
    """All stored constraints plus the raw column order."""
    frame = _load_frame()
    return {
        "constraints": [_record(row) for _, row in frame.iterrows()],
        "columns": list(COLUMNS),
        "authoring_columns": list(AUTHORING_COLUMNS),
    }


@router.post("", status_code=201)
def create_constraint(payload: ConstraintCreate, request: Request = None) -> dict[str, Any]:
    scope_type = _validate_scope(payload.scope_type)
    effect = _validate_effect(payload.effect)
    where = _validate_where(payload.where)
    new_row = {
        "constraint_id": uuid.uuid4().hex[:12],
        "scope_type": scope_type,
        "scope_value": str(payload.scope_value or "").strip(),
        "channel": str(payload.channel or "").strip(),
        "effect": effect,
        "offset_seconds": _num_cell(payload.offset_seconds),
        "offset_min_seconds": _num_cell(payload.offset_min_seconds),
        "offset_max_seconds": _num_cell(payload.offset_max_seconds),
        "count": _num_cell(payload.count),
        "duration_seconds": _num_cell(payload.duration_seconds),
        "duration_min_seconds": _num_cell(payload.duration_min_seconds),
        "duration_max_seconds": _num_cell(payload.duration_max_seconds),
        "order_index": _num_cell(payload.order_index),
        "notes": str(payload.notes or ""),
        "where_json": _where_json_cell(where),
    }
    with _STORE_LOCK:
        frame = _load_frame()
        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        _snapshot_before_write(request)
        _write_frame(frame)
        return _record(frame.iloc[-1])


def _locate(frame: pd.DataFrame, constraint_id: str) -> int:
    mask = frame["constraint_id"].astype(str) == constraint_id
    if not mask.any():
        raise constraints_sentence.refuse("constraint_gone", 404)
    return int(frame.index[mask][0])


@router.put("/{constraint_id}")
def update_constraint(constraint_id: str, payload: ConstraintUpdate,
                      request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_frame()
        index = _locate(frame, constraint_id)
        if payload.scope_type is not None:
            frame.at[index, "scope_type"] = _validate_scope(payload.scope_type)
        if payload.effect is not None:
            frame.at[index, "effect"] = _validate_effect(payload.effect)
        if payload.scope_value is not None:
            frame.at[index, "scope_value"] = str(payload.scope_value).strip()
        if payload.channel is not None:
            frame.at[index, "channel"] = str(payload.channel).strip()
        for column in _FLOAT_COLUMNS + _INT_COLUMNS:
            value = getattr(payload, column)
            if value is not None:
                frame.at[index, column] = str(value)
        if payload.notes is not None:
            frame.at[index, "notes"] = str(payload.notes)
        if payload.where is not None:
            where = _validate_where(payload.where)
            frame.at[index, "where_json"] = _where_json_cell(where)
        _snapshot_before_write(request)
        _write_frame(frame)
        return _record(frame.loc[index])


@router.delete("/{constraint_id}")
def delete_constraint(constraint_id: str, request: Request = None) -> dict[str, Any]:
    with _STORE_LOCK:
        frame = _load_frame()
        index = _locate(frame, constraint_id)
        frame = frame.drop(index=index).reset_index(drop=True)
        _snapshot_before_write(request)
        _write_frame(frame)
    return {"deleted": constraint_id}


@router.get("/options")
def scope_options() -> dict[str, Any]:
    """Option lists the dashboard needs to build a scoped placement constraint.

    Includes the frozen predicate field/operator schema, the programmes and
    genres of the operator's own channel, the active daypart and weekday
    vocabularies, and the operator_channel whose breaks the constraints scope to.

    The programme, genre and channel lists are scoped through
    :mod:`kairos_api.channel_scope`, because this payload feeds the condition
    builder's value picker and that is an operator surface. Measured on the
    reference EPG before the scope was applied: 418 titles, of which 106 are the
    operator's and 312 are three rivals' whole lineups, and all four channel
    names. ``scope`` is the disclosure that travels with the lists, and when
    nothing is declared yet it says why they are empty and how to fill them.
    """
    scoped = _operator_scope_options()
    return {
        "scope_types": sorted(_SCOPES),
        "effects": sorted(_EFFECTS),
        "programmes": scoped["programmes"],
        "genres": scoped["genres"],
        "channels": scoped["channels"],
        "weekdays": _weekday_options(),
        "dayparts": _daypart_options_list(),
        "predicate_fields": _predicate_field_schema(),
        "operator_channel": _load_operator_channel(),
        "available_channels": scoped["channels"],
        "scope": scoped["scope"],
    }


def _build_segments(channel: Optional[str], day: Optional[str],
                    daily_input: Optional[str]) -> list:
    """Back-compat seam: the preview's segments alone.

    The segments and the engine kwargs are built together by
    :func:`kairos_api.preview_inputs.preview_inputs` (the commit path's seams);
    this keeps the historical entry point for callers that only need the
    segments.
    """
    from kairos_api.preview_inputs import preview_inputs

    return preview_inputs(channel, day, daily_input)[0]


@router.get("/effect")
def constraint_effect(
    channel: str | None = None,
    day: str | None = None,
    daily_input: str | None = None,
) -> dict[str, Any]:
    """Preview the weekly schedule WITH vs WITHOUT the stored constraints.

    Builds real segments for the requested channel-day and runs the commit path's
    own core (:func:`kairos.optimize.day_core._optimize_one_day`) twice: once
    without the constraints and once with them, reporting per-segment break-count
    deltas plus any constraints the resolver skipped (with the reason). The stored
    manual overrides and the demand fold apply on BOTH legs, exactly as the weekly
    recompute applies them, so the delta isolates the constraints and the absolute
    numbers are the plan the commit would write. This is honest about where
    constraints bite: a position pin forces a segment's count, a forbid zeroes it,
    and a count pin sets it.
    """
    from kairos.optimize.day_core import _optimize_one_day
    from kairos_api.overrides import _resolved_store_overrides
    from kairos_api.preview_inputs import preview_inputs

    try:
        segments, engine_kwargs = preview_inputs(channel, day, daily_input)
    except Exception as exc:  # pragma: no cover - data/environment dependent
        raise constraints_sentence.refuse("segments_failed", 503)
    if not segments:
        raise constraints_sentence.refuse("no_segments", 404)

    # Resolve once for the honest report (matched / skipped); the WITH leg passes
    # the raw constraint list into the shared core, which resolves them again
    # through the same single resolver, so the report and the plan cannot drift.
    constraints = load_constraints(CONSTRAINTS_PATH)
    placement_pins, count_pins, forbids, skipped = resolve_constraints(
        segments, constraints, operator_channel=engine_kwargs["operator_channel"],
    )
    # The stored manual overrides ride along on BOTH legs (the commit path applies
    # them beside the constraints), resolved through the same anchor guard. None
    # when the store is empty, mirroring the commit's argument exactly.
    active_overrides, _stale = _resolved_store_overrides(segments)
    stored = active_overrides if active_overrides.overrides else None

    baseline = _optimize_one_day(segments, overrides=stored, **engine_kwargs)
    constrained = _optimize_one_day(
        segments, constraints=constraints, overrides=stored, **engine_kwargs,
    )

    base_counts = {s.segment_id: s.num_breaks for s in baseline.segments}
    new_counts = {s.segment_id: s.num_breaks for s in constrained.segments}
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
        "summary": {
            "before_total_breaks": baseline.total_breaks,
            "after_total_breaks": constrained.total_breaks,
            "before_revenue": round(baseline.total_revenue, 2),
            "after_revenue": round(constrained.total_revenue, 2),
            "changed_segments": len(changed),
            "matched_segments": len(set(placement_pins) | set(count_pins) | forbids),
        },
        "changed": changed,
        "skipped_constraints": [
            {"constraint_id": s.constraint_id, "segment_id": s.segment_id, "reason": s.reason}
            for s in skipped
        ],
        "rejected_overrides": [
            {"segment_id": r.segment_id, "kind": r.kind, "requested": r.requested, "reason": r.reason}
            for r in constrained.rejected_overrides
        ],
    }


# The restriction routes ride on this router: they read and write the same store
# through the same lock, so they belong to the same module boundary even though
# the file-size law puts their code next door. Imported at the foot of the module
# so the lazy imports back into this one resolve against a finished module.
from kairos_api.constraints_restrictions import router as _restrictions_router  # noqa: E402

router.include_router(_restrictions_router)
