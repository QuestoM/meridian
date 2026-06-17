"""Unified placement-constraint CRUD plus an honest WITH-vs-WITHOUT preview.

This is the operator-facing seam for the scoped placement-constraint store
(:mod:`kairos.optimize.constraints_store`). It persists constraints to
``data/kairos_constraints.csv`` with the same read-mutate-backup-write style as
:mod:`kairos_api.advertiser_conditions`, serves the option lists the dashboard
needs to build a scoped rule (real programme Titles, channels, weekdays, effects,
scope types), and serves a preview that runs the weekly schedule with and without
the constraints so the operator sees exactly which segments change and which
constraints were skipped.

Honesty rules: scope and effect are validated against the engine vocabularies
before a row is stored; the effect preview reports the resolver's skipped
constraints verbatim (never hiding one that could not be honored); and a preview
that cannot build real segments says so rather than inventing a delta.
"""

from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from kairos.optimize.constraints_store import (
    COLUMNS,
    DEFAULT_CONSTRAINTS_PATH,
    _SCOPES,
    _EFFECTS,
    load_constraints,
    resolve_constraints,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
CONSTRAINTS_PATH = DATA_DIR / "kairos_constraints.csv"

router = APIRouter(prefix="/api/constraints", tags=["constraints"])

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
    """

    model_config = ConfigDict(populate_by_name=True)

    scope_type: str
    effect: str
    scope_value: str = ""
    channel: str = ""
    offset_seconds: Optional[float] = None
    offset_min_seconds: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("offset_min_seconds", "offset_seconds_min"))
    offset_max_seconds: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("offset_max_seconds", "offset_seconds_max"))
    count: Optional[int] = Field(
        default=None, validation_alias=AliasChoices("count", "pin_count"))
    duration_seconds: Optional[float] = None
    duration_min_seconds: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("duration_min_seconds", "duration_seconds_min"))
    duration_max_seconds: Optional[float] = Field(
        default=None, validation_alias=AliasChoices("duration_max_seconds", "duration_seconds_max"))
    order_index: Optional[int] = None
    notes: str = ""


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


def _load_frame() -> pd.DataFrame:
    if not CONSTRAINTS_PATH.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    frame = pd.read_csv(CONSTRAINTS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
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
    _backup()
    CONSTRAINTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame[list(COLUMNS)].to_csv(CONSTRAINTS_PATH, index=False, encoding="utf-8-sig")


def _record(row: "pd.Series[Any]") -> dict[str, Any]:
    return {column: str(row.get(column, "")) for column in COLUMNS}


def _validate_scope(scope_type: str) -> str:
    cleaned = str(scope_type or "").strip().lower()
    if cleaned not in _SCOPES:
        raise HTTPException(status_code=400, detail=f"scope_type must be one of {sorted(_SCOPES)}")
    return cleaned


def _validate_effect(effect: str) -> str:
    cleaned = str(effect or "").strip().lower()
    if cleaned not in _EFFECTS:
        raise HTTPException(status_code=400, detail=f"effect must be one of {sorted(_EFFECTS)}")
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
    }


@router.post("", status_code=201)
def create_constraint(payload: ConstraintCreate) -> dict[str, Any]:
    scope_type = _validate_scope(payload.scope_type)
    effect = _validate_effect(payload.effect)
    frame = _load_frame()
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
    }
    frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
    _write_frame(frame)
    return _record(frame.iloc[-1])


def _locate(frame: pd.DataFrame, constraint_id: str) -> int:
    mask = frame["constraint_id"].astype(str) == constraint_id
    if not mask.any():
        raise HTTPException(status_code=404, detail=f"constraint '{constraint_id}' not found")
    return int(frame.index[mask][0])


@router.put("/{constraint_id}")
def update_constraint(constraint_id: str, payload: ConstraintUpdate) -> dict[str, Any]:
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
    _write_frame(frame)
    return _record(frame.loc[index])


@router.delete("/{constraint_id}")
def delete_constraint(constraint_id: str) -> dict[str, Any]:
    frame = _load_frame()
    index = _locate(frame, constraint_id)
    frame = frame.drop(index=index).reset_index(drop=True)
    _write_frame(frame)
    return {"deleted": constraint_id}


def _channel_options() -> list[str]:
    """Real channel names from the reference EPG, sorted and de-duplicated."""
    try:
        from kairos.data.loaders import load_programmes

        frame = load_programmes()
        if "Channel" not in frame.columns:
            return []
        names = {str(c).strip() for c in frame["Channel"].dropna() if str(c).strip()}
        return sorted(names)
    except Exception:
        return []


def _weekday_options() -> list[dict[str, Any]]:
    """ISO weekday tokens 1..7 with bilingual labels (Mon..Sun)."""
    names_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    names_he = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
    return [
        {"key": str(i + 1), "en": names_en[i], "he": names_he[i]}
        for i in range(7)
    ]


@router.get("/options")
def scope_options() -> dict[str, Any]:
    """Option lists the dashboard needs to build a scoped placement constraint.

    Programmes and channels come from the real reference EPG (the same source the
    advertiser-conditions page reuses), weekdays are the ISO 1..7 tokens, and the
    scope_types / effects come straight from the constraint store vocabulary, so
    the dashboard never offers a value the resolver cannot consume.
    """
    from kairos_api.advertiser_conditions import _programme_options

    return {
        "scope_types": sorted(_SCOPES),
        "effects": sorted(_EFFECTS),
        "programmes": _programme_options(),
        "channels": _channel_options(),
        "weekdays": _weekday_options(),
    }


def _build_segments(channel: Optional[str], day: Optional[str], daily_input: Optional[str]):
    """Build real ProgramSegments for the preview, or raise if data is absent.

    Mirrors :func:`kairos_api.overrides._build_segments` so the preview runs over
    the same segments the live optimizer sees.
    """
    from kairos.data import ProgramClassifier
    from kairos.data.loaders import load_daily_input, load_programmes
    from kairos.data.transform import (
        build_segments_from_daily_input,
        build_segments_from_programmes,
    )
    from kairos.model.impact import load_impact_model
    from kairos.optimize.pricing import OptimizerAssumptions, PricingModel

    pricing = PricingModel.from_yaml()
    assumptions = OptimizerAssumptions()
    impact = load_impact_model(ROOT / "models" / "tv_break_posterior.pkl", assumptions=assumptions)
    classifier = ProgramClassifier.from_yaml()
    if daily_input:
        daily = load_daily_input(daily_input)
        return build_segments_from_daily_input(
            daily, classifier, pricing, assumptions=assumptions, impact_model=impact,
        )
    programmes = load_programmes()
    return build_segments_from_programmes(
        programmes, classifier, pricing,
        assumptions=assumptions, impact_model=impact, channel=channel, day=day,
    )


@router.get("/effect")
def constraint_effect(
    channel: str | None = None,
    day: str | None = None,
    daily_input: str | None = None,
) -> dict[str, Any]:
    """Preview the weekly schedule WITH vs WITHOUT the stored constraints.

    Builds real segments for the requested channel-day, runs the break optimizer
    twice (plain, then with the resolved constraints), and reports per-segment
    break-count deltas plus any constraints the resolver skipped (with the reason).
    This is honest about where constraints bite: a position pin forces a segment's
    count, a forbid zeroes it, and a count pin sets it, so the deltas are exactly
    what the weekly recompute would write.
    """
    from kairos.optimize.constraints_store import count_pins_to_overrides
    from kairos.optimize.optimizer import optimize_breaks

    try:
        segments = _build_segments(channel, day, daily_input)
    except Exception as exc:  # pragma: no cover - data/environment dependent
        raise HTTPException(status_code=503, detail=f"Could not build segments for preview: {exc}")
    if not segments:
        raise HTTPException(status_code=404, detail="No segments found for the requested channel-day")

    constraints = load_constraints(CONSTRAINTS_PATH)
    placement_pins, count_pins, forbids, skipped = resolve_constraints(segments, constraints)
    overrides = count_pins_to_overrides(count_pins, forbids)

    baseline = optimize_breaks(segments)
    constrained = optimize_breaks(
        segments, overrides=overrides, placement_pins=placement_pins or None,
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
