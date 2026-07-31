"""Plan, day and break: the board, the break list, and the segment inspector.

The day reads, moved verbatim from dashboard_api.py (the board, the override
targets, the inspector) and catalog_api.py (the break list) as part of the
wave-zero router split. Behaviour is unchanged, including the competitor
boundary: only the operator's own channel produces override targets, and a
segment on any other channel is a 404, never a competitor's plan.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException

from kairos_api import plan_read
from kairos_api.core import (
    DATA_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    KairosSettings,
    _augment_segment_ids,
    _load_break_schedule,
    _load_programmes,
    _load_settings,
    _money,
    _percent,
    _records,
    _row_anchor,
    _safe_number,
    _signature,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_schedule_segments(schedule: pd.DataFrame, settings: KairosSettings) -> dict[str, Any]:
    """The operator's list of valid override targets, OWNED CHANNEL ONLY.

    The weekly schedule loops every channel-day, but the operator may only constrain
    their own channel, so this filters to settings.operator_channel and never emits a
    competitor row. Honest empty (empty list) when no schedule exists or no owned
    channel is configured yet.
    """
    owned = str(settings.operator_channel or "").strip()
    if schedule.empty or not owned:
        return {
            "operator_channel": owned or None,
            "operator_channel_unset": not owned,
            "segments": [],
        }
    frame = _augment_segment_ids(schedule)
    frame = frame[frame["channel"].astype(str).str.strip() == owned]
    segments = [
        {
            "segment_id": str(row.get("segment_id", "")).strip(),
            "channel": owned,
            "day": str(row.get("date", "")).strip(),
            "anchor": _row_anchor(row),
            "state": {
                "num_breaks": int(_safe_number(row.get("num_breaks", 0))),
                "is_gold": bool(row.get("is_gold", False)),
                "predicted_revenue": _money(row.get("predicted_revenue", 0)),
                "retention": round(_percent(row.get("predicted_retention", 0)), 1),
            },
        }
        for _, row in frame.iterrows()
    ]
    return {"operator_channel": owned, "operator_channel_unset": False, "segments": segments}


@lru_cache(maxsize=16)
def _schedule_segments_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_schedule_segments(_load_break_schedule(), _load_settings())


@lru_cache(maxsize=16)
def _break_operations_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return plan_read.build_break_operations(_load_programmes(), _load_break_schedule())


def _segment_overrides(segment_id: str) -> list[dict[str, Any]]:
    """The saved manual overrides targeting one segment, newest first.

    Read straight from data/manual_overrides.csv so the inspector shows the real
    edit state of a segment (what is pinned/forbidden/gold and where it came from,
    manual or a recommendation). Honest empty list when the store is absent or the
    segment carries no override.
    """
    path = DATA_DIR / "manual_overrides.csv"
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception:
        logger.exception("override store read failed for segment detail")
        return []
    if "target_id" not in frame.columns:
        return []
    rows = frame[frame["target_id"].astype(str).str.strip() == segment_id.strip()]
    records = [{str(k): (None if v == "" else v) for k, v in row.items()} for _, row in rows.iterrows()]
    records.reverse()
    return records


def _build_break_library(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {"breaks": []}

    frame = schedule.copy()
    settings = _load_settings()
    # The weekly schedule carries breaks for every channel because the retention
    # model needs competitor rows, but this is an operator-facing candidate list.
    # Scope to the operator's own channel so no competitor break is ever presented
    # as a candidate. An unconfigured (empty) channel is an honest no-op that keeps
    # all rows, matching the resolver convention. The channel is read from settings,
    # never hardcoded.
    operator_channel = str(settings.operator_channel or "").strip()
    if operator_channel and "channel" in frame.columns:
        frame = frame[frame["channel"].astype(str) == operator_channel]
    if frame.empty:
        return {"breaks": []}
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0), errors="coerce").fillna(0)
    frame["total_break_time"] = pd.to_numeric(frame.get("total_break_time", 0), errors="coerce").fillna(0)
    frame["priority"] = frame["predicted_revenue"] * frame["predicted_retention"].clip(lower=0.1)
    frame = frame.sort_values("priority", ascending=False).head(80)
    floor_percent = settings.min_retention_floor * 100
    frame["status"] = frame["predicted_retention"].map(lambda value: "at_risk" if _percent(value) < floor_percent else "ready")
    return {"breaks": _records(frame)}


@lru_cache(maxsize=16)
def _break_library_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_break_library(_load_break_schedule())


@router.get("/api/schedule/segments", tags=["dashboard"])
def schedule_segments() -> dict[str, Any]:
    """The operator's valid override targets on the OWNED channel only.

    The weekly schedule loops every channel-day, but this endpoint enforces the
    competitor boundary and returns segments for settings.operator_channel alone, each
    with its build-order segment_id, its semantic anchor (date, start_clock, program),
    and its current state (num_breaks, is_gold, predicted_revenue, retention). Honest
    empty (200 + empty list) when no schedule exists or no owned channel is set.
    """
    return _schedule_segments_cached(_signature([
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        SETTINGS_PATH,
    ]))


@router.get("/api/schedule/segment/{segment_id:path}", tags=["dashboard"])
def schedule_segment_detail(segment_id: str) -> dict[str, Any]:
    """Full inspector detail for one owned-channel segment.

    Composes the complete saved-plan row (identity, timing, plan, economics, and
    the risk-adjusted retention with its credible interval) with the segment's
    current manual overrides, so a click-to-open inspector can show everything the
    engine knows about a programme and its breaks and what the operator has already
    decided. Enforces the competitor boundary: a segment on any channel other than
    settings.operator_channel returns 404, never a competitor's plan. Honest 404
    when no saved schedule exists or the id is unknown.
    """
    settings = _load_settings()
    owned = str(settings.operator_channel or "").strip()
    schedule = _load_break_schedule()
    if schedule.empty:
        raise HTTPException(status_code=404, detail="No saved weekly schedule on disk")
    frame = _augment_segment_ids(schedule)
    match = frame[frame["segment_id"].astype(str).str.strip() == segment_id.strip()]
    if match.empty:
        raise HTTPException(status_code=404, detail="Unknown segment id")
    row = match.iloc[0]
    channel = str(row.get("channel", "")).strip()
    if owned and channel != owned:
        # Competitor boundary: the operator only inspects and edits their own channel.
        raise HTTPException(status_code=404, detail="Segment is not on the owned channel")

    def _opt_num(value: Any) -> float | None:
        parsed = _safe_number(value, float("nan"))
        return None if math.isnan(parsed) else parsed

    def _opt_int(value: Any) -> int | None:
        num = _opt_num(value)
        return None if num is None else int(round(num))

    detail = {
        "segment_id": segment_id,
        "found": True,
        "owned_channel": owned or None,
        "anchor": _row_anchor(row),
        "identity": {
            "channel": channel,
            "date": str(row.get("date", "")).strip(),
            "day": str(row.get("day", "")).strip(),
            "program_type": str(row.get("program_type", "")).strip(),
            "start_clock": str(row.get("start_time", "")).strip(),
        },
        "plan": {
            "num_breaks": int(_safe_number(row.get("num_breaks", 0))),
            "break_length_seconds": _opt_num(row.get("break_length")),
            "total_break_seconds": _opt_num(row.get("total_break_time")),
            # Blank for a 0-break segment (the CSV honestly omits a position when
            # there are no breaks); the isna guard keeps NaN from rendering "nan".
            "position": None if pd.isna(row.get("position")) else (str(row.get("position", "")).strip() or None),
            "break_type": str(row.get("break_type", "")).strip() or None,
            "is_gold": bool(str(row.get("is_gold", "")).strip().lower() in ("true", "1", "yes")),
        },
        "economics": {
            "predicted_revenue": _money(row.get("predicted_revenue", 0)),
            "base_rate": _opt_num(row.get("base_rate")),
            "baseline_tvr": _opt_num(row.get("baseline_tvr")),
        },
        "retention": {
            "predicted_retention": round(_percent(row.get("predicted_retention", 0)), 2),
            "retention_used": round(_percent(row.get("retention_used", row.get("predicted_retention", 0))), 2),
            "ci_low": (round(_percent(row.get("retention_ci_low")), 2) if _opt_num(row.get("retention_ci_low")) is not None else None),
            "ci_high": (round(_percent(row.get("retention_ci_high")), 2) if _opt_num(row.get("retention_ci_high")) is not None else None),
            "sample_n": _opt_int(row.get("retention_n")),
            "confidence": str(row.get("retention_confidence", "")).strip() or None,
        },
        "overrides": _segment_overrides(segment_id),
    }
    return detail


@router.get("/api/break-operations", tags=["dashboard"])
def break_operations() -> dict[str, Any]:
    # Same key discipline as /api/schedule: the board reads the operator
    # settings and the rate card, so both belong in the cache signature.
    return _break_operations_cached(_signature([
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        SETTINGS_PATH,
        ROOT / "config" / "optimization_weights.yaml",
    ]))


@router.get("/api/break-library", tags=["catalog"])
def break_library() -> dict[str, Any]:
    return _break_library_cached(_signature([OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv", SETTINGS_PATH]))
