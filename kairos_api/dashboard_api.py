"""Dashboard read endpoints: the operator's overview, schedule canvas, break
board, segment inspector, compliance verdict, and decision shortcut.

Thin domain router over the shared kernel (:mod:`kairos_api.core`). The request
model, builders, the revenue-vs-retention frontier machinery, and the cached
wrappers moved verbatim from server.py as part of the modular-monolith carve-up;
behavior is unchanged and server.py re-exports the moved names so existing
references (tests, the assistant and catalog routers, the startup warm-up) keep
resolving against the SAME objects, including the single lru_cache instances and
the one frontier background-thread state and its lock.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from kairos.optimize.guardrails import Break as GuardrailBreak
from kairos.optimize.guardrails import evaluate as evaluate_guardrails
from kairos.optimize.objective import break_revenue as cpp_break_revenue
from kairos.optimize.objective import retention_adjusted_revenue

from kairos_api.core import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    KairosSettings,
    PricingModel,
    _ENGINE_AVAILABLE,
    _augment_segment_ids,
    _load_break_schedule,
    _load_programmes,
    _load_settings,
    _load_spots,
    _model_dump,
    _money,
    _pacing_call_kwargs,
    _percent,
    _plan_segment_index,
    _ratio,
    _row_anchor,
    _safe_number,
    _settings_to_guardrails,
    _signature,
    _summarize_schedule,
    _time_to_seconds,
    run_scenario,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["dashboard"])


class BreakDecisionRequest(BaseModel):
    """Operator decision captured from the dashboard command surface."""

    action: Literal["approve", "reject", "apply_similar"]
    recommendation_id: str | None = Field(default=None)
    break_id: str | None = Field(default=None)
    program_type: str | None = Field(default=None)
    scenario: str | None = Field(default=None)
    note: str | None = Field(default=None, max_length=500)
    # Fields that let an approve/reject resolve into a REAL override. target_id is the
    # owned-channel segment_id (falls back to break_id); kind is the override kind
    # (pin/force/forbid/gold); the anchor trio is copied from the recommendation so a
    # later re-ingest cannot silently rebind the override to a different break.
    target_id: str | None = Field(default=None)
    kind: str | None = Field(default=None)
    value: str | None = Field(default=None)
    gold: bool = Field(default=False)
    anchor_date: str | None = Field(default=None)
    anchor_start: str | None = Field(default=None)
    anchor_title: str | None = Field(default=None)


def _day_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    aliases = {
        "monday": "Mon",
        "mon": "Mon",
        "tuesday": "Tue",
        "tue": "Tue",
        "wednesday": "Wed",
        "wed": "Wed",
        "thursday": "Thu",
        "thu": "Thu",
        "friday": "Fri",
        "fri": "Fri",
        "saturday": "Sat",
        "sat": "Sat",
        "sunday": "Sun",
        "sun": "Sun",
    }
    return aliases.get(text.lower(), text[:3].title())


def _program_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize programme CSV date/time variants into start_dt and end_dt."""

    result = frame.copy()
    if "Start_datetime" in result.columns:
        starts = pd.to_datetime(result["Start_datetime"], errors="coerce")
    elif {"Date", "Start time"}.issubset(result.columns):
        starts = pd.to_datetime(
            result["Date"].astype(str) + " " + result["Start time"].astype(str),
            errors="coerce",
            dayfirst=True,
        )
    else:
        starts = pd.to_datetime(result.get("Start time"), errors="coerce")

    if "End_datetime" in result.columns:
        ends = pd.to_datetime(result["End_datetime"], errors="coerce")
    elif {"Date", "End time"}.issubset(result.columns):
        ends = pd.to_datetime(
            result["Date"].astype(str) + " " + result["End time"].astype(str),
            errors="coerce",
            dayfirst=True,
        )
    else:
        ends = pd.to_datetime(result.get("End time"), errors="coerce")

    result["start_dt"] = starts
    duration = pd.to_numeric(result.get("Duration", 0), errors="coerce").fillna(0)
    result["end_dt"] = ends.where(ends.notna(), result["start_dt"] + pd.to_timedelta(duration, unit="s"))
    result.loc[result["end_dt"] <= result["start_dt"], "end_dt"] = (
        result["start_dt"] + pd.to_timedelta(duration.clip(lower=1800), unit="s")
    )
    return result


def _build_schedule_canvas(programmes: pd.DataFrame, schedule: pd.DataFrame) -> list[dict[str, Any]]:
    if programmes.empty:
        return []

    frame = _program_datetime_columns(programmes)
    frame = frame.dropna(subset=["start_dt"])
    frame["day"] = frame["start_dt"].dt.strftime("%a")
    frame["hour"] = frame["start_dt"].dt.hour
    frame["viewing_points"] = pd.to_numeric(frame.get("TVR", 1.0), errors="coerce").fillna(1.0)

    # Join each EPG program to ITS OWN planned row on (channel, date, HH:MM). The
    # reference EPG has no program_type column, so the previous type-level join
    # collapsed every program to "Other" and stamped a whole-type aggregate (a SUM
    # of every "Other" plan row) plus a constant break count onto all of them. The
    # plan row carries the real per-program revenue/retention/num_breaks and the
    # real program_type, so look it up per program; emit honest null/0 when no plan
    # row exists for that exact slot rather than borrowing another program's number.
    plan_index = _plan_by_program_key(schedule)

    rows: list[dict[str, Any]] = []
    for channel, channel_df in frame.sort_values("start_dt").groupby("Channel"):
        programs = []
        for _, row in channel_df.head(18).iterrows():
            plan_row = plan_index.get(
                (str(channel), row["start_dt"].strftime("%Y-%m-%d"), row["start_dt"].strftime("%H:%M"))
            )
            if plan_row is None:
                program_type = "Other"
                revenue: float | None = None
                retention: float | None = None
                break_count = 0
            else:
                program_type = str(plan_row.get("program_type") or "Other")
                revenue = _money(plan_row.get("predicted_revenue", 0.0))
                retention = round(_percent(plan_row.get("predicted_retention", 0.0)), 1)
                break_count = int(max(0, _safe_number(plan_row.get("num_breaks"), 0)))
            programs.append(
                {
                    "title": row.get("Title", "Untitled"),
                    "program_type": program_type,
                    "day": row["day"],
                    # The real calendar date (YYYY-MM-DD) of this programme's slot,
                    # taken straight from the parsed EPG start datetime. It is what the
                    # segment inspector needs to resolve a grid/strip row to its
                    # channel|date|start_clock segment_id; the weekday abbreviation in
                    # "day" alone cannot. Never fabricated: start_dt is dropna'd above,
                    # so every emitted row has a genuine date.
                    "date": row["start_dt"].strftime("%Y-%m-%d"),
                    "time": row["start_dt"].strftime("%H:%M"),
                    "duration_minutes": round(_safe_number(row.get("Duration"), 3600) / 60),
                    "revenue": revenue,
                    "retention": retention,
                    "break_markers": break_count,
                    "selected": len(programs) == 1 and len(rows) == 0,
                }
            )
        rows.append({"channel": channel, "programs": programs})

    return rows[:6]


def _plan_by_program_key(schedule: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Index the weekly plan by the real per-program key (channel, date, HH:MM).

    The reference EPG carries no program_type, so the old (type, day) lookup
    matched every EPG program to one type-level row and stamped that single row's
    revenue/retention/breaks onto every program of the type (a fabrication). The
    plan's own (channel, date, start_time) is a real per-program key: it matches
    100% of plan rows and the great majority of EPG rows on the reference data. A
    few short fillers share a start minute, so rows that collide on the key are
    aggregated (revenue, num_breaks and break-time summed; the highest-revenue row
    supplies the representative type/retention/position/break_type/base_rate).
    Callers look up each program's own slot and emit honest null/0 when there is
    no match.
    """
    index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if schedule.empty:
        return index

    frame = schedule.copy()
    frame["program_type"] = frame.get("program_type", "Other").fillna("Other").astype(str)
    frame["channel_key"] = frame.get("channel", "").astype(str).str.strip()
    frame["date_key"] = frame.get("date", "").astype(str).str.strip()
    frame["time_key"] = frame.get("start_time", "").astype(str).str.strip().str.slice(0, 5)
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0.0), errors="coerce").fillna(0.0)
    frame["num_breaks"] = pd.to_numeric(frame.get("num_breaks", 0), errors="coerce").fillna(0)
    frame["total_break_time"] = pd.to_numeric(
        frame.get("total_break_time", frame.get("break_length", 0)), errors="coerce"
    ).fillna(0)
    frame["break_length"] = pd.to_numeric(
        frame.get("break_length", frame.get("total_break_time", 120)), errors="coerce"
    ).fillna(120)

    valid = frame[(frame["channel_key"] != "") & (frame["date_key"] != "") & (frame["time_key"] != "")]
    for key, group in valid.groupby(["channel_key", "date_key", "time_key"], sort=False):
        representative = group.sort_values("predicted_revenue", ascending=False).iloc[0]
        record = representative.to_dict()
        record["predicted_revenue"] = float(group["predicted_revenue"].sum())
        record["num_breaks"] = float(group["num_breaks"].sum())
        record["total_break_time"] = float(group["total_break_time"].sum())
        index[(str(key[0]), str(key[1]), str(key[2]))] = record

    return index


def _build_break_operations(programmes: pd.DataFrame, schedule: pd.DataFrame) -> dict[str, Any]:
    if programmes.empty:
        return {"programs": [], "breaks": [], "summary": {"programs": 0, "breaks": 0, "ad_seconds": 0, "revenue": 0}}

    frame = _program_datetime_columns(programmes)
    frame = frame.dropna(subset=["start_dt", "end_dt"]).copy()
    if frame.empty:
        return {"programs": [], "breaks": [], "summary": {"programs": 0, "breaks": 0, "ad_seconds": 0, "revenue": 0}}

    frame["program_type"] = (
        frame["program_type"] if "program_type" in frame.columns
        else frame["programme_type"] if "programme_type" in frame.columns
        else pd.Series("Other", index=frame.index)
    ).fillna("Other").astype(str)
    frame["viewing_points"] = pd.to_numeric(frame.get("TVR", 1.0), errors="coerce").fillna(1.0)
    frame["day_key"] = frame["start_dt"].dt.strftime("%a")
    frame["duration_seconds"] = (frame["end_dt"] - frame["start_dt"]).dt.total_seconds().clip(lower=0)
    frame = frame.sort_values("start_dt").groupby("Channel", dropna=False).head(12).reset_index(drop=True)

    plan_index = _plan_by_program_key(schedule)
    settings = _load_settings()
    _pricing_model: Any = None
    if _ENGINE_AVAILABLE:
        try:
            _pricing_model = PricingModel.from_yaml()
        except Exception:
            logger.exception("pricing config unavailable; per-break premiums will be 1.0")

    programs: list[dict[str, Any]] = []
    breaks: list[dict[str, Any]] = []

    for row_index, row in frame.iterrows():
        channel = str(row.get("Channel") or row.get("channel") or "Channel")
        day = str(row.get("day_key") or "")
        program_id = str(row.get("programme_id") or row.get("id") or f"program-{row_index}")
        program_key = f"{channel}-{program_id}-{row['start_dt'].strftime('%H%M')}"
        duration_seconds = int(_safe_number(row.get("duration_seconds"), 0))
        duration_minutes = round(duration_seconds / 60, 1)
        # Look up THIS program's own planned row on (channel, date, HH:MM); the EPG
        # has no program_type so a type-level join would stamp one row onto every
        # program of the type. No match means no plan for this slot: honest 0/null,
        # not a borrowed aggregate. The real program_type comes from the plan row.
        schedule_row = plan_index.get(
            (channel, row["start_dt"].strftime("%Y-%m-%d"), row["start_dt"].strftime("%H:%M"))
        ) or {}
        program_type = str(schedule_row.get("program_type") or "Other")
        planned_breaks = int(max(0, _safe_number(schedule_row.get("num_breaks"), 0)))
        capacity_breaks = int(max(0, duration_minutes // 18))
        break_count = max(0, min(5, planned_breaks, capacity_breaks if duration_minutes >= 18 else 0))
        has_plan = bool(schedule_row)
        revenue_total = _money(schedule_row.get("predicted_revenue", 0.0)) if has_plan else None
        retention = round(_percent(schedule_row.get("predicted_retention", 0.0)), 1) if has_plan else None
        break_seconds = int(max(30, min(360, _safe_number(schedule_row.get("break_length"), 120))))
        lane = f"{channel} / {day}"

        programs.append(
            {
                "id": program_id,
                "key": program_key,
                "lane": lane,
                "channel": channel,
                "title": row.get("Title", "Untitled"),
                "program_type": program_type,
                "day": day,
                "date": row["start_dt"].date().isoformat(),
                "start_time": row["start_dt"].strftime("%H:%M"),
                "end_time": row["end_dt"].strftime("%H:%M"),
                "duration_minutes": duration_minutes,
                "revenue": revenue_total,
                "retention": retention,
                "break_markers": break_count,
            }
        )

        if break_count == 0:
            continue

        # Per-break figures fall back to 0 only for the arithmetic below; the
        # program row above keeps the honest null when no plan row matched.
        revenue_for_breaks = revenue_total if revenue_total is not None else 0.0
        retention_for_breaks = retention if retention is not None else 0.0

        for break_index in range(1, break_count + 1):
            candidate = row["start_dt"] + pd.Timedelta(seconds=int((duration_seconds / (break_count + 1)) * break_index))
            min_start = row["start_dt"] + pd.Timedelta(minutes=2)
            max_start = row["end_dt"] - pd.Timedelta(seconds=break_seconds + 60)
            if max_start > min_start:
                if candidate < min_start:
                    candidate = min_start
                if candidate > max_start:
                    candidate = max_start
            break_end = candidate + pd.Timedelta(seconds=break_seconds)
            # Gold comes from the PLAN, never a heuristic: the saved row's
            # is_gold flag (set only when the optimizer actually emitted a gold
            # break for the segment) marks this programme's first break. The old
            # prime-time+settings synthesis showed gold that did not exist and
            # even exceeded the daily cap on busy evenings.
            row_gold = str(schedule_row.get("is_gold", "")).strip().lower() in ("true", "1", "yes")
            is_gold = bool(row_gold and break_index == 1)
            reference_revenue = _money(revenue_for_breaks / max(break_count, 1))
            rating_points = _safe_number(row.get("viewing_points"), 0.0)
            # base_rate comes from the optimizer's weekly schedule CSV. Absent
            # means no plan was run; report None rather than inventing 1000.
            raw_base_rate = schedule_row.get("base_rate")
            cpp: float | None = _safe_number(raw_base_rate, -1.0) if raw_base_rate is not None else None
            if cpp is not None and cpp < 0:
                cpp = None
            # Premium from config; never hardcode 1.25 for gold.
            program_premium = _pricing_model.program_premium(program_type) if _pricing_model is not None else 1.0
            break_revenue: float
            if cpp is not None and cpp > 0 and rating_points > 0:
                try:
                    cpp_revenue = cpp_break_revenue(rating_points, break_seconds, cpp, premium=program_premium)
                    break_revenue = _money(retention_adjusted_revenue(cpp_revenue, retention_for_breaks / 100))
                except ValueError:
                    break_revenue = reference_revenue
            else:
                break_revenue = reference_revenue
            breaks.append(
                {
                    "id": f"{program_key}-br-{break_index}",
                    "program_id": program_id,
                    "program_key": program_key,
                    "program_title": row.get("Title", "Untitled"),
                    "lane": lane,
                    "channel": channel,
                    "day": day,
                    "date": row["start_dt"].date().isoformat(),
                    "program_type": program_type,
                    "position": schedule_row.get("position", "middle"),
                    "break_type": schedule_row.get("break_type", "regular"),
                    "break_num_in_program": break_index,
                    "breaks_in_program": break_count,
                    "start_time": candidate.strftime("%H:%M"),
                    "end_time": break_end.strftime("%H:%M"),
                    "duration_sec": break_seconds,
                    "sponsorships_count": 1 if is_gold else 0,
                    "is_gold": is_gold,
                    "source": "Model",
                    "rating_predicted": round(_safe_number(row.get("viewing_points"), 0.0), 2),
                    "cpp": _money(cpp) if cpp is not None else None,
                    "revenue_reference": reference_revenue,
                    "revenue_premium": program_premium,
                    "revenue_calculated": break_revenue,
                    "retention": retention_for_breaks,
                    "status": "at_risk" if retention_for_breaks < settings.min_retention_floor * 100 else "ready",
                }
            )

    return {
        "programs": programs,
        "breaks": breaks,
        "summary": {
            "programs": len(programs),
            "breaks": len(breaks),
            "ad_seconds": int(sum(item["duration_sec"] for item in breaks)),
            "revenue": _money(sum(item["revenue_calculated"] for item in breaks)),
        },
    }


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


def _proposed_kind(risk: str, num_breaks: int, is_gold: bool) -> str | None:
    """The override a review of this break honestly implies, or None when the break
    is inherently advisory (nothing concrete to change).

    A break below the retention floor (High risk) should shed load: lower its count
    when it carries more than one break, or forbid it when it carries a single break.
    A healthy break (at or above the floor) that already earns is worth protecting:
    tag it gold, or pin its count when it is already gold. A zero-break healthy
    segment has nothing to act on.
    """
    if risk == "High":
        if num_breaks > 1:
            return "lower_count"
        if num_breaks == 1:
            return "forbid"
        return None
    if num_breaks <= 0:
        return None
    return "pin" if is_gold else "gold"


def _build_recommendations(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    if schedule.empty:
        return []

    settings = _load_settings()
    owned = str(settings.operator_channel or "").strip()
    frame = _augment_segment_ids(schedule)
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0.0), errors="coerce").fillna(0.0)
    frame["num_breaks"] = pd.to_numeric(frame.get("num_breaks", 0), errors="coerce").fillna(0).astype(int)

    # Competitor boundary: only the operator's OWNED channel produces
    # recommendations, and only owned-channel segments are ever offered as targets.
    # With no channel configured yet, recommendations stay advisory (channel null, no
    # segment mapping) rather than binding to a channel the operator may not own.
    scoped = frame[frame["channel"].astype(str).str.strip() == owned] if owned else frame
    if scoped.empty:
        return []
    scoped = scoped.sort_values(["predicted_revenue", "predicted_retention"], ascending=[False, True])

    # Risk labels are sourced from the operator's configured retention floor, not
    # fixed literals. Below the floor is High; within a 2-point band above it is
    # Medium; clear of the band is Low. This mirrors the honest _risk_from_retention
    # scale so the recommendation risk and the headline risk score agree.
    floor_percent = round(settings.min_retention_floor * 100, 1)

    def _risk_label(retention_percent: float) -> str:
        if retention_percent < floor_percent:
            return "High"
        if retention_percent < floor_percent + 2.0:
            return "Medium"
        return "Low"

    # Identity and grouping key on the segment's REAL distinguishing facts (its
    # programme type, clock, weekday and canonical daypart), never on the
    # constant position/break_type template the CSV once stamped on every row.
    from kairos.data.dayparts import daypart_for_hour

    def _row_hour(value: Any) -> int | None:
        text = str(value or "").strip()
        return int(text[:2]) if len(text) >= 4 and text[:2].isdigit() else None

    scoped = scoped.copy()
    scoped["_daypart"] = scoped["start_time"].map(lambda v: daypart_for_hour(_row_hour(v)) or "")
    scoped["_risk"] = scoped["predicted_retention"].map(lambda v: _risk_label(_percent(v)))

    weekday_he = {"Mon": "ביום שני", "Tue": "ביום שלישי", "Wed": "ביום רביעי", "Thu": "ביום חמישי", "Fri": "ביום שישי", "Sat": "בשבת", "Sun": "ביום ראשון"}
    # Honest per-risk rationale: the copy states what the risk band actually
    # means against the configured floor, instead of one constant sentence.
    rationale_by_risk = {
        "High": ("Predicted retention is below the configured floor; review whether this segment should shed break load.", "השימור החזוי נמוך מרצפת השימור שהוגדרה; כדאי לבחון הפחתת עומס ברייקים במקטע הזה."),
        "Medium": ("Predicted retention is within two points of the floor; review before committing.", "השימור החזוי קרוב לרצפת השימור, בטווח שתי נקודות; מומלץ לבדוק לפני אישור."),
        "Low": ("Revenue is strong and retention clears the floor; consider protecting this placement.", "ההכנסה גבוהה והשימור מעל הרצפה; שקלו להגן על השיבוץ הזה."),
    }

    actions = []
    for idx, row in scoped.head(5).iterrows():
        retention = _percent(row.get("predicted_retention", 0.0))
        revenue = _money(row.get("predicted_revenue", 0))
        num_breaks = int(row.get("num_breaks", 0))
        risk = str(row.get("_risk", "Low"))
        program_type = str(row.get("program_type", "Other"))
        daypart = str(row.get("_daypart", "")).strip()
        start_clock = str(row.get("start_time", "")).strip()
        date = str(row.get("date", "")).strip()
        weekday = str(row.get("day", "")).strip()
        segment_id = str(row.get("segment_id", "")).strip()
        anchor = _row_anchor(row)
        proposed_kind = _proposed_kind(risk, num_breaks, bool(row.get("is_gold", False)))
        # The title carries the real programme identity (type, clock, weekday,
        # date), so five recommendations can never render one generic label.
        unit = "break" if num_breaks == 1 else "breaks" if num_breaks > 1 else "segment"
        title = " ".join(part for part in ["Review the", start_clock, program_type, unit, "on", weekday, date] if part)
        unit_he = "הברייק" if num_breaks == 1 else "הברייקים" if num_breaks > 1 else "המקטע"
        day_phrase_he = weekday_he.get(weekday, "")
        title_he = f"בדיקת {unit_he} בתוכנית {program_type} בשעה {start_clock} {day_phrase_he}, {date}".replace("  ", " ")
        rationale, rationale_he = rationale_by_risk[risk]
        # Candidate owned-channel segments this review resolves to, grouped on the
        # same real facts (programme type, daypart, risk band). Only produced for
        # the owned channel with a real segment_id; never fabricated for an
        # aggregate advisory.
        candidates: list[dict[str, Any]] = []
        if owned and segment_id:
            same = scoped[
                (scoped["program_type"].astype(str) == program_type)
                & (scoped["_daypart"] == daypart)
                & (scoped["_risk"] == risk)
            ]
            candidates = [
                {"segment_id": str(cand.get("segment_id", "")).strip(), "anchor": _row_anchor(cand)}
                for _, cand in same.head(12).iterrows()
            ]
        actionable = bool(owned and segment_id and proposed_kind)
        actions.append(
            {
                "id": f"rec-{idx}",
                "title": title,
                "title_he": title_he,
                "program_type": program_type,
                # Real identity fields behind the title, so consumers can group
                # or render without re-parsing display copy.
                "start_clock": start_clock,
                "date": date,
                "weekday": weekday,
                "daypart": daypart or None,
                "num_breaks": num_breaks,
                "impact": revenue,
                "retention": round(retention, 1),
                "risk": risk,
                "rationale": rationale,
                "rationale_he": rationale_he,
                # Decision-plane enrichment: the owned channel, the concrete
                # owned-channel segment(s) this resolves to, and the override kind the
                # review implies. non_actionable marks an advisory that cannot honestly
                # bind to a segment (no owned channel, or nothing to change).
                "channel": owned or None,
                "actionable": actionable,
                "non_actionable": not actionable,
                "segment_id": segment_id if actionable else None,
                "anchor": anchor if actionable else None,
                "proposed_kind": proposed_kind if actionable else None,
                "candidates": candidates,
            }
        )
    return actions


def _parse_frontier_scope(scope: str | None, settings: KairosSettings) -> dict[str, str | None]:
    """Parse a ``scope=channel:<id>`` or ``scope=day:<date>`` query into run_scenario kwargs.

    Returns ``{"channel": ..., "day": ...}`` to forward to :func:`run_scenario`.
    No scope (None/empty) returns both None, which preserves the current
    whole-default behaviour (run_scenario auto-detects the first channel-day),
    making the unscoped frontier byte-identical to before. Only the operator's
    OWNED channel is selectable: a channel scope that does not match the configured
    operator_channel is rejected (treated as no scope) so no competitor channel can
    ever be requested. An unrecognised prefix is ignored (honest no-op).
    """
    result: dict[str, str | None] = {"channel": None, "day": None}
    text = str(scope or "").strip()
    if not text or ":" not in text:
        return result
    prefix, _, value = text.partition(":")
    prefix = prefix.strip().lower()
    value = value.strip()
    if not value:
        return result
    if prefix == "channel":
        owned = str(settings.operator_channel or "").strip()
        # When an owned channel is configured, only it is selectable. When it is
        # not configured yet, accept the requested channel (no competitor boundary
        # to enforce against) so the feature is usable in the unconfigured state.
        if owned and value != owned:
            return result
        result["channel"] = value
    elif prefix == "day":
        result["day"] = value
    return result


def _frontier_data_signature() -> tuple[tuple[str, int], ...]:
    """A cheap hashable signature of the data files the frontier depends on, so
    the frontier cache invalidates automatically when programmes, spots or the
    planned schedule change on disk."""
    candidates = [
        OUTPUT_DIR / "weekly_break_schedule.csv",
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "reference" / "Spots.xlsx",
        DATA_DIR / "Programmes.csv",
        DATA_DIR / "Spots.csv",
    ]
    sig: list[tuple[str, int]] = []
    for path in candidates:
        try:
            sig.append((str(path), path.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(sig)


@lru_cache(maxsize=32)
def _frontier_points_cached(
    signature: tuple[tuple[str, int], ...],
    channel: str,
    day: str | None,
    saved_floor: float,
    max_breaks_per_hour: int,
    risk_lambda: float,
    revenue_weight: int,
) -> tuple[dict[str, Any], ...]:
    """Trace the genuine revenue-vs-retention Pareto frontier for one owned scope.

    The frontier sweeps the RETENTION FLOOR at the saved revenue weight, and each
    point is the REFINED optimum (``refine=True``), not a greedy approximation.
    This is a deliberate correctness choice backed by measurement: at a fixed
    floor the refined optimum is nearly invariant to the revenue weight above a
    low threshold (once retention already clears the floor, the weight barely
    moves the plan), so a revenue-weight sweep collapses onto a single point, and
    any spread it appears to show is an artifact of the weaker greedy optimizer
    leaving a different amount of revenue on the table at each weight. The
    retention floor is the binding lever: tightening it sheds the lowest-value
    breaks, trading revenue for retention, which is the real tradeoff the
    operator is choosing between. Cached on the data-file ``signature`` plus the
    guardrail inputs so the sweep runs once and is reused across requests.
    """
    del signature  # part of the cache key only
    anchor = round(float(saved_floor), 4)
    floors = sorted({0.72, 0.80, 0.85, 0.90, 0.93, 0.97, anchor})
    pacing = _pacing_call_kwargs()
    points: list[dict[str, Any]] = []
    for floor in floors:
        try:
            payload = run_scenario(
                revenue_weight=revenue_weight,
                retention_floor=floor,
                max_breaks_per_hour=max_breaks_per_hour,
                risk_lambda=risk_lambda,
                channel=channel,
                day=day,
                refine=True,
                **pacing,
            )
        except Exception:
            logger.exception("frontier scenario failed at retention_floor=%s", floor)
            continue
        summary = payload.get("summary", {})
        retention = summary.get("average_retention")
        revenue = summary.get("projected_revenue")
        if retention is None or revenue is None:
            continue
        points.append(
            {
                "retention": round(_safe_number(retention), 1),
                "revenue": round(_safe_number(revenue), 2),
                "retention_floor": round(float(floor), 4),
                "num_breaks": int(_safe_number(summary.get("total_breaks", 0))),
                "selected": abs(float(floor) - anchor) < 1e-9,
            }
        )
    points.sort(key=lambda point: point["retention"])
    return tuple(points)


@lru_cache(maxsize=8)
def _owned_representative_day(signature: tuple[tuple[str, int], ...], owned: str) -> str | None:
    """The owned channel's busiest broadcast day (most programmes), as YYYY-MM-DD.

    The frontier is traced on this single representative day, not across every
    broadcast day of the channel. A real owned channel spans dozens of broadcast
    days and each day is a full refined optimization per retention-floor step, so a
    whole-channel sweep is many minutes of compute; one full day is interactive.
    The busiest day gives the richest, most distinct curve (a thin day collapses
    the Pareto points on top of each other). Ties break to the latest date for a
    deterministic, recent forecast. Returns ``None`` when the channel has no dated
    programmes.
    """
    del signature  # cache key only
    try:
        programmes = _load_programmes()
    except Exception:
        logger.exception("representative-day load failed")
        return None
    if programmes.empty or "Channel" not in programmes.columns or "start_dt" not in programmes.columns:
        return None
    owned_rows = programmes[programmes["Channel"].astype(str) == owned]
    owned_rows = owned_rows[owned_rows["start_dt"].notna()]
    if owned_rows.empty:
        return None
    days = owned_rows["start_dt"].dt.strftime("%Y-%m-%d")
    counts = days.value_counts()
    busiest = counts[counts == counts.max()].index
    return max(busiest)  # YYYY-MM-DD sorts lexicographically; latest of the busiest


# Label id of the additive net-focused scenario point served beside the frontier
# sweep (the whole-schedule optimum under objective_mode='revenue_net').
NET_POINT_ID = "net_focused"


def _scenario_plan_money(
    payload: dict[str, Any], segments: list[Any], risk_lambda: float
) -> dict[str, Any]:
    """Price one run_scenario plan in ILS on the engine's own plan money model.

    Joins the payload's per-segment break counts back to the rebuilt
    ProgramSegment objects and prices the plan with
    :func:`kairos.optimize.revenue_net.plan_revenue_net`: gross is the runner's
    own projected revenue, the retention cost is the per-break audience loss
    priced at the same CPP, and net is their difference. This is the SAME
    per-break cost model the committed plan's /api/yield-per-second money uses,
    so the comparison and the committed story share one basis. Coefficients are
    first risk-adjusted exactly as the optimizer decided
    (:func:`kairos.optimize._segment_math._risk_adjusted_coefficient`, an exact
    identity at risk_lambda 0). Returns the money block, or an honest
    ``{"available": False, "reason": ...}`` when the plan cannot be priced;
    nothing is proxied.
    """
    from dataclasses import replace as _dataclass_replace
    from types import SimpleNamespace

    from kairos.optimize._segment_math import _risk_adjusted_coefficient
    from kairos.optimize.revenue_net import plan_revenue_net

    summary = payload.get("summary", {})
    plan_rows = payload.get("segments", [])
    adjusted = [
        _dataclass_replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda))
        for s in segments
    ]
    shim = SimpleNamespace(
        segments=[
            SimpleNamespace(
                segment_id=str(row.get("segment_id", "")),
                num_breaks=int(_safe_number(row.get("num_breaks", 0))),
                revenue=float(_safe_number(row.get("revenue", 0.0))),
            )
            for row in plan_rows
        ],
        total_revenue=float(_safe_number(summary.get("projected_revenue", 0.0))),
    )
    money = plan_revenue_net(shim, segments=adjusted)
    if not money.get("available"):
        return {"available": False, "reason": str(money.get("reason") or "plan money unavailable")}
    if int(money.get("priced_segments") or 0) < len(plan_rows):
        return {
            "available": False,
            "reason": (
                "scenario plan and rebuilt segments no longer join; "
                "retention cannot be priced honestly"
            ),
        }
    return {
        "available": True,
        "gross": money["revenue_ils"],
        "retention_cost": money["retention_cost_ils"],
        "net": money["revenue_net_ils"],
        "breaks": int(_safe_number(summary.get("total_breaks", 0))),
    }


def _net_bundle_failure(channel: str, day: str | None, reason: str) -> dict[str, Any]:
    """Honest empty net bundle: no point, no money, the reason named."""
    return {
        "channel": channel,
        "day": day,
        "net_point": None,
        "comparison_available": False,
        "reason": reason,
        "current": None,
        "net_focused": None,
    }


@lru_cache(maxsize=32)
def _frontier_net_bundle_cached(
    signature: tuple[tuple[str, int], ...],
    channel: str,
    day: str | None,
    saved_floor: float,
    max_breaks_per_hour: int,
    risk_lambda: float,
    revenue_weight: int,
    objective_mode: str,
) -> dict[str, Any]:
    """One net-focused whole-schedule scenario beside the sweep, with money.

    Runs the SAME scenario runner as the frontier sweep on the same owned scope,
    refined, under the saved guardrails: once at the operator's saved
    ``objective_mode`` (the 'current' side, the saved decision re-evaluated so a
    money block exists on the sweep's own anchor basis) and once under
    ``objective_mode='revenue_net'`` (the net-focused side). When the saved mode
    already is ``revenue_net`` the single run serves both sides. Each plan is
    priced with the per-break retention-cost model
    (:func:`_scenario_plan_money`), so the operator sees gross, the
    model-estimated retention cost, and net on one shared basis. Cached beside
    the point sweep on the same data signature and computed in the same single
    background thread, never inline in a request.
    """
    del signature  # part of the cache key only
    if day is None:
        return _net_bundle_failure(
            channel, day, "owned channel has no dated programmes to scope the comparison"
        )
    pacing = _pacing_call_kwargs()

    def _run(mode: str) -> dict[str, Any]:
        return run_scenario(
            revenue_weight=revenue_weight,
            retention_floor=saved_floor,
            max_breaks_per_hour=max_breaks_per_hour,
            risk_lambda=risk_lambda,
            channel=channel,
            day=day,
            objective_mode=mode,
            **pacing,
        )

    try:
        net_payload = _run("revenue_net")
        current_payload = (
            net_payload if objective_mode == "revenue_net" else _run(objective_mode)
        )
    except Exception:
        logger.exception("net-focused frontier scenario failed")
        return _net_bundle_failure(
            channel, day, "net-focused scenario run failed; see the server log"
        )

    summary = net_payload.get("summary", {})
    net_point: dict[str, Any] | None = None
    if summary.get("average_retention") is not None and summary.get("projected_revenue") is not None:
        # Same fields as a frontier sweep point, plus the label id. 'selected' is
        # honest: True only when the saved objective_mode IS revenue_net.
        net_point = {
            "retention": round(_safe_number(summary.get("average_retention")), 1),
            "revenue": round(_safe_number(summary.get("projected_revenue")), 2),
            "retention_floor": round(float(saved_floor), 4),
            "num_breaks": int(_safe_number(summary.get("total_breaks", 0))),
            "selected": objective_mode == "revenue_net",
            "id": NET_POINT_ID,
        }

    try:
        segments = list(_plan_segment_index(((channel, str(day)),), pacing["settings"]).values())
        money_current = _scenario_plan_money(current_payload, segments, risk_lambda)
        money_net = _scenario_plan_money(net_payload, segments, risk_lambda)
    except Exception:
        logger.exception("net-focused plan pricing failed")
        bundle = _net_bundle_failure(
            channel, day, "plan money pricing failed; see the server log"
        )
        bundle["net_point"] = net_point
        return bundle

    available = bool(money_current.get("available") and money_net.get("available"))
    reason = None
    if not available:
        reason = str(
            (money_current if not money_current.get("available") else money_net).get("reason")
            or "comparison money unavailable"
        )
    return {
        "channel": channel,
        "day": day,
        "net_point": net_point,
        "comparison_available": available,
        "reason": reason,
        "current": money_current if money_current.get("available") else None,
        "net_focused": money_net if money_net.get("available") else None,
    }


# The frontier is a real optimizer sweep and is too slow to trace inline on a cold
# cache, so it is computed in a background thread. These guard the single in-flight
# computation and its result so /api/overview never blocks on the sweep.
_frontier_bg_lock = threading.Lock()
_frontier_bg_state: dict[str, Any] = {
    "key": None, "status": "idle", "points": (), "net_bundle": None,
}


def _frontier_async(settings: KairosSettings, scope: str | None = None) -> tuple[list[dict[str, Any]], str]:
    """Return ``(points, status)`` for the revenue-vs-retention frontier without
    ever blocking the request.

    Each point is an ACTUAL optimization (:func:`kairos.service.run_scenario`),
    not a synthetic offset off one summary: the curve is the genuine Pareto
    trade-off the engine produces as it shifts from retention-first
    (revenue_weight 0) to revenue-first (revenue_weight 100), under the saved
    retention floor, hourly break cap and risk aversion. The point matching the
    saved revenue_weight is marked ``selected``.

    The frontier forecasts the operator's OWNED channel inventory only. Revenue is
    never projected for a competitor channel: competitor programming informs the
    churn/retention model, not the revenue projection (the competitor-information
    boundary). The curve is scoped to ``settings.operator_channel`` on its busiest
    broadcast day (see :func:`_owned_representative_day`); a ``day:<date>`` scope
    narrows it to another day within the owned channel, and a ``channel:<id>`` scope
    is accepted only when it equals the owned channel, so the forecast can never be
    redirected to a competitor.

    Status is one of: ``no_channel`` (no owned channel set yet, points empty: the
    dashboard prompts the operator to pick their channel), ``computing`` (a
    background sweep is in flight, points empty: an honest "forecast is being
    computed" state, never a fabricated curve), or ``ready`` (points populated from
    the finished sweep). The sweep itself is cached on the data-file signature plus
    the guardrails, so it runs once and is reused across requests and weights.
    """
    points, _net_bundle, status = _frontier_state(settings, scope)
    return points, status


def _frontier_state(
    settings: KairosSettings, scope: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    """``(points, net_bundle, status)`` for the frontier machinery, never blocking.

    The single shared engine behind :func:`_frontier_async` (whose points/status
    contract is unchanged), the overview's additive ``net_point``, and the
    ``/api/optimizer/net-comparison`` endpoint. ONE background thread computes
    the point sweep and the net-focused bundle together under one key, so their
    statuses can never disagree and no second background machine exists. The key
    extends the sweep key with the saved ``objective_mode`` (the 'current' side
    of the bundle is evaluated at it), so a mode edit honestly re-enters
    ``computing``; the sweep points themselves are cached without the mode and
    are byte-identical across that transition. ``net_bundle`` is ``None`` until
    status is ``ready``; a ready bundle may still report
    ``comparison_available: False`` with a reason, never invented numbers.
    """
    owned = str(settings.operator_channel or "").strip()
    if not owned:
        return [], None, "no_channel"
    signature = _frontier_data_signature()
    scope_kwargs = _parse_frontier_scope(scope, settings)
    effective_day = scope_kwargs["day"] or _owned_representative_day(signature, owned)
    key = (
        signature,
        owned,
        effective_day,
        float(settings.min_retention_floor),
        int(settings.max_breaks_per_hour),
        float(settings.risk_lambda),
        int(settings.revenue_weight),
        str(getattr(settings, "objective_mode", "blend") or "blend"),
    )
    with _frontier_bg_lock:
        state = _frontier_bg_state
        if state["key"] == key and state["status"] == "ready":
            return (
                [dict(point) for point in state["points"]],
                copy.deepcopy(state.get("net_bundle")),
                "ready",
            )
        if state["key"] == key and state["status"] == "computing":
            return [], None, "computing"
        # New (or stale) key: start a fresh single in-flight computation.
        state["key"] = key
        state["status"] = "computing"
        state["points"] = ()
        state["net_bundle"] = None

    def _compute() -> None:
        try:
            points = _frontier_points_cached(*key[:7])
        except Exception:
            logger.exception("frontier background compute failed")
            points = ()
        try:
            net_bundle = _frontier_net_bundle_cached(*key)
        except Exception:
            logger.exception("frontier net-focused bundle compute failed")
            net_bundle = _net_bundle_failure(
                key[1], key[2], "net-focused computation failed; see the server log"
            )
        with _frontier_bg_lock:
            if _frontier_bg_state["key"] == key:
                _frontier_bg_state["points"] = points
                _frontier_bg_state["net_bundle"] = net_bundle
                _frontier_bg_state["status"] = "ready"

    threading.Thread(target=_compute, name="kairos-frontier", daemon=True).start()
    return [], None, "computing"


def _infer_hourly_ad_seconds(schedule: pd.DataFrame) -> pd.Series:
    if schedule.empty:
        return pd.Series(dtype=float)

    frame = schedule.copy()
    frame["ad_seconds"] = pd.to_numeric(
        frame.get("total_break_time", frame.get("break_length", 0)),
        errors="coerce",
    ).fillna(0)

    if "hour" not in frame.columns:
        candidate = None
        for column in ["start_time", "time", "break_start", "Start time"]:
            if column in frame.columns:
                candidate = pd.to_datetime(frame[column], errors="coerce")
                break
        if candidate is not None:
            frame["hour"] = candidate.dt.hour
        else:
            frame["hour"] = 0

    group_columns = [column for column in ["date", "Channel", "channel", "hour"] if column in frame.columns]
    if not group_columns:
        group_columns = ["hour"]
    return frame.groupby(group_columns)["ad_seconds"].sum()


def _infer_hourly_break_counts(schedule: pd.DataFrame) -> pd.Series:
    if schedule.empty:
        return pd.Series(dtype=float)
    frame = schedule.copy()
    frame["break_count"] = pd.to_numeric(frame.get("num_breaks", 1), errors="coerce").fillna(1)
    group_columns = [column for column in ["date", "Channel", "channel", "hour"] if column in frame.columns]
    if not group_columns:
        group_columns = ["program_type"] if "program_type" in frame.columns else []
    if not group_columns:
        return frame["break_count"]
    return frame.groupby(group_columns)["break_count"].sum()


@lru_cache(maxsize=2)
def _plan_guardrail_items_cached(signature: tuple[tuple[str, int], ...]) -> tuple[GuardrailBreak, ...]:
    """The exact break geometry of the FULL committed plan, for compliance.

    Rebuilds the engine's own segments from the reference EPG, joins them to the
    saved weekly CSV by segment_id, and lays each row's breaks with the same
    _segment_break_objects the optimizer's guardrail check uses, carrying the
    row's true is_gold. This covers every channel-day of the plan (the previous
    source, the break-operations board, truncated to the first 12 programmes per
    channel and synthesized gold flags, so the compliance verdict watched under
    one percent of the plan). Cached on the plan+EPG signature; empty result
    means the geometry could not be joined and callers fall back honestly.
    """
    del signature  # cache key only
    schedule = _load_break_schedule()
    if schedule.empty or "segment_id" not in schedule.columns:
        return ()
    try:
        from kairos.data import ProgramClassifier
        from kairos.data.loaders import load_programmes as _load_prog
        from kairos.data.transform import build_segments_from_programmes
        from kairos.model.impact import load_impact_model
        from kairos.optimize._segment_math import _segment_break_objects
        from kairos.optimize.pricing import OptimizerAssumptions
        from kairos.service import pricing_from_settings

        # segment_id indexes reset per channel-day build, so segments must be
        # rebuilt per (channel, date) pair exactly like the export loop, with
        # the shared resources loaded once.
        programmes = _load_prog()
        settings_map = _model_dump(_load_settings())
        pricing = pricing_from_settings(settings_map)
        assumptions = OptimizerAssumptions()
        impact = load_impact_model(MODELS_DIR / "tv_break_posterior.pkl", assumptions=assumptions)
        classifier = ProgramClassifier.from_yaml()
        pairs = (
            schedule[["channel", "date"]].astype(str).drop_duplicates().itertuples(index=False)
        )
        by_id: dict[str, Any] = {}
        for channel_name, date_str in pairs:
            day_segments = build_segments_from_programmes(
                programmes, classifier, pricing,
                assumptions=assumptions, impact_model=impact,
                channel=channel_name, day=date_str,
            )
            for segment in day_segments:
                by_id[segment.segment_id] = segment
    except Exception:
        logger.exception("plan guardrail geometry unavailable")
        return ()
    frame = _augment_segment_ids(schedule)
    items: list[GuardrailBreak] = []
    joined = 0
    for row in frame.itertuples(index=False):
        segment = by_id.get(str(getattr(row, "segment_id", "")))
        if segment is None:
            continue
        joined += 1
        count = int(_safe_number(getattr(row, "num_breaks", 0)))
        if count <= 0:
            continue
        gold = str(getattr(row, "is_gold", "")).strip().lower() in ("true", "1", "yes")
        items.extend(_segment_break_objects(segment, count, is_gold=gold))
    if joined < len(frame) * 0.99:
        # The EPG no longer matches the saved plan (a re-ingest happened without
        # a recompute). A partial verdict would be dishonest; report nothing and
        # let the caller fall back, with the freshness banner telling the story.
        logger.warning(
            "plan guardrail geometry joined %s of %s rows; falling back", joined, len(frame)
        )
        return ()
    return tuple(items)


def _plan_guardrail_items() -> list[GuardrailBreak]:
    return list(_plan_guardrail_items_cached(_signature([
        OUTPUT_DIR / "weekly_break_schedule.csv",
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        SETTINGS_PATH,
    ])))


def _guardrail_breaks_from_operations(operations: dict[str, Any]) -> list[GuardrailBreak]:
    out: list[GuardrailBreak] = []
    for item in operations.get("breaks", []):
        start_seconds = _time_to_seconds(item.get("start_time"))
        duration_seconds = _safe_number(item.get("duration_sec"), 0)
        if duration_seconds <= 0:
            continue
        out.append(
            GuardrailBreak(
                channel=str(item.get("channel") or "Channel"),
                day=str(item.get("day") or ""),
                hour=int(start_seconds // 3600),
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                program_type=str(item.get("program_type") or "Other"),
                retention=_ratio(item.get("retention")),
                is_gold=bool(item.get("is_gold")),
            )
        )
    return out


def _max_group_sum(items: list[GuardrailBreak], key_fn: Any, value_fn: Any) -> float:
    grouped: dict[Any, float] = {}
    for item in items:
        key = key_fn(item)
        grouped[key] = grouped.get(key, 0.0) + float(value_fn(item))
    return max(grouped.values(), default=0.0)


def _max_group_count(items: list[GuardrailBreak], key_fn: Any) -> int:
    grouped: dict[Any, int] = {}
    for item in items:
        key = key_fn(item)
        grouped[key] = grouped.get(key, 0) + 1
    return max(grouped.values(), default=0)


def _min_break_spacing_seconds(items: list[GuardrailBreak]) -> float | None:
    grouped: dict[tuple[str, str], list[GuardrailBreak]] = {}
    for item in items:
        grouped.setdefault((item.channel, item.day), []).append(item)
    gaps: list[float] = []
    for breaks in grouped.values():
        ordered = sorted(breaks, key=lambda item: item.start_seconds)
        for previous, current in zip(ordered, ordered[1:]):
            gaps.append(current.start_seconds - (previous.start_seconds + previous.duration_seconds))
    return min(gaps) if gaps else None


def _guardrail_compliance_from_breaks(items: list[GuardrailBreak], settings: KairosSettings) -> dict[str, Any] | None:
    if not items:
        return None

    guardrails = _settings_to_guardrails(settings)
    violations = evaluate_guardrails(items, guardrails)
    violation_counts: dict[str, int] = {}
    for violation in violations:
        violation_counts[violation.code] = violation_counts.get(violation.code, 0) + 1

    protected_types = {item.lower() for item in settings.protected_program_types}
    protected_items = [item for item in items if item.program_type.lower() in protected_types]
    max_hourly_seconds = _max_group_sum(items, lambda item: (item.channel, item.day, item.hour), lambda item: item.duration_seconds)
    max_protected_seconds = _max_group_sum(
        protected_items,
        lambda item: (item.channel, item.day, item.hour),
        lambda item: item.duration_seconds,
    )
    min_spacing = _min_break_spacing_seconds(items)
    observed_spacing = min_spacing if min_spacing is not None else settings.min_break_spacing_minutes * 60
    max_daily_seconds = _max_group_sum(items, lambda item: (item.channel, item.day), lambda item: item.duration_seconds)
    max_gold_breaks = _max_group_count(
        [item for item in items if item.is_gold],
        lambda item: (item.channel, item.day),
    )
    min_retention = min((item.retention for item in items), default=0.0)

    checks = [
        {
            "id": "hourly_ad_load",
            "violation_code": "hourly_ad_load",
            "label_en": "Ad minutes per broadcast hour",
            "label_he": "דקות פרסום לשעת שידור",
            "observed": round(max_hourly_seconds / 60, 2),
            "limit": settings.max_ad_minutes_per_hour,
            "unit": "minutes/hour",
        },
        {
            "id": "break_density",
            "violation_code": "breaks_per_hour",
            "label_en": "Breaks per hour",
            "label_he": "מספר ברייקים בשעה",
            "observed": _max_group_count(items, lambda item: (item.channel, item.day, item.hour)),
            "limit": settings.max_breaks_per_hour,
            "unit": "breaks/hour",
        },
        {
            "id": "retention_floor",
            "violation_code": "retention_floor",
            "label_en": "Viewer retention floor",
            "label_he": "רף שימור צפייה",
            "observed": round(min_retention * 100, 1),
            "limit": round(settings.min_retention_floor * 100, 1),
            "unit": "%",
        },
        {
            "id": "protected_programs",
            "violation_code": "hourly_ad_load",
            "label_en": "Protected programme ad load",
            "label_he": "עומס פרסום בתוכן מוגן",
            "observed": round(max_protected_seconds / 60, 2),
            "limit": settings.protected_program_max_ad_minutes_per_hour,
            "unit": "minutes/hour",
        },
        {
            "id": "break_spacing",
            "violation_code": "break_spacing",
            "label_en": "Minimum break spacing",
            "label_he": "מרווח מינימלי בין ברייקים",
            "observed": round(observed_spacing / 60, 2),
            "limit": settings.min_break_spacing_minutes,
            "unit": "minutes",
        },
        {
            "id": "daily_ad_load",
            "violation_code": "daily_ad_load",
            "label_en": "Daily ad load",
            "label_he": "עומס פרסום יומי",
            "observed": round(max_daily_seconds / 60, 2),
            "limit": settings.max_daily_ad_minutes,
            "unit": "minutes/day",
        },
        {
            "id": "gold_breaks",
            "violation_code": "gold_breaks",
            "label_en": "Gold breaks per day",
            "label_he": "ברייקי זהב ביום",
            "observed": max_gold_breaks,
            "limit": settings.gold_breaks_max_per_day,
            "unit": "breaks/day",
        },
    ]

    for check in checks:
        count = violation_counts.get(check["violation_code"], 0)
        if check["id"] == "protected_programs":
            count = sum(
                1
                for violation in violations
                if violation.code == "hourly_ad_load" and "protected programme" in violation.detail
            )
        check["status"] = "at_risk" if count else "compliant"
        check["violations"] = count

    return {
        "checks": checks,
        "violations": [
            {
                "code": violation.code,
                "scope": violation.scope,
                "observed": violation.observed,
                "limit": violation.limit,
                "detail": violation.detail,
            }
            for violation in violations[:200]
        ],
        "status": "at_risk" if violations else "compliant",
    }


def _build_compliance(
    schedule: pd.DataFrame,
    settings: KairosSettings,
    operations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # The verdict is computed from the FULL committed plan's break geometry, not
    # from the break-operations display board (which is truncated to the first
    # programmes per channel for the editor and would silently grade under one
    # percent of the plan). The operations argument is kept for signature
    # compatibility but no longer feeds the verdict.
    del operations
    guardrail_items = _plan_guardrail_items()
    break_level = _guardrail_compliance_from_breaks(guardrail_items, settings)
    if break_level is not None:
        return {
            "profile": settings.profile_name,
            "effective_date": settings.effective_date,
            "source_url": settings.regulatory_source_url,
            "checks": break_level["checks"],
            "violations": break_level["violations"],
            "status": break_level["status"],
            "disclaimer": settings.notes,
        }

    summary = _summarize_schedule(schedule)
    hourly_seconds = _infer_hourly_ad_seconds(schedule)
    hourly_breaks = _infer_hourly_break_counts(schedule)
    max_hourly_minutes = round(float(hourly_seconds.max() / 60), 2) if not hourly_seconds.empty else 0.0
    max_hourly_breaks = int(hourly_breaks.max()) if not hourly_breaks.empty else 0

    protected_minutes = 0.0
    if not schedule.empty and "program_type" in schedule.columns:
        protected_types = {item.lower() for item in settings.protected_program_types}
        protected = schedule[schedule["program_type"].astype(str).str.lower().isin(protected_types)].copy()
        if not protected.empty:
            protected["ad_seconds"] = pd.to_numeric(
                protected.get("total_break_time", protected.get("break_length", 0)),
                errors="coerce",
            ).fillna(0)
            protected_minutes = round(float(protected["ad_seconds"].max() / 60), 2)

    checks = [
        {
            "id": "hourly_ad_load",
            "label_en": "Ad minutes per broadcast hour",
            "label_he": "דקות פרסום לשעת שידור",
            "status": "compliant" if max_hourly_minutes <= settings.max_ad_minutes_per_hour else "at_risk",
            "observed": max_hourly_minutes,
            "limit": settings.max_ad_minutes_per_hour,
            "unit": "minutes/hour",
        },
        {
            "id": "break_density",
            "label_en": "Breaks per hour",
            "label_he": "מספר ברייקים בשעה",
            "status": "compliant" if max_hourly_breaks <= settings.max_breaks_per_hour else "at_risk",
            "observed": max_hourly_breaks,
            "limit": settings.max_breaks_per_hour,
            "unit": "breaks/hour",
        },
        {
            "id": "retention_floor",
            "label_en": "Viewer retention floor",
            "label_he": "רף שימור צפייה",
            # average_retention is None when no schedule has been computed yet;
            # report an honest unknown rather than comparing None to the floor.
            "status": (
                "unknown"
                if summary["average_retention"] is None
                else "compliant"
                if summary["average_retention"] >= settings.min_retention_floor * 100
                else "at_risk"
            ),
            "observed": summary["average_retention"],
            "limit": round(settings.min_retention_floor * 100, 1),
            "unit": "%",
        },
        {
            "id": "protected_programs",
            "label_en": "Protected programme ad load",
            "label_he": "עומס פרסום בתוכן מוגן",
            "status": "compliant"
            if protected_minutes <= settings.protected_program_max_ad_minutes_per_hour
            else "at_risk",
            "observed": protected_minutes,
            "limit": settings.protected_program_max_ad_minutes_per_hour,
            "unit": "minutes/hour",
        },
    ]

    return {
        "profile": settings.profile_name,
        "effective_date": settings.effective_date,
        "source_url": settings.regulatory_source_url,
        "checks": checks,
        "violations": [],
        "status": "at_risk" if any(check["status"] == "at_risk" for check in checks) else "compliant",
        "disclaimer": settings.notes,
    }


@lru_cache(maxsize=16)
def _overview_cached(signature: tuple[tuple[str, int, int], ...], scope: str | None = None) -> dict[str, Any]:
    del signature
    schedule = _load_break_schedule()
    programmes = _load_programmes()
    spots = _load_spots()
    summary = _summarize_schedule(schedule)
    settings = _load_settings()
    break_operations = _build_break_operations(programmes, schedule)
    return {
        "brand": "Kairos",
        "workspace": "KAI Network",
        "data_freshness": datetime.fromtimestamp(
            max(
                [
                    path.stat().st_mtime
                    for path in [
                        OUTPUT_DIR / "weekly_break_schedule.csv",
                        DATA_DIR / "reference" / "Programmes.xlsx",
                        DATA_DIR / "reference" / "Spots.xlsx",
                        DATA_DIR / "Programmes.csv",
                        DATA_DIR / "Spots.csv",
                    ]
                    if path.exists()
                ]
                or [time.time()]
            ),
            tz=timezone.utc,
        ).isoformat(),
        "summary": summary,
        "source_counts": {
            "programmes": int(len(programmes)),
            "spots": int(len(spots)),
            "planned_break_rows": int(len(schedule)),
        },
        "recommendations": _build_recommendations(schedule),
        "frontier_scope": scope or None,
        "settings": _model_dump(settings),
        "compliance": _build_compliance(schedule, settings, break_operations),
    }


@lru_cache(maxsize=16)
def _schedule_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    programmes = _load_programmes()
    break_schedule = _load_break_schedule()
    return {
        "rows": _build_schedule_canvas(programmes, break_schedule),
        "break_operations": _build_break_operations(programmes, break_schedule),
        "break_schedule": break_schedule.head(200).replace({pd.NA: None}).where(pd.notna(break_schedule.head(200)), None).to_dict("records"),
    }


@lru_cache(maxsize=16)
def _schedule_segments_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_schedule_segments(_load_break_schedule(), _load_settings())


@lru_cache(maxsize=16)
def _break_operations_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_break_operations(_load_programmes(), _load_break_schedule())


def _decision_log() -> list[dict[str, Any]]:
    """The operator's decision log, read from the REAL override store.

    Retires the old data/kairos_decisions.json, which was written on every
    approve/reject but read only for display, so the log and the plan could drift.
    Now every row here is a persisted Override that actually resolves through the
    decision plane: an approved recommendation (source=recommendation) or a dismissed
    rejection (status=dismissed, recorded but never applied).
    """
    from kairos.optimize.overrides import (
        OverrideSet,
        SOURCE_RECOMMENDATION,
        STATUS_DISMISSED,
    )

    records: list[dict[str, Any]] = []
    for override in OverrideSet.from_csv().overrides:
        if override.source != SOURCE_RECOMMENDATION and override.status != STATUS_DISMISSED:
            continue
        records.append({
            "id": override.override_id,
            "action": "reject" if override.status == STATUS_DISMISSED else "approve",
            "recommendation_id": override.rec_id or None,
            "break_id": override.target_id,
            "kind": override.kind,
            "value": override.value,
            "status": override.status,
            "note": override.notes,
            "created_at": override.created_at,
            "source": override.source,
            "anchor": {
                "date": override.anchor_date,
                "start_clock": override.anchor_start,
                "program": override.anchor_title,
            },
        })
    records.sort(key=lambda record: str(record.get("created_at") or ""), reverse=True)
    return records


def _resolve_decision(request: BreakDecisionRequest) -> dict[str, Any]:
    """Turn an approve/reject decision into a REAL override (no dead log write).

    Approve creates an active segment override stamped source=recommendation with the
    rec_id and the semantic anchor, so the anchor guard protects it on re-ingest.
    Reject creates a dismissed record (forbid by default) that the plan never applies
    because only active overrides bend the schedule. The console can equivalently POST
    /api/overrides directly; this shortcut just routes through the same honest store.
    """
    from kairos.optimize.overrides import (
        FORBID,
        SOURCE_RECOMMENDATION,
        STATUS_ACTIVE,
        STATUS_DISMISSED,
    )
    from kairos_api.overrides import OverrideCreate, create_override

    target = str(request.target_id or request.break_id or "").strip()
    if not target:
        raise HTTPException(
            status_code=400,
            detail="a target segment_id (target_id or break_id) is required to resolve a decision into an override",
        )
    reject = request.action == "reject"
    kind = str(request.kind or "").strip().lower() or (FORBID if reject else "")
    if not kind:
        raise HTTPException(
            status_code=400,
            detail="kind is required to approve a decision (pin, force, forbid, or gold)",
        )
    payload = OverrideCreate(
        scope="segment",
        target_id=target,
        kind=kind,
        value=str(request.value or ""),
        gold=bool(request.gold),
        notes=str(request.note or ""),
        source=SOURCE_RECOMMENDATION,
        rec_id=str(request.recommendation_id or "").strip(),
        status=STATUS_DISMISSED if reject else STATUS_ACTIVE,
        anchor_date=str(request.anchor_date or "").strip(),
        anchor_start=str(request.anchor_start or "").strip(),
        anchor_title=str(request.anchor_title or "").strip(),
    )
    return create_override(payload)


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


@router.get("/api/compliance")
def compliance() -> dict[str, Any]:
    return _build_compliance(_load_break_schedule(), _load_settings())


@router.get("/api/overview")
def overview(
    scope: str | None = Query(
        default=None,
        description="Optional frontier scope: 'channel:<id>' or 'day:<date>'. Only the owned channel is selectable. Omit for the whole-default frontier.",
    ),
) -> dict[str, Any]:
    body = dict(_overview_cached(
        _signature([
            OUTPUT_DIR / "weekly_break_schedule.csv",
            DATA_DIR / "reference" / "Programmes.xlsx",
            DATA_DIR / "reference" / "Spots.xlsx",
            DATA_DIR / "Programmes.csv",
            DATA_DIR / "Spots.csv",
            SETTINGS_PATH,
        ]),
        scope or None,
    ))
    # The frontier is a slow optimizer sweep, computed in the background so the
    # overview never blocks on it. Merge its current state into the response.
    overview_settings = _load_settings()
    points, net_bundle, status = _frontier_state(overview_settings, scope or None)
    body["frontier"] = points
    body["frontier_status"] = status
    # Additive: the single net-focused scenario computed beside the sweep (the
    # same runner and saved guardrails under objective_mode='revenue_net'),
    # carrying the frontier-point fields plus its label id. Null while the
    # background sweep is computing or when the run failed; never fabricated.
    body["frontier_net_point"] = (net_bundle or {}).get("net_point") if status == "ready" else None
    # Honest disclosure of what the frontier curve actually measures. Each point is
    # a single representative-day REFINED optimum (run_scenario refine=True) on the
    # owned channel, swept across the retention floor at the saved revenue weight,
    # not the whole-week plan behind the projected_revenue headline it sits next to.
    # Surfaced as structured metadata so the dashboard can label the curve and the
    # operator never reads a one-day estimate as the saved weekly total.
    owned_channel = str(overview_settings.operator_channel or "").strip()
    body["frontier_basis"] = {
        "scope": "representative_day",
        "channel": owned_channel or None,
        "method": "refined_floor_sweep",
        "swept": "retention_floor",
        "disclosure": "This frontier sweeps the retention floor at your saved revenue weight, each point a refined single representative-day optimum for the owned channel, not the saved weekly plan total.",
    }
    # Schedule freshness: is the saved schedule the dashboard renders still in
    # step with its inputs? Computed FRESH here (never inside _overview_cached),
    # because a settings/constraints/pricing edit does not clear the overview
    # cache, so a cached verdict would lie. Honest fresh/stale/unknown; never
    # fabricated. Guarded so a freshness failure never breaks the overview.
    try:
        from kairos.export.schedule_freshness import schedule_freshness

        body["schedule_freshness"] = schedule_freshness(ROOT)
    except Exception:  # pragma: no cover - defensive, never blocks the overview
        body["schedule_freshness"] = {"status": "unknown", "computed_at": None, "changed": []}
    return body


@router.get("/api/schedule")
def schedule() -> dict[str, Any]:
    return _schedule_cached(_signature([
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
    ]))


@router.get("/api/schedule/segments")
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


@router.get("/api/schedule/segment/{segment_id:path}")
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


@router.get("/api/break-operations")
def break_operations() -> dict[str, Any]:
    return _break_operations_cached(_signature([
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
    ]))


@router.get("/api/break-decisions")
def break_decisions() -> dict[str, Any]:
    # Display is driven by the real override store, not a parallel decision-log file.
    return {"decisions": _decision_log()}


@router.post("/api/break-decisions")
def create_break_decision(request: BreakDecisionRequest) -> dict[str, Any]:
    # Approve/Reject shortcut: persists a real Override (source=recommendation, rec_id,
    # anchor) rather than a display-only log entry.
    return {"decision": _resolve_decision(request)}
