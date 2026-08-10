"""The shared read layer over the saved plan: programme joins and the break board.

Frozen. Every piece reads this module and no piece writes it. The functions moved
here verbatim from dashboard_api.py because more than one owner reads them: the
week payload embeds the break board under ``break_operations`` and the day board
serves it directly, so neither surface can own the builder.

The leading underscore was dropped on the move, because a module-private name
imported across modules is what created the ambiguity in the first place. The old
names keep resolving from :mod:`kairos_api.dashboard_api` and
:mod:`kairos_api.server`, against these same objects, so no existing import
changes. Signatures are published and frozen in
``docs/ux-gauntlet/contracts/W0-1.md``.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd

from kairos_api.core import (
    _ENGINE_AVAILABLE,
    _load_settings,
    _money,
    _percent,
    _safe_number,
)

logger = logging.getLogger(__name__)


def program_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
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


def plan_by_program_key(schedule: pd.DataFrame) -> dict[tuple[str, str, str], dict[str, Any]]:
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


def build_break_operations(programmes: pd.DataFrame, schedule: pd.DataFrame) -> dict[str, Any]:
    if programmes.empty:
        return {"programs": [], "breaks": [], "summary": {"programs": 0, "breaks": 0, "ad_seconds": 0, "revenue": 0}}

    frame = program_datetime_columns(programmes)
    frame = frame.dropna(subset=["start_dt", "end_dt"]).copy()
    if frame.empty:
        return {"programs": [], "breaks": [], "summary": {"programs": 0, "breaks": 0, "ad_seconds": 0, "revenue": 0}}

    frame["program_type"] = (
        frame["program_type"] if "program_type" in frame.columns
        else frame["programme_type"] if "programme_type" in frame.columns
        else pd.Series("Other", index=frame.index)
    ).fillna("Other").astype(str)
    # Honest ratings: no fillna(1.0). A programme without a measured TVR keeps
    # NaN here and the payload reports null, never a fabricated 1.0 rating point.
    tvr_column = frame["TVR"] if "TVR" in frame.columns else pd.Series(float("nan"), index=frame.index)
    frame["viewing_points"] = pd.to_numeric(tvr_column, errors="coerce")
    frame["day_key"] = frame["start_dt"].dt.strftime("%a")
    frame["duration_seconds"] = (frame["end_dt"] - frame["start_dt"]).dt.total_seconds().clip(lower=0)
    frame = frame.sort_values("start_dt").groupby("Channel", dropna=False).head(12).reset_index(drop=True)

    plan_index = plan_by_program_key(schedule)
    settings = _load_settings()
    # Display-only premium provenance. Sourced through pricing_from_settings so
    # the label matches what the engine actually priced with (the operator's
    # saved rate-card edits included), never a bare from_yaml read that ignores
    # them. It is NEVER multiplied into the break money below: the plan's
    # predicted_revenue already carries every premium.
    _pricing_model: Any = None
    if _ENGINE_AVAILABLE:
        try:
            from kairos.optimize.pricing import pricing_from_settings

            _pricing_model = pricing_from_settings(settings)
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
        broadcast_midnight = row["start_dt"].normalize()
        program_start_seconds = int((row["start_dt"] - broadcast_midnight).total_seconds())
        program_end_seconds = int((row["end_dt"] - broadcast_midnight).total_seconds())
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
                "start_seconds": program_start_seconds,
                "end_seconds": program_end_seconds,
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
            # Plan-derived money: each displayed break carries an equal split of
            # its programme's committed predicted_revenue, and the LAST break
            # absorbs the rounding remainder so the programme's breaks sum back
            # to the plan figure to the cent. Nothing is re-derived: the old
            # path re-priced breaks from the per-second base_rate fed into
            # 30-second CPP units with the programme premium applied a second
            # time (base_rate already contains it), understating the board
            # roughly 56x against the committed plan.
            reference_revenue = _money(revenue_for_breaks / max(break_count, 1))
            if break_index == break_count:
                break_revenue = _money(revenue_for_breaks - reference_revenue * (break_count - 1))
            else:
                break_revenue = reference_revenue
            # base_rate comes from the optimizer's weekly schedule CSV. Absent
            # means no plan was run; report None rather than inventing 1000.
            raw_base_rate = schedule_row.get("base_rate")
            cpp: float | None = _safe_number(raw_base_rate, -1.0) if raw_base_rate is not None else None
            if cpp is not None and cpp < 0:
                cpp = None
            # Premium from the settings-aware rate card, display-only (see the
            # pricing_from_settings note above); never hardcode 1.25 for gold.
            program_premium = _pricing_model.program_premium(program_type) if _pricing_model is not None else 1.0
            # The displayed rating prefers the plan's own baseline_tvr (the
            # basis predicted_revenue was priced on); the EPG TVR is the backup
            # and a missing value stays null instead of a fabricated 1.0.
            rating_value = _safe_number(schedule_row.get("baseline_tvr"), float("nan")) if has_plan else float("nan")
            if math.isnan(rating_value):
                rating_value = _safe_number(row.get("viewing_points"), float("nan"))
            rating_predicted = None if math.isnan(rating_value) else round(rating_value, 2)
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
                    # Seconds from this programme's broadcast-day midnight, not
                    # clock seconds. A 00:10 break inside a 23:30 programme is
                    # therefore 87,000, preserving its next-day position.
                    "start_seconds": int((candidate - broadcast_midnight).total_seconds()),
                    "duration_sec": break_seconds,
                    "sponsorships_count": 1 if is_gold else 0,
                    "is_gold": is_gold,
                    "source": "Model",
                    "rating_predicted": rating_predicted,
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
