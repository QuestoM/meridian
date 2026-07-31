"""Plan, week: the schedule canvas and the sellable supply beside it.

The week reads, moved verbatim from dashboard_api.py (the canvas) and
catalog_api.py (the inventory) as part of the wave-zero router split. Behaviour
is unchanged, including the cache keys, which carry the settings file and the
rate card because both payloads read them.

The break board the canvas payload embeds under ``break_operations`` is served
from the frozen plan-read layer, because the day board serves the same builder.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import APIRouter

from kairos_api import plan_read
from kairos_api.core import (
    DATA_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    _load_break_schedule,
    _load_programmes,
    _load_settings,
    _load_spots,
    _money,
    _percent,
    _records,
    _safe_number,
    _series,
    _signature,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_schedule_canvas(programmes: pd.DataFrame, schedule: pd.DataFrame) -> list[dict[str, Any]]:
    if programmes.empty:
        return []

    frame = plan_read.program_datetime_columns(programmes)
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
    plan_index = plan_read.plan_by_program_key(schedule)

    # The grid is channel (rows) by weekday (columns), so it needs programmes from
    # every weekday, not just the earliest. A former ``head(18)`` took the first 18
    # chronological programmes per channel, which all fell on the single earliest
    # date, leaving six of the seven weekday columns permanently empty. Instead show
    # one representative week shared by every channel: the programmes on the first
    # seven distinct broadcast dates in the data (a consecutive Fri-to-Thu span here),
    # so each weekday column populates and the channels stay comparable on the same
    # week. Each cell aggregates its channel-day, so no per-day cap is applied.
    week_dates = frame["start_dt"].dt.normalize().drop_duplicates().nsmallest(7)
    week_frame = frame[frame["start_dt"].dt.normalize().isin(week_dates)]

    rows: list[dict[str, Any]] = []
    for channel, channel_df in week_frame.sort_values("start_dt").groupby("Channel"):
        programs = []
        for _, row in channel_df.iterrows():
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


@lru_cache(maxsize=16)
def _schedule_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    programmes = _load_programmes()
    break_schedule = _load_break_schedule()
    return {
        "rows": _build_schedule_canvas(programmes, break_schedule),
        "break_operations": plan_read.build_break_operations(programmes, break_schedule),
        "break_schedule": break_schedule.head(200).replace({pd.NA: None}).where(pd.notna(break_schedule.head(200)), None).to_dict("records"),
        # break_schedule is a display slice (first 200 rows); this is the real
        # size of the saved plan so the client can say "200 of N" honestly.
        "break_schedule_total_rows": int(len(break_schedule)),
    }


def _build_inventory(spots: pd.DataFrame) -> dict[str, Any]:
    """Spot inventory for the OPERATOR'S channel only.

    The spots source carries every channel because the retention model needs
    the competitor rows, but inventory is an ownership concept: everything the
    operator can sell sits on their own channel, so a per-channel split of the
    market has no place on this surface. The payload is scoped to
    ``settings.operator_channel`` (whole-frame only when no channel is
    configured yet, disclosed via ``scope_channel``), and the breakdown the
    operator actually plans with is by broadcast daypart, not by channel.
    """
    settings = _load_settings()
    owned = str(settings.operator_channel or "").strip()
    scope_channel: str | None = None
    if not spots.empty and owned and "Channel" in spots.columns:
        scoped = spots[spots["Channel"].astype(str).str.strip() == owned]
        spots = scoped
        scope_channel = owned

    if spots.empty:
        return {
            "summary": {"spots": 0, "revenue": None, "seconds": 0},
            "revenue_available": False,
            "scope_channel": scope_channel,
            "by_daypart": [],
            "by_hour": [],
        }

    frame = spots.copy()
    # Revenue is reported only when the spots source actually carries it. The
    # reference airings export has no revenue column, so fabricating a zero
    # would misstate a real quantity; report an honest unavailable instead.
    has_revenue = "revenue_ils" in frame.columns
    frame["revenue_ils"] = pd.to_numeric(_series(frame, "revenue_ils", 0), errors="coerce").fillna(0)
    frame["Duration"] = pd.to_numeric(_series(frame, "Duration", 0), errors="coerce").fillna(0)
    if "hour_of_day" not in frame.columns and "Start time" in frame.columns:
        # The reference airings export carries the real airing time but no
        # precomputed hour; derive it rather than collapse every spot into a
        # fabricated midnight bucket.
        frame["hour_of_day"] = frame["Start time"].map(
            lambda value: getattr(value, "hour", None)
        )
    frame["hour_of_day"] = pd.to_numeric(_series(frame, "hour_of_day", -1), errors="coerce").fillna(-1).astype(int)
    valid_hours = frame[(frame["hour_of_day"] >= 0) & (frame["hour_of_day"] <= 23)]

    # Broadcast-daypart breakdown on the engine's own taxonomy; hours without a
    # parseable airing time land in an honest "unclassified" bucket rather than
    # being silently dropped.
    try:
        from kairos.data.dayparts import daypart_for_hour
    except Exception:  # pragma: no cover - taxonomy optional
        def daypart_for_hour(_hour: int) -> str:  # type: ignore[misc]
            return "unclassified"
    frame["daypart"] = frame["hour_of_day"].map(
        lambda hour: (daypart_for_hour(int(hour)) or "unclassified") if 0 <= int(hour) <= 23 else "unclassified"
    )
    by_daypart = (
        frame.groupby("daypart", dropna=False)
        .agg(spots=("Campaign", "count"), seconds=("Duration", "sum"), revenue=("revenue_ils", "sum"))
        .reset_index()
        .sort_values("seconds", ascending=False)
    )
    by_hour = (
        valid_hours.groupby("hour_of_day", dropna=False)
        .agg(spots=("Campaign", "count"), seconds=("Duration", "sum"), revenue=("revenue_ils", "sum"))
        .reset_index()
        .sort_values("hour_of_day")
    )
    if not has_revenue:
        by_daypart["revenue"] = None
        by_hour["revenue"] = None

    return {
        "summary": {
            "spots": int(len(frame)),
            "revenue": _money(frame["revenue_ils"].sum()) if has_revenue else None,
            "seconds": int(frame["Duration"].sum()),
        },
        "revenue_available": has_revenue,
        "scope_channel": scope_channel,
        "by_daypart": _records(by_daypart, 8),
        "by_hour": _records(by_hour, 24),
    }


@lru_cache(maxsize=16)
def _inventory_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_inventory(_load_spots())


@router.get("/api/schedule", tags=["dashboard"])
def schedule() -> dict[str, Any]:
    # SETTINGS_PATH and the pricing YAML are part of the key because the break
    # board inside this payload reads the operator settings (retention floor,
    # pricing overrides) and the rate card; without them a settings or rate-card
    # edit kept serving the stale cached board.
    return _schedule_cached(_signature([
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        SETTINGS_PATH,
        ROOT / "config" / "optimization_weights.yaml",
    ]))


@router.get("/api/inventory", tags=["catalog"])
def inventory() -> dict[str, Any]:
    # _load_spots prefers the reference workbook, so the workbook belongs in the
    # cache key; keying on the legacy CSV alone kept serving a stale inventory
    # after a reference re-ingest. The settings file is in the key because the
    # payload is scoped to the operator's channel.
    return _inventory_cached(_signature([
        DATA_DIR / "reference" / "Spots.xlsx",
        DATA_DIR / "Spots.csv",
        SETTINGS_PATH,
    ]))
