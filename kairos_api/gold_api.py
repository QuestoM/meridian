"""Plan, break zoom: ברייקי זהב in the operator's current plan.

Moved verbatim from insights_api.py as part of the wave-zero router split. Gold
status is read from the saved weekly plan's own ``is_gold`` column, so this route
never re-runs the optimizer to recover it, and the premium figures that live only
on the daily spot-pricing path stay null with their source marker rather than
being invented. The route does not mutate state.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()


def _server() -> Any:
    """Lazy handle to server.py helpers (avoids an import cycle at module load)."""
    from kairos_api import server

    return server


def _is_gold_truthy(value: Any) -> bool:
    """Whether a CSV ``is_gold`` cell marks the segment gold.

    Robust to how the round-tripped column typed itself: a native pandas bool, or
    the literal 'True'/'False' text a re-read CSV can carry. Anything else (blank,
    NaN, 'False') is honestly not gold.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "1.0"}


def _cell_or_none(value: Any) -> Optional[str]:
    """A schedule cell as a clean string, or None for blank/NaN, never 'nan'."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    return text


def _build_gold_breaks() -> dict[str, Any]:
    """Gold breaks in the operator's current plan, read from the saved schedule CSV.

    Gold status is now materialized on ``output/weekly_break_schedule.csv`` as the
    per-segment ``is_gold`` column. The export sets it from each plan's placements
    (``is_gold = any(placement.is_gold)`` for the segment), so reading the column
    here is the faithful, cheap source: this endpoint no longer re-runs the whole
    optimizer just to recover gold status. Each gold row is one gold segment,
    carrying its own ``predicted_revenue`` and first-break ``start_time`` from the
    plan. ``realized_premium``/``potential_premium`` are advertiser-attribution
    figures that live only on the daily spot-pricing path, so they stay null with a
    ``source_pending`` marker, never invented.
    """
    server = _server()
    settings = server._load_settings()
    if not settings.sponsorships_enabled:
        return {"available": True, "enabled": False, "reason": "Sponsorships are disabled in settings.", "count": 0, "breaks": [], "by_day": []}
    if not settings.gold_breaks_enabled:
        return {"available": True, "enabled": False, "reason": "Gold breaks are disabled in settings.", "count": 0, "breaks": [], "by_day": []}

    schedule = server._load_break_schedule()
    if schedule.empty:
        return {"available": False, "reason": "No saved weekly schedule on disk.", "count": 0, "breaks": [], "by_day": []}
    if "is_gold" not in schedule.columns:
        return {
            "available": True,
            "enabled": True,
            "count": 0,
            "reason": "Saved weekly schedule predates gold-break tracking; recompute the schedule to populate is_gold.",
            "max_per_day": settings.gold_breaks_max_per_day,
            "breaks": [],
            "by_day": [],
        }

    # Competitor boundary: gold is quoted only for the operator's own channel,
    # like every other money surface. A competitor row with is_gold set must
    # never surface its channel name or figures here.
    owned = str(settings.operator_channel or "").strip()
    if owned and "channel" in schedule.columns:
        schedule = schedule[schedule["channel"].astype(str).str.strip() == owned]
    gold = schedule[schedule["is_gold"].map(_is_gold_truthy)]
    if gold.empty:
        return {
            "available": True,
            "enabled": True,
            "count": 0,
            "reason": "No gold breaks in the current plan (none configured as gold in overrides).",
            "max_per_day": settings.gold_breaks_max_per_day,
            "breaks": [],
            "by_day": [],
        }

    revenue = pd.to_numeric(gold.get("predicted_revenue", 0), errors="coerce").fillna(0.0)
    duration = pd.to_numeric(gold.get("break_length", 0), errors="coerce").fillna(0.0)
    breaks: list[dict[str, Any]] = []
    by_day_counts: dict[str, int] = {}
    for (_, row), rev, dur in zip(gold.iterrows(), revenue, duration):
        day = _cell_or_none(row.get("date")) or ""
        by_day_counts[day] = by_day_counts.get(day, 0) + 1
        breaks.append(
            {
                "segment_id": _cell_or_none(row.get("segment_id")),
                "channel": _cell_or_none(row.get("channel")),
                "day": day,
                "start_time": _cell_or_none(row.get("start_time")),
                "program_type": _cell_or_none(row.get("program_type")),
                "duration_seconds": round(float(dur), 1),
                "revenue": round(float(rev), 2),
                "realized_premium": None,
                "potential_premium": None,
                "premium_source": "source_pending",
                "premium_note": "Gold-break premium is realized on the daily spot-pricing path, not the weekly optimizer.",
            }
        )
    return {
        "available": True,
        "enabled": True,
        "count": len(breaks),
        "max_per_day": settings.gold_breaks_max_per_day,
        "breaks": breaks,
        "by_day": [{"day": day, "count": count} for day, count in sorted(by_day_counts.items())],
    }


@router.get("/api/gold-breaks", tags=["insights"])
def gold_breaks() -> dict[str, Any]:
    return _build_gold_breaks()
