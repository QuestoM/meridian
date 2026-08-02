"""Plan, week: the goal and the progress against it, read together.

Split out of ``week_api`` under the 450-line law and named by the helper rule.
It answers one question for one window: what the plan projects, what somebody
said it should be, and the distance between the two under a threshold that was
published rather than chosen here.

Three rules hold it, and each one is why a figure on this surface can be
trusted.

- **The window is the plan's own.** It comes from the saved plan's planning-week
  slice (:func:`kairos_api.core._summarize_schedule`), which is the same slice
  the Today workspace resolves a target against, so the two surfaces cannot read
  a different week and disagree about whether a target exists. The window is
  printed with every figure, never in a tooltip.
- **The projection is the operator's.** The saved plan carries every channel
  because the retention model is measured against the competitive lineup, so the
  frame is scoped to ``settings.operator_channel`` before anything is summed and
  the scope travels with the payload.
- **The target is somebody's number, never this module's.** There is no plan
  target anywhere in the data and none is derived here: a plan compared against
  itself is always exactly on plan. The store is read through its own accessors
  and the three-state verdict is computed by the store's own rule, so there is
  one implementation of the threshold in the process rather than one per screen.

When no target has been supplied, the verdict is ``unavailable`` with the reason
``no_target``, no variance is reported and no figure is invented. That is the
state the owner could not unblock, recorded honestly, with the path to supply it
named in the payload.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, Query, Request

from kairos_api import channel_scope, target_store
from kairos_api.core import (
    _load_break_schedule,
    _load_settings,
    _summarize_schedule,
    _window_metrics,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["plan-progress"])

# Where a person supplies the number, named so the empty state is a route and
# not a sentence. The control itself is the Today workspace's, and this is the
# door id the frozen session map already carries for it.
TARGET_DOOR = "today"
TARGET_ROUTE = "PUT /api/plan-target"


def _iso(value: Any) -> Optional[str]:
    text = str(value if value is not None else "").strip()[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _metrics_block(metrics: dict[str, Any]) -> dict[str, Any]:
    """The four figures a planner reads, taken from the engine's own window sum."""
    return {
        "revenue": metrics.get("projected_revenue"),
        "breaks": metrics.get("total_breaks"),
        "ad_seconds": metrics.get("total_ad_seconds"),
        "average_retention": metrics.get("average_retention"),
        "retention_basis": metrics.get("retention_basis"),
    }


def _requested_window(
    schedule: Any, period_start: str, period_end: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Metrics for an explicitly asked-for span, scoped to the owned channel.

    Used only when a caller names a window that is not the plan's own week. The
    frame is scoped through the frozen boundary module and summed with the same
    window function the headline uses, so a custom span is the same arithmetic
    on a different set of dates rather than a second implementation.
    """
    owned, note = channel_scope.scope_frame(schedule)
    if owned is None or len(owned) == 0 or "date" not in getattr(owned, "columns", []):
        window = {
            "date_from": period_start,
            "date_to": period_end,
            "n_dates": 0,
            "basis": "requested",
        }
        return window, {"projected_revenue": None, "total_breaks": 0, "total_ad_seconds": 0}
    text = owned["date"].astype(str).str.strip()
    keep = owned[(text >= period_start) & (text <= period_end)]
    dates = keep["date"].astype(str).str.strip() if len(keep) else None
    window = {
        "date_from": period_start,
        "date_to": period_end,
        "n_dates": int(dates.nunique()) if dates is not None else 0,
        "basis": "requested",
        "scope": note,
    }
    if len(keep) == 0:
        return window, {"projected_revenue": None, "total_breaks": 0, "total_ad_seconds": 0}
    return window, _window_metrics(keep, _load_settings())


def build_progress(
    period_start: Optional[str] = None,
    period_end: Optional[str] = None,
) -> dict[str, Any]:
    """The window, its projection, its target and the verdict between them."""
    settings = _load_settings()
    channel = str(settings.operator_channel or "").strip()
    schedule = _load_break_schedule()
    summary = _summarize_schedule(schedule)
    week = summary.get("week") if isinstance(summary.get("week"), dict) else None

    start = _iso(period_start)
    end = _iso(period_end)
    asked = bool(start and end)
    plan_week_start = _iso(week.get("date_from")) if week else None
    plan_week_end = _iso(week.get("date_to")) if week else None

    if asked and (start, end) != (plan_week_start, plan_week_end):
        window, metrics = _requested_window(schedule, start or "", end or "")
        is_plan_week = False
    elif week:
        window = {
            "date_from": plan_week_start,
            "date_to": plan_week_end,
            "n_dates": int(week.get("n_dates") or 0),
            "basis": str(week.get("basis") or ""),
        }
        metrics = week
        is_plan_week = True
    else:
        window = {
            "date_from": _iso(summary.get("date_from")),
            "date_to": _iso(summary.get("date_to")),
            "n_dates": int(summary.get("n_dates") or 0),
            "basis": "plan_span",
        }
        metrics = summary
        is_plan_week = False

    window["is_plan_week"] = is_plan_week
    projected = _metrics_block(metrics)
    record = None
    if channel and window["date_from"] and window["date_to"]:
        record = target_store.target_for(channel, window["date_from"], window["date_to"])
    verdict = target_store.verdict(projected["revenue"], record)
    others = [row for row in target_store.targets_for_channel(channel) if row != record] if channel else []

    return {
        # The plan has to exist before any of this means anything, and an
        # absent plan is a state rather than a zero.
        "available": bool(window["date_from"] and window["date_to"]),
        "channel": channel or None,
        "metric": target_store.DEFAULT_METRIC,
        "currency": "ILS",
        "window": window,
        "projected": projected,
        # The disclosure that travels with the money: what was summed, on which
        # channel, out of how many the source carries.
        "scope": {
            "scope_channel": summary.get("scope_channel"),
            "n_channels_total": summary.get("n_channels_total"),
            "plan_date_from": summary.get("date_from"),
            "plan_date_to": summary.get("date_to"),
            "plan_n_dates": summary.get("n_dates"),
        },
        "target": {
            "state": "set" if record else "unset",
            "amount_ils": None if record is None else record["amount_ils"],
            "at_risk_band_percent": None if record is None else record["at_risk_band_percent"],
            "set_by": None if record is None else record["set_by"],
            "set_at": None if record is None else record["set_at"],
            "note": None if record is None else record["note"],
        },
        "verdict": verdict,
        # A target on a different span is not this window's, and saying so is
        # what stops two surfaces quietly disagreeing about whether one exists.
        "other_windows": others,
        "supply": {"door": TARGET_DOOR, "route": TARGET_ROUTE},
    }


@router.get("/api/plan-progress")
def plan_progress(
    request: Request,
    period_start: Optional[str] = Query(default=None, description="ISO date. Defaults to the plan's own week."),
    period_end: Optional[str] = Query(default=None, description="ISO date. Defaults to the plan's own week."),
) -> dict[str, Any]:
    """The goal and the progress against it, for the plan's own week."""
    body = build_progress(period_start, period_end)
    return target_store.TARGET_WALL.stamp(body, request)
