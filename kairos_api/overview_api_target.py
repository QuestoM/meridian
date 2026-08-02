"""The plan target's three routes: read it, set it, remove it.

Split out of :mod:`kairos_api.overview_api` under the file-size law. The routes
live on their own router which the parent includes, so the mounted surface is
unchanged: one router in ``server.py``, the same three paths, the same tag.

The only rule this module adds to the store behind it is that a window is never
guessed. A caller that names no window is asking about the one the saved plan
currently covers, and that is resolved from the plan itself rather than from
today's date, because the plan on disk and the calendar do not overlap and
answering about a window the plan says nothing about would be answering with a
figure of zero.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

from kairos_api import overview_api_today, target_store
from kairos_api.core import _load_break_schedule, _load_settings, _summarize_schedule

router = APIRouter()


class PlanTargetRequest(BaseModel):
    """The two numbers a person supplies, plus the window they are about."""

    amount_ils: float = Field(..., description="The target amount in shekels for this window.")
    at_risk_band_percent: float = Field(..., description="How far below the target still reads as at risk, in percent.")
    channel: Optional[str] = Field(default=None, description="Defaults to the operator's own channel.")
    period_start: Optional[str] = Field(default=None, description="ISO date. Defaults to the current plan window.")
    period_end: Optional[str] = Field(default=None, description="ISO date. Defaults to the current plan window.")
    note: str = Field(default="", description="Optional free text recorded with the target.")


def resolved_window(period_start: Optional[str], period_end: Optional[str]) -> tuple[str, str]:
    """The window a request is about: the one it named, or the plan's own."""
    if period_start and period_end:
        return str(period_start), str(period_end)
    window = overview_api_today.window_from_summary(_summarize_schedule(_load_break_schedule()))
    return window["date_from"] or "", window["date_to"] or ""


def _owned(channel: Optional[str]) -> str:
    return str(channel or _load_settings().operator_channel or "").strip()


@router.get("/api/plan-target", tags=["dashboard"])
def plan_target(
    request: Request,
    channel: str | None = Query(default=None, description="Defaults to the operator's own channel."),
    period_start: str | None = Query(default=None, description="ISO date. Defaults to the current plan window."),
    period_end: str | None = Query(default=None, description="ISO date. Defaults to the current plan window."),
) -> dict[str, Any]:
    """The target for one window, or the honest unset state with can_edit on it."""
    start, end = resolved_window(period_start, period_end)
    return target_store.payload(_owned(channel), start, end, request)


@router.put("/api/plan-target", tags=["dashboard"])
@target_store.TARGET_WALL.guard()
def set_plan_target(payload: PlanTargetRequest, request: Request) -> dict[str, Any]:
    """Set the target for one window. The number comes from the person, always."""
    owned = _owned(payload.channel)
    start, end = resolved_window(payload.period_start, payload.period_end)
    target_store.save_target(
        channel=owned,
        period_start=start,
        period_end=end,
        amount_ils=payload.amount_ils,
        at_risk_band_percent=payload.at_risk_band_percent,
        note=payload.note,
        request=request,
    )
    return target_store.payload(owned, start, end, request)


@router.delete("/api/plan-target", tags=["dashboard"])
@target_store.TARGET_WALL.guard()
def clear_plan_target(
    request: Request,
    channel: str | None = Query(default=None, description="Defaults to the operator's own channel."),
    period_start: str | None = Query(default=None, description="ISO date. Defaults to the current plan window."),
    period_end: str | None = Query(default=None, description="ISO date. Defaults to the current plan window."),
) -> dict[str, Any]:
    """Remove one window's target. The verdict goes back to unavailable, never to zero."""
    owned = _owned(channel)
    start, end = resolved_window(period_start, period_end)
    target_store.delete_target(owned, start, end, request)
    return target_store.payload(owned, start, end, request)
