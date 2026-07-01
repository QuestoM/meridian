"""Recompute endpoints: the synchronous rebuild and the async job wrapper.

One shared body (_run_recompute) feeds both the compat synchronous endpoint
and the background job, so the two paths cannot drift. Scoped jobs pass the
{channel, day} pairs through to the engine's incremental only_days path.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from kairos_api.core import (
    _ENGINE_AVAILABLE,
    _load_settings,
    _model_dump,
    _read_csv_cached,
    _reference_today,
    build_weekly_schedule,
    write_weekly_schedule,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["recompute"])


def _run_recompute(
    only_days: list[tuple[str, str]] | None = None,
    progress_cb: Any = None,
) -> dict[str, Any]:
    """Shared recompute body: rebuild the weekly CSV from the saved settings.

    Used by both the synchronous endpoint and the async job so the two paths
    cannot drift. Raises ValueError on an empty result so callers surface it
    honestly in their own error channel.
    """
    saved = _load_settings()
    settings_map = _model_dump(saved)
    frame = build_weekly_schedule(
        settings=settings_map,
        revenue_weight=saved.revenue_weight / 100.0,
        risk_lambda=saved.risk_lambda,
        operator_channel=saved.operator_channel,
        today=_reference_today(saved),
        only_days=only_days,
        progress_cb=progress_cb,
    )
    if frame.empty:
        raise ValueError("No segments produced (is data/reference/Programmes.xlsx present?)")
    path = write_weekly_schedule(frame=frame)
    # The cached reader keys on mtime+size, but clear it so the next GET is fresh.
    _read_csv_cached.cache_clear()
    # An incremental merge returns the frame in CSV text space (str dtype) to
    # guarantee byte-identical writes, so aggregate through to_numeric.
    return {
        "ok": True,
        "path": str(path),
        "rows": int(len(frame)),
        "channels": int(frame["channel"].nunique()),
        "days": int(frame["date"].nunique()),
        "total_breaks": int(pd.to_numeric(frame["num_breaks"], errors="coerce").fillna(0).sum()),
        "total_revenue": round(float(pd.to_numeric(frame["predicted_revenue"], errors="coerce").fillna(0).sum()), 2),
        "revenue_weight": saved.revenue_weight,
        "risk_lambda": saved.risk_lambda,
        "scope": [list(pair) for pair in only_days] if only_days else "full",
    }


@router.post("/api/recompute-schedule")
def recompute_schedule() -> dict[str, Any]:
    """Rebuild ``output/weekly_break_schedule.csv`` from the saved settings.

    This is the button that makes the operator's controls real: it runs the engine
    across every channel-day with the saved revenue_weight, risk_lambda, retention
    floor and guardrails, then overwrites the CSV the dashboard's main screens read.
    Without it, changing a setting only affected the live simulation, not the saved
    schedule. Returns a summary so the operator sees the recompute landed.
    """
    if not _ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Optimization engine is not available")
    try:
        return _run_recompute()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - surfaced honestly to the operator
        logger.exception("recompute-schedule failed")
        raise HTTPException(status_code=500, detail=f"recompute failed: {exc}") from exc


class RecomputeJobRequest(BaseModel):
    """Optional scope for an async recompute: a list of {channel, day} pairs."""

    scope: list[dict[str, str]] | None = None


@router.post("/api/jobs/recompute")
def start_recompute_job(request: RecomputeJobRequest | None = None) -> dict[str, Any]:
    """Run the recompute as a background job with honest tri-state status.

    With a scope, only the listed channel-days are re-optimized and merged into
    the saved CSV (the engine falls back to a full rebuild when the saved file
    cannot support an incremental merge). Without a scope, the whole week is
    rebuilt. If a recompute job is already running, its id is returned instead
    of stacking a second rebuild.
    """
    if not _ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Optimization engine is not available")
    from kairos_api import jobs as _jobs

    existing = _jobs.running_job("recompute")
    if existing is not None:
        return {"job_id": existing, "already_running": True}

    only_days: list[tuple[str, str]] | None = None
    if request is not None and request.scope:
        pairs: list[tuple[str, str]] = []
        for item in request.scope:
            channel = str(item.get("channel", "")).strip()
            day = str(item.get("day", "")).strip()
            if not channel or not day:
                raise HTTPException(status_code=422, detail="scope entries need channel and day")
            pairs.append((channel, day))
        only_days = pairs

    holder: dict[str, str] = {}

    def _progress(done: int, total: int) -> None:
        job_id = holder.get("id")
        if job_id:
            _jobs.report_progress(job_id, done, total)

    holder["id"] = _jobs.submit(
        "recompute", _run_recompute, only_days=only_days, progress_cb=_progress
    )
    return {"job_id": holder["id"], "already_running": False}


@router.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    """Honest job record: running, done with a result, or failed with the error."""
    from kairos_api import jobs as _jobs

    record = _jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return record


