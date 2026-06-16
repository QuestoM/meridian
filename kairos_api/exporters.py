"""CSV export endpoints for the Kairos optimized weekly plan.

The export streams the real optimized schedule. It prefers the materialized
output file and falls back to the schedule the existing service builds, so the
download always reflects genuine planner output and never fabricated rows.
"""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"

router = APIRouter(prefix="/api/export", tags=["export"])

EXPORT_FILENAME = "kairos-weekly-schedule.csv"


def _load_plan() -> pd.DataFrame:
    """Load the real optimized weekly plan, preferring the materialized file."""
    materialized = OUTPUT_DIR / "weekly_break_schedule.csv"
    if materialized.exists():
        frame = pd.read_csv(materialized, encoding="utf-8-sig")
        if not frame.empty:
            return frame

    # Fall back to whatever the existing schedule service builds. Imported
    # lazily to avoid a circular import at module load time.
    from kairos_api.server import _load_break_schedule

    return _load_break_schedule()


@router.get("/schedule.csv")
def export_schedule_csv() -> StreamingResponse:
    """Stream the optimized weekly plan as a downloadable CSV."""
    frame = _load_plan()
    if frame is None or frame.empty:
        raise HTTPException(
            status_code=404,
            detail="No optimized weekly plan is available to export. Run the optimizer first.",
        )

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)

    headers = {"Content-Disposition": f'attachment; filename="{EXPORT_FILENAME}"'}
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers=headers)
