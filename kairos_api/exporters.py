"""CSV export endpoints for the Kairos optimized plans.

Two real, read-only exports live here on one router:

  * ``/schedule.csv`` streams the optimized weekly BREAK schedule. It prefers the
    materialized output file and falls back to the schedule the existing service
    builds, so the download always reflects genuine planner output.
  * ``/spots.csv`` streams the per-spot daily pricing ledger produced by
    :func:`kairos.export.spots.price_daily_file`: every priced spot with its
    advertiser-rule premium and revenue, plus every spot dropped by an advertiser
    rule or by a frequency/separation rule, each with its reason. This is the only
    surface that exposes that pipeline; before it, the priced/dropped ledger was
    reachable from tests only.

Neither export fabricates rows. When no data exists the CSV is streamed with its
header row and no data rows, the honest empty answer.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output"

router = APIRouter(prefix="/api/export", tags=["export"])

EXPORT_FILENAME = "kairos-weekly-schedule.csv"
SPOTS_EXPORT_FILENAME = "kairos-daily-spots.csv"

# Column order for the daily spot ledger. One CSV carries priced spots and both
# kinds of dropped spot, distinguished by the leading ``status`` column, so the
# operator sees what was priced, what was dropped, and why, in one download.
SPOTS_COLUMNS = [
    "status",
    "advertiser",
    "campaign",
    "program",
    "position",
    "genre",
    "daypart",
    "duration_seconds",
    "planned_tvr",
    "pricing_type",
    "premium",
    "revenue",
    "placement_value",
    "ad",
    "break_id",
    "rule_id",
    "limit_type",
    "reason",
]


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


def _blank(value: Optional[Any]) -> Any:
    """Empty string for a missing value, so a dropped-spot row leaves priced-only
    columns genuinely blank rather than showing a fabricated zero."""
    return "" if value is None else value


def _load_daily_pricing():
    """Run the real per-spot daily pricing pipeline over the newest daily file.

    Loads every input the same way the rest of kairos_api does: the newest daily
    Wally file via the uploads resolver, the operator's saved pricing edits via
    ``pricing_from_settings`` (the same seam the optimizer, forecast and spot
    export share), and the operator's manual overrides from the persisted CSV. The
    advertiser rules, classifier and frequency/separation rules are loaded from
    their shipped config files by :func:`price_daily_file` itself. Returns ``None``
    when there is no daily file to price, so the endpoint can stream an honest,
    header-only CSV instead of inventing rows.
    """
    from kairos_api.uploads import _newest_daily
    from kairos_api.overrides import OVERRIDES_PATH
    from kairos_api.server import _load_settings
    from kairos.optimize.overrides import OverrideSet
    from kairos.optimize.pricing import pricing_from_settings
    from kairos.export.spots import price_daily_file

    path = _newest_daily()
    if path is None:
        return None

    settings = _load_settings()
    pricing = pricing_from_settings(settings)
    overrides = OverrideSet.from_csv(OVERRIDES_PATH)
    return price_daily_file(path, pricing=pricing, overrides=overrides)


def _spot_records(result) -> list[dict[str, Any]]:
    """Flatten one DailyPricingResult into ordered CSV rows, one per spot.

    Priced spots carry their premium, revenue and placement value. Rule-dropped
    and frequency-dropped spots carry only the fields their record actually holds
    (blank elsewhere) plus the verbatim drop reason, so nothing is fabricated and
    no dropped spot is silently lost.
    """
    records: list[dict[str, Any]] = []
    for spot in result.priced:
        records.append({
            "status": "priced",
            "advertiser": spot.advertiser,
            "campaign": spot.campaign,
            "program": spot.program,
            "position": _blank(spot.position),
            "genre": spot.genre,
            "daypart": _blank(spot.daypart),
            "duration_seconds": spot.duration_seconds,
            "planned_tvr": spot.planned_tvr,
            "pricing_type": spot.pricing_type,
            "premium": spot.premium,
            "revenue": spot.revenue,
            "placement_value": spot.placement_value,
            "ad": spot.ad,
            "break_id": spot.break_id,
            "rule_id": "",
            "limit_type": "",
            "reason": "",
        })
    for drop in result.dropped:
        records.append({
            "status": "dropped_rule",
            "advertiser": drop.advertiser,
            "campaign": drop.campaign,
            "program": drop.program,
            "position": _blank(drop.position),
            "genre": drop.genre,
            "daypart": _blank(drop.daypart),
            "duration_seconds": "",
            "planned_tvr": "",
            "pricing_type": "",
            "premium": "",
            "revenue": "",
            "placement_value": "",
            "ad": "",
            "break_id": "",
            "rule_id": "",
            "limit_type": "",
            "reason": drop.reason,
        })
    for drop in result.frequency_dropped:
        records.append({
            "status": "dropped_frequency",
            "advertiser": drop.advertiser,
            "campaign": drop.campaign,
            "program": "",
            "position": "",
            "genre": "",
            "daypart": "",
            "duration_seconds": "",
            "planned_tvr": "",
            "pricing_type": "",
            "premium": "",
            "revenue": "",
            "placement_value": "",
            "ad": drop.ad,
            "break_id": drop.break_id,
            "rule_id": drop.rule_id,
            "limit_type": drop.limit_type,
            "reason": drop.reason,
        })
    return records


@router.get("/spots.csv")
def export_spots_csv() -> StreamingResponse:
    """Stream the per-spot daily pricing ledger (priced plus dropped) as a CSV.

    Every priced spot carries its applied advertiser-rule premium and revenue;
    every dropped spot carries the reason it was dropped (an advertiser rule or a
    frequency/separation rule). With no daily file to price, or an empty one, the
    CSV is streamed with its header row and no data rows, never fabricated rows.
    """
    result = _load_daily_pricing()
    records = _spot_records(result) if result is not None else []
    frame = pd.DataFrame(records, columns=SPOTS_COLUMNS)

    buffer = io.StringIO()
    frame.to_csv(buffer, index=False)
    buffer.seek(0)

    headers = {"Content-Disposition": f'attachment; filename="{SPOTS_EXPORT_FILENAME}"'}
    return StreamingResponse(iter([buffer.getvalue()]), media_type="text/csv", headers=headers)
