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
    "agency",
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
    "net_revenue",
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


def _csv_response(frame: pd.DataFrame, filename: str, extra_headers: Optional[dict[str, str]] = None) -> StreamingResponse:
    """Stream a DataFrame as a downloadable CSV that Excel opens correctly.

    Encoded utf-8-sig (a BOM prefix) because Excel sniffs the BOM to decode
    Hebrew; without it every channel and programme name renders as mojibake. The
    media type carries an explicit charset for the same reason.
    """
    payload = frame.to_csv(index=False).encode("utf-8-sig")
    headers = {"Content-Disposition": f'attachment; filename="{filename}"'}
    if extra_headers:
        headers.update(extra_headers)
    return StreamingResponse(iter([payload]), media_type="text/csv; charset=utf-8", headers=headers)


def _schedule_freshness_status() -> str:
    """The real fresh/stale/unknown verdict for the saved weekly schedule.

    The same read-only comparison the dashboard staleness banner uses
    (:func:`kairos.export.schedule_freshness.schedule_freshness`); honest
    ``unknown`` when the verdict itself cannot be computed.
    """
    try:
        from kairos.export.schedule_freshness import schedule_freshness

        return str(schedule_freshness(ROOT).get("status") or "unknown")
    except Exception:
        return "unknown"


@router.get("/schedule.csv")
def export_schedule_csv() -> StreamingResponse:
    """Stream the operator's own channel from the optimized weekly plan.

    RULING 009, decided 2026-08-09. Two tests looked as though they demanded
    opposite things, and they did not.

    ``output/weekly_break_schedule.csv`` is the plan of record and it carries
    EVERY channel, because that is what the optimizer computed and what the
    golden reproduces. Nothing here changes that file.

    This ROUTE is a different thing. It is an operator surface, and the boundary
    this product holds everywhere else is that an operator sees exactly one
    channel: the one in settings. A download that streams a rival's programme
    titles and revenue is the same breach as printing them on a screen, and the
    fact that it arrives as a file rather than as pixels changes nothing.

    So the artifact keeps every channel and the route serves one. The test that
    reads "the export really does carry every channel" now reads the FILE, which
    is what its claim was always about.

    When no operator channel is configured the route refuses rather than falling
    back to everything, because an unset boundary is the case where a leak is
    most likely and least noticed.

    The response carries an ``X-Kairos-Schedule-Freshness`` header with the real
    fresh/stale/unknown verdict, so a client can warn before saving a plan whose
    inputs have moved on since it was computed.
    """
    frame = _load_plan()
    if frame is None or frame.empty:
        raise HTTPException(
            status_code=404,
            detail="No optimized weekly plan is available to export. Run the optimizer first.",
        )
    # Through channel_scope, which is the one home for "which channel does this
    # operator own". Reading the settings object directly here would work and
    # would be wrong: every other scoped surface goes through this seam, and a
    # route that answers the same question its own way is how two answers appear.
    from kairos_api import channel_scope

    owned = str(channel_scope.operator_channel() or "").strip()
    if not owned:
        raise HTTPException(
            status_code=409,
            detail=(
                "No operator channel is configured, so this export cannot be scoped to your "
                "own channel. Set the operator channel in Settings first."
            ),
        )
    if "channel" in frame.columns:
        frame = frame[frame["channel"].astype(str).str.strip() == owned]
    return _csv_response(
        frame,
        EXPORT_FILENAME,
        extra_headers={"X-Kairos-Schedule-Freshness": _schedule_freshness_status()},
    )


def _blank(value: Optional[Any]) -> Any:
    """Empty string for a missing value, so a dropped-spot row leaves priced-only
    columns genuinely blank rather than showing a fabricated zero."""
    return "" if value is None else value


def _load_daily_pricing(path=None):
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

    path = Path(path) if path is not None else _newest_daily()
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
            "agency": spot.agency,
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
            "net_revenue": spot.net_revenue,
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
            "agency": drop.agency,
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
            "net_revenue": "",
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
            "agency": "",
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
            "net_revenue": "",
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
    return _csv_response(frame, SPOTS_EXPORT_FILENAME)
