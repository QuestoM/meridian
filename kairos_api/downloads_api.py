"""Sources, downloads: the report shelf and the source-file audit.

Moved verbatim from catalog_api.py as part of the wave-zero router split. The
five reports carry four owner departments, so the shelf itself is load bearing
and survives the rebuild. Every row's status is read from real state: an empty
plan reports empty, and the daily ledger row counts the rows the download
actually carries.

The compliance row composes the frozen plan-read verdict, the same object Rules
serves and Today prints, so the three can never disagree.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter

from kairos_api.core import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    KairosSettings,
    _load_break_schedule,
    _load_settings,
    _signature,
    _summarize_schedule,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _source_file_paths() -> list[Path]:
    """The real source files the data-quality report audits.

    Single source of truth shared with ``/api/files`` so the report's row count
    reflects the actual file set, not a magic constant.
    """
    return [
        DATA_DIR / "Dayparts.csv",
        DATA_DIR / "Programmes.csv",
        DATA_DIR / "Spots.csv",
        DATA_DIR / "rate_card_premiums.csv",
        DATA_DIR / "advertiser_rules.csv",
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        MODELS_DIR / "tv_break_posterior.pkl",
    ]


def _build_reports(schedule: pd.DataFrame, settings: KairosSettings) -> dict[str, Any]:
    # The compliance verdict is the frozen plan read shared with /api/compliance
    # and /api/overview, so all three quote one verdict; imported at call time so
    # the module import graph stays acyclic.
    from kairos_api.plan_read_compliance import build_compliance

    summary = _summarize_schedule(schedule)
    compliance = build_compliance(schedule, settings)
    source_files = _source_file_paths()
    present = sum(1 for path in source_files if path.exists())
    # Status is sourced from the real plan state, not a fixed "ready". An empty
    # schedule (no plan run yet) reports "empty" so the operator sees the honest
    # state instead of a green light backed by zero rows.
    plan_rows = int(len(schedule))
    revenue_rows = int(summary["total_breaks"])
    # Daily spot ledger: the per-spot priced/dropped output of the daily pricing
    # pipeline, downloadable at /api/export/spots.csv. The row count comes from
    # actually running that pipeline over the newest daily file, so it is the
    # exact ledger the download carries; an honest 0 when no daily file exists.
    ledger_rows = 0
    ledger_status = "empty"
    try:
        from kairos_api.exporters import _load_daily_pricing

        ledger = _load_daily_pricing()
        if ledger is not None:
            ledger_rows = int(len(ledger.priced) + len(ledger.dropped) + len(ledger.frequency_dropped))
            ledger_status = "ready" if ledger_rows else "empty"
    except Exception:
        logger.exception("daily spot ledger row count failed")
        ledger_status = "attention"
    return {
        "reports": [
            {"id": "weekly-plan", "title": "Weekly traffic plan", "status": "ready" if plan_rows else "empty", "rows": plan_rows, "owner": "Traffic"},
            {"id": "compliance", "title": "Compliance and guardrails", "status": compliance["status"], "rows": len(compliance["checks"]), "owner": "Legal / Ops"},
            {"id": "revenue", "title": "Revenue forecast", "status": "ready" if revenue_rows else "empty", "rows": revenue_rows, "owner": "Revenue"},
            {"id": "daily-spots", "title": "Daily spot ledger", "status": ledger_status, "rows": ledger_rows, "owner": "Revenue"},
            {"id": "data-quality", "title": "Source file audit", "status": "ready" if present == len(source_files) else "attention", "rows": present, "owner": "Data"},
        ]
    }


@lru_cache(maxsize=16)
def _reports_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_reports(_load_break_schedule(), _load_settings())


@router.get("/api/reports", tags=["catalog"])
def reports() -> dict[str, Any]:
    # The daily spot ledger entry counts the newest daily file's priced ledger,
    # so that file (when present) is part of the cache key.
    paths = [
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        DATA_DIR / "Programmes.csv",
        SETTINGS_PATH,
    ]
    try:
        from kairos_api.uploads import _newest_daily

        newest_daily = _newest_daily()
        if newest_daily is not None:
            paths.append(newest_daily)
    except Exception:
        logger.exception("newest daily file lookup failed for the reports cache key")
    return _reports_cached(_signature(paths))


@router.get("/api/files", tags=["catalog"])
def files() -> dict[str, Any]:
    paths = _source_file_paths()
    return {
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
                "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                if path.exists()
                else None,
            }
            for path in paths
        ]
    }
