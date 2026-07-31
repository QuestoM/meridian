"""Clients: the historical campaign rollup, read from the spots source.

Moved verbatim from catalog_api.py as part of the wave-zero router split. It is a
read of what aired, not a campaign entity: revenue is reported only when the
loaded spots source actually carries it, and the payload says so with
``revenue_available`` rather than ranking on a fabricated zero.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import APIRouter

from kairos_api.core import (
    DATA_DIR,
    _load_spots,
    _records,
    _series,
    _signature,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _build_campaigns(spots: pd.DataFrame) -> dict[str, Any]:
    if spots.empty:
        return {"campaigns": []}

    frame = spots.copy()
    # Revenue is reported only when the spots source actually carries it (the
    # reference airings export does not); otherwise the rollup ranks by spot
    # volume and reports revenue as unavailable rather than a fabricated zero.
    has_revenue = "revenue_ils" in frame.columns
    frame["revenue_ils"] = pd.to_numeric(_series(frame, "revenue_ils", 0), errors="coerce").fillna(0)
    frame["Duration"] = pd.to_numeric(_series(frame, "Duration", 0), errors="coerce").fillna(0)
    # The restructured Spots export may omit the identity/grouping columns. Backfill
    # any that are missing with honest neutral defaults so the rollup degrades to a
    # single bucket instead of crashing the endpoint (KeyError) into a 500.
    frame["Campaign"] = _series(frame, "Campaign", "Unknown campaign")
    frame["advertiser_id"] = _series(frame, "advertiser_id", "")
    frame["Channel"] = _series(frame, "Channel", "")
    frame["Date"] = _series(frame, "Date", "")
    # Date is a DD/MM/YYYY string, so a raw column .max() orders lexicographically
    # (by day-of-month first) rather than chronologically. Parse to a real datetime
    # for the max, then format back to the DD/MM/YYYY display contract. Unparseable
    # dates coerce to NaT and render as an honest empty value, never a fabricated date.
    frame["_date"] = pd.to_datetime(_series(frame, "Date", ""), format="%d/%m/%Y", errors="coerce")
    grouped = (
        frame.groupby(["Campaign", "advertiser_id"], dropna=False)
        .agg(
            spots=("Campaign", "count"),
            seconds=("Duration", "sum"),
            revenue=("revenue_ils", "sum"),
            channels=("Channel", "nunique"),
            last_airing=("_date", "max"),
        )
        .reset_index()
        .sort_values("revenue" if has_revenue else "spots", ascending=False)
        .head(50)
    )
    grouped["last_airing"] = grouped["last_airing"].dt.strftime("%d/%m/%Y").where(grouped["last_airing"].notna(), None)
    if not has_revenue:
        grouped["revenue"] = None
    return {"campaigns": _records(grouped), "revenue_available": has_revenue}


@lru_cache(maxsize=16)
def _campaigns_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_campaigns(_load_spots())


@router.get("/api/campaigns", tags=["catalog"])
def campaigns() -> dict[str, Any]:
    # Same key discipline as /api/inventory: the loader prefers the workbook.
    return _campaigns_cached(_signature([
        DATA_DIR / "reference" / "Spots.xlsx",
        DATA_DIR / "Spots.csv",
    ]))
