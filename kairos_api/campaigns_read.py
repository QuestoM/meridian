"""Clients: the reads behind the client tree, the money board and the rollup.

Three reads live here, and they answer three different questions.

``GET /api/clients`` is the containment an account manager works in: agency,
then the clients under it, then what each has booked, assembled by
:mod:`kairos_api.campaigns_read_clients`.

``GET /api/clients/money`` is the analyst's question: what each client delivered,
gross and net of the agency rebate, with the campaigns and then the individual
spots behind every figure, assembled by :mod:`kairos_api.campaigns_read_money`.
Both of those read the same priced ledger, so an agency total on one is the sum
of the client totals on the other.

``GET /api/campaigns`` is the third and it is older than both. It is the
historical rollup of what aired, moved verbatim from catalog_api.py in the
wave-zero router split, and it stays because it answers a question the other two
do not: which campaign strings the loaded spots source carries at all. Revenue is
reported only when that source actually has it, and the payload says so with
``revenue_available`` rather than ranking on a fabricated zero. The advertiser
column holds to the same rule: the default reference workbook this route reads
carries 12 raw columns and ``advertiser_id`` is not one of them, so a column the
source never supplied is reported ``advertiser_available: false`` instead of a
blank cell with no stated reason. It sits on the same screen as the
operator-scoped money board, so it holds to the same competitor boundary: the
source spots frame carries every channel on disk, and the rollup is scoped to
``settings.operator_channel`` (see :mod:`kairos_api.channel_scope`) before it
groups, with the scope carried in the payload's own ``scope`` block exactly as
the money board carries ``basis``.

``GET /api/campaigns/detail`` is the rollup's own drill: the individual spot
rows behind one campaign-and-advertiser row of the rollup above, so a name on
that screen opens the rows behind it rather than nothing, the way every other
figure on this destination does.
"""

from __future__ import annotations

import datetime
import logging
from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import APIRouter, Request

from kairos_api import channel_scope
from kairos_api.core import (
    DATA_DIR,
    _load_spots,
    _records,
    _series,
    _signature,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _time_of_day(value: Any) -> str:
    """Normalize one raw ``Start time`` cell to an ``HH:MM:SS`` string.

    The reference workbook mixes ``datetime.time`` and ``datetime.datetime``
    objects in the same column (a spot that airs past midnight serializes as a
    full datetime rather than a bare time), and pandas can neither sort nor
    JSON-encode that mix without raising. Reducing every cell to the same
    plain string, here, once, is what lets the detail drill below sort by air
    time instead of crashing the endpoint into a 500.
    """
    if isinstance(value, datetime.datetime):
        value = value.time()
    if isinstance(value, datetime.time):
        return value.strftime("%H:%M:%S")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _prepared_spots(spots: pd.DataFrame) -> tuple[pd.DataFrame, bool, bool]:
    """Common columns, backfilled once, so the rollup and its drill agree.

    Returns the frame plus two honesty flags read off the source *before* any
    backfill: whether ``revenue_ils`` and ``advertiser_id`` were really there.
    The restructured Spots export may omit either identity/money column, and
    when it does the rollup must say so rather than render a blank cell with
    no stated reason, the way ``revenue_available`` already does for money.
    """
    frame = spots.copy()
    has_revenue = "revenue_ils" in frame.columns
    has_advertiser = "advertiser_id" in frame.columns
    frame["revenue_ils"] = pd.to_numeric(_series(frame, "revenue_ils", 0), errors="coerce").fillna(0)
    frame["Duration"] = pd.to_numeric(_series(frame, "Duration", 0), errors="coerce").fillna(0)
    # The restructured Spots export may omit the identity/grouping columns. Backfill
    # any that are missing with honest neutral defaults so the rollup degrades to a
    # single bucket instead of crashing the endpoint (KeyError) into a 500.
    frame["Campaign"] = _series(frame, "Campaign", "Unknown campaign")
    frame["advertiser_id"] = _series(frame, "advertiser_id", "")
    frame["Channel"] = _series(frame, "Channel", "")
    frame["Date"] = _series(frame, "Date", "")
    frame["Start time"] = _series(frame, "Start time", "").map(_time_of_day)
    return frame, has_revenue, has_advertiser


def _build_campaigns(spots: pd.DataFrame) -> dict[str, Any]:
    if spots.empty:
        _, scope = channel_scope.scope_frame(spots, column="Channel")
        return {"campaigns": [], "revenue_available": True, "advertiser_available": True, "scope": scope}

    frame, has_revenue, has_advertiser = _prepared_spots(spots)
    # Date is a DD/MM/YYYY string, so a raw column .max() orders lexicographically
    # (by day-of-month first) rather than chronologically. Parse to a real datetime
    # for the max, then format back to the DD/MM/YYYY display contract. Unparseable
    # dates coerce to NaT and render as an honest empty value, never a fabricated date.
    frame["_date"] = pd.to_datetime(frame["Date"], format="%d/%m/%Y", errors="coerce")
    # This rollup sits on the same screen as the operator-scoped money board, so it
    # must hold to the same competitor boundary: the spots frame carries every
    # channel Spots.csv has (measured, four of them), and an unscoped groupby put
    # rival-channel spots, minutes and last-airing dates on an operator surface.
    # Scoped to settings.operator_channel exactly as /api/clients/money is, with the
    # channels-per-campaign count dropped from the payload: once the frame is one
    # channel that count is always one and cannot mean anything.
    frame, scope = channel_scope.scope_frame(frame, column="Channel")
    grouped = (
        frame.groupby(["Campaign", "advertiser_id"], dropna=False)
        .agg(
            spots=("Campaign", "count"),
            seconds=("Duration", "sum"),
            revenue=("revenue_ils", "sum"),
            last_airing=("_date", "max"),
        )
        .reset_index()
        .sort_values("revenue" if has_revenue else "spots", ascending=False)
        .head(50)
    )
    grouped["last_airing"] = grouped["last_airing"].dt.strftime("%d/%m/%Y").where(grouped["last_airing"].notna(), None)
    if not has_revenue:
        grouped["revenue"] = None
    if not has_advertiser:
        # The whole column is absent from the source, not merely blank on some
        # rows, so every cell is reported unavailable rather than an empty
        # string a reader could mistake for an advertiser that really has no
        # name.
        grouped["advertiser_id"] = None
    return {
        "campaigns": _records(grouped),
        "revenue_available": has_revenue,
        "advertiser_available": has_advertiser,
        "scope": scope,
    }


def _campaign_detail(spots: pd.DataFrame, campaign: str, advertiser: str) -> dict[str, Any]:
    """The individual spot rows behind one rollup row, so its name opens something.

    Same competitor-scoped, honestly-backfilled frame the rollup groups, filtered
    to the one campaign and advertiser a row named and sorted by air time. Capped
    at 200 rows with the true ``count`` carried alongside, so a long campaign
    states a floor rather than a silent truncation.
    """
    if spots.empty:
        return {"spots": [], "count": 0, "revenue_available": True, "scope": None}
    frame, has_revenue, _ = _prepared_spots(spots)
    frame, scope = channel_scope.scope_frame(frame, column="Channel")
    match = frame[
        (frame["Campaign"].astype(str) == campaign) & (frame["advertiser_id"].astype(str) == advertiser)
    ].copy()
    match["_date"] = pd.to_datetime(match["Date"], format="%d/%m/%Y", errors="coerce")
    match = match.sort_values(["_date", "Start time"])
    shown = match[["Date", "Start time", "Duration", "revenue_ils"]].rename(columns={"Start time": "start_time"})
    if not has_revenue:
        shown["revenue_ils"] = None
    return {
        "spots": _records(shown, limit=200),
        "count": len(match),
        "revenue_available": has_revenue,
        "scope": scope,
    }


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


@router.get("/api/campaigns/detail", tags=["catalog"])
def campaign_detail(campaign: str, advertiser: str = "") -> dict[str, Any]:
    """The rows behind one rollup row, so a campaign name opens something.

    ``campaign`` and ``advertiser`` together are the same pair the rollup
    grouped on, so the row a person clicked is the exact set of spots this
    returns rather than every airing of a campaign name reused by a different
    advertiser.
    """
    return _campaign_detail(_load_spots(), campaign, advertiser)


@router.get("/api/clients", tags=["clients"])
def clients(request: Request = None) -> dict[str, Any]:
    """Every agency with its clients, their campaigns and their delivered money."""
    from kairos_api.campaigns_api import CLIENTS_WALL
    from kairos_api.campaigns_read_clients import client_tree

    return CLIENTS_WALL.stamp(dict(client_tree()), request)


@router.get("/api/clients/money", tags=["clients"])
def clients_money(request: Request = None) -> dict[str, Any]:
    """The priced day, grouped by client, campaign, agency, break and spot."""
    from kairos_api.campaigns_api import CLIENTS_WALL
    from kairos_api.campaigns_read_money import board

    return CLIENTS_WALL.stamp(dict(board()), request)


@router.get("/api/clients/money/advertiser/{advertiser:path}", tags=["clients"])
def client_money(advertiser: str) -> dict[str, Any]:
    """One client's figure and every row behind it, for a direct link to it."""
    from kairos_api.campaigns_read_money import money_for_advertiser

    return money_for_advertiser(advertiser)
