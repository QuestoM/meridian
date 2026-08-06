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
``revenue_available`` rather than ranking on a fabricated zero. It sits on the
same screen as the operator-scoped money board, so it holds to the same
competitor boundary: the source spots frame carries every channel on disk, and
the rollup is scoped to ``settings.operator_channel`` (see
:mod:`kairos_api.channel_scope`) before it groups, with the scope carried in
the payload's own ``scope`` block exactly as the money board carries ``basis``.
"""

from __future__ import annotations

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


def _build_campaigns(spots: pd.DataFrame) -> dict[str, Any]:
    if spots.empty:
        _, scope = channel_scope.scope_frame(spots, column="Channel")
        return {"campaigns": [], "scope": scope}

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
    return {"campaigns": _records(grouped), "revenue_available": has_revenue, "scope": scope}


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
