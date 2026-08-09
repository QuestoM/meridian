"""The campaign and flight store, persisted to data/campaigns.csv.

Split out of :mod:`kairos_api.campaigns_api` to keep that module under the
project line limit, and modelled on :mod:`kairos_api.agencies` deliberately:
module lock, timestamped backup, temp file plus ``os.replace``, and a version
snapshot before every manual edit. A reader never sees a torn file and two
concurrent edits cannot lose each other's rows.

Two record kinds live in one file, distinguished by ``record_type``, because a
flight is a line of a campaign and never exists without one. One store means one
lock, one atomic write and one restore point for the pair. The columns each kind
uses are declared below rather than left to the reader to infer.

What a campaign is here, and what it deliberately is not. It is the commercial
object an account manager signs: an advertiser, the agency it is bought through,
the operator's own channel, the money and the rating-point goal committed to, a
window, the terms, and one or more flights each with its own window and its
booked goal. It is **not** a delivery record. This file holds no delivered
figure and no ``delivered`` column, so no surface can invent a pace out of it.
What actually aired is a separate ledger, ``data/campaign_delivery.csv``, which
is derived from the traffic log on disk and reports every day it has no source
for as unknown rather than as zero. A campaign is ended, never deleted, for the
same reason an agency is suspended rather than removed: historic spots keep
resolving to it.

The words this store speaks, its refusals and its validators live in
:mod:`kairos_api.campaigns_api_words` and are re-exported here under the names
they always had. The commitment vocabularies live in
:mod:`kairos_api.campaigns_commitment`.
"""

from __future__ import annotations

import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from kairos_api import campaigns_commitment as commitment
from kairos_api import campaigns_goal_order as goal_order
from kairos_api.store_columns import projected
from kairos_api.campaigns_api_words import (  # noqa: F401 - re-exported store surface
    FIELD_WORDS,
    GOAL_KIND_VOCABULARY,
    GOAL_KINDS,
    STATUS_VOCABULARY,
    STATUSES,
    choice_words,
    field_words,
    refuse,
    validate_amount,
    validate_choice,
    validate_date,
    validate_goal,
    validate_percent,
    validate_window,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
CAMPAIGNS_PATH = DATA_DIR / "campaigns.csv"

CAMPAIGN = "campaign"
FLIGHT = "flight"

COLUMNS = [
    "record_type",
    "campaign_id",
    "flight_id",
    "name",
    "advertiser",
    "agency_id",
    "status",
    "starts_on",
    "ends_on",
    "goal_kind",
    "goal_value",
    "rebate_percent",
    "surcharge_discount_percent",
    "surcharge_weekdays",
    "notes",
    "created_at",
    "created_by",
    "data_source",
    # The commitment half: what was bought, in money and in rating points, on
    # the operator's own channel and against a named audience.
    "channel",
    "brand",
    "category",
    "budget_ils",
    "bonus_ils",
    "rating_goal_points",
    "rating_goal_audience",
    "price_model",
    "priority",
    "pacing_mode",
    # The demo marker every surface reads. A seeded row is never a booking.
    "is_demo",
    "demo_note",
]

# Which columns each record kind actually carries, so a blank cell on the other
# kind reads as "not applicable" rather than as missing data.
CAMPAIGN_FIELDS = (
    "campaign_id", "name", "advertiser", "agency_id", "status", "starts_on",
    "ends_on", "rebate_percent", "surcharge_discount_percent",
    "surcharge_weekdays", "notes", "channel", "brand", "category", "budget_ils",
    "bonus_ils", "rating_goal_points", "rating_goal_audience", "price_model",
    "priority", "pacing_mode", "is_demo", "demo_note",
)
FLIGHT_FIELDS = (
    "campaign_id", "flight_id", "name", "starts_on", "ends_on", "goal_kind",
    "goal_value", "notes", "is_demo", "demo_note",
)

_STORE_LOCK = threading.Lock()



def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_frame() -> pd.DataFrame:
    """Every row, or an empty frame when the store has never been written."""
    if not CAMPAIGNS_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(CAMPAIGNS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _backup() -> None:
    if not CAMPAIGNS_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(CAMPAIGNS_PATH, BACKUP_DIR / f"campaigns_{stamp}.csv")


def write_frame(frame: pd.DataFrame) -> None:
    """Backup, then write atomically, exactly as the sibling stores do."""
    _backup()
    CAMPAIGNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = CAMPAIGNS_PATH.with_name(CAMPAIGNS_PATH.name + ".tmp")
    projected(frame, COLUMNS).to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, CAMPAIGNS_PATH)


def snapshot_before_write(request: Any) -> None:
    """Version the campaigns store before a manual edit writes it."""
    from kairos_api import version_store

    version_store.snapshot_manual_edit(request, "campaigns")


def lock() -> threading.Lock:
    return _STORE_LOCK


def _text(row: Any, column: str) -> str:
    return str(row.get(column, "") or "").strip()


def _float_or_none(raw: Any) -> Optional[float]:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        return round(float(text), 4)
    except (TypeError, ValueError):
        return None


def _commitment(row: Any) -> dict[str, Any]:
    """What was bought: the money, the rating goal and the audience it counts against.

    Every figure is ``None`` when the commitment does not state one, never zero.
    A campaign booked without a rating goal has no rating goal, and a board that
    prints ``0`` there has told the reader the buyer committed to nothing, which
    is a different fact from the buyer not having committed in that unit at all.
    """
    audience = _text(row, "rating_goal_audience")
    entry = commitment.audience_entry(audience)
    return {
        "budget_ils": _float_or_none(row.get("budget_ils")),
        "bonus_ils": _float_or_none(row.get("bonus_ils")),
        "rating_goal_points": _float_or_none(row.get("rating_goal_points")),
        "rating_goal_audience": audience,
        "rating_goal_audience_label_en": (entry or {}).get("label_en", ""),
        "rating_goal_audience_label_he": (entry or {}).get("label_he", ""),
        "rating_goal_measurable": bool((entry or {}).get("measurable", False)),
        "rating_goal_reason_en": (entry or {}).get("reason_en", ""),
        "rating_goal_reason_he": (entry or {}).get("reason_he", ""),
        "price_model": _text(row, "price_model"),
        "priority": _text(row, "priority"),
        "pacing_mode": _text(row, "pacing_mode"),
    }


def campaign_record(row: Any) -> dict[str, Any]:
    """One campaign as the API reports it, with its terms coerced honestly."""
    demo = commitment.is_demo(row.get("is_demo"))
    return {
        "campaign_id": _text(row, "campaign_id"),
        "name": _text(row, "name"),
        "advertiser": _text(row, "advertiser"),
        "agency_id": _text(row, "agency_id"),
        "channel": _text(row, "channel"),
        "brand": _text(row, "brand"),
        "category": _text(row, "category"),
        "status": _text(row, "status") or "active",
        "starts_on": _text(row, "starts_on"),
        "ends_on": _text(row, "ends_on"),
        "rebate_percent": _float_or_none(row.get("rebate_percent")),
        "surcharge_discount_percent": _float_or_none(row.get("surcharge_discount_percent")),
        "surcharge_weekdays": _text(row, "surcharge_weekdays"),
        "notes": _text(row, "notes"),
        "created_at": _text(row, "created_at"),
        "created_by": _text(row, "created_by"),
        "data_source": _text(row, "data_source") or "manual",
        "is_demo": demo,
        "demo_note": _text(row, "demo_note"),
        "demo": commitment.demo_block(demo),
        "commitment": _commitment(row),
    }


def flight_record(row: Any) -> dict[str, Any]:
    """One flight as the API reports it. What aired is a different ledger."""
    demo = commitment.is_demo(row.get("is_demo"))
    return {
        "campaign_id": _text(row, "campaign_id"),
        "flight_id": _text(row, "flight_id"),
        "name": _text(row, "name"),
        "starts_on": _text(row, "starts_on"),
        "ends_on": _text(row, "ends_on"),
        "goal_kind": _text(row, "goal_kind"),
        "goal_value": _float_or_none(row.get("goal_value")),
        "notes": _text(row, "notes"),
        "created_at": _text(row, "created_at"),
        "created_by": _text(row, "created_by"),
        "is_demo": demo,
        "demo": commitment.demo_block(demo),
    }


def campaigns_with_flights(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Every campaign, each carrying its own flights, ordered by start date."""
    campaigns = []
    flights = frame[frame["record_type"].astype(str) == FLIGHT]
    for _, row in frame[frame["record_type"].astype(str) == CAMPAIGN].iterrows():
        record = campaign_record(row)
        own = flights[flights["campaign_id"].astype(str) == record["campaign_id"]]
        record["flights"] = sorted(
            (flight_record(flight) for _, flight in own.iterrows()),
            key=lambda flight: (flight["starts_on"], flight["flight_id"]),
        )
        # Which of the three kinds of order this is, published rather than left
        # to a reader to infer. A goal-based order books no lines at all and is
        # COMPLETE that way, so no surface may render its absent spot list as
        # missing data. See kairos_api.campaigns_goal_order.
        record["order"] = goal_order.order_block(record["commitment"], record["flights"])
        campaigns.append(record)
    campaigns.sort(key=lambda item: (item["status"] != "active", item["starts_on"], item["campaign_id"]))
    return campaigns



def locate_campaign(frame: pd.DataFrame, campaign_id: str) -> int:
    mask = (frame["record_type"].astype(str) == CAMPAIGN) & (frame["campaign_id"].astype(str) == campaign_id)
    if not mask.any():
        raise refuse(
            404,
            f"Campaign '{campaign_id}' was not found, so nothing was changed",
            f"הקמפיין ⁦{campaign_id}⁩ לא נמצא, ולכן דבר לא שונה",
        )
    return int(frame.index[mask][0])


def locate_flight(frame: pd.DataFrame, campaign_id: str, flight_id: str) -> int:
    mask = (
        (frame["record_type"].astype(str) == FLIGHT)
        & (frame["campaign_id"].astype(str) == campaign_id)
        & (frame["flight_id"].astype(str) == flight_id)
    )
    if not mask.any():
        raise refuse(
            404,
            f"Flight '{flight_id}' was not found on campaign '{campaign_id}', so nothing was changed",
            f"טיסת השידור ⁦{flight_id}⁩ לא נמצאה בקמפיין ⁦{campaign_id}⁩, ולכן דבר לא שונה",
        )
    return int(frame.index[mask][0])


def next_campaign_id(frame: pd.DataFrame) -> str:
    """The next free CMP_nnn, so nobody has to invent an identifier."""
    used = {
        _text(row, "campaign_id")
        for _, row in frame[frame["record_type"].astype(str) == CAMPAIGN].iterrows()
    }
    index = 1
    while f"CMP_{index:03d}" in used:
        index += 1
    return f"CMP_{index:03d}"


def next_flight_id(frame: pd.DataFrame, campaign_id: str) -> str:
    own = frame[
        (frame["record_type"].astype(str) == FLIGHT)
        & (frame["campaign_id"].astype(str) == campaign_id)
    ]
    used = {_text(row, "flight_id") for _, row in own.iterrows()}
    index = 1
    while f"{campaign_id}_F{index}" in used:
        index += 1
    return f"{campaign_id}_F{index}"


def blank_row() -> dict[str, str]:
    return {column: "" for column in COLUMNS}


def append(frame: pd.DataFrame, row: dict[str, str]) -> pd.DataFrame:
    """Add one row, stamped with the moment it was written.

    The stamp is set on a falsy value rather than on a missing key, because
    :func:`blank_row` seeds every column with an empty string, so a
    ``setdefault`` here finds the key present and stamps nothing. That is how
    every campaign and every flight written before this reached disk with an
    empty ``created_at``, which is a record that cannot say when it was made.
    """
    if not str(row.get("created_at") or "").strip():
        row["created_at"] = _now()
    return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
