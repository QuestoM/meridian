"""The decision ledger: a measured shortfall, what was done about it, and who acted.

A make-good is the compensating delivery a channel owes a client when a flight
finishes under the goal it was booked against. Before this store the product held
the shortfall as a fraction returned by a projection function and nothing else:
no record, no remedy, no approval, no link back to the flight it belongs to.

**Two kinds of record, because the job has two endings.** The job this ledger
serves is done when every at-risk campaign has either an act taken against it or
an explicit decision to accept the risk, and both are recorded. A ledger that held
only the first left an at-risk campaign somebody had read and accepted looking
exactly like one nobody had opened. So a record is a ``make_good``, raised against
a measured shortfall and offered, settled, declined or withdrawn; or it is an
``acceptance``, the recorded decision that the risk on this row is accepted as it
stands. An acceptance derives nothing: it stamps the figures the board showed at
the instant of the decision, and it never becomes an offer.

Four rules hold this store honest and each one is enforced here rather than
explained on the surface.

**The deficit is measured by this product, never supplied by the caller.** A raise
request names a campaign; the figures are taken from the pacing board at the
instant of the raise and stamped onto the row with the instant they were counted
at. A caller cannot post a shortfall.

**The deficit carries how certain it is.** ``measured_closed`` is a flight that has
ended with every broadcast day sourced, so the number is final. ``booked_short``
is a flight still running whose remaining days all carry a source and still do not
reach the goal. ``to_date`` is the gap against the pace reference at the day the
ledger was counted at, which moves as the flight runs. A row never claims more
certainty than the ledger it was measured from.

**The offer is a human's number and says so.** What a channel may offer against a
shortfall is a commercial rule this product was never given, so nothing here
proposes an offer, derives an entitlement or reserves inventory. The ledger records
the value and the window a person entered, in the shortfall's own unit, and the
sign-off rule is published as not configured.

**Nothing is deleted.** A make-good that should not have been raised is withdrawn,
which is a state with an actor and an instant, exactly as the sibling stores end a
record rather than remove it.
"""

from __future__ import annotations

import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# The controlled values, the vocabularies and the published sentences live beside
# this module under the section 8.2 helper naming rule, because this file passed
# the 450-line law. They are re-exported here so that every existing importer of
# ``makegood_store`` keeps reading exactly the names it always read.
from kairos_api.makegood_store_words import (  # noqa: F401
    ACCEPTANCE,
    ACCEPTED,
    BOOKED_SHORT,
    CLOSE_REASONS,
    DECLINED,
    DEFICIT_KIND_VOCABULARY,
    ENTRY_STATE,
    KIND_VOCABULARY,
    MAKE_GOOD,
    MEASURED_CLOSED,
    NEEDS_OFFER,
    NO_INVENTORY_EN,
    NO_INVENTORY_HE,
    OFFERED,
    OTHER,
    RAISED,
    REASON_REQUIRED,
    SETTLED,
    SIGN_OFF_EN,
    SIGN_OFF_HE,
    SIGN_OFF_PATH_EN,
    SIGN_OFF_PATH_HE,
    STATE_VOCABULARY,
    TO_DATE,
    TRANSITIONS,
    UNDO_REASON,
    UNDO_STATE,
    WITHDRAWN,
    close_reasons,
    reason_allowed,
    undo_block,
)
from kairos_api.store_columns import projected

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
MAKE_GOODS_PATH = DATA_DIR / "make_goods.csv"

COLUMNS = [
    "make_good_id",
    # Which of the two endings this row records. Blank reads as a make-good, so a
    # ledger written before the second kind existed still loads exactly as it was.
    "kind",
    "campaign_id",
    "campaign_name",
    "advertiser",
    "channel",
    "flight_starts_on",
    "flight_ends_on",
    # What was measured, in the goal's own unit, at the instant the ledger was counted at.
    "unit",
    "goal_value",
    "counted_value",
    "deficit_value",
    "deficit_kind",
    "counted_as_of",
    "days_counted",
    "days_in_flight",
    "unsourced_days",
    # The ledger's own life.
    "state",
    "raised_at",
    "raised_by",
    "raised_note",
    # The offer, which is a person's number and never this product's.
    "offer_value",
    "offer_window_start",
    "offer_window_end",
    "offered_at",
    "offered_by",
    "offer_note",
    "closed_at",
    "closed_by",
    # Why a record was closed without a delivery being made. A controlled value
    # and never free text, because a withdrawal recorded as a note nobody has to
    # write is a record that cannot be counted, compared or answered for.
    "close_reason",
    "close_note",
    "is_demo",
]

_STORE_LOCK = threading.Lock()


def now_stamp() -> str:
    """The instant an act is recorded at, in UTC, to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lock() -> threading.Lock:
    return _STORE_LOCK


def load_frame() -> pd.DataFrame:
    """Every make-good, or an empty frame when the ledger has never been written."""
    if not MAKE_GOODS_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(MAKE_GOODS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _backup() -> None:
    if not MAKE_GOODS_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(MAKE_GOODS_PATH, BACKUP_DIR / f"make_goods_{stamp}.csv")


def write_frame(frame: pd.DataFrame) -> None:
    """Back up, then write atomically, exactly as the sibling stores do."""
    _backup()
    MAKE_GOODS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = MAKE_GOODS_PATH.with_name(MAKE_GOODS_PATH.name + ".tmp")
    projected(frame, COLUMNS).to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, MAKE_GOODS_PATH)


def blank_row() -> dict[str, str]:
    return {column: "" for column in COLUMNS}


def next_id(frame: pd.DataFrame) -> str:
    """The next MG_ id, one above the highest already used."""
    highest = 0
    for value in frame.get("make_good_id", []):
        text = str(value or "").strip()
        if text.startswith("MG_") and text[3:].isdigit():
            highest = max(highest, int(text[3:]))
    return f"MG_{highest + 1:04d}"


def _text(row: Any, column: str) -> str:
    return str(row.get(column, "") or "").strip()


def _number(raw: Any) -> Optional[float]:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        return round(float(text), 4)
    except (TypeError, ValueError):
        return None


def _truth(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"true", "yes", "1", "y"}


def record(row: Any) -> dict[str, Any]:
    """One make-good as the API reports it, with every figure a number or None."""
    state = _text(row, "state") or RAISED
    return {
        "make_good_id": _text(row, "make_good_id"),
        "kind": _text(row, "kind") or MAKE_GOOD,
        "campaign_id": _text(row, "campaign_id"),
        "campaign_name": _text(row, "campaign_name"),
        "advertiser": _text(row, "advertiser"),
        "channel": _text(row, "channel"),
        "flight": {
            "starts_on": _text(row, "flight_starts_on"),
            "ends_on": _text(row, "flight_ends_on"),
            "days_counted": int(_number(row.get("days_counted")) or 0),
            "days_in_flight": int(_number(row.get("days_in_flight")) or 0),
            "unsourced_days": int(_number(row.get("unsourced_days")) or 0),
        },
        "shortfall": {
            "unit": _text(row, "unit"),
            "goal_value": _number(row.get("goal_value")),
            "counted_value": _number(row.get("counted_value")),
            "deficit_value": _number(row.get("deficit_value")),
            "deficit_kind": _text(row, "deficit_kind"),
            "counted_as_of": _text(row, "counted_as_of"),
        },
        "state": state,
        "next_states": sorted(TRANSITIONS.get(state, frozenset())),
        "raised_at": _text(row, "raised_at"),
        "raised_by": _text(row, "raised_by"),
        "raised_note": _text(row, "raised_note"),
        "offer": {
            "value": _number(row.get("offer_value")),
            "window_start": _text(row, "offer_window_start"),
            "window_end": _text(row, "offer_window_end"),
            "offered_at": _text(row, "offered_at"),
            "offered_by": _text(row, "offered_by"),
            "note": _text(row, "offer_note"),
        },
        "closed_at": _text(row, "closed_at"),
        "closed_by": _text(row, "closed_by"),
        "close_reason": _text(row, "close_reason"),
        "close_note": _text(row, "close_note"),
        "is_demo": _truth(row.get("is_demo")),
    }


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Every make-good, newest raise first, so the ledger reads as a timeline."""
    rows = [record(row) for _, row in frame.iterrows()]
    rows.sort(key=lambda item: (item["raised_at"], item["make_good_id"]), reverse=True)
    return rows


def locate(frame: pd.DataFrame, make_good_id: str) -> int:
    """The frame index of one make-good, or -1 when the ledger does not hold it."""
    mask = frame["make_good_id"].astype(str) == str(make_good_id)
    if not mask.any():
        return -1
    return int(frame.index[mask][0])


def transition_allowed(current: str, target: str) -> bool:
    return target in TRANSITIONS.get(current, frozenset())


def open_for(frame: pd.DataFrame, campaign_id: str, kind: str = MAKE_GOOD) -> list[str]:
    """The ids of the records of one kind already open against a campaign.

    Open means not settled and not withdrawn. A second raise against a campaign
    that already has one open is a duplicate, and the caller refuses it by name
    rather than quietly adding a second row for the same shortfall.

    The kind is part of the question. An accepted risk is not a make-good, so a
    campaign whose risk somebody took on last week must still be able to carry a
    make-good when the shortfall becomes owed, and a shared duplicate check would
    have silently forbidden exactly that.
    """
    out: list[str] = []
    for _, row in frame.iterrows():
        if _text(row, "campaign_id") != str(campaign_id):
            continue
        if (_text(row, "kind") or MAKE_GOOD) != kind:
            continue
        if (_text(row, "state") or RAISED) in (SETTLED, WITHDRAWN):
            continue
        out.append(_text(row, "make_good_id"))
    return out


def sign_off_block() -> dict[str, Any]:
    """What this ledger does not decide, published beside everything it does."""
    return {
        "configured": False,
        "reason_en": SIGN_OFF_EN,
        "reason_he": SIGN_OFF_HE,
        "path_forward_en": SIGN_OFF_PATH_EN,
        "path_forward_he": SIGN_OFF_PATH_HE,
        "offer_reserves_nothing_en": NO_INVENTORY_EN,
        "offer_reserves_nothing_he": NO_INVENTORY_HE,
    }


def vocabularies() -> dict[str, Any]:
    """Every controlled word the ledger prints, both languages."""
    return {
        "states": [dict(entry) for entry in STATE_VOCABULARY],
        "kinds": [dict(entry) for entry in KIND_VOCABULARY],
        "deficit_kinds": [dict(entry) for entry in DEFICIT_KIND_VOCABULARY],
        "transitions": {state: sorted(nexts) for state, nexts in TRANSITIONS.items()},
        # The reasons a closing transition takes, published beside the states so
        # a surface offers exactly the ones the store will accept.
        "close_reasons": {state: [dict(entry) for entry in entries]
                          for state, entries in CLOSE_REASONS.items()},
        "reason_required": sorted(REASON_REQUIRED),
        "reason_needing_a_note": OTHER,
        # How a decision is reversed, so the control that offers it reads the
        # rule rather than holding a second copy of it.
        "undo": undo_block(),
    }
