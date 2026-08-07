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
    "close_note",
    "is_demo",
]

MAKE_GOOD = "make_good"
ACCEPTANCE = "acceptance"

RAISED = "raised"
OFFERED = "offered"
SETTLED = "settled"
DECLINED = "declined"
WITHDRAWN = "withdrawn"
ACCEPTED = "accepted"

# The entry state each kind is written in. Nothing else may be written directly.
ENTRY_STATE = {MAKE_GOOD: RAISED, ACCEPTANCE: ACCEPTED}

# Which state may follow which. A settled or withdrawn record is finished, so it
# has no successor: reopening one would rewrite a record somebody acted on. One
# table covers both kinds because no state is shared between their two paths
# except the ending, so a state names its kind without ambiguity.
TRANSITIONS: dict[str, frozenset[str]] = {
    RAISED: frozenset({OFFERED, WITHDRAWN}),
    OFFERED: frozenset({SETTLED, DECLINED, WITHDRAWN}),
    DECLINED: frozenset({OFFERED, WITHDRAWN}),
    ACCEPTED: frozenset({WITHDRAWN}),
    SETTLED: frozenset(),
    WITHDRAWN: frozenset(),
}

# The states that require an offer to exist before they can be entered, because
# settling or declining nothing is not a thing a person can mean.
NEEDS_OFFER = frozenset({SETTLED, DECLINED})

MEASURED_CLOSED = "measured_closed"
BOOKED_SHORT = "booked_short"
TO_DATE = "to_date"

STATE_VOCABULARY = (
    {
        "value": RAISED,
        "label_en": "Raised",
        "label_he": "נפתח",
        "meaning_en": "The shortfall is recorded with the figures it was measured from.",
        "meaning_he": "החוסר נרשם יחד עם הנתונים שממנו נמדד.",
    },
    {
        "value": OFFERED,
        "label_en": "Offered",
        "label_he": "הוצעה השלמה",
        "meaning_en": "Compensating delivery has been offered to the client.",
        "meaning_he": "הוצעה ללקוח השלמת שידור.",
    },
    {
        "value": SETTLED,
        "label_en": "Settled",
        "label_he": "נסגר",
        "meaning_en": "The client accepted the offer and the make-good is closed.",
        "meaning_he": "הלקוח קיבל את ההצעה והפיצוי נסגר.",
    },
    {
        "value": DECLINED,
        "label_en": "Declined",
        "label_he": "נדחה",
        "meaning_en": "The client refused the offer. A new offer may follow it.",
        "meaning_he": "הלקוח דחה את ההצעה. אפשר להציע הצעה חדשה אחריה.",
    },
    {
        "value": WITHDRAWN,
        "label_en": "Withdrawn",
        "label_he": "בוטל",
        "meaning_en": "The record should not have been opened, and the row records who withdrew it.",
        "meaning_he": "לא היה מקום לפתוח את הרשומה, והשורה רושמת מי ביטל אותה.",
    },
    {
        "value": ACCEPTED,
        "label_en": "Risk taken on",
        "label_he": "הסיכון התקבל",
        "meaning_en": "A person read the row and decided the risk stands as it is. The figures are the ones the board showed at that instant.",
        "meaning_he": "אדם קרא את השורה והחליט שהסיכון נשאר כפי שהוא. הנתונים הם אלה שהלוח הציג באותו רגע.",
    },
)

KIND_VOCABULARY = (
    {
        "value": MAKE_GOOD,
        "label_en": "Make-good",
        "label_he": "פיצוי שידור",
        "meaning_en": "Compensating delivery raised against a measured shortfall.",
        "meaning_he": "השלמת שידור שנפתחה מול חוסר נמדד.",
    },
    {
        "value": ACCEPTANCE,
        "label_en": "Risk taken on",
        "label_he": "קבלת הסיכון",
        "meaning_en": "A recorded decision to leave the risk on this campaign as it stands.",
        "meaning_he": "החלטה רשומה להשאיר את הסיכון בקמפיין הזה כפי שהוא.",
    },
)

DEFICIT_KIND_VOCABULARY = (
    {
        "value": MEASURED_CLOSED,
        "label_en": "Measured, flight closed",
        "label_he": "נמדד, הטיסה נסגרה",
        "meaning_en": "The flight has ended and every broadcast day of it carries a source, so the figure is final.",
        "meaning_he": "הטיסה הסתיימה ולכל יום שידור שלה יש מקור, ולכן הנתון סופי.",
    },
    {
        "value": BOOKED_SHORT,
        "label_en": "Booked short",
        "label_he": "ההזמנה חסרה",
        "meaning_en": "Every remaining day carries a source and the spots on them do not reach the goal.",
        "meaning_he": "לכל יום שנותר יש מקור והתשדירים שבהם אינם מגיעים ליעד.",
    },
    {
        "value": TO_DATE,
        "label_en": "Gap to date",
        "label_he": "פער עד כה",
        "meaning_en": "The gap against the pace reference at the counted day. It moves as the flight runs.",
        "meaning_he": "הפער מול ייחוס הקצב ביום שנספר. הוא משתנה ככל שהטיסה מתקדמת.",
    },
)

SIGN_OFF_EN = (
    "This product was not given the commercial rule for what a make-good may be offered against or who "
    "signs one off, so it records who acted and refuses to derive an entitlement."
)
SIGN_OFF_HE = (
    "המוצר לא קיבל את הכלל המסחרי לגבי מה מותר להציע כפיצוי או מי מאשר אותו, ולכן הוא רושם מי פעל "
    "ונמנע מלגזור זכאות."
)
SIGN_OFF_PATH_EN = "Supply the offer rule and the approver role, and this ledger will enforce them."
SIGN_OFF_PATH_HE = "ספקו את כלל ההצעה ואת תפקיד המאשר, וספר הפיצויים יאכוף אותם."

NO_INVENTORY_EN = (
    "An offer is a recorded value and a window. It does not reserve inventory: the spots are booked on "
    "the plan, and this row links to the campaign they belong to."
)
NO_INVENTORY_HE = (
    "הצעה היא ערך רשום וחלון תאריכים. היא אינה משריינת מלאי: התשדירים מוזמנים בתוכנית, והשורה הזו "
    "מקשרת לקמפיין שאליו הם שייכים."
)

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
    frame[COLUMNS].to_csv(tmp, index=False, encoding="utf-8-sig")
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
    }
