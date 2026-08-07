"""Clients, pacing: the two acts that write, and the words a refused one uses.

The router holds the routes and this module holds what a write puts on a row. It
is split out for the same reason the arithmetic and the copy are split out: the
row a write composes is the part a critic has to be able to read in one screen,
and the part most likely to be wrong in a way nobody notices.

Two acts write here and they are opposites, so they are written side by side.
Raising a make-good records that the channel owes compensating delivery. Taking
the risk on records that a person read the same row and decided it stands. Both
stamp the figures the board measured at the instant of the act and neither takes
a figure from the caller, which is what keeps the ledger from ever holding a
number nobody computed.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException

from kairos_api import makegood_store as ledger
from kairos_api import pacing_alerts_api_board as board

OFFER_ORDER_EN = "The offer window ends before it starts."
OFFER_ORDER_HE = "חלון ההצעה מסתיים לפני שהוא מתחיל."
OFFER_VALUE_EN = "An offer carries a value above zero in the shortfall's own unit."
OFFER_VALUE_HE = "הצעה נושאת ערך גדול מאפס ביחידה של החוסר עצמו."
NEEDS_OFFER_EN = "A make-good is settled or declined against an offer, and this one carries none yet."
NEEDS_OFFER_HE = "פיצוי נסגר או נדחה מול הצעה, ולפיצוי הזה עדיין אין הצעה."

NOWHERE_EN = "nowhere, it is finished"
NOWHERE_HE = "לשום מצב, הוא סגור"
UNKNOWN_STATE_EN = "a state this ledger does not hold"
UNKNOWN_STATE_HE = "מצב שספר ההחלטות אינו מחזיק"

# A refusal is written per kind and not assembled from a noun and a verb, because
# Hebrew agrees the verb with the noun's gender and a sentence stitched from parts
# reads as broken to the only people who will ever read it.
REFUSED_EN = {
    ledger.MAKE_GOOD: "A make-good that is {now} does not move to {want}. From here it moves to: {allowed}.",
    ledger.ACCEPTANCE: "A recorded decision that is {now} does not move to {want}. From here it moves to: {allowed}.",
}
REFUSED_HE = {
    ledger.MAKE_GOOD: "פיצוי שמצבו {now} אינו עובר למצב {want}. מכאן הוא עובר אל: {allowed}.",
    ledger.ACCEPTANCE: "החלטה רשומה שמצבה {now} אינה עוברת למצב {want}. מכאן היא עוברת אל: {allowed}.",
}


def refuse(status_code: int, message_en: str, message_he: str,
           opens: Optional[dict[str, str]] = None) -> HTTPException:
    """One refusal in both languages, the shape every write on this spine already sends."""
    detail: dict[str, Any] = {"message_en": message_en, "message_he": message_he}
    if opens and opens.get("kind") and opens.get("id"):
        detail["opens"] = {"kind": str(opens["kind"]), "id": str(opens["id"])}
    return HTTPException(status_code=status_code, detail=detail)


def state_words(state: str) -> tuple[str, str]:
    """One state in the reader's own word, never the store's key.

    The keys are how the rows are stored. A person reads the label the ledger
    publishes beside them, and a refusal is the one place a raw key used to reach
    a screen in both languages at once.
    """
    for entry in ledger.STATE_VOCABULARY:
        if entry["value"] == state:
            return str(entry["label_en"]), str(entry["label_he"])
    return UNKNOWN_STATE_EN, UNKNOWN_STATE_HE


def refuse_transition(current: str, target: str, kind: str) -> HTTPException:
    """A move the machine does not hold, stated in labels the reader has already seen."""
    now_en, now_he = state_words(current)
    want_en, want_he = state_words(target)
    allowed = sorted(ledger.TRANSITIONS.get(current, frozenset()))
    allowed_en = ", ".join(state_words(state)[0] for state in allowed) or NOWHERE_EN
    allowed_he = ", ".join(state_words(state)[1] for state in allowed) or NOWHERE_HE
    shape = kind if kind in REFUSED_EN else ledger.MAKE_GOOD
    return refuse(
        409,
        REFUSED_EN[shape].format(now=now_en, want=want_en, allowed=allowed_en),
        REFUSED_HE[shape].format(now=now_he, want=want_he, allowed=allowed_he),
    )


def new_row(record_id: str, kind: str, row: dict[str, Any], view: dict[str, Any],
            deficit: dict[str, Any], note: str, actor: str) -> dict[str, str]:
    """The ledger row an act writes, every figure taken from the board rather than the request."""
    flight = row["flight"]
    fresh = ledger.blank_row()
    fresh.update({
        "make_good_id": record_id,
        "kind": kind,
        "campaign_id": row["campaign_id"],
        "campaign_name": row["name"],
        "advertiser": row["advertiser"],
        "channel": row["channel"],
        "flight_starts_on": flight["starts_on"],
        "flight_ends_on": flight["ends_on"],
        "unit": deficit["unit"],
        "goal_value": f"{deficit['goal_value']}",
        "counted_value": f"{deficit['counted_value']}",
        "deficit_value": f"{deficit['deficit_value']}",
        "deficit_kind": deficit["deficit_kind"],
        "counted_as_of": str(view.get("as_of", {}).get("instant", "")),
        "days_counted": f"{flight['days_counted']}",
        "days_in_flight": f"{flight['days']}",
        "unsourced_days": f"{deficit['unsourced_days']}",
        "state": ledger.ENTRY_STATE[kind],
        "raised_at": ledger.now_stamp(),
        "raised_by": actor,
        "raised_note": note,
        "is_demo": "true" if row["is_demo"] else "false",
    })
    return fresh


def apply_move(frame: Any, index: int, target: str, payload: Any, actor: str) -> None:
    """Write one transition onto the row, validating what the target state needs."""
    stamp = ledger.now_stamp()
    if target == ledger.OFFERED:
        value = payload.offer_value
        if value is None or float(value) <= 0:
            raise refuse(400, OFFER_VALUE_EN, OFFER_VALUE_HE)
        start = payload.offer_window_start.strip()
        end = payload.offer_window_end.strip()
        if start and end and board.parse_date(end) and board.parse_date(start):
            if board.parse_date(end) < board.parse_date(start):
                raise refuse(400, OFFER_ORDER_EN, OFFER_ORDER_HE)
        frame.at[index, "offer_value"] = f"{round(float(value), 2)}"
        frame.at[index, "offer_window_start"] = start
        frame.at[index, "offer_window_end"] = end
        frame.at[index, "offered_at"] = stamp
        frame.at[index, "offered_by"] = actor
        frame.at[index, "offer_note"] = payload.note.strip()
    elif target in ledger.NEEDS_OFFER:
        if not str(frame.at[index, "offer_value"] or "").strip():
            raise refuse(409, NEEDS_OFFER_EN, NEEDS_OFFER_HE)
    if target in (ledger.SETTLED, ledger.DECLINED, ledger.WITHDRAWN):
        frame.at[index, "closed_at"] = stamp
        frame.at[index, "closed_by"] = actor
        frame.at[index, "close_note"] = payload.note.strip()
    frame.at[index, "state"] = target
