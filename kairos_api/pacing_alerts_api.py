"""Clients, pacing: is a campaign behind, and what is owed when it is.

Three reads and two writes, over two stores this module never edits.

``GET /api/pacing`` is the board an account manager opens in the morning: every
campaign on the operator's channel, worst pacing first, each row carrying the
verdict, the figures it was computed from, the published trigger that decided it,
and the one thing to do about it. ``GET`` and ``POST /api/make-goods`` are the
ledger of what is owed, and ``POST /api/make-goods/{id}/state`` moves one along.

``GET /api/make-good-alerts`` is the older projection over
``campaign_flights.csv`` and it is unchanged. That file is still a header-only
seed, so the route still answers ``data_available: false`` with the reason. It is
kept because it is the signal the optimizer's own pacing weights read, and because
Bar 3 forbids removing a working answer; the board above is a different ledger and
says so on every payload.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos_api import makegood_store as ledger
from kairos_api import pacing_alerts_api_board as board
from kairos_api import pacing_alerts_api_read as read
from kairos_api import pacing_alerts_api_words as words
from kairos_api.affiliation_wall import Wall

logger = logging.getLogger(__name__)

router = APIRouter()

# Pacing is a run-side commercial act, so affiliation does not gate it and role
# does: any account reads the board, a write role raises and moves a make-good.
# The wall is on the read so the control tells the truth before it is pressed.
PACING_WALL = Wall(company_only=False)

DUPLICATE_EN = "This campaign already has an open make-good. Open it rather than raising a second one."
DUPLICATE_HE = "לקמפיין הזה כבר פתוח פיצוי שידור. פתחו אותו במקום לפתוח פיצוי שני."
UNKNOWN_MAKE_GOOD_EN = "The ledger holds no make-good with that id."
UNKNOWN_MAKE_GOOD_HE = "ספר הפיצויים אינו מחזיק פיצוי עם המזהה הזה."
NEEDS_OFFER_EN = "A make-good is settled or declined against an offer, and this one carries none yet."
NEEDS_OFFER_HE = "פיצוי נסגר או נדחה מול הצעה, ולפיצוי הזה עדיין אין הצעה."
OFFER_ORDER_EN = "The offer window ends before it starts."
OFFER_ORDER_HE = "חלון ההצעה מסתיים לפני שהוא מתחיל."
OFFER_VALUE_EN = "An offer carries a value above zero in the shortfall's own unit."
OFFER_VALUE_HE = "הצעה נושאת ערך גדול מאפס ביחידה של החוסר עצמו."


def refuse(status_code: int, message_en: str, message_he: str,
           opens: Optional[dict[str, str]] = None) -> HTTPException:
    """One refusal in both languages, the shape every write on this spine already sends."""
    detail: dict[str, Any] = {"message_en": message_en, "message_he": message_he}
    if opens and opens.get("kind") and opens.get("id"):
        detail["opens"] = {"kind": str(opens["kind"]), "id": str(opens["id"])}
    return HTTPException(status_code=status_code, detail=detail)


def _actor(request: "Request | None") -> str:
    """Who acted, when the session says so and blank when it does not."""
    if request is None:
        return ""
    try:
        session = getattr(request.state, "session", None)
        return str(getattr(session, "username", "") or "")
    except Exception:  # noqa: BLE001 - attribution must never fail a write
        return ""


class RaiseMakeGood(BaseModel):
    """A raise names a campaign and nothing else that is a figure.

    The shortfall is measured by this product from the pacing board at the instant
    of the raise. A caller cannot post a goal, a counted figure or a deficit, which
    is what stops the ledger from ever holding a number nobody computed.
    """

    campaign_id: str = Field(min_length=1, max_length=64)
    note: str = Field(default="", max_length=500)


class MoveMakeGood(BaseModel):
    """One transition, with the offer fields the target state needs."""

    state: str = Field(min_length=1, max_length=32)
    offer_value: Optional[float] = None
    offer_window_start: str = Field(default="", max_length=32)
    offer_window_end: str = Field(default="", max_length=32)
    note: str = Field(default="", max_length=500)


@router.get("/api/pacing", tags=["clients"])
@PACING_WALL.guard()
def pacing_board(request: Request = None) -> dict[str, Any]:
    """Every campaign on the operator's channel, paced against its booked goal.

    Worst first. Each row carries both goal lines, the day rows behind every
    figure, and either a verdict or the named reason there is not one. Nothing
    here is projected: the counted figures are sums over the delivery ledger's own
    sourced days, and a flight day with no source is stated rather than assumed
    empty.
    """
    payload = read.board_payload()
    payload["make_goods"] = _open_index()
    return PACING_WALL.stamp(payload, request)


def _open_index() -> dict[str, list[str]]:
    """Which campaigns already carry an open make-good, so a row never offers a duplicate."""
    frame = ledger.load_frame()
    index: dict[str, list[str]] = {}
    for row_record in ledger.records(frame):
        if row_record["state"] in (ledger.SETTLED, ledger.WITHDRAWN):
            continue
        index.setdefault(row_record["campaign_id"], []).append(row_record["make_good_id"])
    return index


@router.get("/api/make-goods", tags=["clients"])
@PACING_WALL.guard()
def make_goods(request: Request = None) -> dict[str, Any]:
    """The make-good ledger: what was measured, what was offered, and who acted."""
    payload = read.ledger_payload(ledger.load_frame())
    return PACING_WALL.stamp(payload, request)


@router.post("/api/make-goods", tags=["clients"], status_code=201)
@PACING_WALL.guard()
def raise_make_good(payload: RaiseMakeGood, request: Request = None) -> dict[str, Any]:
    """Raise a make-good against a campaign's measured shortfall.

    Refused when the operator's channel holds no such campaign, when the campaign
    already has an open make-good, and when there is no measured shortfall to raise
    against. That last refusal is the important one: it is what keeps the ledger
    from holding a figure the board could not compute.
    """
    campaign_id = payload.campaign_id.strip()
    view = read.board_payload()
    row = read.find_row(view, campaign_id)
    if row is None:
        raise refuse(404, read.UNKNOWN_CAMPAIGN_EN, read.UNKNOWN_CAMPAIGN_HE)
    as_of_day = board.parse_date(view.get("as_of", {}).get("instant"))
    deficit = read.deficit_for(row, as_of_day)
    if deficit is None:
        raise refuse(409, read.NOTHING_TO_RAISE_EN, read.NOTHING_TO_RAISE_HE,
                     opens={"kind": "campaign", "id": campaign_id})

    with ledger.lock():
        frame = ledger.load_frame()
        already = ledger.open_for(frame, campaign_id)
        if already:
            raise refuse(409, DUPLICATE_EN, DUPLICATE_HE,
                         opens={"kind": "make_good", "id": already[0]})
        record_id = ledger.next_id(frame)
        new_row = _raised_row(record_id, row, view, deficit, payload.note.strip(), _actor(request))
        import pandas as pd

        frame = pd.concat([frame, pd.DataFrame([new_row])], ignore_index=True)
        ledger.write_frame(frame)
        stored = ledger.record(new_row)
    return PACING_WALL.stamp({"make_good": stored}, request)


def _raised_row(record_id: str, row: dict[str, Any], view: dict[str, Any],
                deficit: dict[str, Any], note: str, actor: str) -> dict[str, str]:
    """The ledger row a raise writes, every figure taken from the board rather than the request."""
    flight = row["flight"]
    new_row = ledger.blank_row()
    new_row.update({
        "make_good_id": record_id,
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
        "state": ledger.RAISED,
        "raised_at": ledger.now_stamp(),
        "raised_by": actor,
        "raised_note": note,
        "is_demo": "true" if row["is_demo"] else "false",
    })
    return new_row


@router.post("/api/make-goods/{make_good_id}/state", tags=["clients"])
@PACING_WALL.guard()
def move_make_good(make_good_id: str, payload: MoveMakeGood, request: Request = None) -> dict[str, Any]:
    """Move one make-good to its next state, recording who did it and when.

    A transition the state machine does not allow is refused with the states that
    are allowed from here, so a caller is told the shape of the machine rather than
    guessing at it. Settling or declining needs an offer to exist first.
    """
    target = payload.state.strip()
    actor = _actor(request)
    with ledger.lock():
        frame = ledger.load_frame()
        index = ledger.locate(frame, make_good_id)
        if index < 0:
            raise refuse(404, UNKNOWN_MAKE_GOOD_EN, UNKNOWN_MAKE_GOOD_HE)
        current = str(frame.at[index, "state"] or ledger.RAISED)
        if not ledger.transition_allowed(current, target):
            allowed = ", ".join(sorted(ledger.TRANSITIONS.get(current, frozenset()))) or "none"
            raise refuse(
                409,
                f"A make-good in {current} does not move to {target}. Allowed from here: {allowed}.",
                f"פיצוי במצב {current} אינו עובר ל{target}. מותר מכאן: {allowed}.",
            )
        _apply(frame, index, target, payload, actor)
        ledger.write_frame(frame)
        stored = ledger.record(frame.loc[index])
    return PACING_WALL.stamp({"make_good": stored}, request)


def _apply(frame: Any, index: int, target: str, payload: MoveMakeGood, actor: str) -> None:
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


@router.get("/api/make-good-alerts", tags=["insights"])
def make_good_alerts() -> dict[str, Any]:
    """At-risk campaigns from :func:`kairos.optimize.pacing.project_make_goods`.

    Data-pending: ``campaign_flights.csv`` is header-only until the owner uploads
    real flights, so ``load_campaigns`` returns ``[]`` and this returns an empty
    alert list with ``data_available: false``. It never fabricates an alert.
    """
    try:
        from kairos.optimize.pacing import load_campaigns, project_make_goods
    except Exception as exc:  # pragma: no cover - module optional
        return {"alerts": [], "data_available": False, "reason": f"Pacing module unavailable: {str(exc)[:200]}"}

    settings = _server()._load_settings()
    today = _reference_today(settings)
    campaigns = load_campaigns()
    if not campaigns:
        return {
            "alerts": [],
            "data_available": False,
            "reason": "campaign_flights.csv has no campaign rows yet (header-only seed).",
            "as_of": today.isoformat(),
        }

    at_risk = project_make_goods(campaigns, today)
    alerts = [
        {
            "campaign_id": c.campaign_id,
            "elapsed_frac": round(c.elapsed_frac, 4),
            "delivered_frac": round(c.delivered_frac, 4),
            "projected_frac": round(c.projected_frac, 4),
            "projected_shortfall": round(c.projected_shortfall, 4),
        }
        for c in at_risk
    ]
    return {"alerts": alerts, "data_available": True, "count": len(alerts), "as_of": today.isoformat()}


def _server() -> Any:
    """Lazy handle to server.py helpers (avoids an import cycle at module load)."""
    from kairos_api import server

    return server


def _reference_today(settings: Any) -> Any:
    """The reference date the pacing projection runs against (settings.effective_date)."""
    from datetime import date

    text = str(getattr(settings, "effective_date", "") or "").strip()
    parts = text.split("-")
    if len(parts) == 3 and all(p.isdigit() for p in parts):
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            pass
    return date.today()


__all__ = ["router", "words", "PACING_WALL"]
