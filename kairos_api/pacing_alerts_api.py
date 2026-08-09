"""Clients, pacing: is a campaign behind, and what is decided when it is.

Four reads and three writes, over two stores this module never edits.

``GET /api/pacing`` is the board an account manager opens in the morning: every
campaign on the operator's channel, worst pacing first, each row carrying the
verdict, the figures it was computed from, the published trigger that decided it,
and the one thing to do about it. ``GET /api/pacing/{id}/days`` is the drill
behind one row of it. ``GET`` and ``POST /api/make-goods`` are the ledger of what
is owed, and ``POST /api/make-goods/{id}/state`` moves one along.

``POST /api/pacing/{id}/accept`` is the other ending. The job this serves is done
when every at-risk campaign has an act taken against it or an explicit decision to
accept the risk, and both are recorded. Without the second one a campaign somebody
read and accepted is indistinguishable from one nobody opened, which is the half
of the job a ledger of raises alone cannot close.

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

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from kairos_api import makegood_store as ledger
from kairos_api import pacing_alerts_api_board as board
from kairos_api import pacing_alerts_api_read as read
from kairos_api import pacing_alerts_api_wire as wire
from kairos_api import pacing_alerts_api_words as words
from kairos_api import pacing_alerts_api_write as write
from kairos_api.affiliation_wall import Wall

logger = logging.getLogger(__name__)

router = APIRouter()

refuse = write.refuse

# Pacing is a run-side commercial act, so affiliation does not gate it and role
# does: any account reads the board, a write role raises and moves a make-good.
# The wall is on the read so the control tells the truth before it is pressed.
PACING_WALL = Wall(company_only=False)

DUPLICATE_EN = "This campaign already has an open make-good. Open it rather than raising a second one."
DUPLICATE_HE = "לקמפיין הזה כבר פתוח פיצוי שידור. פתחו אותו במקום לפתוח פיצוי שני."
DUPLICATE_ACCEPT_EN = "The risk on this campaign was already taken on. Open that record rather than writing a second one."
DUPLICATE_ACCEPT_HE = "הסיכון בקמפיין הזה כבר התקבל. פתחו את הרשומה הקיימת במקום לכתוב שנייה."
UNKNOWN_MAKE_GOOD_EN = "The ledger holds no record with that id."
UNKNOWN_MAKE_GOOD_HE = "ספר ההחלטות אינו מחזיק רשומה עם המזהה הזה."


def _stamp(payload: dict[str, Any], request: "Request | None") -> dict[str, Any]:
    """The wall's own ``can_edit`` answer, plus that refusal in the second language.

    ``Wall.stamp`` writes one string and the constants behind it are Hebrew only,
    so an English reader on a viewer account met a Hebrew sentence where every
    other word on the screen was English. The wall's string is left exactly as it
    is and the pair is published beside it, so a client that reads only
    ``can_edit_reason`` is unaffected and one that reads the pair can word the
    refusal in the language the reader is in.
    """
    stamped = PACING_WALL.stamp(payload, request)
    stamped.pop("can_edit_reason_en", None)
    stamped.pop("can_edit_reason_he", None)
    stamped.update(words.edit_refusal_block(stamped.get("can_edit_reason", "")))
    return stamped


def _actor(request: "Request | None") -> str:
    """Who acted, when the session says so and blank when it does not.

    The identity comes from ``affiliation_wall.session_for``, which is the
    accessor W0-4 published for exactly this. Nothing in this package writes
    ``request.state.session``, so reading that attribute returned a blank name on
    every write and the ledger recorded nobody, which is the one thing a ledger
    exists to do.
    """
    if request is None:
        return ""
    try:
        from kairos_api.affiliation_wall import session_for

        return str((session_for(request) or {}).get("username") or "").strip()
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


class AcceptRisk(BaseModel):
    """Taking a risk on names a campaign and a note, and no figure at all.

    The figures on the record are the ones the board measured at the instant of
    the decision, exactly as a raise takes them, so the two endings are recorded
    against the same numbers and can be read side by side.
    """

    note: str = Field(default="", max_length=500)


class MoveMakeGood(BaseModel):
    """One transition, with the offer fields the target state needs."""

    state: str = Field(min_length=1, max_length=32)
    offer_value: Optional[float] = None
    offer_window_start: str = Field(default="", max_length=32)
    offer_window_end: str = Field(default="", max_length=32)
    # A transition that closes a record without a delivery carries a controlled
    # reason from the ledger's own published list. The store refuses the move
    # without one, so it is optional here and required there.
    reason: str = Field(default="", max_length=64)
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
    frame = ledger.load_frame()
    payload["make_goods"] = _open_index(frame, ledger.MAKE_GOOD)
    payload["acceptances"] = _open_index(frame, ledger.ACCEPTANCE)
    # Decisions the assistant has PROPOSED and nobody has approved. A separate
    # key from the two above because it is a separate fact: these have changed
    # nothing, are owed by nobody, and must never be counted as ledger records.
    # What they are for is a row that would otherwise offer a second raise of a
    # decision already waiting for an approver.
    payload["proposed"] = _proposed_index()
    payload["needs_a_decision"] = list(words.NEEDS_A_DECISION)
    return _stamp(payload, request)


def _proposed_index() -> dict[str, Any]:
    """Pending assistant proposals per campaign, or empty when unreadable."""
    from kairos_api.assistant_pacing_propose import pending_index

    return pending_index()


def _open_index(frame: Any, kind: str) -> dict[str, list[str]]:
    """Which campaigns already carry an open record of one kind, so a row never offers a duplicate."""
    index: dict[str, list[str]] = {}
    for row_record in ledger.records(frame):
        if row_record["state"] in (ledger.SETTLED, ledger.WITHDRAWN):
            continue
        if row_record["kind"] != kind:
            continue
        index.setdefault(row_record["campaign_id"], []).append(row_record["make_good_id"])
    return index


@router.get("/api/pacing/{campaign_id}/days", tags=["clients"])
@PACING_WALL.guard()
def pacing_days(campaign_id: str, request: Request = None) -> dict[str, Any]:
    """The broadcast days behind one campaign's figures, read when a reader opens them.

    The board is a list somebody triages and the days are the drill behind one row
    of it, so the days ride their own read. Measured on the shipped data they were
    144 KB of a 366 KB board payload, and they are the one term that grows as
    campaigns times flight days.
    """
    payload = read.days_payload(campaign_id)
    if payload is None:
        raise refuse(404, read.UNKNOWN_CAMPAIGN_EN, read.UNKNOWN_CAMPAIGN_HE)
    return _stamp(payload, request)


@router.get("/api/make-goods", tags=["clients"])
@PACING_WALL.guard()
def make_goods(request: Request = None) -> dict[str, Any]:
    """The make-good ledger: what was measured, what was offered, and who acted."""
    payload = read.ledger_payload(ledger.load_frame())
    return _stamp(payload, request)


@router.post("/api/make-goods", tags=["clients"], status_code=201)
@PACING_WALL.guard()
def raise_make_good(payload: RaiseMakeGood, request: Request = None) -> dict[str, Any]:
    """Raise a make-good against a campaign's measured shortfall.

    Refused when the operator's channel holds no such campaign, when the campaign
    already has an open make-good, and when there is no measured shortfall to raise
    against. That last refusal is the important one: it is what keeps the ledger
    from holding a figure the board could not compute.
    """
    return _write_decision(ledger.MAKE_GOOD, payload.campaign_id.strip(), payload.note.strip(),
                           _actor(request), request)


@router.post("/api/pacing/{campaign_id}/accept", tags=["clients"], status_code=201)
@PACING_WALL.guard()
def accept_risk(campaign_id: str, payload: AcceptRisk, request: Request = None) -> dict[str, Any]:
    """Record the decision that the risk on this campaign stands as it is.

    This is the second way a row on the board is finished with. It changes no
    figure and reserves nothing: the campaign keeps its verdict and its place in
    the order, and the row now carries who decided and when. Refused on a campaign
    the board is not asking a decision about, because accepting a risk that was
    never stated is not a thing a person can mean.
    """
    return _write_decision(ledger.ACCEPTANCE, campaign_id.strip(), payload.note.strip(),
                           _actor(request), request)


DUPLICATES = {
    ledger.MAKE_GOOD: (DUPLICATE_EN, DUPLICATE_HE, "make_good"),
    ledger.ACCEPTANCE: (DUPLICATE_ACCEPT_EN, DUPLICATE_ACCEPT_HE, "acceptance"),
}

# Why a raise was refused, in the words the board publishes for the same rule. A
# row that reaches no rung at all and a row that reaches only the gap to date are
# two different facts and they send the reader to two different places.
REFUSED_RAISE = {
    "nothing_measured": (read.NOTHING_TO_RAISE_EN, read.NOTHING_TO_RAISE_HE),
    "not_owed_yet": (
        f"{words.NOT_OWED_YET_EN} {words.NOT_OWED_YET_PATH_EN}",
        f"{words.NOT_OWED_YET_HE} {words.NOT_OWED_YET_PATH_HE}",
    ),
}


def _write_decision(kind: str, campaign_id: str, note: str, actor: str,
                    request: "Request | None" = None) -> dict[str, Any]:
    """One act on one campaign, measured from the board and written to the ledger.

    Both acts take the same route through this function so that neither can ever
    stamp a figure the other would not have. What differs between them is only
    which rows they are allowed on, which is decided by the reader below.

    ``actor`` is passed rather than derived here so a second caller can attribute
    the row to somebody the request does not name. The assistant's action plane
    is that caller: an approved proposal reaches this function with no HTTP
    request at all, and deriving the actor internally recorded a blank name, which
    is the one thing this ledger exists not to do. It is a parameter and NOT a
    field on the request models on purpose: the routes are the only public door,
    they pass the session's own answer, and a caller therefore cannot post an
    actor of its choosing.
    """
    # The board a write measures from is the full shape, never the wire's
    # collapsed one, so an act is never decided on a row a reader could not read.
    view = wire.expand(read.board_payload())
    row = read.find_row(view, campaign_id)
    if row is None:
        raise refuse(404, read.UNKNOWN_CAMPAIGN_EN, read.UNKNOWN_CAMPAIGN_HE)
    as_of_day = board.parse_date(view.get("as_of", {}).get("instant"))
    if kind == ledger.ACCEPTANCE:
        deficit = read.acceptance_figures(row, as_of_day)
        refusal = (words.ACCEPT_NOT_AT_RISK_EN, words.ACCEPT_NOT_AT_RISK_HE)
    else:
        # One rule for one act. A gap to date is a measured figure and not a
        # debt, so it is refused here in the same words the board publishes and
        # with the path that makes the raise available.
        deficit, why = read.raisable_deficit(row, as_of_day)
        refusal = REFUSED_RAISE.get(why, (read.NOTHING_TO_RAISE_EN, read.NOTHING_TO_RAISE_HE))
    if deficit is None:
        raise refuse(409, refusal[0], refusal[1], opens={"kind": "campaign", "id": campaign_id})

    duplicate_en, duplicate_he, opens_kind = DUPLICATES[kind]
    with ledger.lock():
        frame = ledger.load_frame()
        already = ledger.open_for(frame, campaign_id, kind)
        if already:
            raise refuse(409, duplicate_en, duplicate_he, opens={"kind": opens_kind, "id": already[0]})
        record_id = ledger.next_id(frame)
        fresh = write.new_row(record_id, kind, row, view, deficit, note, actor)
        import pandas as pd

        frame = pd.concat([frame, pd.DataFrame([fresh])], ignore_index=True)
        ledger.write_frame(frame)
        stored = ledger.record(fresh)
    # The answer to an act names how to reverse the act. Both writes here can be
    # fired by a single keystroke on a focused row, so the confirmation of one
    # has to be able to offer the undo without the surface holding its own copy
    # of which transition and which reason an undo is.
    return _stamp({"make_good": stored, "undo": ledger.undo_block()}, request)


@router.post("/api/make-goods/{make_good_id}/state", tags=["clients"])
@PACING_WALL.guard()
def move_make_good(make_good_id: str, payload: MoveMakeGood, request: Request = None) -> dict[str, Any]:
    """Move one ledger record to its next state, recording who did it and when.

    A transition the state machine does not allow is refused with the states that
    are allowed from here, so a caller is told the shape of the machine rather than
    guessing at it. Settling or declining needs an offer to exist first, and an
    accepted risk moves only to withdrawn because it was never an offer.
    """
    return _move_decision(make_good_id, payload, _actor(request), request)


def _move_decision(make_good_id: str, payload: MoveMakeGood, actor: str,
                   request: "Request | None" = None) -> dict[str, Any]:
    """The transition itself, with the acting account passed in.

    Split from the route for the reason :func:`_write_decision` records: the
    assistant's action plane applies an approved move with no HTTP request, and
    the row has to name whoever approved it rather than nobody.
    """
    target = payload.state.strip()
    with ledger.lock():
        frame = ledger.load_frame()
        index = ledger.locate(frame, make_good_id)
        if index < 0:
            raise refuse(404, UNKNOWN_MAKE_GOOD_EN, UNKNOWN_MAKE_GOOD_HE)
        current = str(frame.at[index, "state"] or ledger.RAISED)
        kind = str(frame.at[index, "kind"] or ledger.MAKE_GOOD)
        if not ledger.transition_allowed(current, target):
            raise write.refuse_transition(current, target, kind)
        write.apply_move(frame, index, target, payload, actor)
        ledger.write_frame(frame)
        stored = ledger.record(frame.loc[index])
    return _stamp({"make_good": stored}, request)


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


__all__ = ["router", "words", "write", "PACING_WALL"]
