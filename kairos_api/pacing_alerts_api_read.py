"""Clients, pacing: the reads the board stands on, and the deficit it may raise.

The glue between two stores the account manager already owns and the arithmetic in
:mod:`kairos_api.pacing_alerts_api_board`. It reads and never writes them, because
the campaign store and the delivery ledger belong to the clients destination and a
pacing view that edited them would be a second writer on one record.

The competitor boundary is applied here rather than at the router, so every caller
of a pacing read inherits it and cannot forget to. The operator owns exactly one
channel, taken from settings, and a campaign on any other channel never reaches a
row, a count or a make-good.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from kairos_api import channel_scope
from kairos_api import makegood_store as ledger
from kairos_api import pacing_alerts_api_board as board
from kairos_api import pacing_alerts_api_words as words

NOTHING_TO_RAISE_EN = (
    "This campaign has no measured shortfall to raise a make-good against. Its pacing row states what "
    "is missing before a figure could exist."
)
NOTHING_TO_RAISE_HE = (
    "לקמפיין הזה אין חוסר נמדד שאפשר לפתוח מולו פיצוי שידור. שורת הקצב שלו אומרת מה חסר לפני שיכול "
    "להיות נתון."
)
UNKNOWN_CAMPAIGN_EN = "No campaign on this operator's channel carries that id."
UNKNOWN_CAMPAIGN_HE = "אין קמפיין בערוץ של המפעיל הזה שנושא את המזהה הזה."


def _campaigns() -> list[dict[str, Any]]:
    from kairos_api import campaigns_api_store as store

    return store.campaigns_with_flights(store.load_frame())


def _delivery() -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """The delivery ledger grouped by campaign, and the instant it was counted at.

    Both come off the ledger's own public frame. The instant is the newest stamp
    any row carries and the basis is the sentence the ledger writes beside it, so
    this board dates itself by the ledger rather than by the reader's clock.
    """
    from kairos_api import campaigns_delivery as delivery

    frame = delivery.load_frame()
    stamps = [str(value).strip() for value in frame.get("counted_as_of", []) if str(value).strip()]
    bases = [str(value).strip() for value in frame.get("counted_as_of_basis", []) if str(value).strip()]
    return delivery.days_by_campaign(), {
        "instant": max(stamps) if stamps else "",
        "basis": bases[0] if bases else "",
    }


def board_payload() -> dict[str, Any]:
    """Every campaign on the operator's channel, paced, worst first, with its basis.

    ``as_of`` is the instant the delivery ledger split aired from scheduled at, and
    it is the only clock this board reads. Nothing here calls the wall clock,
    because a pacing figure dated by the reader's machine rather than by the
    ledger it was counted from is a figure nobody can reproduce.
    """
    campaigns, scope = channel_scope.scope_records(_campaigns(), key="channel")
    grouped, as_of = _delivery()
    as_of_day = board.parse_date(as_of.get("instant"))
    rows = board.build_rows(campaigns, grouped, as_of_day)
    sourced = sum(1 for row in rows if row.get("flight") and row["flight"]["days_sourced"])
    marking = board.collapse_demo(rows)
    return {
        "available": bool(rows),
        "rows": rows,
        "counts": board.counts(rows),
        "campaigns_with_a_source": sourced,
        "demo_marking": marking,
        "as_of": as_of,
        "scope": scope,
        "trigger": words.trigger_block(),
        "counted_basis_en": words.COUNTED_BASIS_EN,
        "counted_basis_he": words.COUNTED_BASIS_HE,
        "no_source_en": words.NO_SOURCE_EN,
        "no_source_he": words.NO_SOURCE_HE,
        "path_forward_en": words.NO_SOURCE_PATH_EN,
        "path_forward_he": words.NO_SOURCE_PATH_HE,
        "vocabulary": words.vocabularies(),
    }


def days_payload(campaign_id: str) -> Optional[dict[str, Any]]:
    """The broadcast days behind one campaign's figures, or None off the operator's channel.

    The drill behind a board row, read on demand. It goes through the same scope
    the board does, so a campaign on a rival channel has no day read either and the
    boundary is not something a second caller could forget to apply.
    """
    campaigns, _ = channel_scope.scope_records(_campaigns(), key="channel")
    match = next((one for one in campaigns if str(one.get("campaign_id", "")) == str(campaign_id)), None)
    if match is None:
        return None
    grouped, as_of = _delivery()
    days = grouped.get(str(campaign_id), [])
    sources = sorted({str(day.get("source_file") or "") for day in days if day.get("source_file")})
    return {
        "campaign_id": str(campaign_id),
        "days": days,
        "count": len(days),
        "sources": sources,
        "as_of": as_of,
    }


def find_row(payload: dict[str, Any], campaign_id: str) -> Optional[dict[str, Any]]:
    """One board row by campaign id, or None when the operator's channel holds no such campaign."""
    for row in payload.get("rows", []):
        if row.get("campaign_id") == str(campaign_id):
            return row
    return None


def _line_for(row: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The goal line a make-good would be raised in: the rating goal, or the money goal."""
    rating = row.get("rating")
    if rating and rating.get("goal") is not None and rating["pace"].get("code") != "unmeasurable":
        return rating
    money = row.get("money")
    if money and money.get("goal") is not None:
        return money
    return None


def deficit_for(row: dict[str, Any], as_of_day: Optional[date]) -> Optional[dict[str, Any]]:
    """The measured shortfall a make-good may be raised against, or None with no invention.

    Three ladders, most certain first, and the row takes the first one that holds.
    A campaign that reaches none of them has no measured shortfall, and the caller
    refuses the raise rather than recording a figure nobody computed.
    """
    line = _line_for(row)
    flight = row.get("flight")
    if line is None or flight is None:
        return None
    goal = float(line["goal"])
    counted = line["counted"]
    closed = bool(as_of_day and board.parse_date(flight["ends_on"]) and as_of_day >= board.parse_date(flight["ends_on"]))
    unsourced = len(flight["unsourced_elapsed_days"]) + len(flight["unsourced_remaining_days"])
    booked = float(counted["booked_total"])

    if closed and unsourced == 0 and booked < goal:
        return _deficit(line, goal, booked, goal - booked, ledger.MEASURED_CLOSED, unsourced)
    if line["forward"]["state"] == words.SHORT_CERTAIN:
        remaining = float(line["forward"]["remaining_to_goal"] or 0.0)
        if remaining > 0:
            return _deficit(line, goal, booked, remaining, ledger.BOOKED_SHORT, unsourced)
    gap = line["pace"].get("gap_to_reference")
    if line["pace"]["verdict"] in (words.BEHIND, words.AT_RISK) and gap:
        through = float(counted["through_counted_day"])
        return _deficit(line, goal, through, float(gap), ledger.TO_DATE, unsourced)
    return None


def _deficit(line: dict[str, Any], goal: float, counted: float, deficit: float,
             kind: str, unsourced: int) -> dict[str, Any]:
    return {
        "unit": line["unit"],
        "goal_value": round(goal, 2),
        "counted_value": round(counted, 2),
        "deficit_value": round(deficit, 2),
        "deficit_kind": kind,
        "unsourced_days": unsourced,
    }


def acceptance_figures(row: dict[str, Any], as_of_day: Optional[date]) -> Optional[dict[str, Any]]:
    """The measured state an acceptance stamps, or None when the row is not one to accept.

    Only a row the board is asking a decision about may have its risk taken on,
    and the record carries the same measured figures a make-good would have been
    raised against. Nothing is derived for it and nothing is softened: taking a
    risk on is a decision about a number, so the number goes on the record.
    """
    if row.get("headline", {}).get("verdict") not in words.NEEDS_A_DECISION:
        return None
    return deficit_for(row, as_of_day)


def ledger_payload(frame: Any) -> dict[str, Any]:
    """The decision ledger as the API reports it, with what it does not decide beside it.

    Both endings ride one read because they are one ledger. ``make_goods`` is kept
    as its own list because that is the shape this piece published and a consumer
    of it must not have to learn a new one to keep working.
    """
    rows = ledger.records(frame)
    live = [row for row in rows if row["state"] not in (ledger.SETTLED, ledger.WITHDRAWN)]
    made = [row for row in rows if row["kind"] == ledger.MAKE_GOOD]
    accepted = [row for row in rows if row["kind"] == ledger.ACCEPTANCE]
    return {
        "available": True,
        "decisions": rows,
        "make_goods": made,
        "acceptances": accepted,
        "count": len(rows),
        "open_count": sum(1 for row in live if row["kind"] == ledger.MAKE_GOOD),
        "accepted_count": sum(1 for row in live if row["kind"] == ledger.ACCEPTANCE),
        "acceptance_means_en": words.ACCEPT_MEANING_EN,
        "acceptance_means_he": words.ACCEPT_MEANING_HE,
        "sign_off": ledger.sign_off_block(),
        "vocabulary": ledger.vocabularies(),
    }
