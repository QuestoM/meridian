"""The states the day board reports instead of a number it cannot read.

Split out of :mod:`kairos_api.break_api` under the 450-line cap, and it is a
coherent module rather than a leftover: everything here is an answer the product
gives when the honest answer is not a figure.

* **No day to open.** No operator channel, or no saved plan on it. That is a
  state of the deployment, not a fault of the request, so the route answers 200
  with the reason and empty collections, exactly as ``/api/plan/days``,
  ``/api/schedule/segments`` and ``/api/gold-breaks`` already do.
* **Delivered money.** The saved plan covers 2024-11-01 to 2024-11-30 and the one
  daily spot file covers 2025-04-27, so no planned break has a ledger behind it.
  Delivered is therefore tri-state and never carries an amount it did not read.
* **The restrictions in force.** Read only, so a refused move explains itself
  without anybody opening documentation. Authoring belongs to Rules.
* **A saved placement the plan no longer binds.** The one below that is about
  money rather than about a missing input, and it is the reason this module grew.

Every reason travels in both languages. An honest empty state a Hebrew operator
reads in English is only half honest, and the machine ``state`` is what a surface
should branch on in any case.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

NO_CHANNEL = "No operator channel is set, so there is no day to open. Set it in account settings."
NO_CHANNEL_HE = "לא הוגדר ערוץ למפעיל, ולכן אין יום לפתיחה. קבעו אותו בהגדרות החשבון."
NO_PLAN = "No saved weekly plan covers this channel, so there is no day to open."
NO_PLAN_HE = "אין תוכנית שבועית שמורה לערוץ הזה, ולכן אין יום לפתיחה."

# A gold mark that reached nothing. The English half of the refused case is the
# engine's own verbatim reason, so only the Hebrew half is written here.
GOLD_REFUSED_HE = "הסימון נשמר אך המנוע דחה אותו, והנימוק שלו מופיע באנגלית לצד ההודעה."
GOLD_UNMARKED = "The mark is stored, but the plan came back with no gold break in this programme."
GOLD_UNMARKED_HE = "הסימון נשמר, אך התוכנית חזרה בלי ברייק זהב בתוכנית הזו."


def no_day_payload(channel: str) -> dict[str, Any]:
    """There is no day to open, said as a state and never as a figure.

    Every collection is empty and every total is absent rather than zero, because
    a zero is a measurement and this is the absence of one.
    """
    return {
        "available": False,
        "reason": NO_PLAN if channel else NO_CHANNEL,
        "reason_he": NO_PLAN_HE if channel else NO_CHANNEL_HE,
        "operator_channel": channel or None,
        "day": None,
        "basis": None,
        "programmes": [],
        "breaks": [],
        "unbound_placements": [],
        "totals": None,
        "compliance": None,
        "hours": [],
        "restrictions": {"count": 0, "items": [], "skipped": []},
        "gold": None,
        "guardrails": None,
    }


def delivered_state(day: str) -> dict[str, Any]:
    """Whether a spot ledger covers this day, so delivered money can be real.

    Tri-state and never a figure when it is not one: ``real`` only when a daily
    file covers the day, ``unavailable`` with the coverage named when it does
    not, and ``unknown`` when the daily folder cannot be read at all.
    """
    from kairos.export.spots_coverage import daily_input_days

    try:
        covered = daily_input_days()
    except Exception:  # noqa: BLE001 - an unreadable folder is a state, not a crash
        logger.exception("daily input coverage read failed")
        return {
            "state": "unknown",
            "amount": None,
            "reason": "The daily spot folder could not be read.",
            "reason_he": "לא ניתן לקרוא את תיקיית התשדירים היומיים.",
        }
    wanted = str(day or "").strip()
    if wanted and wanted in covered:
        return {"state": "real", "amount": None, "covered_by": sorted(covered), "reason": "", "reason_he": ""}
    return {
        "state": "unavailable",
        "amount": None,
        "covered_by": sorted(covered),
        "reason": "No spot ledger covers this broadcast day.",
        "reason_he": "אין יומן תשדירים המכסה את יום השידור הזה.",
        # NAMES AN UPLOAD THIS PRODUCT ACTUALLY ACCEPTS. This used to say "supply
        # a delivery or as-run feed through Sources", and Sources has no as-run
        # kind: its seven inputs are programmes, daily, spots, dayparts,
        # advertiser_rules, rate_card and campaign_flights. An operator following
        # that sentence reached the upload screen and found nothing to do. The
        # daily ad log is the delivery record this engine reads today -- it is
        # what `daily_input_days()` above is looking for -- so it is what the
        # sentence names. A real As Run feed is a separate, later integration; the
        # instruction must not promise it before it exists.
        "path_forward": "Upload the daily ad log covering this day through Sources.",
        "path_forward_he": "העלו דרך מקורות את קובץ הפרסומות היומי המכסה את היום הזה.",
    }


UNBOUND_FORWARD = "Remove the saved placement to delete the restriction that carries it and let the plan place these breaks itself."
UNBOUND_FORWARD_HE = "הסירו את הנעיצה השמורה כדי למחוק את המגבלה שנושאת אותה ולתת לתוכנית למקם את הברייקים בעצמה."
RESTRICTION_IN_FORCE = "The restriction this record wrote is still in force on this day."
RESTRICTION_IN_FORCE_HE = "המגבלה שהרשומה הזו כתבה עדיין בתוקף ביום הזה."
RESTRICTION_ABSENT = "The restriction this record names is no longer in the store, so only the record is left."
RESTRICTION_ABSENT_HE = "המגבלה שהרשומה מציינת כבר אינה במאגר, ולכן נותרה רק הרשומה."
RESTRICTION_UNKNOWN = "The restriction store could not be read, so whether it is still in force is unknown."
RESTRICTION_UNKNOWN_HE = "לא ניתן לקרוא את מאגר המגבלות, ולכן לא ידוע אם היא עדיין בתוקף."
SEGMENT_ABSENT = "This programme is not in the plan for this day, so the record names a break that does not exist."
SEGMENT_ABSENT_HE = "רצועת השידור הזו אינה בתוכנית ליום הזה, ולכן הרשומה מציינת ברייק שאינו קיים."


def _replanned(here: int, ordinal: int) -> str:
    breaks = "1 break" if here == 1 else f"{here} breaks"
    return f"The plan now places {breaks} in this programme, and this record names break {ordinal}."


def _replanned_he(here: int, ordinal: int) -> str:
    breaks = "ברייק אחד" if here == 1 else f"{here} ברייקים"
    return f"התוכנית מציבה כעת {breaks} ברצועת השידור הזו, והרשומה הזו מציינת ברייק {ordinal}."


def _number(value: Any) -> Any:
    """A stored figure back as a number, or None. Never a zero it did not read."""
    try:
        return round(float(str(value).strip()), 1)
    except (TypeError, ValueError):
        return None


def _flag(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def _restriction_state(constraint_id: str) -> dict[str, Any]:
    """Whether the restriction a record names is still in the store. Tri-state."""
    wanted = str(constraint_id or "").strip()
    if not wanted:
        return {"state": "absent", "reason": RESTRICTION_ABSENT, "reason_he": RESTRICTION_ABSENT_HE}
    try:
        from kairos_api.overrides import _stored_constraints

        present = any(str(row.constraint_id) == wanted for row in _stored_constraints())
    except Exception:  # noqa: BLE001 - an unreadable store is a state, not a crash
        logger.exception("restriction lookup failed for %s", wanted)
        return {"state": "unknown", "reason": RESTRICTION_UNKNOWN, "reason_he": RESTRICTION_UNKNOWN_HE}
    if present:
        return {"state": "in_force", "reason": RESTRICTION_IN_FORCE, "reason_he": RESTRICTION_IN_FORCE_HE}
    return {"state": "absent", "reason": RESTRICTION_ABSENT, "reason_he": RESTRICTION_ABSENT_HE}


def unbound_placements(plan: Any, saved: dict[str, Any], breaks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The day's saved placements that no longer name a break the plan carries.

    Measured on ``רשת 13 / 2024-11-01``, and this is the hole it closes. Pin
    ``001~2`` one snap unit to the right and save: the engine re-plans that
    programme from four breaks down to one, the day falls from 1,067,845.55 to
    1,020,401.35, and the id ``001~2`` stops existing. Every route back was on the
    chip that carried the id, so a reload left 47,444.20 ILS spent with the record
    and the restriction both still on disk and nothing on the surface to point at.

    Identity is ``segment_id + ordinal``, which the contract's section 9 raises as
    an identity nuisance; it is not, it is the hole under the reversibility claim.
    So the partition here is exact: a record whose break id is in the plan is
    reachable on its chip, a record whose break id is not is returned here, and
    every saved record is in exactly one of the two. The surface offers one
    inverse per record, never two and never none.
    """
    live = {record["break_id"] for record in breaks}
    counts: dict[str, int] = {}
    for record in breaks:
        counts[record["segment_id"]] = counts.get(record["segment_id"], 0) + 1
    rows: list[dict[str, Any]] = []
    for break_id, record in sorted(saved.items()):
        if break_id in live:
            continue
        segment_id = str(record.get("segment_id", "")).strip()
        ordinal = _number(record.get("ordinal")) or 0
        here = counts.get(segment_id, 0)
        in_plan = plan.segment(segment_id) is not None
        rows.append({
            "break_id": break_id,
            "segment_id": segment_id,
            "ordinal": int(ordinal),
            "constraint_id": str(record.get("constraint_id", "")).strip(),
            "programme": str(record.get("programme", "")),
            "channel": str(record.get("channel", "")),
            "day": str(record.get("day", "")),
            "offset_seconds": _number(record.get("offset_seconds")),
            "duration_seconds": _number(record.get("duration_seconds")),
            "is_gold": _flag(record.get("is_gold")),
            "actor": str(record.get("actor", "")),
            "saved_at": str(record.get("saved_at", "")),
            "note": str(record.get("note", "")),
            "state": "segment_replanned" if in_plan else "segment_absent",
            "breaks_in_segment": here,
            "restriction": _restriction_state(record.get("constraint_id", "")),
            "reason": _replanned(here, int(ordinal)) if in_plan else SEGMENT_ABSENT,
            "reason_he": _replanned_he(here, int(ordinal)) if in_plan else SEGMENT_ABSENT_HE,
            "path_forward": UNBOUND_FORWARD,
            "path_forward_he": UNBOUND_FORWARD_HE,
        })
    return rows


def restrictions_for(plan: Any) -> dict[str, Any]:
    """The stored placement restrictions that bind this day, in plain terms."""
    try:
        from kairos.optimize.constraints_store import resolve_constraints

        from kairos_api.overrides import _stored_constraints

        stored = _stored_constraints()
        if not stored:
            return {"count": 0, "items": [], "skipped": []}
        pins, count_pins, forbids, skipped = resolve_constraints(
            list(plan.segments), stored, operator_channel=plan.channel,
        )
    except Exception:  # noqa: BLE001 - an unreadable store is a state, not a crash
        logger.exception("restriction resolve failed")
        return {"count": 0, "items": [], "skipped": [], "reason": "The restriction store could not be read."}
    items = [
        {"segment_id": segment_id, "effect": "fixed placement", "breaks": len(rows)}
        for segment_id, rows in sorted(pins.items())
    ] + [
        {"segment_id": segment_id, "effect": "fixed break count", "count": count}
        for segment_id, count in sorted(count_pins.items())
    ] + [
        {"segment_id": segment_id, "effect": "no breaks here"}
        for segment_id in sorted(forbids)
    ]
    return {
        "count": len(items),
        "items": items,
        "skipped": [
            {"constraint_id": row.constraint_id, "segment_id": row.segment_id, "reason": row.reason}
            for row in skipped
        ],
    }
