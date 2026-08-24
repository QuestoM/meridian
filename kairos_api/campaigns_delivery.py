"""What a campaign actually aired, where, and what is still ahead of it.

``data/campaign_delivery.csv`` is a derived ledger, not a booking. One row per
campaign per broadcast day, and each row is in exactly one of three states:

``aired``
    The traffic log on disk records these spots and their time is before the
    counted-as-of instant. The figures are real and the row names the file.

``scheduled``
    The traffic log records these spots and their time is at or after the
    counted-as-of instant, so on that day they are still to come.

``unknown``
    The day falls inside the campaign's flight and this product has no per-spot
    source for it. The figures are blank, never zero. A zero here would tell the
    reader the campaign delivered nothing that day, which is a claim nobody
    measured.

Because ``unknown`` days exist, every counted figure is reported as a **floor**
and says so: at least this much has been counted, and this many days of the
flight carry no source at all. A percentage that silently divides by a partial
count is the exact fabrication this module refuses to make.

Two more honest boundaries ride on every payload.

**Rating points here are the planned break rating** the traffic log carries for
the break each spot sits in, on the all-viewers base. They are not a post-campaign
panel report of what was delivered, because no such report is on disk. A campaign
whose goal names any other audience reports its progress as unknown rather than
comparing two different currencies.

**Money here is engine-priced, not invoiced.** It is the same per-spot ledger the
money board, the agency summary and the spots export all read, so a figure here
can never disagree with the same figure there. Nothing in this product invoices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from kairos_api import campaigns_commitment as commitment

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DELIVERY_PATH = DATA_DIR / "campaign_delivery.csv"

AIRED = "aired"
SCHEDULED = "scheduled"
UNKNOWN = "unknown"

COLUMNS = [
    "campaign_id",
    "broadcast_date",
    "air_state",
    "channel",
    "spots",
    "seconds",
    "rating_points_planned",
    "spend_ils",
    "spots_dropped_by_rule",
    "dropped_rule_id",
    "figures_basis",
    "source_file",
    "counted_as_of",
    "counted_as_of_basis",
    "is_demo",
    "note",
]

AIR_STATE_VOCABULARY = (
    {
        "value": AIRED,
        "label_en": "Aired",
        "label_he": "שודר",
        "meaning_en": "The traffic log records these spots and their time has passed.",
        "meaning_he": "יומן השידור רושם את התשדירים האלה והשעה שלהם עברה.",
    },
    {
        "value": SCHEDULED,
        "label_en": "Scheduled, not aired yet",
        "label_he": "מתוזמן, טרם שודר",
        "meaning_en": "The traffic log records these spots and their time is still ahead.",
        "meaning_he": "יומן השידור רושם את התשדירים האלה והשעה שלהם עוד לפנינו.",
    },
    {
        "value": UNKNOWN,
        "label_en": "Unknown",
        "label_he": "לא ידוע",
        "meaning_en": "This day is inside the flight and no per-spot source exists for it.",
        "meaning_he": "היום הזה נמצא בתוך הטיסה ואין עבורו מקור ברמת התשדיר.",
    },
)

FLOOR_EN = (
    "This is a floor, not a total. It counts only the broadcast days this product holds a per-spot "
    "source for. Days with no source are listed as unknown and are not counted as zero."
)
FLOOR_HE = (
    "זהו רף תחתון ולא סכום. הוא סופר רק את ימי השידור שיש למערכת עבורם מקור ברמת התשדיר. ימים בלי "
    "מקור מופיעים כלא ידועים ואינם נספרים כאפס."
)
RATING_BASIS_EN = (
    "Planned break rating from the traffic log, on the all-viewers base. Not a post-campaign panel "
    "report of delivered rating points."
)
RATING_BASIS_HE = (
    "רייטינג ברייקים מתוכנן מתוך יומן השידור, על בסיס כלל הצופים. אינו דוח פאנל של נקודות רייטינג "
    "שסופקו בפועל."
)
SPEND_BASIS_EN = (
    "Engine-priced from the same per-spot ledger the money board reads. Nothing here is invoiced."
)
SPEND_BASIS_HE = (
    "מתומחר במנוע מאותו ספר תשדירים שלוח הכספים קורא. דבר מכאן אינו מחויב בחשבונית."
)
NO_SOURCE_EN = "No delivery row matches any campaign on this payload, so what aired is unknown."
NO_SOURCE_HE = "אין שורת אספקה שמתאימה לקמפיין כלשהו בתשובה הזו, ולכן מה ששודר אינו ידוע."
NO_SOURCE_PATH_EN = (
    "Upload a daily traffic file for the flight days, then run scripts/seed_campaigns.py to rebuild "
    "the delivery ledger from it."
)
NO_SOURCE_PATH_HE = (
    "העלו קובץ שידור יומי לימי הטיסה, ואז הריצו את scripts/seed_campaigns.py כדי לבנות מחדש את ספר "
    "האספקה ממנו."
)
GOAL_UNMEASURABLE_EN = (
    "The goal names a target audience this product has no panel breakdown for, so progress against "
    "it is unknown rather than measured against a different base."
)
GOAL_UNMEASURABLE_HE = (
    "היעד נוקב בקהל יעד שאין למערכת עבורו פילוח פאנל, ולכן ההתקדמות מולו אינה ידועה ואינה נמדדת מול "
    "בסיס אחר."
)
NO_GOAL_EN = "This campaign carries no goal in this unit, so there is nothing to measure against."
NO_GOAL_HE = "הקמפיין הזה אינו נושא יעד ביחידה הזו, ולכן אין מול מה למדוד."


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


def load_frame() -> pd.DataFrame:
    """Every delivery row, or an empty frame when the ledger has never been written."""
    if not DELIVERY_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(DELIVERY_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def day_record(row: Any) -> dict[str, Any]:
    """One broadcast day of one campaign, with its state and its basis."""
    state = _text(row, "air_state") or UNKNOWN
    known = state in (AIRED, SCHEDULED)
    return {
        "broadcast_date": _text(row, "broadcast_date"),
        "air_state": state,
        "channel": _text(row, "channel"),
        "spots": int(_number(row.get("spots")) or 0) if known else None,
        "seconds": _number(row.get("seconds")) if known else None,
        "rating_points_planned": _number(row.get("rating_points_planned")) if known else None,
        "spend_ils": _number(row.get("spend_ils")) if known else None,
        "spots_dropped_by_rule": int(_number(row.get("spots_dropped_by_rule")) or 0) if known else None,
        "dropped_rule_id": _text(row, "dropped_rule_id"),
        "figures_basis": _text(row, "figures_basis"),
        "source_file": _text(row, "source_file"),
        "note": _text(row, "note"),
        "is_demo": commitment.is_demo(row.get("is_demo")),
    }


def days_by_campaign() -> dict[str, list[dict[str, Any]]]:
    """Every delivery day, grouped by campaign and ordered as a broadcast week runs."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, row in load_frame().iterrows():
        record = day_record(row)
        grouped.setdefault(_text(row, "campaign_id"), []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: item["broadcast_date"])
    return grouped


def _as_of(frame: pd.DataFrame) -> dict[str, Any]:
    """The instant the aired and scheduled split was taken at, and where it came from."""
    stamps = [str(value).strip() for value in frame.get("counted_as_of", []) if str(value).strip()]
    bases = [str(value).strip() for value in frame.get("counted_as_of_basis", []) if str(value).strip()]
    return {
        "instant": max(stamps) if stamps else "",
        "basis": bases[0] if bases else "",
    }


def _totals(days: list[dict[str, Any]], state: str) -> dict[str, Any]:
    """One state's figures. ``days`` counts rows in this state, not calendar days.

    One broadcast day can carry an aired row and a scheduled row at once, which
    is the normal state of the day an operator is standing in. So these counts
    do not add up to the flight length and are not meant to: ``sourced_days`` and
    ``flight_days`` on the block above are the unambiguous denominators.
    """
    rows = [day for day in days if day["air_state"] == state]
    return {
        "days": len(rows),
        "spots": sum(day["spots"] or 0 for day in rows),
        "seconds": round(sum(day["seconds"] or 0.0 for day in rows), 2),
        "rating_points_planned": round(sum(day["rating_points_planned"] or 0.0 for day in rows), 4),
        "spend_ils": round(sum(day["spend_ils"] or 0.0 for day in rows), 2),
        "spots_dropped_by_rule": sum(day["spots_dropped_by_rule"] or 0 for day in rows),
    }


def _progress(counted: Optional[float], goal: Optional[float], *, measurable: bool,
              sourced: bool, unmeasurable_reason: tuple[str, str]) -> dict[str, Any]:
    """One goal's progress in the tri-state: measured, unavailable, or unknown.

    A percentage is published only when there is a goal to divide by, the
    product can count in the goal's own currency, and at least one day of the
    flight has a source at all. That last gate is the subtle one: with no
    source, ``counted`` is zero and the division yields a confident
    ``0%``, which tells the reader the campaign has delivered nothing when what
    is true is that nobody knows. Every other case returns ``None`` with the
    reason, so no surface can draw a bar out of a number nobody computed.
    """
    if not sourced:
        return {"percent": None, "state": UNKNOWN, "reason_en": NO_SOURCE_EN, "reason_he": NO_SOURCE_HE}
    if goal is None:
        return {"percent": None, "state": UNKNOWN, "reason_en": NO_GOAL_EN, "reason_he": NO_GOAL_HE}
    if not measurable:
        return {
            "percent": None,
            "state": "unavailable",
            "reason_en": unmeasurable_reason[0],
            "reason_he": unmeasurable_reason[1],
        }
    if goal <= 0:
        return {"percent": None, "state": UNKNOWN, "reason_en": NO_GOAL_EN, "reason_he": NO_GOAL_HE}
    return {
        "percent": round(100.0 * float(counted or 0.0) / float(goal), 2),
        "state": "floor",
        "reason_en": FLOOR_EN,
        "reason_he": FLOOR_HE,
    }


def booking_rule_sentences(days_by_campaign_index: dict[str, list[dict[str, Any]]]
                           ) -> dict[str, dict[str, Any]]:
    """What each rule the ledger named actually capped, composed once per read.

    The ledger names the rule that left spots out of a day by ``dropped_rule_id``
    and by nothing else. That id is an engine key: the clients drawer printed it
    raw beside the dropped count, so a reader was told how many spots a rule
    removed and shown a token they cannot act on instead of the cap.

    The sentence is imported from the pacing vocabulary rather than written
    again here, because that module's own header says it is the one place this
    product turns this engine artefact into words, and two translators for one
    rule is how a product comes to name one thing two ways. It is composed once
    for the whole payload because the rule file is read from disk on every call:
    per campaign that is one file read per campaign for one shared answer.
    """
    try:
        from kairos_api import pacing_alerts_api_words as words
    except Exception:  # noqa: BLE001 - an unreadable vocabulary is unknown, not a crash
        return {}
    ids = [
        day.get("dropped_rule_id")
        for days in days_by_campaign_index.values()
        for day in days
        if (day.get("spots_dropped_by_rule") or 0) > 0
    ]
    try:
        return words.booking_rules(ids)
    except Exception:  # noqa: BLE001
        return {}


def delivery_for(campaign: dict[str, Any], days: list[dict[str, Any]],
                 as_of: dict[str, Any],
                 booking_rules: Optional[dict[str, dict[str, Any]]] = None) -> dict[str, Any]:
    """One campaign's delivery: what aired, what is still to come, what is unknown.

    ``available`` is false when no day of this campaign carries a source at all,
    and the reason names the missing feed. It is never false-because-empty and
    never true-because-a-row-exists: a row in the ``unknown`` state is a stated
    gap, not a delivery.
    """
    aired = _totals(days, AIRED)
    scheduled = _totals(days, SCHEDULED)
    unknown_days = [day for day in days if day["air_state"] == UNKNOWN]
    sourced_dates = {day["broadcast_date"] for day in days if day["air_state"] != UNKNOWN}
    sourced = len(sourced_dates)
    terms = campaign.get("commitment") or {}
    measurable = bool(terms.get("rating_goal_measurable"))
    return {
        "available": sourced > 0,
        # The two denominators a surface may divide by. Every other count on this
        # block is a count of rows and does not add up to a calendar.
        "sourced_days": sourced,
        "flight_days": sourced + len(unknown_days),
        "reason_en": "" if sourced else NO_SOURCE_EN,
        "reason_he": "" if sourced else NO_SOURCE_HE,
        "path_forward_en": "" if sourced else NO_SOURCE_PATH_EN,
        "path_forward_he": "" if sourced else NO_SOURCE_PATH_HE,
        "as_of": as_of,
        "aired": aired,
        "scheduled": scheduled,
        "unknown": {
            "days": len(unknown_days),
            "dates": [day["broadcast_date"] for day in unknown_days],
            "reason_en": FLOOR_EN,
            "reason_he": FLOOR_HE,
        },
        # Only the rules this campaign's own days named, so a surface holding one
        # delivery block can say what was capped without the id reaching a screen
        # and without being handed the whole payload's vocabulary.
        "booking_rules": {
            rule_id: block
            for rule_id, block in (booking_rules or {}).items()
            if rule_id in {
                str(day.get("dropped_rule_id") or "")
                for day in days
                if (day.get("spots_dropped_by_rule") or 0) > 0
            }
        },
        "rating_basis_en": RATING_BASIS_EN,
        "rating_basis_he": RATING_BASIS_HE,
        "spend_basis_en": SPEND_BASIS_EN,
        "spend_basis_he": SPEND_BASIS_HE,
        "rating_progress": _progress(
            aired["rating_points_planned"],
            terms.get("rating_goal_points"),
            measurable=measurable,
            sourced=sourced > 0,
            unmeasurable_reason=(GOAL_UNMEASURABLE_EN, GOAL_UNMEASURABLE_HE),
        ),
        "budget_progress": _progress(
            aired["spend_ils"],
            terms.get("budget_ils"),
            measurable=True,
            sourced=sourced > 0,
            unmeasurable_reason=("", ""),
        ),
        "days": days,
    }


def attach(campaigns: list[dict[str, Any]]) -> dict[str, Any]:
    """Hang delivery and creative on every campaign, and report the payload's own state.

    Returns the payload-level ``delivery`` block. It is available only when at
    least one campaign on this payload has at least one sourced day, so an empty
    store answers "unknown" rather than "nothing delivered", and a store full of
    campaigns nobody has a traffic file for answers the same way.
    """
    from kairos_api import campaigns_assets

    frame = load_frame()
    as_of = _as_of(frame)
    grouped = days_by_campaign()
    assets = campaigns_assets.assets_by_campaign()
    rules = booking_rule_sentences(grouped)
    sourced_campaigns = 0
    for campaign in campaigns:
        days = grouped.get(campaign["campaign_id"], [])
        campaign["delivery"] = delivery_for(campaign, days, as_of, rules)
        own_assets = assets.get(campaign["campaign_id"], [])
        campaign["assets"] = own_assets
        campaign["assets_summary"] = campaigns_assets.summarise(own_assets)
        if campaign["delivery"]["available"]:
            sourced_campaigns += 1
    return {
        "available": sourced_campaigns > 0,
        "campaigns_with_a_source": sourced_campaigns,
        # A populated reason is what the board reads as "unavailable", so both
        # halves clear together when a source exists. The floor note below is the
        # sentence that belongs beside a figure that does exist.
        "reason_en": "" if sourced_campaigns else NO_SOURCE_EN,
        "reason_he": "" if sourced_campaigns else NO_SOURCE_HE,
        "path_forward_en": "" if sourced_campaigns else NO_SOURCE_PATH_EN,
        "path_forward_he": "" if sourced_campaigns else NO_SOURCE_PATH_HE,
        "as_of": as_of,
        "air_state_vocabulary": [dict(entry) for entry in AIR_STATE_VOCABULARY],
        "floor_note_en": FLOOR_EN,
        "floor_note_he": FLOOR_HE,
        "rating_basis_en": RATING_BASIS_EN,
        "rating_basis_he": RATING_BASIS_HE,
        "spend_basis_en": SPEND_BASIS_EN,
        "spend_basis_he": SPEND_BASIS_HE,
    }
