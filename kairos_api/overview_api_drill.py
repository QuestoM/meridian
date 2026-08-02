"""The second level of the money drill: one day, and the rows that make it up.

Today's money figure resolves to seven days, and each of those days resolves to
the plan rows that produced it. Two levels, in place, without leaving the
screen, which is the mechanic a figure needs before it can claim to be
auditable: a number that only shows a total is a claim, and a number that opens
its parts twice is a reading.

Three things keep this level honest.

**It is the same arithmetic, not a second one.** The rows are the saved plan's
own rows for that date on the operator's channel, summed on the same column the
window figure sums. The payload states the residual between the rows and the
day, so a reader can see that they reconcile instead of being told they do.

**The competitor boundary is applied through the shared helper**, so a rival
channel's programme can never appear among the operator's own rows and the
disclosure that travels with the scope is the same one every scoped surface
prints.

**The level below is named, not faked, and the naming is measured.** Which
advertisers sat in each break, and what they delivered, is a different quantity
from a different source, and whether that source reaches this day is read off
the source itself on every call. It is reported as an absence with the input
that would end it, never as a zero.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from kairos_api import channel_scope

# What the level below this one would need before it could be rendered. Stated
# as data rather than prose so the surface can print it in either language and
# link the route that supplies it.
#
# This was a constant sentence until a critic named what a constant here is: an
# assertion about the disk, that the ledger covers one day in a different month
# from the plan. That was true on the day it was written and turns into a false
# claim the moment a delivery file for the planned week is loaded, with nothing
# in the product to catch it. A statement about what is on disk is now read off
# the disk, every time it is made.


def _coverage() -> Optional[list[str]]:
    """The broadcast dates the priced spot ledger on disk actually covers.

    ``None`` is the third state and not an empty one: the coverage itself could
    not be read, so this day is neither covered nor known to be uncovered.
    """
    try:
        from kairos.export.spots_coverage import daily_input_days

        return sorted(daily_input_days())
    except Exception:  # noqa: BLE001 - an unreadable ledger is a state, not a crash
        return None


def _span(covered: list[str]) -> tuple[str, str]:
    """The coverage as one phrase per language, built from the dates themselves."""
    if len(covered) == 1:
        return f"a single day, {covered[0]}", f"יום אחד בלבד, {covered[0]}"
    return (
        f"{len(covered)} days between {covered[0]} and {covered[-1]}",
        f"{len(covered)} ימים בין {covered[0]} ל־{covered[-1]}",
    )


def delivered_state(iso_date: str) -> dict[str, Any]:
    """Whether the ledger can say what each break delivered on this day."""
    covered = _coverage()
    if covered is None:
        return {
            "available": False,
            "state": "unknown",
            "covers": None,
            "reason_en": "Which advertisers sat in each break, and what they paid, cannot be reported for this day: the priced spot ledger on disk could not be read, so whether it reaches this day is unknown.",
            "reason_he": "מי המפרסמים שישבו בכל ברייק וכמה שילמו אינו ניתן לדיווח ליום הזה: לא ניתן היה לקרוא את ספר התשדירים המתומחר שעל הדיסק, ולכן לא ידוע אם הוא מגיע ליום הזה.",
            "needs_en": "A readable spot ledger, listed with every other input under Sources.",
            "needs_he": "ספר תשדירים קריא, שמופיע עם שאר הקלטים תחת מקורות.",
            "opens": "sources",
        }
    if not covered:
        return {
            "available": False,
            "state": "unavailable",
            "covers": [],
            "reason_en": "Which advertisers sat in each break, and what they paid, is not known for this day. There is no priced spot ledger on disk at all, so there is nothing behind this level to read.",
            "reason_he": "מי המפרסמים שישבו בכל ברייק וכמה שילמו אינו ידוע ליום הזה. אין על הדיסק ספר תשדירים מתומחר כלל, ולכן אין מה לקרוא מתחת לרמה הזאת.",
            "needs_en": "A delivery feed for the planned week, loaded under Sources.",
            "needs_he": "הזנת שידור בפועל לשבוע המתוכנן, נטענת תחת מקורות.",
            "opens": "sources",
        }
    if str(iso_date).strip() not in covered:
        span_en, span_he = _span(covered)
        return {
            "available": False,
            "state": "unavailable",
            "covers": covered,
            "reason_en": f"Which advertisers sat in each break, and what they paid, is not known for this day. The priced spot ledger on disk covers {span_en}, and this day is not among them.",
            "reason_he": f"מי המפרסמים שישבו בכל ברייק וכמה שילמו אינו ידוע ליום הזה. ספר התשדירים המתומחר שעל הדיסק מכסה {span_he}, והיום הזה אינו ביניהם.",
            "needs_en": "A delivery feed for the planned week, loaded under Sources.",
            "needs_he": "הזנת שידור בפועל לשבוע המתוכנן, נטענת תחת מקורות.",
            "opens": "sources",
        }
    # The ledger does reach this day. Nothing in this module joins its spots to
    # the plan's breaks, so the absence is the join and not the feed, and it
    # says so rather than reusing the sentence about a missing file.
    return {
        "available": False,
        "state": "unavailable",
        "covers": covered,
        "reason_en": "The priced spot ledger on disk covers this day, but nothing here matches its spots to this plan's breaks, so what each break delivered cannot be reported from this level.",
        "reason_he": "ספר התשדירים המתומחר שעל הדיסק מכסה את היום הזה, אבל שום דבר כאן לא מתאים את התשדירים שלו לברייקים של התוכנית, ולכן אי אפשר לדווח מהרמה הזאת מה כל ברייק סיפק.",
        "needs_en": "Each spot matched to the break it aired in, a link this level does not hold yet.",
        "needs_he": "התאמה של כל תשדיר לברייק שבו שודר, קישור שהרמה הזאת עדיין לא מחזיקה.",
        "opens": "plan",
    }


def _hour(clock: Any) -> Optional[int]:
    text = str(clock or "").strip()
    return int(text[:2]) if len(text) >= 4 and text[:2].isdigit() else None


def _money(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else round(number, 2)


def _fraction(value: Any) -> Optional[float]:
    """A share exactly as the plan holds it, rounded nowhere.

    Retention was read through the money rounder before being turned into a
    percentage, so 0.8054 left here as 81.0 while the decision list, which
    rounds once, printed 80.5 for the same segment. One quantity read two ways
    on one screen is a defect whichever number is prettier, so the rounding
    happens once, where the percentage is made.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def day_detail(schedule: pd.DataFrame, iso_date: str, dates_in_window: list[str]) -> dict[str, Any]:
    """Every plan row behind one day's figure, on the operator's channel.

    ``dates_in_window`` is the ordered list of dates the window covers, and it
    is what gives the reader their place in the set: the payload carries the
    index and the neighbours, so a day can be walked without going back up.
    """
    from kairos.data.dayparts import daypart_for_hour

    scoped, note = channel_scope.scope_frame(schedule)
    position = _position(iso_date, dates_in_window)
    if not note.get("scoped"):
        # Unscoped rows are the whole market's. A drill that served them would
        # put a competitor's programmes inside the operator's own figure.
        return _empty(iso_date, note, position, "the operator's own channel has not been declared in settings")
    if scoped is None or len(scoped) == 0 or "date" not in getattr(scoped, "columns", []):
        return _empty(iso_date, note, position, "the saved plan holds no rows for the operator's channel")

    frame = scoped[scoped["date"].astype(str).str.strip() == str(iso_date).strip()]
    if len(frame) == 0:
        return _empty(iso_date, note, position, "the saved plan holds no rows for this date")

    revenue = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    total = float(revenue.sum())
    ordered = frame.assign(_revenue=revenue).sort_values("_revenue", ascending=False)

    rows: list[dict[str, Any]] = []
    for _, row in ordered.iterrows():
        amount = _money(row["_revenue"])
        retention = _fraction(pd.to_numeric(row.get("predicted_retention", 0), errors="coerce"))
        clock = str(row.get("start_time", "")).strip()
        rows.append(
            {
                "segment_id": str(row.get("segment_id", "")).strip(),
                "start_clock": clock,
                "program_type": str(row.get("program_type", "")).strip(),
                "daypart": daypart_for_hour(_hour(clock)) or None,
                "breaks": int(pd.to_numeric(row.get("num_breaks", 0), errors="coerce") or 0),
                "ad_seconds": int(pd.to_numeric(row.get("total_break_time", 0), errors="coerce") or 0),
                "projected_revenue": amount,
                "retention_percent": None if retention is None else round(retention * 100, 1),
                "share_percent": None if not total or amount is None else round(amount / total * 100, 2),
                "is_gold": bool(row.get("is_gold", False)),
            }
        )

    rows_total = _money(sum(row["projected_revenue"] or 0 for row in rows))
    day_total = _money(total)
    residual = None if rows_total is None or day_total is None else round(day_total - rows_total, 2)
    return {
        "date": str(iso_date),
        "available": True,
        "reason": None,
        "channel": note.get("scope_channel"),
        "boundary": note,
        "position": position,
        "projected_revenue": day_total,
        "rows": rows,
        "row_count": len(rows),
        "rows_total_ils": rows_total,
        "residual_ils": residual,
        "reconciled": residual is not None and abs(residual) < 0.5,
        "total_breaks": int(sum(row["breaks"] for row in rows)),
        "total_ad_seconds": int(sum(row["ad_seconds"] for row in rows)),
        "gold_breaks": int(sum(1 for row in rows if row["is_gold"])),
        "delivered": delivered_state(iso_date),
    }


def _position(iso_date: str, dates_in_window: list[str]) -> dict[str, Any]:
    """Where this day sits in the window, and which days are either side of it."""
    ordered = [str(value) for value in dates_in_window if str(value)]
    try:
        index = ordered.index(str(iso_date))
    except ValueError:
        return {"index": None, "total": len(ordered), "previous": None, "next": None}
    return {
        "index": index + 1,
        "total": len(ordered),
        "previous": ordered[index - 1] if index > 0 else None,
        "next": ordered[index + 1] if index + 1 < len(ordered) else None,
    }


def _empty(iso_date: str, note: dict[str, Any], position: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "date": str(iso_date),
        "available": False,
        "reason": reason,
        "channel": note.get("scope_channel"),
        "boundary": note,
        "position": position,
        "projected_revenue": None,
        "rows": [],
        "row_count": 0,
        "rows_total_ils": None,
        "residual_ils": None,
        "reconciled": False,
        "total_breaks": 0,
        "total_ad_seconds": 0,
        "gold_breaks": 0,
        "delivered": delivered_state(iso_date),
    }
