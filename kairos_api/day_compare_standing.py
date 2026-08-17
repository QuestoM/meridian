"""Two dimensions a money delta cannot answer: what is left to sell, and who is owed.

A scheduler comparing two versions of one Tuesday is asked to weigh three things
at once, and the product could only ever show them one. Revenue is the easy one.
The other two are the ones that get a channel into trouble.

**Inventory.** A version that earns more by loading the day toward its licence
ceiling has spent something: the seconds a late request could still have been
sold into. That is not a cost anywhere in the revenue figure, and a day with
forty seconds left is a different day from one with nine minutes left.

**Contractual standing.** Some of those seconds are already promised. The
obligations engine (:mod:`kairos.trade.obligations`) measures every committed
term of every approved agreement against the delivery ledger; what it cannot see
is a plan that has not happened yet. This module gives it that: it re-measures
each obligation against the day each version would actually produce.

The link between a plan and an obligation is stated, not guessed. A day's booked
spots need seconds; a version of the day supplies seconds; when a version supplies
fewer than the day already owes, the shortfall must come out of somebody's
campaign. This module prorates that shortfall across the day's booked campaigns
by their own share of the day, scales that day's delivery rows by the resulting
factor, and re-runs the obligations engine on the result.

Three honesty rules, all of them load-bearing.

- **It is a projection and it says so.** ``basis`` is ``projection`` on every
  payload, the proration method is named in full, and nothing here claims to be
  a measurement of what aired.
- **Both sides go through the identical path.** The baseline is projected with
  the same function, the same frame and the same scaling, so a difference
  between the two sides is a difference between the two plans and never an
  artifact of one side taking a shortcut.
- **No source means unknown, never zero.** A day with no per-spot delivery row
  reports that the effect is unmeasurable and why; it does not report that no
  commitment is affected.
"""

from __future__ import annotations

from datetime import date as date_type
from typing import Any, Optional

import pandas as pd

# Alarm ladder severity, worst last. UNKNOWN is deliberately not on the ladder:
# it is not a degree of trouble, it is the absence of a comparison.
_SEVERITY = {"on_track": 0, "watch": 1, "at_risk": 2, "breached": 3}

ADVANCES = "advances"
ENDANGERS = "endangers"
BREAKS = "breaks"
UNCHANGED = "unchanged"
UNKNOWN = "unknown"

_VERDICT_HE = {
    ADVANCES: "מקדם",
    ENDANGERS: "מסכן",
    BREAKS: "מפר",
    UNCHANGED: "ללא שינוי",
    UNKNOWN: "לא ידוע",
}

_PRORATION_HE = (
    "היום מספק שניות פרסום; התשדירים המוזמנים ליום צורכים שניות. כשגרסה מספקת "
    "פחות שניות ממה שכבר מוזמן, המחסור מחולק בין הקמפיינים של אותו יום לפי חלקם "
    "בשניות היום, ושורות האספקה של אותו יום מוכפלות במקדם שנוצר. זו תחזית לפי "
    "שיטה מוצהרת ולא מדידה של מה ששודר."
)

_SCALED_COLUMNS = ("spots", "seconds", "rating_points_planned", "spend_ils")


def _sum(frame: pd.DataFrame, column: str) -> float:
    if frame is None or frame.empty or column not in frame.columns:
        return 0.0
    return float(pd.to_numeric(frame[column], errors="coerce").fillna(0).sum())


def hour_of(value: Any) -> Optional[int]:
    """The clock hour of an ``HH:MM`` plan cell, or None when it is unreadable."""
    text = str(value or "").strip()
    head, _, _rest = text.partition(":")
    try:
        hour = int(head)
    except (TypeError, ValueError):
        return None
    return hour if 0 <= hour <= 25 else None


# ------------------------------------------------------------------- inventory

def inventory_consequence(rows: pd.DataFrame, caps: dict[str, Any]) -> dict[str, Any]:
    """What this version of the day leaves sellable, against the licence ceilings.

    A cap the settings do not define is reported as ``None`` with the remaining
    figure ``None`` beside it, because "there is no cap" and "the cap is zero"
    are opposite facts and only one of them is ever true.
    """
    ad_seconds = round(_sum(rows, "total_break_time"), 1)
    breaks = int(round(_sum(rows, "num_breaks")))
    daily_cap = caps.get("max_daily_ad_seconds")
    hourly_seconds_cap = caps.get("max_ad_seconds_per_hour")
    hourly_breaks_cap = caps.get("max_breaks_per_hour")

    hours: dict[int, dict[str, float]] = {}
    unplaced = 0
    if rows is not None and not rows.empty:
        for _index, row in rows.iterrows():
            hour = hour_of(row.get("start_time"))
            if hour is None:
                unplaced += 1
                continue
            bucket = hours.setdefault(hour, {"breaks": 0.0, "ad_seconds": 0.0})
            bucket["breaks"] += float(pd.to_numeric(row.get("num_breaks"), errors="coerce") or 0)
            bucket["ad_seconds"] += float(pd.to_numeric(row.get("total_break_time"), errors="coerce") or 0)

    hour_rows = []
    breaks_headroom = 0
    seconds_headroom = 0.0
    for hour in sorted(hours):
        bucket = hours[hour]
        used_breaks = int(round(bucket["breaks"]))
        used_seconds = round(bucket["ad_seconds"], 1)
        room_breaks = None if hourly_breaks_cap is None else int(hourly_breaks_cap) - used_breaks
        room_seconds = None if hourly_seconds_cap is None else round(float(hourly_seconds_cap) - used_seconds, 1)
        if room_breaks is not None:
            breaks_headroom += max(0, room_breaks)
        if room_seconds is not None:
            seconds_headroom += max(0.0, room_seconds)
        hour_rows.append({
            "hour": hour, "breaks": used_breaks, "ad_seconds": used_seconds,
            "breaks_remaining": room_breaks, "ad_seconds_remaining": room_seconds,
            "over_breaks": room_breaks is not None and room_breaks < 0,
            "over_ad_seconds": room_seconds is not None and room_seconds < 0,
        })

    daily_remaining = None if daily_cap is None else round(float(daily_cap) - ad_seconds, 1)
    return {
        "ad_seconds_planned": ad_seconds,
        "breaks_planned": breaks,
        "hours_covered": len(hour_rows),
        "rows_without_a_clock_time": unplaced,
        "daily_ad_seconds_cap": None if daily_cap is None else float(daily_cap),
        "daily_ad_seconds_remaining": daily_remaining,
        "over_daily_cap": daily_remaining is not None and daily_remaining < 0,
        # The hour-by-hour headroom, summed only over hours this day actually
        # covers. An hour with no programme has no headroom to sell into here,
        # so it is not counted as free capacity that does not exist.
        "hourly_breaks_remaining": None if hourly_breaks_cap is None else breaks_headroom,
        "hourly_ad_seconds_remaining": None if hourly_seconds_cap is None else round(seconds_headroom, 1),
        "hours": hour_rows,
        "cap_note_he": (
            "תקרה שאינה מוגדרת בהגדרות מדווחת כלא קיימת ולא כאפס"
            if daily_cap is None or hourly_breaks_cap is None else ""
        ),
    }


# --------------------------------------------------------- contractual standing

def _day_mask(delivery: pd.DataFrame, channel: str, date: str) -> pd.Series:
    dates = delivery["broadcast_date"].astype(str).str.slice(0, 10).str.strip()
    mask = dates == str(date).strip()
    if "channel" in delivery.columns and str(channel or "").strip():
        mask = mask & (delivery["channel"].astype(str).str.strip() == str(channel).strip())
    return mask


def _projected_delivery(delivery: pd.DataFrame, channel: str, date: str,
                        factor: float) -> pd.DataFrame:
    """The delivery ledger with only this channel-day's rows scaled by ``factor``."""
    if delivery is None or delivery.empty or factor == 1.0:
        return delivery
    work = delivery.copy()
    mask = _day_mask(work, channel, date)
    for column in _SCALED_COLUMNS:
        if column not in work.columns:
            continue
        numeric = pd.to_numeric(work.loc[mask, column], errors="coerce")
        work.loc[mask, column] = (numeric * factor).where(numeric.notna(), work.loc[mask, column])
    return work


def _capacity_factor(supplied_seconds: float, booked_seconds: float) -> float:
    """How much of the day's booked airtime this version can actually carry."""
    if booked_seconds <= 0:
        return 1.0
    if supplied_seconds >= booked_seconds:
        return 1.0
    return max(0.0, supplied_seconds / booked_seconds)


def _snapshot_index(snapshots: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(snap.get("obligation_id")): snap for snap in snapshots}


def _figure(snapshot: dict[str, Any]) -> dict[str, Any]:
    standing = snapshot.get("standing") or {}
    return {
        "alarm": snapshot.get("alarm"),
        "alarm_reason": snapshot.get("alarm_reason"),
        "counted": standing.get("counted"),
        "unit": standing.get("unit"),
        "projection": snapshot.get("projection"),
        "target": (snapshot.get("target") or {}).get("value"),
    }


def _verdict(before: dict[str, Any], after: dict[str, Any]) -> tuple[str, str]:
    """How one obligation fares under a version, against the same obligation today."""
    before_rank = _SEVERITY.get(str(before.get("alarm")))
    after_rank = _SEVERITY.get(str(after.get("alarm")))
    if before_rank is None or after_rank is None:
        return UNKNOWN, "אין בסיס מדוד להשוואה בין הגרסאות עבור התחייבות זו"
    if after_rank > before_rank:
        if str(after.get("alarm")) == "breached":
            return BREAKS, "הגרסה מביאה את ההתחייבות למצב הפרה"
        return ENDANGERS, "הגרסה מחמירה את מצב ההתחייבות"
    if after_rank < before_rank:
        return ADVANCES, "הגרסה משפרת את מצב ההתחייבות"
    projected_before = before.get("projection")
    projected_after = after.get("projection")
    if projected_before is not None and projected_after is not None:
        if projected_after < projected_before:
            return ENDANGERS, "רמת האזעקה זהה אך התחזית לסגירת ההתחייבות יורדת"
        if projected_after > projected_before:
            return ADVANCES, "רמת האזעקה זהה והתחזית לסגירת ההתחייבות עולה"
    return UNCHANGED, "ההתחייבות אינה מושפעת מהגרסה הזו"


def _agreement_snapshots(delivery: pd.DataFrame, campaigns: pd.DataFrame,
                         links: pd.DataFrame, today: date_type,
                         approved: list[tuple[dict[str, Any], dict[str, Any]]],
                         ) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    from kairos.trade import obligations as ob

    inputs = ob.Inputs(delivery=delivery, campaigns=campaigns, agency_links=links,
                       today=today, preferred_rate=None)
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for head, termset in approved:
        for snapshot in ob.evaluate_all(termset, head, inputs):
            out.append((head, snapshot))
    return out


def contractual_standing(
    *,
    channel: str,
    date: str,
    baseline_rows: pd.DataFrame,
    side_rows: pd.DataFrame,
    approved: list[tuple[dict[str, Any], dict[str, Any]]],
    delivery: pd.DataFrame,
    campaigns: pd.DataFrame,
    links: pd.DataFrame,
    today: date_type,
) -> dict[str, Any]:
    """Which commitments this version advances, endangers or breaks, and why.

    ``approved`` is the (head, termset) pairs of the agreements that reached an
    approved version; an agreement still in review binds nothing and is not
    measured here. Every figure on both sides comes out of the same
    :func:`kairos.trade.obligations.evaluate_all` call shape, so the comparison
    is between two plans rather than between two methods.
    """
    booked = 0.0
    day_rows = 0
    if delivery is not None and not delivery.empty and "broadcast_date" in delivery.columns:
        mask = _day_mask(delivery, channel, date)
        day_rows = int(mask.sum())
        booked = round(_sum(delivery[mask], "seconds"), 1)
    baseline_seconds = round(_sum(baseline_rows, "total_break_time"), 1)
    side_seconds = round(_sum(side_rows, "total_break_time"), 1)

    if not approved:
        return {
            "available": False,
            "reason": "no approved agreement carries a committed term, so nothing is measured",
            "reason_he": "אין הסכם מאושר עם מונח מחייב, ולכן אין מה למדוד",
            "counts": {}, "obligations": [],
        }
    if day_rows == 0:
        return {
            "available": False,
            "reason": f"the delivery ledger holds no row for {channel} on {date}, so this day's "
                      "effect on commitments is unknown rather than nil",
            "reason_he": "ספר האספקה אינו מחזיק שורה ליום הזה בערוץ הזה, ולכן השפעת היום על "
                         "ההתחייבויות אינה ידועה ואינה אפס",
            "counts": {}, "obligations": [],
            "day_capacity": {
                "baseline_seconds": baseline_seconds, "side_seconds": side_seconds,
                "booked_seconds": booked, "delivery_rows": 0,
            },
        }

    baseline_factor = _capacity_factor(baseline_seconds, booked)
    side_factor = _capacity_factor(side_seconds, booked)
    before = _snapshot_index([
        snapshot for _head, snapshot in _agreement_snapshots(
            _projected_delivery(delivery, channel, date, baseline_factor),
            campaigns, links, today, approved)
    ])
    heads: dict[str, dict[str, Any]] = {}
    after_pairs = _agreement_snapshots(
        _projected_delivery(delivery, channel, date, side_factor),
        campaigns, links, today, approved)
    counts = {ADVANCES: 0, ENDANGERS: 0, BREAKS: 0, UNCHANGED: 0, UNKNOWN: 0}
    items: list[dict[str, Any]] = []
    for head, snapshot in after_pairs:
        obligation_id = str(snapshot.get("obligation_id"))
        heads[obligation_id] = head
        baseline_snapshot = before.get(obligation_id)
        if baseline_snapshot is None:
            continue
        side_figure = _figure(snapshot)
        base_figure = _figure(baseline_snapshot)
        verdict, reason_he = _verdict(base_figure, side_figure)
        counts[verdict] += 1
        items.append({
            "obligation_id": obligation_id,
            "agreement_id": snapshot.get("agreement_id"),
            "agreement_title": head.get("title"),
            "counterparty": head.get("counterparty") or {},
            "term_id": snapshot.get("term_id"),
            "verdict": verdict,
            "verdict_he": _VERDICT_HE[verdict],
            "reason_he": reason_he,
            "baseline": base_figure,
            "side": side_figure,
        })
    items.sort(key=lambda item: (
        {BREAKS: 0, ENDANGERS: 1, ADVANCES: 2, UNCHANGED: 3, UNKNOWN: 4}[item["verdict"]],
        str(item["obligation_id"]),
    ))
    return {
        "available": True,
        "basis": "projection",
        "method": {
            "name": "capacity-prorated day delivery",
            "note_he": _PRORATION_HE,
            "scaled_columns": list(_SCALED_COLUMNS),
        },
        "day_capacity": {
            "baseline_seconds": baseline_seconds,
            "side_seconds": side_seconds,
            "booked_seconds": booked,
            "delivery_rows": day_rows,
            "baseline_factor": round(baseline_factor, 6),
            "side_factor": round(side_factor, 6),
            "shortfall_seconds": round(max(0.0, booked - side_seconds), 1),
        },
        "counts": counts,
        "obligations": items,
    }
