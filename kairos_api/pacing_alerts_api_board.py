"""Clients, pacing: what a campaign has delivered against what it committed to.

The arithmetic, and none of the copy. Every figure here is summed from two stores
the account manager already owns: the campaign and its flights, and the derived
delivery ledger that says which broadcast days of the flight this product holds a
per-spot source for and which it does not.

Three honest boundaries decide the shape of every row.

**A day with no source is not a day with no delivery.** The ledger states that
difference and this module never collapses it. A flight day carrying no source
makes the figure for that day unknown, and it is the elapsed days, not the whole
flight, that decide whether a pace can be stated at all.

**Pace to date is only stated when the elapsed window is complete.** When every
broadcast day of the flight that has already run carries a source, what was
delivered to date is a total rather than a floor and a ratio against it is real.
When one elapsed day is missing, the ratio would be arithmetic over a hole, so the
verdict is unknown and the row names the missing days.

**The forward half is separate from the pace half, because they fail
separately.** A campaign can be exactly on pace to date and have six unbooked days
ahead of it, and a reader who is shown one number cannot tell those apart.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Optional

from kairos_api import pacing_alerts_api_words as words

AIRED = "aired"
SCHEDULED = "scheduled"


def parse_date(value: Any) -> Optional[date]:
    """A YYYY-MM-DD or leading ISO instant as a date, or None when it is neither."""
    text = str(value or "").strip()
    if not text:
        return None
    head = text.split(" ")[0].split("T")[0]
    parts = head.split("-")
    if len(parts) != 3 or not all(part.isdigit() for part in parts):
        return None
    try:
        return date(int(parts[0]), int(parts[1]), int(parts[2]))
    except ValueError:
        return None


def _span(start: date, end: date) -> list[date]:
    """Every calendar day of the flight, start and end inclusive."""
    if end < start:
        return [start]
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def _flight_dates(campaign: dict[str, Any]) -> tuple[Optional[date], Optional[date], str]:
    """The flight window, taken from the flights when they state one and the campaign otherwise.

    A campaign carries the booked window and each flight carries its own. Where
    flights exist the flown window is the union of theirs, because that is what
    was actually scheduled to run; where none does, the campaign's own dates are
    the only statement of the window there is.
    """
    flights = campaign.get("flights") or []
    starts = [parse_date(flight.get("starts_on")) for flight in flights]
    ends = [parse_date(flight.get("ends_on")) for flight in flights]
    starts = [value for value in starts if value is not None]
    ends = [value for value in ends if value is not None]
    if starts and ends:
        kinds = {str(flight.get("goal_kind") or "").strip() for flight in flights}
        kinds.discard("")
        return min(starts), max(ends), (sorted(kinds)[0] if len(kinds) == 1 else "")
    return parse_date(campaign.get("starts_on")), parse_date(campaign.get("ends_on")), ""


def _sum(days: list[dict[str, Any]], field: str) -> float:
    return float(sum(day.get(field) or 0.0 for day in days))


def _round(value: float, places: int) -> float:
    return round(float(value), places)


def _line(
    *,
    unit: str,
    goal: Optional[float],
    measurable: bool,
    measurable_code: str,
    field: str,
    elapsed: list[dict[str, Any]],
    ahead: list[dict[str, Any]],
    days_counted: int,
    flight_days: int,
    elapsed_gap: list[str],
    ahead_gap: list[str],
    started: bool,
    sourced_any: bool,
    places: int,
) -> dict[str, Any]:
    """One goal line: the counted figures, the reference, the pace and the forward state.

    ``elapsed`` and ``ahead`` are the ledger rows on or before, and after, the day
    the ledger was counted at. ``elapsed_gap`` and ``ahead_gap`` are the flight
    days on each side that carry no row at all, which is the only thing that can
    turn a real figure into an unknown one.
    """
    delivered = _sum([day for day in elapsed if day["air_state"] == AIRED], field)
    booked_today = _sum([day for day in elapsed if day["air_state"] == SCHEDULED], field)
    booked_ahead = _sum(ahead, field)
    through_today = delivered + booked_today
    booked_total = through_today + booked_ahead

    counted = {
        "delivered": _round(delivered, places),
        "booked_not_aired": _round(booked_today, places),
        "through_counted_day": _round(through_today, places),
        "booked_total": _round(booked_total, places),
        "days_counted": days_counted,
        "days_in_flight": flight_days,
    }

    if goal is None:
        state = dict(words.reason("no_goal"))
        return {"unit": unit, "goal": None, "counted": counted, "reference": None,
                "pace": {"verdict": words.UNKNOWN, "ratio": None, **state},
                "forward": {"state": words.NOT_BOOKED_YET, "remaining_to_goal": None, **state}}
    if not measurable:
        state = dict(words.reason(measurable_code))
        return {"unit": unit, "goal": _round(goal, places), "counted": counted, "reference": None,
                "pace": {"verdict": words.UNKNOWN, "ratio": None, **state},
                "forward": {"state": words.NOT_BOOKED_YET, "remaining_to_goal": None, **state}}

    reference = None
    if flight_days > 0 and days_counted > 0:
        reference = {
            "expected_through_counted_day": _round(goal * days_counted / flight_days, places),
            "rule_en": words.EVEN_REFERENCE_EN,
            "rule_he": words.EVEN_REFERENCE_HE,
        }

    pace = _pace(through_today, reference, elapsed_gap, started, sourced_any, places)
    forward = _forward(goal, booked_total, ahead_gap, places)
    return {
        "unit": unit,
        "goal": _round(goal, places),
        "counted": counted,
        "reference": reference,
        "pace": pace,
        "forward": forward,
    }


def _pace(
    through_today: float,
    reference: Optional[dict[str, Any]],
    elapsed_gap: list[str],
    started: bool,
    sourced_any: bool,
    places: int,
) -> dict[str, Any]:
    """The verdict to date, or the named reason there is not one."""
    if not started:
        return {"verdict": words.UNKNOWN, "ratio": None, **words.reason("not_started")}
    if elapsed_gap:
        # No sourced day at all is a different fact from a hole in an otherwise
        # counted window, and the two send the reader to different places: one
        # needs a feed, the other needs the days it names.
        code = "no_source" if not sourced_any else "gap_in_elapsed"
        state = dict(words.reason(code))
        state["unsourced_elapsed_days"] = list(elapsed_gap)
        return {"verdict": words.UNKNOWN, "ratio": None, **state}
    if reference is None:
        return {"verdict": words.UNKNOWN, "ratio": None, **words.reason("no_source")}
    expected = float(reference["expected_through_counted_day"])
    if expected <= 0:
        return {"verdict": words.UNKNOWN, "ratio": None, **words.reason("not_started")}
    ratio = through_today / expected
    if ratio >= words.ON_PACE_RATIO:
        verdict = words.ON_PACE
    elif ratio >= words.AT_RISK_RATIO:
        verdict = words.AT_RISK
    else:
        verdict = words.BEHIND
    return {
        "verdict": verdict,
        "ratio": round(ratio, 4),
        "gap_to_reference": _round(max(0.0, expected - through_today), places),
        "code": "",
        "reason_en": "",
        "reason_he": "",
        "path_forward_en": "",
        "path_forward_he": "",
    }


def _forward(goal: float, booked_total: float, ahead_gap: list[str], places: int) -> dict[str, Any]:
    """What the rest of the flight says, which fails separately from the pace."""
    remaining = max(0.0, goal - booked_total)
    block = {
        "remaining_to_goal": _round(remaining, places),
        "unsourced_remaining_days": list(ahead_gap),
        "code": "",
        "path_forward_en": "",
        "path_forward_he": "",
    }
    if booked_total >= goal:
        return {**block, "state": words.COVERED,
                "reason_en": words.FORWARD_COVERED_EN, "reason_he": words.FORWARD_COVERED_HE}
    if not ahead_gap:
        return {**block, "state": words.SHORT_CERTAIN,
                "reason_en": words.FORWARD_SHORT_EN, "reason_he": words.FORWARD_SHORT_HE}
    return {
        **block,
        "state": words.NOT_BOOKED_YET,
        "reason_en": words.FORWARD_OPEN_EN,
        "reason_he": words.FORWARD_OPEN_HE,
        "path_forward_en": words.FORWARD_OPEN_PATH_EN,
        "path_forward_he": words.FORWARD_OPEN_PATH_HE,
    }


def _rank(row: dict[str, Any]) -> tuple[int, float, str]:
    """Worst first: behind, then at risk, then unknown with a reason, then on pace."""
    order = {words.BEHIND: 0, words.AT_RISK: 1, words.UNKNOWN: 2, words.ON_PACE: 3}
    headline = row["headline"]
    ratio = headline.get("ratio")
    return (order.get(headline["verdict"], 4), ratio if ratio is not None else 9.9, row["campaign_id"])


def campaign_row(campaign: dict[str, Any], days: list[dict[str, Any]], as_of_day: Optional[date]) -> dict[str, Any]:
    """One campaign on the board: its flight, its two goal lines, and its day rows."""
    start, end, goal_kind = _flight_dates(campaign)
    terms = campaign.get("commitment") or {}
    base = {
        "campaign_id": campaign.get("campaign_id", ""),
        "name": campaign.get("name", ""),
        "advertiser": campaign.get("advertiser", ""),
        "agency_id": campaign.get("agency_id", ""),
        "channel": campaign.get("channel", ""),
        "status": campaign.get("status", ""),
        "is_demo": bool(campaign.get("is_demo")),
        "demo": campaign.get("demo") or {},
        "goal_kind": goal_kind,
        "days": days,
    }
    if start is None or end is None:
        state = dict(words.reason("no_flight_dates"))
        return {**base, "flight": None, "rating": None, "money": None,
                "headline": {"unit": "", "verdict": words.UNKNOWN, "ratio": None, **state}}

    span = _span(start, end)
    counted_day = as_of_day if as_of_day is not None else start - timedelta(days=1)
    elapsed_span = [day for day in span if day <= counted_day]
    ahead_span = [day for day in span if day > counted_day]
    sourced = {day["broadcast_date"] for day in days if day["air_state"] in (AIRED, SCHEDULED)}
    elapsed_rows = [day for day in days if day["air_state"] in (AIRED, SCHEDULED)
                    and parse_date(day["broadcast_date"]) is not None
                    and parse_date(day["broadcast_date"]) <= counted_day]
    ahead_rows = [day for day in days if day["air_state"] in (AIRED, SCHEDULED)
                  and parse_date(day["broadcast_date"]) is not None
                  and parse_date(day["broadcast_date"]) > counted_day]
    elapsed_gap = [day.isoformat() for day in elapsed_span if day.isoformat() not in sourced]
    ahead_gap = [day.isoformat() for day in ahead_span if day.isoformat() not in sourced]

    shared = {
        "elapsed": elapsed_rows,
        "ahead": ahead_rows,
        "days_counted": len(elapsed_span),
        "flight_days": len(span),
        "elapsed_gap": elapsed_gap,
        "ahead_gap": ahead_gap,
        "started": bool(elapsed_span),
        "sourced_any": bool(sourced),
    }
    rating = _line(
        unit=words.RATING_POINTS,
        goal=terms.get("rating_goal_points"),
        measurable=bool(terms.get("rating_goal_measurable")),
        measurable_code="unmeasurable",
        field="rating_points_planned",
        places=2,
        **shared,
    )
    money = _line(
        unit=words.ILS,
        goal=terms.get("budget_ils"),
        measurable=True,
        measurable_code="unmeasurable",
        field="spend_ils",
        places=2,
        **shared,
    )
    rating["audience"] = {
        "value": terms.get("rating_goal_audience", ""),
        "label_en": terms.get("rating_goal_audience_label_en", ""),
        "label_he": terms.get("rating_goal_audience_label_he", ""),
        "measurable": bool(terms.get("rating_goal_measurable")),
    }
    headline_line = rating if rating["goal"] is not None else money
    headline = {"unit": headline_line["unit"], **headline_line["pace"]}
    return {
        **base,
        "flight": {
            "starts_on": start.isoformat(),
            "ends_on": end.isoformat(),
            "days": len(span),
            "days_counted": len(elapsed_span),
            "days_remaining": len(ahead_span),
            "days_sourced": len(sourced),
            "unsourced_elapsed_days": elapsed_gap,
            "unsourced_remaining_days": ahead_gap,
        },
        "rating": rating,
        "money": money,
        "headline": headline,
    }


def build_rows(campaigns: list[dict[str, Any]], grouped: dict[str, list[dict[str, Any]]],
               as_of_day: Optional[date]) -> list[dict[str, Any]]:
    """Every campaign as a board row, worst pacing first."""
    rows = [campaign_row(campaign, grouped.get(campaign.get("campaign_id", ""), []), as_of_day)
            for campaign in campaigns]
    rows.sort(key=_rank)
    return rows


def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many rows landed in each verdict, so a header states the board rather than guessing it."""
    tally = {words.BEHIND: 0, words.AT_RISK: 0, words.ON_PACE: 0, words.UNKNOWN: 0}
    for row in rows:
        verdict = row["headline"]["verdict"]
        if verdict in tally:
            tally[verdict] += 1
    tally["total"] = len(rows)
    tally["demo"] = sum(1 for row in rows if row["is_demo"])
    return tally
