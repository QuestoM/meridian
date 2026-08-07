"""One break, opened: what it is, where it sits, what it earns, and what binds it.

The break inspector's payload. Everything on it resolves to something a person
can go to, because the rule that removes dead ends applies to a drawer as much as
to a page: the programme opens, the hour opens, the restriction that pins it
opens, and the figure that is missing says which file would supply it.

Each of those three is a control on the surface and not a sentence in this file.
The programme title opens the programme's own record through
``programme.segment_id`` with the channel and day beside it in ``identity``; the
hour opens the breaks the plan puts in it through ``guardrails.hour_breaks``,
each of which is an addressable break id; and a saved placement opens the record
naming who saved it, when, and which restriction carries it.

Three honesty rules govern this shape.

* **Projected money is the optimizer's own credit to this break**, not a share of
  the programme divided by the break count. The engine credits a marginal revenue
  at insertion and those credits sum back to the day exactly.
* **The stated basis reproduces the stated amount.** A break is not priced on the
  programme's baseline rating: the engine prices it on the rating that survives
  once the break is present, which is why the fourth break of a programme earns
  less than its first. Measured on ``רשת 13 / 2024-11-01``, ``001~1`` to
  ``001~4``: cpp 60, baseline rating 1.7, unit 1 s, premium 0.92 and a 120 s break
  give 11,260.80, while the plan credits 10,711.71, 10,162.61, 9,613.52 and
  9,064.43. The four ratios are 1 minus k times the 0.048762 retention cost, so a
  basis naming only the baseline rating overstated the first break by 549.09 and
  the fourth by 2,196.37. The rating each break is actually priced at now travels
  with the figure, and a test multiplies the printed inputs back out.
* **Delivered money is a state.** There is no spot ledger covering any planned
  day, so it reports unavailable with the reason and the path forward, and it
  will never carry a figure it did not read.
* **The retention cost carries its own uncertainty.** The credible interval, the
  sample the estimate rests on, and the confidence label all travel with the
  point, so a person can see how much to trust the number they are deciding on.
"""

from __future__ import annotations

from typing import Any, Optional

from kairos_api import break_api_board as board
from kairos_api import break_api_pod as pod
from kairos_api import break_store, break_store_pins


def _clock(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{(total // 3600) % 24:02d}:{(total % 3600) // 60:02d}:{total % 60:02d}"


def _segment_plan(plan: break_store.DayPlan, segment_id: str) -> Optional[Any]:
    for row in plan.result.segments:
        if row.segment_id == segment_id:
            return row
    return None


def _priced_rating(segment: Any, segment_plan: Optional[Any], ordinal: int) -> tuple[float, float]:
    """The retention this break was priced at, and the rating that follows from it.

    The engine's own seam is called rather than its arithmetic restated, because a
    second implementation of a price is a second price. ``_segment_retention`` is
    the one place the per-break coefficient enters retention, and
    ``_marginal_revenue`` multiplies its answer by ``baseline_tvr`` to get the
    rating it credits the k-th break at.

    The coefficient is taken from the plan rather than from the segment. When
    ``risk_lambda`` is above zero the optimizer decides on a more conservative
    coefficient than the point estimate, and ``retention_cost_used`` is that
    decided value, so pricing on the point estimate would disagree with the
    engine on exactly the days the operator most needs to trust it.
    """
    from dataclasses import replace

    from kairos.optimize._segment_math import _segment_retention

    priced = segment
    if segment_plan is not None:
        priced = replace(segment, impact_coefficient=float(segment_plan.retention_cost_used))
    retention = float(_segment_retention(priced, int(ordinal)))
    return retention, float(segment.baseline_tvr) * retention


def build_detail(plan: break_store.DayPlan, segment_id: str, ordinal: int) -> Optional[dict[str, Any]]:
    """Assemble one break's full record, or None when the id names no real break."""
    segment = plan.segment(segment_id)
    if segment is None:
        return None
    grouped = break_store.placements_by_segment(plan.result)
    placements = grouped.get(segment_id, [])
    if ordinal > len(placements):
        return None
    placement = placements[ordinal - 1]
    _counts, pins = break_store.arrangement(plan)
    items = board._breaks_for_guardrails(plan, pins)
    start = float(placement.start_seconds)
    duration = float(placement.duration_seconds)
    segment_start = float(segment.start_seconds)
    segment_plan = _segment_plan(plan, segment_id)
    hour_rows = {row["hour"]: row for row in board.hour_load(items, plan.guardrails)}
    break_id = break_store.break_id(segment_id, ordinal)
    saved = break_store_pins.for_day(plan.day).get(break_id)
    retention_here, rating_here = _priced_rating(segment, segment_plan, ordinal)

    from kairos_api.break_api_states import delivered_state

    return {
        "break_id": break_id,
        "identity": {
            "segment_id": segment_id,
            "ordinal": ordinal,
            "breaks_in_programme": len(placements),
            "channel": placement.channel,
            "day": placement.day,
        },
        "programme": {
            "segment_id": segment_id,
            "title": segment.program_title,
            "genre": segment.program_type,
            "start_clock": _clock(segment_start),
            "end_clock": _clock(segment_start + float(segment.duration_seconds)),
            "duration_seconds": round(float(segment.duration_seconds), 1),
            "baseline_rating": round(float(segment.baseline_tvr), 3),
            "rate_per_point": round(float(segment.cpp), 4),
            "premium": round(float(segment.premium), 4),
            "rate_unit_seconds": round(float(segment.unit_seconds), 3),
        },
        "placement": {
            "start_clock": _clock(start),
            "end_clock": _clock(start + duration),
            "start_seconds": round(start, 1),
            "duration_seconds": round(duration, 1),
            "offset_seconds": round(max(0.0, start - segment_start), 1),
            "hour": int(placement.hour),
            "source": "operator" if saved else "plan",
            "saved_placement": saved,
        },
        "money": {
            "projected": {
                "state": "real",
                "amount": round(float(placement.revenue), 2),
                "currency": "ILS",
                "basis": "the plan's own credit to this break, this channel and day",
                "basis_he": "הזיכוי של התוכנית לברייק הזה, בערוץ הזה וביום הזה",
                "retention_at_this_break": round(retention_here, 6),
                "rating_at_this_break": round(rating_here, 6),
                "formula": "rate per point times the rating this break is priced at, times its length over the rate unit, times the premium",
                "formula_he": "מחיר לנקודת רייטינג כפול הרייטינג שלפיו מתומחר הברייק הזה, כפול אורכו חלקי יחידת המחירון, כפול הפרמיה",
                "rating_formula": "the rating this break is priced at is the programme's baseline rating times the retention that holds once this break is present, so each further break in a programme earns less than the one before it",
                "rating_formula_he": "הרייטינג שלפיו מתומחר הברייק הזה הוא רייטינג הבסיס של התוכנית כפול השימור שמתקיים ברגע שהברייק קיים, ולכן כל ברייק נוסף בתוכנית מכניס פחות מקודמו",
            },
            "delivered": delivered_state(plan.day),
        },
        "retention": _retention(segment, segment_plan, placement),
        "gold": {
            "is_gold": bool(placement.is_gold),
            "scope": "programme",
            "max_per_day": int(plan.guardrails.gold_breaks_max_per_day),
        },
        "guardrails": {
            "hour": hour_rows.get(int(placement.hour)),
            "hour_breaks": _hour_breaks(plan, int(placement.hour)),
            "spacing": board.spacing_around([item for item in items if item.start_seconds != start], start),
            "min_break_spacing_seconds": float(plan.guardrails.min_break_spacing_seconds),
        },
        "compliance": board.compliance(items, plan.guardrails),
        # The individual ads inside a break are modelled now. The daily traffic
        # file already carries a break identifier per ad, which is the input the
        # earlier state here asked for, so this reads the pod that covers this
        # break's window instead of declaring the contents unavailable in every
        # case. It stays a state when no traffic file covers the day, and it now
        # names the days that are covered rather than only the missing input.
        "contents": pod.contents_state(plan.day, start, duration),
        "basis": board.basis(plan),
    }


def _hour_breaks(plan: break_store.DayPlan, hour: int) -> list[dict[str, Any]]:
    """Every break the plan puts in this clock hour, in time order.

    The hour row above it states a load against a licence limit, and a load is
    the sum of objects. Without them the figure is a dead end: a person reading
    that an hour carries 480 s against a 720 s limit has no way from the number
    to the breaks that make it, which is the one thing they would do next.

    Read from the same records the board serves rather than from a second walk of
    the placements, because two walks of one plan are two plans on the day they
    disagree.
    """
    rows = [
        {
            "break_id": record["break_id"],
            "start_seconds": record["start_seconds"],
            "start_clock": _clock(record["start_seconds"]),
            "duration_seconds": record["duration_seconds"],
            "programme": record["programme"],
            "segment_id": record["segment_id"],
            "is_gold": record["is_gold"],
            "projected_revenue": record["projected_revenue"],
        }
        for record in break_store.break_records(plan)
        if int(record["hour"]) == int(hour)
    ]
    rows.sort(key=lambda row: row["start_seconds"])
    return rows


def _retention(segment: Any, segment_plan: Optional[Any], placement: Any) -> dict[str, Any]:
    """The retention side of this break, with the uncertainty it actually carries."""
    record: dict[str, Any] = {
        "programme_retention": round(float(placement.retention), 6),
        "cost_per_break": round(float(segment.impact_coefficient), 6),
        "ci_low": None if segment.impact_ci_low is None else round(float(segment.impact_ci_low), 6),
        "ci_high": None if segment.impact_ci_high is None else round(float(segment.impact_ci_high), 6),
        "sample_breaks": int(segment.impact_n),
        "confidence": str(segment.impact_confidence),
        "first_break_multiplier": round(float(segment.first_break_multiplier), 4),
    }
    if segment_plan is not None:
        record["cost_used"] = round(float(segment_plan.retention_cost_used), 6)
        record["cost_point"] = round(float(segment_plan.retention_cost_point), 6)
    return record
