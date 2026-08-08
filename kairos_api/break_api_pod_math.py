"""The arithmetic on one pod: the sum, the span, the difference, and the gaps.

Split out of :mod:`kairos_api.break_api_pod` under the 450-line cap, and it is a
coherent module rather than a leftover: everything here is a subtraction over
numbers a traffic file declared, and nothing here reads a file or shapes a
payload.

Three rules run through all of it.

**A missing length is missing, not zero.** A spot that declares no length makes
the sum a floor rather than a total, and it stops the difference being served at
all. A figure that silently absorbed an absent length would understate a pod by
exactly the length nobody declared, which is the one lie this surface exists to
prevent.

**A negative difference is an overflow and is named as one.** A pod whose spots
run past its own span is a real and expensive condition, so it is reported rather
than clamped to zero.

**A declared break length is a fourth figure and it is often absent.** The plan is
the only place in the product that declares how long a break is meant to run, so
the comparison against it resolves when a plan covers the pod's day and places a
break over its start, and otherwise reports unavailable with both windows named.

**The uncovered figure is decomposed, not left as one number.** A pod's span
starts at the break's own declared start, which can sit well before the first
spot. That dead air and the holes between spots are two different things a
traffic operator needs to see separately, so the span before the first spot is
its own figure and the gaps between spots stay theirs; the two sum to the
uncovered total less any overlap.

**A declared position is a verdict, not a silent pass.** A spot's ``position``
is the rank the traffic file itself gives it inside the priced block. It is not
evidence of what a campaign bought, and this module never says it is: measured
on the shipped file, every pod's positions run 1 to N contiguously in the file's
own order and reach 26, which no preferred-position agreement describes. When
the pod's current order does not honour the file's own rank, that disagreement
is reported here as a fact about the order; nothing is refused, because a
traffic operator sometimes reorders a pod for a reason the position column does
not carry, and this module's job is to make the disagreement impossible to miss
rather than to relitigate a decision already made.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

NO_DECLARED = "No plan covers this day, so there is no declared break length to measure the pod against."
NO_DECLARED_HE = "אין תוכנית המכסה את היום הזה, ולכן אין אורך ברייק מוצהר להשוות אליו את התוכן."
NO_BREAK_HERE = "The plan for this day places no break over this pod's start time, so it declares no length for it."
NO_BREAK_HERE_HE = "התוכנית ליום הזה אינה מציבה ברייק מעל שעת ההתחלה של התוכן הזה, ולכן אינה מצהירה עבורו על אורך."

LOAD_BASIS = "the sum of the spots' own declared lengths in the traffic file"
LOAD_BASIS_HE = "סכום האורכים המוצהרים של התשדירים בקובץ הטראפיק"
SPAN_BASIS = "from the break's declared start to the last declared spot's own end, so it includes any dead air before the first spot"
SPAN_BASIS_HE = "מתחילת הברייק המוצהרת ועד סוף התשדיר המוצהר האחרון, ולכן כולל אוויר מת לפני התשדיר הראשון אם יש כזה"
UNFILLED_BASIS = "the span less the declared load, negative when the spots run past the span"
UNFILLED_BASIS_HE = "המשך פחות העומס המוצהר, שלילי כאשר התשדירים חורגים מהמשך"
HEAD_GAP_BASIS = "from the break's declared start to the first spot's declared start"
HEAD_GAP_BASIS_HE = "מתחילת הברייק המוצהרת ועד תחילת התשדיר הראשון המוצהרת"
DECLARED_BASIS = "the length the operator's own plan declares for the break covering this pod's start"
DECLARED_BASIS_HE = "האורך שהתוכנית של המפעיל מצהירה עליו עבור הברייק שמכסה את תחילת התוכן הזה"


def figure(seconds: Optional[float], basis: str, basis_he: str) -> dict[str, Any]:
    """A figure that was computed, or the absence of one. Never a zero either way."""
    if seconds is None:
        return {"state": "unknown", "seconds": None, "basis": basis, "basis_he": basis_he}
    return {"state": "real", "seconds": round(float(seconds), 1), "basis": basis, "basis_he": basis_he}


def pod_arithmetic(break_start: Optional[float], spots: list[dict[str, Any]]) -> dict[str, Any]:
    """The declared load against the pod's own span, and the difference in seconds.

    The continuity between consecutive spots is the same subtraction one size
    smaller, and it finds the holes inside a pod that a total cannot see. Measured
    on the shipped break at ``2025-04-27 20:40:09``: 569 s declared across 28
    spots, a span of 634 s, 65 s that no spot covers, and two internal gaps of
    13 s and 10 s.
    """
    lengths = [spot["duration"]["seconds"] for spot in spots]
    missing = sum(1 for value in lengths if value is None)
    load = sum(value for value in lengths if value is not None) if lengths else None
    ends = [spot["end_seconds"] for spot in spots if spot["end_seconds"] is not None]
    span = None if break_start is None or not ends else max(ends) - break_start
    unfilled = None if load is None or span is None or missing else span - load
    gaps, overlaps, gap_seconds, overlap_seconds = 0, 0, 0.0, 0.0
    ordered = [spot for spot in spots if spot["start_seconds"] is not None]
    for earlier, later in zip(ordered, ordered[1:]):
        if earlier["end_seconds"] is None:
            continue
        difference = round(later["start_seconds"] - earlier["end_seconds"], 1)
        if difference > 0:
            gaps += 1
            gap_seconds += difference
        elif difference < 0:
            overlaps += 1
            overlap_seconds += -difference
    head_gap = None
    if break_start is not None and ordered:
        head_gap = round(ordered[0]["start_seconds"] - break_start, 1)
    return {
        "spot_count": len(spots),
        "spots_missing_a_length": missing,
        "declared_load": figure(load, LOAD_BASIS, LOAD_BASIS_HE),
        "span": figure(span, SPAN_BASIS, SPAN_BASIS_HE),
        "unfilled": figure(unfilled, UNFILLED_BASIS, UNFILLED_BASIS_HE),
        "gap_before_first_spot": figure(head_gap, HEAD_GAP_BASIS, HEAD_GAP_BASIS_HE),
        "gaps_between_spots": {"count": gaps, "seconds": round(gap_seconds, 1)},
        "overlaps_between_spots": {"count": overlaps, "seconds": round(overlap_seconds, 1)},
    }


def declared_length(day: str, break_start: Optional[float]) -> dict[str, Any]:
    """The plan's own declared length for the break this pod sits in, or a state.

    The plan covers 2024-11-01 to 2024-11-30 and the one traffic file on disk
    covers 2025-04-27, so today this resolves to unavailable on every real pod,
    with both windows named. Supply a traffic file for a planned day and the
    comparison lights up with no change here, which is what
    ``tests/test_p10_pod_declared_length.py`` drives.
    """
    from kairos_api import break_store

    try:
        days = break_store.plan_days()
    except Exception:  # noqa: BLE001 - an unreadable plan is a state, not a crash
        logger.exception("plan day list read failed")
        days = []
    if str(day).strip() not in days:
        return {
            "state": "unavailable",
            "seconds": None,
            "reason": NO_DECLARED,
            "reason_he": NO_DECLARED_HE,
            "plan_covers": days[:1] + days[-1:] if days else [],
        }
    try:
        plan = break_store.day_plan(str(day).strip())
        records = break_store.break_records(plan)
    except Exception:  # noqa: BLE001 - a day that will not build is a state here
        logger.exception("day plan build failed for pod declared length on %s", day)
        return {"state": "unknown", "seconds": None, "reason": NO_DECLARED, "reason_he": NO_DECLARED_HE}
    over = [
        record for record in records
        if break_start is not None
        and float(record["start_seconds"]) <= break_start < float(record["end_seconds"])
    ]
    if not over:
        return {"state": "unavailable", "seconds": None, "reason": NO_BREAK_HERE, "reason_he": NO_BREAK_HERE_HE}
    match = over[0]
    return {
        "state": "real",
        "seconds": round(float(match["duration_seconds"]), 1),
        "break_id": match["break_id"],
        "programme": match["programme"],
        "basis": DECLARED_BASIS,
        "basis_he": DECLARED_BASIS_HE,
    }


def against_declared(declared: dict[str, Any], arithmetic: dict[str, Any]) -> dict[str, Any]:
    """The declared break length less the declared load: the gap, or the overflow.

    One spot with no declared length withholds the whole verdict, because a pod
    measured against a length while one of its own lengths is missing would report
    a gap that is at least partly the missing declaration.
    """
    length = declared.get("seconds")
    load = arithmetic["declared_load"].get("seconds")
    if length is None or load is None or arithmetic["spots_missing_a_length"]:
        return {
            "state": "unavailable",
            "seconds": None,
            "verdict": "unknown",
            "reason": declared.get("reason", NO_DECLARED),
            "reason_he": declared.get("reason_he", NO_DECLARED_HE),
        }
    difference = round(float(length) - float(load), 1)
    verdict = "exact" if difference == 0 else ("gap" if difference > 0 else "overflow")
    return {
        "state": "real",
        "seconds": abs(difference),
        "signed_seconds": difference,
        "verdict": verdict,
        "declared_seconds": round(float(length), 1),
        "load_seconds": round(float(load), 1),
    }


def position_violations(spots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Declared positions the pod's own current order does not honour.

    A numbered position names a spot's rank among the priced, positioned spots
    in the break, not its absolute place in the whole pod. Measured on the
    shipped break at ``2025-04-27 20:40:09``: six unpositioned sponsorship
    billboards air before position 1, so the spot the file places at position 1 is the
    seventh spot in the pod and the seventh in this ranking, not the first. So
    the check ranks only the spots that carry a real ordinal or Last, in the
    order this pod is currently shown in, and compares each rank to what it was
    the file declares. An unpositioned or unknown spot carries no declared rank and is
    skipped entirely, never counted and never checked.
    """
    ranked = [item for item in spots if (item.get("position") or {}).get("kind") in ("ordinal", "last")]
    total = len(ranked)
    violations: list[dict[str, Any]] = []
    for rank, item in enumerate(ranked, start=1):
        position = item.get("position") or {}
        if position.get("kind") == "ordinal":
            contracted = position.get("ordinal")
            if contracted is not None and int(contracted) != rank:
                violations.append({"spot_key": item["spot_key"], "contracted_position": str(contracted), "current_rank": rank})
        elif position.get("kind") == "last" and total and rank != total:
            violations.append({"spot_key": item["spot_key"], "contracted_position": "L", "current_rank": rank})
    return violations


def verification_errors(spots: list[dict[str, Any]], violations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Every real error this pod carries, named against the spot it belongs to.

    The trade's own step is verification, then finalising. This is the error
    list that step needs: a copy version whose own length disagrees with the
    booked duration, a spot with no declared length at all, and a spot not
    airing in the slot the traffic file declares for it. Nothing here is
    invented; each
    entry restates a check already computed elsewhere in this module or in
    :mod:`kairos_api.break_api_pod_spots`.
    """
    errors: list[dict[str, Any]] = []
    for item in spots:
        copy_check = item.get("copy_length") or {}
        if copy_check.get("state") == "disagrees":
            errors.append({
                "kind": "copy_length",
                "spot_key": item["spot_key"],
                "detail": f"The copy version names {copy_check['copy_seconds']:g}s, booked at {copy_check['booked_seconds']:g}s",
                "detail_he": f"שם הגרסה נוקב ב-{copy_check['copy_seconds']:g} שנ', בעוד ההזמנה היא {copy_check['booked_seconds']:g} שנ'",
            })
        if (item.get("duration") or {}).get("state") != "real":
            errors.append({
                "kind": "missing_length",
                "spot_key": item["spot_key"],
                "detail": "This spot declares no length.",
                "detail_he": "לתשדיר הזה אין אורך מוצהר.",
            })
    for violation in violations:
        errors.append({
            "kind": "position_order",
            "spot_key": violation["spot_key"],
            "detail": f"The traffic file places this spot at {violation['contracted_position']}, currently ranked {violation['current_rank']} among the positioned spots",
            "detail_he": f"קובץ הטראפיק מציב את התשדיר הזה במיקום {violation['contracted_position']}, וכעת הוא מדורג {violation['current_rank']} מבין התשדירים הממוקמים",
        })
    return errors
