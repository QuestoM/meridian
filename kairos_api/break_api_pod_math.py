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
SPAN_BASIS = "from this break's declared start to the end of its last declared spot"
SPAN_BASIS_HE = "מהתחלת הברייק המוצהרת ועד סוף התשדיר המוצהר האחרון שלו"
UNFILLED_BASIS = "the span less the declared load, negative when the spots run past the span"
UNFILLED_BASIS_HE = "המשך פחות העומס המוצהר, שלילי כאשר התשדירים חורגים מהמשך"
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
    return {
        "spot_count": len(spots),
        "spots_missing_a_length": missing,
        "declared_load": figure(load, LOAD_BASIS, LOAD_BASIS_HE),
        "span": figure(span, SPAN_BASIS, SPAN_BASIS_HE),
        "unfilled": figure(unfilled, UNFILLED_BASIS, UNFILLED_BASIS_HE),
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
