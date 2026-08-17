"""Where a version's money difference comes from, programme by programme.

This is the piece the whole comparison surface is for. A number beside a number
is not a comparison; a decision-maker needs the CAUSE, and the cause has to be
arithmetic rather than a sentence somebody wrote.

So the difference between two versions of a day is decomposed from the row-level
diff itself. Each programme lands in exactly one bucket according to what
actually moved on it - a break added, a break removed, a length changed, a price
moved, a programme entering or leaving the board - and each bucket is cut by
daypart, which is what turns a figure into the sentence a person reads: "breaks
added in prime time: +2 breaks, 2 programmes: +250,000 ILS".

Two rules make it trustworthy.

**It sums exactly, in integer agorot.** Cells sum to buckets, buckets sum to the
total, and the total equals the scoped headline difference - by construction, not
by luck, because nothing is rounded until the payload is built. Rounding each
level separately is exactly how an explanation drifts a few agorot from the
number it explains, and an explanation that does not add up is worse than none.

**A residue is a finding, never an absorption.** The truth the attribution is
measured against is the money the two frames actually carry, not the money the
segment-keyed diff managed to key. Anything left over becomes an explicit
``unattributed`` bucket carrying its own reason, so a row this code cannot key to
a programme surfaces as an unexplained difference instead of quietly inflating a
neighbouring bucket that would then be wrong.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from kairos_api import day_compare_standing as standing

BREAKS_ADDED = "breaks_added"
BREAKS_REMOVED = "breaks_removed"
LENGTH_CHANGED = "length_changed"
REPRICED = "repriced"
SEGMENT_ADDED = "segment_added"
SEGMENT_REMOVED = "segment_removed"
UNATTRIBUTED = "unattributed"

BUCKETS = (BREAKS_ADDED, BREAKS_REMOVED, LENGTH_CHANGED, REPRICED,
           SEGMENT_ADDED, SEGMENT_REMOVED, UNATTRIBUTED)

_BUCKET_HE = {
    BREAKS_ADDED: "ברייקים נוספו",
    BREAKS_REMOVED: "ברייקים הוסרו",
    LENGTH_CHANGED: "אורך הברייקים שונה",
    REPRICED: "התמחור השתנה",
    SEGMENT_ADDED: "תוכניות נוספו ללוח",
    SEGMENT_REMOVED: "תוכניות ירדו מהלוח",
    UNATTRIBUTED: "לא ניתן לשייך לגורם",
}

_NO_DAYPART_HE = "שעה לא מסווגת"
_TOLERANCE_SECONDS = 0.05


def _cents(value: Any) -> int:
    """One money figure as integer agorot. Unreadable reads as zero contribution."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number != number:  # NaN
        return 0
    return int(round(number * 100))


def _ils(cents: int) -> float:
    return round(cents / 100.0, 2)


def _number(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if number == number else 0.0


def _daypart_of(start_time: Any) -> tuple[Optional[str], str]:
    from kairos.data.dayparts import daypart_for_hour, dayparts

    hour = standing.hour_of(start_time)
    key = daypart_for_hour(hour if hour is None or hour <= 23 else hour % 24)
    if key is None:
        return None, _NO_DAYPART_HE
    label = next((part.label_he for part in dayparts() if part.key == key), key)
    return key, label


def _row_index(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """One day's rows keyed by the optimizer's own segment id, ready to diff."""
    index: dict[str, dict[str, Any]] = {}
    if frame is None or frame.empty:
        return index
    for _position, row in frame.iterrows():
        segment_id = str(row.get("segment_id") or "").strip()
        if not segment_id:
            continue
        key, label = _daypart_of(row.get("start_time"))
        index[segment_id] = {
            "segment_id": segment_id,
            "breaks": int(round(_number(row.get("num_breaks")))),
            "ad_seconds": _number(row.get("total_break_time")),
            "revenue_cents": _cents(row.get("predicted_revenue")),
            "start_time": str(row.get("start_time") or ""),
            "programme": str(row.get("program_type") or ""),
            "daypart": key,
            "daypart_he": label,
        }
    return index


def _revenue_agorot(frame: pd.DataFrame) -> int:
    """Every agora a frame carries, on the SAME basis as the headline.

    The headline the decision-maker reads is plan_version_store._totals:
    the float column summed once, rounded once to two decimals. This truth
    figure must be computed identically — rounding each row to agorot first
    and summing summed DIFFERENTLY from the headline on sub-agora fractions,
    so the payload could claim exact:True while disagreeing with the number
    it explains. With one basis, any per-row rounding drift lands in
    ``residual`` and is printed as unattributed instead of denied.
    """
    if frame is None or frame.empty or "predicted_revenue" not in frame.columns:
        return 0
    revenue = pd.to_numeric(frame["predicted_revenue"], errors="coerce").fillna(0)
    return _cents(round(float(revenue.sum()), 2))


def _classify(before: Optional[dict[str, Any]], after: Optional[dict[str, Any]]) -> str:
    """Which single fact about this programme moved. Exhaustive by construction."""
    if before is None:
        return SEGMENT_ADDED
    if after is None:
        return SEGMENT_REMOVED
    if after["breaks"] > before["breaks"]:
        return BREAKS_ADDED
    if after["breaks"] < before["breaks"]:
        return BREAKS_REMOVED
    if abs(after["ad_seconds"] - before["ad_seconds"]) > _TOLERANCE_SECONDS:
        return LENGTH_CHANGED
    return REPRICED


def _cell_sentence(bucket: str, daypart_he: str, revenue_cents: int,
                   breaks_delta: int, seconds_delta: float, segments: int) -> str:
    parts = [f"{_BUCKET_HE[bucket]} ב{daypart_he}"]
    if breaks_delta:
        parts.append(f"{breaks_delta:+d} ברייקים")
    elif abs(seconds_delta) > _TOLERANCE_SECONDS:
        parts.append(f"{seconds_delta:+,.0f} שניות פרסום")
    parts.append(f"{segments} תוכניות")
    return ", ".join(parts) + f": {_ils(revenue_cents):+,.0f} ₪"


def attribute(baseline_rows: pd.DataFrame, side_rows: pd.DataFrame) -> dict[str, Any]:
    """Where a version's money difference comes from, programme by programme.

    Returns the cells (one per bucket-and-daypart pair that moved), the buckets
    they roll into, and the total - every one of them a sum of the same integer
    agorot, so the three levels agree exactly. ``residual`` is the part of the
    scoped revenue difference that landed in no cell; it is zero by construction
    and is reported rather than assumed.
    """
    before = _row_index(baseline_rows)
    after = _row_index(side_rows)
    if not before and not after:
        return {"available": False,
                "reason": "neither side carries a segment_id, so a placement-level diff "
                          "cannot be keyed to a programme",
                "reason_he": "אף צד אינו נושא מזהה תוכנית, ולכן לא ניתן לגזור השוואה ברמת השיבוץ"}

    cells: dict[tuple[str, Optional[str]], dict[str, Any]] = {}
    changed_segments: list[dict[str, Any]] = []
    for segment_id in sorted(set(before) | set(after)):
        base = before.get(segment_id)
        side = after.get(segment_id)
        revenue_delta = (side["revenue_cents"] if side else 0) - (base["revenue_cents"] if base else 0)
        breaks_delta = (side["breaks"] if side else 0) - (base["breaks"] if base else 0)
        seconds_delta = (side["ad_seconds"] if side else 0.0) - (base["ad_seconds"] if base else 0.0)
        if (revenue_delta == 0 and breaks_delta == 0
                and abs(seconds_delta) <= _TOLERANCE_SECONDS
                and base is not None and side is not None):
            continue
        bucket = _classify(base, side)
        shape = side or base
        key = (bucket, shape["daypart"])
        cell = cells.setdefault(key, {
            "bucket": bucket, "bucket_he": _BUCKET_HE[bucket],
            "daypart": shape["daypart"], "daypart_he": shape["daypart_he"],
            "revenue_delta_agorot": 0, "breaks_delta": 0,
            "ad_seconds_delta": 0.0, "segment_ids": [],
        })
        cell["revenue_delta_agorot"] += revenue_delta
        cell["breaks_delta"] += breaks_delta
        cell["ad_seconds_delta"] += seconds_delta
        cell["segment_ids"].append(segment_id)
        changed_segments.append({
            "segment_id": segment_id,
            "bucket": bucket,
            "programme": shape["programme"],
            "start_time": shape["start_time"],
            "daypart": shape["daypart"],
            "breaks_before": base["breaks"] if base else None,
            "breaks_after": side["breaks"] if side else None,
            "ad_seconds_before": round(base["ad_seconds"], 1) if base else None,
            "ad_seconds_after": round(side["ad_seconds"], 1) if side else None,
            "revenue_before": _ils(base["revenue_cents"]) if base else None,
            "revenue_after": _ils(side["revenue_cents"]) if side else None,
            "revenue_delta": _ils(revenue_delta),
        })

    total_agorot = sum(cell["revenue_delta_agorot"] for cell in cells.values())
    # The truth the attribution is measured against is the money the two frames
    # actually carry, NOT the money the segment-keyed diff managed to key. Taking
    # it from the index instead would make an unkeyable row disappear from both
    # sides of the check and the residue would always read zero.
    scoped_agorot = _revenue_agorot(side_rows) - _revenue_agorot(baseline_rows)
    residual = scoped_agorot - total_agorot
    if residual:
        # Never folded into a neighbouring bucket: an unexplained agora is a
        # finding about this code, and it is printed as one.
        cells[(UNATTRIBUTED, None)] = {
            "bucket": UNATTRIBUTED, "bucket_he": _BUCKET_HE[UNATTRIBUTED],
            "daypart": None, "daypart_he": _NO_DAYPART_HE,
            "revenue_delta_agorot": residual, "breaks_delta": 0,
            "ad_seconds_delta": 0.0, "segment_ids": [],
            "reason_he": "הפרש שנותר ללא שיוך לגורם; מדווח במפורש ואינו מגולגל לסעיף אחר",
        }
        total_agorot += residual

    ordered = sorted(cells.values(),
                     key=lambda cell: (-abs(cell["revenue_delta_agorot"]), cell["bucket"]))
    rendered = []
    for cell in ordered:
        rendered.append({
            **{key: value for key, value in cell.items() if key != "revenue_delta_agorot"},
            "revenue_delta": _ils(cell["revenue_delta_agorot"]),
            "ad_seconds_delta": round(cell["ad_seconds_delta"], 1),
            "segments": len(cell["segment_ids"]),
            "sentence_he": cell.get("reason_he") or _cell_sentence(
                cell["bucket"], cell["daypart_he"], cell["revenue_delta_agorot"],
                cell["breaks_delta"], cell["ad_seconds_delta"], len(cell["segment_ids"]),
            ),
        })

    buckets = []
    for bucket in BUCKETS:
        members = [cell for cell in ordered if cell["bucket"] == bucket]
        if not members:
            continue
        buckets.append({
            "bucket": bucket, "bucket_he": _BUCKET_HE[bucket],
            "revenue_delta": _ils(sum(cell["revenue_delta_agorot"] for cell in members)),
            "breaks_delta": sum(cell["breaks_delta"] for cell in members),
            "ad_seconds_delta": round(sum(cell["ad_seconds_delta"] for cell in members), 1),
            "segments": sum(len(cell["segment_ids"]) for cell in members),
            "dayparts": [cell["daypart"] for cell in members],
        })
    return {
        "available": True,
        "revenue_delta": _ils(total_agorot),
        "cells": rendered,
        "buckets": buckets,
        "changed_segments": changed_segments,
        "reconciliation": {
            "scoped_revenue_delta": _ils(scoped_agorot),
            "attributed_revenue_delta": _ils(total_agorot),
            "difference": _ils(residual if residual else 0),
            "exact": residual == 0,
            "unit": "agorot, summed as integers so the cells, the buckets and the total agree exactly",
        },
    }
