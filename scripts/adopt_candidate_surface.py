"""What a candidate would drop, gain or move beyond its coefficients.

Two artifacts can carry identical coefficients and still be different engine
inputs. Measured on this tree, the placebo-corrected candidate predicts exactly
what the shipped model predicts, break for break, and yet drops nine metadata
keys and the calibrated interval on every one of the 36 cells. A registry that
reported it as changing nothing would be reporting something false, and the
money measurement cannot always catch the loss: two artifacts can produce the
same plan today and price risk differently the moment an operator moves
risk_lambda.

Split out of ``adopt_candidate_adoption.py`` under the naming rule of section
8.2 when that file reached the 450-line cap.
"""

from __future__ import annotations

from typing import Any

# What the engine actually reads out of a coefficients artifact, each with the
# line that reads it, so this list is a measurement and not an opinion. A
# candidate that drops one of these is a narrower artifact than the one it would
# replace, and the money measurement cannot always see the loss: two artifacts
# can produce the same plan today and price risk differently the moment an
# operator moves risk_lambda.
ENGINE_READ_METADATA = {
    "first_break_multiplier": "kairos/service.py:117 folds it into the optimizer assumptions",
    "computed_at": "kairos/model/freshness.py:42 dates the operator's freshness banner",
    "source_fingerprints": "kairos/model/freshness.py:40 decides whether the plan is stale",
}

ENGINE_READ_DETAIL = {
    "coefficient": "kairos/model/impact.py:321 is the retention cost itself",
    "ci_low": "kairos/optimize/_segment_math.py:49 prices the risk_lambda decision",
    "ci_high": "kairos/optimize/_segment_math.py:49 prices the risk_lambda decision",
    "n": "kairos/model/impact.py:327 carries the weight behind the cell",
}


def _interval_moves(shipped_detail: dict[str, Any],
                    candidate_detail: dict[str, Any]) -> dict[str, Any]:
    """How far the credible bands move, cell by cell, kept apart from the points.

    A candidate can leave every point coefficient alone and still reprice the
    risk_lambda decision, because that decision reads the interval and not the
    point. This is reported whether or not the plan moves today, since today's
    plan is computed at one risk weight and the operator can move it.
    """
    moved, largest, largest_cell = 0, 0.0, None
    for cell, before in shipped_detail.items():
        after = candidate_detail.get(cell)
        if not isinstance(before, dict) or not isinstance(after, dict):
            continue
        for key in ("ci_low", "ci_high"):
            left, right = before.get(key), after.get(key)
            if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
                continue
            gap = abs(float(right) - float(left))
            if gap > 1e-12:
                moved += 1
            if gap > largest:
                largest, largest_cell = gap, f"{cell}.{key}"
    return {"bounds_moved": moved, "max_abs_move": round(largest, 9),
            "max_abs_move_at": largest_cell,
            "read_by": ENGINE_READ_DETAIL["ci_low"]}


def artifact_surface(shipped: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    """What the candidate would drop, gain or move, beyond its coefficients.

    Two artifacts can carry identical coefficients and still be different
    engine inputs. Measured on this tree, the placebo-corrected candidate
    predicts exactly what the shipped model predicts and drops nine metadata
    keys and the calibrated interval on every cell. A registry that reported it
    as changing nothing would be reporting something false.
    """
    shipped_meta = shipped.get("metadata") if isinstance(shipped.get("metadata"), dict) else {}
    candidate_meta = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
    shipped_detail = shipped.get("detail") if isinstance(shipped.get("detail"), dict) else {}
    candidate_detail = candidate.get("detail") if isinstance(candidate.get("detail"), dict) else {}

    detail_keys_before: set[str] = set()
    detail_keys_after: set[str] = set()
    for cell in shipped_detail.values():
        if isinstance(cell, dict):
            detail_keys_before |= set(cell)
    for cell in candidate_detail.values():
        if isinstance(cell, dict):
            detail_keys_after |= set(cell)

    metadata_dropped = sorted(set(shipped_meta) - set(candidate_meta))
    detail_dropped = sorted(detail_keys_before - detail_keys_after)
    engine_metadata = [key for key in metadata_dropped if key in ENGINE_READ_METADATA]
    engine_detail = [key for key in detail_dropped if key in ENGINE_READ_DETAIL]
    return {
        "intervals": _interval_moves(shipped_detail, candidate_detail),
        "metadata_dropped": metadata_dropped,
        "metadata_added": sorted(set(candidate_meta) - set(shipped_meta)),
        "detail_fields_dropped": detail_dropped,
        "detail_fields_added": sorted(detail_keys_after - detail_keys_before),
        "cells_dropped": sorted(set(shipped.get("coefficients") or {}) - set(candidate.get("coefficients") or {})),
        "engine_inputs_dropped": [
            {"field": key, "read_by": ENGINE_READ_METADATA[key], "where": "metadata"}
            for key in engine_metadata
        ] + [
            {"field": key, "read_by": ENGINE_READ_DETAIL[key], "where": "detail"}
            for key in engine_detail
        ],
    }
