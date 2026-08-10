"""Measured WITH-vs-WITHOUT effects for stored constraints or one draft."""

from __future__ import annotations

from typing import Any, Optional, Sequence

from kairos.optimize.constraints_store import PlacementConstraint, resolve_constraints
from kairos_api import constraints_sentence


def measure(
    constraints: Sequence[PlacementConstraint],
    channel: Optional[str] = None,
    day: Optional[str] = None,
    daily_input: Optional[str] = None,
) -> dict[str, Any]:
    """Run the commit path once without and once with the supplied constraints."""
    from kairos.optimize.day_core import _optimize_one_day
    from kairos_api.overrides import _resolved_store_overrides
    from kairos_api.preview_inputs import preview_inputs

    try:
        segments, engine_kwargs = preview_inputs(channel, day, daily_input)
    except Exception:  # pragma: no cover - data/environment dependent
        raise constraints_sentence.refuse("segments_failed", 503)
    if not segments:
        raise constraints_sentence.refuse("no_segments", 404)

    placement_pins, count_pins, forbids, skipped = resolve_constraints(
        segments, constraints, operator_channel=engine_kwargs["operator_channel"],
    )
    active_overrides, _stale = _resolved_store_overrides(segments)
    stored = active_overrides if active_overrides.overrides else None
    baseline = _optimize_one_day(segments, overrides=stored, **engine_kwargs)
    constrained = _optimize_one_day(
        segments, constraints=constraints, overrides=stored, **engine_kwargs,
    )
    base_counts = {segment.segment_id: segment.num_breaks for segment in baseline.segments}
    new_counts = {segment.segment_id: segment.num_breaks for segment in constrained.segments}
    changed = [
        {
            "segment_id": segment_id,
            "before": base_counts.get(segment_id, 0),
            "after": new_counts.get(segment_id, 0),
        }
        for segment_id in sorted(new_counts)
        if base_counts.get(segment_id, 0) != new_counts.get(segment_id, 0)
    ]
    return {
        "channel": channel,
        "day": day,
        "summary": {
            "before_total_breaks": baseline.total_breaks,
            "after_total_breaks": constrained.total_breaks,
            "before_revenue": round(baseline.total_revenue, 2),
            "after_revenue": round(constrained.total_revenue, 2),
            "changed_segments": len(changed),
            "matched_segments": len(set(placement_pins) | set(count_pins) | forbids),
        },
        "changed": changed,
        "skipped_constraints": [
            {"constraint_id": item.constraint_id, "segment_id": item.segment_id, "reason": item.reason}
            for item in skipped
        ],
        "rejected_overrides": [
            {"segment_id": item.segment_id, "kind": item.kind, "requested": item.requested, "reason": item.reason}
            for item in constrained.rejected_overrides
        ],
    }
