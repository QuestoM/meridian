"""Override validation and placement-pin logic for the Kairos optimizer.

Handles converting operator segment constraints (pin / force / forbid / gold)
and explicit placement pins into the per-segment floors, caps, gold flags, and
pin side-maps that the greedy allocator and refiner consume.  Any override or
pin that would breach a hard guardrail is rejected and recorded rather than
silently applied.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

from kairos.optimize.guardrails import Guardrails, is_compliant
from kairos.optimize._types import PlacementPin, ProgramSegment, RejectedOverride
from kairos.optimize._segment_math import _group_breaks, _segment_break_objects

_EPSILON = 1e-9


def _apply_segment_overrides(
    segs: list[ProgramSegment],
    groups: dict[tuple[str, str], list[ProgramSegment]],
    guardrails: Guardrails,
    constraints: dict[str, dict[str, object]],
) -> tuple[dict[str, int], dict[str, int], dict[str, bool], list[RejectedOverride]]:
    """Turn segment constraints into per-segment floors, caps, gold flags.

    Returns ``(floors, caps, gold_by_id, rejected)``. A forbid pins the cap to 0.
    A pin sets floor == cap == the pinned count. A force lifts the floor. Gold
    tags the segment's breaks. Any pin or force that exceeds ``max_breaks`` or
    that makes its channel-day infeasible at the requested floor is rejected and
    left out (the segment falls back to a 0 floor and its normal cap), so an
    infeasible override never breaches a hard guardrail.
    """
    floors: dict[str, int] = {s.segment_id: 0 for s in segs}
    caps: dict[str, int] = {s.segment_id: s.max_breaks for s in segs}
    gold_by_id: dict[str, bool] = {}
    rejected: list[RejectedOverride] = []

    for segment in segs:
        entry = constraints.get(segment.segment_id)
        if not entry:
            continue
        if entry.get("gold"):
            gold_by_id[segment.segment_id] = True
        if entry.get("forbid"):
            floors[segment.segment_id] = 0
            caps[segment.segment_id] = 0
            continue
        pin = entry.get("pin")
        if pin is not None:
            requested = int(pin)
            if requested > segment.max_breaks:
                rejected.append(RejectedOverride(
                    segment_id=segment.segment_id, kind="pin", requested=requested,
                    reason=f"pinned count {requested} exceeds max_breaks {segment.max_breaks}",
                ))
            else:
                floors[segment.segment_id] = requested
                caps[segment.segment_id] = requested
            continue
        minimum = entry.get("min")
        if minimum is not None:
            requested = int(minimum)
            if requested > segment.max_breaks:
                rejected.append(RejectedOverride(
                    segment_id=segment.segment_id, kind="force", requested=requested,
                    reason=f"forced minimum {requested} exceeds max_breaks {segment.max_breaks}",
                ))
            else:
                floors[segment.segment_id] = requested

    # Verify each channel-day is compliant at its floors. If a pinned or forced
    # floor makes the group breach a guardrail (for example spacing), back that
    # override out and report it, rather than ship an out-of-policy plan.
    for group in groups.values():
        _reject_infeasible_floors(group, floors, caps, gold_by_id, guardrails, constraints, rejected)

    return floors, caps, gold_by_id, rejected


def _placements_in_bounds(segment: ProgramSegment, pins: Sequence[PlacementPin]) -> Optional[str]:
    """Reason the pins are invalid for the segment, or None when they are valid.

    Each break must sit inside the segment (``0 <= offset`` and
    ``offset + duration <= segment.duration_seconds``) and, ordered by offset, no
    two breaks may overlap (``prev.offset + prev.duration <= next.offset``).
    """
    for pin in pins:
        if pin.duration_seconds <= 0:
            return f"placement duration {pin.duration_seconds} must be positive"
        if pin.offset_seconds < 0:
            return f"placement offset {pin.offset_seconds} is before the segment start"
        if pin.offset_seconds + pin.duration_seconds > segment.duration_seconds + _EPSILON:
            return (
                f"placement at {pin.offset_seconds}s + {pin.duration_seconds}s exceeds "
                f"segment duration {segment.duration_seconds}s"
            )
    ordered = sorted(pins, key=lambda p: p.offset_seconds)
    for previous, current in zip(ordered, ordered[1:]):
        if previous.offset_seconds + previous.duration_seconds > current.offset_seconds + _EPSILON:
            return "placements overlap within the segment"
    return None


def _apply_placement_pins(
    segs: list[ProgramSegment],
    groups: dict[tuple[str, str], list[ProgramSegment]],
    guardrails: Guardrails,
    placement_pins: Optional[Mapping[str, Sequence[PlacementPin]]],
    floors: dict[str, int],
    caps: dict[str, int],
    gold_by_id: dict[str, bool],
    rejected: list[RejectedOverride],
) -> dict[str, Sequence[PlacementPin]]:
    """Validate explicit placement pins and force the segments that carry them.

    For each pinned segment: validate the pins are in-bounds and non-overlapping,
    then run the guardrail checks on the pinned geometry for its channel-day. If
    either fails, the segment's pins are dropped (it falls back to a 0 floor and
    its normal cap) and a ``RejectedOverride(kind="placement")`` is recorded.
    A valid pin set forces ``floor == cap == len(pins)`` so every tier leaves the
    segment fixed, and is returned in the side map every emit / revenue path reads.
    """
    placements: dict[str, Sequence[PlacementPin]] = {}
    if not placement_pins:
        return placements

    seg_by_id = {s.segment_id: s for s in segs}
    for segment_id, pins in placement_pins.items():
        segment = seg_by_id.get(segment_id)
        if segment is None or not pins:
            continue
        reason = _placements_in_bounds(segment, pins)
        if reason is None:
            # Check the pinned geometry against the spacing / load guardrails in
            # isolation (per-segment); the channel-day check below catches breaches
            # that only show up once the whole group's pinned breaks are combined.
            probe = _segment_break_objects(segment, len(pins), pins=pins)
            if not is_compliant(probe, guardrails):
                reason = "pinned breaks breach a guardrail (spacing/load) for the segment"
        if reason is not None:
            rejected.append(RejectedOverride(
                segment_id=segment_id, kind="placement", requested=len(pins), reason=reason,
            ))
            # A rejected placement drops the segment to 0 breaks (the operator asked
            # for explicit geometry that cannot be honored; falling back to free
            # optimization would silently substitute different breaks).
            floors[segment_id] = 0
            caps[segment_id] = 0
            continue
        # Per-break gold lives on the individual PlacementPin (honored in the emit
        # path); it is NOT promoted to a segment-level gold flag, so a single gold
        # pin does not gild every break in the segment.
        placements[segment_id] = pins
        floors[segment_id] = len(pins)
        caps[segment_id] = len(pins)

    # Combined channel-day guardrail check: a group whose pinned floors breach a
    # guardrail has the largest pinned segment backed out one at a time until the
    # group's floor state is compliant, mirroring _reject_infeasible_floors.
    for group in groups.values():
        _reject_infeasible_placements(group, floors, caps, gold_by_id, guardrails, placements, rejected)
    return placements


def _reject_infeasible_placements(
    group: list[ProgramSegment],
    floors: dict[str, int],
    caps: dict[str, int],
    gold_by_id: dict[str, bool],
    guardrails: Guardrails,
    placements: dict[str, Sequence[PlacementPin]],
    rejected: list[RejectedOverride],
) -> None:
    """Back out pinned segments in a group until its floor geometry is compliant.

    Only placement-pinned segments are candidates (a non-pinned floor is left for
    the override path to handle), removed largest first, each recorded as a
    ``placement`` rejection.
    """
    state = {s.segment_id: floors[s.segment_id] for s in group}
    while not is_compliant(_group_breaks(group, state, gold_by_id, placements), guardrails):
        candidates = [s for s in group if s.segment_id in placements]
        if not candidates:
            break  # the infeasibility is not from a placement; leave it for reporting
        worst = max(candidates, key=lambda s: len(placements[s.segment_id]))
        rejected.append(RejectedOverride(
            segment_id=worst.segment_id, kind="placement", requested=len(placements[worst.segment_id]),
            reason="pinned breaks breach a guardrail for the channel-day (spacing/load)",
        ))
        del placements[worst.segment_id]
        state[worst.segment_id] = 0
        floors[worst.segment_id] = 0
        caps[worst.segment_id] = worst.max_breaks


def _reject_infeasible_floors(
    group: list[ProgramSegment],
    floors: dict[str, int],
    caps: dict[str, int],
    gold_by_id: dict[str, bool],
    guardrails: Guardrails,
    constraints: dict[str, dict[str, object]],
    rejected: list[RejectedOverride],
) -> None:
    """Back out pin/force floors in a group until its floor state is compliant.

    Removes the largest offending override-floored segment one at a time until
    the group's floors are guardrail-compliant, recording each as rejected. A
    forbid (cap 0) is never the cause and never backed out.
    """
    state = {s.segment_id: floors[s.segment_id] for s in group}
    while not is_compliant(_group_breaks(group, state, gold_by_id), guardrails):
        candidates = [
            s for s in group
            if state[s.segment_id] > 0 and _is_override_floored(s.segment_id, constraints)
        ]
        if not candidates:
            break  # the infeasibility is not from an override; leave it for reporting
        worst = max(candidates, key=lambda s: state[s.segment_id])
        entry = constraints.get(worst.segment_id, {})
        kind = "pin" if entry.get("pin") is not None else "force"
        rejected.append(RejectedOverride(
            segment_id=worst.segment_id, kind=kind, requested=state[worst.segment_id],
            reason="override floor breaks a guardrail for its channel-day (spacing/load)",
        ))
        state[worst.segment_id] = 0
        floors[worst.segment_id] = 0
        if kind == "pin":
            caps[worst.segment_id] = worst.max_breaks


def _is_override_floored(segment_id: str, constraints: dict[str, dict[str, object]]) -> bool:
    entry = constraints.get(segment_id, {})
    return entry.get("pin") is not None or entry.get("min") is not None
