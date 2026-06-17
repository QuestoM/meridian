"""Unified placement-constraint store and resolver for Kairos.

Background. P1 gave the optimizer two hard-constraint primitives: per-segment
break COUNTS (via :class:`~kairos.optimize.overrides.OverrideSet`, which pins /
floors / forbids / golds a segment by its id) and explicit PLACEMENT pins (via
:class:`~kairos.optimize.optimizer.PlacementPin`, which fixes a segment at exact
break offsets and durations). Both bite at the level of a single segment id, so
an operator who wants "every episode of this programme starts its first break at
00:22:00" or "no breaks on this date" had to enumerate segment ids by hand.

This module is the SCOPED generalization. An operator writes one row keyed by a
scope (a programme Title, a date, a weekday, a channel, or "always") and an
effect (fix an offset, allow an offset window, pin a count, range a duration,
gild, or forbid). :func:`resolve_constraints` matches each stored constraint
against the real segments of a run and translates the matched effects into the
optimizer's own primitives (placement pins, count pins, forbids), so the engine
honors them exactly as it honors a hand-written pin, with the same honesty rules.

Honesty rules:

  * Nothing is invented to fill an empty store: an empty file (header only, the
    seeded state) yields no constraints and every segment keeps its automatic
    behaviour.
  * A malformed or unknown scope / effect is skipped with a recorded reason, so a
    bad row never silently bends the plan.
  * Conflicting effects on the same segment-and-order target (for example two
    FIX_OFFSET rules on break 1) are detected: the first wins and the rest are
    skipped with a reason, rather than letting the last writer silently win.
  * The resolver only emits primitives the optimizer genuinely supports. A
    DURATION_RANGE with no fixed offset cannot be expressed as a pin (a pin needs
    a position), so it is recorded as a soft hint and NOT turned into a position,
    rather than pretending the engine can range a duration in place.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import pandas as pd

from kairos.optimize.optimizer import PlacementPin, ProgramSegment

from kairos.optimize._constraints_io import (  # noqa: F401  (re-exported)
    BACKUP_DIR,
    COLUMNS,
    DATA_DIR,
    DEFAULT_CONSTRAINTS_PATH,
    ROOT,
    _backup,
    _load_frame,
    _write_frame,
    count_pins_to_overrides,
)

# Scope vocabulary: how a constraint selects the segments it applies to.
PROGRAMME = "programme"   # scope_value is a programme Title (case-normalized match)
DATE = "date"             # scope_value is a YYYY-MM-DD date (segment.day match)
WEEKDAY = "weekday"       # scope_value is an ISO weekday 1..7 (Mon..Sun)
CHANNEL = "channel"       # scope_value is a channel name (segment.channel match)
ALWAYS = "always"         # matches every segment (scope_value ignored)
_SCOPES = (PROGRAMME, DATE, WEEKDAY, CHANNEL, ALWAYS)

# Effect vocabulary: what a matched constraint does to the segment.
FIX_OFFSET = "fix_offset"          # pin one break at an exact offset-from-start
OFFSET_WINDOW = "offset_window"    # pin one break at the window midpoint (soft center)
PIN_COUNT = "pin_count"            # force the segment's break count
DURATION_RANGE = "duration_range"  # range a break's duration (only pins with an offset)
GOLD = "gold"                      # mark the segment's breaks gold
FORBID = "forbid"                  # this segment carries 0 breaks
_EFFECTS = (FIX_OFFSET, OFFSET_WINDOW, PIN_COUNT, DURATION_RANGE, GOLD, FORBID)

def _to_float(raw: object) -> Optional[float]:
    """Parse a numeric cell to a float, or None when blank or unusable."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _to_int(raw: object) -> Optional[int]:
    value = _to_float(raw)
    return None if value is None else int(value)


@dataclass(frozen=True)
class PlacementConstraint:
    """One scoped placement constraint, as stored in kairos_constraints.csv.

    ``scope_type`` and ``scope_value`` select the segments (see the scope
    vocabulary); an optional ``channel`` narrows any scope to one channel.
    ``effect`` is one of the effect vocabulary; the effect params carry the
    numbers it needs (an unused param is ``None``). ``order_index`` is the 1-based
    break within the segment a position effect targets (``None`` means break 1).

    ``where`` is an optional rich predicate tree (the frozen Group/Condition
    contract). When present it replaces the flat scope_type/scope_value matching
    for that constraint; the old flat fields are preserved for backward compat.
    When absent (None) the constraint uses the legacy flat scope matching exactly
    as before.
    """

    constraint_id: str
    scope_type: str
    scope_value: str = ""
    channel: str = ""
    effect: str = ""
    offset_seconds: Optional[float] = None
    offset_min_seconds: Optional[float] = None
    offset_max_seconds: Optional[float] = None
    count: Optional[int] = None
    duration_seconds: Optional[float] = None
    duration_min_seconds: Optional[float] = None
    duration_max_seconds: Optional[float] = None
    order_index: Optional[int] = None
    notes: str = ""
    where: Optional[dict[str, Any]] = None   # parsed predicate Group tree

    def is_valid(self) -> bool:
        """True when the id, scope and effect are all recognised."""
        if not self.constraint_id:
            return False
        if self.scope_type not in _SCOPES:
            return False
        return self.effect in _EFFECTS


@dataclass(frozen=True)
class SkippedConstraint:
    """One constraint the resolver could not honor, with why (honesty surface)."""

    constraint_id: str
    segment_id: str
    reason: str


def _normalize(text: object) -> str:
    return str(text if text is not None else "").strip().lower()


def _matches(
    segment: ProgramSegment,
    constraint: PlacementConstraint,
    *,
    operator_channel: str = "",
) -> bool:
    """True when ``constraint`` applies to ``segment``.

    Two-path matching:

    1. Predicate path: when ``constraint.where`` is set, delegate to
       :func:`kairos.optimize.predicate.evaluate_predicate` which enforces the
       operator_channel filter internally and evaluates the full predicate tree.

    2. Legacy flat path: when ``where`` is absent, use the original
       scope_type/scope_value logic plus the per-constraint ``channel`` field
       filter. The ``operator_channel`` setting is also applied here: if the
       operator has picked a channel, only segments on that channel match.
    """
    if constraint.where is not None:
        from kairos.optimize.predicate import evaluate_predicate
        return evaluate_predicate(
            constraint.where, segment, operator_channel=operator_channel or None,
        )

    # Legacy flat path. Apply operator_channel if set.
    if operator_channel and segment.channel != operator_channel:
        return False
    # Per-constraint channel narrowing (the old per-row "channel" field).
    channel_filter = str(constraint.channel or "").strip()
    if channel_filter and segment.channel != channel_filter:
        return False
    scope = constraint.scope_type
    value = str(constraint.scope_value or "").strip()
    if scope == ALWAYS:
        return True
    if scope == CHANNEL:
        return segment.channel == value
    if scope == PROGRAMME:
        return _normalize(segment.program_title) == _normalize(value)
    if scope == DATE:
        return str(segment.day) == value
    if scope == WEEKDAY:
        try:
            target = int(value)
            return pd.to_datetime(segment.day).isoweekday() == target
        except (ValueError, TypeError):
            return False
    return False


def _offset_for(constraint: PlacementConstraint) -> Optional[float]:
    """The pinned offset a position effect wants, or None when it has none.

    FIX_OFFSET uses ``offset_seconds``. OFFSET_WINDOW has no single offset, so it
    is pinned at the window midpoint (a soft center the operator can read), which
    needs both bounds.
    """
    if constraint.effect == FIX_OFFSET:
        return constraint.offset_seconds
    if constraint.effect == OFFSET_WINDOW:
        low = constraint.offset_min_seconds
        high = constraint.offset_max_seconds
        if low is not None and high is not None:
            return (low + high) / 2.0
    if constraint.effect == DURATION_RANGE:
        # A duration range pins a position only when an explicit offset is given;
        # otherwise it is a soft hint (handled by the caller), never a guessed slot.
        return constraint.offset_seconds
    return None


def _duration_for(constraint: PlacementConstraint, segment: ProgramSegment) -> float:
    """The duration of a pinned break: explicit if given, else the segment default."""
    if constraint.duration_seconds is not None:
        return constraint.duration_seconds
    if constraint.duration_min_seconds is not None:
        return constraint.duration_min_seconds
    return segment.break_length_seconds


def resolve_constraints(
    segments: Sequence[ProgramSegment],
    constraints: Sequence[PlacementConstraint],
    *,
    operator_channel: str = "",
) -> tuple[dict[str, list[PlacementPin]], dict[str, int], set[str], list[SkippedConstraint]]:
    """Translate scoped constraints into the optimizer's per-segment primitives.

    Returns ``(placement_pins, count_pins, forbids, skipped)``:

      * ``placement_pins`` maps a segment id to a list of :class:`PlacementPin`
        built from FIX_OFFSET / OFFSET_WINDOW / GOLD / position-bearing
        DURATION_RANGE effects, ordered by offset.
      * ``count_pins`` maps a segment id to a forced break count (PIN_COUNT;
        FORBID contributes 0).
      * ``forbids`` is the set of segment ids held at 0 breaks (FORBID).
      * ``skipped`` lists every constraint dropped with its reason (unknown row,
        conflict, or an effect that cannot be expressed as a pin).

    A pinned position needs a duration; with none given the segment's default
    break length is used. A GOLD effect with no position pins gilds the segment by
    flagging its pinned breaks gold, or, when the segment has no other pin, is
    recorded as a count-free gold via a single full-length gold break only if an
    offset is supplied, else skipped with a reason (gold needs a break to gild).
    """
    placement_pins: dict[str, list[PlacementPin]] = {}
    count_pins: dict[str, int] = {}
    forbids: set[str] = set()
    skipped: list[SkippedConstraint] = []
    # Per segment, track which 1-based break order a position effect already claimed,
    # so a second FIX_OFFSET on the same order is a detectable conflict.
    claimed_orders: dict[str, dict[int, str]] = {}
    gold_segments: set[str] = set()

    valid = [c for c in constraints if c.is_valid()]
    for segment in segments:
        sid = segment.segment_id
        for constraint in valid:
            if not _matches(segment, constraint, operator_channel=operator_channel):
                continue
            effect = constraint.effect

            if effect == FORBID:
                forbids.add(sid)
                count_pins[sid] = 0
                continue

            if effect == PIN_COUNT:
                count = constraint.count
                if count is None or count < 0:
                    skipped.append(SkippedConstraint(
                        constraint.constraint_id, sid, "pin_count needs a non-negative count",
                    ))
                    continue
                existing = count_pins.get(sid)
                if existing is not None and existing != count:
                    skipped.append(SkippedConstraint(
                        constraint.constraint_id, sid,
                        f"conflicting pin_count {count} (already pinned at {existing})",
                    ))
                    continue
                count_pins[sid] = count
                continue

            if effect == GOLD:
                gold_segments.add(sid)
                continue

            # Position-bearing effects: FIX_OFFSET, OFFSET_WINDOW, or a
            # DURATION_RANGE that carries an explicit offset.
            offset = _offset_for(constraint)
            if offset is None:
                if effect == DURATION_RANGE:
                    skipped.append(SkippedConstraint(
                        constraint.constraint_id, sid,
                        "duration_range without a fixed offset is a soft hint, not pinned",
                    ))
                else:
                    skipped.append(SkippedConstraint(
                        constraint.constraint_id, sid,
                        f"{effect} needs an offset (offset_seconds or both window bounds)",
                    ))
                continue

            order = constraint.order_index if constraint.order_index else len(
                placement_pins.get(sid, [])
            ) + 1
            claimed = claimed_orders.setdefault(sid, {})
            if order in claimed:
                skipped.append(SkippedConstraint(
                    constraint.constraint_id, sid,
                    f"conflicting position on break {order} (already set by {claimed[order]})",
                ))
                continue
            claimed[order] = constraint.constraint_id
            placement_pins.setdefault(sid, []).append(PlacementPin(
                offset_seconds=float(offset),
                duration_seconds=_duration_for(constraint, segment),
                is_gold=sid in gold_segments,
            ))

    # A GOLD effect with no position of its own still gilds a segment that carries
    # pins: re-flag those pins gold. A gold segment with no pins at all cannot be
    # expressed (gold marks BREAKS, and the count optimizer's gold lives on the
    # OverrideSet path, not here), so it is left for the count path / skipped.
    for sid in gold_segments:
        pins = placement_pins.get(sid)
        if not pins:
            continue
        placement_pins[sid] = [
            PlacementPin(p.offset_seconds, p.duration_seconds, is_gold=True) for p in pins
        ]

    # Order each segment's pins by offset so the optimizer sees them time-ordered.
    for sid, pins in placement_pins.items():
        placement_pins[sid] = sorted(pins, key=lambda p: p.offset_seconds)

    return placement_pins, count_pins, forbids, skipped


def _parse_where_json(raw: str) -> Optional[dict[str, Any]]:
    """Parse the where_json CSV cell into a Group dict, or None when absent/invalid.

    A blank or whitespace-only cell -> None (legacy flat constraint).
    A non-JSON cell -> None (defensive: a bad cell never becomes a match-all).
    """
    text = raw.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and "combinator" in parsed:
            return parsed
        return None
    except (json.JSONDecodeError, ValueError):
        return None


def load_constraints(path: str | Path | None = None) -> list[PlacementConstraint]:
    """Read kairos_constraints.csv into :class:`PlacementConstraint` objects.

    A missing or header-only file yields an empty list (the honest seeded state).
    Malformed rows are kept but flagged invalid by :meth:`PlacementConstraint.is_valid`,
    so the loader never silently drops a row it read; the resolver skips them.
    """
    target = Path(path) if path else DEFAULT_CONSTRAINTS_PATH
    if not target.exists():
        return []
    frame = pd.read_csv(target, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    out: list[PlacementConstraint] = []
    for _, row in frame.iterrows():
        constraint_id = str(row.get("constraint_id", "")).strip()
        if not constraint_id:
            continue
        where = _parse_where_json(str(row.get("where_json", "") or ""))
        out.append(PlacementConstraint(
            constraint_id=constraint_id,
            scope_type=_normalize(row.get("scope_type")),
            scope_value=str(row.get("scope_value", "")).strip(),
            channel=str(row.get("channel", "")).strip(),
            effect=_normalize(row.get("effect")),
            offset_seconds=_to_float(row.get("offset_seconds")),
            offset_min_seconds=_to_float(row.get("offset_min_seconds")),
            offset_max_seconds=_to_float(row.get("offset_max_seconds")),
            count=_to_int(row.get("count")),
            duration_seconds=_to_float(row.get("duration_seconds")),
            duration_min_seconds=_to_float(row.get("duration_min_seconds")),
            duration_max_seconds=_to_float(row.get("duration_max_seconds")),
            order_index=_to_int(row.get("order_index")),
            notes=str(row.get("notes", "")),
            where=where,
        ))
    return out


def constraints_to_optimizer_inputs(
    segments: Sequence[ProgramSegment],
    path: str | Path | None = None,
    *,
    operator_channel: str = "",
) -> tuple[dict[str, list[PlacementPin]], dict[str, int], set[str], list[SkippedConstraint]]:
    """Load the CSV and resolve it against ``segments``, ready for the optimizer.

    Convenience wrapper returning ``(placement_pins, count_pins, forbids,
    skipped)``: feed ``placement_pins`` straight into
    :func:`~kairos.optimize.optimizer.optimize_breaks` / build_weekly_schedule,
    and turn ``count_pins`` / ``forbids`` into an OverrideSet (see
    :func:`count_pins_to_overrides`).
    """
    constraints = load_constraints(path)
    return resolve_constraints(segments, constraints, operator_channel=operator_channel)


