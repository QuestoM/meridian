"""Tests for the unified scoped placement-constraint store and resolver.

These prove an operator can pin / shape / forbid breaks by SCOPE (programme,
date, weekday, channel, always) and that each scoped constraint is translated
into the optimizer's own primitives (placement pins, count pins, forbids) and
honored exactly. They also prove conflicting constraints are skipped with a
reason rather than letting the last writer silently win, and that the weekly
export honors a constraints CSV end to end.
"""

from __future__ import annotations

import pandas as pd

from kairos.export.schedule import build_weekly_schedule
from kairos.optimize.constraints_store import (
    COLUMNS,
    PlacementConstraint,
    count_pins_to_overrides,
    load_constraints,
    resolve_constraints,
)
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks

GR = Guardrails()


def make_segment(**overrides) -> ProgramSegment:
    base = dict(
        segment_id="s1",
        channel="קשת 12",
        day="2026-06-15",            # a Monday
        start_seconds=20 * 3600.0,   # 20:00
        duration_seconds=3600.0,     # one hour, room for spaced breaks
        program_type="Drama",
        program_title="חתונה ממבט ראשון",
        baseline_tvr=10.0,
        cpp=1000.0,
        impact_coefficient=0.0,
        retention_baseline=1.0,
        premium=1.0,
        is_gold=False,
        max_breaks=4,
        break_length_seconds=120.0,
        unit_seconds=1.0,
    )
    base.update(overrides)
    return ProgramSegment(**base)


def test_fix_offset_programme_scope_pins_the_break() -> None:
    segment = make_segment()
    constraint = PlacementConstraint(
        constraint_id="c1", scope_type="programme",
        scope_value="חתונה ממבט ראשון", effect="fix_offset",
        offset_seconds=22 * 60, duration_seconds=90.0, order_index=1,
    )
    pins, counts, forbids, skipped = resolve_constraints([segment], [constraint])
    assert not skipped
    assert "s1" in pins and len(pins["s1"]) == 1
    assert pins["s1"][0].offset_seconds == 22 * 60
    assert pins["s1"][0].duration_seconds == 90.0

    result = optimize_breaks([segment], GR, revenue_weight=1.0, placement_pins=pins)
    assert result.segments[0].num_breaks == 1   # count forced to the pin count
    placement = result.placements[0]
    assert placement.start_seconds == segment.start_seconds + 22 * 60
    assert placement.duration_seconds == 90.0
    assert result.is_compliant


def test_fix_offset_programme_scope_does_not_match_other_titles() -> None:
    other = make_segment(segment_id="s2", program_title="חדשות הערב")
    constraint = PlacementConstraint(
        constraint_id="c1", scope_type="programme",
        scope_value="חתונה ממבט ראשון", effect="fix_offset", offset_seconds=22 * 60,
    )
    pins, _, _, _ = resolve_constraints([other], [constraint])
    assert pins == {}


def test_pin_count_date_scope_forces_count_on_that_date_only() -> None:
    on_date = make_segment(segment_id="s_mon", day="2026-06-15")
    off_date = make_segment(segment_id="s_tue", day="2026-06-16")
    constraint = PlacementConstraint(
        constraint_id="c1", scope_type="date", scope_value="2026-06-15",
        effect="pin_count", count=3,
    )
    pins, counts, forbids, skipped = resolve_constraints([on_date, off_date], [constraint])
    assert counts == {"s_mon": 3}
    assert "s_tue" not in counts

    overrides = count_pins_to_overrides(counts, forbids)
    result = optimize_breaks([on_date, off_date], GR, revenue_weight=1.0, overrides=overrides)
    by_id = {s.segment_id: s.num_breaks for s in result.segments}
    assert by_id["s_mon"] == 3              # forced on the matched date
    assert by_id["s_tue"] == on_date.max_breaks  # untouched, free optimization


def test_weekday_scope_gold_flags_matching_weekday_only() -> None:
    # 2026-06-15 is a Monday (isoweekday 1); 2026-06-17 is a Wednesday (3).
    monday = make_segment(segment_id="s_mon", day="2026-06-15")
    wednesday = make_segment(segment_id="s_wed", day="2026-06-17")
    gold = PlacementConstraint(
        constraint_id="g1", scope_type="weekday", scope_value="1", effect="gold",
    )
    # A position pin gives gold a break to gild on the Monday segment.
    pin = PlacementConstraint(
        constraint_id="p1", scope_type="weekday", scope_value="1",
        effect="fix_offset", offset_seconds=30 * 60,
    )
    pins, _, _, _ = resolve_constraints([monday, wednesday], [gold, pin])
    assert "s_mon" in pins and pins["s_mon"][0].is_gold is True
    assert "s_wed" not in pins                      # weekday 3 does not match


def test_forbid_always_scope_zeroes_breaks() -> None:
    segment = make_segment()
    constraint = PlacementConstraint(
        constraint_id="c1", scope_type="always", effect="forbid",
    )
    pins, counts, forbids, skipped = resolve_constraints([segment], [constraint])
    assert "s1" in forbids and counts["s1"] == 0

    overrides = count_pins_to_overrides(counts, forbids)
    result = optimize_breaks([segment], GR, revenue_weight=1.0, overrides=overrides)
    assert result.segments[0].num_breaks == 0


def test_resolve_skips_conflicting_pair_with_reason() -> None:
    segment = make_segment()
    a = PlacementConstraint(
        constraint_id="a", scope_type="always", effect="fix_offset",
        offset_seconds=10 * 60, order_index=1,
    )
    b = PlacementConstraint(
        constraint_id="b", scope_type="always", effect="fix_offset",
        offset_seconds=40 * 60, order_index=1,   # same break order -> conflict
    )
    pins, counts, forbids, skipped = resolve_constraints([segment], [a, b])
    assert len(pins["s1"]) == 1                  # only the first won
    assert pins["s1"][0].offset_seconds == 10 * 60
    assert len(skipped) == 1
    assert skipped[0].constraint_id == "b"
    assert "conflicting position" in skipped[0].reason


def test_columns_round_trip_through_csv(tmp_path) -> None:
    path = tmp_path / "kairos_constraints.csv"
    row = {column: "" for column in COLUMNS}
    row.update({
        "constraint_id": "c1", "scope_type": "channel", "scope_value": "קשת 12",
        "effect": "pin_count", "count": "2",
    })
    pd.DataFrame([row])[list(COLUMNS)].to_csv(path, index=False, encoding="utf-8-sig")
    loaded = load_constraints(path)
    assert len(loaded) == 1
    assert loaded[0].scope_type == "channel"
    assert loaded[0].count == 2


def test_build_weekly_schedule_honors_constraints_csv(tmp_path) -> None:
    # Build two real synthetic segments via a programmes frame: same channel-day,
    # so the FORBID-by-channel constraint zeroes both vs the unconstrained run.
    frame = pd.DataFrame({
        "Channel": ["קשת 12", "קשת 12"],
        "Title": ["דרמה א", "דרמה ב"],
        "start_dt": pd.to_datetime(["2026-06-15 20:00", "2026-06-15 21:30"]),
        "Duration": [3600.0, 3600.0],
        "TVR": [10.0, 8.0],
    })

    without = build_weekly_schedule(programmes=frame, revenue_weight=1.0)
    assert without["num_breaks"].sum() > 0       # some breaks placed unconstrained

    path = tmp_path / "kairos_constraints.csv"
    row = {column: "" for column in COLUMNS}
    row.update({
        "constraint_id": "f1", "scope_type": "channel",
        "scope_value": "קשת 12", "effect": "forbid",
    })
    pd.DataFrame([row])[list(COLUMNS)].to_csv(path, index=False, encoding="utf-8-sig")

    with_constraints = build_weekly_schedule(
        programmes=frame, revenue_weight=1.0, constraints_path=str(path),
    )
    assert int(with_constraints["num_breaks"].sum()) == 0   # forbid zeroed the channel
