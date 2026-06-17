"""Tests for explicit per-segment break placement pins.

These prove the operator can pin exact breaks (absolute offset, per-break
duration, gold flag) and that the optimizer honors them as a hard constraint:
the pinned segment is fixed at the pinned count and emits breaks at exactly the
pinned positions and durations, with revenue summed over the variable per-break
durations. Invalid pins (out of bounds, overlapping, or guardrail-breaching) are
dropped to zero and reported in ``rejected_overrides`` rather than silently bent.
"""

from __future__ import annotations

import pandas as pd

from kairos.export.schedule import build_weekly_schedule
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.objective import break_revenue
from kairos.optimize.optimizer import PlacementPin, ProgramSegment, optimize_breaks
from kairos.optimize.overrides import Override, OverrideSet

GR = Guardrails()


def make_segment(**overrides) -> ProgramSegment:
    base = dict(
        segment_id="s1",
        channel="קשת 12",
        day="Mon",
        start_seconds=20 * 3600.0,   # 20:00
        duration_seconds=3600.0,     # one hour, long enough for two spaced breaks
        program_type="Drama",
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


def test_placement_pins_emit_exact_breaks_and_force_count() -> None:
    segment = make_segment()
    pins = [
        PlacementPin(offset_seconds=22 * 60, duration_seconds=90.0, is_gold=False),
        PlacementPin(offset_seconds=50 * 60, duration_seconds=120.0, is_gold=True),
    ]
    result = optimize_breaks(
        [segment], GR, revenue_weight=1.0,
        placement_pins={"s1": pins},
    )
    assert result.segments[0].num_breaks == 2  # count is forced, not re-optimized
    placements = sorted(result.placements, key=lambda p: p.start_seconds)
    assert len(placements) == 2
    # Absolute positions = segment.start_seconds + offset.
    assert placements[0].start_seconds == segment.start_seconds + 22 * 60
    assert placements[1].start_seconds == segment.start_seconds + 50 * 60
    # Per-break durations are honored.
    assert placements[0].duration_seconds == 90.0
    assert placements[1].duration_seconds == 120.0
    # Only the second break is gold.
    assert placements[0].is_gold is False
    assert placements[1].is_gold is True
    assert result.is_compliant
    assert not result.rejected_overrides


def test_pinned_count_is_not_re_optimized_even_revenue_only() -> None:
    # Revenue-only would normally fill to max_breaks (4); the 2 pins fix it at 2.
    segment = make_segment(max_breaks=4)
    pins = [
        PlacementPin(offset_seconds=22 * 60, duration_seconds=90.0),
        PlacementPin(offset_seconds=50 * 60, duration_seconds=120.0),
    ]
    result = optimize_breaks([segment], GR, revenue_weight=1.0, placement_pins={"s1": pins})
    assert result.segments[0].num_breaks == 2


def test_revenue_is_sum_of_per_break_durations() -> None:
    segment = make_segment(impact_coefficient=0.0)  # retention stays 1.0
    pins = [
        PlacementPin(offset_seconds=22 * 60, duration_seconds=90.0),
        PlacementPin(offset_seconds=50 * 60, duration_seconds=120.0),
    ]
    result = optimize_breaks([segment], GR, revenue_weight=1.0, placement_pins={"s1": pins})
    effective_tvr = segment.baseline_tvr  # retention 1.0 with zero impact
    expected = (
        break_revenue(effective_tvr, 90.0, segment.cpp, unit_seconds=1.0, premium=1.0)
        + break_revenue(effective_tvr, 120.0, segment.cpp, unit_seconds=1.0, premium=1.0)
    )
    assert result.segments[0].revenue == expected
    # And NOT 2 * one fixed length.
    two_fixed = 2 * break_revenue(
        effective_tvr, segment.break_length_seconds, segment.cpp, unit_seconds=1.0, premium=1.0,
    )
    assert result.segments[0].revenue != two_fixed


def test_out_of_bounds_pin_is_dropped_and_reported() -> None:
    segment = make_segment(duration_seconds=3600.0)
    # offset + duration = 3590 + 120 = 3710 > 3600, out of bounds.
    pins = [PlacementPin(offset_seconds=3590.0, duration_seconds=120.0)]
    result = optimize_breaks([segment], GR, revenue_weight=1.0, placement_pins={"s1": pins})
    assert result.segments[0].num_breaks == 0
    assert result.placements == ()
    rejected = [r for r in result.rejected_overrides if r.segment_id == "s1"]
    assert len(rejected) == 1
    assert rejected[0].kind == "placement"
    assert rejected[0].requested == 1


def test_overlapping_pins_are_rejected() -> None:
    segment = make_segment(duration_seconds=3600.0)
    # First break runs 0..600; second starts at 300, overlapping.
    pins = [
        PlacementPin(offset_seconds=0.0, duration_seconds=600.0),
        PlacementPin(offset_seconds=300.0, duration_seconds=120.0),
    ]
    result = optimize_breaks([segment], GR, revenue_weight=1.0, placement_pins={"s1": pins})
    assert result.segments[0].num_breaks == 0
    rejected = [r for r in result.rejected_overrides if r.segment_id == "s1"]
    assert len(rejected) == 1
    assert rejected[0].kind == "placement"


def test_program_title_threads_through_segment() -> None:
    segment = make_segment(program_title="The Late Show")
    result = optimize_breaks([segment], GR, revenue_weight=0.0)
    # program_title is a frozen-dataclass field; it does not affect optimization
    # but is carried so cross-date constraints can match on it later.
    assert segment.program_title == "The Late Show"
    assert result.is_compliant


def _tiny_programmes() -> pd.DataFrame:
    """A one-channel, one-day programmes frame the schedule path can optimise."""
    base = pd.Timestamp("2026-01-05 20:00:00")
    rows = [
        {"Title": "Evening Drama", "Channel": "קשת 12", "start_dt": base,
         "Duration": 3600.0, "TVR": 10.0},
    ]
    return pd.DataFrame(rows)


def test_weekly_schedule_honors_pin_override_count() -> None:
    programmes = _tiny_programmes()
    # First find the optimizer's default (no override) num_breaks for the segment.
    default_frame = build_weekly_schedule(programmes, overrides=OverrideSet(), revenue_weight=1.0)
    assert len(default_frame) == 1
    default_breaks = int(default_frame.iloc[0]["num_breaks"])

    # The single segment's id is f"{date}|{channel}|000".
    seg_id = "2026-01-05|קשת 12|000"
    pin_count = 1 if default_breaks != 1 else 2
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="segment", target_id=seg_id, kind="pin", value=str(pin_count)),
    ])
    pinned_frame = build_weekly_schedule(programmes, overrides=overrides, revenue_weight=1.0)
    assert len(pinned_frame) == 1
    assert int(pinned_frame.iloc[0]["num_breaks"]) == pin_count
    # The pin moved the count off the optimizer default (the override took effect).
    assert pin_count != default_breaks
