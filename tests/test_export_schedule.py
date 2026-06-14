"""Tests for the weekly schedule exporter.

A synthetic multi-channel, multi-day frame keeps these in the fast gate while
still running the real classifier, pricing, transform and optimizer end to end,
and proving the output carries the real columns the dashboard reads (so its
placeholder fallback never has to fire).
"""

from __future__ import annotations

import pandas as pd

from kairos.export.schedule import COLUMNS, build_weekly_schedule, write_weekly_schedule


def make_frame() -> pd.DataFrame:
    rows = [
        ("חדשות הערב", "קשת 12", "2024-11-04 20:00:00", 3600, 6.0),
        ("התוכנית הראשונה", "קשת 12", "2024-11-04 21:00:00", 3600, 6.0),
        ("התוכנית השנייה", "קשת 12", "2024-11-04 22:00:00", 3600, 5.0),
        ("מהדורת חדשות", "רשת 13", "2024-11-05 20:00:00", 3600, 4.0),
        ("תוכנית אירוח", "רשת 13", "2024-11-05 21:00:00", 3600, 4.0),
    ]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "Duration", "TVR"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    return frame


def test_schedule_carries_the_columns_the_dashboard_reads() -> None:
    schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
    assert list(schedule.columns) == COLUMNS
    # Every channel-day is covered: 3 segments on Keshet, 2 on Reshet.
    assert len(schedule) == 5
    assert set(schedule["channel"]) == {"קשת 12", "רשת 13"}
    # Days are normalised to the dashboard's abbreviation key.
    assert set(schedule["day"]).issubset({"Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"})


def test_numbers_are_real_not_fabricated() -> None:
    schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
    # Revenue is earned, retention is a real probability, breaks are placed.
    assert schedule["predicted_revenue"].sum() > 0
    assert (schedule["num_breaks"] >= 0).all()
    assert schedule["num_breaks"].sum() > 0
    assert ((schedule["predicted_retention"] >= 0) & (schedule["predicted_retention"] <= 1)).all()
    # base_rate is the effective per-second rate (CPP times premium), always positive.
    assert (schedule["base_rate"] > 0).all()
    # total_break_time is num_breaks times the break length, never invented.
    expected = schedule["num_breaks"] * schedule["break_length"]
    assert (schedule["total_break_time"] - expected).abs().max() < 1e-6


def test_zero_revenue_weight_places_no_breaks_and_earns_nothing() -> None:
    schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=0.0)
    assert (schedule["num_breaks"] == 0).all()
    assert schedule["predicted_revenue"].sum() == 0
    # A programme with no breaks keeps its full baseline retention, never a guess.
    assert (schedule["predicted_retention"] == 1.0).all()


def test_break_type_is_derived_from_length() -> None:
    schedule = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
    # The default break length is 120s, which is the medium band.
    assert set(schedule["break_type"]).issubset({"short", "medium", "long"})
    assert (schedule["break_type"] == "medium").all()


def test_write_round_trips_through_csv(tmp_path) -> None:
    frame = build_weekly_schedule(programmes=make_frame(), revenue_weight=1.0)
    target = tmp_path / "weekly_break_schedule.csv"
    written = write_weekly_schedule(target, frame=frame)
    assert written == target
    reloaded = pd.read_csv(target)
    assert list(reloaded.columns) == COLUMNS
    assert len(reloaded) == len(frame)
