"""Tests for the Meridian data-preparation helpers.

These use small synthetic frames so the pure-pandas helpers (bucketing, break
detection, pricing-class mapping, retention) are proven without Meridian, xarray
or the real xlsx. Tests that need the real data or the Meridian object are marked
``realdata`` so the fast gate can skip them with ``-m "not realdata"``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.model.prepare import (
    build_meridian_input_data,
    daily_retention,
    identify_breaks,
    keyed_breaks,
    length_bucket,
    match_break_to_programme,
    meridian_available,
    position_bucket,
    pricing_class_lookup,
)


# --- bucketing thresholds ---------------------------------------------------

@pytest.mark.parametrize(
    "ratio, expected",
    [
        (0.0, "first"),
        (0.33, "first"),
        (0.34, "middle"),
        (0.66, "middle"),
        (0.67, "last"),
        (1.0, "last"),
    ],
)
def test_position_bucket(ratio: float, expected: str) -> None:
    assert position_bucket(ratio) == expected


@pytest.mark.parametrize(
    "seconds, expected",
    [
        (0.0, "short"),
        (89.9, "short"),
        (90.0, "standard"),
        (179.9, "standard"),
        (180.0, "long"),
        (600.0, "long"),
    ],
)
def test_length_bucket(seconds: float, expected: str) -> None:
    assert length_bucket(seconds) == expected


# --- break detection on a tiny synthetic spots frame ------------------------

def _spot(channel: str, start: str, duration: float) -> dict:
    return {"Channel": channel, "air_dt": pd.Timestamp(start), "Duration": duration}


def test_identify_breaks_groups_adjacent_and_drops_singletons() -> None:
    # Two spots 10s apart (one break), then a 60s gap to a lone spot (dropped).
    rows = [
        _spot("A", "2024-11-04 20:00:00", 30),   # ends 20:00:30
        _spot("A", "2024-11-04 20:00:40", 30),   # starts 10s after the 30s spot ends
        _spot("A", "2024-11-04 20:05:00", 30),   # 4+ min later, lone spot -> dropped
    ]
    breaks = identify_breaks(pd.DataFrame(rows))
    assert len(breaks) == 1
    only = breaks.iloc[0]
    assert only["channel"] == "A"
    assert only["num_spots"] == 2
    # 20:00:00 -> 20:01:10 is 70 seconds.
    assert only["break_seconds"] == 70.0


def test_identify_breaks_splits_on_large_gap() -> None:
    rows = [
        _spot("A", "2024-11-04 20:00:00", 30),
        _spot("A", "2024-11-04 20:00:35", 30),   # 5s gap -> same break
        _spot("A", "2024-11-04 21:00:00", 30),   # 1h later -> new run
        _spot("A", "2024-11-04 21:00:35", 30),   # 5s gap -> joins the second break
    ]
    breaks = identify_breaks(pd.DataFrame(rows))
    assert len(breaks) == 2


def test_identify_breaks_empty_frame() -> None:
    breaks = identify_breaks(pd.DataFrame(columns=["Channel", "air_dt", "Duration"]))
    assert breaks.empty


# --- pricing-class mapping (reuses transform._pricing_classes) ---------------

def _programmes() -> pd.DataFrame:
    rows = [
        ("חדשות הערב", "A", "2024-11-04 20:00:00", "2024-11-04 21:00:00", 3600.0),
        ("התוכנית הראשונה", "A", "2024-11-04 21:00:00", "2024-11-04 22:00:00", 3600.0),
        ("התוכנית השנייה", "A", "2024-11-04 22:00:00", "2024-11-04 23:00:00", 3600.0),
        ("תוכנית רביעית", "A", "2024-11-04 23:00:00", "2024-11-05 00:00:00", 3600.0),
    ]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "end", "Duration"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    frame["end_dt"] = pd.to_datetime(frame["end"])
    return frame


def test_pricing_class_lookup_sequence_after_news() -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    lookup = pricing_class_lookup(_programmes(), classifier)
    records = lookup[("A", "2024-11-04")]
    classes = [r["pricing_class"] for r in records]
    # News, then the first two main shows, then Other.
    assert classes == ["News", "PrimeShow1", "PrimeShow2", "Other"]


def test_match_break_to_programme_reads_class_and_position() -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    lookup = pricing_class_lookup(_programmes(), classifier)
    # A break 6 minutes into the 60-minute news programme: ratio 0.1 -> first.
    pricing_class, position = match_break_to_programme(
        "A",
        pd.Timestamp("2024-11-04 20:06:00"),
        pd.Timestamp("2024-11-04 20:07:00"),
        lookup,
    )
    assert pricing_class == "News"
    assert position == "first"


def test_match_break_unmatched_is_other_middle() -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    lookup = pricing_class_lookup(_programmes(), classifier)
    # A break on a day with no programmes is honestly Other/middle.
    pricing_class, position = match_break_to_programme(
        "A",
        pd.Timestamp("2024-12-25 10:00:00"),
        pd.Timestamp("2024-12-25 10:01:00"),
        lookup,
    )
    assert pricing_class == "Other"
    assert position == "middle"


def test_keyed_breaks_emits_engine_channel_names() -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    # A two-spot, ~80s break early in the news programme.
    spots = pd.DataFrame([
        _spot("A", "2024-11-04 20:06:00", 30),
        _spot("A", "2024-11-04 20:06:40", 40),   # ends 20:07:20 -> 80s break, < 90 short
    ])
    keyed = keyed_breaks(spots, _programmes(), classifier)
    assert len(keyed) == 1
    row = keyed.iloc[0]
    assert row["program_type"] == "News"
    assert row["break_position"] == "first"
    assert row["break_length"] == "short"
    assert row["channel_name"] == "News_first_short"
    assert row["day"] == "2024-11-04"


# --- daily retention (real ratio, no fabrication) ---------------------------

def test_daily_retention_is_mean_over_peak() -> None:
    rows = [
        {"date": pd.Timestamp("2024-11-04"), "channel": "A", "tvr": 2.0},
        {"date": pd.Timestamp("2024-11-04"), "channel": "A", "tvr": 4.0},  # peak 4, mean 3 -> 0.75
        {"date": pd.Timestamp("2024-11-04"), "channel": "B", "tvr": 1.0},
        {"date": pd.Timestamp("2024-11-04"), "channel": "B", "tvr": 1.0},  # peak 1, mean 1 -> 1.0
    ]
    retention = daily_retention(pd.DataFrame(rows))
    # Day retention is the channel average of (0.75, 1.0) = 0.875.
    assert retention["2024-11-04"] == pytest.approx(0.875)


def test_daily_retention_skips_days_without_positive_peak() -> None:
    rows = [
        {"date": pd.Timestamp("2024-11-04"), "channel": "A", "tvr": 0.0},
        {"date": pd.Timestamp("2024-11-04"), "channel": "A", "tvr": 0.0},
    ]
    retention = daily_retention(pd.DataFrame(rows))
    # No positive peak means no fabricated number: the day is simply absent.
    assert "2024-11-04" not in retention


# --- meridian-dependent path (skipped unless meridian + xarray present) ------

def _multiday_programmes() -> pd.DataFrame:
    """Programmes across three consecutive days for a valid Meridian time axis."""
    frames = []
    for day in ("2024-11-04", "2024-11-05", "2024-11-06"):
        frame = _programmes().copy()
        shift = pd.Timestamp(day) - pd.Timestamp("2024-11-04")
        frame["start_dt"] = frame["start_dt"] + shift
        frame["end_dt"] = frame["end_dt"] + shift
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


@pytest.mark.realdata
@pytest.mark.skipif(not meridian_available(), reason="meridian/xarray not installed")
def test_build_meridian_input_data_smoke() -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    # Meridian needs a regularly spaced time axis, so span three days.
    spots = pd.DataFrame([
        _spot("A", f"2024-11-0{d} 20:06:00", 30) for d in (4, 5, 6)
    ] + [
        _spot("A", f"2024-11-0{d} 20:06:40", 40) for d in (4, 5, 6)
    ])
    dayparts = pd.DataFrame([
        {"date": pd.Timestamp(f"2024-11-0{d}"), "channel": "A", "tvr": tvr}
        for d in (4, 5, 6) for tvr in (2.0, 4.0)
    ])
    data = build_meridian_input_data(
        programmes=_multiday_programmes(),
        spots=spots,
        dayparts=dayparts,
        classifier=classifier,
    )
    channels = list(data.media.coords["media_channel"].values)
    assert "News_first_short" in channels
    assert len(channels) == 36


@pytest.mark.skipif(meridian_available(), reason="only meaningful when meridian is absent")
def test_build_raises_without_meridian() -> None:
    with pytest.raises(RuntimeError):
        build_meridian_input_data(programmes=_programmes())
