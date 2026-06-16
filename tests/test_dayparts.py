"""Tests for the canonical Israeli TV daypart taxonomy.

These prove the taxonomy covers the whole clock, that the prime window matches the
engine's prime-time pricing, that the midnight-wrapping night daypart works, and
that a missing or invalid time stays honestly unclassified.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.data.dayparts import (
    PRIME_KEY,
    daypart_for_hour,
    daypart_for_timestamp,
    daypart_keys,
    daypart_options,
    dayparts,
    is_daypart_key,
)


def test_every_clock_hour_maps_to_exactly_one_daypart() -> None:
    # Full coverage with no gaps and no overlaps: each of the 24 hours lands in one
    # and only one daypart, so a spot can never be unclassified for a real clock time.
    for hour in range(24):
        matches = [part.key for part in dayparts() if part.contains_hour(hour)]
        assert len(matches) == 1, f"hour {hour} mapped to {matches}"
        assert daypart_for_hour(hour) == matches[0]


def test_prime_window_matches_the_engine_prime_hours() -> None:
    # Prime time is 20:00-22:59, the same window the prime-time pricing uses.
    for hour in (20, 21, 22):
        assert daypart_for_hour(hour) == PRIME_KEY
    assert daypart_for_hour(19) != PRIME_KEY
    assert daypart_for_hour(23) != PRIME_KEY


def test_night_wraps_midnight() -> None:
    # The night daypart runs 23:00-06:00, so it must claim both late-night and the
    # small hours, the part a non-wrapping range would drop.
    assert daypart_for_hour(23) == "night"
    assert daypart_for_hour(0) == "night"
    assert daypart_for_hour(5) == "night"
    assert daypart_for_hour(6) == "morning"


@pytest.mark.parametrize(
    "hour, expected",
    [(7, "morning"), (13, "noon"), (18, "evening"), (21, "prime"), (2, "night")],
)
def test_named_dayparts(hour: int, expected: str) -> None:
    assert daypart_for_hour(hour) == expected


def test_missing_or_invalid_hour_is_unclassified() -> None:
    assert daypart_for_hour(None) is None
    assert daypart_for_hour(24) is None
    assert daypart_for_hour(-1) is None
    assert daypart_for_hour("not-a-number") is None


def test_daypart_for_timestamp() -> None:
    assert daypart_for_timestamp(pd.Timestamp("2024-11-04 21:30:00")) == "prime"
    assert daypart_for_timestamp(pd.Timestamp("2024-11-04 08:00:00")) == "morning"
    assert daypart_for_timestamp(pd.NaT) is None


def test_keys_and_validation() -> None:
    keys = daypart_keys()
    assert keys == ("morning", "noon", "evening", "prime", "night")
    assert is_daypart_key("prime")
    assert not is_daypart_key("All")
    assert not is_daypart_key("")


def test_options_carry_bilingual_labels_and_windows() -> None:
    options = daypart_options()
    assert {o["key"] for o in options} == set(daypart_keys())
    prime = next(o for o in options if o["key"] == "prime")
    assert prime["he"] == "פריים טיים"
    assert prime["en"] == "Prime time"
    assert prime["start_hour"] == 20 and prime["end_hour"] == 23
