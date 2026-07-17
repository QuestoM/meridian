"""Regression tests for the segment transform's edge-value handling.

Two proven data edges: a NaN Duration is truthy (``nan or 0.0`` stays nan) and
``nan <= 0`` is False, so before the guard it slipped past the non-positive
skip and emitted a NaN-length segment; and the next-programme presence test in
the daily path must be ``is not None``, not truthiness, so a programme starting
exactly at midnight (0.0 seconds) counts as present.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from kairos.data import ProgramClassifier
from kairos.data.transform import (
    _DEFAULT_PROGRAMME_SECONDS,
    build_segments_from_daily_input,
    build_segments_from_programmes,
)
from kairos.optimize.pricing import PricingModel


@pytest.fixture(scope="module")
def pricing() -> PricingModel:
    return PricingModel.from_yaml()


@pytest.fixture(scope="module")
def classifier() -> ProgramClassifier:
    return ProgramClassifier.from_yaml()


def _programmes_frame() -> pd.DataFrame:
    rows = [
        ("חדשות הערב", "קשת 12", "20:00:00", 3600.0, 5.0),
        ("תוכנית אמצע", "קשת 12", "21:00:00", float("nan"), 5.0),
        ("תוכנית סיום", "קשת 12", "22:00:00", 3600.0, 4.0),
    ]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "time", "Duration", "TVR"])
    frame["start_dt"] = pd.to_datetime("2024-11-04 " + frame["time"])
    return frame


def test_nan_duration_row_is_skipped_not_emitted(pricing, classifier) -> None:
    """A NaN Duration must be skipped like a non-positive one, never emitted.

    Before the guard, ``float(nan or 0.0)`` stayed NaN and ``nan <= 0`` was
    False, so the row produced a segment with duration_seconds = NaN that
    poisoned every downstream sum.
    """
    segments = build_segments_from_programmes(
        _programmes_frame(), classifier, pricing, channel="קשת 12",
    )
    assert len(segments) == 2, "the NaN-duration programme must be dropped"
    assert all(math.isfinite(s.duration_seconds) for s in segments)
    assert all(s.duration_seconds > 0 for s in segments)


def test_zero_duration_still_skipped(pricing, classifier) -> None:
    frame = _programmes_frame()
    frame.loc[1, "Duration"] = 0.0
    segments = build_segments_from_programmes(frame, classifier, pricing, channel="קשת 12")
    assert len(segments) == 2


def _daily_frame(program_starts: list[tuple[str, str]]) -> pd.DataFrame:
    rows = []
    for program, start in program_starts:
        rows.append(
            {
                "date": "2025-04-27",
                "program": program,
                "program_start": start,
                "planned_tvr": 5.0,
            }
        )
    frame = pd.DataFrame(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def test_daily_midnight_programme_bounds_are_finite(pricing, classifier) -> None:
    """The daily path: a programme starting at midnight (0.0 seconds) is a real
    presence, and programme lengths run to the next programme's start with the
    documented fallback for the last."""
    frame = _daily_frame([("תוכנית לילה", "00:00"), ("תוכנית בוקר", "01:00")])
    segments = build_segments_from_daily_input(frame, classifier, pricing)
    assert len(segments) == 2
    assert segments[0].start_seconds == 0.0
    assert segments[0].duration_seconds == 3600.0, "length must run to the next start"
    assert segments[1].duration_seconds == _DEFAULT_PROGRAMME_SECONDS


def test_daily_equal_starts_fall_back_to_default_length(pricing, classifier) -> None:
    """Two programmes at the same start (including both at midnight) cannot
    bound each other, so both keep the documented fallback length, and the
    presence test never mistakes the 0.0 start for a missing next programme."""
    frame = _daily_frame([("תוכנית א", "00:00"), ("תוכנית ב", "00:00")])
    segments = build_segments_from_daily_input(frame, classifier, pricing)
    assert len(segments) == 2
    assert all(s.duration_seconds == _DEFAULT_PROGRAMME_SECONDS for s in segments)
    assert all(math.isfinite(s.duration_seconds) for s in segments)
