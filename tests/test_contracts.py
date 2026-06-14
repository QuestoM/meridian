"""Tests for the Kairos data-contract validators.

The fast tests use small synthetic DataFrames that mirror the SHAPE the
loaders produce, so they run with no file I/O. One opt-in real-data smoke test
loads the committed reference xlsx through the existing loaders and asserts the
report is valid; it is marked ``realdata`` so the fast gate can exclude it with
``-m "not realdata"``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.data.contracts import (
    ValidationReport,
    Violation,
    validate_daily_input,
    validate_dayparts,
    validate_programmes,
    validate_spots,
)

KNOWN_CHANNEL = "קשת 12"


# ---------------------------------------------------------------------------
# Synthetic frame builders (valid baselines)
# ---------------------------------------------------------------------------


def _valid_programmes() -> pd.DataFrame:
    start = pd.to_datetime(["2024-11-01 20:00", "2024-11-01 21:00"])
    end = pd.to_datetime(["2024-11-01 20:30", "2024-11-01 21:45"])
    return pd.DataFrame(
        {
            "Title": ["News", "Prime"],
            "Channel": [KNOWN_CHANNEL, KNOWN_CHANNEL],
            "Duration": [1800, 2700],
            "TVR": [5.4, 8.1],
            "start_dt": start,
            "end_dt": end,
        }
    )


def _valid_spots() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Campaign": ["A", "B"],
            "Channel": [KNOWN_CHANNEL, KNOWN_CHANNEL],
            "Duration": [30, 20],
            "TVR": [3.1, 0.0],
            "air_dt": pd.to_datetime(["2024-11-01 20:10", "2024-11-01 20:11"]),
        }
    )


def _valid_dayparts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-11-01", "2024-11-01"]),
            "timeband": ["20:00", "20:01"],
            "channel": [KNOWN_CHANNEL, KNOWN_CHANNEL],
            "tvr": [4.2, 4.3],
        }
    )


def _valid_daily_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-04-27", "2025-04-27"]),
            "advertiser": ["Acme", "Beta"],
            "campaign": ["Spring", "Launch"],
            "program": ["Prime", "Prime"],
            "duration_sec": [30, 45],
            "position_in_break": [1, 2],
            "planned_tvr": [6.0, 7.5],
            "price": [np.nan, np.nan],
            "status": [np.nan, np.nan],
        }
    )


# ---------------------------------------------------------------------------
# Violation / report basics
# ---------------------------------------------------------------------------


def test_violation_rejects_bad_severity() -> None:
    with pytest.raises(ValueError):
        Violation("TVR", "x", "y", "fatal")


def test_report_str_and_validity() -> None:
    report = ValidationReport("demo")
    assert report.is_valid
    assert "VALID" in str(report)
    report.add("TVR", "negative_values", "1 below zero", "error")
    report.add("TVR", "nan_values", "2 NaN", "warning")
    assert not report.is_valid
    assert len(report.errors) == 1
    assert len(report.warnings) == 1
    text = str(report)
    assert "INVALID" in text
    assert "negative_values" in text


# ---------------------------------------------------------------------------
# Valid frames pass
# ---------------------------------------------------------------------------


def test_valid_programmes_passes() -> None:
    report = validate_programmes(_valid_programmes())
    assert report.is_valid, str(report)
    assert not report.warnings


def test_valid_spots_passes() -> None:
    report = validate_spots(_valid_spots())
    assert report.is_valid, str(report)


def test_valid_dayparts_passes() -> None:
    report = validate_dayparts(_valid_dayparts())
    assert report.is_valid, str(report)


def test_valid_daily_input_passes() -> None:
    report = validate_daily_input(_valid_daily_input())
    assert report.is_valid, str(report)


# ---------------------------------------------------------------------------
# Missing column -> error
# ---------------------------------------------------------------------------


def test_missing_column_is_error() -> None:
    frame = _valid_programmes().drop(columns=["TVR"])
    report = validate_programmes(frame)
    assert not report.is_valid
    codes = {(v.field, v.code) for v in report.errors}
    assert ("TVR", "missing_column") in codes


# ---------------------------------------------------------------------------
# Negative TVR -> error
# ---------------------------------------------------------------------------


def test_negative_tvr_is_error() -> None:
    frame = _valid_programmes()
    frame.loc[0, "TVR"] = -1.0
    report = validate_programmes(frame)
    assert not report.is_valid
    codes = {(v.field, v.code) for v in report.errors}
    assert ("TVR", "negative_values") in codes


def test_negative_tvr_spots_is_error() -> None:
    frame = _valid_spots()
    frame.loc[1, "TVR"] = -0.5
    report = validate_spots(frame)
    assert ("TVR", "negative_values") in {
        (v.field, v.code) for v in report.errors
    }


# ---------------------------------------------------------------------------
# Non-positive duration -> error
# ---------------------------------------------------------------------------


def test_zero_duration_is_error() -> None:
    frame = _valid_programmes()
    frame.loc[0, "Duration"] = 0
    report = validate_programmes(frame)
    assert ("Duration", "non_positive_values") in {
        (v.field, v.code) for v in report.errors
    }


# ---------------------------------------------------------------------------
# NaN TVR -> warning, never silently invented, stays valid
# ---------------------------------------------------------------------------


def test_nan_tvr_is_warning_not_error() -> None:
    frame = _valid_programmes()
    frame.loc[0, "TVR"] = np.nan
    report = validate_programmes(frame)
    assert report.is_valid, str(report)
    warn_codes = {(v.field, v.code) for v in report.warnings}
    assert ("TVR", "nan_values") in warn_codes
    # The validator must not fabricate: the frame still holds a NaN.
    assert frame["TVR"].isna().sum() == 1


def test_nan_daypart_tvr_is_warning() -> None:
    frame = _valid_dayparts()
    frame.loc[0, "tvr"] = np.nan
    report = validate_dayparts(frame)
    assert report.is_valid, str(report)
    assert ("tvr", "nan_values") in {
        (v.field, v.code) for v in report.warnings
    }


# ---------------------------------------------------------------------------
# Non-numeric coercion failure -> error
# ---------------------------------------------------------------------------


def test_non_numeric_tvr_is_error() -> None:
    frame = _valid_programmes()
    frame["TVR"] = frame["TVR"].astype(object)
    frame.loc[0, "TVR"] = "not-a-number"
    report = validate_programmes(frame)
    assert ("TVR", "non_numeric_values") in {
        (v.field, v.code) for v in report.errors
    }


# ---------------------------------------------------------------------------
# Unknown / NaN channel -> warning
# ---------------------------------------------------------------------------


def test_unknown_channel_is_warning() -> None:
    frame = _valid_programmes()
    frame.loc[0, "Channel"] = "BBC One"
    report = validate_programmes(frame)
    # Unknown channel does not invalidate the frame.
    assert report.is_valid, str(report)
    assert ("Channel", "unknown_channel") in {
        (v.field, v.code) for v in report.warnings
    }


# ---------------------------------------------------------------------------
# end before start -> error
# ---------------------------------------------------------------------------


def test_end_before_start_is_error() -> None:
    frame = _valid_programmes()
    frame.loc[0, "end_dt"] = frame.loc[0, "start_dt"] - pd.Timedelta(hours=1)
    report = validate_programmes(frame)
    assert ("end_dt", "end_before_start") in {
        (v.field, v.code) for v in report.errors
    }


# ---------------------------------------------------------------------------
# None frame -> error, not a crash
# ---------------------------------------------------------------------------


def test_none_frame_is_error_not_crash() -> None:
    report = validate_programmes(None)
    assert not report.is_valid
    assert ("<frame>", "missing_frame") in {
        (v.field, v.code) for v in report.errors
    }


# ---------------------------------------------------------------------------
# Opt-in real-data smoke (excluded from the fast gate with -m "not realdata")
# ---------------------------------------------------------------------------


@pytest.mark.realdata
def test_real_reference_data_validates() -> None:
    from kairos.data.loaders import (
        DAILY_DIR,
        load_dayparts,
        load_daily_input,
        load_programmes,
        load_spots,
    )

    daily_file = DAILY_DIR / "Wally_Prime_Reshet_Example_2025-04-27.csv"
    if not daily_file.exists():
        pytest.skip("real reference data not present")

    assert validate_programmes(load_programmes()).is_valid
    assert validate_spots(load_spots()).is_valid
    assert validate_dayparts(load_dayparts()).is_valid
    assert validate_daily_input(load_daily_input(daily_file)).is_valid
