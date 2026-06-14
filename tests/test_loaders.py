"""Tests for the Kairos data loaders against the real reference data.

These run on the committed reference files (real Israeli TV data), so they
assert the actual shapes and the two date conventions the loaders must handle.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.data import loaders
from kairos.data.loaders import (
    DAILY_DIR,
    load_daily_input,
    load_dayparts,
    load_programmes,
    load_spots,
)

DAILY_FILE = DAILY_DIR / "Wally_Prime_Reshet_Example_2025-04-27.csv"


def test_load_programmes_shape_and_dates() -> None:
    frame = load_programmes()
    assert len(frame) == 8704
    assert not [c for c in frame.columns if str(c).startswith("Unnamed")]
    assert pd.api.types.is_datetime64_any_dtype(frame["start_dt"])
    assert pd.api.types.is_datetime64_any_dtype(frame["end_dt"])
    both = frame.dropna(subset=["start_dt", "end_dt"])
    # After the cross-midnight fix, no programme ends before it starts.
    assert (both["end_dt"] >= both["start_dt"]).all()
    # DD/MM parsing: November 2024 data, so months are all 11.
    assert set(frame["start_dt"].dropna().dt.month.unique()) == {11}


def test_load_spots_shape_and_numeric() -> None:
    frame = load_spots()
    assert len(frame) == 50386
    assert pd.api.types.is_datetime64_any_dtype(frame["air_dt"])
    assert pd.api.types.is_numeric_dtype(frame["Pos. Block 1"])
    assert pd.api.types.is_numeric_dtype(frame["TVR"])
    assert frame["TVR"].max() > 0


def test_load_dayparts_melts_to_long() -> None:
    frame = load_dayparts()
    # 43,200 minute rows x 4 channels.
    assert len(frame) == 43200 * 4
    assert set(frame.columns) == {"date", "timeband", "channel", "tvr"}
    assert set(frame["channel"].unique()) == set(loaders.CHANNELS)
    assert pd.api.types.is_datetime64_any_dtype(frame["date"])
    assert pd.api.types.is_numeric_dtype(frame["tvr"])


def test_load_daily_input_canonical_columns() -> None:
    frame = load_daily_input(DAILY_FILE)
    assert len(frame) == 175
    for column in ("date", "advertiser", "campaign", "program", "duration_sec",
                   "position_in_break", "planned_tvr", "price", "status"):
        assert column in frame.columns, column
    # Daily input is a single day in April 2025 (M/D/YYYY parsing).
    dates = frame["date"].dropna()
    assert dates.dt.year.unique().tolist() == [2025]
    assert dates.dt.month.unique().tolist() == [4]
    # Price is an optimizer output, intentionally empty in the input.
    assert frame["price"].isna().all()
    assert pd.api.types.is_numeric_dtype(frame["planned_tvr"])
    assert frame["planned_tvr"].max() > 0


def test_daily_input_missing_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_daily_input(DAILY_DIR / "does_not_exist.csv")
