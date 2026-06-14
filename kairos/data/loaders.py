"""Loaders for Kairos reference data and the daily optimization input.

Each loader reads one real source file and returns a clean, typed DataFrame
with canonical columns. The two date conventions in the data are handled
explicitly here so nothing downstream has to guess:

  - reference xlsx (Spots/Programmes/Dayparts): dates are DD/MM/YYYY
  - daily input csv (Wally ...): dates are M/D/YYYY

Dayparts is stored wide (one TVR column per channel); load_dayparts melts it
to long form so it can be joined by (channel, date, minute).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = ROOT / "data" / "reference"
DAILY_DIR = ROOT / "data" / "daily_input"

# The four channels present in the reference data.
CHANNELS = ("קשת 12", "רשת 13", "כאן 11", "עכשיו 14")

# Hebrew -> canonical English column names for the daily optimization input.
DAILY_COLUMN_MAP = {
    "תאריך": "date",
    "שעה": "spot_time",
    "שעת התחלת ברייק": "break_start",
    "משרד / MB": "agency",
    "סוג תשדיר": "spot_type",
    "מפרסם": "advertiser",
    "קמפיין": "campaign",
    "שם גרסה": "creative",
    "House Number": "house_number",
    "אורך תשדיר": "duration_sec",
    "תוכנית מוזמנת": "program",
    "שעת התחלת תוכנית": "program_start",
    "סוג ברייק": "break_type",
    "סוג תמחור": "pricing_type",
    "מחיר": "price",
    "רייטינג ברייקים מתוכנן": "planned_tvr",
    "מיקום בברייק": "position_in_break",
    "סטטוס": "status",
}


def _drop_index_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop the saved pandas row index that xlsx exports keep as 'Unnamed: 0'."""
    return frame.drop(columns=[c for c in frame.columns if str(c).startswith("Unnamed")])


def _combine_datetime(dates: pd.Series, times: pd.Series, *, dayfirst: bool) -> pd.Series:
    return pd.to_datetime(
        dates.astype(str).str.strip() + " " + times.astype(str).str.strip(),
        errors="coerce",
        dayfirst=dayfirst,
    )


def load_programmes(path: str | Path | None = None) -> pd.DataFrame:
    """Load the programme (EPG) log with parsed start_dt and end_dt."""
    path = Path(path) if path else REFERENCE_DIR / "Programmes.xlsx"
    frame = _drop_index_column(pd.read_excel(path))
    frame["start_dt"] = _combine_datetime(frame["Date"], frame["Start time"], dayfirst=True)
    frame["end_dt"] = _combine_datetime(frame["Date"], frame["End time"], dayfirst=True)
    # Programmes that cross midnight have end < start; push end to the next day.
    crosses_midnight = frame["end_dt"].notna() & (frame["end_dt"] < frame["start_dt"])
    frame.loc[crosses_midnight, "end_dt"] += pd.Timedelta(days=1)
    frame["Duration"] = pd.to_numeric(frame.get("Duration"), errors="coerce")
    frame["TVR"] = pd.to_numeric(frame.get("TVR"), errors="coerce")
    return frame.reset_index(drop=True)


def load_spots(path: str | Path | None = None) -> pd.DataFrame:
    """Load the aired-spots log with parsed air datetime and numeric fields."""
    path = Path(path) if path else REFERENCE_DIR / "Spots.xlsx"
    frame = _drop_index_column(pd.read_excel(path))
    frame["air_dt"] = _combine_datetime(frame["Date"], frame["Start time"], dayfirst=True)
    for column in ("Duration", "Pos. Block 1", "Spots Block 1", "TVR"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def load_dayparts(path: str | Path | None = None) -> pd.DataFrame:
    """Load the minute-level TVR matrix melted to long form.

    Returns columns: date, timeband, channel, tvr (one row per channel-minute).
    """
    path = Path(path) if path else REFERENCE_DIR / "Dayparts.xlsx"
    frame = _drop_index_column(pd.read_excel(path))
    channel_columns = [c for c in CHANNELS if c in frame.columns]
    long = frame.melt(
        id_vars=["Dates", "Timebands"],
        value_vars=channel_columns,
        var_name="channel",
        value_name="tvr",
    )
    long = long.rename(columns={"Dates": "date", "Timebands": "timeband"})
    long["date"] = pd.to_datetime(long["date"], errors="coerce", dayfirst=True)
    long["tvr"] = pd.to_numeric(long["tvr"], errors="coerce")
    return long.reset_index(drop=True)


def load_daily_input(path: str | Path) -> pd.DataFrame:
    """Load a daily optimization input csv, renaming Hebrew columns to canonical.

    The price and status columns are intentionally empty in the input (they are
    optimizer outputs); they are preserved as nullable columns.
    """
    frame = pd.read_csv(Path(path), encoding="utf-8")
    frame = frame.rename(columns={k: v for k, v in DAILY_COLUMN_MAP.items() if k in frame.columns})
    if "date" in frame.columns:
        # Daily input dates are M/D/YYYY (US-style), unlike the reference xlsx.
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce", dayfirst=False)
    for column in ("duration_sec", "position_in_break", "planned_tvr", "price"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)
