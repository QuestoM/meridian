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
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
REFERENCE_DIR = DATA_DIR / "reference"
DAILY_DIR = DATA_DIR / "daily_input"

# Uploaded-CSV fallback for each reference workbook. The upload center
# (kairos_api.uploads) writes an operator-provided programmes/spots/dayparts
# file as a flat CSV under data/ (data/Programmes.csv, data/Spots.csv,
# data/Dayparts.csv). The engine reads the reference xlsx, so those uploads were
# shadowed with no way to take effect. These loaders now resolve the xlsx FIRST
# and fall back to the CSV only when the xlsx is absent, so the xlsx always wins
# when both exist (present-day behaviour is unchanged) and an upload becomes the
# live input the moment the reference workbook is removed. The CSV columns are a
# superset of the xlsx (the enriched export adds computed columns); the parse
# reads only the shared columns, so the fallback is value-compatible.
REFERENCE_CSV_FALLBACK = {
    "Programmes.xlsx": DATA_DIR / "Programmes.csv",
    "Spots.xlsx": DATA_DIR / "Spots.csv",
    "Dayparts.xlsx": DATA_DIR / "Dayparts.csv",
}


def _resolve_reference_path(default_xlsx: Path) -> Path:
    """Return the reference workbook to read: the xlsx if present, else the CSV.

    The xlsx wins whenever it exists, so a deployment that ships the reference
    workbooks behaves exactly as before. Only when the xlsx is absent do we adopt
    the uploaded CSV equivalent (where kairos_api.uploads stores it). If neither
    exists we return the xlsx path unchanged, so the real FileNotFoundError still
    surfaces honestly rather than being masked. An explicit caller-supplied path
    is never rerouted here (this is only consulted for the default location).
    """
    if default_xlsx.exists():
        return default_xlsx
    fallback = REFERENCE_CSV_FALLBACK.get(default_xlsx.name)
    if fallback is not None and fallback.exists():
        return fallback
    return default_xlsx


def _read_reference_table(path_str: str) -> pd.DataFrame:
    """Read a reference table as a DataFrame, dispatching on the file suffix.

    An ``.xlsx``/``.xls`` path is read with :func:`pandas.read_excel` exactly as
    before; a ``.csv`` fallback is read with ``utf-8-sig`` so the enriched
    export's BOM is stripped and the header matches the workbook's. Any other
    suffix is treated as Excel, preserving the prior default.
    """
    if path_str.lower().endswith(".csv"):
        return pd.read_csv(path_str, encoding="utf-8-sig")
    return pd.read_excel(path_str)

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
    """Combine a date column and a time-of-day column into one air/start datetime.

    The reference xlsx is inconsistent: some Start/End time cells arrive as plain
    "HH:MM:SS" strings, others as Excel time serials that openpyxl surfaces as a
    full "1900-01-01 HH:MM:SS" datetime. Naive string concatenation
    ("01/11/2024 1900-01-01 00:19:48") fails to parse on the latter and silently
    coerced the row to NaT, dropping it downstream (measured on the real data:
    3,262 of 50,386 spots = 6.5%, plus ~10% of programme start/end times, all of
    them real post-midnight airings). So parse the date and the time-of-day
    independently and add them: the time-of-day is the offset of the parsed time
    from its own midnight, which is identical whether the cell was a bare time
    string or an Excel 1900-epoch datetime. A genuinely unparseable date or time
    still yields NaT (no fabrication), it is only the spurious concatenation loss
    that is recovered.
    """
    day = pd.to_datetime(
        dates.astype(str).str.strip(), errors="coerce", dayfirst=dayfirst
    ).dt.normalize()
    # Parse the time-of-day fast. Without a format pandas falls back to per-element
    # dateutil parsing, which is minutes-slow on 50k rows (the recurring "Could not
    # infer format" warning). The bare "HH:MM:SS" cells parse vectorised against a
    # fixed format; only the Excel-serial "1900-01-01 HH:MM:SS" leftovers fall
    # through to the general parser. The result is value-identical, just faster.
    times_str = times.astype(str).str.strip()
    moment = pd.to_datetime(times_str, format="%H:%M:%S", errors="coerce")
    leftover = moment.isna() & times_str.str.contains(":", na=False)
    if leftover.any():
        moment = moment.where(
            ~leftover, pd.to_datetime(times_str.where(leftover), errors="coerce")
        )
    time_of_day = moment - moment.dt.normalize()
    return day + time_of_day


@lru_cache(maxsize=8)
def _load_programmes_parsed(path_str: str, mtime_ns: int, size: int) -> pd.DataFrame:
    """Parse the programmes xlsx once per (path, mtime, size). The reference parse
    runs the date-combination twice over the whole EPG; the optimizer frontier,
    the channel picker, and the dashboard all load it repeatedly, so cache the
    parse and let callers copy it."""
    del mtime_ns, size
    frame = _drop_index_column(_read_reference_table(path_str))
    frame["start_dt"] = _combine_datetime(frame["Date"], frame["Start time"], dayfirst=True)
    frame["end_dt"] = _combine_datetime(frame["Date"], frame["End time"], dayfirst=True)
    # Programmes that cross midnight have end < start; push end to the next day.
    crosses_midnight = frame["end_dt"].notna() & (frame["end_dt"] < frame["start_dt"])
    frame.loc[crosses_midnight, "end_dt"] += pd.Timedelta(days=1)
    frame["Duration"] = pd.to_numeric(frame.get("Duration"), errors="coerce")
    frame["TVR"] = pd.to_numeric(frame.get("TVR"), errors="coerce")
    return frame.reset_index(drop=True)


def load_programmes(path: str | Path | None = None) -> pd.DataFrame:
    """Load the programme (EPG) log with parsed start_dt and end_dt.

    The parse is memoized on the source file signature (path, mtime, size) and a
    fresh copy is returned, so repeated loads within one process are cheap while
    callers stay free to mutate their copy. A missing file falls through to the
    uncached parse so the real FileNotFoundError surfaces honestly.
    """
    path = Path(path) if path else _resolve_reference_path(REFERENCE_DIR / "Programmes.xlsx")
    try:
        stat = path.stat()
    except OSError:
        return _load_programmes_parsed.__wrapped__(str(path), 0, 0)
    return _load_programmes_parsed(str(path), stat.st_mtime_ns, stat.st_size).copy()


@lru_cache(maxsize=8)
def _load_spots_parsed(path_str: str, mtime_ns: int, size: int) -> pd.DataFrame:
    """Parse the spots xlsx once per (path, mtime, size). On the real 50k-row file
    the date-combination is tens of seconds; the dashboard inventory, campaigns,
    and overview builders each load it, so cache the parse and let callers copy."""
    del mtime_ns, size
    frame = _drop_index_column(_read_reference_table(path_str))
    frame["air_dt"] = _combine_datetime(frame["Date"], frame["Start time"], dayfirst=True)
    for column in ("Duration", "Pos. Block 1", "Spots Block 1", "TVR"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def load_spots(path: str | Path | None = None) -> pd.DataFrame:
    """Load the aired-spots log with parsed air datetime and numeric fields.

    Memoized on the source file signature (path, mtime, size); a fresh copy is
    returned so callers may mutate freely. A missing file falls through to the
    uncached parse so the real FileNotFoundError surfaces honestly.
    """
    path = Path(path) if path else _resolve_reference_path(REFERENCE_DIR / "Spots.xlsx")
    try:
        stat = path.stat()
    except OSError:
        return _load_spots_parsed.__wrapped__(str(path), 0, 0)
    return _load_spots_parsed(str(path), stat.st_mtime_ns, stat.st_size).copy()


def load_dayparts(path: str | Path | None = None) -> pd.DataFrame:
    """Load the minute-level TVR matrix melted to long form.

    Returns columns: date, timeband, channel, tvr (one row per channel-minute).
    """
    path = Path(path) if path else _resolve_reference_path(REFERENCE_DIR / "Dayparts.xlsx")
    frame = _drop_index_column(_read_reference_table(str(path)))
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
