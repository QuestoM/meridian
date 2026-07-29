"""Deterministic Israeli broadcast calendar features.

This module turns a Gregorian date into the calendar facts that shape Israeli
TV audiences: the Sunday-to-Saturday week (weekend is Friday and Saturday
only), shabbat and yom tov (the composite ``religious_blackout``, the days on
which religiously observant viewers do not turn on the TV), chol hamoed,
Hanukkah, school holidays, election days, and the Israeli season.

It is the DETERMINISTIC half of the calendar story: everything here is
computable in advance from the bundled table
``kairos/config/israel_calendar.csv`` plus the weekday. Operator-maintained
events (wars, special events, intensities) live in the separate events store
and its ``event_*`` annotation seam; this module emits a parallel ``cal_*``
feature family and never reads or duplicates operator events.

Everything is date-level. Jewish days begin at sundown the prior evening, so
clock-level nuance (the Friday candle-lighting hour after which observant
households are already offline, and the corresponding motzei-shabbat return)
is deliberately out of scope here and is a v2 refinement at the break-time
level. The bundled table states its own verification caveat: verify against
the official calendar before operational use.

All functions are pure: loading is tolerant (a missing table yields an empty
calendar, weekday and season still work), nothing here writes state, and
``annotate_calendar`` returns a new frame without mutating its input.
"""

from __future__ import annotations

import csv
import datetime as dt
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Bundled table location; the single source for the deterministic calendar.
CALENDAR_PATH = Path(__file__).resolve().parents[1] / "config" / "israel_calendar.csv"

# Known row kinds, most specific first. When several rows cover the same date
# (Tisha BeAv inside the school summer break, an election during Sukkot), the
# reported holiday_kind is the highest-priority covering kind.
KIND_PRIORITY = (
    "yom_tov",
    "chol_hamoed",
    "hanukkah",
    "election",
    "holiday_civil",
    "school_summer",
    "other",
)
_KIND_RANK = {kind: rank for rank, kind in enumerate(KIND_PRIORITY)}

# Feature keys emitted by calendar_features, in emission order. annotate_calendar
# adds each as a cal_ prefixed column.
FEATURE_KEYS = (
    "weekday_iso",
    "is_shabbat",
    "is_erev_shabbat",
    "is_yom_tov",
    "is_erev_yom_tov",
    "is_chol_hamoed",
    "is_hanukkah",
    "is_school_holiday",
    "is_election_day",
    "season",
    "holiday_kind",
    "religious_blackout",
)


@dataclass(frozen=True)
class CalendarRow:
    """One inclusive date range from the bundled calendar table."""

    start_date: dt.date
    end_date: dt.date
    name_he: str
    kind: str
    is_yom_tov: bool
    is_school_holiday: bool
    notes: str

    def covers(self, d: dt.date) -> bool:
        return self.start_date <= d <= self.end_date


def _parse_flag(raw: str) -> bool:
    return str(raw).strip().lower() in ("1", "true", "yes")


def load_calendar(path: Path | str | None = None) -> tuple[CalendarRow, ...]:
    """Load the calendar table tolerantly.

    A missing file yields an empty tuple (weekday and season features still
    work without the table). Comment lines starting with ``#`` and rows that
    fail to parse are skipped with a debug log, never raised, so one bad row
    cannot take down scoring.
    """
    target = Path(path) if path is not None else CALENDAR_PATH
    if not target.exists():
        logger.debug("israel calendar table missing at %s; empty calendar", target)
        return ()

    rows: list[CalendarRow] = []
    with target.open("r", encoding="utf-8", newline="") as handle:
        data_lines = (line for line in handle if not line.lstrip().startswith("#"))
        for record in csv.DictReader(data_lines):
            try:
                start = dt.date.fromisoformat(str(record["start_date"]).strip())
                end = dt.date.fromisoformat(str(record["end_date"]).strip())
            except (KeyError, TypeError, ValueError):
                logger.debug("skipping unparseable calendar row: %r", record)
                continue
            if end < start:
                logger.debug("skipping inverted calendar range: %r", record)
                continue
            rows.append(
                CalendarRow(
                    start_date=start,
                    end_date=end,
                    name_he=str(record.get("name_he") or "").strip(),
                    kind=str(record.get("kind") or "other").strip() or "other",
                    is_yom_tov=_parse_flag(record.get("is_yom_tov", "")),
                    is_school_holiday=_parse_flag(record.get("is_school_holiday", "")),
                    notes=str(record.get("notes") or "").strip(),
                )
            )
    return tuple(rows)


@lru_cache(maxsize=8)
def _load_cached(path_str: str) -> tuple[CalendarRow, ...]:
    """Per-path cache so repeated single-date calls do not re-read the file.

    The table is static and checked in; edits during a live process need
    ``_load_cached.cache_clear()`` (or a restart) to be seen.
    """
    return load_calendar(path_str)


def _resolve_rows(
    rows: tuple[CalendarRow, ...] | None, path: Path | str | None
) -> tuple[CalendarRow, ...]:
    if rows is not None:
        return rows
    target = Path(path) if path is not None else CALENDAR_PATH
    return _load_cached(str(target))


@lru_cache(maxsize=8)
def _yom_tov_dates(rows: tuple[CalendarRow, ...]) -> frozenset[dt.date]:
    """Every individual date flagged is_yom_tov, ranges expanded."""
    dates: set[dt.date] = set()
    for row in rows:
        if row.is_yom_tov:
            day = row.start_date
            while day <= row.end_date:
                dates.add(day)
                day += dt.timedelta(days=1)
    return frozenset(dates)


def season_of(d: dt.date) -> str:
    """The Israeli season of a date: summer, autumn, winter, or spring.

    The bands are Israeli, not the European meteorological quarters: summer is
    June through September (the long dry heat runs well into September, the
    school break sits inside it, and evening routines stay summer-shaped until
    the Tishrei holidays), autumn is October and November (first rains, clocks
    back to winter time in late October, school routine fully resumed), winter
    is December through February (early darkness and the peak of indoor
    evening viewing), and spring is March through May (daylight saving starts
    in late March and daylight stretches the early evening).
    """
    month = d.month
    if 6 <= month <= 9:
        return "summer"
    if month in (10, 11):
        return "autumn"
    if month in (12, 1, 2):
        return "winter"
    return "spring"


def calendar_features(
    d: dt.date,
    *,
    rows: tuple[CalendarRow, ...] | None = None,
    path: Path | str | None = None,
) -> dict[str, object]:
    """All deterministic calendar features for one Gregorian date.

    Returns a dict with exactly the keys in :data:`FEATURE_KEYS`:

    * ``weekday_iso``: ISO weekday, 1 Monday through 7 Sunday. The Israeli
      week runs Sunday to Saturday and the weekend is Friday and Saturday
      only; the two weekend flags below encode that directly.
    * ``is_shabbat``: Saturday. ``is_erev_shabbat``: Friday.
    * ``is_yom_tov``: the date sits on a table row flagged is_yom_tov.
    * ``is_erev_yom_tov``: the NEXT day is yom tov and this day is not itself
      yom tov (so Rosh Hashana day one reads as yom tov, not as erev; a chol
      hamoed Hoshana Rabbah is both chol hamoed and erev yom tov).
    * ``is_chol_hamoed`` / ``is_hanukkah`` / ``is_election_day``: covering row
      of that kind exists. ``is_school_holiday``: any covering row has the
      school-holiday flag.
    * ``season``: :func:`season_of`. ``holiday_kind``: the highest-priority
      covering kind per :data:`KIND_PRIORITY`, or ``''`` when no row covers
      the date. ``religious_blackout``: is_shabbat or is_yom_tov, the owner's
      composite for days when religious viewers do not watch TV.

    ``rows`` short-circuits loading (used by :func:`annotate_calendar`);
    ``path`` points at an alternative table, and a missing table degrades to
    weekday and season features only.
    """
    table = _resolve_rows(rows, path)
    covering = [row for row in table if row.covers(d)]
    yom_tov_days = _yom_tov_dates(table)

    weekday_iso = d.isoweekday()
    is_shabbat = weekday_iso == 6
    is_yom_tov = d in yom_tov_days
    kinds = {row.kind for row in covering}
    holiday_kind = ""
    if covering:
        holiday_kind = min(kinds, key=lambda kind: _KIND_RANK.get(kind, len(_KIND_RANK)))

    return {
        "weekday_iso": weekday_iso,
        "is_shabbat": is_shabbat,
        "is_erev_shabbat": weekday_iso == 5,
        "is_yom_tov": is_yom_tov,
        "is_erev_yom_tov": (d + dt.timedelta(days=1)) in yom_tov_days and not is_yom_tov,
        "is_chol_hamoed": "chol_hamoed" in kinds,
        "is_hanukkah": "hanukkah" in kinds,
        "is_school_holiday": any(row.is_school_holiday for row in covering),
        "is_election_day": "election" in kinds,
        "season": season_of(d),
        "holiday_kind": holiday_kind,
        "religious_blackout": is_shabbat or is_yom_tov,
    }


# Neutral values for rows whose date fails to parse: every flag off, no season
# claimed, weekday_iso 0 (outside the ISO 1..7 range, so it can never be
# mistaken for a real weekday).
_NEUTRAL_FEATURES: dict[str, object] = {
    key: (0 if key == "weekday_iso" else "" if key in ("season", "holiday_kind") else False)
    for key in FEATURE_KEYS
}


def annotate_calendar(
    frame: pd.DataFrame,
    date_column: str,
    *,
    rows: tuple[CalendarRow, ...] | None = None,
    path: Path | str | None = None,
) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``cal_`` prefixed calendar columns.

    Purely additive: the returned frame has the same rows in the same order,
    every existing column byte-identical, plus one ``cal_<key>`` column per
    :data:`FEATURE_KEYS` entry derived from ``date_column``. Unparseable or
    missing dates get the neutral feature values instead of raising. The
    input frame is not mutated. This is the seam the training measurement
    frame and forward scoring call, beside the operator-events ``event_*``
    columns.
    """
    if date_column not in frame.columns:
        raise KeyError(f"date column {date_column!r} not in frame")

    table = _resolve_rows(rows, path)
    result = frame.copy()
    stamps = pd.to_datetime(result[date_column], errors="coerce")

    feature_rows = [
        _NEUTRAL_FEATURES if pd.isna(stamp)
        else calendar_features(stamp.date(), rows=table)
        for stamp in stamps
    ]
    for key in FEATURE_KEYS:
        result[f"cal_{key}"] = [features[key] for features in feature_rows]
    return result
