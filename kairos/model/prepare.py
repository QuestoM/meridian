"""Turn the real TV reference data into a Meridian ``InputData`` object.

This is the data-preparation seam between the Kairos loaders and the Meridian
impact model. It detects commercial breaks in the aired-spots log, matches each
break to its containing programme, derives the three break attributes the
optimizer reasons about (positional pricing class, position, length bucket), and
aggregates break seconds and viewer retention to a daily, single-geo tensor.

The channel vocabulary is the engine vocabulary, not genres: a media channel is
``f"{program_type}_{break_position}_{break_length}"`` exactly as
:class:`kairos.model.spec.ChannelDescriptor.from_parts` builds it, so the trained
posterior's ``media_channel`` coordinates round-trip into the optimizer's lookups
(see :func:`kairos.model.impact._extract_coefficients`). The pricing class is
computed with the same :func:`kairos.data.transform._pricing_classes` the
optimizer uses, so training ``program_type`` matches the optimizer's assignment.

Honesty rules mirror :mod:`kairos.data.transform`: retention is the real ratio
from the daypart data and a day with no measurable retention keeps the neutral
baseline; breaks in unmatched programmes are ``"Other"``. Meridian and xarray are
imported behind a guard, so the pure-pandas helpers import and unit-test cleanly
without them, and :func:`build_meridian_input_data` raises a clear RuntimeError
when Meridian is absent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.data.transform import (
    DEFAULT_NEWS_KEYWORDS,
    DEFAULT_NUM_MAIN_SHOWS,
    _pricing_classes,
)
from kairos.model.spec import ChannelDescriptor, ChannelSpec, build_channel_spec

logger = logging.getLogger(__name__)

# Guarded optional imports: the data layer must import on a desktop Python that
# carries neither Meridian nor xarray. They stay None when absent, and only
# build_meridian_input_data touches them (after raising a clear error if missing).
try:  # pragma: no cover - exercised only where Meridian is installed
    from meridian.data import input_data as _meridian_input_data
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _meridian_input_data = None

try:  # pragma: no cover - exercised only where xarray is installed
    import xarray as _xr
except Exception:  # noqa: BLE001
    _xr = None

# Break detection: spots <= this gap apart belong to the same break, and a break
# needs at least this many spots. Matches tv_break_data_transformer's defaults.
_MAX_SPOT_GAP_SECONDS = 15.0
_MIN_SPOTS_PER_BREAK = 2

# Position thresholds on the break's relative offset into its programme.
_FIRST_MAX_RATIO = 0.33
_MIDDLE_MAX_RATIO = 0.66

# Length-bucket thresholds in seconds, matching kairos.data.transform's buckets
# (which match kairos.model.spec.DEFAULT_BREAK_LENGTHS).
_SHORT_MAX_SECONDS = 90.0
_STANDARD_MAX_SECONDS = 180.0

_OTHER = "Other"
_GEO = "Israel"
_POPULATION = 1_000_000.0

_DAY_ORDER = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)


def meridian_available() -> bool:
    """Return True when both Meridian and xarray imported successfully."""
    return _meridian_input_data is not None and _xr is not None


def position_bucket(relative_position: float) -> str:
    """Map a break's relative offset into its programme to first/middle/last.

    ``relative_position`` is seconds-into-programme divided by programme duration.
    The engine vocabulary buckets are first (<=0.33), middle (<=0.66), last.
    """
    if relative_position <= _FIRST_MAX_RATIO:
        return "first"
    if relative_position <= _MIDDLE_MAX_RATIO:
        return "middle"
    return "last"


def length_bucket(break_seconds: float) -> str:
    """Map a break length in seconds to the short/standard/long bucket."""
    if break_seconds < _SHORT_MAX_SECONDS:
        return "short"
    if break_seconds < _STANDARD_MAX_SECONDS:
        return "standard"
    return "long"


def identify_breaks(spots: pd.DataFrame) -> pd.DataFrame:
    """Group adjacent spots per channel into commercial breaks.

    ``spots`` must carry the columns produced by
    :func:`kairos.data.loaders.load_spots` (``Channel``, ``air_dt``,
    ``Duration`` in seconds). Spots within :data:`_MAX_SPOT_GAP_SECONDS` of the
    previous spot's end join the same break; a run of fewer than
    :data:`_MIN_SPOTS_PER_BREAK` spots is dropped (it is not a break).

    Returns one row per break with columns: channel, break_start, break_end,
    break_seconds, num_spots. Returns an empty frame when there are no breaks.
    """
    columns = ["channel", "break_start", "break_end", "break_seconds", "num_spots"]
    if spots.empty:
        return pd.DataFrame(columns=columns)

    frame = spots[spots["air_dt"].notna()].copy()
    frame["Duration"] = pd.to_numeric(frame.get("Duration"), errors="coerce").fillna(0.0)
    frame["end_dt"] = frame["air_dt"] + pd.to_timedelta(frame["Duration"], unit="s")
    frame = frame.sort_values(["Channel", "air_dt"]).reset_index(drop=True)

    breaks: list[dict[str, Any]] = []
    for channel, channel_spots in frame.groupby("Channel", sort=False):
        current: list[pd.Series] = []
        for _, spot in channel_spots.iterrows():
            if not current:
                current = [spot]
                continue
            gap = (spot["air_dt"] - current[-1]["end_dt"]).total_seconds()
            if gap <= _MAX_SPOT_GAP_SECONDS:
                current.append(spot)
            else:
                _append_break(breaks, channel, current)
                current = [spot]
        _append_break(breaks, channel, current)

    return pd.DataFrame(breaks, columns=columns)


def _append_break(breaks: list[dict[str, Any]], channel: str, spots: list[pd.Series]) -> None:
    """Append one break to ``breaks`` if the run is long enough to count."""
    if len(spots) < _MIN_SPOTS_PER_BREAK:
        return
    break_start = spots[0]["air_dt"]
    break_end = spots[-1]["end_dt"]
    breaks.append(
        {
            "channel": channel,
            "break_start": break_start,
            "break_end": break_end,
            "break_seconds": float((break_end - break_start).total_seconds()),
            "num_spots": len(spots),
        }
    )


def pricing_class_lookup(
    programmes: pd.DataFrame,
    classifier: ProgramClassifier,
    *,
    news_keywords: tuple[str, ...] = tuple(DEFAULT_NEWS_KEYWORDS),
    num_main_shows: int = DEFAULT_NUM_MAIN_SHOWS,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    """Build a (channel, day) -> ordered programme records lookup.

    Each programme record carries start_dt, end_dt, duration and the positional
    pricing class assigned by :func:`kairos.data.transform._pricing_classes` over
    that channel-day in air order. This is the exact logic the optimizer uses, so
    the training program_type matches the optimizer's assignment.
    """
    frame = programmes[programmes["start_dt"].notna()].copy()
    frame["day"] = frame["start_dt"].dt.strftime("%Y-%m-%d")
    lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}

    for (channel, day), group in frame.groupby(["Channel", "day"], sort=False):
        ordered = group.sort_values("start_dt").reset_index(drop=True)
        titles_categories = [
            (str(title), classifier.classify(title).category) for title in ordered["Title"]
        ]
        classes = _pricing_classes(
            titles_categories, news_keywords=news_keywords, num_main_shows=num_main_shows,
        )
        records: list[dict[str, Any]] = []
        for row, pricing_class in zip(ordered.itertuples(index=False), classes):
            records.append(
                {
                    "start_dt": getattr(row, "start_dt"),
                    "end_dt": getattr(row, "end_dt"),
                    "duration": float(getattr(row, "Duration", 0.0) or 0.0),
                    "pricing_class": pricing_class,
                }
            )
        lookup[(str(channel), day)] = records
    return lookup


def match_break_to_programme(
    channel: str,
    break_start: pd.Timestamp,
    break_end: pd.Timestamp,
    lookup: dict[tuple[str, str], list[dict[str, Any]]],
) -> tuple[str, str]:
    """Return (pricing_class, position_bucket) for a break.

    Finds the programme on the same channel-day whose span contains the break and
    reads its positional pricing class and the break's position bucket. A break in
    no matched programme honestly becomes (``"Other"``, ``"middle"``); middle is
    the neutral position when the offset is unknown.
    """
    day = pd.Timestamp(break_start).strftime("%Y-%m-%d")
    records = lookup.get((channel, day), [])
    for record in records:
        start = record["start_dt"]
        end = record["end_dt"]
        if pd.isna(start) or pd.isna(end):
            continue
        if start <= break_start and end >= break_end:
            duration = record["duration"]
            if duration > 0:
                offset = (break_start - start).total_seconds()
                return record["pricing_class"], position_bucket(offset / duration)
            return record["pricing_class"], "middle"
    return _OTHER, "middle"


def keyed_breaks(
    spots: pd.DataFrame,
    programmes: pd.DataFrame,
    classifier: ProgramClassifier,
    *,
    news_keywords: tuple[str, ...] = tuple(DEFAULT_NEWS_KEYWORDS),
    num_main_shows: int = DEFAULT_NUM_MAIN_SHOWS,
) -> pd.DataFrame:
    """Detect breaks and tag each with its engine-vocabulary channel and day.

    Returns the breaks frame extended with program_type, break_position,
    break_length, channel_name (the engine channel id) and day (YYYY-MM-DD).
    Pure pandas: no Meridian needed.
    """
    breaks = identify_breaks(spots)
    columns = [
        "channel", "break_start", "break_end", "break_seconds", "num_spots",
        "program_type", "break_position", "break_length", "channel_name", "day",
    ]
    if breaks.empty:
        return pd.DataFrame(columns=columns)

    lookup = pricing_class_lookup(
        programmes, classifier, news_keywords=news_keywords, num_main_shows=num_main_shows,
    )

    program_types: list[str] = []
    positions: list[str] = []
    lengths: list[str] = []
    names: list[str] = []
    days: list[str] = []
    for row in breaks.itertuples(index=False):
        pricing_class, position = match_break_to_programme(
            str(getattr(row, "channel")),
            getattr(row, "break_start"),
            getattr(row, "break_end"),
            lookup,
        )
        length = length_bucket(float(getattr(row, "break_seconds")))
        descriptor = ChannelDescriptor.from_parts(pricing_class, position, length)
        program_types.append(pricing_class)
        positions.append(position)
        lengths.append(length)
        names.append(descriptor.name)
        days.append(pd.Timestamp(getattr(row, "break_start")).strftime("%Y-%m-%d"))

    breaks = breaks.copy()
    breaks["program_type"] = program_types
    breaks["break_position"] = positions
    breaks["break_length"] = lengths
    breaks["channel_name"] = names
    breaks["day"] = days
    return breaks


def daily_retention(dayparts: pd.DataFrame) -> dict[str, float]:
    """Compute mean viewer retention per day from the daypart TVR series.

    Retention is each channel-day's mean minute TVR over its peak minute TVR (a
    hold ratio), averaged across channels for the day. A day with no positive
    peak contributes nothing (no fabricated rating), so it is absent from the
    returned day (YYYY-MM-DD) -> retention map and keeps the neutral baseline.
    """
    frame = dayparts[dayparts["date"].notna()].copy()
    frame["tvr"] = pd.to_numeric(frame["tvr"], errors="coerce")
    frame["day"] = frame["date"].dt.strftime("%Y-%m-%d")

    retention: dict[str, float] = {}
    for day, day_group in frame.groupby("day", sort=True):
        ratios: list[float] = []
        for _channel, channel_group in day_group.groupby("channel", sort=False):
            values = channel_group["tvr"].dropna()
            peak = values.max()
            if pd.notna(peak) and peak > 0:
                ratios.append(float(values.mean() / peak))
        if ratios:
            retention[day] = float(np.mean(ratios))
    return retention


def _date_range(*frames: pd.Series) -> list[str]:
    """Collect the sorted YYYY-MM-DD strings spanning every supplied date series."""
    stamps: list[pd.Timestamp] = []
    for series in frames:
        valid = pd.to_datetime(series, errors="coerce").dropna()
        stamps.extend(valid.dt.normalize().tolist())
    clean = sorted({s for s in stamps if pd.notna(s) and 2010 <= s.year <= 2100})
    if not clean:
        return []
    full = pd.date_range(start=clean[0], end=clean[-1], freq="D")
    return [d.strftime("%Y-%m-%d") for d in full]


def _control_frame(time_values: list[str]) -> tuple[np.ndarray, list[str]]:
    """Build day-of-week one-hots plus is_weekend controls for the time axis."""
    control_names = [f"day_of_week_{day}" for day in _DAY_ORDER] + ["is_weekend"]
    controls = np.zeros((1, len(time_values), len(control_names)), dtype=float)
    for time_idx, day in enumerate(time_values):
        stamp = pd.Timestamp(day)
        day_name = stamp.day_name()
        if day_name in _DAY_ORDER:
            controls[0, time_idx, _DAY_ORDER.index(day_name)] = 1.0
        controls[0, time_idx, len(_DAY_ORDER)] = 1.0 if stamp.weekday() >= 5 else 0.0
    return controls, control_names


def build_meridian_input_data(
    *,
    programmes: Optional[pd.DataFrame] = None,
    spots: Optional[pd.DataFrame] = None,
    dayparts: Optional[pd.DataFrame] = None,
    programmes_path: str | Path | None = None,
    spots_path: str | Path | None = None,
    dayparts_path: str | Path | None = None,
    channel_spec: Optional[ChannelSpec] = None,
    classifier: Optional[ProgramClassifier] = None,
) -> Any:
    """Build a Meridian ``InputData`` from the real TV reference data.

    Frames may be passed directly (the pure-pandas path for tests) or loaded from
    ``data/reference`` via the Kairos loaders when omitted. The media tensor sums
    break seconds per engine channel per day, the KPI is mean viewer retention per
    day, controls are day-of-week one-hots plus is_weekend, geo is ["Israel"], and
    the time axis is the daily date range across the data. Raises a clear
    RuntimeError when Meridian or xarray is absent; no rating is fabricated.
    """
    if not meridian_available():
        raise RuntimeError(
            "build_meridian_input_data requires the meridian and xarray libraries, "
            "which are not installed. Install google-meridian (with tensorflow and "
            "xarray) on a supported Python before preparing the training data."
        )

    programmes = load_programmes(programmes_path) if programmes is None else programmes
    spots = load_spots(spots_path) if spots is None else spots
    dayparts = load_dayparts(dayparts_path) if dayparts is None else dayparts
    classifier = classifier or ProgramClassifier.from_yaml()
    spec = channel_spec or build_channel_spec()

    breaks = keyed_breaks(spots, programmes, classifier)
    retention = daily_retention(dayparts)

    time_values = _date_range(
        programmes["start_dt"],
        spots["air_dt"],
        dayparts["date"],
    )
    if not time_values:
        raise RuntimeError("the reference data carried no usable dates; cannot build a time axis")
    time_index = {day: idx for idx, day in enumerate(time_values)}

    channel_names = list(spec.channel_names)
    channel_index = {name: idx for idx, name in enumerate(channel_names)}

    kpi = np.ones((1, len(time_values)), dtype=float)
    media = np.zeros((1, len(time_values), len(channel_names)), dtype=float)
    media_spend = np.zeros(len(channel_names), dtype=float)

    for day, value in retention.items():
        time_idx = time_index.get(day)
        if time_idx is not None:
            kpi[0, time_idx] = float(np.clip(value, 0.01, 1.5))

    for row in breaks.itertuples(index=False):
        time_idx = time_index.get(getattr(row, "day"))
        media_idx = channel_index.get(getattr(row, "channel_name"))
        if time_idx is None or media_idx is None:
            continue
        seconds = float(getattr(row, "break_seconds"))
        media[0, time_idx, media_idx] += seconds
        media_spend[media_idx] += seconds

    # Meridian needs strictly positive spend per channel; replace zeros with 1.0.
    media_spend = np.where(media_spend > 0, media_spend, 1.0)

    controls, control_names = _control_frame(time_values)

    logger.info(
        "Built Meridian InputData: %d days, %d channels, %d breaks keyed, %d days with retention.",
        len(time_values),
        len(channel_names),
        len(breaks),
        len(retention),
    )

    return _meridian_input_data.InputData(
        kpi=_xr.DataArray(
            kpi,
            dims=["geo", "time"],
            coords={"geo": [_GEO], "time": time_values},
            name="kpi",
        ),
        kpi_type="non_revenue",
        controls=_xr.DataArray(
            controls,
            dims=["geo", "time", "control_variable"],
            coords={"geo": [_GEO], "time": time_values, "control_variable": control_names},
            name="controls",
        ),
        population=_xr.DataArray(
            np.array([_POPULATION]),
            dims=["geo"],
            coords={"geo": [_GEO]},
            name="population",
        ),
        media=_xr.DataArray(
            media,
            dims=["geo", "media_time", "media_channel"],
            coords={"geo": [_GEO], "media_time": time_values, "media_channel": channel_names},
            name="media",
        ),
        media_spend=_xr.DataArray(
            media_spend,
            dims=["media_channel"],
            coords={"media_channel": channel_names},
            name="media_spend",
        ),
    )
