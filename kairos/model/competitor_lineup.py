"""Competitor lineup opposite the operator's channel, from the spots history.

The owner ask behind this module: expected rating must account for the parallel
programmes on the other channels. The multi-channel aired-spots log carries four
channels of spots with programme titles, air clocks and observed TVR, so the
competitor lineup at any historical moment, and each rival title's typical
audience strength, are derivable from the history itself. Forward (plan-week)
dates resolve their lineup through the published competitor EPG
(:mod:`kairos.model.future_epg`) when that file is present.

What this module computes
-------------------------
1. :func:`collapse_airings`: spots -> programme airings. Spots sharing a channel
   and title, consecutive in air order with gaps no larger than
   :data:`MAX_INTRA_AIRING_GAP_MINUTES`, collapse into one airing whose span runs
   from the first spot's clock to the last spot's clock plus its duration, and
   whose ``mean_tvr`` is the mean of the spots' observed TVR. The span is
   necessarily the BREAK-observed span, not the full programme span: spots exist
   only where the rival took a break, so an airing understates the programme's
   true start/end by up to one programme act on each side. That limitation is
   inherent to a spots-derived lineup and is stated here rather than papered over.
2. :func:`title_strengths`: each (channel, title)'s historical mean TVR,
   empirical-Bayes shrunk toward its channel's mean by airing count, so a title
   seen once cannot spike the pressure signal on the strength of one noisy
   airing: ``strength = (n * title_mean + K * channel_mean) / (n + K)`` with
   ``K =`` :data:`EB_PRIOR_AIRINGS`.
3. :func:`lineup_frame`: per requested date, the piecewise-constant competitor
   lineup as change-point windows with ``competitor_pressure`` = the sum of the
   opposite titles' strengths. A title with no usable TVR history contributes
   0.0 (honest: its audience is unknown, nothing is invented) while still being
   listed in ``competitor_titles``.
4. :func:`pressure_for_window`: the overlap-weighted pressure for an arbitrary
   query window against those change-point windows.

Null versus zero (load-bearing)
-------------------------------
A date the history does not cover and the forward EPG does not cover yields ONE
full-day row with ``competitor_pressure`` NaN (null): the lineup is unknown and
the audience model must treat the competitor family as not applicable there,
never as "no competition". A date that IS covered but has no opposite airing in
some window yields 0.0 for that window: a known-empty lineup. The per-date
resolution source is reported in ``frame.attrs["coverage"]``.

Overlap convention (documented contract)
----------------------------------------
Lineup windows are change-point cells, so pressure is constant within each
window by construction. A query window that straddles cells receives the
time-weighted mean: ``sum(pressure_i * overlap_seconds_i) / sum(overlap_i)``,
each window weighted by its overlap fraction of the query window. If any
overlapped window is null the whole query is null: partial knowledge is not
averaged with ignorance.
"""

from __future__ import annotations

from typing import Any, Iterable, Optional

import pandas as pd

# A same-title gap larger than this splits two airings. Asserted, not measured:
# commercial breaks inside one airing recur every few minutes; a repeat airing
# of the same title later in the day sits hours away. 60 minutes separates the
# two regimes with a wide margin on this schedule data.
MAX_INTRA_AIRING_GAP_MINUTES = 60.0

# Empirical-Bayes prior strength, in pseudo-airings, for shrinking a title's
# mean TVR toward its channel mean. Asserted, not measured: with K=5 a
# one-airing title sits 5/6 of the way toward the channel mean, and a title
# with a month of airings keeps essentially its own mean.
EB_PRIOR_AIRINGS = 5.0

SECONDS_PER_DAY = 86_400

LINEUP_COLUMNS = (
    "date",
    "start_seconds",
    "end_seconds",
    "competitor_pressure",
    "competitor_titles",
)


def collapse_airings(spots: pd.DataFrame) -> pd.DataFrame:
    """Collapse an aired-spots log into programme airings.

    ``spots`` needs columns ``Channel``, ``Title``, ``air_dt``, ``Duration``,
    and optionally ``TVR`` (as loaded by :func:`kairos.data.loaders.load_spots`).
    Returns one row per airing: ``channel``, ``title``, ``start_dt``, ``end_dt``,
    ``mean_tvr`` (NaN when no spot in the airing carries a TVR), ``spot_count``.
    Rows with no parseable air time or an empty title are dropped, not guessed.
    """
    columns = ["channel", "title", "start_dt", "end_dt", "mean_tvr", "spot_count"]
    if spots is None or spots.empty:
        return pd.DataFrame(columns=columns)
    frame = spots.copy()
    frame = frame[frame["air_dt"].notna() & frame["Title"].notna()]
    frame["Title"] = frame["Title"].astype(str).str.strip()
    frame = frame[frame["Title"] != ""]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    frame["Duration"] = pd.to_numeric(frame.get("Duration"), errors="coerce").fillna(0.0)
    if "TVR" in frame.columns:
        frame["TVR"] = pd.to_numeric(frame["TVR"], errors="coerce")
    else:
        frame["TVR"] = float("nan")
    frame = frame.sort_values(["Channel", "air_dt"], kind="stable")
    gap = pd.Timedelta(minutes=MAX_INTRA_AIRING_GAP_MINUTES)

    records: list[dict[str, Any]] = []
    for channel, group in frame.groupby("Channel", sort=False):
        current: Optional[dict[str, Any]] = None
        for row in group.itertuples(index=False):
            start = pd.Timestamp(getattr(row, "air_dt"))
            end = start + pd.Timedelta(seconds=float(getattr(row, "Duration")))
            title = str(getattr(row, "Title"))
            tvr = getattr(row, "TVR")
            if (
                current is not None
                and current["title"] == title
                and start - current["end_dt"] <= gap
            ):
                current["end_dt"] = max(current["end_dt"], end)
                current["spot_count"] += 1
                if pd.notna(tvr):
                    current["tvrs"].append(float(tvr))
            else:
                if current is not None:
                    records.append(current)
                current = {
                    "channel": str(channel),
                    "title": title,
                    "start_dt": start,
                    "end_dt": end,
                    "spot_count": 1,
                    "tvrs": [float(tvr)] if pd.notna(tvr) else [],
                }
        if current is not None:
            records.append(current)

    for record in records:
        tvrs = record.pop("tvrs")
        record["mean_tvr"] = (sum(tvrs) / len(tvrs)) if tvrs else float("nan")
    out = pd.DataFrame(records, columns=columns)
    return out.sort_values(["channel", "start_dt"], kind="stable").reset_index(drop=True)


def title_strengths(airings: pd.DataFrame) -> pd.DataFrame:
    """Each (channel, title)'s EB-shrunk historical mean TVR.

    Only airings with an observed ``mean_tvr`` count toward ``n_airings`` and the
    means. ``strength = (n * title_mean + K * channel_mean) / (n + K)`` with
    ``K =`` :data:`EB_PRIOR_AIRINGS`; a title with zero measured airings gets a
    NaN strength (unknown, never fabricated from the prior alone). Returns
    columns ``channel``, ``title``, ``n_airings``, ``title_mean_tvr``,
    ``channel_mean_tvr``, ``strength``.
    """
    columns = ["channel", "title", "n_airings", "title_mean_tvr", "channel_mean_tvr", "strength"]
    if airings is None or airings.empty:
        return pd.DataFrame(columns=columns)
    measured = airings[airings["mean_tvr"].notna()]
    channel_means = measured.groupby("channel")["mean_tvr"].mean()
    records: list[dict[str, Any]] = []
    for (channel, title), group in airings.groupby(["channel", "title"], sort=False):
        with_tvr = group[group["mean_tvr"].notna()]
        n = int(len(with_tvr))
        channel_mean = float(channel_means.get(channel, float("nan")))
        if n == 0 or pd.isna(channel_mean):
            title_mean = float("nan")
            strength = float("nan")
        else:
            title_mean = float(with_tvr["mean_tvr"].mean())
            strength = (n * title_mean + EB_PRIOR_AIRINGS * channel_mean) / (
                n + EB_PRIOR_AIRINGS
            )
        records.append(
            {
                "channel": str(channel),
                "title": str(title),
                "n_airings": n,
                "title_mean_tvr": title_mean,
                "channel_mean_tvr": channel_mean,
                "strength": strength,
            }
        )
    return pd.DataFrame(records, columns=columns).reset_index(drop=True)


def _normalize_dates(dates: Iterable[Any]) -> list[pd.Timestamp]:
    """The requested dates as unique, sorted, midnight-normalized timestamps."""
    normalized: set[pd.Timestamp] = set()
    for value in dates:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp):
            raise ValueError(f"unparseable lineup date: {value!r}")
        normalized.add(stamp.normalize())
    return sorted(normalized)


def _windows_for_day(
    day: pd.Timestamp,
    intervals: list[tuple[pd.Timestamp, pd.Timestamp, str, str]],
    strength_by_key: dict[tuple[str, str], float],
) -> list[dict[str, Any]]:
    """Change-point windows partitioning [day, day+1) from (start, end, channel, title).

    Boundaries are every clipped interval edge plus the day edges; within each
    cell every interval either covers the whole cell or none of it, so pressure
    is exact per cell (no partial overlap exists at this stage; partial overlap
    arises only for query windows, handled in :func:`pressure_for_window`).
    Adjacent cells with identical pressure and titles are merged.
    """
    day_end = day + pd.Timedelta(days=1)
    clipped: list[tuple[pd.Timestamp, pd.Timestamp, str, str]] = []
    for start, end, channel, title in intervals:
        lo, hi = max(start, day), min(end, day_end)
        if hi > lo:
            clipped.append((lo, hi, channel, title))
    edges = {day, day_end}
    for lo, hi, _c, _t in clipped:
        edges.add(lo)
        edges.add(hi)
    ordered = sorted(edges)
    windows: list[dict[str, Any]] = []
    for cell_start, cell_end in zip(ordered, ordered[1:]):
        covering = [
            (channel, title)
            for lo, hi, channel, title in clipped
            if lo <= cell_start and hi >= cell_end
        ]
        pressure = sum(
            strength
            for key in covering
            if pd.notna(strength := strength_by_key.get(key, float("nan")))
        )
        titles = ";".join(sorted({title for _channel, title in covering}))
        start_s = int((cell_start - day).total_seconds())
        end_s = int((cell_end - day).total_seconds())
        if windows and windows[-1]["competitor_pressure"] == pressure and windows[-1][
            "competitor_titles"
        ] == titles:
            windows[-1]["end_seconds"] = end_s
        else:
            windows.append(
                {
                    "date": day.date().isoformat(),
                    "start_seconds": start_s,
                    "end_seconds": end_s,
                    "competitor_pressure": float(pressure),
                    "competitor_titles": titles,
                }
            )
    return windows


def _null_day_row(day: pd.Timestamp) -> dict[str, Any]:
    """The honest full-day row for a date whose lineup is unknown."""
    return {
        "date": day.date().isoformat(),
        "start_seconds": 0,
        "end_seconds": SECONDS_PER_DAY,
        "competitor_pressure": float("nan"),
        "competitor_titles": "",
    }


def lineup_frame(
    dates: Iterable[Any],
    owned_channel: str,
    *,
    spots: Optional[pd.DataFrame] = None,
    epg: Optional[pd.DataFrame] = "__load__",  # type: ignore[assignment]
) -> pd.DataFrame:
    """The competitor lineup windows for the requested dates.

    Returns a DataFrame with exactly :data:`LINEUP_COLUMNS`. Per date, the
    resolution order is: the spots HISTORY when any rival airing starts on that
    date (real observed lineup); else the FORWARD EPG when its date window
    covers the date (published plan, audience strength still strictly
    historical); else one full-day row with null pressure, because an unknown
    lineup is null, never 0.0 pretending to know. ``frame.attrs["coverage"]``
    maps each ISO date to its source: ``history``, ``forward_epg`` or
    ``unknown``. ``spots`` and ``epg`` default to the real loaders
    (:func:`kairos.data.loaders.load_spots`,
    :func:`kairos.model.future_epg.load_future_competitor_epg`); pass ``epg=None``
    to run with no forward EPG. The owned channel never contributes pressure.
    """
    if spots is None:
        from kairos.data.loaders import load_spots

        spots = load_spots()
    if isinstance(epg, str) and epg == "__load__":
        from kairos.model.future_epg import load_future_competitor_epg

        epg, _status = load_future_competitor_epg()

    airings = collapse_airings(spots)
    strengths = title_strengths(airings)
    strength_by_key = {
        (row.channel, row.title): float(row.strength)
        for row in strengths.itertuples(index=False)
    }
    rival_airings = airings[airings["channel"] != str(owned_channel)]

    epg_intervals: list[tuple[pd.Timestamp, pd.Timestamp, str, str]] = []
    epg_window: Optional[tuple[pd.Timestamp, pd.Timestamp]] = None
    if epg is not None and not epg.empty:
        usable = epg[epg["start_dt"].notna() & epg["end_dt"].notna()]
        rivals_epg = usable[usable["Channel"].astype(str) != str(owned_channel)]
        for row in rivals_epg.itertuples(index=False):
            epg_intervals.append(
                (
                    pd.Timestamp(getattr(row, "start_dt")),
                    pd.Timestamp(getattr(row, "end_dt")),
                    str(getattr(row, "Channel")),
                    str(getattr(row, "Title")).strip(),
                )
            )
        if epg_intervals:
            epg_window = (
                min(lo for lo, _hi, _c, _t in epg_intervals).normalize(),
                max(lo for lo, _hi, _c, _t in epg_intervals).normalize(),
            )

    rows: list[dict[str, Any]] = []
    coverage: dict[str, str] = {}
    for day in _normalize_dates(dates):
        day_end = day + pd.Timedelta(days=1)
        starts_today = rival_airings[
            (rival_airings["start_dt"] >= day) & (rival_airings["start_dt"] < day_end)
        ]
        if not starts_today.empty:
            overlapping = rival_airings[
                (rival_airings["start_dt"] < day_end) & (rival_airings["end_dt"] > day)
            ]
            intervals = [
                (row.start_dt, row.end_dt, row.channel, row.title)
                for row in overlapping.itertuples(index=False)
            ]
            rows.extend(_windows_for_day(day, intervals, strength_by_key))
            coverage[day.date().isoformat()] = "history"
        elif epg_window is not None and epg_window[0] <= day <= epg_window[1]:
            intervals = [
                (lo, hi, channel, title)
                for lo, hi, channel, title in epg_intervals
                if lo < day_end and hi > day
            ]
            rows.extend(_windows_for_day(day, intervals, strength_by_key))
            coverage[day.date().isoformat()] = "forward_epg"
        else:
            rows.append(_null_day_row(day))
            coverage[day.date().isoformat()] = "unknown"

    frame = pd.DataFrame(rows, columns=list(LINEUP_COLUMNS))
    frame["competitor_pressure"] = frame["competitor_pressure"].astype(float)
    frame.attrs["coverage"] = coverage
    return frame


def pressure_for_window(
    frame: pd.DataFrame,
    date: Any,
    start_seconds: float,
    end_seconds: float,
) -> dict[str, Any]:
    """The overlap-weighted competitor pressure for one query window.

    ``frame`` is a :func:`lineup_frame` result; the query window is
    ``[start_seconds, end_seconds)`` on ``date``, clipped to the date's
    ``[0, 86400)``. The weighting convention: each lineup window contributes its
    pressure times its overlap fraction of the query window (time-weighted
    mean). Returns ``competitor_pressure`` (float, or None when the date is
    absent from the frame or any overlapped window is null),
    ``competitor_titles`` (sorted unique titles over the overlap) and
    ``covered_seconds``.
    """
    day_iso = pd.Timestamp(date).normalize().date().isoformat()
    lo = max(0.0, float(start_seconds))
    hi = min(float(SECONDS_PER_DAY), float(end_seconds))
    day_rows = frame[frame["date"] == day_iso]
    if day_rows.empty or hi <= lo:
        return {"competitor_pressure": None, "competitor_titles": [], "covered_seconds": 0.0}
    weighted = 0.0
    covered = 0.0
    titles: set[str] = set()
    for row in day_rows.itertuples(index=False):
        overlap = min(hi, float(row.end_seconds)) - max(lo, float(row.start_seconds))
        if overlap <= 0:
            continue
        if pd.isna(row.competitor_pressure):
            return {
                "competitor_pressure": None,
                "competitor_titles": [],
                "covered_seconds": 0.0,
            }
        weighted += float(row.competitor_pressure) * overlap
        covered += overlap
        if row.competitor_titles:
            titles.update(str(row.competitor_titles).split(";"))
    if covered <= 0:
        return {"competitor_pressure": None, "competitor_titles": [], "covered_seconds": 0.0}
    return {
        "competitor_pressure": weighted / covered,
        "competitor_titles": sorted(titles),
        "covered_seconds": covered,
    }
