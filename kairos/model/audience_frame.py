"""Observation frames for the audience (expected TVR) model.

The audience model predicts the rating a programme slot is expected to draw
(expected TVR), distinct from the retention model that prices the cost of
interrupting it. Its unit of observation is one programme in one clock-hour
slot on one date and channel, aggregated honestly from the multi-channel
aired-spots history: every spot in a group carries the real measured TVR of
its break, and the observation's ``tvr`` is the plain mean of those measured
values, never an invented number.

Two frames are built here:

  * :func:`build_training_frame` turns the spots history into training
    observations annotated with the deterministic Israeli calendar
    (``cal_*`` columns, :mod:`kairos.data.israel_calendar`), the operator
    events store (``event_*`` columns, :mod:`kairos.model.event_gate`), the
    canonical series key and the classified genre.
  * :func:`prediction_frame` normalizes prediction-surface rows
    (``date, channel, program_title, start_seconds, duration_seconds``) into
    the same vocabulary, so the model scores with exactly the features it
    trained on.

Competitor pressure is attached tolerantly by :func:`attach_pressure`: the
lineup source (:mod:`kairos.model.competitor_lineup`, built separately) may be
absent, and a date without a known lineup yields NaN pressure, which the model
treats as family-not-applicable for that row. Nothing is ever guessed.

All functions are pure: inputs are copied, never mutated, and no state is
written.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.israel_calendar import annotate_calendar
from kairos.data.loaders import load_spots
from kairos.data.title_features import canonicalize_series
from kairos.model.event_gate import annotate_event_columns

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600

# The broadcast slot bands, a deterministic function of the clock hour. These
# are the audience-level dayparts (when people watch), not the pricing classes:
# late night runs past midnight into the small hours, overnight is the dead
# zone, and prime is the 20:00-23:00 core.
SLOT_BANDS = ("overnight", "morning", "afternoon", "access", "prime", "late")

# Columns the prediction surface requires on its input rows (frozen contract).
PREDICTION_COLUMNS = (
    "date",
    "channel",
    "program_title",
    "start_seconds",
    "duration_seconds",
)

# The training-frame columns before calendar/event annotation, in order.
_CORE_COLUMNS = (
    "date",
    "channel",
    "title",
    "series_key",
    "genre",
    "slot_hour",
    "slot_band",
    "start_seconds",
    "tvr",
    "n_spots",
)


def slot_band_of_hour(hour: int) -> str:
    """The slot band of a clock hour (0..23)."""
    hour = int(hour) % 24
    if 2 <= hour < 6:
        return "overnight"
    if 6 <= hour < 12:
        return "morning"
    if 12 <= hour < 17:
        return "afternoon"
    if 17 <= hour < 20:
        return "access"
    if 20 <= hour < 23:
        return "prime"
    return "late"


def _genre_lookup(
    titles: Iterable[str], classifier: Optional[ProgramClassifier]
) -> dict[str, str]:
    """Classify each unique title once (the classifier is deterministic)."""
    classifier = classifier or ProgramClassifier.from_yaml()
    return {title: classifier.classify(title).category for title in titles}


def _annotate(
    frame: pd.DataFrame,
    *,
    calendar_path=None,
    events_path=None,
) -> pd.DataFrame:
    """Add the ``cal_*`` and ``event_*`` columns keyed on the ``date`` column.

    ``break_start`` is set to the observation date so the operator-events seam
    (which joins by break date) and the temporal fold builder (which orders by
    ``break_start``) both work on this frame unchanged.
    """
    frame = annotate_calendar(frame, "date", path=calendar_path)
    frame["break_start"] = frame["date"]
    return annotate_event_columns(frame, events_path)


def _empty_training_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=list(_CORE_COLUMNS))


def build_training_frame(
    spots: Optional[pd.DataFrame] = None,
    *,
    classifier: Optional[ProgramClassifier] = None,
    calendar_path=None,
    events_path=None,
) -> pd.DataFrame:
    """One training observation per (date, channel, title, clock hour).

    ``spots`` is the aired-spots history in the shape of
    :func:`kairos.data.loaders.load_spots` (``Channel``, ``Title``, ``TVR``,
    ``air_dt``); the default loads the reference file. Spots without a
    parseable air time or a non-negative numeric TVR are dropped (no rating is
    ever invented). The observation ``tvr`` is the plain mean of the group's
    spot TVRs and ``n_spots`` counts them, so the aggregation is auditable.
    """
    spots = load_spots() if spots is None else spots
    needed = {"Channel", "Title", "TVR", "air_dt"}
    if spots.empty or not needed.issubset(spots.columns):
        return _empty_training_frame()

    work = spots.loc[spots["air_dt"].notna(), ["Channel", "Title", "TVR", "air_dt"]].copy()
    work["tvr"] = pd.to_numeric(work["TVR"], errors="coerce")
    work = work[work["tvr"].notna() & (work["tvr"] >= 0)]
    if work.empty:
        return _empty_training_frame()

    stamps = pd.to_datetime(work["air_dt"])
    work["date"] = stamps.dt.normalize()
    work["slot_hour"] = stamps.dt.hour.astype(int)
    work["_seconds"] = (
        stamps.dt.hour * SECONDS_PER_HOUR + stamps.dt.minute * 60 + stamps.dt.second
    ).astype(float)

    grouped = (
        work.groupby(["date", "Channel", "Title", "slot_hour"], sort=True)
        .agg(
            tvr=("tvr", "mean"),
            start_seconds=("_seconds", "min"),
            n_spots=("tvr", "size"),
        )
        .reset_index()
        .rename(columns={"Channel": "channel", "Title": "title"})
    )
    grouped["channel"] = grouped["channel"].astype(str)
    grouped["title"] = grouped["title"].astype(str)
    grouped["series_key"] = grouped["title"].map(canonicalize_series)
    genres = _genre_lookup(grouped["title"].unique(), classifier)
    grouped["genre"] = grouped["title"].map(genres)
    grouped["slot_band"] = grouped["slot_hour"].map(slot_band_of_hour)
    grouped = grouped[list(_CORE_COLUMNS)]

    frame = _annotate(grouped, calendar_path=calendar_path, events_path=events_path)
    return (
        frame.sort_values(["date", "channel", "start_seconds", "title"], kind="stable")
        .reset_index(drop=True)
    )


def prediction_frame(
    rows: pd.DataFrame,
    *,
    classifier: Optional[ProgramClassifier] = None,
    calendar_path=None,
    events_path=None,
) -> pd.DataFrame:
    """Normalize prediction rows into the training vocabulary.

    ``rows`` must carry :data:`PREDICTION_COLUMNS`. The slot is derived from
    ``start_seconds`` exactly as training derives it from the air time, the
    series key and genre come from the same canonicalizer and classifier, and
    the same calendar and event annotations are added. An unparseable date
    keeps the neutral calendar features (the annotator's contract), it never
    raises.
    """
    missing = [c for c in PREDICTION_COLUMNS if c not in rows.columns]
    if missing:
        raise KeyError(f"prediction rows missing required column(s): {missing}")

    out = pd.DataFrame(index=rows.index)
    out["date"] = pd.to_datetime(rows["date"], errors="coerce").dt.normalize()
    out["channel"] = rows["channel"].astype(str)
    out["title"] = rows["program_title"].astype(str)
    seconds = pd.to_numeric(rows["start_seconds"], errors="coerce").fillna(0.0)
    out["start_seconds"] = seconds.astype(float)
    out["duration_seconds"] = pd.to_numeric(rows["duration_seconds"], errors="coerce")
    out["slot_hour"] = (seconds // SECONDS_PER_HOUR).astype(int) % 24
    out["slot_band"] = out["slot_hour"].map(slot_band_of_hour)
    out["series_key"] = out["title"].map(canonicalize_series)
    genres = _genre_lookup(out["title"].unique(), classifier)
    out["genre"] = out["title"].map(genres)
    return _annotate(out, calendar_path=calendar_path, events_path=events_path)


def attach_pressure(
    frame: pd.DataFrame,
    owned_channel: str,
    *,
    lineup_frame_fn: Optional[Callable] = None,
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Per-row competitor pressure opposite the owned channel, or an honest reason.

    Returns ``(pressure, reason)``. ``pressure`` is a float array aligned to
    ``frame`` positions where NaN marks a row for which the pressure is
    unknown or undefined (a row on a rival channel, a date without a lineup, a
    null pressure window); the audience model treats NaN as
    family-not-applicable for that row. ``pressure`` is None (with a one-line
    ``reason``) when the lineup source cannot contribute at all: the module is
    absent, no operator channel is configured, or the lineup call fails or
    returns nothing. Nothing here fabricates a zero.
    """
    if lineup_frame_fn is None:
        try:
            from kairos.model.competitor_lineup import lineup_frame as lineup_frame_fn
        except Exception:  # noqa: BLE001 - absence of the parallel module is expected
            return None, (
                "competitor lineup source unavailable "
                "(kairos.model.competitor_lineup could not be imported); "
                "the family cannot be evaluated"
            )
    if not owned_channel:
        return None, (
            "no operator channel configured in settings; competitor pressure "
            "opposite the owned channel is undefined"
        )
    dates = sorted(
        {stamp.date().isoformat() for stamp in frame["date"] if pd.notna(stamp)}
    )
    if not dates:
        return None, "no dated observations; competitor pressure cannot be joined"
    try:
        lineup = lineup_frame_fn(dates, owned_channel)
    except Exception as exc:  # noqa: BLE001 - a broken lineup must not break training
        logger.warning("competitor lineup_frame failed: %s", exc)
        return None, f"competitor lineup could not be built: {exc}"
    if lineup is None or len(lineup) == 0:
        return None, "competitor lineup is empty for the measured window"

    windows: dict[str, list[tuple[float, float, float]]] = {}
    for row in lineup.itertuples(index=False):
        stamp = pd.to_datetime(getattr(row, "date"), errors="coerce")
        raw = getattr(row, "competitor_pressure")
        if pd.isna(stamp):
            continue
        value = float("nan") if raw is None or pd.isna(raw) else float(raw)
        windows.setdefault(stamp.date().isoformat(), []).append(
            (float(getattr(row, "start_seconds")), float(getattr(row, "end_seconds")), value)
        )

    values = np.full(len(frame), np.nan)
    positions = zip(frame["date"].tolist(), frame["channel"].tolist(), frame["start_seconds"].tolist())
    for index, (stamp, channel, start) in enumerate(positions):
        if pd.isna(stamp) or str(channel) != owned_channel:
            continue
        for win_start, win_end, value in windows.get(stamp.date().isoformat(), ()):
            if win_start <= float(start) < win_end:
                values[index] = value
                break
    return values, None
