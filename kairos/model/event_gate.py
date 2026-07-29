"""Calendar-event annotation seam and the gated event retention layer.

The owner's ask: when a new war or another calendar event happens, updating the
events store must flow correctly into future model training, not only into the
weekly pricing computation (which already reads the store through
``pricing_activation.events``). This module is that training seam, in two parts.

Annotation seam (always on, purely additive)
--------------------------------------------
:func:`annotate_event_columns` joins the operator's events store
(``data/calendar_events.csv``, writes owned by ``kairos_api/events_api.py``)
onto the per-break measurement frame by the break's calendar date, adding three
columns and touching nothing else:

  * ``event_active``    0/1, whether the break date lies inside any ACTIVE event,
  * ``event_intensity`` int, the intensity of the covering event (0 when none),
  * ``event_type``      the covering event's type ('' when none).

When several events overlap a date, the MAX-INTENSITY event wins (ties broken
by earliest start date, then type, so the result is deterministic). An
open-ended event (empty end date, a war without a declared end) covers every
date from its start onward. The columns are annotation only: the pooling keys
on ``channel_name`` and ``log_effect`` exclusively, so the emitted coefficients
are byte-identical with or without the seam (proven in
tests/test_qa8_event_training_seam.py).

Gated event layer (measured, self-deciding each rebuild)
--------------------------------------------------------
:func:`event_layer_gate` copies the series-gate discipline
(:mod:`kairos.model.series_gate`): five temporal folds over the measured
breaks, out-of-sample RMSE with versus without an additive ``event_active``
contrast on top of the genre-cell means, and a +2 percent held-out improvement
bar. The verdict is re-measured on EVERY rebuild and written to the
coefficients JSON metadata under :data:`METADATA_KEY`, so the layer
self-activates the day history with genuine event contrast lands, with no code
change. While the verdict is off (today's honest state: the 30-day window sits
entirely inside wartime, so there is no on/off contrast to measure) the gate
never alters any coefficient; it is a recorded verdict, exactly like the
detrend-seasonality gate.

Why ``event_active`` only, not an intensity slope: identifying a per-unit
intensity effect separately from the plain on/off offset requires at least two
distinct intensity levels among the ACTIVE training breaks of every fold, with
real off-days for anchoring. The current window is covered end to end by one
war, so an intensity regressor would be collinear with the offset and any
fitted slope would be noise wearing a number. The annotation carries
``event_intensity`` per break so the richer fit can be added the day the data
supports it; until then the gate measures the honest, identifiable contrast.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.model.series_gate import (
    GATE_FOLDS,
    HOLDOUT_FRACTION,
    _fold_index_sets,
)
from kairos.optimize.event_pricing import DEFAULT_EVENTS_PATH

logger = logging.getLogger(__name__)

# The coefficients-JSON metadata key the gate verdict is written under. The
# events API's model-context payload reads this exact key (tri-state honest:
# absent metadata reads as verdict "unknown" there).
METADATA_KEY = "event_layer_gate"

# The three additive annotation columns the seam adds to the effects frame.
ANNOTATION_COLUMNS: tuple[str, ...] = ("event_active", "event_intensity", "event_type")

# Relative held-out RMSE improvement the event layer must achieve over the
# genre-only baseline to be activated: the same +2 percent bar the series and
# counter-programming gates use, so every optional layer earns its way in
# against the same standard.
EVENT_GATE_MIN_RELATIVE_IMPROVEMENT = 0.02

# Minimum number of test breaks for the gate to run at all (matches the series
# gate's floor) and minimum breaks PER ARM (inside vs outside an event) so a
# handful of breaks on one side can never claim a measured contrast.
_MIN_TEST_BREAKS = 10
_MIN_ARM_BREAKS = 10


@dataclass(frozen=True)
class TrainingEvent:
    """One active stored event as the training seam sees it.

    ``end`` is None for an open-ended event (covers every date from ``start``
    onward). ``intensity`` defaults to 1 when the stored value does not parse:
    an active event with an unreadable intensity is still an event.
    """

    start: date
    end: "date | None"
    intensity: int
    event_type: str


def load_training_events(path: str | Path | None = None) -> list[TrainingEvent]:
    """Read the ACTIVE stored calendar events for training annotation.

    The same tolerant-reader discipline as
    :func:`kairos.optimize.event_pricing.load_price_events`, but for the
    training seam: every active event counts regardless of its price
    multiplier (annotation is about what HAPPENED, not what the operator
    asserts about prices). Inactive rows, rows without a parseable start date,
    and rows whose end precedes their start are skipped. Returns an empty list
    when the store is missing, so a fresh install annotates honestly with
    zeros rather than failing.
    """
    events_path = Path(path) if path is not None else DEFAULT_EVENTS_PATH
    if not events_path.exists():
        return []
    events: list[TrainingEvent] = []
    with open(events_path, "r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("active", "")).strip().lower() != "true":
                continue
            try:
                start = date.fromisoformat(str(row.get("start_date", "")).strip())
            except ValueError:
                continue
            end: "date | None" = None
            end_raw = str(row.get("end_date") or "").strip()
            if end_raw:
                try:
                    end = date.fromisoformat(end_raw)
                except ValueError:
                    continue
                if end < start:
                    continue
            try:
                intensity = int(float(str(row.get("intensity") or "").strip() or 1))
            except ValueError:
                intensity = 1
            events.append(
                TrainingEvent(
                    start=start,
                    end=end,
                    intensity=intensity,
                    event_type=str(row.get("type") or "").strip(),
                )
            )
    return events


def _covering_event(
    day: date, events: list[TrainingEvent]
) -> "TrainingEvent | None":
    """The event covering ``day``, max intensity winning; None when uncovered.

    Ties on intensity break by earliest start date, then type, so overlapping
    same-intensity events resolve deterministically.
    """
    best: "TrainingEvent | None" = None
    for event in events:
        if day < event.start:
            continue
        if event.end is not None and day > event.end:
            continue
        if best is None or (
            (-event.intensity, event.start, event.event_type)
            < (-best.intensity, best.start, best.event_type)
        ):
            best = event
    return best


def annotate_event_columns(
    effects: pd.DataFrame, path: str | Path | None = None
) -> pd.DataFrame:
    """Return ``effects`` with the three event annotation columns added.

    Joins on the break's calendar date (``break_start``). Purely additive: the
    input columns are carried through untouched and the pooling never reads the
    new columns, so coefficients are byte-identical with or without the seam.
    A frame without ``break_start`` (or an empty frame) gets the honest
    defaults: no break can be dated, so nothing is inside an event.
    """
    frame = effects.copy()
    if frame.empty or "break_start" not in frame.columns:
        frame["event_active"] = pd.Series([0] * len(frame), index=frame.index, dtype=int)
        frame["event_intensity"] = pd.Series([0] * len(frame), index=frame.index, dtype=int)
        frame["event_type"] = pd.Series([""] * len(frame), index=frame.index, dtype=object)
        return frame

    events = load_training_events(path)
    days = pd.to_datetime(frame["break_start"], errors="coerce")
    cache: dict[date, tuple[int, int, str]] = {}
    actives: list[int] = []
    intensities: list[int] = []
    types: list[str] = []
    for stamp in days:
        if pd.isna(stamp):
            actives.append(0)
            intensities.append(0)
            types.append("")
            continue
        day = stamp.date()
        found = cache.get(day)
        if found is None:
            event = _covering_event(day, events)
            found = (
                (1, event.intensity, event.event_type) if event is not None else (0, 0, "")
            )
            cache[day] = found
        actives.append(found[0])
        intensities.append(found[1])
        types.append(found[2])
    frame["event_active"] = pd.Series(actives, index=frame.index, dtype=int)
    frame["event_intensity"] = pd.Series(intensities, index=frame.index, dtype=int)
    frame["event_type"] = pd.Series(types, index=frame.index, dtype=object)
    return frame


def _fold_rmses(train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, float]:
    """Out-of-sample (baseline_rmse, event_rmse) for one train/test split.

    Baseline: each test break predicted by its genre cell's training mean
    (global training mean for an unseen cell). Event layer: the same
    prediction plus an additive ``event_active`` contrast fitted on the
    training residuals (mean residual inside events minus mean residual
    outside). A fold whose training data lacks either arm fits a zero contrast
    (honest cold start), so the layer can never help on a fold where it could
    not be measured.
    """
    cell_means = train.groupby("channel_name")["log_effect"].mean().to_dict()
    global_mean = float(train["log_effect"].mean()) if not train.empty else 0.0

    def _predict_base(frame: pd.DataFrame) -> np.ndarray:
        return np.array(
            [cell_means.get(str(c), global_mean) for c in frame["channel_name"]]
        )

    train_resid = train["log_effect"].to_numpy() - _predict_base(train)
    active_mask = train["event_active"].to_numpy() == 1
    if active_mask.any() and (~active_mask).any():
        delta = float(np.mean(train_resid[active_mask]) - np.mean(train_resid[~active_mask]))
    else:
        delta = 0.0

    y_true = test["log_effect"].to_numpy()
    y_base = _predict_base(test)
    y_event = y_base + delta * (test["event_active"].to_numpy() == 1)
    baseline_rmse = float(np.sqrt(np.mean((y_true - y_base) ** 2)))
    event_rmse = float(np.sqrt(np.mean((y_true - y_event) ** 2)))
    return baseline_rmse, event_rmse


def _gate_result(
    verdict: str,
    reason: str,
    held_out_delta_pct: "float | None",
    measured_at: "str | None",
) -> dict[str, object]:
    """The frozen metadata shape: exactly these four keys, nothing else."""
    return {
        "verdict": verdict,
        "reason": reason,
        "held_out_delta_pct": held_out_delta_pct,
        "measured_at": (
            measured_at
            if measured_at is not None
            else datetime.now(timezone.utc).isoformat()
        ),
    }


def event_layer_gate(
    effects: pd.DataFrame,
    path: str | Path | None = None,
    *,
    min_relative_improvement: float = EVENT_GATE_MIN_RELATIVE_IMPROVEMENT,
    measured_at: "str | None" = None,
) -> dict[str, object]:
    """Decide whether the measured event layer earns activation, held out.

    Returns the frozen metadata dict ``{"verdict": "on"|"off", "reason": str,
    "held_out_delta_pct": float|None, "measured_at": iso}``.
    ``held_out_delta_pct`` is the fold-mean relative RMSE improvement of the
    event layer over the genre-only baseline, in percent (positive means the
    layer predicted held-out breaks better); None when the comparison could
    not be run at all. The verdict is "on" only when the improvement clears
    ``min_relative_improvement`` (2 percent), the same bar as the series gate.

    Accepts a frame already carrying the annotation columns (used unchanged),
    or annotates it here from the events store at ``path``. ``measured_at``
    defaults to the wall clock; tests pass a fixed value for determinism.
    """
    needed = {"channel_name", "log_effect"}
    if effects.empty or not needed.issubset(effects.columns):
        return _gate_result(
            "off",
            "no break effects available; the event layer cannot be evaluated",
            None,
            measured_at,
        )

    work = (
        effects
        if set(ANNOTATION_COLUMNS).issubset(effects.columns)
        else annotate_event_columns(effects, path)
    )
    work = work[[c for c in work.columns if c in
                 ("channel_name", "log_effect", "break_start", *ANNOTATION_COLUMNS)]].copy()

    n_total = len(work)
    n_test_target = max(1, int(round(n_total * HOLDOUT_FRACTION)))
    if n_test_target < _MIN_TEST_BREAKS:
        return _gate_result(
            "off",
            f"too few measured breaks ({n_total}) to hold out a reliable test set; "
            "event layer stays off",
            None,
            measured_at,
        )

    n_active = int((work["event_active"] == 1).sum())
    n_inactive = n_total - n_active
    if n_active == 0 or n_inactive == 0:
        side = "inside" if n_inactive == 0 else "outside"
        return _gate_result(
            "off",
            f"no event on/off contrast in the measured window: all {n_total} "
            f"measured breaks lie {side} an active calendar event, so an event "
            "retention effect cannot be separated from the baseline; the gate "
            "re-measures automatically once history with both conditions exists",
            None,
            measured_at,
        )
    if min(n_active, n_inactive) < _MIN_ARM_BREAKS:
        return _gate_result(
            "off",
            f"event contrast too thin to measure: {n_active} breaks inside events "
            f"vs {n_inactive} outside (need at least {_MIN_ARM_BREAKS} per arm); "
            "event layer stays off",
            None,
            measured_at,
        )

    method, pairs = _fold_index_sets(work, work, n_total, n_test_target)
    positional = work.reset_index(drop=True)
    improvements: list[float] = []
    for train_pos, test_pos in pairs:
        train = positional.iloc[train_pos]
        test = positional.iloc[test_pos]
        baseline_rmse, event_rmse = _fold_rmses(train, test)
        if baseline_rmse <= 0.0:
            return _gate_result(
                "off",
                "baseline RMSE is zero on at least one fold (degenerate data); "
                "the event gate cannot compare; event layer stays off",
                None,
                measured_at,
            )
        improvements.append((baseline_rmse - event_rmse) / baseline_rmse)

    statistic = float(np.mean(improvements))
    delta_pct = float(100.0 * statistic)
    fold_kind = "temporal folds" if method == "fold_mean_temporal" else "seeded splits"
    if statistic > min_relative_improvement:
        reason = (
            f"the event_active layer beats the genre-only baseline by "
            f"{delta_pct:.1f} percent held-out RMSE on average over {len(pairs)} "
            f"{fold_kind} ({n_active} breaks inside events, {n_inactive} outside; "
            f"bar {100.0 * min_relative_improvement:.0f} percent); event layer activated"
        )
        verdict = "on"
    else:
        reason = (
            f"the event_active layer does not beat the genre-only baseline by the "
            f"required {100.0 * min_relative_improvement:.0f} percent held-out RMSE "
            f"margin (measured {delta_pct:.1f} percent over {len(pairs)} {fold_kind}; "
            f"{n_active} breaks inside events, {n_inactive} outside); event layer stays off"
        )
        verdict = "off"
    logger.info("Event gate: %s", reason)
    return _gate_result(verdict, reason, delta_pct, measured_at)
