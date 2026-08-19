"""Gated factor families for the audience (expected TVR) model.

The audience model is a pooled multiplicative base (see
:mod:`kairos.model.audience_model`) times one optional factor per family. A
family earns activation the same way every optional Kairos layer does
(:mod:`kairos.model.series_gate`, :mod:`kairos.model.event_gate`): five
temporal folds over the observations, out-of-sample RMSE with versus without
the factor on top of the base, and a +2 percent held-out improvement bar. The
verdict is re-measured on every rebuild, so each family self-activates the day
the data genuinely supports it, and a family whose feature source is absent or
whose window carries no contrast records an honest off verdict with the
reason, never an error and never a forced number.

The eight families are frozen in :data:`FAMILIES`. Seven are cell factors
(a shrunk-toward-zero mean residual per cell, multiplicative in log space);
``competitor_lineup`` is a slope on the continuous competitor pressure, fitted
by least squares on the base residuals, with NaN pressure meaning
family-not-applicable for that row.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

from kairos.model.series_gate import HOLDOUT_FRACTION, _fold_index_sets

logger = logging.getLogger(__name__)

# The frozen family names, exactly as the artifact's gates block records them.
FAMILIES = (
    "weekday_slot",
    "series",
    "calendar_school_and_chol_hamoed",
    "calendar_hanukkah",
    "calendar_religious_blackout",
    "season",
    "operator_events",
    "competitor_lineup",
)

# Relative held-out RMSE improvement a family must achieve over the pooled
# base to activate: the same +2 percent bar the series and event gates use.
AUDIENCE_GATE_MIN_RELATIVE_IMPROVEMENT = 0.02

# Floors mirroring the event gate: minimum held-out observations for the gate
# to run, minimum observations per contrast arm, and minimum observations per
# cell for a multi-cell family to claim a contrast at all.
_MIN_TEST_OBSERVATIONS = 10
_MIN_ARM_OBSERVATIONS = 10
_MIN_CELL_OBSERVATIONS = 10

# Cell-key separator. Channel names carry no pipe and the series canonicalizer
# strips punctuation (including the pipe), so the join is unambiguous; any
# residual pipe in a component is replaced defensively.
CELL_SEPARATOR = "|"

# The flag families: rows where the flag is set form the measured arm, rows
# where it is not are the reference (factor exactly 1.0 by construction).
_FLAG_DESCRIPTIONS = {
    "calendar_school_and_chol_hamoed": "school-holiday or chol-hamoed",
    "calendar_hanukkah": "Hanukkah",
    "calendar_religious_blackout": "religious-blackout (shabbat or yom tov)",
    "operator_events": "operator-event",
}
_FLAGGED_CELL = "flagged"


def cell_key(*parts: object) -> str:
    """Join cell-key components deterministically."""
    return CELL_SEPARATOR.join(
        str(part).replace(CELL_SEPARATOR, "/") for part in parts
    )


def _flag_values(frame: pd.DataFrame, family: str) -> np.ndarray:
    if family == "calendar_school_and_chol_hamoed":
        return (
            frame["cal_is_school_holiday"].astype(bool)
            | frame["cal_is_chol_hamoed"].astype(bool)
        ).to_numpy()
    if family == "calendar_hanukkah":
        return frame["cal_is_hanukkah"].astype(bool).to_numpy()
    if family == "calendar_religious_blackout":
        return frame["cal_religious_blackout"].astype(bool).to_numpy()
    if family == "operator_events":
        return (frame["event_active"].astype(int) == 1).to_numpy()
    raise ValueError(f"{family!r} is not a flag family")


def family_cells(frame: pd.DataFrame, family: str) -> list[Optional[str]]:
    """The per-row factor cell key for one family; None marks a reference or
    not-applicable row (which the factor leaves exactly at the base)."""
    if family == "weekday_slot":
        return [
            cell_key(weekday, band) if 1 <= int(weekday) <= 7 else None
            for weekday, band in zip(frame["cal_weekday_iso"], frame["slot_band"])
        ]
    if family == "series":
        return [
            cell_key(channel, series) if series else None
            for channel, series in zip(frame["channel"], frame["series_key"])
        ]
    if family == "season":
        return [str(season) if season else None for season in frame["cal_season"]]
    if family in _FLAG_DESCRIPTIONS:
        return [
            _FLAGGED_CELL if flagged else None for flagged in _flag_values(frame, family)
        ]
    raise ValueError(f"{family!r} has no cell representation")


def fit_cell_deltas(
    residuals: np.ndarray, cells: list[Optional[str]], shrinkage_k: float
) -> dict[str, float]:
    """Per-cell log-space delta: the cell's residual mean shrunk toward zero.

    ``delta = sum(residuals in cell) / (n + k)``: a thin cell is pulled hard
    toward the base (factor near 1.0), a rich cell is trusted near its own
    mean. This is the same pseudo-count pooling shape the retention layers use.
    """
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for value, cell in zip(residuals, cells):
        if cell is None:
            continue
        sums[cell] = sums.get(cell, 0.0) + float(value)
        counts[cell] = counts.get(cell, 0) + 1
    return {
        cell: sums[cell] / (counts[cell] + float(shrinkage_k)) for cell in sorted(sums)
    }


def cell_deltas_for(
    cells: list[Optional[str]], table: dict[str, float]
) -> np.ndarray:
    """Per-row log deltas from a fitted table (0.0 for reference/unseen cells)."""
    return np.array(
        [0.0 if cell is None else float(table.get(cell, 0.0)) for cell in cells]
    )


def fit_pressure_beta(
    residuals: np.ndarray, pressure: np.ndarray
) -> Optional[dict[str, float]]:
    """Least-squares slope of the base residuals on competitor pressure.

    Fitted only on rows whose pressure is known (finite). Returns None when
    fewer than two such rows exist or the pressure carries no variance, so a
    contrast-free fold honestly contributes nothing.
    """
    mask = np.isfinite(pressure)
    if int(mask.sum()) < 2:
        return None
    x = pressure[mask]
    y = residuals[mask]
    reference = float(np.mean(x))
    centered = x - reference
    denominator = float(np.sum(centered**2))
    if denominator <= 0.0:
        return None
    return {
        "beta": float(np.sum(centered * y) / denominator),
        "reference": reference,
    }


def pressure_deltas_for(
    pressure: np.ndarray, payload: Optional[dict[str, float]]
) -> np.ndarray:
    """Per-row log deltas from a fitted pressure slope (0.0 where unknown)."""
    deltas = np.zeros(len(pressure))
    if payload is None:
        return deltas
    mask = np.isfinite(pressure)
    deltas[mask] = float(payload["beta"]) * (
        pressure[mask] - float(payload["reference"])
    )
    return deltas


def contrast_reason(
    frame: pd.DataFrame, family: str, pressure: Optional[np.ndarray] = None
) -> Optional[str]:
    """A one-line reason when the window carries no honest contrast, else None."""
    total = len(frame)
    if family == "competitor_lineup":
        assert pressure is not None
        finite = pressure[np.isfinite(pressure)]
        if len(finite) < _MIN_ARM_OBSERVATIONS:
            return (
                f"only {len(finite)} of {total} observations carry a known "
                f"competitor pressure (need at least {_MIN_ARM_OBSERVATIONS}); "
                "no competitor contrast to measure"
            )
        if float(np.std(finite)) <= 0.0:
            return (
                "competitor pressure is constant across the measured window; "
                "no contrast to measure"
            )
        return None

    if family in _FLAG_DESCRIPTIONS:
        description = _FLAG_DESCRIPTIONS[family]
        flagged = int(np.sum(_flag_values(frame, family)))
        reference = total - flagged
        if flagged == 0:
            return (
                f"no {description} days in the measured window; nothing to "
                "contrast against ordinary days"
            )
        if reference == 0:
            return (
                f"every one of the {total} observations falls on {description} "
                "days; no ordinary days to contrast against"
            )
        if min(flagged, reference) < _MIN_ARM_OBSERVATIONS:
            return (
                f"contrast too thin to measure: {flagged} observations on "
                f"{description} days vs {reference} ordinary (need at least "
                f"{_MIN_ARM_OBSERVATIONS} per arm)"
            )
        return None

    cells = family_cells(frame, family)
    counts: dict[str, int] = {}
    for cell in cells:
        if cell is not None:
            counts[cell] = counts.get(cell, 0) + 1
    rich = sum(1 for n in counts.values() if n >= _MIN_CELL_OBSERVATIONS)
    if rich < 2:
        return (
            f"fewer than two {family} cells with at least "
            f"{_MIN_CELL_OBSERVATIONS} observations ({rich} of {len(counts)} "
            "cells qualify); no contrast to measure"
        )
    return None


def _gate(
    verdict: str,
    reason: str,
    held_out_delta_pct: Optional[float],
    measured_at: Optional[str],
) -> dict[str, object]:
    """The frozen gate shape: exactly these four keys."""
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


def gate_off(reason: str, measured_at: Optional[str] = None) -> dict[str, object]:
    """An honest off verdict for a family that could not be evaluated."""
    return _gate("off", reason, None, measured_at)


def gate_family(
    frame: pd.DataFrame,
    family: str,
    base_fit: Callable[[pd.DataFrame], Callable[[pd.DataFrame], np.ndarray]],
    *,
    log_tvr: np.ndarray,
    shrinkage_k: float,
    pressure: Optional[np.ndarray] = None,
    active: Sequence[str] = (),
    min_relative_improvement: float = AUDIENCE_GATE_MIN_RELATIVE_IMPROVEMENT,
    measured_at: Optional[str] = None,
) -> dict[str, object]:
    """Decide one family's verdict on held-out observations.

    MEASURED AGAINST THE MODEL AS IT WILL ACTUALLY BE, not against the bare
    base. ``active`` names the families already activated on this rebuild; each
    is fitted on the fold's own base residual and added to the reference, which
    is exactly what :meth:`AudienceModel.score` does with them at predict time.

    The alternative — scoring every family alone against the base — is what this
    used to do, and it admits a family that then makes the model worse. Measured
    on the real frame over five temporal folds: a repeat family scores +11%
    against the bare base and, added to the two families that ship, takes the
    composed model 14.21% the wrong way. Both families that ship pass either
    way (+25.5% and +9.5% composed), so this costs nothing today and refuses
    that one honestly.

    ``frame`` must be positionally indexed (``reset_index(drop=True)``) with
    ``log_tvr`` (and ``pressure`` for the competitor family) aligned to its
    positions. ``base_fit`` refits the pooled base on each fold's training
    rows and returns the fold's log-base predictor, so every fold's factor is
    measured against a base that never saw the test rows.
    """
    reason = contrast_reason(frame, family, pressure)
    if reason is not None:
        return gate_off(reason, measured_at)

    n_total = len(frame)
    n_test_target = max(1, int(round(n_total * HOLDOUT_FRACTION)))
    if n_test_target < _MIN_TEST_OBSERVATIONS:
        return gate_off(
            f"too few observations ({n_total}) to hold out a reliable test "
            f"set; the {family} factor stays off",
            measured_at,
        )

    method, pairs = _fold_index_sets(frame, frame, n_total, n_test_target)
    improvements: list[float] = []
    for train_positions, test_positions in pairs:
        train = frame.iloc[train_positions]
        test = frame.iloc[test_positions]
        predict = base_fit(train)
        residuals = log_tvr[train_positions] - predict(train)
        # The reference is the base PLUS whatever is already activated, each
        # fitted the way the model fits it: on the base residual, then summed.
        reference = predict(test)
        for prior in active:
            if prior == family or prior == "competitor_lineup":
                continue
            prior_table = fit_cell_deltas(
                residuals, family_cells(train, prior), shrinkage_k)
            reference = reference + np.nan_to_num(np.asarray(
                cell_deltas_for(family_cells(test, prior), prior_table), dtype=float))
        if family == "competitor_lineup":
            payload = fit_pressure_beta(residuals, pressure[train_positions])
            deltas = pressure_deltas_for(pressure[test_positions], payload)
        else:
            table = fit_cell_deltas(residuals, family_cells(train, family), shrinkage_k)
            deltas = cell_deltas_for(family_cells(test, family), table)
        y_true = log_tvr[test_positions]
        y_base = reference
        rmse_base = float(np.sqrt(np.mean((y_true - y_base) ** 2)))
        if rmse_base <= 0.0:
            return gate_off(
                "base RMSE is zero on at least one fold (degenerate data); "
                f"the gate cannot compare; the {family} factor stays off",
                measured_at,
            )
        rmse_family = float(np.sqrt(np.mean((y_true - y_base - deltas) ** 2)))
        improvements.append((rmse_base - rmse_family) / rmse_base)

    statistic = float(np.mean(improvements))
    delta_pct = float(100.0 * statistic)
    fold_kind = "temporal folds" if method == "fold_mean_temporal" else "seeded splits"
    bar_pct = 100.0 * min_relative_improvement
    if statistic > min_relative_improvement:
        reason = (
            f"the {family} factor beats the pooled base by {delta_pct:.1f} "
            f"percent held-out RMSE on average over {len(pairs)} {fold_kind} "
            f"(bar {bar_pct:.0f} percent); factor activated"
        )
        verdict = "on"
    else:
        reason = (
            f"the {family} factor does not beat the pooled base by the "
            f"required {bar_pct:.0f} percent held-out RMSE margin (measured "
            f"{delta_pct:.1f} percent over {len(pairs)} {fold_kind}); factor "
            "stays off"
        )
        verdict = "off"
    logger.info("Audience gate [%s]: %s", family, reason)
    return _gate(verdict, reason, delta_pct, measured_at)
