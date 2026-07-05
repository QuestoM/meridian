"""Held-out gate for the season-aware detrend baseline (evaluate, never enable).

The shipped detrend baseline (:func:`kairos.model.measure._baseline_levels`)
is one typical audience curve per channel, averaged over every day in the data
window. On the one-month reference data that is exactly right. On a 24-month
window the same average smears winter into summer and the "typical" level at a
minute stops being typical of any actual day, biasing every measured
coefficient. The "month_minute" mode
(:func:`kairos.model.measure._seasonal_baseline_levels`) keeps one curve per
calendar month of the broadcast day, with a documented minimum-sample fallback
to the global curve for thin cells.

Gate design (same discipline as :mod:`kairos.model.series_gate` and
:mod:`kairos.model.competitor_gate`)
-------------------------------------
* A deterministic 20 percent of broadcast DAYS are held out. Whole days, not
  rows: seasonality is day-level structure, and predicting a withheld day's
  minute curve from the other days is the honest test of it.
* GLOBAL prediction: the training days' (channel, minute) mean TVR.
* MONTH_MINUTE prediction: the training days' (channel, month, minute) mean
  where the cell carries at least ``min_samples`` days, else the global mean
  (the same fallback the measurement mode applies).
* Minutes the global baseline cannot predict (channel-minute never seen in
  training) are skipped in BOTH arms, so the comparison is like for like.
* month_minute is recommended only when its RMSE beats the global RMSE by at
  least :data:`DETREND_GATE_MIN_RELATIVE_IMPROVEMENT` (2 percent, the bar
  every optional layer clears).

Evaluate-only (load-bearing). Nothing reads this verdict to switch the
measurement. The rebuild records it in the coefficients JSON metadata so the
decision is on the table at the two-year data drop; actually passing
``baseline_seasonality="month_minute"`` to
:func:`kairos.model.measure.break_effects` stays an explicit owner decision.
On a single-month window the two baselines coincide by construction (one month
means one seasonal cell per minute), so the verdict is honestly "no
improvement" and nothing moves.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from kairos.model.measure import (
    _SEASONAL_MIN_SAMPLES,
    _baseline_levels,
    _dayparts_frame,
    _seasonal_baseline_levels,
)

logger = logging.getLogger(__name__)

# Relative RMSE improvement month_minute must achieve over the global baseline
# to be recommended. Matches SERIES_GATE_MIN_RELATIVE_IMPROVEMENT and
# COUNTERPROGRAMMING_MIN_RELATIVE_IMPROVEMENT so every optional layer clears
# the same bar.
DETREND_GATE_MIN_RELATIVE_IMPROVEMENT = 0.02
HOLDOUT_FRACTION = 0.20
_HOLDOUT_SEED = 42
# Below this many held-out days the RMSE contrast is too noisy to trust, so
# the gate abstains (mode stays global).
_MIN_TEST_DAYS = 5


def detrend_seasonality_gate(
    dayparts: pd.DataFrame,
    *,
    min_relative_improvement: float = DETREND_GATE_MIN_RELATIVE_IMPROVEMENT,
    holdout_fraction: float = HOLDOUT_FRACTION,
    min_samples: int = _SEASONAL_MIN_SAMPLES,
) -> dict[str, object]:
    """Compare global vs month_minute baseline skill on held-out days.

    ``dayparts`` is the raw frame from :func:`kairos.data.loaders.load_dayparts`
    (columns ``date``, ``timeband``, ``channel``, ``tvr``). Returns a dict with
    ``detrend_seasonality_recommended`` (bool), ``detrend_seasonality_holdout``
    (rmse_global, rmse_month_minute, n_test_minutes, n_test_days,
    relative_improvement, min_relative_improvement) and
    ``detrend_seasonality_reason`` (one line). Deterministic: fixed seed, no
    clock. Fails safely to "not recommended" when the data is too thin to
    split or to score.
    """
    # Guard before _dayparts_frame: that helper assumes at least one row (its
    # timeband split produces no columns on an empty frame), and a frame
    # missing the audience columns cannot be evaluated either way.
    needed = {"date", "timeband", "channel", "tvr"}
    empty_reason = "no daypart audience rows; seasonality cannot be evaluated"
    if dayparts.empty or not needed.issubset(dayparts.columns):
        return _result(
            recommended=False, rmse_global=None, rmse_month_minute=None,
            n_test_minutes=0, n_test_days=0,
            min_relative_improvement=min_relative_improvement,
            reason=empty_reason,
        )
    frame = _dayparts_frame(dayparts)
    if frame.empty:
        return _result(
            recommended=False, rmse_global=None, rmse_month_minute=None,
            n_test_minutes=0, n_test_days=0,
            min_relative_improvement=min_relative_improvement,
            reason=empty_reason,
        )

    day_series = frame["date"].dt.normalize()
    days = sorted(pd.unique(day_series))
    n_days = len(days)
    n_test_days = max(1, int(round(n_days * holdout_fraction)))
    if n_test_days < _MIN_TEST_DAYS:
        return _result(
            recommended=False, rmse_global=None, rmse_month_minute=None,
            n_test_minutes=0, n_test_days=n_test_days,
            min_relative_improvement=min_relative_improvement,
            reason=(
                f"too few held-out days ({n_test_days} < {_MIN_TEST_DAYS}) from a "
                f"{n_days}-day window; mode stays global"
            ),
        )

    rng = np.random.default_rng(_HOLDOUT_SEED)
    perm = rng.permutation(n_days)
    test_days = {days[i] for i in perm[:n_test_days]}
    test_mask = day_series.isin(test_days)
    train = frame[~test_mask]
    test = frame[test_mask]

    global_levels = _baseline_levels(train)
    seasonal_levels = _seasonal_baseline_levels(train, min_samples=min_samples)

    y_true: list[float] = []
    y_global: list[float] = []
    y_seasonal: list[float] = []
    months = test["date"].dt.month
    for channel, month, mod, tvr in zip(
        test["channel"].astype(str), months, test["mod"], test["tvr"]
    ):
        level = global_levels.get((channel, int(mod)))
        if level is None:
            continue  # unpredictable for both arms; skip like for like
        y_true.append(float(tvr))
        y_global.append(level)
        y_seasonal.append(seasonal_levels.get((channel, int(month), int(mod)), level))

    if not y_true:
        return _result(
            recommended=False, rmse_global=None, rmse_month_minute=None,
            n_test_minutes=0, n_test_days=n_test_days,
            min_relative_improvement=min_relative_improvement,
            reason="no scorable held-out minutes; mode stays global",
        )

    truth = np.asarray(y_true)
    rmse_global = float(np.sqrt(np.mean((truth - np.asarray(y_global)) ** 2)))
    rmse_month_minute = float(np.sqrt(np.mean((truth - np.asarray(y_seasonal)) ** 2)))
    pct = (
        100.0 * (rmse_global - rmse_month_minute) / rmse_global
        if rmse_global > 0 else 0.0
    )
    if rmse_global <= 0.0:
        recommended = False
        reason = "degenerate holdout (zero global-baseline RMSE); mode stays global"
    elif rmse_month_minute < rmse_global * (1.0 - min_relative_improvement):
        recommended = True
        reason = (
            f"month_minute RMSE ({rmse_month_minute:.5f}) beats the global RMSE "
            f"({rmse_global:.5f}) by {pct:.1f}% (threshold "
            f"{min_relative_improvement * 100:.0f}%); season-aware detrend is "
            "recommended, activation stays an owner decision"
        )
    else:
        recommended = False
        reason = (
            f"month_minute RMSE ({rmse_month_minute:.5f}) does not beat the global "
            f"RMSE ({rmse_global:.5f}) by the required "
            f"{min_relative_improvement * 100:.0f}% (actual improvement {pct:.1f}%); "
            "mode stays global"
        )
    logger.info("Detrend seasonality gate: %s", reason)
    return _result(
        recommended=recommended, rmse_global=rmse_global,
        rmse_month_minute=rmse_month_minute, n_test_minutes=len(y_true),
        n_test_days=n_test_days,
        min_relative_improvement=min_relative_improvement, reason=reason,
    )


def _result(
    *,
    recommended: bool,
    rmse_global,
    rmse_month_minute,
    n_test_minutes: int,
    n_test_days: int,
    min_relative_improvement: float,
    reason: str,
) -> dict[str, object]:
    relative = (
        (rmse_global - rmse_month_minute) / rmse_global
        if rmse_global and rmse_month_minute is not None and rmse_global > 0
        else None
    )
    return {
        "detrend_seasonality_recommended": recommended,
        "detrend_seasonality_holdout": {
            "rmse_global": rmse_global,
            "rmse_month_minute": rmse_month_minute,
            "n_test_minutes": n_test_minutes,
            "n_test_days": n_test_days,
            "relative_improvement": relative,
            "min_relative_improvement": min_relative_improvement,
        },
        "detrend_seasonality_reason": reason,
    }
