"""Scoring a set of forecasts against what was actually measured.

The walk (:mod:`kairos.model.forecast_backtest`) decides WHICH rows a model is
allowed to be scored on; this module decides what the score MEANS. Kept apart
because they fail differently: a leak is a bug in the walk, and a flattering
statistic is a bug here.

Four numbers, and what each is for.

``mae`` and ``rmse`` are in rating points, the unit the plan prices in. ``bias``
is the MEAN SIGNED error, kept separate because a forecast that is 0.4 points
high on everything is a different failure from one that is 0.4 points off in both
directions, and only the first is fixable by a constant. ``mape`` is reported
over observations at or above :data:`MAPE_TVR_FLOOR` only -- a percentage error
against a measured rating of 0.0 is unbounded, so those rows are excluded,
COUNTED, and named. ``interval_coverage`` is the honesty check on the published
band: at the stated level the observed rating should fall inside it that often,
and coverage far below the level means the band is too narrow no matter how good
the point forecast is.

**Two objectives, both reported.** The audience model's gates admitted each
family on held-out RMSE IN LOG SPACE. The plan prices in arithmetic rating
points. Those objectives can disagree, and on the real one-month window they do,
so every metrics block carries the log-space pair beside the points pair and
:func:`verdict_for` states which way each went. Reporting only the objective the
model was tuned on would be picking the scoreboard after the game.

**The pre-model baseline is scored on the same rows.** The plain historical mean
is what this product priced on before the model existed. A model error with
nothing beside it is a number with nothing to be better than.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from kairos.model.audience_model import TVR_FLOOR

# Observations below this measured rating are excluded from MAPE (a percentage
# error against a measured 0.0 is unbounded), counted, and named on the payload.
MAPE_TVR_FLOOR = 0.05


def unavailable(reason: str, **extra: Any) -> dict[str, Any]:
    """The frozen honest-absence shape the measurement returns everywhere."""
    return {"available": False, "reason": reason, **extra}


def metrics_for(
    observed: np.ndarray, predicted: np.ndarray, historical: np.ndarray,
    low: np.ndarray, high: np.ndarray, level: float,
) -> dict[str, Any]:
    """Point error, bias, MAPE with its exclusions, and interval coverage."""
    n = int(len(observed))
    if n == 0:
        return unavailable("no scored observations in this cell", n=0)
    error = predicted - observed
    hist_error = historical - observed
    scorable = observed >= MAPE_TVR_FLOOR
    banded = np.isfinite(low) & np.isfinite(high)
    inside = banded & (observed >= low) & (observed <= high)
    log_observed = np.log(np.maximum(observed, TVR_FLOOR))
    log_error = np.log(np.maximum(predicted, TVR_FLOOR)) - log_observed
    log_hist_error = np.log(np.maximum(historical, TVR_FLOOR)) - log_observed
    out: dict[str, Any] = {
        "available": True,
        "n": n,
        "mae": round(float(np.mean(np.abs(error))), 4),
        "rmse": round(float(np.sqrt(np.mean(error**2))), 4),
        "bias": round(float(np.mean(error)), 4),
        "mean_observed": round(float(np.mean(observed)), 4),
        "historical_mae": round(float(np.mean(np.abs(hist_error))), 4),
        "historical_rmse": round(float(np.sqrt(np.mean(hist_error**2))), 4),
        "historical_bias": round(float(np.mean(hist_error)), 4),
        "log_rmse": round(float(np.sqrt(np.mean(log_error**2))), 4),
        "historical_log_rmse": round(float(np.sqrt(np.mean(log_hist_error**2))), 4),
        "mape": None,
        "mape_n": int(scorable.sum()),
        "mape_excluded_n": int(n - scorable.sum()),
        "mape_excluded_reason": (
            f"observations with a measured rating below {MAPE_TVR_FLOOR} points are "
            "excluded from MAPE because a percentage error against them is unbounded"
        ),
        "interval_level": round(float(level), 4),
        "interval_n": int(banded.sum()),
        "interval_missing_n": int(n - banded.sum()),
        "interval_coverage": None,
    }
    if int(scorable.sum()):
        out["mape"] = round(float(
            100.0 * np.mean(np.abs(error[scorable]) / observed[scorable])
        ), 3)
    if int(banded.sum()):
        out["interval_coverage"] = round(float(inside.sum() / banded.sum()), 4)
        out["interval_mean_width"] = round(float(np.mean(high[banded] - low[banded])), 4)
    return out


def breakdown_for(
    keys: np.ndarray, observed: np.ndarray, predicted: np.ndarray,
    historical: np.ndarray, low: np.ndarray, high: np.ndarray, level: float,
) -> dict[str, Any]:
    """Per-cell metrics, ``n`` always reported, thin cells never hidden."""
    out: dict[str, Any] = {}
    for key in sorted({str(k) for k in keys}):
        mask = keys == key
        out[key] = metrics_for(
            observed[mask], predicted[mask], historical[mask],
            low[mask], high[mask], level,
        )
    return out


def verdict_for(overall: dict[str, Any]) -> dict[str, Any]:
    """Where the model beats the pre-model historical mean, and where it does not.

    Both objectives, side by side, because they can disagree. A log-space win
    does not carry over to points: ``exp`` of a mean of logs estimates the
    geometric centre, which sits below the arithmetic mean of a right-skewed
    rating distribution, so a model fitted and gated in logs can be honest in
    logs and systematically low in points. A negative ``bias`` beside a log-space
    win is the signature of exactly that, and this block names it.
    """
    if not overall.get("available"):
        return {"available": False, "reason": overall.get("reason", "nothing scored")}
    log_gain = overall["historical_log_rmse"] - overall["log_rmse"]
    points_gain = overall["historical_mae"] - overall["mae"]
    beats_log = log_gain > 0
    beats_points = points_gain > 0
    return {
        "available": True,
        "beats_historical_in_log_space": bool(beats_log),
        "beats_historical_in_points": bool(beats_points),
        "log_rmse_gain": round(float(log_gain), 4),
        "log_rmse_gain_pct": round(
            100.0 * log_gain / overall["historical_log_rmse"], 2
        ) if overall["historical_log_rmse"] else None,
        "points_mae_gain": round(float(points_gain), 4),
        "points_mae_gain_pct": round(
            100.0 * points_gain / overall["historical_mae"], 2
        ) if overall["historical_mae"] else None,
        "interval_is_conservative": bool(
            overall.get("interval_coverage") is not None
            and overall["interval_coverage"] > overall["interval_level"]
        ),
        "headline_en": (
            ("the model beats" if beats_log else "the model does not beat")
            + " the pre-model historical mean on the log-space objective its gates "
            + "were measured on, and "
            + ("beats" if beats_points else "does NOT beat")
            + " it in arithmetic rating points, the unit the plan prices in"
        ),
        "headline_he": (
            ("המודל מנצח את" if beats_log else "המודל אינו מנצח את")
            + " הממוצע ההיסטורי במרחב הלוג, המדד שעל פיו נמדדו השערים, ו"
            + ("מנצח" if beats_points else "אינו מנצח")
            + " אותו בנקודות רייטינג, היחידה שבה התוכנית מתומחרת"
        ),
        "mechanism_note_en": (
            "a negative bias beside a log-space win is the retransformation "
            "shortfall: the exponential of a log-space level estimates the "
            "geometric centre, which sits below the arithmetic mean of a "
            "right-skewed rating distribution"
        ) if (beats_log and not beats_points and overall["bias"] < 0) else None,
    }
