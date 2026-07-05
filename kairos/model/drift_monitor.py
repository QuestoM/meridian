"""Weekly level-drift monitor for the measured retention base.

The pooled retention coefficients assume the audience level they were measured
on is stationary week to week. The uncertainty-calibration review
(docs/model-validation/uncertainty-calibration.md, section 1 finding 4) showed
it is not: on the reference month the weekly grand mean of the measured log
effect moved +0.0202 between the newest week and the base it was pooled with,
about twice the pooled coefficient's 95 percent half-width (~0.0094 in log
space). Nonstationarity of the LEVEL, not cell ordering, is therefore the
binding risk the credible intervals cannot see, and the review's ranked fix 4
asks for a weekly monitor with a recompute trigger.

This module is that monitor. :func:`level_drift` takes the same measured
break-effects frame the coefficient rebuild already produces (one row per
break with ``log_effect`` and ``break_start``, see
:func:`kairos.model.measure.break_effects`) and returns a self-describing
block for the artifact metadata:

  * ``weekly_levels``   per-week mean log effect of the measurement base,
                        7-day blocks anchored at the first measured break
                        (the review's convention).
  * ``drift_per_week``  the week-over-week level change: the mean log effect
                        of the last 7 measured days minus the mean of the
                        preceding base, with ``drift_se`` (two-sample SE).
                        This is the review's headline statistic.
  * ``slope_per_week``  the supporting trend view: break-level OLS slope of
                        the log effect on the week index, with ``slope_se``.
  * ``binding``         True when ``|drift_per_week|`` exceeds
                        ``BINDING_HALF_WIDTH_MULTIPLE`` times the pooled
                        coefficient's 95 percent half-width, i.e. the level
                        moved further in one week than the published
                        uncertainty band can absorb. ``criterion`` spells the
                        rule out so the number cannot be misread.

Everything is measured from the data passed in; nothing is fabricated. With
under :data:`MIN_WINDOW_DAYS` days of measured breaks the drift statistic
cannot distinguish drift from noise across a single block boundary, so the
block reports ``status`` ``insufficient_data`` with the drift fields ``None``
(an honest absent state, never a guessed verdict). Pure pandas and numpy, no
clock, no randomness: the same frame always yields the same block.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from kairos.model.measure import _cell_stats, _pooled_within_variance

# Days in one monitoring block; the trailing window compared against the base.
WEEK_DAYS = 7
# Minimum measured span (first to last break, inclusive) for a drift verdict:
# two full weeks, so the trailing window and its base are both real weeks.
MIN_WINDOW_DAYS = 14
# The binding rule: |drift_per_week| must exceed this multiple of the pooled
# coefficient's 95 percent half-width. The review's criterion is 2x: a level
# move of about twice the band means nonstationarity binds before cell order.
BINDING_HALF_WIDTH_MULTIPLE = 2.0

_CRITERION = (
    "binding when |drift_per_week| > {mult:g} x the pooled coefficient's 95 percent "
    "half-width in log-effect space; drift_per_week is the mean log effect of the "
    "last {days} measured days minus the mean of the preceding base"
).format(mult=BINDING_HALF_WIDTH_MULTIPLE, days=WEEK_DAYS)


def _absent(reason: str, *, n_breaks: int, window_days: int,
            weekly_levels: list[dict[str, Any]]) -> dict[str, Any]:
    """The honest absent state: measurement impossible, no verdict invented."""
    return {
        "status": "insufficient_data",
        "reason": reason,
        "n_breaks": n_breaks,
        "n_weeks": len(weekly_levels),
        "window_days": window_days,
        "weekly_levels": weekly_levels,
        "drift_per_week": None,
        "drift_se": None,
        "slope_per_week": None,
        "slope_se": None,
        "pooled_half_width_95": None,
        "binding_threshold": None,
        "binding": None,
        "criterion": _CRITERION,
    }


def _weekly_levels(work: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-week mean log effect, 7-day blocks from the first measured break."""
    levels: list[dict[str, Any]] = []
    for week, group in work.groupby("week"):
        values = group["log_effect"].to_numpy(dtype=float)
        se = (
            float(np.std(values, ddof=1) / np.sqrt(len(values)))
            if len(values) > 1
            else None
        )
        levels.append(
            {
                "week": int(week),
                "n": int(len(values)),
                "mean_log_effect": float(np.mean(values)),
                "se": se,
            }
        )
    return levels


def _pooled_half_width(work: pd.DataFrame) -> Optional[float]:
    """The pooled coefficient's 95 percent half-width in log-effect space.

    ``1.96 * sqrt(s_p^2 / N)``: the half-width of the precision-weighted grand
    mean the empirical-Bayes pooling publishes (the pooled constant that the
    decision-robustness review measured as doing all the decision work). The
    within-cell variance comes from the same :mod:`kairos.model.measure`
    helpers the pooling itself uses; when the frame carries no cell column or
    the pooled variance is degenerate, the plain sample variance of the log
    effects stands in, and ``None`` means it cannot be computed at all.
    """
    n = len(work)
    if n < 2:
        return None
    s2 = float("nan")
    if "channel_name" in work.columns:
        s2 = _pooled_within_variance(_cell_stats(work))
    if not np.isfinite(s2) or s2 <= 0:
        s2 = float(np.var(work["log_effect"].to_numpy(dtype=float), ddof=1))
    if not np.isfinite(s2) or s2 <= 0:
        return None
    return float(1.96 * np.sqrt(s2 / n))


def _ols_slope(work: pd.DataFrame) -> tuple[Optional[float], Optional[float]]:
    """Break-level OLS slope of log effect on the week index, with its SE."""
    x = work["week"].to_numpy(dtype=float)
    y = work["log_effect"].to_numpy(dtype=float)
    sxx = float(np.sum((x - x.mean()) ** 2))
    if len(x) < 3 or sxx <= 0:
        return None, None
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / sxx)
    residuals = y - (y.mean() + slope * (x - x.mean()))
    s2 = float(np.sum(residuals**2) / (len(y) - 2))
    se = float(np.sqrt(s2 / sxx)) if s2 > 0 else None
    return slope, se


def level_drift(effects: pd.DataFrame) -> dict[str, Any]:
    """Measure the weekly level drift of the coefficient measurement base.

    ``effects`` is the frame :func:`kairos.model.measure.break_effects`
    returns (``log_effect`` and ``break_start`` are required; ``channel_name``
    sharpens the half-width when present). Returns the metadata block
    described in the module docstring, with ``status`` ``measured`` or the
    honest ``insufficient_data``.
    """
    if (
        effects is None
        or effects.empty
        or not {"log_effect", "break_start"}.issubset(effects.columns)
    ):
        return _absent(
            "no measured break effects with timestamps; level drift cannot be measured",
            n_breaks=0, window_days=0, weekly_levels=[],
        )

    work = effects.dropna(subset=["log_effect", "break_start"]).copy()
    if work.empty:
        return _absent(
            "no measured break effects with timestamps; level drift cannot be measured",
            n_breaks=0, window_days=0, weekly_levels=[],
        )
    work["break_start"] = pd.to_datetime(work["break_start"])
    first = work["break_start"].min().normalize()
    last = work["break_start"].max().normalize()
    window_days = int((last - first).days) + 1
    work["week"] = ((work["break_start"] - first).dt.days // WEEK_DAYS) + 1
    weekly_levels = _weekly_levels(work)
    n_breaks = int(len(work))

    if window_days < MIN_WINDOW_DAYS:
        return _absent(
            (
                f"only {window_days} measured day(s) on record; at least "
                f"{MIN_WINDOW_DAYS} days (two weeks) are needed for a drift verdict"
            ),
            n_breaks=n_breaks, window_days=window_days, weekly_levels=weekly_levels,
        )

    # The review's drift statistic: the newest measured week against the base
    # the coefficients were pooled with. The window is anchored at the END of
    # the data so the statistic always describes the most recent level move.
    cutoff = last - pd.Timedelta(days=WEEK_DAYS - 1)
    recent = work.loc[work["break_start"] >= cutoff, "log_effect"].to_numpy(dtype=float)
    base = work.loc[work["break_start"] < cutoff, "log_effect"].to_numpy(dtype=float)
    if len(recent) == 0 or len(base) == 0:
        return _absent(
            "the trailing week or its base holds no measured breaks; drift cannot be measured",
            n_breaks=n_breaks, window_days=window_days, weekly_levels=weekly_levels,
        )
    drift = float(np.mean(recent) - np.mean(base))
    drift_se = (
        float(np.sqrt(np.var(recent, ddof=1) / len(recent) + np.var(base, ddof=1) / len(base)))
        if len(recent) > 1 and len(base) > 1
        else None
    )
    slope, slope_se = _ols_slope(work)

    half_width = _pooled_half_width(work)
    threshold = (
        float(BINDING_HALF_WIDTH_MULTIPLE * half_width) if half_width is not None else None
    )
    binding = bool(abs(drift) > threshold) if threshold is not None else None

    return {
        "status": "measured",
        "reason": None,
        "n_breaks": n_breaks,
        "n_weeks": len(weekly_levels),
        "window_days": window_days,
        "weekly_levels": weekly_levels,
        "drift_per_week": drift,
        "drift_se": drift_se,
        "slope_per_week": slope,
        "slope_se": slope_se,
        "pooled_half_width_95": half_width,
        "binding_threshold": threshold,
        "binding": binding,
        "criterion": _CRITERION,
    }
