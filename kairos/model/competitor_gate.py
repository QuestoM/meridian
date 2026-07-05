"""Automatic held-out gate for the counter-programming retention covariate.

Same discipline as :mod:`kairos.model.series_gate`: the covariate earns its way
into the model ONLY by improving out-of-sample prediction of the per-break
retention effect on the real data. Otherwise the machinery ships OFF with the
measured verdict recorded, and the plan is exactly what it was without it.

Gate design
-----------
* A deterministic random 20 percent of measured breaks are held out.
* WITHOUT prediction: the training-set mean log_effect of the break's genre
  cell (the same baseline the series gate uses).
* WITH prediction: the same cell mean computed on COMPETITION-ADJUSTED training
  effects, plus the forward betas' contribution for the test break's own
  competitor context: ``cell_mean_adj + sum(beta_f * (x_f - reference_f))``.
  The betas come from :func:`kairos.model.competitor_model.fit_competitor_betas`
  fitted on the TRAINING split only (within-cell OLS; the training-only rival
  co-breaking feature is a fit-time control and never predicts).
* The covariate is recommended ON iff its RMSE beats the WITHOUT RMSE by at
  least :data:`COUNTERPROGRAMMING_MIN_RELATIVE_IMPROVEMENT` (2 percent, the
  same margin the series gate uses).

The decision, both RMSEs, the split size and the fitted betas are returned so
the coefficients JSON metadata can carry the full audit trail.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from kairos.model.competitor_features import (
    EXTENDED_ALL_FEATURES,
    EXTENDED_FORWARD_FEATURES,
)
from kairos.model.competitor_model import (
    adjust_effects_for_forward_competition,
    fit_competitor_betas,
)

logger = logging.getLogger(__name__)

# Relative RMSE improvement required to recommend the covariate ON. Matches
# SERIES_GATE_MIN_RELATIVE_IMPROVEMENT so every optional layer clears one bar.
COUNTERPROGRAMMING_MIN_RELATIVE_IMPROVEMENT = 0.02
HOLDOUT_FRACTION = 0.20
_HOLDOUT_SEED = 42
_MIN_TEST_BREAKS = 10


def counterprogramming_holdout_gate(
    effects: pd.DataFrame,
    *,
    min_relative_improvement: float = COUNTERPROGRAMMING_MIN_RELATIVE_IMPROVEMENT,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> dict[str, object]:
    """Evaluate WITH vs WITHOUT the counter-programming covariate out of sample.

    ``effects`` must carry ``channel_name``, ``log_effect`` and the
    :data:`~kairos.model.competitor_features.EXTENDED_ALL_FEATURES` columns
    (from :func:`kairos.model.competitor_model.measure_effects_with_competitors`).

    Returns a dict with ``counterprogramming_active`` (bool: recommend ON),
    ``counterprogramming_holdout`` (rmse_without, rmse_with, n_test),
    ``counterprogramming_reason`` (one line) and ``counterprogramming_betas``
    (feature -> beta summary from the training split). Fails safely OFF when
    the data is too thin to split or the betas cannot be fitted.
    """
    missing = [c for c in ("channel_name", "log_effect", *EXTENDED_ALL_FEATURES) if c not in effects.columns]
    if effects.empty or missing:
        return _result(
            active=False, rmse_without=None, rmse_with=None, n_test=0, betas={},
            min_relative_improvement=min_relative_improvement,
            reason=(
                "no measured effects with competitor features available"
                if effects.empty
                else f"missing columns {missing}; covariate cannot be evaluated"
            ),
        )

    work = effects.dropna(subset=["log_effect", *EXTENDED_ALL_FEATURES]).reset_index(drop=True)
    n_total = len(work)
    n_test = max(1, int(round(n_total * holdout_fraction)))
    if n_test < _MIN_TEST_BREAKS:
        return _result(
            active=False, rmse_without=None, rmse_with=None, n_test=n_test, betas={},
            min_relative_improvement=min_relative_improvement,
            reason=(
                f"too few test breaks ({n_test} < {_MIN_TEST_BREAKS}) after the holdout split; "
                "covariate left off"
            ),
        )

    rng = np.random.default_rng(_HOLDOUT_SEED)
    idx = rng.permutation(n_total)
    test = work.iloc[idx[:n_test]]
    train = work.iloc[idx[n_test:]]

    # WITHOUT: plain genre-cell means from the raw training effects.
    cell_means = train.groupby("channel_name")["log_effect"].mean().to_dict()
    global_mean = float(train["log_effect"].mean())
    y_true = test["log_effect"].to_numpy()
    y_without = np.array([
        cell_means.get(str(r.channel_name), global_mean)
        for r in test.itertuples(index=False)
    ])

    # WITH: betas fit on the training split; cell means on the adjusted
    # training effects; each test prediction adds back the forward
    # contribution of ITS OWN competitor context.
    betas = fit_competitor_betas(train, feature_names=EXTENDED_ALL_FEATURES)
    forward = {name: cb for name, cb in betas.items() if cb.role == "forward"}
    if not forward:
        return _result(
            active=False, rmse_without=_rmse(y_true, y_without), rmse_with=None,
            n_test=n_test, betas=_beta_summary(betas),
            min_relative_improvement=min_relative_improvement,
            reason="forward competitor betas could not be fitted on the training split; covariate left off",
        )
    adjusted_train = adjust_effects_for_forward_competition(train, betas)
    cell_means_adj = adjusted_train.groupby("channel_name")["log_effect"].mean().to_dict()
    global_mean_adj = float(adjusted_train["log_effect"].mean())
    contribution = np.zeros(len(test), dtype=float)
    for name, cb in forward.items():
        contribution += cb.beta * (test[name].to_numpy(dtype=float) - cb.reference)
    y_with = np.array([
        cell_means_adj.get(str(r.channel_name), global_mean_adj)
        for r in test.itertuples(index=False)
    ]) + contribution

    rmse_without = _rmse(y_true, y_without)
    rmse_with = _rmse(y_true, y_with)
    threshold = rmse_without * (1.0 - min_relative_improvement)
    pct = 100.0 * (rmse_without - rmse_with) / rmse_without if rmse_without > 0 else 0.0
    if rmse_without <= 0.0:
        active = False
        reason = "degenerate holdout (zero baseline RMSE); covariate left off"
    elif rmse_with < threshold:
        active = True
        reason = (
            f"counter-programming RMSE ({rmse_with:.5f}) beats the no-covariate RMSE "
            f"({rmse_without:.5f}) by {pct:.1f}% (threshold {min_relative_improvement*100:.0f}%); "
            "covariate recommended ON"
        )
    else:
        active = False
        reason = (
            f"counter-programming RMSE ({rmse_with:.5f}) does not beat the no-covariate RMSE "
            f"({rmse_without:.5f}) by the required {min_relative_improvement*100:.0f}% "
            f"(actual improvement {pct:.1f}%); covariate left off"
        )
    logger.info("Counter-programming gate: %s", reason)
    return _result(
        active=active, rmse_without=rmse_without, rmse_with=rmse_with,
        n_test=n_test, betas=_beta_summary(betas),
        min_relative_improvement=min_relative_improvement, reason=reason,
    )


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _beta_summary(betas) -> dict[str, dict[str, float | str]]:
    return {
        name: {
            "beta": cb.beta, "se": cb.se, "ci_low": cb.ci_low, "ci_high": cb.ci_high,
            "role": cb.role, "reference": cb.reference,
        }
        for name, cb in betas.items()
    }


def _result(
    *, active: bool, rmse_without, rmse_with, n_test: int, betas, reason: str,
    min_relative_improvement: float = COUNTERPROGRAMMING_MIN_RELATIVE_IMPROVEMENT,
) -> dict[str, object]:
    # relative_improvement is the gate's delta (positive = WITH is better) and
    # min_relative_improvement its pass threshold, carried alongside the raw
    # RMSEs so an artifact reader can audit pass/fail without recomputing.
    relative = (
        (rmse_without - rmse_with) / rmse_without
        if rmse_without and rmse_with is not None and rmse_without > 0
        else None
    )
    return {
        "counterprogramming_active": active,
        "counterprogramming_holdout": {
            "rmse_without": rmse_without,
            "rmse_with": rmse_with,
            "n_test": n_test,
            "relative_improvement": relative,
            "min_relative_improvement": min_relative_improvement,
        },
        "counterprogramming_betas": betas,
        "counterprogramming_reason": reason,
        "forward_features": list(EXTENDED_FORWARD_FEATURES),
    }
