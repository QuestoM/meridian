"""Automatic held-out gate for the counter-programming retention covariate.

Same discipline as :mod:`kairos.model.series_gate`: the covariate earns its way
into the model ONLY by improving out-of-sample prediction of the per-break
retention effect on the real data. Otherwise the machinery ships OFF with the
measured verdict recorded, and the plan is exactly what it was without it.

Gate design
-----------
* WITHOUT prediction: the training-set mean log_effect of the break's genre
  cell (the same baseline the series gate uses).
* WITH prediction: the same cell mean computed on COMPETITION-ADJUSTED training
  effects, plus the forward betas' contribution for the test break's own
  competitor context: ``cell_mean_adj + sum(beta_f * (x_f - reference_f))``.
  The betas come from :func:`kairos.model.competitor_model.fit_competitor_betas`
  fitted on the TRAINING split only (within-cell OLS; the training-only rival
  co-breaking feature is a fit-time control and never predicts).
* The covariate is recommended ON iff its improvement over WITHOUT exceeds
  :data:`COUNTERPROGRAMMING_MIN_RELATIVE_IMPROVEMENT` (2 percent, the same
  margin the series gate uses).

Gate statistic (fold-averaged)
------------------------------
As measured in docs/model-validation/uncertainty-calibration.md section 4, a
single 80/20 split leaves the gate statistic with avoidable split noise (the
sign of this covariate's tiny improvement is pure seed noise), so the default
statistic is the MEAN relative improvement over :data:`GATE_FOLDS` temporal
folds (contiguous blocks in break_start order), same 2 percent threshold.
Where time order is unavailable it averages :data:`GATE_FOLDS` seeded 80/20
splits instead. ``statistic_method="single_split"`` keeps the legacy split,
value for value. Verified on the real reference month before fold averaging
became the default: single split -0.11 percent, temporal fold mean -0.06
percent (fold sd 0.44pp), both far below the +2 percent bar, so the shipped
verdict (covariate OFF) is unchanged.

The decision, both RMSEs, the statistic method, the split size and the fitted
betas are returned so the coefficients JSON metadata can carry the full audit
trail.
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

# Fold-averaged statistic (see module docstring). Defaults mirror the series
# gate: fold_mean became the default only after the real-data verdict was
# proven identical to the legacy single split.
GATE_FOLDS = 5
STATISTIC_FOLD_MEAN = "fold_mean"
STATISTIC_SINGLE_SPLIT = "single_split"
COUNTERPROGRAMMING_GATE_STATISTIC_METHOD = STATISTIC_FOLD_MEAN


def counterprogramming_holdout_gate(
    effects: pd.DataFrame,
    *,
    min_relative_improvement: float = COUNTERPROGRAMMING_MIN_RELATIVE_IMPROVEMENT,
    holdout_fraction: float = HOLDOUT_FRACTION,
    statistic_method: str | None = None,
) -> dict[str, object]:
    """Evaluate WITH vs WITHOUT the counter-programming covariate out of sample.

    ``effects`` must carry ``channel_name``, ``log_effect`` and the
    :data:`~kairos.model.competitor_features.EXTENDED_ALL_FEATURES` columns
    (from :func:`kairos.model.competitor_model.measure_effects_with_competitors`).

    Returns a dict with ``counterprogramming_active`` (bool: recommend ON),
    ``counterprogramming_holdout`` (rmse_without, rmse_with, n_test, the
    relative improvement, gate_statistic_method, folds, fold_sd),
    ``counterprogramming_reason`` (one line) and ``counterprogramming_betas``
    (feature -> beta summary; fitted on the training split for the legacy
    single split, on the full frame for the fold statistic, display only
    either way). Under the fold statistic the RMSEs are fold means and
    ``n_test`` counts every tested break across folds. Fails safely OFF when
    the data is too thin to split or the betas cannot be fitted.
    """
    method = (
        COUNTERPROGRAMMING_GATE_STATISTIC_METHOD
        if statistic_method is None
        else statistic_method
    )
    if method not in (STATISTIC_FOLD_MEAN, STATISTIC_SINGLE_SPLIT):
        raise ValueError(
            f"unknown statistic_method {method!r}; expected "
            f"{STATISTIC_FOLD_MEAN!r} or {STATISTIC_SINGLE_SPLIT!r}"
        )

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

    keep = ["channel_name", "log_effect", *EXTENDED_ALL_FEATURES]
    if "break_start" in effects.columns:
        keep.append("break_start")
    work = effects[keep].dropna(subset=["log_effect", *EXTENDED_ALL_FEATURES]).reset_index(drop=True)
    n_total = len(work)
    n_test = max(1, int(round(n_total * holdout_fraction)))
    if n_test < _MIN_TEST_BREAKS:
        return _result(
            active=False, rmse_without=None, rmse_with=None, n_test=n_test, betas={},
            min_relative_improvement=min_relative_improvement, method=method,
            reason=(
                f"too few test breaks ({n_test} < {_MIN_TEST_BREAKS}) after the holdout split; "
                "covariate left off"
            ),
        )

    if method == STATISTIC_SINGLE_SPLIT:
        return _single_split_gate(work, n_total, n_test, min_relative_improvement)
    return _fold_mean_gate(work, n_total, n_test, min_relative_improvement)


def _split_rmses(
    train: pd.DataFrame, test: pd.DataFrame,
) -> tuple[float, float | None, dict]:
    """(rmse_without, rmse_with, betas) for one train/test split.

    ``rmse_with`` is None when no forward betas could be fitted on the
    training split, so the caller can fail the gate honestly.
    """
    # WITHOUT: plain genre-cell means from the raw training effects.
    cell_means = train.groupby("channel_name")["log_effect"].mean().to_dict()
    global_mean = float(train["log_effect"].mean())
    y_true = test["log_effect"].to_numpy()
    y_without = np.array([
        cell_means.get(str(r.channel_name), global_mean)
        for r in test.itertuples(index=False)
    ])
    rmse_without = _rmse(y_true, y_without)

    # WITH: betas fit on the training split; cell means on the adjusted
    # training effects; each test prediction adds back the forward
    # contribution of ITS OWN competitor context.
    betas = fit_competitor_betas(train, feature_names=EXTENDED_ALL_FEATURES)
    forward = {name: cb for name, cb in betas.items() if cb.role == "forward"}
    if not forward:
        return rmse_without, None, betas
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
    return rmse_without, _rmse(y_true, y_with), betas


def _single_split_gate(
    work: pd.DataFrame, n_total: int, n_test: int, min_relative_improvement: float,
) -> dict[str, object]:
    """The legacy deterministic 80/20 split, computed exactly as before."""
    rng = np.random.default_rng(_HOLDOUT_SEED)
    idx = rng.permutation(n_total)
    test = work.iloc[idx[:n_test]]
    train = work.iloc[idx[n_test:]]

    rmse_without, rmse_with, betas = _split_rmses(train, test)
    if rmse_with is None:
        return _result(
            active=False, rmse_without=rmse_without, rmse_with=None,
            n_test=n_test, betas=_beta_summary(betas),
            min_relative_improvement=min_relative_improvement,
            method=STATISTIC_SINGLE_SPLIT,
            reason="forward competitor betas could not be fitted on the training split; covariate left off",
        )

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
        method=STATISTIC_SINGLE_SPLIT,
    )


def _fold_pairs(work: pd.DataFrame, n_total: int, n_test: int) -> tuple[str, list]:
    """Temporal fold pairs when break_start is usable, else seeded splits."""
    if (
        "break_start" in work.columns
        and n_total >= GATE_FOLDS
        and pd.to_datetime(work["break_start"], errors="coerce").notna().all()
    ):
        order = np.argsort(
            pd.to_datetime(work["break_start"]).to_numpy(), kind="stable"
        )
        pairs = []
        for block in np.array_split(order, GATE_FOLDS):
            mask = np.zeros(n_total, dtype=bool)
            mask[block] = True
            pairs.append((np.flatnonzero(~mask), block))
        return "fold_mean_temporal", pairs
    pairs = []
    for k in range(GATE_FOLDS):
        rng = np.random.default_rng(_HOLDOUT_SEED + k)
        idx = rng.permutation(n_total)
        pairs.append((idx[n_test:], idx[:n_test]))
    return "seed_mean", pairs


def _fold_mean_gate(
    work: pd.DataFrame, n_total: int, n_test_target: int,
    min_relative_improvement: float,
) -> dict[str, object]:
    """The fold-averaged statistic: mean relative improvement over the folds."""
    method, pairs = _fold_pairs(work, n_total, n_test_target)
    # Display betas for the audit trail: one deterministic fit on the full
    # frame (the per-fold training fits drive the statistic itself).
    betas_all = fit_competitor_betas(work, feature_names=EXTENDED_ALL_FEATURES)

    improvements: list[float] = []
    withouts: list[float] = []
    withs: list[float] = []
    n_test = 0
    for train_pos, test_pos in pairs:
        train = work.iloc[train_pos]
        test = work.iloc[test_pos]
        rmse_without, rmse_with, _betas = _split_rmses(train, test)
        if rmse_with is None:
            return _result(
                active=False, rmse_without=rmse_without, rmse_with=None,
                n_test=n_test + len(test), betas=_beta_summary(betas_all),
                min_relative_improvement=min_relative_improvement,
                method=method, folds=len(pairs),
                reason=(
                    "forward competitor betas could not be fitted on at least one "
                    "training fold; covariate left off"
                ),
            )
        if rmse_without <= 0.0:
            return _result(
                active=False, rmse_without=rmse_without, rmse_with=rmse_with,
                n_test=n_test + len(test), betas=_beta_summary(betas_all),
                min_relative_improvement=min_relative_improvement,
                method=method, folds=len(pairs),
                reason="degenerate holdout (zero baseline RMSE on a fold); covariate left off",
            )
        withouts.append(rmse_without)
        withs.append(rmse_with)
        improvements.append((rmse_without - rmse_with) / rmse_without)
        n_test += len(test)

    statistic = float(np.mean(improvements))
    fold_sd = float(np.std(improvements, ddof=1)) if len(improvements) > 1 else None
    without_mean = float(np.mean(withouts))
    with_mean = float(np.mean(withs))
    fold_kind = "temporal folds" if method == "fold_mean_temporal" else "seeded splits"
    sd_text = f"{100.0 * fold_sd:.1f}pp" if fold_sd is not None else "n/a"

    active = statistic > min_relative_improvement
    if active:
        reason = (
            f"counter-programming RMSE (fold mean {with_mean:.5f}) beats the no-covariate "
            f"RMSE (fold mean {without_mean:.5f}) by {100.0 * statistic:.1f}% on average over "
            f"{len(pairs)} {fold_kind} (fold sd {sd_text}, threshold "
            f"{min_relative_improvement*100:.0f}%); covariate recommended ON"
        )
    else:
        reason = (
            f"counter-programming RMSE (fold mean {with_mean:.5f}) does not beat the "
            f"no-covariate RMSE (fold mean {without_mean:.5f}) by the required "
            f"{min_relative_improvement*100:.0f}% (mean improvement {100.0 * statistic:.1f}% "
            f"over {len(pairs)} {fold_kind}, fold sd {sd_text}); covariate left off"
        )
    logger.info("Counter-programming gate: %s", reason)
    return _result(
        active=active, rmse_without=without_mean, rmse_with=with_mean,
        n_test=n_test, betas=_beta_summary(betas_all),
        min_relative_improvement=min_relative_improvement, reason=reason,
        method=method, folds=len(pairs), fold_sd=fold_sd,
        relative_improvement=statistic,
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


_UNSET = object()


def _result(
    *, active: bool, rmse_without, rmse_with, n_test: int, betas, reason: str,
    min_relative_improvement: float = COUNTERPROGRAMMING_MIN_RELATIVE_IMPROVEMENT,
    method: str | None = None, folds: int | None = None,
    fold_sd: float | None = None, relative_improvement=_UNSET,
) -> dict[str, object]:
    # relative_improvement is the gate's decision statistic (positive = WITH is
    # better) and min_relative_improvement its pass threshold, carried
    # alongside the raw RMSEs so an artifact reader can audit pass/fail
    # without recomputing. For the single split it derives from the two RMSEs;
    # the fold path passes its mean-over-folds statistic explicitly.
    if relative_improvement is _UNSET:
        relative_improvement = (
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
            "relative_improvement": relative_improvement,
            "min_relative_improvement": min_relative_improvement,
            "gate_statistic_method": method,
            "folds": folds,
            "fold_sd": fold_sd,
        },
        "counterprogramming_betas": betas,
        "counterprogramming_reason": reason,
        "forward_features": list(EXTENDED_FORWARD_FEATURES),
    }
