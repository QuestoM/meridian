"""Automatic held-out gate for the series retention layer.

The series layer (genre -> series -> episode pooling) adds per-title
coefficients on top of the genre-cell coefficients. It is only useful when the
data genuinely supports per-title distinctions -- that is, when title-level
predictions outperform genre-level predictions on held-out breaks. When the
data is thin (e.g., a single month) or titles are too sparse, the gate fails
and only the genre layer is emitted, which is today's behavior.

Gate design
-----------
* Genre-only prediction: each test break's predicted log_effect is the
  training-set mean for its (channel_name) cell.
* Genre+series prediction: the training-set mean for its (channel_name,
  series_key) cell, falling back to the (channel_name) genre mean for any
  series not seen in training (honest cold-start).
* Out-of-sample RMSE is computed for both predictions.
* The series layer is activated if and only if the improvement of series over
  genre exceeds SERIES_GATE_MIN_RELATIVE_IMPROVEMENT (2 %). A marginal or
  within-noise improvement does not activate the layer; a genuine title-level
  signal does.

Gate statistic (fold-averaged)
------------------------------
The uncertainty-calibration review (docs/model-validation/
uncertainty-calibration.md, section 4) measured that a single 80/20 split
carries about 4.1pp of split-to-split dispersion, while averaging the same
comparison over folds roughly halves it without changing any verdict. The
gate statistic is therefore the MEAN relative improvement over
:data:`GATE_FOLDS` temporal folds (contiguous blocks in break_start order,
every break predicted exactly once), against the SAME 2 % threshold. Where
temporal folds are impossible (no usable break_start), it averages over
:data:`GATE_FOLDS` seeded 80/20 splits instead. The legacy single-split
statistic remains available via ``statistic_method="single_split"`` and is
computed exactly as before, value for value.

Verified on the real reference month before the fold statistic became the
default: single split -8.3 %, temporal fold mean -8.5 % (fold sd 3.4pp), both
far below the +2 % bar, so the shipped verdict (series layer OFF) is
unchanged by construction and by measurement.

The decision and the statistic are recorded in the JSON metadata as
``series_layer_active`` (bool), ``series_gate_holdout`` (dict with
``genre_rmse``, ``series_rmse``, ``n_test``, ``relative_improvement``,
``gate_statistic_method``, ``folds``, ``fold_sd``) and ``series_gate_reason``
(a one-line human-readable explanation), so any reader can audit the decision.

When ``--series force-on`` or ``--series force-off`` is passed on the command
line, the gate is bypassed and the override takes effect, but the metadata
still records the gate numbers alongside the reason "forced by --series flag".
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from kairos.data.title_features import canonicalize_series

logger = logging.getLogger(__name__)

# Relative RMSE improvement the series layer must achieve over the genre-only
# baseline to be activated. 0.02 means the series RMSE must be at least 2 %
# lower than the genre RMSE; a smaller difference is treated as a tie.
SERIES_GATE_MIN_RELATIVE_IMPROVEMENT = 0.02

# Fraction of breaks withheld as the test set. 0.2 (20 %) balances leaving
# enough training data for stable cell means and enough test data for reliable
# RMSE estimates. Used by the single-split statistic and the seeded fallback.
HOLDOUT_FRACTION = 0.20

# Reproducible random seed so consecutive script runs give the same split.
_HOLDOUT_SEED = 42

# Minimum number of test breaks needed to run the gate. Below this the RMSE
# estimate is too noisy to trust, so the gate abstains (series omitted).
_MIN_TEST_BREAKS = 10

# How many folds (or seeded splits) the fold-averaged statistic uses.
GATE_FOLDS = 5
# Statistic methods: the fold average (temporal folds, or seeded splits where
# time order is unavailable) and the legacy single deterministic 80/20 split.
STATISTIC_FOLD_MEAN = "fold_mean"
STATISTIC_SINGLE_SPLIT = "single_split"
# Default statistic. fold_mean was made the default only after the real-data
# verdict was proven identical to the single split (module docstring numbers).
SERIES_GATE_STATISTIC_METHOD = STATISTIC_FOLD_MEAN


def series_holdout_gate(
    effects: pd.DataFrame,
    *,
    min_relative_improvement: float = SERIES_GATE_MIN_RELATIVE_IMPROVEMENT,
    holdout_fraction: float = HOLDOUT_FRACTION,
    statistic_method: str | None = None,
) -> dict[str, object]:
    """Evaluate whether the series layer beats genre-only on held-out breaks.

    Returns a dict with keys:
      ``series_layer_active``  bool: True iff the layer should be emitted.
      ``series_gate_holdout``  dict: genre_rmse, series_rmse, n_test,
                               relative_improvement, gate_statistic_method,
                               folds, fold_sd. Under the fold statistic the
                               RMSEs are fold means and n_test counts every
                               tested break across folds.
      ``series_gate_reason``   str: one-line human explanation of the decision.

    ``statistic_method`` selects :data:`STATISTIC_FOLD_MEAN` (default) or
    :data:`STATISTIC_SINGLE_SPLIT` (the legacy 80/20 split, unchanged value
    for value). When ``effects`` is empty or too small to split, the gate
    fails safely: ``series_layer_active`` is False and the reason explains why.
    """
    method = SERIES_GATE_STATISTIC_METHOD if statistic_method is None else statistic_method
    if method not in (STATISTIC_FOLD_MEAN, STATISTIC_SINGLE_SPLIT):
        raise ValueError(
            f"unknown statistic_method {method!r}; expected "
            f"{STATISTIC_FOLD_MEAN!r} or {STATISTIC_SINGLE_SPLIT!r}"
        )

    # Guard: need the title column and enough data.
    if effects.empty or "title" not in effects.columns:
        return _gate_result(
            active=False, genre_rmse=None, series_rmse=None, n_test=0,
            reason="no break effects available; series layer cannot be evaluated",
        )

    # Derive series keys on the full frame (needed for both train and test).
    work = effects[["channel_name", "log_effect", "title"]].copy()
    work["series_key"] = work["title"].map(canonicalize_series)

    n_total = len(work)
    n_test_target = max(1, int(round(n_total * holdout_fraction)))
    if n_test_target < _MIN_TEST_BREAKS:
        return _gate_result(
            active=False, genre_rmse=None, series_rmse=None, n_test=n_test_target,
            reason=(
                f"too few test breaks ({n_test_target} < {_MIN_TEST_BREAKS}) "
                "after the holdout split; series layer omitted"
            ),
        )

    if method == STATISTIC_SINGLE_SPLIT:
        return _single_split_gate(work, n_total, n_test_target, min_relative_improvement)
    return _fold_mean_gate(
        work, effects, n_total, n_test_target, min_relative_improvement,
    )


def _holdout_rmses(train: pd.DataFrame, test: pd.DataFrame) -> tuple[float, float]:
    """Out-of-sample (genre_rmse, series_rmse) for one train/test split."""
    # Genre-only cell means from training data.
    genre_means = (
        train.groupby("channel_name")["log_effect"].mean().to_dict()
    )
    # Series-level means from training data (within each genre cell).
    series_means: dict[tuple[str, str], float] = {}
    for (cell, key), grp in train.groupby(["channel_name", "series_key"]):
        if key:  # skip empty keys (unmatched titles)
            series_means[(str(cell), str(key))] = float(grp["log_effect"].mean())

    # Predict for test breaks.
    global_train_mean = float(train["log_effect"].mean()) if not train.empty else 0.0
    y_true = test["log_effect"].to_numpy()

    # Genre-only predictions.
    y_genre = np.array([
        genre_means.get(str(row.channel_name), global_train_mean)
        for row in test.itertuples(index=False)
    ])

    # Genre+series predictions (fall back to genre mean for unseen series).
    y_series = np.array([
        series_means.get(
            (str(row.channel_name), str(row.series_key)),
            genre_means.get(str(row.channel_name), global_train_mean),
        )
        for row in test.itertuples(index=False)
    ])

    genre_rmse = float(np.sqrt(np.mean((y_true - y_genre) ** 2)))
    series_rmse = float(np.sqrt(np.mean((y_true - y_series) ** 2)))
    return genre_rmse, series_rmse


def _single_split_gate(
    work: pd.DataFrame, n_total: int, n_test_target: int,
    min_relative_improvement: float,
) -> dict[str, object]:
    """The legacy deterministic 80/20 split, computed exactly as before."""
    rng = np.random.default_rng(_HOLDOUT_SEED)
    idx = rng.permutation(n_total)
    test_idx = set(idx[:n_test_target].tolist())
    train_mask = pd.Series([i not in test_idx for i in range(n_total)], index=work.index)

    train = work[train_mask]
    test = work[~train_mask]
    n_test = len(test)

    if n_test < _MIN_TEST_BREAKS:
        return _gate_result(
            active=False, genre_rmse=None, series_rmse=None, n_test=n_test,
            method=STATISTIC_SINGLE_SPLIT,
            reason=(
                f"too few test breaks ({n_test} < {_MIN_TEST_BREAKS}) after the holdout split; "
                "series layer omitted"
            ),
        )

    genre_rmse, series_rmse = _holdout_rmses(train, test)

    # Gate: series must be strictly better by the minimum relative margin.
    threshold = genre_rmse * (1.0 - min_relative_improvement)
    if genre_rmse <= 0.0:
        active = False
        improvement = None
        reason = (
            "genre RMSE is zero (degenerate data); series gate cannot compare; series layer omitted"
        )
    elif series_rmse < threshold:
        active = True
        improvement = (genre_rmse - series_rmse) / genre_rmse
        reason = (
            f"series RMSE ({series_rmse:.5f}) beats genre RMSE ({genre_rmse:.5f}) "
            f"by {100.0 * improvement:.1f}% (threshold {min_relative_improvement * 100:.0f}%); "
            "series layer activated"
        )
    else:
        active = False
        improvement = (genre_rmse - series_rmse) / genre_rmse if genre_rmse > 0 else 0.0
        reason = (
            f"series RMSE ({series_rmse:.5f}) does not beat genre RMSE ({genre_rmse:.5f}) "
            f"by the required {min_relative_improvement * 100:.0f}% margin "
            f"(actual improvement {100.0 * improvement:.1f}%); series layer omitted"
        )

    logger.info("Series gate: %s", reason)
    return _gate_result(
        active=active, genre_rmse=genre_rmse, series_rmse=series_rmse, n_test=n_test,
        reason=reason, method=STATISTIC_SINGLE_SPLIT, relative_improvement=improvement,
    )


def _fold_index_sets(
    work: pd.DataFrame, effects: pd.DataFrame, n_total: int, n_test_target: int,
) -> tuple[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Build the fold (train_positions, test_positions) pairs.

    Temporal folds when every row carries a usable ``break_start``: the rows
    are ordered by time (stable sort) and cut into :data:`GATE_FOLDS`
    contiguous blocks, so every break is tested exactly once against a model
    trained on the other weeks. Otherwise :data:`GATE_FOLDS` seeded 80/20
    splits (seeds ``_HOLDOUT_SEED + k``), the "seed averaged" fallback.
    """
    starts = None
    if "break_start" in effects.columns and len(effects) == n_total:
        candidate = pd.to_datetime(effects["break_start"], errors="coerce")
        if candidate.notna().all() and n_total >= GATE_FOLDS:
            starts = candidate.to_numpy()
    if starts is not None:
        order = np.argsort(starts, kind="stable")
        pairs = []
        for block in np.array_split(order, GATE_FOLDS):
            test_mask = np.zeros(n_total, dtype=bool)
            test_mask[block] = True
            pairs.append((np.flatnonzero(~test_mask), block))
        return "fold_mean_temporal", pairs

    pairs = []
    for k in range(GATE_FOLDS):
        rng = np.random.default_rng(_HOLDOUT_SEED + k)
        idx = rng.permutation(n_total)
        pairs.append((idx[n_test_target:], idx[:n_test_target]))
    return "seed_mean", pairs


def _fold_mean_gate(
    work: pd.DataFrame, effects: pd.DataFrame, n_total: int, n_test_target: int,
    min_relative_improvement: float,
) -> dict[str, object]:
    """The fold-averaged statistic: mean relative improvement over the folds."""
    positional = work.reset_index(drop=True)
    method, pairs = _fold_index_sets(work, effects, n_total, n_test_target)

    improvements: list[float] = []
    genre_rmses: list[float] = []
    series_rmses: list[float] = []
    n_test = 0
    for train_pos, test_pos in pairs:
        train = positional.iloc[train_pos]
        test = positional.iloc[test_pos]
        genre_rmse, series_rmse = _holdout_rmses(train, test)
        if genre_rmse <= 0.0:
            return _gate_result(
                active=False, genre_rmse=genre_rmse, series_rmse=series_rmse,
                n_test=n_test + len(test), method=method, folds=len(pairs),
                reason=(
                    "genre RMSE is zero on at least one fold (degenerate data); "
                    "series gate cannot compare; series layer omitted"
                ),
            )
        genre_rmses.append(genre_rmse)
        series_rmses.append(series_rmse)
        improvements.append((genre_rmse - series_rmse) / genre_rmse)
        n_test += len(test)

    statistic = float(np.mean(improvements))
    fold_sd = float(np.std(improvements, ddof=1)) if len(improvements) > 1 else None
    genre_mean = float(np.mean(genre_rmses))
    series_mean = float(np.mean(series_rmses))
    fold_kind = "temporal folds" if method == "fold_mean_temporal" else "seeded splits"
    sd_text = f"{100.0 * fold_sd:.1f}pp" if fold_sd is not None else "n/a"

    active = statistic > min_relative_improvement
    if active:
        reason = (
            f"series RMSE (fold mean {series_mean:.5f}) beats genre RMSE "
            f"(fold mean {genre_mean:.5f}) by {100.0 * statistic:.1f}% on average over "
            f"{len(pairs)} {fold_kind} (fold sd {sd_text}, threshold "
            f"{min_relative_improvement * 100:.0f}%); series layer activated"
        )
    else:
        reason = (
            f"series RMSE (fold mean {series_mean:.5f}) does not beat genre RMSE "
            f"(fold mean {genre_mean:.5f}) by the required "
            f"{min_relative_improvement * 100:.0f}% margin (mean improvement "
            f"{100.0 * statistic:.1f}% over {len(pairs)} {fold_kind}, fold sd {sd_text}); "
            "series layer omitted"
        )

    logger.info("Series gate: %s", reason)
    return _gate_result(
        active=active, genre_rmse=genre_mean, series_rmse=series_mean, n_test=n_test,
        reason=reason, method=method, folds=len(pairs), fold_sd=fold_sd,
        relative_improvement=statistic,
    )


def _gate_result(
    *,
    active: bool,
    genre_rmse: float | None,
    series_rmse: float | None,
    n_test: int,
    reason: str,
    method: str | None = None,
    folds: int | None = None,
    fold_sd: float | None = None,
    relative_improvement: float | None = None,
) -> dict[str, object]:
    return {
        "series_layer_active": active,
        "series_gate_holdout": {
            "genre_rmse": genre_rmse,
            "series_rmse": series_rmse,
            "n_test": n_test,
            "relative_improvement": relative_improvement,
            "gate_statistic_method": method,
            "folds": folds,
            "fold_sd": fold_sd,
        },
        "series_gate_reason": reason,
    }
