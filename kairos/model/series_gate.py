"""Automatic held-out gate for the series retention layer.

The series layer (genre -> series -> episode pooling) adds per-title
coefficients on top of the genre-cell coefficients. It is only useful when the
data genuinely supports per-title distinctions -- that is, when title-level
predictions outperform genre-level predictions on held-out breaks. When the
data is thin (e.g., a single month) or titles are too sparse, the gate fails
and only the genre layer is emitted, which is today's behavior.

Gate design
-----------
* A random 20 % of breaks (deterministic seed for reproducibility) are held
  out as the test set; 80 % form the training set.
* Genre-only prediction: each test break's predicted log_effect is the
  training-set mean for its (channel_name) cell.
* Genre+series prediction: the training-set mean for its (channel_name,
  series_key) cell, falling back to the (channel_name) genre mean for any
  series not seen in training (honest cold-start).
* Out-of-sample RMSE is computed for both predictions.
* The series layer is activated if and only if series_rmse < genre_rmse *
  (1 - SERIES_GATE_MIN_RELATIVE_IMPROVEMENT). With the default 2 % tolerance,
  a marginal or within-noise improvement does not activate the layer; a genuine
  title-level signal does.

The decision and both RMSE values are recorded in the JSON metadata as
``series_layer_active`` (bool), ``series_gate_holdout`` (dict with
``genre_rmse``, ``series_rmse``, ``n_test``) and ``series_gate_reason`` (a
one-line human-readable explanation), so any reader can audit the decision.

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
# RMSE estimates.
HOLDOUT_FRACTION = 0.20

# Reproducible random seed so consecutive script runs give the same split.
_HOLDOUT_SEED = 42

# Minimum number of test breaks needed to run the gate. Below this the RMSE
# estimate is too noisy to trust, so the gate abstains (series omitted).
_MIN_TEST_BREAKS = 10


def series_holdout_gate(
    effects: pd.DataFrame,
    *,
    min_relative_improvement: float = SERIES_GATE_MIN_RELATIVE_IMPROVEMENT,
    holdout_fraction: float = HOLDOUT_FRACTION,
) -> dict[str, object]:
    """Evaluate whether the series layer beats genre-only on held-out breaks.

    Returns a dict with keys:
      ``series_layer_active``  bool: True iff the layer should be emitted.
      ``series_gate_holdout``  dict: genre_rmse, series_rmse, n_test.
      ``series_gate_reason``   str: one-line human explanation of the decision.

    When ``effects`` is empty or too small to split, the gate fails safely:
    ``series_layer_active`` is False and the reason explains why.
    """
    # Guard: need the title column and enough data.
    if effects.empty or "title" not in effects.columns:
        return _gate_result(
            active=False,
            genre_rmse=None,
            series_rmse=None,
            n_test=0,
            reason="no break effects available; series layer cannot be evaluated",
        )

    # Derive series keys on the full frame (needed for both train and test).
    work = effects[["channel_name", "log_effect", "title"]].copy()
    work["series_key"] = work["title"].map(canonicalize_series)

    n_total = len(work)
    n_test_target = max(1, int(round(n_total * holdout_fraction)))
    if n_test_target < _MIN_TEST_BREAKS:
        return _gate_result(
            active=False,
            genre_rmse=None,
            series_rmse=None,
            n_test=n_test_target,
            reason=(
                f"too few test breaks ({n_test_target} < {_MIN_TEST_BREAKS}) "
                "after the holdout split; series layer omitted"
            ),
        )

    # Deterministic split: reproducible across runs so the gate decision is stable.
    rng = np.random.default_rng(_HOLDOUT_SEED)
    idx = rng.permutation(n_total)
    test_idx = set(idx[:n_test_target].tolist())
    train_mask = pd.Series([i not in test_idx for i in range(n_total)], index=work.index)

    train = work[train_mask]
    test = work[~train_mask]
    n_test = len(test)

    if n_test < _MIN_TEST_BREAKS:
        return _gate_result(
            active=False,
            genre_rmse=None,
            series_rmse=None,
            n_test=n_test,
            reason=(
                f"too few test breaks ({n_test} < {_MIN_TEST_BREAKS}) after the holdout split; "
                "series layer omitted"
            ),
        )

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

    # Gate: series must be strictly better by the minimum relative margin.
    threshold = genre_rmse * (1.0 - min_relative_improvement)
    if genre_rmse <= 0.0:
        active = False
        reason = (
            "genre RMSE is zero (degenerate data); series gate cannot compare; series layer omitted"
        )
    elif series_rmse < threshold:
        active = True
        pct_better = 100.0 * (genre_rmse - series_rmse) / genre_rmse
        reason = (
            f"series RMSE ({series_rmse:.5f}) beats genre RMSE ({genre_rmse:.5f}) "
            f"by {pct_better:.1f}% (threshold {min_relative_improvement * 100:.0f}%); "
            "series layer activated"
        )
    else:
        active = False
        pct = 100.0 * (genre_rmse - series_rmse) / genre_rmse if genre_rmse > 0 else 0.0
        reason = (
            f"series RMSE ({series_rmse:.5f}) does not beat genre RMSE ({genre_rmse:.5f}) "
            f"by the required {min_relative_improvement * 100:.0f}% margin "
            f"(actual improvement {pct:.1f}%); series layer omitted"
        )

    logger.info("Series gate: %s", reason)
    return _gate_result(
        active=active,
        genre_rmse=genre_rmse,
        series_rmse=series_rmse,
        n_test=n_test,
        reason=reason,
    )


def _gate_result(
    *,
    active: bool,
    genre_rmse: float | None,
    series_rmse: float | None,
    n_test: int,
    reason: str,
) -> dict[str, object]:
    return {
        "series_layer_active": active,
        "series_gate_holdout": {
            "genre_rmse": genre_rmse,
            "series_rmse": series_rmse,
            "n_test": n_test,
        },
        "series_gate_reason": reason,
    }
