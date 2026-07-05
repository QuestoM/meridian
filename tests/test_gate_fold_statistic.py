"""Fold-averaged gate statistic: verdict preservation and metadata contract.

The gates' decision statistic moved from a single 80/20 split to the mean
relative improvement over 5 temporal folds (seeded splits where time order is
unavailable), same 2 percent threshold. These tests prove on synthetic frames
with known truth that the fold statistic reaches the SAME verdict as the
legacy single split in both directions (a real signal activates, no signal
stays off), that the method used is recorded, and that the legacy path is
still selectable. The real-data verdict preservation is pinned by
tests/test_drift_rebuild_metadata.py (realdata marker).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.model.competitor_gate import counterprogramming_holdout_gate
from kairos.model.series_gate import series_holdout_gate

_GATE_STAT_KEYS = {"gate_statistic_method", "folds", "fold_sd", "relative_improvement"}


# ---------------------------------------------------------------------------
# Synthetic frames with known truth
# ---------------------------------------------------------------------------

def _series_frame(*, diverging: bool, with_time: bool, n: int = 240) -> pd.DataFrame:
    """One genre cell, two interleaved series spread evenly across the month.

    With ``diverging`` the second series sits far from the genre mean (a real
    title-level signal); without it both series share one distribution, so the
    series layer can only add noise.
    """
    rng = np.random.default_rng(7)
    rows = []
    for i in range(n):
        loser = i % 2 == 1
        mean = -0.25 if (diverging and loser) else -0.05
        rows.append(
            {
                "channel_name": "News_first_short",
                "title": "Big Loser Show" if loser else "Normal Show",
                "log_effect": float(rng.normal(mean, 0.01)),
            }
        )
    frame = pd.DataFrame(rows)
    if with_time:
        frame["break_start"] = [
            pd.Timestamp("2024-11-01 20:00") + pd.Timedelta(hours=3 * i) for i in range(n)
        ]
    return frame


def _competitor_frame(*, planted_beta: float, with_time: bool, n: int = 300) -> pd.DataFrame:
    """Two cells; strength varies within cells; other features constant."""
    rng = np.random.default_rng(7)
    cells = np.where(rng.random(n) < 0.5, "News_first_short", "Other_last_long")
    strength = rng.uniform(0.0, 10.0, size=n)
    noise = rng.normal(0.0, 0.01, size=n)
    frame = pd.DataFrame(
        {
            "channel_name": cells,
            "log_effect": planted_beta * strength + noise,
            "competitor_strength": strength,
            "competitor_genre_contrast": np.full(n, 0.5),
            "competitor_prog_start": np.zeros(n),
            "competitor_in_break": np.full(n, 0.3),
        }
    )
    if with_time:
        frame["break_start"] = [
            pd.Timestamp("2024-11-01 18:00") + pd.Timedelta(hours=2 * i) for i in range(n)
        ]
    return frame


# ---------------------------------------------------------------------------
# Series gate
# ---------------------------------------------------------------------------

def test_series_fold_statistic_activates_on_real_signal_like_single_split() -> None:
    frame = _series_frame(diverging=True, with_time=True)
    fold = series_holdout_gate(frame)
    single = series_holdout_gate(frame, statistic_method="single_split")

    assert fold["series_layer_active"] is True
    assert single["series_layer_active"] is True  # same verdict, both directions

    holdout = fold["series_gate_holdout"]
    assert _GATE_STAT_KEYS <= set(holdout)
    assert holdout["gate_statistic_method"] == "fold_mean_temporal"
    assert holdout["folds"] == 5
    assert holdout["fold_sd"] is not None and holdout["fold_sd"] >= 0.0
    assert holdout["relative_improvement"] > 0.02
    assert holdout["n_test"] == len(frame)  # every break tested exactly once


def test_series_fold_statistic_stays_off_without_signal_like_single_split() -> None:
    frame = _series_frame(diverging=False, with_time=True)
    fold = series_holdout_gate(frame)
    single = series_holdout_gate(frame, statistic_method="single_split")

    assert fold["series_layer_active"] is False
    assert single["series_layer_active"] is False
    assert fold["series_gate_holdout"]["relative_improvement"] < 0.02


def test_series_falls_back_to_seed_mean_without_timestamps() -> None:
    frame = _series_frame(diverging=True, with_time=False)
    result = series_holdout_gate(frame)
    holdout = result["series_gate_holdout"]
    assert result["series_layer_active"] is True
    assert holdout["gate_statistic_method"] == "seed_mean"
    assert holdout["folds"] == 5


def test_series_single_split_path_is_labelled_and_unchanged_shape() -> None:
    frame = _series_frame(diverging=True, with_time=True)
    result = series_holdout_gate(frame, statistic_method="single_split")
    holdout = result["series_gate_holdout"]
    assert holdout["gate_statistic_method"] == "single_split"
    assert holdout["folds"] is None
    assert holdout["fold_sd"] is None
    # Legacy split size: 20 percent of the frame.
    assert holdout["n_test"] == round(0.2 * len(frame))


def test_series_gate_rejects_unknown_statistic_method() -> None:
    with pytest.raises(ValueError):
        series_holdout_gate(_series_frame(diverging=True, with_time=True),
                            statistic_method="bootstrap")


def test_series_gate_stat_keys_present_even_when_gate_abstains() -> None:
    result = series_holdout_gate(
        pd.DataFrame(columns=["channel_name", "title", "log_effect"])
    )
    assert _GATE_STAT_KEYS <= set(result["series_gate_holdout"])


# ---------------------------------------------------------------------------
# Counter-programming gate
# ---------------------------------------------------------------------------

def test_competitor_fold_statistic_activates_on_planted_beta_like_single_split() -> None:
    frame = _competitor_frame(planted_beta=-0.02, with_time=True)
    fold = counterprogramming_holdout_gate(frame)
    single = counterprogramming_holdout_gate(frame, statistic_method="single_split")

    assert fold["counterprogramming_active"] is True
    assert single["counterprogramming_active"] is True

    holdout = fold["counterprogramming_holdout"]
    assert _GATE_STAT_KEYS <= set(holdout)
    assert holdout["gate_statistic_method"] == "fold_mean_temporal"
    assert holdout["folds"] == 5
    assert holdout["fold_sd"] is not None
    assert holdout["relative_improvement"] > 0.02
    assert holdout["n_test"] == len(frame)
    # The display betas still recover the planted effect.
    beta = fold["counterprogramming_betas"]["competitor_strength"]["beta"]
    assert beta == pytest.approx(-0.02, abs=0.005)


def test_competitor_fold_statistic_stays_off_on_noise_like_single_split() -> None:
    frame = _competitor_frame(planted_beta=0.0, with_time=True)
    fold = counterprogramming_holdout_gate(frame)
    single = counterprogramming_holdout_gate(frame, statistic_method="single_split")

    assert fold["counterprogramming_active"] is False
    assert single["counterprogramming_active"] is False
    assert "covariate left off" in fold["counterprogramming_reason"]
    assert fold["counterprogramming_holdout"]["relative_improvement"] < 0.02


def test_competitor_falls_back_to_seed_mean_without_timestamps() -> None:
    frame = _competitor_frame(planted_beta=-0.02, with_time=False)
    result = counterprogramming_holdout_gate(frame)
    holdout = result["counterprogramming_holdout"]
    assert result["counterprogramming_active"] is True
    assert holdout["gate_statistic_method"] == "seed_mean"
    assert holdout["folds"] == 5


def test_competitor_gate_rejects_unknown_statistic_method() -> None:
    with pytest.raises(ValueError):
        counterprogramming_holdout_gate(
            _competitor_frame(planted_beta=0.0, with_time=False),
            statistic_method="jackknife",
        )
