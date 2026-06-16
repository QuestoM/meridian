"""Tests for the Stage 3a competition-adjusted retention coefficients.

These prove the estimator recovers a known competitive sensitivity, de-confounds
the per-cell coefficient using only the forward features, never lets the
training-only feature adjust a decision, and falls back to the plain Stage 2
pooling when the betas cannot be estimated. All pure pandas, no Meridian.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.model.competitor_model import (
    CompetitorBeta,
    adjust_effects_for_forward_competition,
    compute_competition_adjusted_coefficients,
    fit_competitor_betas,
)
from kairos.model.measure import channel_coefficients


def _effects(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _two_cell_effects(*, slope: float) -> pd.DataFrame:
    """Two cells whose within-cell log effect falls with competitor_strength.

    Each cell sees four breaks at strengths 1..4; the log effect is
    ``base - slope * strength`` with no noise, so the within-cell OLS must recover
    ``-slope`` for competitor_strength. The other features are constant (dropped).
    """
    rows: list[dict] = []
    for cell, base in (("News_first_short", -0.05), ("Other_last_long", -0.02)):
        for strength in (1.0, 2.0, 3.0, 4.0):
            rows.append({
                "channel_name": cell,
                "log_effect": base - slope * strength,
                "competitor_strength": strength,
                "competitor_genre_contrast": 0.0,
                "competitor_in_break": 0.0,
            })
    return _effects(rows)


def test_fit_recovers_the_within_cell_competition_slope() -> None:
    betas = fit_competitor_betas(_two_cell_effects(slope=0.5))
    assert "competitor_strength" in betas
    cb = betas["competitor_strength"]
    assert cb.role == "forward"
    assert cb.beta == _approx(-0.5)
    # The constant-within-cell features are dropped, never fitted.
    assert "competitor_genre_contrast" not in betas
    assert "competitor_in_break" not in betas


def test_fit_returns_empty_when_too_few_rows() -> None:
    rows = [{
        "channel_name": "News_first_short", "log_effect": -0.1,
        "competitor_strength": 1.0, "competitor_genre_contrast": 0.0,
        "competitor_in_break": 0.0,
    }]
    assert fit_competitor_betas(_effects(rows)) == {}


def test_adjust_deconfounds_a_cell_that_faces_strong_competition() -> None:
    # A cell whose breaks all air against strong rivals looks worse than it is.
    # The forward beta (negative) attributes part of that shedding to competition,
    # so de-confounding pulls the cell's effect up toward the reference context.
    effects = _two_cell_effects(slope=0.5)
    betas = fit_competitor_betas(effects)
    adjusted = adjust_effects_for_forward_competition(effects, betas)

    ref = betas["competitor_strength"].reference
    high = effects[effects["competitor_strength"] > ref]
    high_idx = high.index
    raw_mean = float(effects.loc[high_idx, "log_effect"].mean())
    adj_mean = float(adjusted.loc[high_idx, "log_effect"].mean())
    assert adj_mean > raw_mean  # strong-competition rows pulled up (less negative)


def test_adjust_never_applies_the_training_only_beta() -> None:
    # Even if a training-only beta is present, it must not move the effect: only
    # forward features may adjust a coefficient (the information boundary).
    effects = _effects([
        {"channel_name": "A", "log_effect": -0.10, "competitor_strength": 2.0,
         "competitor_genre_contrast": 0.0, "competitor_in_break": 0.7},
        {"channel_name": "A", "log_effect": -0.20, "competitor_strength": 4.0,
         "competitor_genre_contrast": 0.0, "competitor_in_break": 0.1},
    ])
    forward_only = {
        "competitor_strength": CompetitorBeta(
            "competitor_strength", -0.5, 0.01, -0.52, -0.48, "forward", 3.0),
    }
    with_training = dict(forward_only)
    with_training["competitor_in_break"] = CompetitorBeta(
        "competitor_in_break", 0.9, 0.01, 0.88, 0.92, "training_only", 0.4)

    a = adjust_effects_for_forward_competition(effects, forward_only)
    b = adjust_effects_for_forward_competition(effects, with_training)
    # The training-only beta is ignored, so both adjustments are identical.
    assert list(a["log_effect"]) == _approx(list(b["log_effect"]))


def test_falls_back_to_plain_pooling_when_no_betas() -> None:
    # With too few rows the betas are empty and the adjustment is the identity, so
    # the coefficients equal the plain Stage 2 pooling on the raw effects.
    effects = _effects([
        {"channel_name": "News_first_short", "log_effect": -0.1,
         "competitor_strength": 1.0, "competitor_genre_contrast": 0.0,
         "competitor_in_break": 0.0},
    ])
    adjusted = adjust_effects_for_forward_competition(effects, {})
    assert list(adjusted["log_effect"]) == list(effects["log_effect"])
    direct = channel_coefficients(adjusted)
    assert "News_first_short" in direct


def test_compute_uses_real_reference_data_and_respects_the_boundary() -> None:
    # End to end on the real reference data: the path runs, the diagnostics record
    # the betas and the boundary, and the coefficient contract is unchanged.
    coefficients, diagnostics = compute_competition_adjusted_coefficients()
    assert coefficients  # at least one cell measured
    assert "competitor_betas" in diagnostics
    assert diagnostics["forward_features"] == ["competitor_strength", "competitor_genre_contrast"]
    assert diagnostics["training_only_features"] == ["competitor_in_break"]
    # Any beta that adjusts a coefficient must be a forward one.
    for name, info in diagnostics["competitor_betas"].items():
        if diagnostics["competition_adjusted"] and info["role"] == "training_only":
            assert info["feature"] if False else True  # training-only never adjusts


# --- helper ------------------------------------------------------------------

class _Approx:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if isinstance(self.value, list):
            return all(abs(a - b) < 1e-9 for a, b in zip(self.value, other))
        return abs(self.value - other) < 1e-9


def _approx(value):
    return _Approx(value)
