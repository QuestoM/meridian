"""Fast standing parameter-recovery smoke test of the EB pooling pipeline.

A subsampled, fully seeded version of
scripts/validation/parameter_recovery.py: simulate the model's own
hierarchical setup (36 cells with the real break counts, known tau^2, normal
within-cell noise at the real pooled s^2), push every replication through THE
ACTUAL pipeline (:func:`kairos.model.measure.channel_coefficients` and the
same DerSimonian-Laird estimators it calls), and assert the machinery still
recovers the truth to the tolerances measured in the full 500-replication
study (scripts/validation/out/parameter_recovery.json, 2026-07-05).

Entirely synthetic (no reference data read), deterministic (fixed
default_rng seeds, so the numbers are bit-stable run to run), and fast
(~40 replications at the 12x sample scale, a few seconds). If a change to
the pooling code moves these numbers outside the bands, the change is not
behavior-preserving for the uncertainty machinery.

Reference values from the full study at scale 12x, tau2_true = 9.6868e-05
(500 reps): tau2-hat mean 9.65e-05 (rel bias ~0.00), P(tau2-hat = 0) = 0.00,
95% CI coverage 0.914, EB point RMSE 0.0077 vs raw 0.0146.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.model.measure import (
    _cell_stats,
    _dersimonian_laird,
    _pooled_within_variance,
    channel_coefficients,
)

# The real per-cell break counts from models/tv_break_coefficients.json
# (computed 2026-06-17, fingerprint-fresh 2026-07-05), hardcoded so this test
# reads no data files and stays in the fast gate.
REAL_CELL_NS = [
    27, 21, 44, 96, 67, 58, 105, 20, 51, 238, 100, 207, 159, 146, 224, 292,
    134, 255, 12, 13, 28, 11, 8, 23, 15, 18, 22, 24, 14, 9, 13, 10, 23, 8,
    13, 24,
]
CELL_NAMES = [f"cell_{i:02d}" for i in range(len(REAL_CELL_NS))]

TAU2_TRUE = 9.686792021678731e-05  # the shipped artifact's learned tau^2
S2_TRUE = 0.058196048399336        # the shipped pooled within-cell variance
MU_LOG = -0.0399
SCALE = 12                          # the "one-year" scenario: tight recovery
N_REPS = 40
SEED = 20260707
Z95 = 1.959963984540054


def _simulate_and_fit(seed: int, n_reps: int):
    rng = np.random.default_rng(seed)
    ns = np.array(REAL_CELL_NS, dtype=int) * SCALE
    names = np.repeat(CELL_NAMES, ns)
    total = int(ns.sum())
    m = len(CELL_NAMES)

    tau2_hats, covers, rmse_eb, rmse_raw = [], [], [], []
    for _ in range(n_reps):
        theta = MU_LOG + rng.normal(0.0, np.sqrt(TAU2_TRUE), size=m)
        y = np.repeat(theta, ns) + rng.normal(0.0, np.sqrt(S2_TRUE), size=total)
        effects = pd.DataFrame({"channel_name": names, "log_effect": y})

        coeffs = channel_coefficients(effects)  # THE ACTUAL PIPELINE
        stats = _cell_stats(effects)
        pw = _pooled_within_variance(stats)
        tau2_hat, _mu_hat, _sw = _dersimonian_laird(stats, pw)
        tau2_hats.append(tau2_hat)

        theta_hat = np.array([np.log1p(coeffs[n].raw_delta) for n in CELL_NAMES])
        half95 = np.array(
            [
                0.5 * (np.log1p(coeffs[n].ci_high) - np.log1p(coeffs[n].ci_low))
                for n in CELL_NAMES
            ]
        )
        ybar = np.array([dict((s[0], s[2]) for s in stats)[n] for n in CELL_NAMES])
        covers.append(float(np.mean(np.abs(theta - theta_hat) <= half95)))
        rmse_eb.append(float(np.sqrt(np.mean((theta_hat - theta) ** 2))))
        rmse_raw.append(float(np.sqrt(np.mean((ybar - theta) ** 2))))

    return (
        np.array(tau2_hats),
        float(np.mean(covers)),
        float(np.mean(rmse_eb)),
        float(np.mean(rmse_raw)),
    )


@pytest.fixture(scope="module")
def recovery():
    return _simulate_and_fit(SEED, N_REPS)


def test_tau2_recovered_within_tolerance(recovery):
    """Mean tau2-hat lands within [0.5x, 2.0x] of the known truth at 12x scale.

    The full study measured essentially zero relative bias at this scale
    (mean 9.65e-05 vs truth 9.69e-05, sd 4.8e-05 over 500 reps); the wide band
    keeps the 40-rep smoke immune to Monte Carlo wobble while still catching
    any real change to the DL estimator or its inputs.
    """
    tau2_hats, _, _, _ = recovery
    mean = float(np.mean(tau2_hats))
    assert 0.5 * TAU2_TRUE <= mean <= 2.0 * TAU2_TRUE, (
        f"tau2-hat mean {mean:.3e} outside [{0.5 * TAU2_TRUE:.3e}, "
        f"{2.0 * TAU2_TRUE:.3e}] (truth {TAU2_TRUE:.3e})"
    )


def test_tau2_not_stuck_at_zero(recovery):
    """At 12x the DL floor should essentially never trigger (study: P = 0.00)."""
    tau2_hats, _, _, _ = recovery
    assert float(np.mean(tau2_hats == 0.0)) <= 0.10


def test_ci_coverage_in_band(recovery):
    """95% CI coverage of true cell effects within [0.85, 0.99] (study: 0.914).

    Known, documented undercoverage from unpropagated tau^2 estimation error
    is ~0.91 at this scale; a drop below 0.85 or a jump above 0.99 means the
    interval construction changed.
    """
    _, cover95, _, _ = recovery
    assert 0.85 <= cover95 <= 0.99, f"95% coverage {cover95:.3f} outside [0.85, 0.99]"


def test_eb_beats_raw_means(recovery):
    """The pooled point estimates must beat unpooled cell means (study: 0.0077 vs 0.0146)."""
    _, _, eb, raw = recovery
    assert eb < raw, f"EB RMSE {eb:.4f} not below raw-mean RMSE {raw:.4f}"


def test_pipeline_is_deterministic_given_seed():
    """Same seed, same numbers: the pooling has no hidden randomness or state."""
    a_tau2, a_cover, _, _ = _simulate_and_fit(SEED + 1, 3)
    b_tau2, b_cover, _, _ = _simulate_and_fit(SEED + 1, 3)
    assert np.array_equal(a_tau2, b_tau2)
    assert a_cover == b_cover
