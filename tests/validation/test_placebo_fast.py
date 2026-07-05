"""Standing fast placebo check for the retention-cost measurement.

Guards the 2026-07 causal-identification review
(docs/model-validation/causal-identification.md) against silent rot: a
seeded, subsampled version of the placebo / negative-control computation
(scripts/validation/run_placebo.py) re-runs on every realdata test pass.

What it asserts, and why each bound is where it is (all calibrated against
the full-sample review on the November 2024 reference data):

  * the measurement machinery still reproduces the shipped pooled effect
    (full run: mu = -0.03984 log, delta -0.03906). If this moves, the shipped
    coefficients and this review no longer describe the same model;
  * pseudo-breaks measured with the same machinery at eligible no-break
    minutes do NOT read as spurious COST: the full-sample placebo mean is
    +0.0149 log (95% cluster CI [+0.0081, +0.0217] on the delta scale), i.e.
    the machinery's no-break counterfactual is a positive within-show drift.
    The subsampled placebo mean must stay inside (-0.005, +0.040): the lower
    bound fails if the machinery starts manufacturing negative cost at
    no-break minutes, the upper if the drift explodes;
  * the placebo stays well above the real-break effect (the causal contrast
    the permutation test found at p = 0.0005 survives in sign).

Deterministic: numpy default_rng(42), fixed subsample of 400 source breaks,
one pseudo-break each. Runtime is dominated by the reference-data load
(realdata marker, excluded from the fast gate like every xlsx-reading test).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

_SPEC = importlib.util.spec_from_file_location(
    "kairos_validation_common", REPO / "scripts" / "validation" / "common.py")
common = importlib.util.module_from_spec(_SPEC)
sys.modules["kairos_validation_common"] = common  # dataclasses need the module registered
_SPEC.loader.exec_module(common)

SUBSAMPLE = 400
MIN_MEASURED_PSEUDO = 250
PLACEBO_LOG_BOUNDS = (-0.005, 0.040)
FULL_POOLED_MU = -0.03984282196759275
POOLED_TOLERANCE = 0.004


@pytest.mark.realdata
def test_placebo_negative_control_fast():
    bundle = common.load_bundle(verbose=False)
    effects = bundle.effects

    # 1. The machinery still measures what the review reviewed.
    pooled = common.dl_pool(effects)
    assert abs(pooled["mu"] - FULL_POOLED_MU) < POOLED_TOLERANCE, (
        f"pooled effect moved: mu={pooled['mu']:+.5f} vs reviewed "
        f"{FULL_POOLED_MU:+.5f}; re-run the causal-identification review "
        f"(scripts/validation/) before trusting its conclusions")

    # 2. Seeded subsample of source breaks, one matched pseudo-break each.
    rng = np.random.default_rng(common.SEED)
    matched_rows = effects[effects["prog_key"].notna()].reset_index(drop=True)
    take = rng.choice(len(matched_rows), size=min(SUBSAMPLE, len(matched_rows)),
                      replace=False)
    subset = matched_rows.iloc[np.sort(take)]
    pseudo = common.sample_matched_pseudo(bundle, rng, k=1, effects=subset)

    assert len(pseudo) >= MIN_MEASURED_PSEUDO, (
        f"only {len(pseudo)} pseudo-breaks measurable of {len(subset)} sampled; "
        f"the placebo eligibility rules or the data changed materially")

    # 3. No manufactured cost at no-break minutes (the review's core finding).
    placebo_mean = float(pseudo["log_effect"].mean())
    lo, hi = PLACEBO_LOG_BOUNDS
    assert lo < placebo_mean < hi, (
        f"placebo mean log-effect {placebo_mean:+.5f} left ({lo}, {hi}): the "
        f"measurement machinery now reads no-break minutes as "
        f"{'spurious cost' if placebo_mean <= lo else 'runaway drift'}; the "
        f"shipped coefficients' causal reading is no longer covered by the "
        f"2026-07 review")

    # 4. The causal contrast keeps its sign with room to spare.
    assert placebo_mean - pooled["mu"] > 0.02, (
        f"real-vs-placebo gap collapsed: placebo {placebo_mean:+.5f}, real "
        f"{pooled['mu']:+.5f}")
