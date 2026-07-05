"""Rebuild-equivalence proof for the gated coefficient rebuild.

scripts/compute_measured_coefficients.py now runs the counter-programming
held-out gate (and records the evaluate-only detrend seasonality verdict) on
every rebuild. Law 9 demands that TODAY this changes nothing: with the gate
failing on the one-month data, a fresh rebuild into a temp path must reproduce
the shipped models/tv_break_coefficients.json decision byte-equivalently, all
36 flat coefficients exactly equal, with only the NEW metadata keys (and the
computed_at timestamp) differing.

The second test proves the self-activation path actually works without
waiting for the data drop: forcing the covariate on emits competition-adjusted
coefficients plus the fitted betas, while the gate verdict stays honestly
recorded. Both tests write ONLY under tmp_path; the shipped artifact is never
touched.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compute_measured_coefficients.py"
SHIPPED = ROOT / "models" / "tv_break_coefficients.json"

# Exactly the metadata keys the gate wiring added. The equivalence assertion
# below fails if the rebuild ever adds, renames or drops a key silently.
# The placebo_correction_* keys are the drift-correction layer's always-written
# verdict (docs/model-validation/causal-identification.md fix 1+2): measured on
# every rebuild, applied only under an explicit --placebo-correction force-on.
NEW_METADATA_KEYS = {
    "counterprogramming_active",
    "counterprogramming_holdout",
    "counterprogramming_reason",
    "counterprogramming_forward_features",
    "detrend_baseline_mode",
    "detrend_seasonality_recommended",
    "detrend_seasonality_holdout",
    "detrend_seasonality_reason",
    "placebo_correction_active",
    "placebo_correction",
    "placebo_correction_reason",
    # Interval-calibration keys (docs/model-validation/uncertainty-calibration.md
    # P2+P3): the measured verdict is always written; the calibrated bands and
    # moderated variances are applied only under an explicit force-on.
    "interval_method",
    "moderated_variances",
    "bootstrap_B",
    "interval_seed",
    "prior_df",
    "width_factor_measured",
    "interval_calibration_reason",
    # Weekly level-drift monitor (docs/model-validation/uncertainty-calibration.md
    # finding 4): measured on every rebuild, disclosure-only (never moves a
    # coefficient); detailed assertions live in tests/test_drift_rebuild_metadata.py.
    "level_drift",
}


def _run_rebuild(tmp_path: Path, *extra: str) -> tuple[dict, str]:
    out = tmp_path / "coeffs.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUTF8"] = "1"
    env.pop("KAIROS_SERIES_LAYER", None)
    env.pop("KAIROS_COUNTERPROGRAMMING", None)
    env.pop("KAIROS_PLACEBO_CORRECTION", None)
    env.pop("KAIROS_INTERVAL_CALIBRATION", None)
    env.pop("KAIROS_MODERATED_VARIANCES", None)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), *extra],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(out.read_text(encoding="utf-8")), proc.stdout


@pytest.mark.realdata
def test_rebuild_reproduces_shipped_decision_byte_equivalently(tmp_path) -> None:
    fresh, stdout = _run_rebuild(tmp_path)
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))

    # The 36 coefficients the optimizer consumes: exactly equal, value for value.
    assert len(shipped["coefficients"]) == 36
    assert fresh["coefficients"] == shipped["coefficients"]
    assert fresh["method"] == shipped["method"]
    # The series block stays absent, as shipped.
    assert ("series" in fresh) == ("series" in shipped)
    # Per-cell detail: the decision-bearing fields are identical. (ci bounds
    # can differ at the last ulp across platforms; the shipped artifact was
    # built on Windows, so they are not asserted bytewise here.)
    for name, detail in shipped["detail"].items():
        assert fresh["detail"][name]["coefficient"] == detail["coefficient"]
        assert fresh["detail"][name]["raw_delta"] == detail["raw_delta"]
        assert fresh["detail"][name]["n"] == detail["n"]

    meta_fresh, meta_shipped = fresh["metadata"], shipped["metadata"]

    # The covariate is OFF today, exactly as the science wave measured, and
    # the verdict is fully recorded: pass/fail, both RMSEs, delta, threshold.
    assert meta_fresh["counterprogramming_active"] is False
    assert "counterprogramming_betas" not in meta_fresh
    hold = meta_fresh["counterprogramming_holdout"]
    assert hold["rmse_without"] is not None and hold["rmse_with"] is not None
    assert hold["n_test"] > 0
    assert hold["min_relative_improvement"] == 0.02
    assert hold["relative_improvement"] < hold["min_relative_improvement"]
    assert "covariate left off" in meta_fresh["counterprogramming_reason"]

    # Detrend seasonality: evaluated, recorded, NOT enabled. One month of
    # data cannot show seasonal structure, so the verdict is honest.
    assert meta_fresh["detrend_baseline_mode"] == "global"
    assert meta_fresh["detrend_seasonality_recommended"] is False
    assert meta_fresh["detrend_seasonality_holdout"]["min_relative_improvement"] == 0.02

    # Placebo-drift correction: applied by default (the measured within-show
    # build is subtracted, content-only baseline in force). The recorded drift
    # must be the review's positive value, never a fabricated zero, and the
    # corrected pooled charge is auditable straight from the metadata.
    assert meta_fresh["placebo_correction_active"] is True
    drift = meta_fresh["placebo_correction"]
    assert drift["baseline"] == "content_only"
    assert drift["n_pseudo"] > 0
    assert drift["pooled_drift"] > 0
    assert set(drift["per_genre_drift"]) == set(drift["per_genre_n"])
    assert drift["pooled_corrected_delta"] == pytest.approx(-0.0496, abs=0.002)
    assert "active by default" in meta_fresh["placebo_correction_reason"]

    # Every pre-existing decision and measurement summary is unchanged.
    for key in (
        "channels", "total_breaks_measured", "negative_cells",
        "pooling_method", "series_layer_active", "series_layer",
        "series_count", "first_break_multiplier", "first_break_active",
        "first_break_n_first", "first_break_n_later",
    ):
        assert meta_fresh[key] == meta_shipped[key], key

    # The shipped artifact already carries every layer's metadata; a fresh
    # rebuild adds nothing and removes nothing.
    assert set(meta_fresh) == set(meta_shipped)


@pytest.mark.realdata
def test_force_on_emits_competition_adjusted_coefficients(tmp_path) -> None:
    # Placebo correction (on by default) overrides the competition adjustment,
    # so isolate the counter-programming layer with both default layers off.
    fresh, stdout = _run_rebuild(
        tmp_path, "--counterprogramming", "force-on",
        "--placebo-correction", "force-off", "--interval-calibration", "force-off",
    )
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    meta = fresh["metadata"]

    assert meta["counterprogramming_active"] is True
    assert "forced by --counterprogramming force-on" in meta["counterprogramming_reason"]
    betas = meta["counterprogramming_betas"]
    assert any(spec["role"] == "forward" for spec in betas.values())
    # The training-only control is fitted but tagged so it can never reach a
    # forward path (future_epg.forward_adjustment filters and asserts on role).
    assert betas["competitor_in_break"]["role"] == "training_only"

    # Activation genuinely moves the emitted numbers: same 36 cells, at least
    # one de-confounded coefficient differs from the shipped plain ones.
    assert set(fresh["coefficients"]) == set(shipped["coefficients"])
    assert fresh["coefficients"] != shipped["coefficients"]
    assert "counter-programming ACTIVE" in stdout
