"""Rebuild-time drift metadata and the /api/impact drift block.

Proves, on the real reference data, that a fresh coefficient rebuild into a
temp path (the shipped artifact is never touched):

  * always writes the ``level_drift`` block with the required keys
    (weekly_levels, drift_per_week, drift_se, binding, criterion), measured
    from the same month the coefficients are pooled on, and that the measured
    verdict matches the uncertainty-calibration review (drift about +0.0202,
    binding True against twice the pooled half-width);
  * records the fold-averaged gate statistic for both held-out gates
    (gate_statistic_method, folds, fold_sd) with the verdicts UNCHANGED
    (series layer still off, counter-programming still off) and the flat
    coefficients byte-equal to the shipped ones, so the statistic upgrade is
    disclosure-only;

and that GET /api/impact exposes a drift block honestly: the measurement when
the artifact carries one, an explicit "unavailable" (never a fabricated
verdict) when it predates the monitor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compute_measured_coefficients.py"
SHIPPED = ROOT / "models" / "tv_break_coefficients.json"

DRIFT_REQUIRED_KEYS = {
    "weekly_levels", "drift_per_week", "drift_se", "binding", "criterion",
}
GATE_STAT_KEYS = {"gate_statistic_method", "folds", "fold_sd"}


def _run_rebuild(tmp_path: Path) -> dict:
    out = tmp_path / "coeffs.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUTF8"] = "1"
    env.pop("KAIROS_SERIES_LAYER", None)
    env.pop("KAIROS_COUNTERPROGRAMMING", None)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out)],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.realdata
def test_rebuild_writes_measured_drift_and_fold_gate_metadata(tmp_path) -> None:
    fresh = _run_rebuild(tmp_path)
    meta = fresh["metadata"]

    # --- level drift: present, measured, and matching the review's numbers.
    drift = meta["level_drift"]
    assert DRIFT_REQUIRED_KEYS <= set(drift)
    assert drift["status"] == "measured"
    assert drift["n_weeks"] == 5
    assert len(drift["weekly_levels"]) == 5
    assert sum(w["n"] for w in drift["weekly_levels"]) == meta["total_breaks_measured"]
    # The review measured the trailing-week level shift at +0.0202 against a
    # binding threshold of twice the pooled half-width (2 x 0.0094 = 0.0188).
    assert drift["drift_per_week"] == pytest.approx(0.0202, abs=2e-3)
    assert drift["drift_se"] == pytest.approx(0.0117, abs=2e-3)
    assert drift["binding_threshold"] == pytest.approx(0.0188, abs=1e-3)
    assert drift["binding"] is True
    assert "half-width" in drift["criterion"]

    # --- series gate: fold-averaged statistic recorded, verdict unchanged.
    series_hold = meta["series_gate_holdout"]
    assert GATE_STAT_KEYS <= set(series_hold)
    assert series_hold["gate_statistic_method"] == "fold_mean_temporal"
    assert series_hold["folds"] == 5
    assert series_hold["fold_sd"] > 0.0
    # The series layer is measurably WORSE than genre out of sample (about
    # -8.5 percent mean improvement over temporal folds), far below the +2
    # percent activation bar: still off.
    assert series_hold["relative_improvement"] < -0.02
    assert meta["series_layer_active"] is False

    # --- counter-programming gate: same statistic upgrade, same verdict.
    cp_hold = meta["counterprogramming_holdout"]
    assert GATE_STAT_KEYS <= set(cp_hold)
    assert cp_hold["gate_statistic_method"] == "fold_mean_temporal"
    assert cp_hold["folds"] == 5
    assert cp_hold["fold_sd"] > 0.0
    # The covariate's improvement is within noise of zero, below the bar.
    assert abs(cp_hold["relative_improvement"]) < 0.02
    assert meta["counterprogramming_active"] is False

    # --- the money proof: the statistic upgrade and the drift monitor are
    # disclosure-only. Every flat coefficient equals the shipped artifact.
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert fresh["coefficients"] == shipped["coefficients"]


# ---------------------------------------------------------------------------
# /api/impact drift block (no real-data rebuild required)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def client() -> TestClient:
    from kairos_api.server import app

    return TestClient(app, raise_server_exceptions=False)


def test_api_impact_exposes_drift_block_honestly(client) -> None:
    response = client.get("/api/impact")
    assert response.status_code == 200
    body = response.json()
    assert "drift" in body
    drift = body["drift"]
    assert isinstance(drift, dict)
    status = drift.get("status")
    assert status in {"measured", "insufficient_data", "unavailable"}
    if status == "unavailable":
        # Artifact predates the monitor: an explicit reason, no invented verdict.
        assert drift.get("reason")
        assert "binding" not in drift or drift.get("binding") is None
    else:
        # A rebuilt artifact carries the full measured (or honestly absent) block.
        assert DRIFT_REQUIRED_KEYS <= set(drift)


def test_measured_impact_summary_carries_level_drift_metadata(tmp_path) -> None:
    from kairos_api.server import _load_measured_impact_summary

    payload = {
        "method": "measured_detrended_pooled",
        "metadata": {
            "total_breaks_measured": 6,
            "level_drift": {
                "status": "measured",
                "weekly_levels": [{"week": 1, "n": 3, "mean_log_effect": -0.04, "se": 0.01}],
                "drift_per_week": 0.02,
                "drift_se": 0.0129,
                "binding": False,
                "criterion": "binding when |drift_per_week| > 2 x the half-width",
            },
        },
        "detail": {},
    }
    path = tmp_path / "tv_break_coefficients.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    summary = _load_measured_impact_summary(path)
    assert summary["metadata"]["level_drift"]["drift_per_week"] == 0.02
    assert summary["metadata"]["level_drift"]["binding"] is False
