"""Tests for kairos.model.interval_calibration (P2 interval honesty + P3
moderated variances) and its wiring in scripts/compute_measured_coefficients.py.

Fast tests are synthetic and seeded. The realdata tests run the rebuild script
into tmp paths only; the shipped artifact is never written.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kairos.model import interval_calibration as ic
from kairos.model.measure import (
    _cell_stats,
    _dersimonian_laird,
    _pooled_within_variance,
    channel_coefficients,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compute_measured_coefficients.py"
SHIPPED = ROOT / "models" / "tv_break_coefficients.json"

# The real artifact's per-cell break counts (36 cells, November 2024), so the
# synthetic coverage tests exercise the actual thin-to-thick cell profile.
REAL_NS = np.array([
    292, 255, 238, 224, 207, 159, 146, 134, 105, 100, 96, 67, 58, 51, 44, 28,
    27, 24, 24, 23, 23, 22, 21, 20, 18, 15, 14, 13, 13, 13, 12, 11, 10, 9, 8, 8,
])
TAU2_SHIPPED = 9.686792021678731e-05
S2_SHIPPED = 0.058196048399336
MU_LOG = -0.0399


def _synthetic_effects(seed: int = 7, tau2: float = 2e-4, s2: float = 0.05,
                       n_cells: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ns = rng.integers(5, 60, size=n_cells)
    theta = MU_LOG + rng.normal(0.0, math.sqrt(tau2), n_cells)
    rows = []
    for i in range(n_cells):
        for v in theta[i] + rng.normal(0.0, math.sqrt(s2), ns[i]):
            rows.append({"channel_name": f"c{i:02d}", "log_effect": float(v)})
    return pd.DataFrame(rows)


def _simulate_stats(rng, ns, tau2, sigma2_cells):
    """One synthetic month at the sufficient-statistic level, known truth."""
    m = len(ns)
    theta_true = MU_LOG + rng.normal(0.0, math.sqrt(tau2), m) if tau2 > 0 else np.full(m, MU_LOG)
    ybar = theta_true + rng.normal(0.0, 1.0, m) * np.sqrt(sigma2_cells / ns)
    rss = rng.chisquare(ns - 1) * sigma2_cells
    return theta_true, ybar, rss


def test_digamma_trigamma_match_scipy():
    scipy_special = pytest.importorskip("scipy.special")
    xs = np.concatenate([np.linspace(0.05, 5.0, 40), np.linspace(6.0, 80.0, 20)])
    assert np.max(np.abs(ic.digamma(xs) - scipy_special.digamma(xs))) < 1e-9
    assert np.max(np.abs(ic.trigamma(xs) - scipy_special.polygamma(1, xs))) < 1e-9


def test_trigamma_inverse_roundtrip():
    for y in (0.001, 0.05, 0.5, 2.0, 20.0):
        x = ic.trigamma_inverse(y)
        assert abs(float(ic.trigamma(x)) - y) < 1e-8 * max(1.0, y)
    assert ic.trigamma_inverse(0.0) == float("inf")


def test_moderated_variance_hand_fixture():
    # d0 = 4, s2_0 = 0.05; cells with df 4 and 1: exact posterior-mean algebra.
    prior = ic.PriorDf(df=4.0, s2=0.05, n_cells_used=2, method="moment")
    out = ic.moderate_variances(np.array([4.0, 1.0]), np.array([0.1, 0.02]), prior)
    assert out[0] == pytest.approx((4 * 0.05 + 4 * 0.1) / 8, abs=1e-15)   # 0.075
    assert out[1] == pytest.approx((4 * 0.05 + 1 * 0.02) / 5, abs=1e-15)  # 0.044
    # df = 0 cell takes the prior s2_0 outright.
    out0 = ic.moderate_variances(np.array([0.0]), np.array([np.nan]), prior)
    assert out0[0] == pytest.approx(0.05, abs=1e-15)
    # Infinite prior df: every cell gets s2_0; unavailable: unchanged.
    inf_prior = ic.PriorDf(df=float("inf"), s2=0.03, n_cells_used=5, method="infinite")
    assert np.allclose(ic.moderate_variances(np.array([4.0]), np.array([0.2]), inf_prior), 0.03)
    una = ic.PriorDf(df=0.0, s2=float("nan"), n_cells_used=1, method="unavailable")
    assert np.allclose(ic.moderate_variances(np.array([4.0]), np.array([0.2]), una), 0.2)


def test_prior_df_moment_recovery():
    # Simulate the limma model with known d0 = 8, s2_0 = 0.05, df = 10:
    # 1/sigma2_c ~ chi2(d0) / (d0 s2_0); s2_c | sigma2_c ~ sigma2_c chi2(df)/df.
    rng = np.random.default_rng(42)
    m, d0, s2_0, df = 2000, 8.0, 0.05, 10
    sigma2 = d0 * s2_0 / rng.chisquare(d0, size=m)
    s2 = sigma2 * rng.chisquare(df, size=m) / df
    prior = ic.estimate_prior_df(np.full(m, float(df)), s2)
    assert prior.method == "moment"
    assert 6.0 < prior.df < 10.5
    assert abs(prior.s2 / s2_0 - 1.0) < 0.2


def test_general_dl_matches_shipped_on_homoskedastic():
    eff = _synthetic_effects(seed=3)
    stats = _cell_stats(eff)
    s2p = _pooled_within_variance(stats)
    tau2_m, mu_m, sw_m = _dersimonian_laird(stats, s2p)
    ns = np.array([s[1] for s in stats], dtype=float)
    ybar = np.array([s[2] for s in stats])
    tau2_g, mu_g, sw_g = ic.dl_general(ybar, s2p / ns)
    assert tau2_g == pytest.approx(tau2_m, rel=1e-12, abs=1e-18)
    assert mu_g == pytest.approx(mu_m, rel=1e-12)
    assert sw_g == pytest.approx(sw_m, rel=1e-12)


def test_plugin_matches_channel_coefficients():
    eff = _synthetic_effects(seed=5)
    res = ic.calibrate_intervals(eff, bootstrap_b=50, seed=1)
    coeffs = channel_coefficients(eff)
    for i, name in enumerate(res.names):
        assert math.log1p(coeffs[name].raw_delta) == pytest.approx(res.theta[i], abs=1e-14)
        half = 0.5 * (math.log1p(coeffs[name].ci_high) - math.log1p(coeffs[name].ci_low))
        assert half == pytest.approx(1.96 * res.post_sd_plugin[i], abs=1e-14)


def test_bootstrap_determinism():
    eff = _synthetic_effects(seed=11)
    a = ic.calibrate_intervals(eff, bootstrap_b=300, seed=99)
    b = ic.calibrate_intervals(eff, bootstrap_b=300, seed=99)
    assert np.array_equal(a.mix_mean, b.mix_mean)
    assert np.array_equal(a.mix_sd, b.mix_sd)
    lo_a, hi_a = a.interval(0.95)
    lo_b, hi_b = b.interval(0.95)
    assert np.array_equal(lo_a, lo_b) and np.array_equal(hi_a, hi_b)
    c = ic.calibrate_intervals(eff, bootstrap_b=300, seed=100)
    assert not np.array_equal(a.mix_mean, c.mix_mean)


def test_apply_preserves_points_bitwise_and_widens():
    eff = _synthetic_effects(seed=7)
    coeffs = channel_coefficients(eff)
    res = ic.calibrate_intervals(eff, bootstrap_b=400, seed=2)
    out = ic.apply_calibrated_intervals(coeffs, res)
    widths_new, widths_old = [], []
    for name, c in coeffs.items():
        assert out[name].coefficient == c.coefficient  # bit-for-bit
        assert out[name].raw_delta == c.raw_delta
        assert out[name].n == c.n
        widths_new.append(math.log1p(out[name].ci_high) - math.log1p(out[name].ci_low))
        widths_old.append(math.log1p(c.ci_high) - math.log1p(c.ci_low))
    assert np.mean(np.array(widths_new) / np.array(widths_old)) > 1.0
    assert res.width_factor() > 1.0


def test_predictive_band_wider_than_ci_per_cell():
    eff = _synthetic_effects(seed=9)
    res = ic.calibrate_intervals(eff, bootstrap_b=300, seed=3)
    bands = ic.predictive_bands(res)
    lo, hi = res.interval(0.95)
    assert set(bands) == set(res.names)
    for i, name in enumerate(res.names):
        p_lo, p_hi = bands[name]
        assert p_lo < float(np.expm1(lo[i]))
        assert p_hi > float(np.expm1(hi[i]))


def test_calibration_unavailable_on_single_break_cells():
    eff = pd.DataFrame(
        {"channel_name": [f"c{i}" for i in range(6)], "log_effect": np.linspace(-0.1, 0.0, 6)}
    )
    with pytest.raises(ValueError, match="calibration unavailable"):
        ic.calibrate_intervals(eff, bootstrap_b=50, seed=1)


def test_calibrated_coverage_improves_on_seeded_synthetic():
    """The P2 acceptance check: at the real cell profile and the shipped
    tau2/s2 anchors, the plug-in 95% band undercovers (the panel measured
    0.826 over 500 reps) and the bootstrap mixture must recover most of the
    gap. Tolerances are loose because this is the fast 40-rep variant; the
    full referee numbers live in the adoption report."""
    rng = np.random.default_rng(20260707)
    reps, boot_b = 40, 250
    plug95, mix95, mix50 = [], [], []
    sigma2 = np.full(len(REAL_NS), S2_SHIPPED)
    for r in range(reps):
        theta_true, ybar, rss = _simulate_stats(rng, REAL_NS, TAU2_SHIPPED, sigma2)
        res = ic.calibrate_intervals(
            stats=(tuple(f"c{i}" for i in range(len(REAL_NS))), REAL_NS, ybar, rss),
            bootstrap_b=boot_b, seed=1000 + r,
        )
        err = np.abs(theta_true - res.theta)
        plug95.append(float(np.mean(err <= 1.96 * res.post_sd_plugin)))
        lo, hi = res.interval(0.95)
        mix95.append(float(np.mean((theta_true >= lo) & (theta_true <= hi))))
        lo5, hi5 = res.interval(0.50)
        mix50.append(float(np.mean((theta_true >= lo5) & (theta_true <= hi5))))
    naive, calibrated = float(np.mean(plug95)), float(np.mean(mix95))
    assert naive < 0.92  # the undercoverage the panel measured is present
    assert calibrated >= naive + 0.05  # the mixture recovers most of the gap
    assert calibrated >= 0.88
    assert 0.40 <= float(np.mean(mix50)) <= 0.75  # central mass not degenerate


def test_moderated_variances_fix_heteroskedastic_coverage():
    """The P3 acceptance check: with the real 23x per-cell variance span the
    pooled-s2 plug-in band collapses (the panel measured 0.67 at 95); the
    moderated fit plus bootstrap must restore most of the coverage AND cut
    the shrinkage-weight error."""
    rng = np.random.default_rng(20260708)
    m = len(REAL_NS)
    # Log-spaced per-cell variances spanning ~23x around the shipped s2,
    # shuffled so variance is not aligned with cell size.
    span = np.geomspace(0.0052, 0.1186, m)
    sigma2 = span[rng.permutation(m)]
    reps, boot_b = 30, 200
    plug95_naive, mix95_mod, berr_naive, berr_mod = [], [], [], []
    for r in range(reps):
        theta_true, ybar, rss = _simulate_stats(rng, REAL_NS, TAU2_SHIPPED, sigma2)
        stats = (tuple(f"c{i}" for i in range(m)), REAL_NS, ybar, rss)
        res_n = ic.calibrate_intervals(stats=stats, bootstrap_b=2, seed=2000 + r)
        res_m = ic.calibrate_intervals(
            stats=stats, bootstrap_b=boot_b, seed=2000 + r, moderated=True
        )
        assert res_m.moderated is True
        err_n = np.abs(theta_true - res_n.theta)
        plug95_naive.append(float(np.mean(err_n <= 1.96 * res_n.post_sd_plugin)))
        lo, hi = res_m.interval(0.95)
        mix95_mod.append(float(np.mean((theta_true >= lo) & (theta_true <= hi))))
        sig2bar = sigma2 / REAL_NS
        b_true = sig2bar / (sig2bar + TAU2_SHIPPED)
        for res, sink in ((res_n, berr_naive), (res_m, berr_mod)):
            vm = res.s2_cell / REAL_NS
            sink.append(float(np.mean(np.abs(vm / (vm + res.tau2) - b_true))))
    naive95 = float(np.mean(plug95_naive))
    moderated95 = float(np.mean(mix95_mod))
    assert naive95 < 0.85  # heteroskedastic collapse reproduced
    assert moderated95 >= naive95 + 0.10  # material coverage repair
    assert float(np.mean(berr_mod)) < float(np.mean(berr_naive))  # better weights


# ---------------------------------------------------------------------------
# Wiring (realdata): the rebuild script's OFF byte-equivalence and force-on
# semantics. Writes ONLY under tmp_path; the shipped artifact is never touched.
# ---------------------------------------------------------------------------

MY_METADATA_KEYS = {
    "interval_method", "moderated_variances", "bootstrap_B", "interval_seed",
    "prior_df", "width_factor_measured", "interval_calibration_reason",
}


def _run_rebuild(tmp_path: Path, *extra: str) -> dict:
    out = tmp_path / "coeffs.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUTF8"] = "1"
    for var in ("KAIROS_SERIES_LAYER", "KAIROS_COUNTERPROGRAMMING",
                "KAIROS_PLACEBO_CORRECTION", "KAIROS_INTERVAL_CALIBRATION",
                "KAIROS_MODERATED_VARIANCES"):
        env.pop(var, None)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), *extra],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.mark.realdata
def test_rebuild_default_reproduces_shipped_calibrated_artifact(tmp_path) -> None:
    """Calibration ships ON by default: a no-flag rebuild reproduces the shipped
    artifact (calibrated bands, predictive keys) with every point untouched."""
    fresh = _run_rebuild(tmp_path)
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert fresh["coefficients"] == shipped["coefficients"]
    for name, d in shipped["detail"].items():
        f = fresh["detail"][name]
        assert f["coefficient"] == d["coefficient"]
        assert f["raw_delta"] == d["raw_delta"]
        assert f["n"] == d["n"]
        assert f["ci_low"] == d["ci_low"] and f["ci_high"] == d["ci_high"]
        assert f["predictive_low"] == d["predictive_low"]
        assert f["predictive_high"] == d["predictive_high"]
    meta = fresh["metadata"]
    assert meta["interval_method"] == "bootstrap"
    assert meta["moderated_variances"] is False
    assert meta["bootstrap_B"] == ic.DEFAULT_BOOTSTRAP_B
    assert meta["interval_seed"] == ic.DEFAULT_CALIBRATION_SEED
    assert isinstance(meta["width_factor_measured"], float)
    assert 1.0 < meta["width_factor_measured"] < 4.0
    assert isinstance(meta["prior_df"], float) and meta["prior_df"] > 0
    assert "active by default" in meta["interval_calibration_reason"]


@pytest.mark.realdata
def test_rebuild_force_off_narrows_to_plugin_bands_points_untouched(tmp_path) -> None:
    """force-off is the diagnostic path: plug-in bands (strictly narrower than
    the shipped calibrated bands), no predictive keys, points untouched."""
    fresh = _run_rebuild(tmp_path, "--interval-calibration", "force-off")
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    assert fresh["coefficients"] == shipped["coefficients"]
    for name, d in shipped["detail"].items():
        f = fresh["detail"][name]
        assert f["coefficient"] == d["coefficient"]
        assert f["raw_delta"] == d["raw_delta"]
        # Shipped calibrated 95% band strictly wider than the plug-in band.
        w_shipped = math.log1p(d["ci_high"]) - math.log1p(d["ci_low"])
        w_plugin = math.log1p(f["ci_high"]) - math.log1p(f["ci_low"])
        assert w_shipped > w_plugin
        assert "predictive_low" not in f and "predictive_high" not in f
        # Shipped predictive keys stay separate from and wider than shipped ci.
        assert d["predictive_low"] < d["ci_low"]
        assert d["predictive_high"] > d["ci_high"]
    meta = fresh["metadata"]
    assert meta["interval_method"] == "naive"
    assert meta["moderated_variances"] is False
    assert meta["width_factor_measured"] > 1.0
    assert "force-off" in meta["interval_calibration_reason"]


@pytest.mark.realdata
def test_rebuild_moderated_force_on_reweights_points(tmp_path) -> None:
    fresh = _run_rebuild(tmp_path, "--moderated-variances", "force-on")
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    meta = fresh["metadata"]
    assert meta["moderated_variances"] is True
    assert meta["interval_method"] == "bootstrap"  # calibration is on by default
    assert isinstance(meta["prior_df"], float) and meta["prior_df"] > 0
    assert "moderated per-cell variances APPLIED" in meta["interval_calibration_reason"]
    # Re-weighted DL legitimately moves points: same cells, different values.
    assert set(fresh["coefficients"]) == set(shipped["coefficients"])
    assert fresh["coefficients"] != shipped["coefficients"]
