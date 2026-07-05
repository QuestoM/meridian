"""Credible-interval coverage and calibration of the Kairos retention model.

Temporal holdout on the real month (November 2024, 30 days of measurable
breaks): TRAIN on the first 23 days, EVALUATE on the last 7. Per-cell
predictions and credible intervals come from THE SHIPPED pipeline
(:func:`kairos.model.measure.channel_coefficients` run on the training split);
this script only inverts the exp transform to recover the posterior sd, and
rescales z for the 50/80/95 nominal levels.

What is measured
----------------
1. COVERAGE of the shipped intervals, per nominal level (50/80/95):
   a. of realized held-out CELL MEANS -- naive (shipped parameter interval as
      published) and noise-adjusted (interval widened by the held-out mean's
      own sampling noise s2/n_test, which is the fair test of the latent-mean
      interval since a 7-day cell mean is itself a noisy estimate);
   b. of INDIVIDUAL held-out break outcomes -- against the shipped interval
      (which is a mean-level interval, so this quantifies how far the shipped
      band understates single-break risk) and against a proper predictive
      interval (posterior variance + within-cell variance).
   Binomial Wilson CIs on every coverage estimate; cell-cluster bootstrap CIs
   for the individual-outcome coverages (breaks within a cell share the cell
   effect, so a plain binomial CI would overstate precision).
   For each naive coverage we also compute the MODEL-IMPLIED value (what
   coverage the model itself expects given the target's sampling noise), so
   under- or over-coverage can be attributed to the interval rather than to
   the noisy target.

2. CALIBRATION: predicted vs realized retention effect on held-out breaks, in
   the pipeline's log-effect space. Decile-binned curve, per-break OLS slope
   and intercept with cell-cluster bootstrap CIs (predictions are constant
   within a cell, so clustering is mandatory), a cell-level weighted
   regression cross-check, an out-of-sample R^2 vs the global-constant
   baseline, a shrinkage-multiplier sweep (RMSE-optimal multiplier m on the
   EB deviations; m == 1 is the shipped prediction, m < 1 means shrink more),
   and the extra shrinkage translated into an implied between-cell tau^2.

3. SENSITIVITY: a fully leak-free variant where the training coefficients are
   measured from train-window frames only and the held-out outcomes are
   measured with the test week's own dayparts (own-week detrend baseline), so
   no minute of held-out audience touches the training side. The primary
   analysis above shares the full-month detrend baseline exactly as the
   shipped rebuild would (the baseline is a nuisance curve, ~25% of which is
   test-week data; the sensitivity bounds what that sharing changes).

Deterministic: numpy default_rng(20260705) for the bootstrap only; everything
else is closed-form on the real data. Runtime ~40 s. Results are written to
scripts/validation/out/coverage_holdout.json.

Run:  /Users/home/.venvs/meridian/bin/python scripts/validation/coverage_holdout.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _valcommon import (  # noqa: E402
    TRAIN_END,
    Z_LEVELS,
    coeff_log_params,
    env_snapshot,
    load_effects_full,
    load_frames,
    normal_cdf,
    print_snapshot,
    wilson_interval,
    write_results,
)

BOOT_REPS = 4000
BOOT_SEED = 20260705


def split_effects(effects: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mask = effects["break_start"] <= TRAIN_END
    return effects[mask].reset_index(drop=True), effects[~mask].reset_index(drop=True)


def fit_train(effects_train: pd.DataFrame):
    """Run the SHIPPED pooling on the training split and unpack log-space params."""
    from kairos.model.measure import between_cell_variance, channel_coefficients

    coeffs = channel_coefficients(effects_train)
    diag = between_cell_variance(effects_train)
    cells = {}
    for name, c in coeffs.items():
        theta, post_sd = coeff_log_params(c)
        cells[name] = {"theta": theta, "post_sd": post_sd, "n_train": c.n}
    return cells, diag


def coverage_tables(cells: dict, diag: dict, test: pd.DataFrame) -> dict:
    """Coverage of cell means and individual outcomes, per nominal level."""
    s2 = float(diag["pooled_within_var"])

    per_cell = []
    for name, grp in test.groupby("channel_name"):
        if name not in cells:
            continue
        y = grp["log_effect"].to_numpy()
        per_cell.append(
            {
                "cell": str(name),
                "theta": cells[name]["theta"],
                "post_sd": cells[name]["post_sd"],
                "n_train": cells[name]["n_train"],
                "n_test": int(len(y)),
                "test_mean": float(np.mean(y)),
            }
        )
    cell_frame = pd.DataFrame(per_cell)

    # Individual outcomes joined to their cell's parameters.
    test_in = test[test["channel_name"].isin(cells.keys())].copy()
    test_in["theta"] = test_in["channel_name"].map(lambda c: cells[c]["theta"])
    test_in["post_sd"] = test_in["channel_name"].map(lambda c: cells[c]["post_sd"])

    out = {"n_cells": int(len(cell_frame)), "n_individual": int(len(test_in)), "levels": {}}
    rng = np.random.default_rng(BOOT_SEED)
    cell_ids = test_in["channel_name"].to_numpy()
    unique_cells = np.unique(cell_ids)

    for level, z in Z_LEVELS.items():
        err_cell = np.abs(cell_frame["test_mean"] - cell_frame["theta"]).to_numpy()
        sd_param = cell_frame["post_sd"].to_numpy()
        sd_adj = np.sqrt(sd_param**2 + s2 / cell_frame["n_test"].to_numpy())

        cover_naive = err_cell <= z * sd_param
        cover_adj = err_cell <= z * sd_adj
        # What the model itself expects the naive number to be, given that the
        # target (a finite test-week mean) is noisy: realized - theta ~
        # N(0, post_sd^2 + s2/n_test) under the model, but the band is z*post_sd.
        implied_naive_cell = float(
            np.mean(2.0 * np.asarray(normal_cdf(z * sd_param / sd_adj)) - 1.0)
        )

        err_ind = np.abs(test_in["log_effect"] - test_in["theta"]).to_numpy()
        sd_i = test_in["post_sd"].to_numpy()
        sd_pred = np.sqrt(sd_i**2 + s2)
        cover_ind_naive = err_ind <= z * sd_i
        cover_ind_pred = err_ind <= z * sd_pred
        implied_naive_ind = float(np.mean(2.0 * np.asarray(normal_cdf(z * sd_i / sd_pred)) - 1.0))

        # Cell-cluster bootstrap for the individual coverages.
        boot_naive, boot_pred = [], []
        for _ in range(BOOT_REPS):
            take = rng.choice(unique_cells, size=len(unique_cells), replace=True)
            idx = np.concatenate([np.flatnonzero(cell_ids == c) for c in take])
            boot_naive.append(float(np.mean(cover_ind_naive[idx])))
            boot_pred.append(float(np.mean(cover_ind_pred[idx])))

        def pct(v: list[float]) -> tuple[float, float]:
            return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))

        k_naive, k_adj = int(cover_naive.sum()), int(cover_adj.sum())
        n_cells = len(cover_naive)
        out["levels"][f"{level:.2f}"] = {
            "cell_naive": {
                "coverage": float(np.mean(cover_naive)),
                "k": k_naive, "n": n_cells,
                "wilson95": wilson_interval(k_naive, n_cells),
                "model_implied": implied_naive_cell,
            },
            "cell_noise_adjusted": {
                "coverage": float(np.mean(cover_adj)),
                "k": k_adj, "n": n_cells,
                "wilson95": wilson_interval(k_adj, n_cells),
            },
            "individual_shipped_interval": {
                "coverage": float(np.mean(cover_ind_naive)),
                "k": int(cover_ind_naive.sum()), "n": int(len(cover_ind_naive)),
                "cluster_boot95": pct(boot_naive),
                "model_implied": implied_naive_ind,
            },
            "individual_predictive_interval": {
                "coverage": float(np.mean(cover_ind_pred)),
                "k": int(cover_ind_pred.sum()), "n": int(len(cover_ind_pred)),
                "cluster_boot95": pct(boot_pred),
            },
        }
    out["cell_detail"] = cell_frame.to_dict(orient="records")
    return out


def _ols(xv: np.ndarray, yv: np.ndarray) -> tuple[float, float]:
    """Slope and intercept of y on x; (nan, nan) when x has zero variance."""
    vx = float(np.var(xv))
    if vx <= 1e-18:
        return float("nan"), float("nan")
    b = float(np.cov(xv, yv, bias=True)[0, 1] / vx)
    a = float(np.mean(yv) - b * np.mean(xv))
    return b, a


def _cluster_boot_slope(
    x: np.ndarray, y: np.ndarray, cell_ids: np.ndarray, seed: int
) -> dict:
    """OLS slope/intercept with cell-cluster bootstrap CIs (x constant within cell)."""
    slope, intercept = _ols(x, y)
    rng = np.random.default_rng(seed)
    unique_cells = np.unique(cell_ids)
    cell_index = {c: np.flatnonzero(cell_ids == c) for c in unique_cells}
    boot_slopes, boot_intercepts = [], []
    for _ in range(BOOT_REPS):
        take = rng.choice(unique_cells, size=len(unique_cells), replace=True)
        idx = np.concatenate([cell_index[c] for c in take])
        b, a = _ols(x[idx], y[idx])
        if np.isfinite(b):
            boot_slopes.append(b)
            boot_intercepts.append(a)
    def ci(v):
        return (
            (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
            if v else (float("nan"), float("nan"))
        )
    return {
        "slope": slope,
        "slope_ci95_cluster_boot": ci(boot_slopes),
        "intercept": intercept,
        "intercept_ci95_cluster_boot": ci(boot_intercepts),
        "boot_valid_frac": len(boot_slopes) / BOOT_REPS,
    }


def calibration(cells: dict, diag: dict, train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Predicted vs realized on held-out breaks, slope/intercept, extra shrinkage.

    Two predictors are calibrated:
      * the SHIPPED EB prediction theta_i (post-shrinkage). When the training
        split's tau2-hat is 0 every theta_i equals the grand mean, the predictor
        is constant and its slope is undefined -- that degeneracy is reported,
        not papered over.
      * the RAW (unpooled) train cell means. Their held-out slope is the direct
        empirical estimate of how much of a raw cell deviation is real, i.e. the
        shrinkage factor the data wants; comparing it with the EB (1 - B_i)
        answers "should we shrink more or less than DL does".
    """
    test_in = test[test["channel_name"].isin(cells.keys())].copy()
    x_eb = test_in["channel_name"].map(lambda c: cells[c]["theta"]).to_numpy(dtype=float)
    y = test_in["log_effect"].to_numpy(dtype=float)
    cell_ids = test_in["channel_name"].to_numpy()

    from kairos.model.measure import _cell_stats, _dersimonian_laird, _pooled_within_variance

    stats = _cell_stats(train)
    pooled_within = _pooled_within_variance(stats)
    tau2_train, mu_train, _sw = _dersimonian_laird(stats, pooled_within)

    raw_means = train.groupby("channel_name")["log_effect"].mean().to_dict()
    x_raw = np.array([raw_means[str(c)] for c in cell_ids], dtype=float)
    n_train_map = {name: n for name, n, _m, _r in stats}
    n_i = np.array([n_train_map[str(c)] for c in cell_ids], dtype=float)

    eb_fit = _cluster_boot_slope(x_eb, y, cell_ids, BOOT_SEED + 1)
    raw_fit = _cluster_boot_slope(x_raw, y, cell_ids, BOOT_SEED + 2)

    # Decile-binned curve over held-out breaks, on the raw-mean predictor (the
    # EB predictor can be constant); ties broken by rank so bins stay equal.
    order = np.argsort(x_raw, kind="stable")
    bins = np.array_split(order, 10)
    decile_rows = []
    for i, idx in enumerate(bins):
        decile_rows.append(
            {
                "decile": i + 1,
                "n": int(len(idx)),
                "pred_mean_raw": float(np.mean(x_raw[idx])),
                "pred_mean_eb": float(np.mean(x_eb[idx])),
                "realized_mean": float(np.mean(y[idx])),
                "realized_se": float(np.std(y[idx], ddof=1) / np.sqrt(len(idx))),
            }
        )

    # Out-of-sample skill vs the global constant (train grand mean).
    grand = float(np.mean(train["log_effect"]))
    sse_grand = float(np.sum((y - grand) ** 2))
    r2_eb = 1.0 - float(np.sum((y - x_eb) ** 2)) / sse_grand
    r2_raw = 1.0 - float(np.sum((y - x_raw) ** 2)) / sse_grand

    # Shrinkage-multiplier sweep on the RAW deviations: pred_m = mu + m*(ybar - mu).
    # m == 0 is the global constant, m == 1 is no pooling; the EB prediction uses
    # m_i = (1 - B_i) per cell. The RMSE-optimal m is the single shrinkage factor
    # the held-out week would have wanted.
    sweep = []
    for m in np.round(np.arange(0.0, 1.51, 0.025), 3):
        pred = mu_train + m * (x_raw - mu_train)
        sweep.append({"m": float(m), "rmse": float(np.sqrt(np.mean((y - pred) ** 2)))})
    best = min(sweep, key=lambda r: r["rmse"])

    # EB's own per-cell keep-factors (1 - B_i) under the train tau2 and, for
    # context, under the full-month artifact tau2.
    def keep_factors(tau2: float) -> np.ndarray:
        return (n_i * tau2) / (pooled_within + n_i * tau2)

    keep_train = keep_factors(tau2_train)
    tau2_full_artifact = 9.686792021678731e-05  # shipped artifact metadata
    keep_full = keep_factors(tau2_full_artifact)

    # Method-of-moments tau2 implied by the raw-mean calibration slope:
    # slope_theory(tau2) = tau2 / (tau2 + E_w[s2/n_i]) with E_w over test breaks.
    vbar = float(np.mean(pooled_within / n_i))
    b_raw = raw_fit["slope"]
    if np.isfinite(b_raw) and 0.0 < b_raw < 1.0:
        tau2_implied_raw = b_raw / (1.0 - b_raw) * vbar
    elif np.isfinite(b_raw) and b_raw <= 0.0:
        tau2_implied_raw = 0.0
    else:
        tau2_implied_raw = float("nan")

    # Residual shape: is the within-cell noise normal? (feeds the predictive-
    # interval over/undercoverage diagnosis).
    resid = train["log_effect"] - train.groupby("channel_name")["log_effect"].transform("mean")
    r = resid.to_numpy()
    m2 = float(np.mean(r**2)); m3 = float(np.mean(r**3)); m4 = float(np.mean(r**4))
    skew = m3 / m2**1.5
    ex_kurt = m4 / m2**2 - 3.0

    # Weekly grand means: the drift the intervals must survive.
    wk = test_in.copy()
    full = pd.concat([train, test], ignore_index=True)
    full["week"] = ((full["break_start"] - full["break_start"].min()).dt.days // 7) + 1
    weekly = [
        {
            "week": int(wnum),
            "n": int(len(g)),
            "mean": float(g["log_effect"].mean()),
            "se": float(g["log_effect"].std(ddof=1) / np.sqrt(len(g))),
        }
        for wnum, g in full.groupby("week")
    ]

    return {
        "n_test_breaks": int(len(y)),
        "eb_predictor": eb_fit
        | {
            "degenerate_constant_predictor": bool(np.var(x_eb) <= 1e-18),
            "note": (
                "tau2-hat on the training split is 0, so every EB prediction equals "
                "the grand mean and the slope is undefined"
                if np.var(x_eb) <= 1e-18
                else "EB predictions vary across cells"
            ),
        },
        "raw_predictor": raw_fit,
        "deciles": decile_rows,
        "r2_oos_eb_vs_global_constant": r2_eb,
        "r2_oos_raw_unpooled_means": r2_raw,
        "train_grand_mean": grand,
        "test_grand_mean": float(np.mean(y)),
        "test_grand_mean_se": float(np.std(y, ddof=1) / np.sqrt(len(y))),
        "level_shift_train_to_test": float(np.mean(y) - grand),
        "weekly_grand_means": weekly,
        "shrinkage_sweep_raw": sweep,
        "rmse_optimal_multiplier_raw": best,
        "eb_keep_factor_train": {
            "tau2": float(tau2_train),
            "mean": float(np.mean(keep_train)),
            "median": float(np.median(keep_train)),
        },
        "eb_keep_factor_full_month_artifact": {
            "tau2": tau2_full_artifact,
            "mean": float(np.mean(keep_full)),
            "median": float(np.median(keep_full)),
        },
        "tau2_implied_by_raw_slope": tau2_implied_raw,
        "pseudo_count_implied_by_raw_slope": (
            float(pooled_within / tau2_implied_raw)
            if np.isfinite(tau2_implied_raw) and tau2_implied_raw > 0
            else None
        ),
        "train_residual_shape": {"skew": skew, "excess_kurtosis": ex_kurt, "sd": float(np.sqrt(m2))},
    }


def tau2_stability(effects: pd.DataFrame) -> list[dict]:
    """DL tau2-hat over the full month and sliding sub-windows: how fragile is
    the learned between-cell variance at the current sample size?  Uses the
    ACTUAL pipeline estimators on subsets of the shipped full-month effects."""
    from kairos.model.measure import _cell_stats, _dersimonian_laird, _pooled_within_variance

    windows = [
        ("full month (days 1-30)", None, None),
        ("days 1-23 (train split)", 1, 23),
        ("days 8-30", 8, 30),
        ("days 1-15", 1, 15),
        ("days 16-30", 16, 30),
        ("days 4-26", 4, 26),
    ]
    day = effects["break_start"].dt.day
    rows = []
    for label, lo, hi in windows:
        sub = effects if lo is None else effects[(day >= lo) & (day <= hi)]
        stats = _cell_stats(sub)
        pw = _pooled_within_variance(stats)
        tau2, mu, _sw = _dersimonian_laird(stats, pw)
        rows.append(
            {
                "window": label,
                "n_breaks": int(len(sub)),
                "n_cells": int(len(stats)),
                "tau2_hat": float(tau2),
                "mu_hat": float(mu),
                "pooled_within": float(pw),
                "pseudo_count": float(pw / tau2) if tau2 > 0 else None,
            }
        )
    return rows


def leak_free_sensitivity(frames) -> dict:
    """Train coefficients from train-window frames only; test outcomes measured
    with the test week's own dayparts (own-week detrend baseline)."""
    from kairos.model.measure import break_effects, between_cell_variance, channel_coefficients

    spots, programmes, dayparts, classifier = frames
    cut = TRAIN_END
    spots_tr = spots[spots["air_dt"] <= cut]
    spots_te = spots[spots["air_dt"] > cut]
    day_tr = dayparts[dayparts["date"] <= cut.normalize()]
    day_te = dayparts[dayparts["date"] > cut.normalize()]

    eff_tr = break_effects(spots_tr, programmes, day_tr, classifier)
    eff_te = break_effects(spots_te, programmes, day_te, classifier)

    coeffs = channel_coefficients(eff_tr)
    diag = between_cell_variance(eff_tr)
    cells = {}
    for name, c in coeffs.items():
        theta, post_sd = coeff_log_params(c)
        cells[name] = {"theta": theta, "post_sd": post_sd, "n_train": c.n}

    tables = coverage_tables(cells, diag, eff_te)
    calib = calibration(cells, diag, eff_tr, eff_te)
    return {
        "n_train_breaks": int(len(eff_tr)),
        "n_test_breaks": int(len(eff_te)),
        "tau2_train": float(diag["tau2"]),
        "pooled_within_train": float(diag["pooled_within_var"]),
        "coverage": {
            lvl: {
                "cell_naive": tables["levels"][lvl]["cell_naive"]["coverage"],
                "cell_noise_adjusted": tables["levels"][lvl]["cell_noise_adjusted"]["coverage"],
                "individual_shipped": tables["levels"][lvl]["individual_shipped_interval"]["coverage"],
                "individual_predictive": tables["levels"][lvl]["individual_predictive_interval"]["coverage"],
            }
            for lvl in tables["levels"]
        },
        "calibration_slope_eb": calib["eb_predictor"]["slope"],
        "calibration_slope_eb_ci95": calib["eb_predictor"]["slope_ci95_cluster_boot"],
        "calibration_slope_raw": calib["raw_predictor"]["slope"],
        "calibration_slope_raw_ci95": calib["raw_predictor"]["slope_ci95_cluster_boot"],
        "rmse_optimal_multiplier_raw": calib["rmse_optimal_multiplier_raw"],
        "level_shift_train_to_test": calib["level_shift_train_to_test"],
    }


def main() -> None:
    t0 = time.time()
    snapshot = env_snapshot()
    print_snapshot(snapshot)

    frames = load_frames()
    effects = load_effects_full(frames)
    train, test = split_effects(effects)
    print(
        f"[split] train {len(train)} breaks ({train['break_start'].min().date()} .. "
        f"{train['break_start'].max().date()}), test {len(test)} breaks "
        f"({test['break_start'].min().date()} .. {test['break_start'].max().date()})"
    )

    cells, diag = fit_train(train)
    print(
        f"[train fit] {len(cells)} cells, tau2={diag['tau2']:.6g}, "
        f"pooled within-var={diag['pooled_within_var']:.6g}, method={diag['method']}"
    )

    tables = coverage_tables(cells, diag, test)
    print("\n=== COVERAGE (primary split, shared full-month detrend baseline) ===")
    print(f"{'nominal':>8} | {'cell naive':>22} | {'cell adj.':>10} | "
          f"{'indiv shipped':>22} | {'indiv predictive':>22}")
    for lvl, row in tables["levels"].items():
        cn = row["cell_naive"]
        ca = row["cell_noise_adjusted"]
        i1 = row["individual_shipped_interval"]
        i2 = row["individual_predictive_interval"]
        print(
            f"{lvl:>8} | {cn['coverage']:.3f} ({cn['k']}/{cn['n']}) impl {cn['model_implied']:.3f}"
            f" | {ca['coverage']:>10.3f} | {i1['coverage']:.3f} impl {i1['model_implied']:.3f}"
            f"        | {i2['coverage']:.3f} [{i2['cluster_boot95'][0]:.3f},{i2['cluster_boot95'][1]:.3f}]"
        )

    calib = calibration(cells, diag, train, test)
    print("\n=== CALIBRATION (held-out breaks, log-effect space) ===")
    eb = calib["eb_predictor"]
    raw = calib["raw_predictor"]
    if eb["degenerate_constant_predictor"]:
        print("EB predictor: DEGENERATE on this split (train tau2-hat = 0 -> every cell "
              "predicts the grand mean; slope undefined)")
    else:
        print(f"EB predictor slope {eb['slope']:.3f}  CI95 {eb['slope_ci95_cluster_boot']}")
    print(
        f"RAW cell-mean predictor slope {raw['slope']:.3f}  CI95 {raw['slope_ci95_cluster_boot']}, "
        f"intercept {raw['intercept']:.4f}"
    )
    print(
        f"R2 oos: EB {calib['r2_oos_eb_vs_global_constant']:+.5f}, "
        f"raw unpooled means {calib['r2_oos_raw_unpooled_means']:+.5f}"
    )
    print(
        f"level shift train->test {calib['level_shift_train_to_test']:+.5f} "
        f"(test mean se {calib['test_grand_mean_se']:.5f})"
    )
    print(
        f"RMSE-optimal shrinkage multiplier on raw deviations m*={calib['rmse_optimal_multiplier_raw']['m']}"
        f" (EB keep-factor: train mean {calib['eb_keep_factor_train']['mean']:.3f}, "
        f"full-month artifact mean {calib['eb_keep_factor_full_month_artifact']['mean']:.3f})"
    )
    print(
        f"tau2 implied by raw slope {calib['tau2_implied_by_raw_slope']:.3g} "
        f"(train DL tau2 {calib['eb_keep_factor_train']['tau2']:.3g}, "
        f"artifact tau2 {calib['eb_keep_factor_full_month_artifact']['tau2']:.3g})"
    )
    print(
        f"train residual shape: sd {calib['train_residual_shape']['sd']:.4f}, "
        f"skew {calib['train_residual_shape']['skew']:+.2f}, "
        f"excess kurtosis {calib['train_residual_shape']['excess_kurtosis']:+.2f}"
    )

    stability = tau2_stability(effects)
    print("\n=== TAU2 STABILITY (actual DL estimator on sub-windows) ===")
    for row in stability:
        pc = f"{row['pseudo_count']:.0f}" if row["pseudo_count"] else "inf (full pooling)"
        print(
            f"  {row['window']:<26} n={row['n_breaks']:>5}  tau2={row['tau2_hat']:.3e}  "
            f"pseudo-count={pc}"
        )

    sens = leak_free_sensitivity(frames)
    print("\n=== SENSITIVITY (leak-free: train-only fit, own-week test baseline) ===")
    print(
        f"train {sens['n_train_breaks']} / test {sens['n_test_breaks']} breaks; "
        f"raw slope {sens['calibration_slope_raw']:.3f} CI95 {sens['calibration_slope_raw_ci95']}; "
        f"level shift {sens['level_shift_train_to_test']:+.5f}"
    )
    for lvl, row in sens["coverage"].items():
        print(
            f"  {lvl}: cell naive {row['cell_naive']:.3f} adj {row['cell_noise_adjusted']:.3f} | "
            f"indiv shipped {row['individual_shipped']:.3f} predictive {row['individual_predictive']:.3f}"
        )

    write_results(
        "coverage_holdout.json",
        {
            "env": snapshot,
            "split": {
                "train_end": str(TRAIN_END),
                "n_train": int(len(train)),
                "n_test": int(len(test)),
            },
            "train_diag": {k: diag[k] for k in ("tau2", "pooled_within_var", "n_cells", "method")},
            "coverage": tables,
            "calibration": calib,
            "tau2_stability": stability,
            "leak_free_sensitivity": sens,
            "runtime_seconds": round(time.time() - t0, 1),
        },
    )
    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
