"""Parameter-recovery simulation of the DerSimonian-Laird EB pooling pipeline.

Simulates the EXACT hierarchical setup the Kairos retention model assumes --
36 cells with the REAL per-cell break counts from the shipped artifact, known
true between-cell variance tau^2 and true cell effects drawn from it, within-
cell noise matched to the real pooled within-variance -- and pushes every
replication through THE ACTUAL pipeline code
(:func:`kairos.model.measure.channel_coefficients`, plus the same
``_cell_stats`` / ``_pooled_within_variance`` / ``_dersimonian_laird``
estimators it calls). Nothing is reimplemented; the pipeline is treated as a
black box whose outputs are scored against the known truth.

Scenarios
---------
* GRID (normal, homoskedastic noise -- the model's own assumptions):
  tau^2_true in {0, tau2_artifact, 4 x tau2_artifact} crossed with sample
  scale in {1x (today), 12x, 24x (the two-year data drop)}.
* EMPIRICAL-TAILS variant: within-cell noise resampled from the REAL pooled
  residuals of the full-month measurement (excess kurtosis ~ +7.9), rescaled
  to the same s^2, at tau2_artifact -- does heavy-tailed real noise break the
  normal-theory pooling?
* HETEROSKEDASTIC variant: each cell keeps its REAL within-cell variance
  (0.021 .. 0.135 across cells) while the pipeline assumes one pooled s^2 --
  the misspecification actually present in the data.

Metrics per scenario (500 replications, seeded)
-----------------------------------------------
* tau2-hat: mean, median, sd, relative bias, P(tau2-hat == 0) (the DL floor).
* Coverage of the shipped per-cell credible intervals at nominal 50/80/95
  against the TRUE cell effects, with replication-clustered MC error.
* Oracle-coverage decomposition: the same interval formula fed the TRUE tau^2
  and mu, isolating how much miscoverage is due to ESTIMATING tau^2.
* Shrinkage-weight error: mean |B-hat_i - B-true_i| and mean signed error.
* EB point-estimate RMSE vs the raw (unpooled) cell means -- does the pooling
  actually reduce error, and by how much at each scale.
* Mean 95% interval half-width -- does the interval sharpen ~1/sqrt(scale).

Deterministic: numpy default_rng(20260706) master seed, per-replication
spawned seeds. Runtime ~4-6 minutes (500 reps; the rep count is a knob).

Run:  /Users/home/.venvs/meridian/bin/python scripts/validation/parameter_recovery.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _valcommon import (  # noqa: E402
    ARTIFACT,
    Z95,
    Z_LEVELS,
    env_snapshot,
    load_effects_full,
    print_snapshot,
    write_results,
)

MASTER_SEED = 20260706
N_REPS = 500

# Real-data anchors (verified against the shipped artifact at runtime).
TAU2_ARTIFACT = 9.686792021678731e-05
S2_ARTIFACT = 0.058196048399336
MU_LOG = -0.0399  # grand mean log effect; location does not affect recovery metrics


def real_cell_ns() -> tuple[list[str], np.ndarray]:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    detail = payload["detail"]
    names = sorted(detail.keys())
    ns = np.array([int(detail[n]["n"]) for n in names], dtype=int)
    return names, ns


def real_residual_pool_and_cell_vars() -> tuple[np.ndarray, dict[str, float]]:
    """Centered per-cell residuals of the REAL full-month measurement, rescaled
    to the artifact s^2 (for the empirical-tails variant), plus each cell's own
    real within-cell variance (for the heteroskedastic variant)."""
    effects = load_effects_full()
    resid = (
        effects["log_effect"]
        - effects.groupby("channel_name")["log_effect"].transform("mean")
    ).to_numpy()
    pool = resid * np.sqrt(S2_ARTIFACT / np.var(resid))
    cell_vars = effects.groupby("channel_name")["log_effect"].var(ddof=1).to_dict()
    return pool, {str(k): float(v) for k, v in cell_vars.items()}


def one_replication(
    rng: np.random.Generator,
    names: list[str],
    ns: np.ndarray,
    tau2_true: float,
    noise: str,
    pool: np.ndarray | None,
    cell_sigma2: np.ndarray | None,
) -> dict:
    """Simulate one month-equivalent, push through the ACTUAL pipeline, score it."""
    from kairos.model.measure import (
        _cell_stats,
        _dersimonian_laird,
        _pooled_within_variance,
        channel_coefficients,
    )

    m = len(names)
    theta_true = MU_LOG + rng.normal(0.0, np.sqrt(tau2_true), size=m)

    total = int(ns.sum())
    reps_theta = np.repeat(theta_true, ns)
    if noise == "normal":
        y = reps_theta + rng.normal(0.0, np.sqrt(S2_ARTIFACT), size=total)
        sigma2_true = np.full(m, S2_ARTIFACT)
    elif noise == "empirical":
        y = reps_theta + rng.choice(pool, size=total, replace=True)
        sigma2_true = np.full(m, S2_ARTIFACT)
    elif noise == "hetero":
        sd = np.repeat(np.sqrt(cell_sigma2), ns)
        y = reps_theta + rng.normal(0.0, 1.0, size=total) * sd
        sigma2_true = cell_sigma2.copy()
    else:
        raise ValueError(noise)

    effects = pd.DataFrame({"channel_name": np.repeat(names, ns), "log_effect": y})

    # THE ACTUAL PIPELINE. channel_coefficients internally reruns the same
    # _cell_stats -> _pooled_within_variance -> _dersimonian_laird path; we call
    # the estimators once more explicitly only to record tau2-hat and mu-hat.
    coeffs = channel_coefficients(effects)
    stats = _cell_stats(effects)
    pw_hat = _pooled_within_variance(stats)
    tau2_hat, mu_hat, _sw = _dersimonian_laird(stats, pw_hat)

    theta_hat = np.empty(m)
    half95 = np.empty(m)
    for j, name in enumerate(names):
        c = coeffs[name]
        theta_hat[j] = np.log1p(c.raw_delta)
        half95[j] = 0.5 * (np.log1p(c.ci_high) - np.log1p(c.ci_low))

    ybar = np.array([dict((s[0], s[2]) for s in stats)[n] for n in names])

    # Oracle interval: same normal-normal formula with TRUE tau2, mu and sigma2
    # (the truth baseline the pipeline is scored against; diagnostic only).
    sig2_bar = sigma2_true / ns
    b_true = sig2_bar / (sig2_bar + tau2_true) if tau2_true > 0 else np.ones(m)
    theta_oracle = MU_LOG + (1.0 - b_true) * (ybar - MU_LOG)
    var_oracle = (1.0 - b_true) * sig2_bar

    sig2_bar_hat = pw_hat / ns
    b_hat = sig2_bar_hat / (sig2_bar_hat + tau2_hat) if tau2_hat > 0 else np.ones(m)

    err = np.abs(theta_true - theta_hat)
    err_oracle = np.abs(theta_true - theta_oracle)
    cover = {}
    cover_oracle = {}
    for level, z in Z_LEVELS.items():
        cover[level] = float(np.mean(err <= (z / Z95) * half95))
        cover_oracle[level] = float(np.mean(err_oracle <= z * np.sqrt(var_oracle)))

    # |error| in units of the shipped 95% half-width: the L-quantile of this,
    # divided by z_L/z95, is the multiplicative widening the shipped interval
    # needs to reach nominal level L.
    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = np.where(half95 > 0, err / half95, np.inf)

    return {
        "ratios": ratios,
        "tau2_hat": float(tau2_hat),
        "mu_hat": float(mu_hat),
        "pw_hat": float(pw_hat),
        "cover": cover,
        "cover_oracle": cover_oracle,
        "b_abs_err": float(np.mean(np.abs(b_hat - b_true))),
        "b_signed_err": float(np.mean(b_hat - b_true)),
        "rmse_eb": float(np.sqrt(np.mean((theta_hat - theta_true) ** 2))),
        "rmse_raw": float(np.sqrt(np.mean((ybar - theta_true) ** 2))),
        "mean_half95": float(np.mean(half95)),
    }


def run_scenario(
    label: str,
    names: list[str],
    ns_base: np.ndarray,
    scale: int,
    tau2_true: float,
    noise: str,
    pool: np.ndarray | None,
    cell_sigma2: np.ndarray | None,
    n_reps: int,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    ns = ns_base * scale
    rows = [
        one_replication(rng, names, ns, tau2_true, noise, pool, cell_sigma2)
        for _ in range(n_reps)
    ]
    tau2s = np.array([r["tau2_hat"] for r in rows])
    out = {
        "label": label,
        "noise": noise,
        "scale": scale,
        "tau2_true": tau2_true,
        "n_reps": n_reps,
        "total_breaks": int(ns.sum()),
        "tau2_hat_mean": float(np.mean(tau2s)),
        "tau2_hat_median": float(np.median(tau2s)),
        "tau2_hat_sd": float(np.std(tau2s, ddof=1)),
        "tau2_hat_p10": float(np.percentile(tau2s, 10)),
        "tau2_hat_p90": float(np.percentile(tau2s, 90)),
        "prob_tau2_zero": float(np.mean(tau2s == 0.0)),
        "tau2_bias": float(np.mean(tau2s) - tau2_true),
        "tau2_rel_bias": (
            float((np.mean(tau2s) - tau2_true) / tau2_true) if tau2_true > 0 else None
        ),
        "b_abs_err_mean": float(np.mean([r["b_abs_err"] for r in rows])),
        "b_signed_err_mean": float(np.mean([r["b_signed_err"] for r in rows])),
        "rmse_eb_mean": float(np.mean([r["rmse_eb"] for r in rows])),
        "rmse_raw_mean": float(np.mean([r["rmse_raw"] for r in rows])),
        "mean_half95": float(np.mean([r["mean_half95"] for r in rows])),
    }
    pooled_ratios = np.concatenate([r["ratios"] for r in rows])
    pooled_ratios = pooled_ratios[np.isfinite(pooled_ratios)]
    for level, z in Z_LEVELS.items():
        per_rep = np.array([r["cover"][level] for r in rows])
        per_rep_oracle = np.array([r["cover_oracle"][level] for r in rows])
        key = f"{level:.2f}"
        out[f"cover_{key}"] = float(np.mean(per_rep))
        # Replications are independent; cells within one share tau2-hat, so the
        # MC error clusters by replication.
        out[f"cover_{key}_mc_se"] = float(np.std(per_rep, ddof=1) / np.sqrt(n_reps))
        out[f"cover_oracle_{key}"] = float(np.mean(per_rep_oracle))
        # Widening factor: multiply the shipped level-L interval by this to hit
        # nominal L against the true cell effects.
        if len(pooled_ratios):
            out[f"width_inflation_{key}"] = float(
                np.quantile(pooled_ratios, level) / (z / Z95)
            )
    return out


def main() -> None:
    t0 = time.time()
    snapshot = env_snapshot()
    print_snapshot(snapshot)

    names, ns = real_cell_ns()
    print(
        f"[setup] {len(names)} cells from artifact, n: min {ns.min()} median "
        f"{int(np.median(ns))} max {ns.max()} total {ns.sum()}; "
        f"tau2_true(shipped)={TAU2_ARTIFACT:.4g}, s2={S2_ARTIFACT:.4g}"
    )
    pool, cell_var_map = real_residual_pool_and_cell_vars()
    cell_sigma2 = np.array([cell_var_map[n] for n in names])
    print(
        f"[setup] real residual pool n={len(pool)}, real per-cell within-var "
        f"range [{cell_sigma2.min():.4f}, {cell_sigma2.max():.4f}]"
    )

    scenarios = []
    sid = 0
    for tau2_true, tag in [
        (0.0, "tau2=0"),
        (TAU2_ARTIFACT, "tau2=shipped"),
        (4 * TAU2_ARTIFACT, "tau2=4x"),
    ]:
        for scale in (1, 12, 24):
            sid += 1
            scenarios.append(
                dict(
                    label=f"normal {tag} scale={scale}x",
                    tau2_true=tau2_true, scale=scale, noise="normal",
                    seed=MASTER_SEED + sid,
                )
            )
    for scale in (1, 12, 24):
        sid += 1
        scenarios.append(
            dict(
                label=f"empirical-tails tau2=shipped scale={scale}x",
                tau2_true=TAU2_ARTIFACT, scale=scale, noise="empirical",
                seed=MASTER_SEED + sid,
            )
        )
    for scale in (1, 12, 24):
        sid += 1
        scenarios.append(
            dict(
                label=f"hetero tau2=shipped scale={scale}x",
                tau2_true=TAU2_ARTIFACT, scale=scale, noise="hetero",
                seed=MASTER_SEED + sid,
            )
        )

    results = []
    for sc in scenarios:
        t1 = time.time()
        res = run_scenario(
            sc["label"], names, ns, sc["scale"], sc["tau2_true"], sc["noise"],
            pool, cell_sigma2, N_REPS, sc["seed"],
        )
        res["runtime_s"] = round(time.time() - t1, 1)
        results.append(res)
        rb = f"{res['tau2_rel_bias']:+.2f}" if res["tau2_rel_bias"] is not None else "  n/a"
        print(
            f"[{res['label']:<42}] tau2-hat mean {res['tau2_hat_mean']:.3e} "
            f"(rel bias {rb}, P(=0) {res['prob_tau2_zero']:.2f}) | "
            f"cover 50/80/95 {res['cover_0.50']:.3f}/{res['cover_0.80']:.3f}/{res['cover_0.95']:.3f} "
            f"(oracle {res['cover_oracle_0.95']:.3f}) | B abs err {res['b_abs_err_mean']:.3f} | "
            f"RMSE EB/raw {res['rmse_eb_mean']:.4f}/{res['rmse_raw_mean']:.4f} | "
            f"half95 {res['mean_half95']:.4f} | infl95 {res.get('width_inflation_0.95', float('nan')):.2f} "
            f"| {res['runtime_s']}s"
        )

    write_results(
        "parameter_recovery.json",
        {
            "env": snapshot,
            "master_seed": MASTER_SEED,
            "n_reps": N_REPS,
            "cells": {"names": names, "ns": ns.tolist()},
            "anchors": {"tau2_artifact": TAU2_ARTIFACT, "s2_artifact": S2_ARTIFACT, "mu_log": MU_LOG},
            "scenarios": results,
            "runtime_seconds": round(time.time() - t0, 1),
        },
    )
    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
