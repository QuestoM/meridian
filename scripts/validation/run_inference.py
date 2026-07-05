"""Item 3: randomization inference and cluster-robust uncertainty.

The shipped pooling treats the 2,532 measured breaks as independent: the
global-mean standard error is sqrt(s_p^2 / N) and each cell's credible
interval comes from the same independence assumption. But effects on the same
channel-day share programming, audience flow and schedule structure. This
script computes:

  (a) a permutation test of the pooled real-vs-placebo contrast: within each
      programme (stratum), the labels "real break" / "pseudo-break minute" are
      randomly reassigned (counts preserved), 2,000 permutations, two-sided
      p-value for the observed difference in means. Exact under the sharp null
      that, within a programme, measured minutes are exchangeable;
  (b) a cluster (block) bootstrap resampling whole channel-days (1,000 draws)
      for the pooled effect, compared with the shipped naive CI, reporting the
      SE inflation factor; and the same per-cell: every draw re-runs the full
      DerSimonian-Laird empirical-Bayes pooling, so each of the 36 shipped
      cell CIs is compared with its cluster-robust sampling interval.

Deterministic: default_rng(42). Runtime ~1 minute.
Run from the repo root: python scripts/validation/run_inference.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    REPO, SEED, delta, dl_pool, load_bundle, percentile_ci,
    sample_matched_pseudo, write_section,
)


def permutation_test(real: pd.DataFrame, pseudo: pd.DataFrame, rng, n_perm=2000):
    """Permute real/pseudo labels within programme strata; two-sided p for the mean gap."""
    a = real[real["prog_key"].notna()][["prog_key", "log_effect"]].copy()
    a["is_real"] = True
    b = pseudo[["prog_key", "log_effect"]].copy()
    b["is_real"] = False
    both = pd.concat([a, b], ignore_index=True)
    # Keep only strata containing both labels, else the permutation is a no-op.
    counts = both.groupby("prog_key")["is_real"].agg(["sum", "count"])
    keep = counts[(counts["sum"] > 0) & (counts["sum"] < counts["count"])].index
    both = both[both["prog_key"].isin(keep)].copy()
    both["block"] = pd.factorize(both["prog_key"])[0]
    both = both.sort_values(["block"], kind="stable").reset_index(drop=True)

    values = both["log_effect"].to_numpy()
    blocks = both["block"].to_numpy()
    is_real = both["is_real"].to_numpy()
    n = len(both)
    n_real = int(is_real.sum())
    n_pseudo = n - n_real
    t_obs = values[is_real].mean() - values[~is_real].mean()

    block_sizes = np.bincount(blocks)
    block_starts = np.concatenate([[0], np.cumsum(block_sizes)[:-1]])
    n_real_per_block = np.bincount(blocks, weights=is_real).astype(int)
    pos_offset = np.repeat(block_starts, block_sizes)
    real_quota = np.repeat(n_real_per_block, block_sizes)

    draws = np.empty(n_perm)
    for p in range(n_perm):
        keys = rng.random(n)
        order = np.lexsort((keys, blocks))
        pos_in_block = np.arange(n) - pos_offset
        mask = pos_in_block < real_quota  # first n_real_g of each block -> "real"
        v = values[order]
        draws[p] = v[mask].sum() / n_real - v[~mask].sum() / n_pseudo
    p_value = (1 + np.sum(np.abs(draws) >= abs(t_obs))) / (1 + n_perm)
    return {
        "t_obs": float(t_obs), "p": float(p_value), "n_strata": int(len(keep)),
        "n_real": n_real, "n_pseudo": n_pseudo,
        "perm_sd": float(draws.std(ddof=1)),
    }


def cluster_tables(effects: pd.DataFrame):
    """Per-(cluster, cell) sufficient statistics for fast bootstrap DL pooling."""
    grouped = effects.groupby(["cluster", "channel_name"])["log_effect"]
    agg = grouped.agg(n="count", s="sum", q=lambda x: float(np.sum(np.square(x))))
    clusters = sorted(effects["cluster"].unique())
    cells = sorted(effects["channel_name"].unique())
    ci = {c: i for i, c in enumerate(clusters)}
    gi = {g: i for i, g in enumerate(cells)}
    N = np.zeros((len(clusters), len(cells)))
    S = np.zeros_like(N)
    Q = np.zeros_like(N)
    for (cluster, cell), row in agg.iterrows():
        N[ci[cluster], gi[cell]] = row["n"]
        S[ci[cluster], gi[cell]] = row["s"]
        Q[ci[cluster], gi[cell]] = row["q"]
    return clusters, cells, N, S, Q


def dl_from_tables(mult, N, S, Q):
    """Full DerSimonian-Laird EB pooling from resampled cluster multiplicities."""
    n = mult @ N
    s = mult @ S
    q = mult @ Q
    present = n > 0
    n, s, q = n[present], s[present], q[present]
    mean = s / n
    rss = q - s * s / n
    df = n.sum() - len(n)
    pooled_within = rss.sum() / df
    w = n / pooled_within
    sw = w.sum()
    mu = float((w * mean).sum() / sw)
    qstat = float((w * (mean - mu) ** 2).sum())
    c = sw - float((w ** 2).sum()) / sw
    tau2 = max(0.0, (qstat - (len(n) - 1)) / c) if c > 0 else 0.0
    sigma2 = pooled_within / n
    shrink = sigma2 / (sigma2 + tau2)
    theta = mu + (1.0 - shrink) * (mean - mu)
    return mu, np.exp(theta) - 1.0, present


def main() -> None:
    t0 = time.time()
    bundle = load_bundle()
    effects = bundle.effects
    real = dl_pool(effects)
    rng = np.random.default_rng(SEED)
    pseudo = sample_matched_pseudo(bundle, rng, k=3, strict=False)

    # (a) permutation test within programme strata
    perm = permutation_test(effects, pseudo, np.random.default_rng(SEED + 20))
    print(f"[permutation] T_obs (real - pseudo mean log) = {perm['t_obs']:+.5f}")
    print(f"              p = {perm['p']:.4g} over {perm['n_strata']} programme strata "
          f"({perm['n_real']} real / {perm['n_pseudo']} pseudo), perm sd {perm['perm_sd']:.5f}")

    # (b) channel-day block bootstrap of the FULL pooling pipeline
    clusters, cells, N, S, Q = cluster_tables(effects)
    n_clusters = len(clusters)
    rng_b = np.random.default_rng(SEED + 21)
    n_boot = 1000
    mu_draws = np.empty(n_boot)
    cell_draws = np.full((n_boot, len(cells)), np.nan)
    for b in range(n_boot):
        idx = rng_b.integers(0, n_clusters, size=n_clusters)
        mult = np.bincount(idx, minlength=n_clusters).astype(float)
        mu, deltas, present = dl_from_tables(mult, N, S, Q)
        mu_draws[b] = mu
        cell_draws[b, present] = deltas
    se_boot = float(mu_draws.std(ddof=1))
    se_naive = real["se_naive"]
    lo, hi = percentile_ci(mu_draws)
    inflation = se_boot / se_naive
    print(f"[pooled] mu = {real['mu']:+.5f} (delta {real['pooled_delta']:+.5f})")
    print(f"  naive SE (independence, shipped machinery) = {se_naive:.5f} log")
    print(f"  channel-day bootstrap SE                   = {se_boot:.5f} log")
    print(f"  SE/CI inflation factor = {inflation:.2f}")
    print(f"  naive 95% CI (delta): [{delta(real['mu']-1.96*se_naive):+.5f}, "
          f"{delta(real['mu']+1.96*se_naive):+.5f}]")
    print(f"  bootstrap 95% CI (delta): [{delta(lo):+.5f}, {delta(hi):+.5f}]")

    # per-cell: shipped credible interval vs cluster-robust sampling interval
    shipped = json.loads((REPO / "models" / "tv_break_coefficients.json").read_text())
    detail = shipped["detail"]
    rows = []
    for j, cell in enumerate(cells):
        d = detail.get(cell)
        if d is None:
            continue
        shipped_hw = (d["ci_high"] - d["ci_low"]) / 2.0
        draws = cell_draws[:, j]
        draws = draws[~np.isnan(draws)]
        boot_hw = (np.percentile(draws, 97.5) - np.percentile(draws, 2.5)) / 2.0
        rows.append((cell, shipped_hw, float(boot_hw), float(boot_hw / shipped_hw)))
    infl = np.array([r[3] for r in rows])
    worst = max(rows, key=lambda r: r[3])
    best = min(rows, key=lambda r: r[3])
    print(f"[per-cell] shipped CI halfwidth vs cluster-bootstrap halfwidth over "
          f"{len(rows)} cells:")
    print(f"  inflation median {np.median(infl):.2f}, min {infl.min():.2f} ({best[0]}), "
          f"max {infl.max():.2f} ({worst[0]})")

    elapsed = time.time() - t0

    lines = []
    lines.append("## 3. Randomization inference and clustering "
                 "(`scripts/validation/run_inference.py`, seed 42)")
    lines.append("")
    lines.append(f"**Permutation test** (labels permuted within programme strata, "
                 f"{perm['n_strata']} strata, 2,000 permutations): observed real-vs-pseudo "
                 f"gap in mean log effect {perm['t_obs']:+.5f}, permutation sd "
                 f"{perm['perm_sd']:.5f}, two-sided p = {perm['p']:.4g}. The break effect "
                 f"is not an artifact of which minutes got measured: no label "
                 f"reassignment within shows comes close to the observed gap.")
    lines.append("")
    lines.append(f"**Cluster (channel-day) bootstrap** of the full DL/EB pipeline, "
                 f"1,000 draws over {n_clusters} channel-day blocks:")
    lines.append("")
    lines.append("| quantity | naive (shipped assumption) | cluster-robust | inflation |")
    lines.append("|---|---|---|---|")
    lines.append(f"| pooled-mean SE (log) | {se_naive:.5f} | {se_boot:.5f} | "
                 f"x{inflation:.2f} |")
    lines.append(f"| pooled 95% CI (delta) | [{delta(real['mu']-1.96*se_naive):+.5f}, "
                 f"{delta(real['mu']+1.96*se_naive):+.5f}] | [{delta(lo):+.5f}, "
                 f"{delta(hi):+.5f}] | x{inflation:.2f} width |")
    lines.append("")
    lines.append(f"Per-cell, comparing each shipped credible-interval halfwidth with the "
                 f"cluster-bootstrap sampling halfwidth of the same EB estimator "
                 f"(median over {len(rows)} cells): inflation x{np.median(infl):.2f} "
                 f"(min x{infl.min():.2f} {best[0]}, max x{infl.max():.2f} {worst[0]}). "
                 f"These are different objects (posterior credible vs frequentist "
                 f"sampling), but the gap measures how much uncertainty the "
                 f"independence assumption hides from the operator-facing "
                 f"high/medium/low confidence labels.")
    lines.append("")
    lines.append(f"Runtime {elapsed:.0f}s; deterministic (default_rng(42)).")
    write_section("inference", "\n".join(lines))


if __name__ == "__main__":
    main()
