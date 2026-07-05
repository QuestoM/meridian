"""Item 1: placebo / negative control for the retention-cost measurement.

Samples pseudo-break times at minutes where NO commercial break aired,
respecting the eligibility rules the real measurement effectively enforces
(windows inside the programme, full distance from every detected break, the
machinery's own positive-audience drop rules), then runs the EXACT shipped
measurement arithmetic on them. If the machinery is unbiased, the pseudo
effects center on zero; any negative placebo mean is spurious "cost"
manufactured by the detrend/window machinery, and the shipped coefficients
are inflated by that amount.

Designs:
  * MATCHED (primary): for each real measured break, up to 3 pseudo-breaks in
    the SAME programme with the SAME floor-minute duration, at uniformly
    sampled eligible minutes. Composition (channel, show, genre, time of day)
    matches the real measurement by construction.
  * MATCHED-STRICT: eligibility additionally excludes every single-spot ad
    run, so pseudo windows contain no commercial airtime at all (a pure
    machinery control; the primary design mirrors the machinery's own
    break-level clip standard instead).
  * UNIFORM (design B): one pseudo-break per programme across the whole EPG,
    duration drawn from the real break-duration distribution; plus the subset
    of programmes containing no detected break at all.

Deterministic: numpy default_rng(42) everywhere. Runtime ~2 minutes.
Run from the repo root: python scripts/validation/run_placebo.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    SEED, SHIPPED_POOLED_DELTA, cluster_bootstrap_mean, delta, dl_pool,
    joint_cluster_bootstrap, load_bundle, percentile_ci, sample_matched_pseudo,
    sample_uniform_pseudo, write_section,
)


def describe(name: str, frame, rng, real_mean_log: float) -> dict:
    """Mean, cluster-aware CI and share-of-real for one placebo sample."""
    mean_log = float(frame["log_effect"].mean())
    draws = cluster_bootstrap_mean(frame, rng, n_boot=1000)
    lo, hi = percentile_ci(draws)
    se = float(np.nanstd(draws, ddof=1))
    out = {
        "name": name, "n": len(frame),
        "n_clusters": frame["cluster"].nunique(),
        "mean_log": mean_log, "mean_delta": delta(mean_log),
        "ci_lo_log": lo, "ci_hi_log": hi,
        "ci_lo_delta": delta(lo), "ci_hi_delta": delta(hi),
        "se_log": se,
        "z_vs_zero": mean_log / se if se > 0 else float("nan"),
        "share_of_real": mean_log / real_mean_log,
    }
    print(f"  {name}: n={out['n']} ({out['n_clusters']} channel-day clusters)")
    print(f"    mean log_effect = {mean_log:+.5f}  -> delta {out['mean_delta']:+.5f}")
    print(f"    95% cluster-bootstrap CI (delta): [{out['ci_lo_delta']:+.5f}, {out['ci_hi_delta']:+.5f}]"
          f"  z vs 0 = {out['z_vs_zero']:+.2f}")
    print(f"    share of the real pooled effect it could explain: {100*out['share_of_real']:.1f}%")
    return out


def main() -> None:
    t0 = time.time()
    bundle = load_bundle()
    effects = bundle.effects
    real = dl_pool(effects)
    real_mean_log = real["mu"]
    print(f"[real] n={real['n']} pooled mu={real_mean_log:+.5f} "
          f"delta={real['pooled_delta']:+.5f} (shipped {SHIPPED_POOLED_DELTA})")

    # --- matched placebo (primary) ------------------------------------------
    rng = np.random.default_rng(SEED)
    matched = sample_matched_pseudo(bundle, rng, k=3, strict=False)
    n_src = effects["prog_key"].notna().sum()
    covered = matched["prog_key"].nunique()
    print(f"[matched] {len(matched)} pseudo effects from {covered} programmes "
          f"(source: {n_src} programme-matched real breaks)")
    res_matched = describe("matched", matched, np.random.default_rng(SEED + 1), real_mean_log)

    # --- matched, strict (no single-spot ad airtime in any window) ----------
    rng = np.random.default_rng(SEED)
    matched_strict = sample_matched_pseudo(bundle, rng, k=3, strict=True)
    res_strict = describe("matched-strict", matched_strict,
                          np.random.default_rng(SEED + 2), real_mean_log)

    # --- uniform over the EPG (design B) -------------------------------------
    rng = np.random.default_rng(SEED)
    uniform = sample_uniform_pseudo(bundle, rng, per_programme=1, strict=False)
    res_uniform = describe("uniform-EPG", uniform, np.random.default_rng(SEED + 3), real_mean_log)
    breakless = uniform[~uniform["programme_has_break"]]
    res_breakless = describe("uniform, breakless programmes only", breakless,
                             np.random.default_rng(SEED + 4), real_mean_log)

    # --- bias-corrected real effect (joint bootstrap, shared clusters) -------
    rng = np.random.default_rng(SEED + 5)
    joint = joint_cluster_bootstrap({"real": effects, "pseudo": matched}, rng, n_boot=1000)
    corrected_draws = joint["real"] - joint["pseudo"]
    corrected = real_mean_log - float(matched["log_effect"].mean())
    c_lo, c_hi = percentile_ci(corrected_draws)
    print(f"[corrected] real minus placebo (log) = {corrected:+.5f} -> delta {delta(corrected):+.5f}")
    print(f"            95% joint cluster-bootstrap CI (delta): "
          f"[{delta(c_lo):+.5f}, {delta(c_hi):+.5f}]")

    # --- where does the placebo effect live? ---------------------------------
    print("[matched placebo by program_type]")
    by_type = matched.groupby("program_type")["log_effect"].agg(["mean", "count"])
    for name, row in by_type.iterrows():
        print(f"    {name:12s} mean={row['mean']:+.5f} n={int(row['count'])}")
    print("[matched placebo by pseudo position]")
    by_pos = matched.groupby("pseudo_position")["log_effect"].agg(["mean", "count"])
    for name, row in by_pos.iterrows():
        print(f"    {name:8s} mean={row['mean']:+.5f} n={int(row['count'])}")

    elapsed = time.time() - t0

    lines = []
    lines.append("## 1. Placebo / negative control "
                 "(`scripts/validation/run_placebo.py`, seed 42)")
    lines.append("")
    lines.append(f"Real pooled effect reproduced from the current data: mu = "
                 f"{real_mean_log:+.5f} (log), delta = {real['pooled_delta']:+.5f} "
                 f"(shipped: {SHIPPED_POOLED_DELTA}). Pseudo-breaks were sampled at "
                 f"minutes with no detected break, windows fully inside the "
                 f"programme and clear of every detected break span, then measured "
                 f"with the exact shipped arithmetic (same window means, same "
                 f"detrend curve, same drop rules).")
    lines.append("")
    lines.append("| design | n | clusters | placebo mean (delta) | 95% cluster CI (delta) "
                 "| z vs 0 | share of real effect |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in (res_matched, res_strict, res_uniform, res_breakless):
        lines.append(
            f"| {r['name']} | {r['n']} | {r['n_clusters']} | {r['mean_delta']:+.5f} "
            f"| [{r['ci_lo_delta']:+.5f}, {r['ci_hi_delta']:+.5f}] | {r['z_vs_zero']:+.2f} "
            f"| {100*r['share_of_real']:.1f}% |")
    lines.append("")
    lines.append(f"Matched placebo (primary): mean pseudo effect "
                 f"{res_matched['mean_delta']:+.5f} explains "
                 f"{100*res_matched['share_of_real']:.1f}% of the shipped -0.0391.")
    lines.append("")
    lines.append(f"**Placebo-corrected real effect** (real mean minus matched placebo "
                 f"mean, joint channel-day bootstrap): delta = {delta(corrected):+.5f}, "
                 f"95% CI [{delta(c_lo):+.5f}, {delta(c_hi):+.5f}]. "
                 f"Implied multiplicative correction to every shipped coefficient: "
                 f"x{delta(corrected)/real['pooled_delta']:.3f}.")
    lines.append("")
    lines.append("Placebo mean by genre cell dimension (matched design, log scale): "
                 + "; ".join(f"{name} {row['mean']:+.5f} (n={int(row['count'])})"
                             for name, row in by_type.iterrows())
                 + ". By pseudo position: "
                 + "; ".join(f"{name} {row['mean']:+.5f}" for name, row in by_pos.iterrows())
                 + ".")
    lines.append("")
    lines.append(f"Runtime {elapsed:.0f}s; fully deterministic (default_rng(42)).")
    write_section("placebo", "\n".join(lines))


if __name__ == "__main__":
    main()
