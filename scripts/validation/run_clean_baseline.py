"""Fix preview: rebuild the detrend baseline from content-only minutes.

Items 1-2 found the mechanism that biases the shipped coefficients TOWARD
ZERO: the detrend baseline (`_baseline_levels`) is the month-mean TVR at each
broadcast minute INCLUDING minutes when a break was airing on other days.
Breaks recur at similar clock minutes across days, so the baseline is
depressed exactly where real after-windows sit (in-break density gap after
minus before window: +0.085 for real breaks vs +0.010 for placebo minutes,
standardized difference +0.46). A depressed expected level after the break
absorbs part of the true shedding into the "expected" curve.

This script measures the fix directly, WITHOUT editing any product source: it
rebuilds the baseline excluding every minute covered by commercial airtime
(two variants: detected breaks only, and every ad-air run including single
spots), temporarily rebinds `kairos.model.measure._baseline_levels` for the
duration of one `break_effects` call (runtime patch, restored in a finally),
and reports:

  * the average in-break audience dip that contaminates the shipped baseline;
  * the pooled effect under the clean baseline (full shipped pipeline: same
    break detection, clipping, pooling);
  * the matched placebo re-measured under the same clean baseline (does the
    machinery drift shrink once the contamination is gone?);
  * the placebo-corrected effect under the clean baseline, i.e. the best
    causal estimate this review can construct;
  * how far the 36 shipped cell coefficients would move.

Deterministic: default_rng(42). Runtime ~2 minutes.
Run from the repo root: python scripts/validation/run_clean_baseline.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    SEED, delta, dl_pool, joint_cluster_bootstrap, load_bundle, min_to_ts,
    percentile_ci, sample_matched_pseudo, write_section,
)

import kairos.model.measure as measure_mod  # noqa: E402
from kairos.model.measure import (  # noqa: E402
    _broadcast_minute, _dayparts_frame, channel_coefficients,
)


def in_ad_mask(frame, spans: dict) -> np.ndarray:
    """Boolean mask over daypart rows: minute lies inside a commercial span."""
    from common import to_min
    minute = frame["ts"].map(to_min)
    mask = np.zeros(len(frame), dtype=bool)
    for channel, (starts, ends) in spans.items():
        rows = frame["channel"] == channel
        m = minute[rows].to_numpy()
        j = np.searchsorted(starts, m, side="right") - 1
        hit = (j >= 0) & (ends[np.clip(j, 0, len(ends) - 1)] >= m)
        mask[np.where(rows)[0]] = hit
    return mask


def clean_baseline(frame, spans: dict) -> dict:
    keep = frame[~in_ad_mask(frame, spans)]
    grouped = keep.groupby(["channel", "mod"])["tvr"].mean()
    return {(str(ch), int(mod)): float(v) for (ch, mod), v in grouped.items()}


def run_pipeline_with_baseline(bundle, baseline_dict) -> "pd.DataFrame":
    """Run the FULL shipped break_effects with a substituted baseline curve."""
    original = measure_mod._baseline_levels
    try:
        measure_mod._baseline_levels = lambda frame: baseline_dict
        return measure_mod.break_effects(
            bundle.spots, bundle.programmes, bundle.dayparts, bundle.classifier)
    finally:
        measure_mod._baseline_levels = original


def main() -> None:
    t0 = time.time()
    bundle = load_bundle()
    shipped = dl_pool(bundle.effects)
    frame = _dayparts_frame(bundle.dayparts)

    # How deep is the dip the shipped baseline averages in?
    ad_mask = in_ad_mask(frame, bundle.break_spans)
    inbreak = frame[ad_mask]
    excess = []
    for row in inbreak.itertuples(index=False):
        base = bundle.baseline.get((str(row.channel), _broadcast_minute(row.ts)))
        if base and base > 0 and row.tvr and row.tvr > 0:
            excess.append(np.log(row.tvr) - np.log(base))
    dip = delta(float(np.mean(excess)))
    share = float(ad_mask.mean())
    print(f"[contamination] {100*share:.1f}% of channel-minutes are inside a detected "
          f"break; mean in-break audience vs shipped baseline: {100*dip:+.1f}%")

    results = {}
    for name, spans in (("breaks-only", bundle.break_spans), ("all-ad-airtime", bundle.ad_spans)):
        base_clean = clean_baseline(frame, spans)
        effects_clean = run_pipeline_with_baseline(bundle, base_clean)
        pooled_clean = dl_pool(effects_clean)

        # placebo under the same clean baseline: same sampler, same seed
        saved = bundle.baseline
        bundle.baseline = base_clean
        rng = np.random.default_rng(SEED)
        pseudo_clean = sample_matched_pseudo(bundle, rng, k=3, strict=False)
        bundle.baseline = saved

        rngj = np.random.default_rng(SEED + 30)
        effects_clean = effects_clean.copy()
        effects_clean["cluster"] = (effects_clean["channel"].astype(str) + "|" +
                                    effects_clean["break_start"].dt.strftime("%Y-%m-%d"))
        joint = joint_cluster_bootstrap(
            {"real": effects_clean, "pseudo": pseudo_clean}, rngj, n_boot=1000)
        corrected_mu = pooled_clean["mu"] - float(pseudo_clean["log_effect"].mean())
        c_lo, c_hi = percentile_ci(joint["real"] - joint["pseudo"])

        results[name] = {
            "n": pooled_clean["n"],
            "pooled": pooled_clean["pooled_delta"],
            "pseudo_mean": delta(float(pseudo_clean["log_effect"].mean())),
            "pseudo_n": len(pseudo_clean),
            "corrected": delta(corrected_mu),
            "ci": (delta(c_lo), delta(c_hi)),
        }
        r = results[name]
        print(f"[{name}] pooled {r['pooled']:+.5f} (n={r['n']}), "
              f"placebo {r['pseudo_mean']:+.5f} (n={r['pseudo_n']}), "
              f"corrected {r['corrected']:+.5f} CI [{r['ci'][0]:+.5f}, {r['ci'][1]:+.5f}]")

        if name == "all-ad-airtime":
            coefs_clean = channel_coefficients(effects_clean)
            shipped_coefs = channel_coefficients(bundle.effects)
            shifts = np.array([coefs_clean[c].coefficient - shipped_coefs[c].coefficient
                               for c in shipped_coefs if c in coefs_clean])
            results[name]["coef_shift_mean"] = float(shifts.mean())
            results[name]["coef_shift_max"] = float(shifts[np.argmax(np.abs(shifts))])
            print(f"  36-cell coefficient shift: mean {shifts.mean():+.5f}, "
                  f"max |shift| {shifts[np.argmax(np.abs(shifts))]:+.5f}")

    elapsed = time.time() - t0
    ra = results["all-ad-airtime"]
    rb = results["breaks-only"]

    lines = []
    lines.append("## 5. Fix preview: content-only detrend baseline "
                 "(`scripts/validation/run_clean_baseline.py`, seed 42)")
    lines.append("")
    lines.append(f"The shipped baseline averages break minutes into the 'typical' "
                 f"curve: {100*share:.1f}% of channel-minutes lie inside a detected "
                 f"break, and audience during those minutes runs {100*dip:+.1f}% vs "
                 f"the shipped baseline at the same broadcast minute. Rebuilding the "
                 f"baseline from content-only minutes and re-running the ENTIRE "
                 f"shipped pipeline (runtime rebind of `_baseline_levels`; no source "
                 f"edit):")
    lines.append("")
    lines.append("| baseline | pooled delta | matched placebo mean | placebo-corrected "
                 "delta | 95% CI (joint cluster bootstrap) |")
    lines.append("|---|---|---|---|---|")
    lines.append(f"| shipped (ad minutes included) | -0.03906 | +0.01506 | -0.05331 "
                 f"| [-0.06459, -0.04260] |")
    lines.append(f"| clean, breaks-only excluded | {rb['pooled']:+.5f} "
                 f"| {rb['pseudo_mean']:+.5f} | {rb['corrected']:+.5f} "
                 f"| [{rb['ci'][0]:+.5f}, {rb['ci'][1]:+.5f}] |")
    lines.append(f"| clean, all ad airtime excluded | {ra['pooled']:+.5f} "
                 f"| {ra['pseudo_mean']:+.5f} | {ra['corrected']:+.5f} "
                 f"| [{ra['ci'][0]:+.5f}, {ra['ci'][1]:+.5f}] |")
    lines.append("")
    lines.append(f"Moving to the content-only baseline shifts the raw pooled effect "
                 f"from -0.03906 to {ra['pooled']:+.5f} and moves each of the 36 "
                 f"shipped cell coefficients by {ra['coef_shift_mean']:+.5f} on "
                 f"average (largest single-cell move {ra['coef_shift_max']:+.5f}). "
                 f"The residual placebo mean under the clean baseline "
                 f"({ra['pseudo_mean']:+.5f}) is within-show audience drift that the "
                 f"baseline cannot and should not absorb; the honest per-break cost "
                 f"is the placebo-corrected {ra['corrected']:+.5f}.")
    lines.append("")
    lines.append(f"Runtime {elapsed:.0f}s; deterministic (default_rng(42)).")
    write_section("cleanbase", "\n".join(lines))


if __name__ == "__main__":
    main()
