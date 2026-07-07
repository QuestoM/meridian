"""Optimizer's-curse bootstrap: honest confidence bands on the F1-over-greedy gain.

The optimizer picks a break plan partly BECAUSE that plan's retention coefficients
were estimated in its favour (winner's curse), so a reported "+X%" gain is biased
upward. This measures that bias directly with a day-level block bootstrap of the
retention measurement dataset.

Method (set by the lead):
  * Measure per-break retention effects ONCE on the real month (break_effects).
    Each break's log_effect is computed independently on real audience, so a
    day-level block resample of these measured units is a valid bootstrap of the
    pooled coefficients while preserving within-day correlation (the whole day's
    breaks move together).
  * For each resample: draw the 30 measurement days with replacement, refit the
    empirical-Bayes pooling (channel_coefficients, DerSimonian-Laird tau^2) and the
    first-break multiplier (first_break_gate) with the SHIPPED fitting code, build
    an impact model from the resampled coefficients, and re-run the optimizer
    (greedy and greedy+F1) on the 12 evaluation channel-days.
  * Record (a) the nominal F1-over-greedy gain under the resample's OWN
    coefficients, and (b) the honest gain: each resample's chosen greedy and F1
    plans re-evaluated under the ORIGINAL full-data coefficients. (b) measures how
    much of the nominal gain survives estimation noise.

Objective basis: revenue_net (pure ILS net of retention cost). It is the cleanly
additively-separable engine truth (no global scale, no global clamp), the same
basis the +4.29% F1 figure was reported on, so the gain is confound-free.

Evaluation set: the 12 real channel-days behind the recorded +4.29% F1 figure
(4 channels x 2024-11-01..03). Optimizer run through the engine primitive
optimize_breaks with default Guardrails, revenue_weight 0.5, risk_lambda 0.0.

No data/ writes. Outputs live under analysis/bootstrap-curse/.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.data.transform import build_segments_from_programmes
from kairos.model.impact import PosteriorImpactModel, RetentionEstimate
from kairos.model.measure import (
    between_cell_variance,
    break_effects,
    channel_coefficients,
    confidence_label,
    first_break_gate,
)
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import OptimizerAssumptions, PricingModel
from kairos.optimize.revenue_net import segment_net_revenue
from kairos.service import _build_classifier

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results.csv"
SUMMARY_JSON = HERE / "summary.json"
SUMMARY_MD = HERE / "SUMMARY.md"

N_RESAMPLES = 300
SEED = 12345
REVENUE_WEIGHT = 0.5
RISK_LAMBDA = 0.0
OBJECTIVE_MODE = "revenue_net"

CHANNELS = ["כאן 11", "עכשיו 14",
            "קשת 12", "רשת 13"]
EVAL_DAYS = ["2024-11-01", "2024-11-02", "2024-11-03"]

_DEFAULT_COEF = OptimizerAssumptions().retention_impact_per_break


def build_model(coeffs) -> PosteriorImpactModel:
    """A PosteriorImpactModel from a channel_coefficients() dict (point + CI + n)."""
    cmap: dict[str, float] = {}
    detail: dict[str, RetentionEstimate] = {}
    for name, c in coeffs.items():
        cmap[name] = c.coefficient
        detail[name] = RetentionEstimate(
            coefficient=c.coefficient, ci_low=c.ci_low, ci_high=c.ci_high,
            n=c.n, confidence=confidence_label(c.n, c.ci_low, c.ci_high),
        )
    return PosteriorImpactModel(
        cmap, default=_DEFAULT_COEF, source="measured", detail=detail,
    )


def net_of_plan(segments, counts) -> float:
    """Sum the per-segment revenue-net-of-retention (ILS) at the plan's counts."""
    return float(sum(segment_net_revenue(s, counts[s.segment_id]) for s in segments))


def optimize_plan(segments, guardrails, refine: bool) -> dict[str, int]:
    """Return the chosen break count per segment id for greedy (refine False)/F1."""
    result = optimize_breaks(
        segments, guardrails,
        revenue_weight=REVENUE_WEIGHT, risk_lambda=RISK_LAMBDA,
        refine=refine, objective_mode=OBJECTIVE_MODE,
    )
    return {sp.segment_id: sp.num_breaks for sp in result.segments}


def main() -> None:
    wall0 = time.time()
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()
    classifier = _build_classifier()
    pricing = PricingModel.from_yaml()
    guardrails = Guardrails()

    # Measure once. Each row is one real break's detrended retention effect; the
    # break's calendar day tags the bootstrap block.
    effects = break_effects(spots, programmes, dayparts, classifier)
    effects = effects.copy()
    effects["day"] = pd.to_datetime(effects["break_start"]).dt.strftime("%Y-%m-%d")
    unique_days = sorted(effects["day"].dropna().unique().tolist())
    rows_by_day = {d: effects[effects["day"] == d] for d in unique_days}

    # Original full-data reference: coefficients M0 + its first-break gate. These
    # define the honest re-evaluation basis and the point-estimate anchor gain.
    base_coeffs = channel_coefficients(effects)
    base_gate = first_break_gate(effects)
    base_model = build_model(base_coeffs)
    base_assumptions = OptimizerAssumptions(
        first_break_multiplier=float(base_gate["first_break_multiplier"]),
    )

    eval_cds = [(ch, d) for d in EVAL_DAYS for ch in CHANNELS]
    base_segments = {
        cd: build_segments_from_programmes(
            programmes, classifier, pricing,
            assumptions=base_assumptions, impact_model=base_model,
            channel=cd[0], day=cd[1],
        )
        for cd in eval_cds
    }

    setup_s = time.time() - wall0
    print(f"setup {setup_s:.1f}s  effects_rows={len(effects)}  days={len(unique_days)}  "
          f"base_gate_mult={base_gate['first_break_multiplier']} "
          f"active={base_gate['first_break_active']}")

    # Point-estimate anchor (idx 0): the reported F1-over-greedy gain on the full
    # data itself, no resampling. Nominal == honest here by construction.
    records: list[dict] = []
    anchor = run_one(base_model, base_assumptions, base_segments,
                     base_segments, guardrails, eval_cds)
    anchor.update({"resample": 0, "kind": "point_estimate_anchor",
                   "gate_mult": float(base_gate["first_break_multiplier"]),
                   "gate_active": bool(base_gate["first_break_active"]),
                   "tau2": float(between_cell_variance(effects)["tau2"])})
    records.append(anchor)
    print(f"anchor nominal%={anchor['nominal_gain_pct']:.4f} "
          f"honest%={anchor['honest_gain_pct']:.4f}")

    rng = np.random.default_rng(SEED)
    t_loop = time.time()
    for i in range(1, N_RESAMPLES + 1):
        sample_days = rng.choice(unique_days, size=len(unique_days), replace=True)
        eff_r = pd.concat([rows_by_day[d] for d in sample_days], ignore_index=True)
        coeffs_r = channel_coefficients(eff_r)
        gate_r = first_break_gate(eff_r)
        diag_r = between_cell_variance(eff_r)
        model_r = build_model(coeffs_r)
        assumptions_r = OptimizerAssumptions(
            first_break_multiplier=float(gate_r["first_break_multiplier"]),
        )
        segs_r = {
            cd: build_segments_from_programmes(
                programmes, classifier, pricing,
                assumptions=assumptions_r, impact_model=model_r,
                channel=cd[0], day=cd[1],
            )
            for cd in eval_cds
        }
        rec = run_one(model_r, assumptions_r, segs_r, base_segments,
                      guardrails, eval_cds)
        rec.update({"resample": i, "kind": "bootstrap",
                    "gate_mult": float(gate_r["first_break_multiplier"]),
                    "gate_active": bool(gate_r["first_break_active"]),
                    "tau2": float(diag_r["tau2"])})
        records.append(rec)
        if i % 25 == 0:
            rate = (time.time() - t_loop) / i
            eta = rate * (N_RESAMPLES - i)
            print(f"  resample {i}/{N_RESAMPLES}  {rate:.2f}s/it  eta {eta/60:.1f}min")

    df = pd.DataFrame(records)
    df.to_csv(RESULTS_CSV, index=False)
    write_summary(df, setup_s, time.time() - wall0)
    print(f"done total {(time.time()-wall0)/60:.1f}min -> {RESULTS_CSV}")


def run_one(model_r, assumptions_r, segs_r, base_segments, guardrails, eval_cds) -> dict:
    """Optimize each eval channel-day under the resample; score own + base coeffs."""
    g_r = f_r = g_b = f_b = 0.0
    for cd in eval_cds:
        segments = segs_r[cd]
        kg = optimize_plan(segments, guardrails, refine=False)
        kf = optimize_plan(segments, guardrails, refine=True)
        # Nominal: score under the resample's own coefficients.
        g_r += net_of_plan(segments, kg)
        f_r += net_of_plan(segments, kf)
        # Honest: score the SAME chosen counts under the full-data coefficients.
        bs = base_segments[cd]
        g_b += net_of_plan(bs, kg)
        f_b += net_of_plan(bs, kf)
    nominal_ils = f_r - g_r
    honest_ils = f_b - g_b
    return {
        "net_greedy_own": g_r, "net_f1_own": f_r,
        "net_greedy_base": g_b, "net_f1_base": f_b,
        "nominal_gain_ils": nominal_ils,
        "honest_gain_ils": honest_ils,
        "nominal_gain_pct": (nominal_ils / g_r * 100.0) if g_r else float("nan"),
        "honest_gain_pct": (honest_ils / g_b * 100.0) if g_b else float("nan"),
    }


def _pct(series: pd.Series) -> dict:
    a = series.to_numpy(dtype=float)
    a = a[np.isfinite(a)]
    return {
        "p5": float(np.percentile(a, 5)),
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "mean": float(np.mean(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
    }


def write_summary(df: pd.DataFrame, setup_s: float, total_s: float) -> None:
    boot = df[df["kind"] == "bootstrap"]
    anchor = df[df["kind"] == "point_estimate_anchor"].iloc[0]
    nom = _pct(boot["nominal_gain_pct"])
    hon = _pct(boot["honest_gain_pct"])
    nom_ils = _pct(boot["nominal_gain_ils"])
    hon_ils = _pct(boot["honest_gain_ils"])
    # Paired shrinkage: honest/nominal per resample (only where nominal is a real
    # positive gain, so the ratio is meaningful).
    paired = boot[boot["nominal_gain_pct"] > 1e-9].copy()
    paired_ratio = (paired["honest_gain_pct"] / paired["nominal_gain_pct"]).to_numpy()
    paired_ratio = paired_ratio[np.isfinite(paired_ratio)]
    shrink_of_medians = (hon["p50"] / nom["p50"]) if nom["p50"] else float("nan")
    n_gate_on = int(boot["gate_active"].sum())
    n_honest_neg = int((boot["honest_gain_pct"] < 0).sum())
    n_nominal_neg = int((boot["nominal_gain_pct"] < 0).sum())

    summary = {
        "n_resamples": int(len(boot)),
        "eval_channel_days": len(CHANNELS) * len(EVAL_DAYS),
        "objective_mode": OBJECTIVE_MODE,
        "seed": SEED,
        "setup_seconds": round(setup_s, 1),
        "total_minutes": round(total_s / 60.0, 2),
        "point_estimate_anchor": {
            "nominal_gain_pct": float(anchor["nominal_gain_pct"]),
            "honest_gain_pct": float(anchor["honest_gain_pct"]),
            "nominal_gain_ils": float(anchor["nominal_gain_ils"]),
        },
        "nominal_gain_pct": nom,
        "honest_gain_pct": hon,
        "nominal_gain_ils": nom_ils,
        "honest_gain_ils": hon_ils,
        "shrinkage": {
            "median_honest_over_median_nominal": shrink_of_medians,
            "paired_ratio_p5": float(np.percentile(paired_ratio, 5)),
            "paired_ratio_p50": float(np.percentile(paired_ratio, 50)),
            "paired_ratio_p95": float(np.percentile(paired_ratio, 95)),
            "paired_ratio_mean": float(np.mean(paired_ratio)),
        },
        "first_break_gate_activations": n_gate_on,
        "n_nominal_negative": n_nominal_neg,
        "n_honest_negative": n_honest_neg,
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append("# Optimizer's-curse bootstrap: F1-over-greedy honest bands")
    lines.append("")
    lines.append(f"N resamples: {summary['n_resamples']} (day-level block bootstrap, "
                 f"seed {SEED}); {summary['eval_channel_days']} eval channel-days; "
                 f"objective {OBJECTIVE_MODE}; total {summary['total_minutes']} min.")
    lines.append("")
    lines.append("F1-over-greedy gain, nominal (resample's own coefficients) vs honest "
                 "(chosen plans re-scored under original full-data coefficients):")
    lines.append("")
    lines.append("| metric | p5 | p50 | p95 | mean |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(f"| nominal gain % | {nom['p5']:.4f} | {nom['p50']:.4f} | "
                 f"{nom['p95']:.4f} | {nom['mean']:.4f} |")
    lines.append(f"| honest gain % | {hon['p5']:.4f} | {hon['p50']:.4f} | "
                 f"{hon['p95']:.4f} | {hon['mean']:.4f} |")
    lines.append(f"| nominal gain ILS | {nom_ils['p5']:.0f} | {nom_ils['p50']:.0f} | "
                 f"{nom_ils['p95']:.0f} | {nom_ils['mean']:.0f} |")
    lines.append(f"| honest gain ILS | {hon_ils['p5']:.0f} | {hon_ils['p50']:.0f} | "
                 f"{hon_ils['p95']:.0f} | {hon_ils['mean']:.0f} |")
    lines.append("")
    lines.append(f"Point-estimate anchor (full data, no resampling): nominal "
                 f"{summary['point_estimate_anchor']['nominal_gain_pct']:.4f}% "
                 f"({summary['point_estimate_anchor']['nominal_gain_ils']:.0f} ILS over 12 days).")
    lines.append("")
    lines.append("Shrinkage (how much of the nominal gain survives estimation noise):")
    lines.append(f"- median(honest%) / median(nominal%) = {shrink_of_medians:.4f}")
    lines.append(f"- paired honest/nominal ratio p5/p50/p95 = "
                 f"{summary['shrinkage']['paired_ratio_p5']:.4f} / "
                 f"{summary['shrinkage']['paired_ratio_p50']:.4f} / "
                 f"{summary['shrinkage']['paired_ratio_p95']:.4f}")
    lines.append("")
    lines.append(f"First-break gate activated in {n_gate_on}/{summary['n_resamples']} "
                 f"resamples. Nominal gain negative in {n_nominal_neg}; honest gain "
                 f"negative in {n_honest_neg}.")
    SUMMARY_MD.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
