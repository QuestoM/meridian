"""Adversarial verification of the bootstrap-curse finding.

Independently re-runs the load-bearing computations:
  1. Reference point estimate (greedy vs F1 on the 12 headline channel-days,
     original full-data coefficients), scored THREE ways:
       a) the pipeline's own truth_net formula (reimplemented here),
       b) the engine-exact net objective, segment_net_revenue (the quantity the
          revenue_net optimizer actually maximises),
       c) the engine's own plan_revenue_net report on each optimizer result.
     (a) checks reproducibility of the claimed numbers; (b) and (c) check the
     "metric is the true objective" claim.
  2. A fresh mini-bootstrap with a DIFFERENT seed, scored with the engine-exact
     metric on both the nominal and honest bases, to check whether the headline
     conclusion (shrinkage >= 1, zero negative honest gains) survives.

Writes only under analysis/bootstrap-curse/verify/. data/ is read-only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data.classifier import ProgramClassifier  # noqa: E402
from kairos.data.loaders import load_dayparts, load_programmes, load_spots  # noqa: E402
from kairos.data.transform import build_segments_from_programmes  # noqa: E402
from kairos.model.impact import PosteriorImpactModel, RetentionEstimate  # noqa: E402
from kairos.model.measure import break_effects, channel_coefficients, first_break_gate  # noqa: E402
from kairos.optimize._segment_math import _segment_retention, _segment_revenue  # noqa: E402
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine  # noqa: E402
from kairos.optimize.day_core import _optimize_one_day  # noqa: E402
from kairos.optimize.inventory import load_inventory  # noqa: E402
from kairos.optimize.optimizer import optimize_breaks  # noqa: E402
from kairos.optimize.pacing import load_campaigns  # noqa: E402
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings  # noqa: E402
from kairos.optimize.revenue_net import plan_revenue_net, segment_net_revenue  # noqa: E402
from kairos.export.schedule import _load_constraints  # noqa: E402
from kairos.service import _pacing_knobs_from_settings, guardrails_from_settings  # noqa: E402
from kairos_api.core import _load_settings, _model_dump, _reference_today  # noqa: E402

OUT = Path(__file__).resolve().parent
EVAL_DAYS = ["2024-11-01", "2024-11-02", "2024-11-03"]
EVAL_CHANNELS = ["כאן 11", "קשת 12", "רשת 13", "עכשיו 14"]


def pipeline_truth_net(seg, k: int) -> float:
    """The lane pipeline's truth_net, reimplemented from its formula."""
    revenue = _segment_revenue(seg, k)
    cost = (
        seg.cpp * seg.premium * seg.baseline_tvr
        * (1.0 - _segment_retention(seg, k)) * (k * seg.break_length_seconds)
    )
    return revenue - cost


def make_impact_model(coefs, assumptions):
    estimates = {
        name: RetentionEstimate(coefficient=c.coefficient, ci_low=c.ci_low,
                                ci_high=c.ci_high, n=c.n, confidence="medium")
        for name, c in coefs.items()
    }
    return PosteriorImpactModel(
        {name: c.coefficient for name, c in coefs.items()},
        default=assumptions.retention_impact_per_break,
        source="measured", detail=estimates,
    )


class Harness:
    def __init__(self) -> None:
        self.settings_map = _model_dump(_load_settings())
        self.base_assumptions = OptimizerAssumptions()
        self.pricing = pricing_from_settings(self.settings_map, None)
        self.classifier = ProgramClassifier.from_yaml()
        self.guardrails = guardrails_from_settings(self.settings_map)
        self.operator_channel = str(self.settings_map.get("operator_channel", "") or "")
        self.constraints = _load_constraints(None)
        self.demand_engine = AdvertiserRuleEngine.from_files()
        self.inventory_pool = load_inventory()
        self.campaigns = load_campaigns()
        self.pacing_today = _reference_today(_load_settings())
        self.pacing_knobs = _pacing_knobs_from_settings(self.settings_map)
        self.programmes = load_programmes()

    def assumptions_with_fb(self, fb: float) -> OptimizerAssumptions:
        from dataclasses import replace
        return replace(self.base_assumptions,
                       first_break_multiplier=max(self.base_assumptions.first_break_multiplier, fb))

    def segments_for(self, channel, day, impact_model, assumptions):
        return build_segments_from_programmes(
            self.programmes, self.classifier, self.pricing, assumptions=assumptions,
            impact_model=impact_model, channel=channel, day=day)

    def optimize(self, segments, refine: bool):
        return _optimize_one_day(
            segments, guardrails=self.guardrails,
            revenue_weight=self.settings_map["revenue_weight"] / 100.0,
            risk_lambda=0.0, demand_engine=self.demand_engine,
            inventory_pool=self.inventory_pool, campaigns=self.campaigns,
            pacing_today=self.pacing_today, pacing_knobs=self.pacing_knobs,
            constraints=self.constraints, operator_channel=self.operator_channel,
            refine=refine, objective_mode="revenue_net", optimize_fn=optimize_breaks)


def score_plan(counts, seg_index, fn):
    total, skipped = 0.0, 0
    for sid, k in counts.items():
        seg = seg_index.get(sid)
        if seg is None:
            skipped += 1
            continue
        total += fn(seg, k)
    return total, skipped


def run_12(harness, impact, assumptions, base_segindex=None):
    """Optimize the 12 channel-days; return per-basis totals + engine report."""
    tot = {"g_pipe": 0.0, "f_pipe": 0.0, "g_eng": 0.0, "f_eng": 0.0,
           "g_rep": 0.0, "f_rep": 0.0, "g_pipe0": 0.0, "f_pipe0": 0.0,
           "g_eng0": 0.0, "f_eng0": 0.0, "skipped": 0, "days": 0}
    for channel in EVAL_CHANNELS:
        for day in EVAL_DAYS:
            segs = harness.segments_for(channel, day, impact, assumptions)
            if not segs:
                continue
            tot["days"] += 1
            idx = {s.segment_id: s for s in segs}
            g = harness.optimize(segs, refine=False)
            f = harness.optimize(segs, refine=True)
            gc = {p.segment_id: p.num_breaks for p in g.segments}
            fc = {p.segment_id: p.num_breaks for p in f.segments}
            for key, counts in (("g", gc), ("f", fc)):
                v, sk = score_plan(counts, idx, pipeline_truth_net)
                tot[key + "_pipe"] += v
                tot["skipped"] += sk
                v, _ = score_plan(counts, idx, segment_net_revenue)
                tot[key + "_eng"] += v
            tot["g_rep"] += plan_revenue_net(g, segments=segs)["revenue_net_ils"]
            tot["f_rep"] += plan_revenue_net(f, segments=segs)["revenue_net_ils"]
            if base_segindex is not None:
                idx0 = base_segindex[(channel, day)]
                for key, counts in (("g", gc), ("f", fc)):
                    v, _ = score_plan(counts, idx0, pipeline_truth_net)
                    tot[key + "_pipe0"] += v
                    v, _ = score_plan(counts, idx0, segment_net_revenue)
                    tot[key + "_eng0"] += v
    return tot


def resample_effects(effects, days, rng):
    picked = rng.choice(days, size=len(days), replace=True)
    blocks = []
    for rep, d in enumerate(picked):
        blk = effects[effects["day"] == d].copy()
        blk["prog_key"] = blk["prog_key"].astype(str) + f"__v{rep}"
        blocks.append(blk)
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=99173)
    args = ap.parse_args()

    t0 = time.time()
    harness = Harness()
    spots = load_spots()
    dayparts = load_dayparts()
    effects = break_effects(spots, harness.programmes, dayparts, harness.classifier)
    effects["day"] = pd.to_datetime(effects["break_start"]).dt.strftime("%Y-%m-%d")
    days = sorted(effects["day"].dropna().unique().tolist())
    print(f"[load] {time.time()-t0:.1f}s effects={len(effects)} days={len(days)}")

    coefs0 = channel_coefficients(effects)
    fb0 = float(first_break_gate(effects)["first_break_multiplier"])
    assum0 = harness.assumptions_with_fb(fb0)
    impact0 = make_impact_model(coefs0, assum0)
    base_segindex = {}
    for ch in EVAL_CHANNELS:
        for d in EVAL_DAYS:
            base_segindex[(ch, d)] = {s.segment_id: s
                                      for s in harness.segments_for(ch, d, impact0, assum0)}

    ref = run_12(harness, impact0, assum0)
    reference = {
        "fb_multiplier": fb0,
        "eval_channel_days": ref["days"],
        "plans_skipped_segments": ref["skipped"],
        "pipeline_basis": {"greedy": ref["g_pipe"], "f1": ref["f_pipe"],
                           "gain_ils": ref["f_pipe"] - ref["g_pipe"],
                           "gain_pct": (ref["f_pipe"] - ref["g_pipe"]) / ref["g_pipe"] * 100},
        "engine_exact_basis": {"greedy": ref["g_eng"], "f1": ref["f_eng"],
                               "gain_ils": ref["f_eng"] - ref["g_eng"],
                               "gain_pct": (ref["f_eng"] - ref["g_eng"]) / ref["g_eng"] * 100},
        "engine_reported_plan_revenue_net": {"greedy": ref["g_rep"], "f1": ref["f_rep"],
                                             "gain_ils": ref["f_rep"] - ref["g_rep"],
                                             "gain_pct": (ref["f_rep"] - ref["g_rep"]) / ref["g_rep"] * 100},
    }
    print(json.dumps(reference, indent=2))

    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.n):
        eff_r = resample_effects(effects, days, rng)
        coefs_r = channel_coefficients(eff_r)
        fb_r = float(first_break_gate(eff_r)["first_break_multiplier"])
        assum_r = harness.assumptions_with_fb(fb_r)
        impact_r = make_impact_model(coefs_r, assum_r)
        t = run_12(harness, impact_r, assum_r, base_segindex=base_segindex)
        rows.append({
            "resample": i, "fb": fb_r,
            "nominal_eng_ils": t["f_eng"] - t["g_eng"],
            "honest_eng_ils": t["f_eng0"] - t["g_eng0"],
            "nominal_eng_pct": (t["f_eng"] - t["g_eng"]) / t["g_eng"] * 100,
            "honest_eng_pct": (t["f_eng0"] - t["g_eng0"]) / t["g_eng0"] * 100,
            "nominal_pipe_ils": t["f_pipe"] - t["g_pipe"],
            "honest_pipe_ils": t["f_pipe0"] - t["g_pipe0"],
            "nominal_pipe_pct": (t["f_pipe"] - t["g_pipe"]) / t["g_pipe"] * 100,
            "honest_pipe_pct": (t["f_pipe0"] - t["g_pipe0"]) / t["g_pipe0"] * 100,
        })
        if (i + 1) % 5 == 0:
            print(f"  [{i+1}/{args.n}] {time.time()-t0:.0f}s "
                  f"eng nominal={rows[-1]['nominal_eng_ils']:.0f} honest={rows[-1]['honest_eng_ils']:.0f}")

    with (OUT / "verify_resamples.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    def band(key):
        a = np.array([r[key] for r in rows])
        return {"p5": float(np.percentile(a, 5)), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "mean": float(np.mean(a)),
                "min": float(a.min()), "n_negative": int((a < 0).sum())}

    summary = {
        "n_resamples": len(rows), "seed": args.seed,
        "reference": reference,
        "engine_exact": {"nominal_pct": band("nominal_eng_pct"),
                         "honest_pct": band("honest_eng_pct"),
                         "nominal_ils": band("nominal_eng_ils"),
                         "honest_ils": band("honest_eng_ils")},
        "pipeline_formula": {"nominal_pct": band("nominal_pipe_pct"),
                             "honest_pct": band("honest_pipe_pct")},
        "shrinkage_median_engine": (np.percentile([r["honest_eng_pct"] for r in rows], 50)
                                    / np.percentile([r["nominal_eng_pct"] for r in rows], 50)),
        "shrinkage_median_pipeline": (np.percentile([r["honest_pipe_pct"] for r in rows], 50)
                                      / np.percentile([r["nominal_pipe_pct"] for r in rows], 50)),
        "fb_active_frac": float(np.mean([r["fb"] > 1.0 for r in rows])),
    }
    (OUT / "verify_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
