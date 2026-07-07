"""Optimizer's-curse day-block bootstrap: honest confidence bands on F1 gains.

The optimizer picks a plan partly BECAUSE its retention coefficients were
mis-estimated in its favor (winner's curse), so a reported "+X%" F1-over-greedy
gain is structurally biased upward. This measures how much of that gain survives
estimation noise.

Design (set by the lead):
  * Block-bootstrap at the DAY level over the retention measurement dataset
    (preserves within-day correlation). One resample = draw 30 days with
    replacement from the 30-day reference month; each drawn day contributes its
    whole block of measured breaks. Duplicated-day blocks get a unique replicate
    tag so their prog_key programmes are treated as independent draws.
  * For each resample: refit the empirical-Bayes pooling (DerSimonian-Laird
    tau^2, channel_coefficients) and the first-break multiplier (first_break_gate)
    with the SHIPPED fitting code, then re-run the optimizer (greedy and
    greedy+F1) on the 12 evaluation channel-days under the resampled coefficients.
  * Record (a) the F1-over-greedy gain under the resample's OWN coefficients
    (nominal) and (b) the CRUCIAL honest number: each resample's two chosen plans
    re-EVALUATED under the ORIGINAL full-data coefficients (honest).

Metric basis: objective_mode='revenue_net', risk_lambda=0 (the mode + config the
+4.29% F1 headline was measured on; revenue_net is the one cleanly separable
retention-cost-in-ILS objective, so the curse enters through the retention cost
directly and both plans re-score exactly). Gain is net-ILS summed over the 12
channel-days, reported absolute and as percent of the greedy net.

READ-ONLY over data/. Writes only under analysis/bootstrap-curse/.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data.classifier import ProgramClassifier  # noqa: E402
from kairos.data.loaders import (  # noqa: E402
    load_dayparts,
    load_programmes,
    load_spots,
)
from kairos.data.transform import build_segments_from_programmes  # noqa: E402
from kairos.model.impact import PosteriorImpactModel, RetentionEstimate  # noqa: E402
from kairos.model.measure import (  # noqa: E402
    break_effects,
    channel_coefficients,
    first_break_gate,
)
from kairos.optimize._segment_math import (  # noqa: E402
    _segment_retention,
    _segment_revenue,
)
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine  # noqa: E402
from kairos.optimize.day_core import _optimize_one_day  # noqa: E402
from kairos.optimize.inventory import load_inventory  # noqa: E402
from kairos.optimize.optimizer import optimize_breaks  # noqa: E402
from kairos.optimize.pacing import load_campaigns  # noqa: E402
from kairos.optimize.pricing import (  # noqa: E402
    OptimizerAssumptions,
    pricing_from_settings,
)
from kairos.export.schedule import _load_constraints  # noqa: E402
from kairos.service import (  # noqa: E402
    _pacing_knobs_from_settings,
    guardrails_from_settings,
)
from kairos_api.core import (  # noqa: E402
    _load_settings,
    _model_dump,
    _reference_today,
)

OUT = Path(__file__).resolve().parent
OBJECTIVE_MODE = "revenue_net"
RISK_LAMBDA = 0.0
EVAL_DAYS = ["2024-11-01", "2024-11-02", "2024-11-03"]
EVAL_CHANNELS = ["כאן 11", "קשת 12", "רשת 13", "עכשיו 14"]


def _fb_default_ok(assumptions: OptimizerAssumptions) -> float:
    return assumptions.first_break_multiplier


def make_impact_model(coefs, assumptions):
    """A measured PosteriorImpactModel from a channel_coefficients() result dict."""
    estimates = {
        name: RetentionEstimate(
            coefficient=c.coefficient,
            ci_low=c.ci_low,
            ci_high=c.ci_high,
            n=c.n,
            confidence="medium",
        )
        for name, c in coefs.items()
    }
    return PosteriorImpactModel(
        {name: c.coefficient for name, c in coefs.items()},
        default=assumptions.retention_impact_per_break,
        source="measured",
        detail=estimates,
    )


def fit_coefficients(effects: pd.DataFrame):
    """Refit EB pooling + first-break multiplier on a (resampled) effects frame."""
    coefs = channel_coefficients(effects)
    gate = first_break_gate(effects)
    return coefs, float(gate["first_break_multiplier"])


def truth_net(seg, k: int) -> tuple[float, float]:
    """(revenue, cost) in ILS on the engine-exact truth basis at k breaks."""
    revenue = _segment_revenue(seg, k)
    cost = (
        seg.cpp
        * seg.premium
        * seg.baseline_tvr
        * (1.0 - _segment_retention(seg, k))
        * (k * seg.break_length_seconds)
    )
    return revenue, cost


class Harness:
    """Loads shipped resources once; optimizes any channel-day under any model."""

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

    def assumptions_with_fb(self, fb_multiplier: float) -> OptimizerAssumptions:
        from dataclasses import replace

        chosen = max(self.base_assumptions.first_break_multiplier, fb_multiplier)
        return replace(self.base_assumptions, first_break_multiplier=chosen)

    def segments_for(self, channel, day, impact_model, assumptions):
        return build_segments_from_programmes(
            self.programmes,
            self.classifier,
            self.pricing,
            assumptions=assumptions,
            impact_model=impact_model,
            channel=channel,
            day=day,
        )

    def optimize(self, segments, refine: bool):
        return _optimize_one_day(
            segments,
            guardrails=self.guardrails,
            revenue_weight=self.settings_map["revenue_weight"] / 100.0,
            risk_lambda=RISK_LAMBDA,
            demand_engine=self.demand_engine,
            inventory_pool=self.inventory_pool,
            campaigns=self.campaigns,
            pacing_today=self.pacing_today,
            pacing_knobs=self.pacing_knobs,
            constraints=self.constraints,
            operator_channel=self.operator_channel,
            refine=refine,
            objective_mode=OBJECTIVE_MODE,
            optimize_fn=optimize_breaks,
        )


def _net_of_plan(plan_counts: dict, seg_index: dict) -> float:
    total = 0.0
    for sid, k in plan_counts.items():
        seg = seg_index.get(sid)
        if seg is None:
            continue
        rev, cost = truth_net(seg, k)
        total += rev - cost
    return total


def evaluate_resample(harness, impact_r, assumptions_r, base_segindex):
    """Optimize the 12 eval days under resample coefficients; score both bases.

    ``base_segindex`` maps (channel, day) -> {segment_id: seg} valued under the
    ORIGINAL full-data coefficients, so the SAME plan counts re-score on the
    honest basis without any re-optimization.
    """
    greedy_net_r = f1_net_r = 0.0
    greedy_net_0 = f1_net_0 = 0.0
    for channel in EVAL_CHANNELS:
        for day in EVAL_DAYS:
            seg_r = harness.segments_for(channel, day, impact_r, assumptions_r)
            if not seg_r:
                continue
            idx_r = {s.segment_id: s for s in seg_r}
            idx_0 = base_segindex[(channel, day)]
            g = harness.optimize(seg_r, refine=False)
            f = harness.optimize(seg_r, refine=True)
            g_counts = {p.segment_id: p.num_breaks for p in g.segments}
            f_counts = {p.segment_id: p.num_breaks for p in f.segments}
            greedy_net_r += _net_of_plan(g_counts, idx_r)
            f1_net_r += _net_of_plan(f_counts, idx_r)
            greedy_net_0 += _net_of_plan(g_counts, idx_0)
            f1_net_0 += _net_of_plan(f_counts, idx_0)
    return greedy_net_r, f1_net_r, greedy_net_0, f1_net_0


def build_resampled_effects(effects, days, rng):
    """Draw len(days) days with replacement; concatenate blocks with unique tags."""
    picked = rng.choice(days, size=len(days), replace=True)
    blocks = []
    for rep, d in enumerate(picked):
        blk = effects[effects["day"] == d].copy()
        # Unique prog_key per replicate so a duplicated day is an independent draw
        # for the first-break gate (which groups first-vs-later by prog_key).
        blk["prog_key"] = blk["prog_key"].astype(str) + f"__r{rep}"
        blocks.append(blk)
    return pd.concat(blocks, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--time-one", action="store_true")
    ap.add_argument("--seed", type=int, default=20260707)
    args = ap.parse_args()

    t_load = time.time()
    harness = Harness()
    spots = load_spots()
    programmes = harness.programmes
    dayparts = load_dayparts()
    effects = break_effects(spots, programmes, dayparts, harness.classifier)
    effects["day"] = pd.to_datetime(effects["break_start"]).dt.strftime("%Y-%m-%d")
    days = sorted(effects["day"].dropna().unique().tolist())
    print(f"[load] {time.time()-t_load:.1f}s  effects={len(effects)}  days={len(days)}")

    # Original full-data fit (the reference / honest basis).
    coefs_0, fb_0 = fit_coefficients(effects)
    assumptions_0 = harness.assumptions_with_fb(fb_0)
    impact_0 = make_impact_model(coefs_0, assumptions_0)
    base_segindex = {}
    for channel in EVAL_CHANNELS:
        for day in EVAL_DAYS:
            seg0 = harness.segments_for(channel, day, impact_0, assumptions_0)
            base_segindex[(channel, day)] = {s.segment_id: s for s in seg0}

    # Reference point estimate: F1 gain on the 12 days under the ORIGINAL coefs.
    g0, f0, g0b, f0b = evaluate_resample(harness, impact_0, assumptions_0, base_segindex)
    ref = {
        "fb_multiplier": fb_0,
        "greedy_net_ils": g0,
        "f1_net_ils": f0,
        "f1_gain_ils": f0 - g0,
        "f1_gain_pct": (f0 - g0) / g0 * 100 if g0 else float("nan"),
    }
    print(f"[reference] greedy={g0:.0f} f1={f0:.0f} gain={f0-g0:.0f} "
          f"({ref['f1_gain_pct']:.4f}%)  fb={fb_0}")

    rng = np.random.default_rng(args.seed)

    if args.time_one:
        t0 = time.time()
        eff_r = build_resampled_effects(effects, days, rng)
        coefs_r, fb_r = fit_coefficients(eff_r)
        assumptions_r = harness.assumptions_with_fb(fb_r)
        impact_r = make_impact_model(coefs_r, assumptions_r)
        gr, fr, gr0, fr0 = evaluate_resample(
            harness, impact_r, assumptions_r, base_segindex
        )
        dt = time.time() - t0
        print(f"[time-one] one full resample cycle = {dt:.2f}s  fb_r={fb_r}")
        print(f"  nominal gain={fr-gr:.0f}  honest gain={fr0-gr0:.0f}")
        (OUT / "time_one.json").write_text(json.dumps(
            {"seconds_per_resample": dt, "reference": ref}, ensure_ascii=False, indent=2))
        return

    rows = []
    t_start = time.time()
    for i in range(args.n):
        eff_r = build_resampled_effects(effects, days, rng)
        coefs_r, fb_r = fit_coefficients(eff_r)
        assumptions_r = harness.assumptions_with_fb(fb_r)
        impact_r = make_impact_model(coefs_r, assumptions_r)
        gr, fr, gr0, fr0 = evaluate_resample(
            harness, impact_r, assumptions_r, base_segindex
        )
        nominal_ils = fr - gr
        honest_ils = fr0 - gr0
        rows.append({
            "resample": i,
            "fb_multiplier": fb_r,
            "greedy_net_r": gr,
            "f1_net_r": fr,
            "greedy_net_0": gr0,
            "f1_net_0": fr0,
            "nominal_gain_ils": nominal_ils,
            "honest_gain_ils": honest_ils,
            "nominal_gain_pct": nominal_ils / gr * 100 if gr else float("nan"),
            "honest_gain_pct": honest_ils / gr0 * 100 if gr0 else float("nan"),
        })
        if (i + 1) % 10 == 0:
            el = time.time() - t_start
            print(f"  [{i+1}/{args.n}] {el:.0f}s  "
                  f"nominal={nominal_ils:.0f} honest={honest_ils:.0f}")

    _write_outputs(rows, ref, args)


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def _write_outputs(rows, ref, args):
    csv_path = OUT / "bootstrap_results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    nom_pct = np.array([r["nominal_gain_pct"] for r in rows])
    hon_pct = np.array([r["honest_gain_pct"] for r in rows])
    nom_ils = np.array([r["nominal_gain_ils"] for r in rows])
    hon_ils = np.array([r["honest_gain_ils"] for r in rows])

    med_nom = _pct(nom_pct, 50)
    med_hon = _pct(hon_pct, 50)
    summary = {
        "n_resamples": len(rows),
        "objective_mode": OBJECTIVE_MODE,
        "risk_lambda": RISK_LAMBDA,
        "eval_channel_days": len(EVAL_CHANNELS) * len(EVAL_DAYS),
        "seed": args.seed,
        "reference_point_estimate": ref,
        "nominal_gain_pct": {"p5": _pct(nom_pct, 5), "p50": med_nom, "p95": _pct(nom_pct, 95),
                             "mean": float(np.mean(nom_pct))},
        "honest_gain_pct": {"p5": _pct(hon_pct, 5), "p50": med_hon, "p95": _pct(hon_pct, 95),
                            "mean": float(np.mean(hon_pct))},
        "nominal_gain_ils": {"p5": _pct(nom_ils, 5), "p50": _pct(nom_ils, 50),
                             "p95": _pct(nom_ils, 95), "mean": float(np.mean(nom_ils))},
        "honest_gain_ils": {"p5": _pct(hon_ils, 5), "p50": _pct(hon_ils, 50),
                            "p95": _pct(hon_ils, 95), "mean": float(np.mean(hon_ils))},
        "shrinkage_factor_median": (med_hon / med_nom) if med_nom else float("nan"),
        "shrinkage_factor_mean": (float(np.mean(hon_pct)) / float(np.mean(nom_pct))
                                  if np.mean(nom_pct) else float("nan")),
        "frac_honest_negative": float(np.mean(hon_ils < 0)),
        "frac_honest_below_half_nominal": float(np.mean(hon_ils < 0.5 * nom_ils)),
        "fb_active_frac": float(np.mean([r["fb_multiplier"] > 1.0 for r in rows])),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
