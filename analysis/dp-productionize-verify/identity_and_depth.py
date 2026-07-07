"""Flag-off byte identity, depth-13 no-fallback + runtime, and per-day never-worse."""
from __future__ import annotations

import time
from dataclasses import replace

import kairos.optimize.dp_refine as dp_mod
from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.model.impact import load_impact_model
from kairos.optimize.dp_refine import (
    OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET, dp_refine_group,
)
from kairos.optimize.guardrails import Guardrails, evaluate
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings
from kairos.optimize.revenue_net import segment_net_revenue
from kairos.optimize._segment_math import (
    _group_breaks, _risk_adjusted_coefficient, _segment_revenue,
)
from kairos.service import (
    DEFAULT_IMPACT_MODEL_PATH, _apply_first_break_multiplier, _build_classifier,
)

GR = Guardrails()
assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
pricing = pricing_from_settings(None, None)
classifier = _build_classifier()
impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
programmes = load_programmes()


def build(channel, day):
    return build_segments_from_programmes(
        programmes, classifier, pricing, assumptions=assumptions,
        impact_model=impact_model, channel=channel, day=day)


DEPTH13 = ("כאן 11", "2024-11-09")
LARGEST = ("קשת 12", "2024-11-22")
DAYS = [LARGEST, DEPTH13, ("רשת 13", "2024-11-03"), ("קשת 12", "2024-11-30"),
        ("עכשיו 14", "2024-11-06")]


def prep(segs, risk_lambda=0.0):
    adj = [replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda)) for s in segs]
    scale = max(sum(_segment_revenue(s, s.max_breaks) for s in adj), 1e-9)
    total_tvr = sum(s.baseline_tvr for s in adj)
    return adj, scale, total_tvr


def flag_off_identity():
    print("=== CHECK 2: flag-off byte identity (real day, largest) ===")
    segs = build(*LARGEST)
    orig = dp_mod.apply_dp_tier
    # Reference = pre-tier path: dp_refine=True but apply_dp_tier neutralised.
    dp_mod.apply_dp_tier = lambda *a, **k: None
    ref = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                          refine=True, dp_refine=True, objective_mode=OBJECTIVE_BLEND)
    dp_mod.apply_dp_tier = orig
    off = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                          refine=True, dp_refine=False, objective_mode=OBJECTIVE_BLEND)
    ref_counts = {p.segment_id: p.num_breaks for p in ref.segments}
    off_counts = {p.segment_id: p.num_breaks for p in off.segments}
    print("objective repr-equal:", repr(off.objective) == repr(ref.objective), off.objective)
    print("total_revenue repr-equal:", repr(off.total_revenue) == repr(ref.total_revenue), off.total_revenue)
    print("aggregate_retention repr-equal:", repr(off.aggregate_retention) == repr(ref.aggregate_retention))
    print("per-segment counts equal:", off_counts == ref_counts, "n=", len(off_counts))


def depth13_and_never_worse():
    print("\n=== CHECK 4: depth-13 fell_back + runtime, per-day never-worse (export objective) ===")
    for ch, day in DAYS:
        segs = build(ch, day)
        adj, scale, total_tvr = prep(segs, 0.0)
        zero = {s.segment_id: 0 for s in adj}
        for mode in (OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET):
            t0 = time.perf_counter()
            out = dp_refine_group(adj, zero, GR, revenue_weight=0.6, revenue_scale=scale,
                                  total_tvr=total_tvr, objective_mode=mode, net_of=segment_net_revenue)
            el = time.perf_counter() - t0
            tag = "DEPTH13" if (ch, day) == DEPTH13 else ""
            print(f"  {ch} {day} {mode:12s} fell_back={out.fell_back} reason={out.reason!r} "
                  f"depth={out.max_open_depth} peak={out.peak_states} elapsed={el:.3f}s {tag}")
        # never-worse through the shipped optimizer both modes
        for mode in (OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET):
            off = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                                  refine=True, dp_refine=False, objective_mode=mode)
            on = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                                 refine=True, dp_refine=True, objective_mode=mode)
            comp = not evaluate(_group_breaks(list(segs), {p.segment_id: p.num_breaks for p in on.segments}), GR)
            print(f"     never-worse {mode:12s}: on.obj={on.objective:.6f} off.obj={off.objective:.6f} "
                  f"delta={on.objective-off.objective:+.6f} compliant={comp}")


if __name__ == "__main__":
    flag_off_identity()
    depth13_and_never_worse()
