"""Evaluate exact DP vs greedy vs greedy+F1 on the 12 real evaluation days.

The 12 channel-days are the four channels (kan 11, keshet 12, reshet 13,
akhshav 14) crossed with 2024-11-01..03, the same corpus the F1 refiner commit
7cecd35 reported on. Each channel-day is optimized three ways on IDENTICAL
settings and scored on the engine's own objective (blend contribution and
revenue_net ILS), to the cent, per day and in total, with wall times. The DP plan
is checked for engine compliance (primal feasibility) on every day.

Usage: python eval_12day.py [revenue_weight] [risk_lambda]
"""
from __future__ import annotations

import sys
import time

from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.export.schedule import DEFAULT_IMPACT_MODEL_PATH, _build_classifier
from kairos.model.impact import load_impact_model
from kairos.optimize.guardrails import Guardrails, is_compliant
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings
from kairos.optimize._segment_math import _group_breaks, _segment_revenue
from kairos.optimize.revenue_net import segment_net_revenue
from kairos.service import _apply_first_break_multiplier

from dp_exact import _prep, dp_optimize_day, group_objective

CHANNELS = ["כאן 11", "קשת 12", "רשת 13", "עכשיו 14"]
LABELS = {"כאן 11": "kan 11", "קשת 12": "keshet 12", "רשת 13": "reshet 13", "עכשיו 14": "akhshav 14"}
DAYS = ["2024-11-01", "2024-11-02", "2024-11-03"]
GR = Guardrails()


def _load():
    assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
    pricing = pricing_from_settings(None, None)
    classifier = _build_classifier()
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    programmes = load_programmes()
    out = []
    for channel in CHANNELS:
        for day in DAYS:
            segs = build_segments_from_programmes(
                programmes, classifier, pricing,
                assumptions=assumptions, impact_model=impact_model,
                channel=channel, day=day,
            )
            if segs:
                out.append((channel, day, segs))
    return out


def _net_money(segs, counts):
    """Sum of per-segment net ILS (revenue minus retention cost) at counts."""
    return sum(segment_net_revenue(s, counts[s.segment_id]) for s in segs)


def _greedy_counts(segs, *, revenue_weight, risk_lambda, mode, refine):
    t0 = time.perf_counter()
    res = optimize_breaks(segs, GR, revenue_weight=revenue_weight,
                          risk_lambda=risk_lambda, refine=refine, objective_mode=mode)
    dt = time.perf_counter() - t0
    return {p.segment_id: p.num_breaks for p in res.segments}, dt


def run(revenue_weight, risk_lambda):
    groups = _load()
    print(f"12-day evaluation  revenue_weight={revenue_weight}  risk_lambda={risk_lambda}")
    print(f"channel-days loaded: {len(groups)}")
    for mode in ("blend", "revenue_net"):
        print("=" * 100)
        unit = "blend-contribution" if mode == "blend" else "net-ILS"
        print(f"OBJECTIVE MODE = {mode}   (objective unit: {unit})")
        hdr = (f"{'channel':<11} {'day':<11} {'n':>3} "
               f"{'greedy':>16} {'greedy+F1':>16} {'DP(exact)':>16} "
               f"{'DP-F1':>14} {'tG':>6} {'tF1':>6} {'tDP':>6} {'cmp':>4}")
        print(hdr)
        tot = {"greedy": 0.0, "f1": 0.0, "dp": 0.0}
        tot_net = {"greedy": 0.0, "f1": 0.0, "dp": 0.0}
        tt = {"greedy": 0.0, "f1": 0.0, "dp": 0.0}
        worse = better = equal = 0
        noncompliant = 0
        for channel, day, segs in groups:
            adj, scale, tvr = _prep(segs, risk_lambda=risk_lambda)
            gcounts, tg = _greedy_counts(segs, revenue_weight=revenue_weight,
                                         risk_lambda=risk_lambda, mode=mode, refine=False)
            fcounts, tf = _greedy_counts(segs, revenue_weight=revenue_weight,
                                         risk_lambda=risk_lambda, mode=mode, refine=True)
            res = dp_optimize_day(segs, GR, revenue_weight=revenue_weight,
                                  risk_lambda=risk_lambda, objective_mode=mode)
            dcounts, tdp = res.counts, res.elapsed
            gv = group_objective(adj, gcounts, scale, tvr, mode=mode, revenue_weight=revenue_weight)
            fv = group_objective(adj, fcounts, scale, tvr, mode=mode, revenue_weight=revenue_weight)
            dv = group_objective(adj, dcounts, scale, tvr, mode=mode, revenue_weight=revenue_weight)
            comp = is_compliant(_group_breaks(adj, dcounts), GR) and not res.fell_back
            comp_any = is_compliant(_group_breaks(adj, dcounts), GR)
            if not comp_any:
                noncompliant += 1
            gap = dv - fv
            if gap < -1e-9:
                worse += 1
            elif gap > 1e-9:
                better += 1
            else:
                equal += 1
            tot["greedy"] += gv
            tot["f1"] += fv
            tot["dp"] += dv
            tot_net["greedy"] += _net_money(adj, gcounts)
            tot_net["f1"] += _net_money(adj, fcounts)
            tot_net["dp"] += _net_money(adj, dcounts)
            tt["greedy"] += tg
            tt["f1"] += tf
            tt["dp"] += tdp
            flag = "OK" if comp else ("FB" if res.fell_back else "BAD")
            if mode == "blend":
                print(f"{LABELS[channel]:<11} {day:<11} {len(segs):>3} "
                      f"{gv:>16.9f} {fv:>16.9f} {dv:>16.9f} "
                      f"{gap:>14.9f} {tg:>6.3f} {tf:>6.3f} {tdp:>6.3f} {flag:>4}")
            else:
                print(f"{LABELS[channel]:<11} {day:<11} {len(segs):>3} "
                      f"{gv:>16.2f} {fv:>16.2f} {dv:>16.2f} "
                      f"{gap:>14.2f} {tg:>6.3f} {tf:>6.3f} {tdp:>6.3f} {flag:>4}")
        print("-" * 100)
        if mode == "blend":
            print(f"{'TOTAL':<11} {'':<11} {'':>3} "
                  f"{tot['greedy']:>16.9f} {tot['f1']:>16.9f} {tot['dp']:>16.9f} "
                  f"{tot['dp'] - tot['f1']:>14.9f} "
                  f"{tt['greedy']:>6.2f} {tt['f1']:>6.2f} {tt['dp']:>6.2f}")
        else:
            print(f"{'TOTAL':<11} {'':<11} {'':>3} "
                  f"{tot['greedy']:>16.2f} {tot['f1']:>16.2f} {tot['dp']:>16.2f} "
                  f"{tot['dp'] - tot['f1']:>14.2f} "
                  f"{tt['greedy']:>6.2f} {tt['f1']:>6.2f} {tt['dp']:>6.2f}")
        print(f"DP vs greedy+F1 on the {mode} objective: better={better} equal={equal} worse={worse} "
              f"(worse>0 is a bug)")
        print(f"DP plans failing engine is_compliant: {noncompliant} (primal feasibility)")
        print(f"NET-ILS money basis (all three scored on net revenue, whichever objective was maximised):")
        print(f"  greedy    {tot_net['greedy']:>18.2f} ILS")
        print(f"  greedy+F1 {tot_net['f1']:>18.2f} ILS")
        print(f"  DP(exact) {tot_net['dp']:>18.2f} ILS")
        d1 = tot_net['dp'] - tot_net['f1']
        d2 = tot_net['dp'] - tot_net['greedy']
        p1 = 100.0 * d1 / tot_net['f1'] if tot_net['f1'] else 0.0
        p2 = 100.0 * d2 / tot_net['greedy'] if tot_net['greedy'] else 0.0
        print(f"  DP - greedy+F1 : {d1:>18.2f} ILS ({p1:+.2f}%)")
        print(f"  DP - greedy    : {d2:>18.2f} ILS ({p2:+.2f}%)")


if __name__ == "__main__":
    rw = float(sys.argv[1]) if len(sys.argv) > 1 else 0.6
    rl = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0
    run(rw, rl)
