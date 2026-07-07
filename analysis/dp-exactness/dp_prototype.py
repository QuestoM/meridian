"""Exact interval-sweep DP prototype for one Kairos channel-day (free path).

Scope: no overrides, no pins, no demand weights; default guardrails; blend or
revenue_net objective; risk_lambda applied as the engine's pre-pass.

State after deciding segment j (segments sorted by start_seconds):
  (B, G, open_ks) where
    B       = total breaks placed so far (daily budget: B * 120 <= 9600)
    G       = gold breaks so far (<= 3)
    open_ks = tuple of (i, k_i) for OPEN past segments: those whose breaks can
              still interact with future segments' breaks through the spacing
              guardrail (end_i + bl/2 + 420 > start_next) or the hourly
              guardrails (hour of last possible break >= hour of start_next).

All break geometry, retention, revenue, and compliance predicates reuse the
engine's own primitives, so DP truth = engine truth by construction.

Correctness argument (verified against brute force in mode 'brute'):
  * every hour's total is checkable with all contributors present at the first
    transition where the sweep passes the hour, because a contributor stays
    open until then;
  * every globally-consecutive spacing pair is local at the transition where
    its later member is decided, because the earlier member's window covers it;
  * hour overload pruning is monotone-safe (totals only grow, the protected
    flag only lowers the cap).

Modes:
  brute  - exhaustive oracle comparison on prefixes/slices of real groups
  full   - DP vs greedy and greedy+refiner on real channel-days
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data.loaders import load_programmes  # noqa: E402
from kairos.data.transform import build_segments_from_programmes  # noqa: E402
from kairos.export.schedule import DEFAULT_IMPACT_MODEL_PATH, _build_classifier  # noqa: E402
from kairos.model.impact import load_impact_model  # noqa: E402
from kairos.optimize.guardrails import Guardrails, is_compliant  # noqa: E402
from kairos.optimize.optimizer import optimize_breaks  # noqa: E402
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings  # noqa: E402
from kairos.optimize._segment_math import (  # noqa: E402
    _group_breaks,
    _group_objective_contribution,
    _risk_adjusted_coefficient,
    _segment_break_objects,
    _segment_retention,
    _segment_revenue,
)
from kairos.service import _apply_first_break_multiplier  # noqa: E402

GR = Guardrails()
PROTECTED = frozenset(p.lower() for p in GR.protected_program_types)


def _load_groups():
    assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
    pricing = pricing_from_settings(None, None)
    classifier = _build_classifier()
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    programmes = load_programmes()
    channels = sorted(set(programmes["Channel"].dropna().astype(str)))
    days = sorted(set(programmes["start_dt"].dropna().dt.strftime("%Y-%m-%d")))
    for channel in channels:
        for day in days:
            segs = build_segments_from_programmes(
                programmes, classifier, pricing,
                assumptions=assumptions, impact_model=impact_model,
                channel=channel, day=day,
            )
            if segs:
                yield channel, day, sorted(segs, key=lambda s: s.start_seconds)


def _prep(segs, *, risk_lambda: float):
    """Risk pre-pass + run constants, mirroring optimize_breaks exactly."""
    adj = [replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda))
           for s in segs]
    revenue_scale = max(sum(_segment_revenue(s, s.max_breaks) for s in adj), 1e-9)
    total_tvr = sum(s.baseline_tvr for s in adj)
    return adj, revenue_scale, total_tvr


def _contributions(adj, revenue_scale, total_tvr, *, mode, revenue_weight):
    """contribution[i][k] on the active objective (blend linear / net ILS)."""
    if mode == "revenue_net":
        from kairos.optimize.revenue_net import segment_net_revenue
        return [[segment_net_revenue(s, k) for k in range(s.max_breaks + 1)] for s in adj]
    out = []
    for s in adj:
        row = []
        for k in range(s.max_breaks + 1):
            rev = revenue_weight * _segment_revenue(s, k) / revenue_scale
            ret = ((1.0 - revenue_weight) * s.baseline_tvr * _segment_retention(s, k)
                   / total_tvr if total_tvr > 1e-9 else 0.0)
            row.append(rev + ret)
        out.append(row)
    return out


def dp_optimize(adj, contributions):
    """Exact DP over one sorted channel-day. Returns (counts, objective)."""
    n = len(adj)
    bl = adj[0].break_length_seconds
    max_total = int(GR.max_daily_ad_seconds // bl)
    kmax = [s.max_breaks for s in adj]
    # retention floor cap per segment
    for i, s in enumerate(adj):
        cap = 0
        for k in range(s.max_breaks + 1):
            if k == 0 or _segment_retention(s, k) >= GR.min_retention_floor:
                cap = k
            else:
                break
        kmax[i] = min(kmax[i], cap)
    breaks_of = [[_segment_break_objects(s, k) for k in range(kmax[i] + 1)]
                 for i, s in enumerate(adj)]
    starts = [s.start_seconds for s in adj]
    window_end = [max(s.start_seconds + s.duration_seconds + bl / 2.0
                      + GR.min_break_spacing_seconds,
                      (int((s.start_seconds + s.duration_seconds + bl / 2.0) // 3600.0)
                       + 1) * 3600.0) for s in adj]

    def open_set(j_next_start):
        return lambda i: window_end[i] > j_next_start

    def feasible_local(local):
        """local = list of (segment_index, k). Monotone-safe pruning checks."""
        items = []
        for i, k in local:
            items.extend(breaks_of[i][k])
        items.sort(key=lambda b: b.start_seconds)
        hours = {}
        for b in items:
            sec, cnt, prot = hours.get(b.hour, (0.0, 0, False))
            hours[b.hour] = (sec + b.duration_seconds, cnt + 1,
                             prot or b.program_type.lower() in PROTECTED)
        for sec, cnt, prot in hours.values():
            if cnt > GR.max_breaks_per_hour or sec > GR.max_ad_seconds_per_hour:
                return False
            if prot and sec > GR.protected_max_ad_seconds_per_hour:
                return False
        for prev, cur in zip(items, items[1:]):
            gap = cur.start_seconds - (prev.start_seconds + prev.duration_seconds)
            if gap < GR.min_break_spacing_seconds:
                return False
        return True

    # states: {(B, G, open_ks): (value, parent_key, k_chosen)}
    states = {(0, 0, ()): (0.0, None, None)}
    trace = []  # per-step dict for reconstruction
    for j in range(n):
        next_start = starts[j + 1] if j + 1 < n else float("inf")
        keep = open_set(next_start)
        new_states = {}
        for key, (value, _, _) in states.items():
            B, G, open_ks = key
            for k in range(kmax[j] + 1):
                B2 = B + k
                if B2 > max_total:
                    break
                G2 = G + (k if adj[j].is_gold else 0)
                if G2 > GR.gold_breaks_max_per_day:
                    break
                local = list(open_ks) + [(j, k)]
                if not feasible_local(local):
                    continue
                open2 = tuple((i, ki) for i, ki in local if keep(i))
                nk = (B2, G2, open2)
                val = value + contributions[j][k]
                cur = new_states.get(nk)
                if cur is None or val > cur[0]:
                    new_states[nk] = (val, key, k)
        # dominance pruning on B within same (G, open_ks)
        by_rest = {}
        for (B, G, o), payload in new_states.items():
            by_rest.setdefault((G, o), []).append((B, payload))
        pruned = {}
        for (G, o), lst in by_rest.items():
            lst.sort()  # by B ascending
            best = -float("inf")
            for B, payload in lst:
                if payload[0] > best + 1e-12:
                    best = payload[0]
                    pruned[(B, G, o)] = payload
        trace.append(pruned)
        states = pruned
        if not states:
            raise RuntimeError(f"DP infeasible at segment {j}")

    best_key = max(states, key=lambda kk: states[kk][0])
    best_val = states[best_key][0]
    counts = [0] * n
    key = best_key
    for j in range(n - 1, -1, -1):
        val, parent, k = trace[j][key]
        counts[j] = k
        key = parent
    return counts, best_val


def group_objective(adj, counts_by_id, revenue_scale, total_tvr, *, mode, revenue_weight):
    if mode == "revenue_net":
        from kairos.optimize.revenue_net import segment_net_revenue
        return sum(segment_net_revenue(s, counts_by_id[s.segment_id]) for s in adj)
    contribution, _, _ = _group_objective_contribution(
        adj, counts_by_id,
        revenue_weight=revenue_weight, revenue_scale=revenue_scale, total_tvr=total_tvr,
    )
    return contribution


def run_brute(mode: str, revenue_weight: float, risk_lambda: float) -> None:
    """Oracle check: DP vs exhaustive enumeration on real slices."""
    from itertools import product
    groups = list(_load_groups())
    # a spread of groups + the deepest-overlap day; prefix and mid slices
    picks = [0, len(groups) // 3, 2 * len(groups) // 3, len(groups) - 1]
    cases = []
    for gi in picks:
        channel, day, segs = groups[gi]
        cases.append((channel, day, segs[:7]))
        cases.append((channel, day, segs[30:37] if len(segs) > 37 else segs[-7:]))
    for channel, day, segs in groups:
        if channel == "קשת 12" and day == "2024-11-30":
            cases.append((channel, day, segs[40:48]))  # deep-overlap region
    mismatches = 0
    for channel, day, segs in cases:
        adj, scale, tvr = _prep(segs, risk_lambda=risk_lambda)
        contribs = _contributions(adj, scale, tvr, mode=mode, revenue_weight=revenue_weight)
        counts, dp_val = dp_optimize(adj, contribs)
        # brute force with the ENGINE's own compliance + objective
        best_val, best_vec = -float("inf"), None
        ranges = [range(s.max_breaks + 1) for s in adj]
        for vec in product(*ranges):
            state = {s.segment_id: k for s, k in zip(adj, vec)}
            if not is_compliant(_group_breaks(adj, state), GR):
                continue
            val = group_objective(adj, state, scale, tvr,
                                  mode=mode, revenue_weight=revenue_weight)
            if val > best_val:
                best_val, best_vec = val, vec
        dp_state = {s.segment_id: k for s, k in zip(adj, counts)}
        dp_engine_val = group_objective(adj, dp_state, scale, tvr,
                                        mode=mode, revenue_weight=revenue_weight)
        ok_comp = is_compliant(_group_breaks(adj, dp_state), GR)
        match = abs(dp_engine_val - best_val) < 1e-9 and ok_comp
        if not match:
            mismatches += 1
        print(f"{channel} {day} n={len(segs)} DP={dp_engine_val:.9f} "
              f"brute={best_val:.9f} dp_internal={dp_val:.9f} "
              f"compliant={ok_comp} match={match} "
              f"dp_counts={list(counts)} brute_counts={list(best_vec)}")
    print(f"MISMATCHES: {mismatches} of {len(cases)}")


def run_full(mode: str, revenue_weight: float, risk_lambda: float) -> None:
    """DP vs greedy and greedy+refiner across every real channel-day."""
    total_dp_t = 0.0
    worse = 0
    better = 0
    equal = 0
    sum_gain_vs_refined = 0.0
    rows = []
    for channel, day, segs in _load_groups():
        adj, scale, tvr = _prep(segs, risk_lambda=risk_lambda)
        contribs = _contributions(adj, scale, tvr, mode=mode, revenue_weight=revenue_weight)
        t0 = time.perf_counter()
        counts, _ = dp_optimize(adj, contribs)
        dp_t = time.perf_counter() - t0
        total_dp_t += dp_t
        dp_state = {s.segment_id: k for s, k in zip(adj, counts)}
        assert is_compliant(_group_breaks(adj, dp_state), GR), "DP plan not compliant"
        dp_val = group_objective(adj, dp_state, scale, tvr,
                                 mode=mode, revenue_weight=revenue_weight)
        vals = {}
        for refine in (False, True):
            res = optimize_breaks(
                segs, GR, revenue_weight=revenue_weight, risk_lambda=risk_lambda,
                refine=refine, objective_mode=mode,
            )
            st = {p.segment_id: p.num_breaks for p in res.segments}
            vals[refine] = group_objective(adj, st, scale, tvr,
                                           mode=mode, revenue_weight=revenue_weight)
        gap = dp_val - vals[True]
        sum_gain_vs_refined += gap
        if gap < -1e-9:
            worse += 1
        elif gap > 1e-9:
            better += 1
        else:
            equal = equal + 1
        rows.append((gap, channel, day, len(segs), dp_val, vals[False], vals[True], dp_t))
    rows.sort(reverse=True)
    print(f"mode={mode} w={revenue_weight} risk={risk_lambda}")
    print(f"channel-days: {len(rows)}; DP strictly better than greedy+refiner: {better}; "
          f"equal: {equal}; DP WORSE (bug if >0): {worse}")
    print(f"total DP wall time: {total_dp_t:.2f}s "
          f"(mean {total_dp_t / max(len(rows), 1):.3f}s per channel-day)")
    print(f"sum of (DP - refined) objective gaps: {sum_gain_vs_refined:.9f}")
    print("top 8 gaps (gap, channel, day, n, dp, greedy, refined, dp_seconds):")
    for r in rows[:8]:
        print(f"  {r[0]:.9f} {r[1]} {r[2]} n={r[3]} dp={r[4]:.6f} "
              f"greedy={r[5]:.6f} refined={r[6]:.6f} t={r[7]:.3f}s")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "brute"
    mode = sys.argv[2] if len(sys.argv) > 2 else "blend"
    weight = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    risk = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    if which == "brute":
        run_brute(mode, weight, risk)
    else:
        run_full(mode, weight, risk)
