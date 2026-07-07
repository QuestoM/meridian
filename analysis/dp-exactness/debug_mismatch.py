"""Reproduce one validator mismatch and localize the DP defect."""
from __future__ import annotations

import random
import sys
from itertools import product

from kairos.optimize.guardrails import evaluate, is_compliant
from kairos.optimize._segment_math import _group_breaks, _segment_break_objects

import brute_validator as bv
from dp_exact import _prep, _window_ends, _retention_capped_kmax, group_objective, dp_optimize_day

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 20260707
target = int(sys.argv[2]) if len(sys.argv) > 2 else None
rng = random.Random(seed)

for inst in range(250):
    segs, guardrails, mode, rw, rl = bv._make_instance(rng)
    adj, scale, tvr = _prep(segs, risk_lambda=rl)
    b_val, b_vec, _ = bv._brute(adj, guardrails, scale, tvr, mode=mode, revenue_weight=rw)
    res = dp_optimize_day(segs, guardrails, revenue_weight=rw, risk_lambda=rl, objective_mode=mode)
    if b_vec is None:
        continue
    diff = abs(res.objective - b_val)
    if diff <= 1e-6 * (abs(b_val) + 1.0):
        continue
    if target is not None and inst != target:
        continue
    print(f"inst={inst} mode={mode} rw={rw:.4f} rl={rl} dp={res.objective:.4f} brute={b_val:.4f}")
    sadj = sorted(adj, key=lambda s: s.start_seconds)
    order = [s.segment_id for s in sadj]
    print("sweep order:", order)
    we = _window_ends(sadj, guardrails, sadj[0].break_length_seconds)
    kmax = _retention_capped_kmax(sadj, guardrails)
    for i, s in enumerate(sadj):
        print(f"  {s.segment_id} start={s.start_seconds:.0f} end={s.start_seconds+s.duration_seconds:.0f} "
              f"win_end={we[i]:.0f} kmax={kmax[i]}")
    bmap = {s.segment_id: k for s, k in zip(adj, b_vec)}
    print("brute counts:", bmap, "compliant=", is_compliant(_group_breaks(adj, bmap), guardrails))
    print("dp counts:", res.counts, "compliant=", is_compliant(_group_breaks(adj, res.counts), guardrails))
    # replay the DP's per-transition feasible_local along the brute plan
    protected = frozenset(p.lower() for p in guardrails.protected_program_types)
    breaks_of = {s.segment_id: [_segment_break_objects(s, k) for k in range(kmax[i] + 1)]
                 for i, s in enumerate(sadj)}

    def feasible_local(local):
        items = []
        for sid, k in local:
            items.extend(breaks_of[sid][k])
        items.sort(key=lambda b: b.start_seconds)
        hours = {}
        for b in items:
            sec, cnt, prot = hours.get(b.hour, (0.0, 0, False))
            hours[b.hour] = (sec + b.duration_seconds, cnt + 1, prot or b.program_type.lower() in protected)
        for sec, cnt, prot in hours.values():
            if cnt > guardrails.max_breaks_per_hour or sec > guardrails.max_ad_seconds_per_hour:
                return False, "hour"
            if prot and sec > guardrails.protected_max_ad_seconds_per_hour:
                return False, "protected"
        for prev, cur in zip(items, items[1:]):
            gap = cur.start_seconds - (prev.start_seconds + prev.duration_seconds)
            if gap < guardrails.min_break_spacing_seconds:
                return False, "spacing"
        return True, ""

    starts = [s.start_seconds for s in sadj]
    open_ks = []
    ok_all = True
    for j, s in enumerate(sadj):
        k = bmap[s.segment_id]
        if k > kmax[j]:
            print(f"  step {j} {s.segment_id}: brute k={k} EXCEEDS kmax={kmax[j]}  <-- pre-cap drops it")
            ok_all = False
            break
        next_start = starts[j + 1] if j + 1 < len(sadj) else float("inf")
        local = list(open_ks) + [(s.segment_id, k)]
        ok, why = feasible_local(local)
        if not ok:
            print(f"  step {j} {s.segment_id} k={k}: feasible_local REJECTS ({why}) "
                  f"open={[x[0] for x in open_ks]}")
            ok_all = False
            break
        open_ks = [(sid, ki) for sid, ki in local if we[order.index(sid)] > next_start]
    if ok_all:
        print("  brute plan passes EVERY per-transition local check -> state-merge/dominance bug")
    break
