"""Adversarial verification of the dp-exactness claim on REAL channel-days.

Independent of the DP's internals: every candidate plan is judged only by the
ENGINE's own is_compliant (kairos.optimize.guardrails) and scored by the
ENGINE's own money/objective functions (segment_net_revenue on the ORIGINAL
segments, _group_objective_contribution for blend). If any feasible plan beats
the DP plan, exactness is refuted on real data.

Attack neighbourhoods, seeded from the DP plan and from the greedy+F1 plan:
  1. exhaustive single-segment moves (every segment, every k in 0..max_breaks)
  2. exhaustive ordered-pair moves with deltas in {(+1,-1),(-1,+1),(+1,+1),(-1,-1)}
  3. randomized kicks (perturb 2-4 segments to random k) followed by a
     first-improvement hill climb of single moves, N restarts per day

Also reports: per-day DP vs refined net on the engine basis (original segments),
count of segments with positive impact_coefficient (would threaten the DP's
prefix retention-floor cap only if retention were non-monotone in k), and an
engine-compliance verdict for every DP plan.

Usage: PYTHONPATH=<repo>:<repo>/analysis/dp-exactness python attack_dp.py [mode]
"""
from __future__ import annotations

import random
import sys
import time

from dp_prototype import _contributions, _load_groups, _prep, dp_optimize, group_objective
from kairos.optimize.guardrails import Guardrails, is_compliant
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.revenue_net import segment_net_revenue
from kairos.optimize._segment_math import _group_breaks

GR = Guardrails()
TOL = 1e-6

ATTACK_DAYS = [
    ("קשת 12", "2024-11-02"),   # top net gap day
    ("קשת 12", "2024-11-27"),   # 2nd net gap day
    ("קשת 12", "2024-11-09"),   # 3rd net gap day
    ("קשת 12", "2024-11-30"),   # deepest open-depth day (6)
    ("רשת 13", "2024-11-11"),   # top blend gap outside keshet
    ("כאן 11", "2024-11-15"),   # untargeted spot check
    ("עכשיו 14", "2024-11-08"),  # untargeted spot check
]


def plan_value(segs, counts, mode, scale, tvr, revenue_weight):
    """Engine-basis score of a counts vector on the ORIGINAL segments."""
    state = {s.segment_id: k for s, k in zip(segs, counts)}
    if mode == "revenue_net":
        return sum(segment_net_revenue(s, state[s.segment_id]) for s in segs)
    return group_objective(segs, state, scale, tvr, mode=mode, revenue_weight=revenue_weight)


def compliant(segs, counts):
    state = {s.segment_id: k for s, k in zip(segs, counts)}
    return is_compliant(_group_breaks(segs, state), GR)


def attack_day(segs, mode, revenue_weight, rng, restarts=40):
    """Return (dp_val, ref_val, best_found_val, n_improvements_over_dp)."""
    adj, scale, tvr = _prep(segs, risk_lambda=0.0)
    contribs = _contributions(adj, scale, tvr, mode=mode, revenue_weight=revenue_weight)
    dp_counts, _ = dp_optimize(adj, contribs)
    assert compliant(segs, dp_counts), "DP plan fails engine is_compliant"
    dp_val = plan_value(segs, dp_counts, mode, scale, tvr, revenue_weight)

    res = optimize_breaks(segs, GR, revenue_weight=revenue_weight, risk_lambda=0.0,
                          refine=True, objective_mode=mode)
    st = {p.segment_id: p.num_breaks for p in res.segments}
    ref_counts = [st[s.segment_id] for s in segs]
    ref_val = plan_value(segs, ref_counts, mode, scale, tvr, revenue_weight)

    # per-segment value tables for a cheap improving-move filter
    n = len(segs)
    kmax = [s.max_breaks for s in segs]
    table = []
    for i, s in enumerate(segs):
        if mode == "revenue_net":
            table.append([segment_net_revenue(s, k) for k in range(kmax[i] + 1)])
        else:
            row = []
            for k in range(kmax[i] + 1):
                row.append(contribs[i][k])  # engine-identical additive share
            table.append(row)

    best_val = dp_val
    improvements = 0

    def try_counts(cur, cur_val, cand):
        nonlocal best_val, improvements
        delta = 0.0
        for i in range(n):
            if cand[i] != cur[i]:
                delta += table[i][cand[i]] - table[i][cur[i]]
        if cur_val + delta <= best_val + TOL:
            return None
        if not compliant(segs, cand):
            return None
        real = plan_value(segs, cand, mode, scale, tvr, revenue_weight)
        if real > best_val + TOL:
            improvements += 1
            best_val = real
            return real
        return None

    def exhaust_singles(base, base_val):
        for i in range(n):
            for k in range(kmax[i] + 1):
                if k == base[i]:
                    continue
                cand = list(base)
                cand[i] = k
                try_counts(base, base_val, cand)

    def exhaust_pairs(base, base_val):
        deltas = ((1, -1), (-1, 1), (1, 1), (-1, -1))
        for i in range(n):
            for j in range(i + 1, n):
                for di, dj in deltas:
                    ki, kj = base[i] + di, base[j] + dj
                    if not (0 <= ki <= kmax[i] and 0 <= kj <= kmax[j]):
                        continue
                    cand = list(base)
                    cand[i], cand[j] = ki, kj
                    try_counts(base, base_val, cand)

    def hill_climb(start):
        cur = list(start)
        if not compliant(segs, cur):
            return
        cur_val = plan_value(segs, cur, mode, scale, tvr, revenue_weight)
        improved = True
        while improved:
            improved = False
            for i in range(n):
                for k in range(kmax[i] + 1):
                    if k == cur[i]:
                        continue
                    cand = list(cur)
                    cand[i] = k
                    delta = table[i][k] - table[i][cur[i]]
                    if delta <= TOL:
                        continue
                    if compliant(segs, cand):
                        cur = cand
                        cur_val += delta
                        improved = True
                        break
                if improved:
                    break
        if cur_val > best_val + TOL:
            real = plan_value(segs, cur, mode, scale, tvr, revenue_weight)
            if real > best_val + TOL:
                nonlocal_report(real)

    def nonlocal_report(real):
        nonlocal best_val, improvements
        improvements += 1
        best_val = real

    for base, base_val in ((dp_counts, dp_val), (ref_counts, ref_val)):
        exhaust_singles(base, base_val)
        exhaust_pairs(base, base_val)

    for _ in range(restarts):
        kick = list(dp_counts)
        for _ in range(rng.randint(2, 4)):
            i = rng.randrange(n)
            kick[i] = rng.randint(0, kmax[i])
        hill_climb(kick)

    return dp_val, ref_val, best_val, improvements


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "revenue_net"
    revenue_weight = 0.6
    rng = random.Random(20260707)
    groups = {(c, d): segs for c, d, segs in _load_groups()}
    pos_coeff = sum(1 for segs in groups.values() for s in segs if s.impact_coefficient > 0)
    total_segs = sum(len(v) for v in groups.values())
    print(f"corpus: {len(groups)} channel-days, {total_segs} segments, "
          f"{pos_coeff} with impact_coefficient > 0")
    beaten = 0
    for ch, day in ATTACK_DAYS:
        segs = groups.get((ch, day))
        if segs is None:
            print(f"MISSING {ch} {day}")
            continue
        t0 = time.perf_counter()
        dp_val, ref_val, best, imp = attack_day(segs, mode, revenue_weight, rng)
        dt = time.perf_counter() - t0
        verdict = "DP-BEATEN" if best > dp_val + TOL else "dp-holds"
        if best > dp_val + TOL:
            beaten += 1
        print(f"{ch} {day} n={len(segs)} mode={mode} dp={dp_val:.4f} "
              f"refined={ref_val:.4f} attack_best={best:.4f} "
              f"improvements={imp} {verdict} t={dt:.1f}s")
    print(f"days_where_attack_beat_DP: {beaten} of {len(ATTACK_DAYS)}")


if __name__ == "__main__":
    main()
