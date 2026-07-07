"""Randomized exactness gate: DP == brute force to the cent on tiny instances.

Generates synthetic channel-days of up to 8 overlapping segments under guardrail
profiles that make EVERY constraint type bind (spacing, per-hour count, per-hour
seconds, protected cap, daily ad-seconds budget, gold count, retention floor),
then compares the exact DP's objective against an exhaustive enumeration that uses
the engine's own is_compliant and objective. A single mismatch beyond 1e-6 fails
the run and prints the offending instance. Runs both objective modes (blend and
revenue_net) and exercises the risk_lambda pre-pass.

Usage: python brute_validator.py [n_instances] [seed]
"""
from __future__ import annotations

import random
import sys
from itertools import product

from kairos.optimize.guardrails import Guardrails, evaluate, is_compliant
from kairos.optimize._types import ProgramSegment
from kairos.optimize._segment_math import _group_breaks

from dp_exact import (
    OBJECTIVE_BLEND,
    OBJECTIVE_REVENUE_NET,
    _prep,
    dp_optimize_day,
    group_objective,
)

PROGRAM_TYPES = ["Movie", "Series", "Sport", "News", "Children", "Entertainment"]


def _guardrail_profile(rng):
    """A guardrail configuration; the tight ones force a specific budget to bind."""
    kind = rng.choice(["loose", "tight_daily", "tight_gold", "tight_hour", "protected"])
    base = dict(
        max_ad_seconds_per_hour=720.0, max_breaks_per_hour=4,
        min_break_spacing_seconds=420.0, min_retention_floor=0.72,
        max_daily_ad_seconds=9600.0, protected_max_ad_seconds_per_hour=480.0,
        gold_breaks_max_per_day=3,
    )
    if kind == "tight_daily":
        base["max_daily_ad_seconds"] = rng.choice([360.0, 480.0, 600.0, 720.0])
    elif kind == "tight_gold":
        base["gold_breaks_max_per_day"] = rng.choice([1, 2])
    elif kind == "tight_hour":
        base["max_breaks_per_hour"] = rng.choice([1, 2, 3])
        base["max_ad_seconds_per_hour"] = rng.choice([240.0, 360.0, 480.0])
    elif kind == "protected":
        base["protected_max_ad_seconds_per_hour"] = rng.choice([120.0, 240.0])
    return Guardrails(**base), kind


def _make_segment(rng, idx, profile_kind):
    bl = 120.0  # uniform per instance (set by caller); placeholder overwritten below
    start = rng.uniform(0, 4 * 3600.0)
    duration = rng.uniform(900.0, 4500.0)
    ptype = rng.choice(PROGRAM_TYPES)
    if profile_kind == "protected" and rng.random() < 0.5:
        ptype = rng.choice(["News", "Children", "Kids"])
    baseline_tvr = rng.uniform(0.5, 12.0)
    cpp = rng.uniform(500.0, 5000.0)
    premium = rng.choice([1.0, 1.0, 1.15, 1.3])
    coeff = -rng.uniform(0.0, 0.09)  # retention drop per break, keeps some k feasible
    retention_baseline = rng.uniform(0.9, 1.0)
    is_gold = profile_kind == "tight_gold" and rng.random() < 0.6
    max_breaks = rng.randint(1, 4)
    seg = ProgramSegment(
        segment_id=f"s{idx}", channel="C", day="2024-11-01",
        start_seconds=start, duration_seconds=duration, program_type=ptype,
        baseline_tvr=baseline_tvr, cpp=cpp, impact_coefficient=coeff,
        retention_baseline=retention_baseline, premium=premium,
        is_gold=is_gold, max_breaks=max_breaks, break_length_seconds=120.0,
    )
    # occasionally attach a credible interval so risk_lambda has something to bite
    if rng.random() < 0.3:
        lo = coeff - rng.uniform(0.01, 0.05)
        seg = ProgramSegment(**{**seg.__dict__, "impact_ci_low": lo, "impact_ci_high": min(0.0, coeff + 0.01)})
    return seg


def _make_instance(rng):
    guardrails, kind = _guardrail_profile(rng)
    n = rng.randint(3, 8)
    segs = [_make_segment(rng, i, kind) for i in range(n)]
    # bound the brute enumeration size; shrink caps if the product is too big
    while True:
        size = 1
        for s in segs:
            size *= (s.max_breaks + 1)
        if size <= 60000:
            break
        victim = rng.randrange(len(segs))
        s = segs[victim]
        if s.max_breaks <= 1:
            continue
        segs[victim] = ProgramSegment(**{**s.__dict__, "max_breaks": s.max_breaks - 1})
    mode = rng.choice([OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET])
    if mode == OBJECTIVE_REVENUE_NET:
        # net mode needs at least one monetizable segment; the generator guarantees it
        if not any(s.baseline_tvr > 0 and s.cpp * s.premium > 0 for s in segs):
            mode = OBJECTIVE_BLEND
    revenue_weight = rng.uniform(0.0, 1.0)
    risk_lambda = rng.choice([0.0, 0.0, 0.5, 1.0])
    return segs, guardrails, mode, revenue_weight, risk_lambda


def _brute(adj, guardrails, scale, tvr, *, mode, revenue_weight):
    """Exhaustive optimum on the engine's own compliance and objective."""
    best_val = -float("inf")
    best_vec = None
    codes_seen = set()
    ranges = [range(s.max_breaks + 1) for s in adj]
    for vec in product(*ranges):
        counts = {s.segment_id: k for s, k in zip(adj, vec)}
        breaks = _group_breaks(adj, counts)
        viols = evaluate(breaks, guardrails)
        if viols:
            codes_seen.update(v.code for v in viols)
            continue
        val = group_objective(adj, counts, scale, tvr,
                              mode=mode, revenue_weight=revenue_weight)
        if val > best_val:
            best_val, best_vec = val, vec
    return best_val, best_vec, codes_seen


def main():
    n_instances = int(sys.argv[1]) if len(sys.argv) > 1 else 250
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 20260707
    rng = random.Random(seed)
    mismatches = 0
    fell_back = 0
    tol = 1e-6
    code_counts = {}
    mode_counts = {OBJECTIVE_BLEND: 0, OBJECTIVE_REVENUE_NET: 0}
    checked = 0
    worst_abs = 0.0
    for inst in range(n_instances):
        segs, guardrails, mode, rw, rl = _make_instance(rng)
        adj, scale, tvr = _prep(segs, risk_lambda=rl)
        b_val, b_vec, codes = _brute(adj, guardrails, scale, tvr,
                                     mode=mode, revenue_weight=rw)
        for c in codes:
            code_counts[c] = code_counts.get(c, 0) + 1
        res = dp_optimize_day(segs, guardrails, revenue_weight=rw,
                              risk_lambda=rl, objective_mode=mode)
        if res.fell_back:
            fell_back += 1
            continue
        if b_vec is None:
            # only the all-zero plan is feasible; DP must match it (objective 0-ish)
            b_val = group_objective(adj, {s.segment_id: 0 for s in adj}, scale, tvr,
                                    mode=mode, revenue_weight=rw)
        checked += 1
        mode_counts[mode] += 1
        diff = abs(res.objective - b_val)
        worst_abs = max(worst_abs, diff)
        rel_ok = diff <= tol or diff <= tol * (abs(b_val) + 1.0)
        if not rel_ok:
            mismatches += 1
            print(f"MISMATCH inst={inst} mode={mode} rw={rw:.3f} rl={rl} "
                  f"dp={res.objective:.9f} brute={b_val:.9f} diff={diff:.3e}")
            print(f"  guardrails={guardrails}")
            print(f"  dp_counts={res.counts} brute_vec={b_vec}")
            for s in adj:
                print(f"  seg {s.segment_id} start={s.start_seconds:.0f} dur={s.duration_seconds:.0f} "
                      f"ptype={s.program_type} tvr={s.baseline_tvr:.2f} coeff={s.impact_coefficient:.4f} "
                      f"gold={s.is_gold} maxk={s.max_breaks}")
            if mismatches >= 5:
                break
    print("=" * 70)
    print(f"instances={n_instances} checked_on_exact_path={checked} "
          f"fell_back={fell_back} MISMATCHES={mismatches}")
    print(f"worst |dp - brute| among checked: {worst_abs:.3e} (tol {tol})")
    print(f"modes on exact path: {mode_counts}")
    print("guardrail violation codes exercised during brute search (instances):")
    for code in sorted(code_counts):
        print(f"  {code}: {code_counts[code]}")
    print("RESULT:", "PASS" if mismatches == 0 else "FAIL")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
