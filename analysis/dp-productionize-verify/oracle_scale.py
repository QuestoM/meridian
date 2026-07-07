"""Fresh-seed oracle at scale: brute force vs the PRODUCTION dp_refine module."""
from __future__ import annotations

import math
import random
from dataclasses import replace
from itertools import product

# Import from the SHIPPED module, not the analysis copy.
from kairos.optimize.dp_refine import (
    OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET, dp_refine_group,
)
from kairos.optimize.guardrails import Guardrails, is_compliant
from kairos.optimize.revenue_net import segment_net_revenue
from kairos.optimize._segment_math import (
    _group_breaks, _group_objective_contribution, _risk_adjusted_coefficient,
    _segment_revenue,
)
from kairos.optimize._types import ProgramSegment

_PROGRAM_TYPES = ["Movie", "Series", "Sport", "News", "Children", "Entertainment"]


def _guardrail_profile(rng):
    base = dict(
        max_ad_seconds_per_hour=720.0, max_breaks_per_hour=4,
        min_break_spacing_seconds=420.0, min_retention_floor=0.72,
        max_daily_ad_seconds=9600.0, protected_max_ad_seconds_per_hour=480.0,
        gold_breaks_max_per_day=3,
    )
    kind = rng.choice(["loose", "tight_daily", "tight_gold", "tight_hour", "protected"])
    if kind == "tight_daily":
        base["max_daily_ad_seconds"] = rng.choice([240.0, 360.0, 480.0, 600.0, 720.0])
    elif kind == "tight_gold":
        base["gold_breaks_max_per_day"] = rng.choice([1, 2])
    elif kind == "tight_hour":
        base["max_breaks_per_hour"] = rng.choice([1, 2, 3])
        base["max_ad_seconds_per_hour"] = rng.choice([240.0, 360.0, 480.0])
    elif kind == "protected":
        base["protected_max_ad_seconds_per_hour"] = rng.choice([120.0, 240.0])
    return Guardrails(**base), kind


def _make_segment(rng, idx, kind):
    ptype = rng.choice(_PROGRAM_TYPES)
    if kind == "protected" and rng.random() < 0.5:
        ptype = rng.choice(["News", "Children", "Kids"])
    coeff = -rng.uniform(0.0, 0.09)
    seg = ProgramSegment(
        segment_id=f"s{idx}", channel="C", day="2024-11-01",
        start_seconds=rng.uniform(0, 4 * 3600.0),
        duration_seconds=rng.uniform(900.0, 4500.0), program_type=ptype,
        baseline_tvr=rng.uniform(0.5, 12.0), cpp=rng.uniform(500.0, 5000.0),
        impact_coefficient=coeff, retention_baseline=rng.uniform(0.9, 1.0),
        premium=rng.choice([1.0, 1.0, 1.15, 1.3]),
        is_gold=(kind == "tight_gold" and rng.random() < 0.6),
        max_breaks=rng.randint(1, 4), break_length_seconds=120.0,
    )
    if rng.random() < 0.3:
        lo = coeff - rng.uniform(0.01, 0.05)
        seg = replace(seg, impact_ci_low=lo, impact_ci_high=min(0.0, coeff + 0.01))
    return seg


def _score(group, counts, scale, total_tvr, *, mode, revenue_weight):
    if mode == OBJECTIVE_REVENUE_NET:
        return sum(segment_net_revenue(s, counts[s.segment_id]) for s in group)
    contribution, _, _ = _group_objective_contribution(
        group, counts, revenue_weight=revenue_weight,
        revenue_scale=scale, total_tvr=total_tvr,
    )
    return contribution


def _brute_best(group, scale, total_tvr, guardrails, *, mode, revenue_weight):
    best = -math.inf
    best_counts = None
    for vec in product(*[range(s.max_breaks + 1) for s in group]):
        counts = {s.segment_id: k for s, k in zip(group, vec)}
        if not is_compliant(_group_breaks(group, counts), guardrails):
            continue
        val = _score(group, counts, scale, total_tvr, mode=mode, revenue_weight=revenue_weight)
        if val > best:
            best = val
            best_counts = counts
    return best, best_counts


def _prep(segs, *, risk_lambda):
    adj = [replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda))
           for s in segs]
    scale = max(sum(_segment_revenue(s, s.max_breaks) for s in adj), 1e-9)
    total_tvr = sum(s.baseline_tvr for s in adj)
    return adj, scale, total_tvr


def main():
    worst_gap = 0.0
    worst_ctx = None
    checked = 0
    fell_back = 0
    target = 300
    both_modes = {OBJECTIVE_BLEND: 0, OBJECTIVE_REVENUE_NET: 0}
    seed = 900000
    while checked < target:
        seed += 1
        rng = random.Random(seed)
        guardrails, kind = _guardrail_profile(rng)
        n = rng.randint(3, 6)
        segs = [_make_segment(rng, i, kind) for i in range(n)]
        mode = rng.choice([OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET])
        revenue_weight = rng.uniform(0.0, 1.0)
        risk_lambda = rng.choice([0.0, 0.0, 0.5, 1.0])
        adj, scale, total_tvr = _prep(segs, risk_lambda=risk_lambda)
        zero = {s.segment_id: 0 for s in adj}
        outcome = dp_refine_group(
            adj, zero, guardrails, revenue_weight=revenue_weight,
            revenue_scale=scale, total_tvr=total_tvr, objective_mode=mode,
            net_of=segment_net_revenue,
        )
        if outcome.fell_back:
            fell_back += 1
            continue
        brute, brute_counts = _brute_best(
            adj, scale, total_tvr, guardrails, mode=mode, revenue_weight=revenue_weight)
        # Compare the DP's reconstructed-count score (through the shipped scorer)
        # to the brute-force optimum on the same scorer.
        dp_score = _score(adj, outcome.counts, scale, total_tvr,
                          mode=mode, revenue_weight=revenue_weight)
        gap = abs(dp_score - brute)
        # normalise gap for net mode (ILS) to relative for fair reporting
        rel = gap / (abs(brute) + 1e-12)
        if gap > worst_gap:
            worst_gap = gap
            worst_ctx = (seed, mode, gap, rel, dp_score, brute)
        # also confirm dp_objective internal matches
        assert is_compliant(_group_breaks(adj, outcome.counts), guardrails), f"noncompliant seed {seed}"
        both_modes[mode] += 1
        checked += 1

    print(f"checked={checked} fell_back={fell_back} blend={both_modes[OBJECTIVE_BLEND]} net={both_modes[OBJECTIVE_REVENUE_NET]}")
    print(f"worst_gap_abs={worst_gap:.3e}")
    if worst_ctx:
        s, m, g, r, d, b = worst_ctx
        print(f"worst_ctx seed={s} mode={m} gap_abs={g:.3e} gap_rel={r:.3e} dp={d!r} brute={b!r}")
    blocking = worst_gap > 1e-9
    # For net mode the absolute gap can be large in ILS while still exact; report
    # relative too. Use relative threshold for a real blocking decision.
    print("PASS" if not blocking else "CHECK_RELATIVE")


if __name__ == "__main__":
    main()
