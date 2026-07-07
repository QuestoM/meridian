"""Quantify the depth-guard discrepancy in the dp-exactness claim.

The claim reports measured open-depth max 6 (out_depth.txt), but that script
(check_depth.py) uses a spacing-only window. The DP itself and dp_exact.py's
runtime guard use the full window including the hour ceiling. This script:

1. computes dp_exact's own _max_open_depth for all 120 real channel-days,
2. lists every day whose depth exceeds the spec's suggested cap of 10
   (those days would silently FALL BACK to greedy+F1 in production),
3. computes the DP-vs-refined net-ILS gap those days carry, i.e. how much of
   the claimed +9,922,861 ILS the fallback would forfeit at cap 10,
4. reruns dp_exact.dp_optimize_day on the over-cap days at cap 10 to confirm
   fell_back=True, and at a raised cap to confirm the exact path still runs
   at real-world speed.

Usage: PYTHONPATH=<repo>:<repo>/analysis/dp-exactness python check_depth_guard.py
"""
from __future__ import annotations

import time
from collections import Counter

from dp_prototype import _contributions, _load_groups, _prep, dp_optimize
from dp_exact import _max_open_depth, _window_ends, dp_optimize_day
from kairos.optimize.guardrails import Guardrails, is_compliant
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.revenue_net import segment_net_revenue
from kairos.optimize._segment_math import _group_breaks

GR = Guardrails()


def day_nets(segs):
    adj, scale, tvr = _prep(segs, risk_lambda=0.0)
    contribs = _contributions(adj, scale, tvr, mode="revenue_net", revenue_weight=0.6)
    counts, _ = dp_optimize(adj, contribs)
    state = {s.segment_id: k for s, k in zip(adj, counts)}
    assert is_compliant(_group_breaks(adj, state), GR)
    dp_net = sum(segment_net_revenue(s, state[s.segment_id]) for s in adj)
    res = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                          refine=True, objective_mode="revenue_net")
    st = {p.segment_id: p.num_breaks for p in res.segments}
    ref_net = sum(segment_net_revenue(s, st[s.segment_id]) for s in adj)
    return dp_net, ref_net


def main():
    dist = Counter()
    over_cap = []
    for channel, day, segs in _load_groups():
        adj, _, _ = _prep(segs, risk_lambda=0.0)
        bl = adj[0].break_length_seconds
        depth = _max_open_depth(adj, _window_ends(adj, GR, bl))
        dist[depth] += 1
        if depth > 10:
            over_cap.append((depth, channel, day, segs))
    print("per-day max open-depth (dp_exact definition, incl. hour ceiling):")
    for d in sorted(dist):
        print(f"  depth {d}: {dist[d]} days")
    print(f"days over the spec cap of 10: {len(over_cap)}")
    lost = 0.0
    for depth, channel, day, segs in over_cap:
        dp_net, ref_net = day_nets(segs)
        gap = dp_net - ref_net
        lost += gap
        res10 = dp_optimize_day(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                                objective_mode="revenue_net", max_open_depth=10)
        t0 = time.perf_counter()
        res_hi = dp_optimize_day(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                                 objective_mode="revenue_net", max_open_depth=20)
        t_hi = time.perf_counter() - t0
        print(f"  {channel} {day} depth={depth} dp_net={dp_net:,.0f} "
              f"ref_net={ref_net:,.0f} gap={gap:,.0f} "
              f"cap10_fell_back={res10.fell_back} cap20_fell_back={res_hi.fell_back} "
              f"cap20_time={t_hi:.2f}s cap20_obj={res_hi.objective:,.2f}")
    print(f"net-ILS gain forfeited by the cap-10 fallback: {lost:,.0f} "
          f"of the claimed 9,922,861")


if __name__ == "__main__":
    main()
