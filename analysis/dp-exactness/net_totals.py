"""Total net-ILS across all 120 real channel-days: refined engine vs exact DP.

Prints the whole-corpus totals so the DP gain can be stated as a percentage of
the refined engine's net, on the engine's own truth basis.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dp_prototype import _contributions, _load_groups, _prep, dp_optimize  # noqa: E402
from kairos.optimize.guardrails import Guardrails, is_compliant  # noqa: E402
from kairos.optimize.optimizer import optimize_breaks  # noqa: E402
from kairos.optimize.revenue_net import segment_net_revenue  # noqa: E402
from kairos.optimize._segment_math import _group_breaks  # noqa: E402

GR = Guardrails()


def main() -> None:
    tot_dp = tot_ref = tot_greedy = 0.0
    days = 0
    for channel, day, segs in _load_groups():
        adj, scale, tvr = _prep(segs, risk_lambda=0.0)
        contribs = _contributions(adj, scale, tvr, mode="revenue_net", revenue_weight=0.6)
        counts, _ = dp_optimize(adj, contribs)
        state = {s.segment_id: k for s, k in zip(adj, counts)}
        assert is_compliant(_group_breaks(adj, state), GR)
        tot_dp += sum(segment_net_revenue(s, state[s.segment_id]) for s in adj)
        for refine, bucket in ((False, "greedy"), (True, "refined")):
            res = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                                  refine=refine, objective_mode="revenue_net")
            st = {p.segment_id: p.num_breaks for p in res.segments}
            val = sum(segment_net_revenue(s, st[s.segment_id]) for s in adj)
            if refine:
                tot_ref += val
            else:
                tot_greedy += val
        days += 1
    print(f"channel-days={days}")
    print(f"greedy  net total: {tot_greedy:,.0f} ILS")
    print(f"refined net total: {tot_ref:,.0f} ILS")
    print(f"DP      net total: {tot_dp:,.0f} ILS")
    print(f"DP - refined: {tot_dp - tot_ref:,.0f} ILS ({(tot_dp / tot_ref - 1) * 100:.2f}%)")
    print(f"DP - greedy:  {tot_dp - tot_greedy:,.0f} ILS ({(tot_dp / tot_greedy - 1) * 100:.2f}%)")


if __name__ == "__main__":
    main()
