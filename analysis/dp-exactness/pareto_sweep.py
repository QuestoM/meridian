"""Time an exact Pareto frontier sweep: one DP run per revenue_weight value.

The blend objective is a linear scalarization of (revenue, tvr-weighted
retention). Sweeping revenue_weight through the exact DP yields the exact
SUPPORTED Pareto frontier of the feasible set, replacing the 7 heuristic
scenario points. This measures wall time and prints the frontier for one
representative real channel-day (the owned channel's busiest day) and for the
largest real channel-day.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dp_prototype import _contributions, _load_groups, _prep, dp_optimize  # noqa: E402
from kairos.optimize._segment_math import (  # noqa: E402
    _group_breaks,
    _segment_retention,
    _segment_revenue,
)
from kairos.optimize.guardrails import Guardrails, is_compliant  # noqa: E402

GR = Guardrails()
WEIGHTS = [i / 20.0 for i in range(21)]  # 0.00, 0.05, ..., 1.00


def sweep(channel: str, day: str, segs) -> None:
    adj, scale, tvr = _prep(segs, risk_lambda=0.0)
    t0 = time.perf_counter()
    points = []
    for w in WEIGHTS:
        contribs = _contributions(adj, scale, tvr, mode="blend", revenue_weight=w)
        counts, _ = dp_optimize(adj, contribs)
        state = {s.segment_id: k for s, k in zip(adj, counts)}
        assert is_compliant(_group_breaks(adj, state), GR)
        rev = sum(_segment_revenue(s, state[s.segment_id]) for s in adj)
        retw = sum(s.baseline_tvr * _segment_retention(s, state[s.segment_id]) for s in adj)
        points.append((w, rev, retw / tvr if tvr > 1e-9 else 1.0, sum(counts)))
    dt = time.perf_counter() - t0
    distinct = len({(round(p[1], 6), round(p[2], 9)) for p in points})
    print(f"{channel} {day} n={len(segs)}: 21-weight exact sweep in {dt:.2f}s "
          f"({dt / len(WEIGHTS):.2f}s per point), {distinct} distinct frontier points")
    for w, rev, ret, brk in points:
        print(f"  w={w:.2f} revenue={rev:14,.0f} retention={ret:.6f} breaks={brk}")


def main() -> None:
    wanted = {("עכשיו 14", "2024-11-25"), ("קשת 12", "2024-11-22")}
    for channel, day, segs in _load_groups():
        if (channel, day) in wanted:
            sweep(channel, day, segs)


if __name__ == "__main__":
    main()
