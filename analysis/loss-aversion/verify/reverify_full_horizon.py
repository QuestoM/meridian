"""Adversarial re-verification of the full-horizon (120 channel-day) headline.

Re-runs baseline (1.947, w=1) and honest (1.690, w=1) over every real
channel-day using the independent machinery in reverify_12day.py, and scores
both fixed plans in the 1.947, 1.690, and shipped 1.0 evaluation worlds.
Checks the claimed +193,733 (1.947 world) and +211,181 (1.690 world) deltas
and 8631/8704 identical decisions.

Run:
  /Users/home/.venvs/meridian/bin/python analysis/loss-aversion/verify/reverify_full_horizon.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for p in (str(HERE.parents[2]), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import reverify_12day as rv  # noqa: E402


def main() -> int:
    t0 = time.time()
    shared = rv.Shared()
    all_days = tuple(sorted(set(
        shared.programmes["start_dt"].dropna().dt.strftime("%Y-%m-%d")
    )))
    print(f"horizon: {len(all_days)} days x {len(shared.channels)} channels")

    w1947 = rv.World(shared, 1.947)
    w1690 = rv.World(shared, 1.690)
    w1000 = rv.World(shared, 1.0)

    arms = [
        ("baseline_1.947_w1.00", w1947, 1.0),
        ("honest_1.690_w1.00", w1690, 1.0),
    ]
    plans_by_arm = {}
    for name, world, w in arms:
        t = time.time()
        plans_by_arm[name], ncalls = rv.optimize_arm(shared, world, w, days=all_days)
        print(f"optimized {name}: {len(plans_by_arm[name])} channel-days, "
              f"{ncalls} weighted-net calls, {time.time() - t:.1f}s")

    baseline = plans_by_arm[arms[0][0]]
    eval_worlds = {"1.947": w1947, "1.690": w1690, "1.000": w1000}
    base_net = {k: rv.evaluate_strict(shared, ew, baseline)["net"]
                for k, ew in eval_worlds.items()}

    rows = []
    for name, world, w in arms:
        plans = plans_by_arm[name]
        same, total = rv.similarity(plans, baseline)
        for wkey, ew in eval_worlds.items():
            ev = rv.evaluate_strict(shared, ew, plans)
            rows.append({
                "arm": name, "w": w, "eval_world": wkey,
                "gross_ils": round(ev["gross"], 2), "cost_ils": round(ev["cost"], 2),
                "net_ils": round(ev["net"], 2),
                "delta_vs_baseline": round(ev["net"] - base_net[wkey], 2),
                "breaks": ev["breaks"], "violations": ev["violations"],
                "identical": same, "decisions": total,
            })
            print(rows[-1])

    out = HERE / "reverify_full_horizon.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}; wall {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
