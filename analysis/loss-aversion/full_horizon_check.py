"""Full-horizon robustness check for the 12-day loss-aversion sweep.

Re-runs the key arms (baseline 1.947 w=1, honest 1.690 w=1, honest w=1.05,
honest w=1.152) over EVERY real channel-day in the data (all 4 channels x all
loaded days), and evaluates each fixed plan under both evaluation worlds, on
the same engine-exact per-break net basis as the sweep. This tests whether the
12-day findings (honest w=1 slightly beats baseline; w=1.05 reproduces the
baseline plan) hold on the whole horizon, where the historical -4.1M claim
about lowering 1.947 to ~1.69 was measured.

Output: full_horizon_matrix.csv in this directory.

Run:
    /Users/home/.venvs/meridian/bin/python analysis/loss-aversion/full_horizon_check.py
"""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import sweep_loss_aversion as sweep  # noqa: E402


def main() -> int:
    t0 = time.time()
    shared = sweep.Shared()
    all_days = tuple(sorted(
        set(shared.programmes["start_dt"].dropna().dt.strftime("%Y-%m-%d"))
    ))
    sweep.EVAL_DAYS = all_days  # widen the horizon; everything else identical
    print(f"horizon: {len(all_days)} days x {len(shared.channels)} channels")

    world_baseline = sweep.World(shared, sweep.BASELINE_MULT)
    world_honest = sweep.World(shared, sweep.HONEST_MULT)
    arms = [
        ("baseline_1.947_w1.00", world_baseline, 1.0),
        ("honest_1.690_w1.000", world_honest, 1.0),
        ("honest_1.690_w1.050", world_honest, 1.05),
        ("honest_1.690_w1.152", world_honest, 1.152),
    ]

    plans_by_arm = {}
    for name, world, w in arms:
        t = time.time()
        plans_by_arm[name] = sweep.optimize_arm(shared, world, w)
        print(f"optimized {name}: {len(plans_by_arm[name])} channel-days "
              f"in {time.time() - t:.1f}s")

    baseline_plans = plans_by_arm[arms[0][0]]
    eval_worlds = {"1.947": world_baseline, "1.690": world_honest}
    baseline_net = {
        wkey: sweep.evaluate_plan(shared, ew, baseline_plans)["net_ils"]
        for wkey, ew in eval_worlds.items()
    }

    rows = []
    for name, world, w in arms:
        plans = plans_by_arm[name]
        sim, same, total = sweep.plan_similarity(plans, baseline_plans)
        for wkey, eworld in eval_worlds.items():
            ev = sweep.evaluate_plan(shared, eworld, plans)
            rows.append({
                "arm": name, "plan_multiplier": world.multiplier, "w": w,
                "eval_world_multiplier": wkey,
                "gross_ils": round(ev["gross_ils"], 2),
                "cost_ils": round(ev["cost_ils"], 2),
                "net_ils": round(ev["net_ils"], 2),
                "delta_net_vs_baseline_ils": round(ev["net_ils"] - baseline_net[wkey], 2),
                "total_breaks": ev["total_breaks"],
                "violations_total": ev["violations_total"],
                "retention_floor_violations": ev["retention_floor_violations"],
                "identical_decision_share_vs_baseline": round(sim, 6),
                "identical_decisions": same,
                "total_decisions": total,
            })

    out = HERE / "full_horizon_matrix.csv"
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")
    print(f"total wall time {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
