"""Review item 4: does risk_lambda buy tail protection, and at what mean cost?

Three plans are computed on the owned channel's representative day with the
SHIPPED artifact (its real credible intervals) and the shipped decision process
(run_scenario, blend, saved settings, refine=True), differing only in
risk_lambda: 0.0 (the shipped default), 0.5, 1.0. risk_lambda enters the engine
in optimize_breaks, which replaces every segment's coefficient with
conservative_impact(point, ci_low, ci_high, risk_lambda) BEFORE allocating, so
lambda > 0 makes the whole allocation decide against a more pessimistic cost.

Each fixed plan is then priced in ILS across the SAME K seeded coefficient
draws used by decision_sensitivity.py (Normal at the point, CI-implied sd,
truncated to [-1, 0]): the drawn vector plays the role of the unknown truth.
Tail protection would show as a higher P10 of revenue-net for higher lambda;
its price is the drop in mean revenue-net.

Run:  /Users/home/.venvs/meridian/bin/python scripts/validation/risk_lambda_efficacy.py [--k 200] [--seed 42]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import decision_uncertainty_lib as lib

LAMBDAS = (0.0, 0.5, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=200, help="number of coefficient draws")
    parser.add_argument("--seed", type=int, default=42, help="rng seed (same as decision_sensitivity)")
    args = parser.parse_args()

    t0 = time.time()
    ctx = lib.load_context()
    point = {c: float(ctx.detail[c]["coefficient"]) for c in ctx.cells}
    print(f"scope: channel={ctx.channel!r} day={ctx.day}")

    # The three plans: shipped artifact (real CIs) through the real loader, only
    # risk_lambda varies. impact_model=None means run_scenario loads the shipped
    # artifact itself, so the intervals feeding conservative_impact are the real ones.
    plans: dict[float, dict[str, int]] = {}
    for lam in LAMBDAS:
        counts, payload = lib.reoptimize(ctx, impact_model=None, risk_lambda=lam, refine=True)
        plans[lam] = counts
        # NOTE: no revenue-consistency assert here for lambda > 0: the payload's
        # reported revenue is computed at the risk-ADJUSTED coefficients (the
        # engine replaces the point before the rollup), while this review prices
        # plans at point/drawn coefficients on purpose.
        print(f"  lambda={lam}: {sum(counts.values())} breaks "
              f"(payload reports {payload['summary']['total_breaks']}, "
              f"objective {payload['summary']['objective']})")

    if plans[0.0] != ctx.shipped_counts:
        share, changed = lib.hamming_share(plans[0.0], ctx.shipped_counts)
        print(f"WARNING: lambda=0 replay differs from the saved CSV on {changed} segments ({share:.1%}).")

    # Price each fixed plan at the shipped point coefficients (reference)...
    shipped_model = lib.make_impact_model(point, ctx.detail, degenerate_ci=False)
    reference_segments = lib.build_segments(ctx, shipped_model)
    at_point = {lam: lib.evaluate_counts(reference_segments, plans[lam]) for lam in LAMBDAS}

    # ...and across the SAME K drawn truths as decision_sensitivity.py.
    draws = lib.draw_coefficient_vectors(ctx.detail, args.k, seed=args.seed)
    nets: dict[float, list[float]] = {lam: [] for lam in LAMBDAS}
    for index, draw in enumerate(draws):
        model = lib.make_impact_model(draw, ctx.detail, degenerate_ci=True)
        segments = lib.build_segments(ctx, model)
        if index == 0:
            lib.verify_segment_mapping(segments, draw)
        for lam in LAMBDAS:
            nets[lam].append(lib.evaluate_counts(segments, plans[lam])["net_ils"])
        if (index + 1) % 50 == 0:
            print(f"  draw {index + 1}/{args.k} ({time.time() - t0:.0f}s elapsed)")

    print("\n================ ITEM 4: RISK_LAMBDA EFFICACY ================")
    header = (f"{'lambda':>6s} {'breaks':>7s} {'vs l=0':>12s} {'net@point':>13s} "
              f"{'mean net':>13s} {'P10 net':>13s} {'P90 net':>13s}")
    print(header)
    rows: dict[str, dict[str, float]] = {}
    base_counts = plans[0.0]
    for lam in LAMBDAS:
        arr = np.array(nets[lam])
        share, changed = lib.hamming_share(plans[lam], base_counts)
        rows[str(lam)] = {
            "breaks": float(sum(plans[lam].values())),
            "hamming_vs_lambda0_share": share,
            "segments_changed_vs_lambda0": float(changed),
            "net_at_point_ils": at_point[lam]["net_ils"],
            "gross_at_point_ils": at_point[lam]["gross_ils"],
            "retention_cost_at_point_ils": at_point[lam]["retention_cost_ils"],
            "net_mean_ils": float(arr.mean()),
            "net_p10_ils": float(np.percentile(arr, 10)),
            "net_p50_ils": float(np.percentile(arr, 50)),
            "net_p90_ils": float(np.percentile(arr, 90)),
        }
        print(f"{lam:6.1f} {sum(plans[lam].values()):7d} {changed:4.0f} seg ({share:5.1%}) "
              f"{at_point[lam]['net_ils']:13,.0f} {arr.mean():13,.0f} "
              f"{np.percentile(arr, 10):13,.0f} {np.percentile(arr, 90):13,.0f}")

    base = np.array(nets[0.0])
    print("\ndeltas vs lambda=0 (positive = better):")
    for lam in LAMBDAS[1:]:
        arr = np.array(nets[lam])
        print(f"  lambda={lam}: mean {arr.mean() - base.mean():+,.0f} ILS/day, "
              f"P10 {np.percentile(arr, 10) - np.percentile(base, 10):+,.0f}, "
              f"P90 {np.percentile(arr, 90) - np.percentile(base, 90):+,.0f}; "
              f"per-draw net difference: mean {float((arr - base).mean()):+,.0f}, "
              f"worst {float((arr - base).min()):+,.0f}, best {float((arr - base).max()):+,.0f}")

    prov = lib.provenance()
    prov["coefficients_computed_at"] = ctx.metadata.get("computed_at")
    out = lib.write_results("risk_lambda_efficacy.json", {
        "review_item": "4 risk_lambda efficacy",
        "scope": {"channel": ctx.channel, "day": ctx.day},
        "k": args.k, "seed": args.seed,
        "lambdas": list(LAMBDAS),
        "rows": rows,
        "replay_lambda0_equals_shipped_csv": plans[0.0] == ctx.shipped_counts,
        "provenance": prov,
    })
    print(f"\nresults written to {out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
