"""Review items 1+2: decision sensitivity to model uncertainty, and regret.

For K seeded draws of the 36-cell coefficient vector (Normal at the point with
the CI-implied sd, truncated to [-1, 0]):

  * re-optimize the owned channel's representative day with the SHIPPED decision
    process (kairos.service.run_scenario, blend objective, saved settings,
    refine=True) under the drawn coefficients, and
  * reprice BOTH the per-draw optimal plan and the SHIPPED plan (the saved
    weekly CSV's break counts for that day) in ILS at the drawn coefficients,
    using the product's own revenue-net machinery.

Outputs the plan-stability distribution (total breaks, gross, revenue-net,
Hamming distance from the shipped plan) and the regret distribution
(net(re-optimized) - net(shipped), both at the drawn truth), which is the money
that knowing the true coefficients would add over the current process.

Run:  /Users/home/.venvs/meridian/bin/python scripts/validation/decision_sensitivity.py [--k 200] [--seed 42]

Read-only with respect to the product: builds plans in memory via run_scenario,
never writes output/weekly_break_schedule.csv, never edits models/.
"""

from __future__ import annotations

import argparse
import time

import numpy as np

import decision_uncertainty_lib as lib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--k", type=int, default=200, help="number of coefficient draws")
    parser.add_argument("--seed", type=int, default=42, help="rng seed for the draws")
    parser.add_argument("--day", type=str, default=None, help="override the representative day")
    args = parser.parse_args()

    t0 = time.time()
    ctx = lib.load_context(day=args.day)
    print(f"scope: channel={ctx.channel!r} day={ctx.day} "
          f"(shipped: {sum(ctx.shipped_counts.values())} breaks over {len(ctx.shipped_counts)} segments, "
          f"gross {ctx.shipped_day_revenue:,.2f} ILS)")
    print(f"settings: revenue_weight={ctx.settings['revenue_weight']} "
          f"risk_lambda={ctx.settings['risk_lambda']} floor={ctx.settings['min_retention_floor']} "
          f"max_breaks_per_hour={ctx.settings['max_breaks_per_hour']}")

    # ---- Baseline: replay the shipped process at the shipped coefficients. ----
    base_counts, base_payload = lib.reoptimize(ctx)
    if base_counts != ctx.shipped_counts:
        share, changed = lib.hamming_share(base_counts, ctx.shipped_counts)
        print(f"WARNING: live replay differs from the saved CSV on {changed} segments "
              f"({share:.1%}); the saved schedule is stale relative to the current engine. "
              "Regret below is measured against the CSV plan (the plan of record).")
    shipped_model = lib.make_impact_model(
        {c: float(ctx.detail[c]["coefficient"]) for c in ctx.cells}, ctx.detail,
        degenerate_ci=False,
    )
    base_segments = lib.build_segments(ctx, shipped_model)
    lib.verify_segment_mapping(base_segments, {c: float(ctx.detail[c]["coefficient"]) for c in ctx.cells})
    base_eval = lib.evaluate_counts(base_segments, base_counts)
    lib.assert_revenue_consistency(base_payload, base_eval)
    shipped_eval_at_point = lib.evaluate_counts(base_segments, ctx.shipped_counts)
    print(f"shipped plan at shipped coefficients: gross {shipped_eval_at_point['gross_ils']:,.2f}, "
          f"retention cost {shipped_eval_at_point['retention_cost_ils']:,.2f}, "
          f"net {shipped_eval_at_point['net_ils']:,.2f} ILS")

    # ---- K draws. ----
    draws = lib.draw_coefficient_vectors(ctx.detail, args.k, seed=args.seed)
    records: list[dict[str, float]] = []
    for index, draw in enumerate(draws):
        model = lib.make_impact_model(draw, ctx.detail, degenerate_ci=True)
        counts, payload = lib.reoptimize(ctx, model, refine=True)
        segments = lib.build_segments(ctx, model)
        if index == 0 or (index + 1) % 50 == 0:
            lib.verify_segment_mapping(segments, draw)
        eval_opt = lib.evaluate_counts(segments, counts)
        lib.assert_revenue_consistency(payload, eval_opt)
        eval_shipped = lib.evaluate_counts(segments, ctx.shipped_counts)
        share, changed = lib.hamming_share(counts, ctx.shipped_counts)
        records.append({
            "breaks_opt": eval_opt["breaks"],
            "gross_opt_ils": eval_opt["gross_ils"],
            "net_opt_ils": eval_opt["net_ils"],
            "net_shipped_ils": eval_shipped["net_ils"],
            "gross_shipped_ils": eval_shipped["gross_ils"],
            "cost_shipped_ils": eval_shipped["retention_cost_ils"],
            "regret_ils": eval_opt["net_ils"] - eval_shipped["net_ils"],
            "hamming_share": share,
            "hamming_count": float(changed),
        })
        if (index + 1) % 25 == 0:
            print(f"  draw {index + 1}/{args.k} ({time.time() - t0:.0f}s elapsed)")

    def col(name: str) -> np.ndarray:
        return np.array([r[name] for r in records])

    breaks = col("breaks_opt")
    gross = col("gross_opt_ils")
    net_opt = col("net_opt_ils")
    net_shipped = col("net_shipped_ils")
    regret = col("regret_ils")
    hamming = col("hamming_share")
    hamming_n = col("hamming_count")

    print("\n================ ITEM 1: DECISION SENSITIVITY ================")
    print(f"K={args.k} draws, seed={args.seed}; per-cell sd = CI width / 3.92, truncated to [-1, 0]")
    print(f"total breaks (re-optimized): min {breaks.min():.0f} / p10 {np.percentile(breaks,10):.0f} / "
          f"median {np.median(breaks):.0f} / p90 {np.percentile(breaks,90):.0f} / max {breaks.max():.0f} "
          f"(shipped: {sum(ctx.shipped_counts.values())})")
    print(f"gross revenue (re-optimized, at draw): mean {gross.mean():,.0f}  "
          f"P10 {np.percentile(gross,10):,.0f}  P90 {np.percentile(gross,90):,.0f} ILS")
    print(f"revenue-net (re-optimized, at draw):   mean {net_opt.mean():,.0f}  "
          f"P10 {np.percentile(net_opt,10):,.0f}  P90 {np.percentile(net_opt,90):,.0f} ILS")
    print(f"revenue-net (SHIPPED plan, at draw):   mean {net_shipped.mean():,.0f}  "
          f"P10 {np.percentile(net_shipped,10):,.0f}  P90 {np.percentile(net_shipped,90):,.0f} ILS")
    print(f"Hamming vs shipped plan: mean {hamming.mean():.2%} of segments "
          f"({hamming_n.mean():.1f} of {len(ctx.shipped_counts)}), "
          f"P10 {np.percentile(hamming,10):.2%}, P90 {np.percentile(hamming,90):.2%}, "
          f"max {hamming.max():.2%}; identical plans in {(hamming_n == 0).mean():.1%} of draws")

    print("\n================ ITEM 2: REGRET UNDER MISSPECIFICATION ================")
    print(f"regret = net(per-draw re-optimized) - net(shipped plan), both priced at the drawn truth")
    print(f"mean {regret.mean():,.0f}  median {np.median(regret):,.0f}  "
          f"P90 {np.percentile(regret,90):,.0f}  max {regret.max():,.0f} ILS/day")
    print(f"zero-regret draws (plan already optimal for the process): {(np.abs(regret) < 0.005).mean():.1%}; "
          f"negative-regret draws (blend process, not net-optimal): {(regret < -0.005).mean():.1%}")
    print(f"extrapolated week (x7, owned channel): mean {7*regret.mean():,.0f}  "
          f"P90 {7*np.percentile(regret,90):,.0f} ILS")
    day_net = shipped_eval_at_point["net_ils"]
    print(f"as a share of the day's net ({day_net:,.0f} ILS): mean {regret.mean()/day_net:.3%}, "
          f"P90 {np.percentile(regret,90)/day_net:.3%}")

    prov = lib.provenance()
    prov["coefficients_computed_at"] = ctx.metadata.get("computed_at")
    out = lib.write_results("decision_sensitivity.json", {
        "review_item": "1+2 decision sensitivity and regret",
        "scope": {"channel": ctx.channel, "day": ctx.day,
                  "segments": len(ctx.shipped_counts),
                  "shipped_breaks": sum(ctx.shipped_counts.values())},
        "k": args.k, "seed": args.seed,
        "decision_process": {
            "entry": "kairos.service.run_scenario",
            "objective_mode": "blend",
            "revenue_weight": ctx.settings["revenue_weight"],
            "risk_lambda": ctx.settings["risk_lambda"],
            "refine": True,
        },
        "baseline": {
            "replay_equals_shipped_csv": base_counts == ctx.shipped_counts,
            "shipped_plan_at_point": shipped_eval_at_point,
        },
        "summary": {
            "breaks_opt": lib.percentiles(breaks, (0, 10, 50, 90, 100)) | {"mean": float(breaks.mean())},
            "gross_opt_ils": lib.percentiles(gross) | {"mean": float(gross.mean())},
            "net_opt_ils": lib.percentiles(net_opt) | {"mean": float(net_opt.mean())},
            "net_shipped_ils": lib.percentiles(net_shipped) | {"mean": float(net_shipped.mean())},
            "regret_ils_per_day": lib.percentiles(regret, (10, 50, 90, 100)) | {"mean": float(regret.mean())},
            "regret_ils_per_week_x7": {"mean": float(7 * regret.mean()),
                                       "p90": float(7 * np.percentile(regret, 90))},
            "hamming_share": lib.percentiles(hamming) | {"mean": float(hamming.mean()),
                                                          "max": float(hamming.max())},
            "hamming_segments_mean": float(hamming_n.mean()),
            "identical_plan_share": float((hamming_n == 0).mean()),
            "zero_regret_share": float((np.abs(regret) < 0.005).mean()),
            "negative_regret_share": float((regret < -0.005).mean()),
        },
        "draws": records,
        "provenance": prov,
    })
    print(f"\nresults written to {out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
