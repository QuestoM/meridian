"""Review item 3: what is the retention model worth in the plan, in ILS?

Three retention models drive the SAME decision process (run_scenario, blend,
saved settings, refine=True) on the owned channel's representative day:

  (a) the shipped 36-cell EB-pooled model,
  (b) one global constant = the n-weighted pooled mean of the 36 cells
      (the skeptic audit's claim of what the model effectively is), and
  (c) zero retention cost (revenue-only: breaks are treated as free).

All three resulting plans are then priced under the SHIPPED coefficients as the
reference truth, with the product's own revenue-net machinery. The (a)-(b) gap
is what the 36-cell structure earns TODAY over a single constant; the (a)-(c)
gap is what having any retention model at all is worth to the plan.

Run:  /Users/home/.venvs/meridian/bin/python scripts/validation/model_value.py
"""

from __future__ import annotations

import time

import decision_uncertainty_lib as lib


def main() -> None:
    t0 = time.time()
    ctx = lib.load_context()
    point = {c: float(ctx.detail[c]["coefficient"]) for c in ctx.cells}
    mu = lib.pooled_mean(ctx.detail)
    print(f"scope: channel={ctx.channel!r} day={ctx.day}")
    print(f"pooled mean (n-weighted over 36 cells): {mu:+.6f} per break")
    print(f"effective per-class coefficients the plan decides with (36-cell model): "
          f"{ {k: round(v, 5) for k, v in lib.class_means_at_standard(point).items()} }")

    # Reference truth: the shipped coefficients (with their real intervals).
    shipped_model = lib.make_impact_model(point, ctx.detail, degenerate_ci=False)
    reference_segments = lib.build_segments(ctx, shipped_model)
    lib.verify_segment_mapping(reference_segments, point)

    models = {
        "36cell_shipped": shipped_model,
        "global_constant": lib.constant_model(mu, ctx.detail),
        "zero_retention_cost": lib.constant_model(0.0, ctx.detail),
    }

    rows: dict[str, dict[str, float]] = {}
    plans: dict[str, dict[str, int]] = {}
    for name, model in models.items():
        counts, payload = lib.reoptimize(ctx, model, refine=True)
        plans[name] = counts
        # Consistency: the harness segments under this model reprice the service's
        # own plan to the cent (guards against drift from concurrent engine edits).
        own_segments = lib.build_segments(ctx, model)
        lib.assert_revenue_consistency(payload, lib.evaluate_counts(own_segments, counts))
        evaluation = lib.evaluate_counts(reference_segments, counts)   # reference truth
        share, changed = lib.hamming_share(counts, ctx.shipped_counts)
        rows[name] = evaluation | {"hamming_vs_shipped": share, "segments_changed": float(changed)}

    if plans["36cell_shipped"] != ctx.shipped_counts:
        print("WARNING: the 36-cell replay differs from the saved CSV plan; "
              "the saved schedule is stale relative to the current engine.")

    print("\n================ ITEM 3: VALUE OF THE MODEL (evaluated at SHIPPED coefficients) ================")
    header = f"{'model':22s} {'breaks':>7s} {'gross ILS':>14s} {'ret. cost':>12s} {'net ILS':>14s} {'vs shipped plan':>16s}"
    print(header)
    for name, row in rows.items():
        print(f"{name:22s} {row['breaks']:7.0f} {row['gross_ils']:14,.0f} "
              f"{row['retention_cost_ils']:12,.0f} {row['net_ils']:14,.0f} "
              f"{row['segments_changed']:8.0f} seg ({row['hamming_vs_shipped']:.1%})")

    net_a = rows["36cell_shipped"]["net_ils"]
    net_b = rows["global_constant"]["net_ils"]
    net_c = rows["zero_retention_cost"]["net_ils"]
    print(f"\nvalue of 36-cell structure over one global constant: {net_a - net_b:+,.2f} ILS/day "
          f"({(net_a - net_b) / net_a:+.4%} of the day's net); plans differ on "
          f"{rows['global_constant']['segments_changed']:.0f} segment(s)")
    print(f"value of any retention model over ignoring retention: {net_a - net_c:+,.2f} ILS/day "
          f"({(net_a - net_c) / net_a:+.4%}); zero-cost plan places "
          f"{rows['zero_retention_cost']['breaks'] - rows['36cell_shipped']['breaks']:+.0f} extra breaks")
    print(f"weekly (x7): 36-cell over constant {7 * (net_a - net_b):+,.0f} ILS; "
          f"model over none {7 * (net_a - net_c):+,.0f} ILS")

    prov = lib.provenance()
    prov["coefficients_computed_at"] = ctx.metadata.get("computed_at")
    out = lib.write_results("model_value.json", {
        "review_item": "3 value of the model",
        "scope": {"channel": ctx.channel, "day": ctx.day},
        "reference_truth": "shipped 36-cell coefficients",
        "pooled_mean": mu,
        "effective_class_coefficients": lib.class_means_at_standard(point),
        "plans_evaluated_at_reference": rows,
        "deltas_ils_per_day": {
            "structure_36cell_minus_constant": net_a - net_b,
            "model_minus_none": net_a - net_c,
        },
        "replay_equals_shipped_csv": plans["36cell_shipped"] == ctx.shipped_counts,
        "provenance": prov,
    })
    print(f"\nresults written to {out}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
