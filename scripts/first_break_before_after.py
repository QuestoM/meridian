"""Real-data before/after: a day's schedule with the first-break multiplier off vs on.

Builds segments from the real Wally daily input, runs the optimizer with the
measured impact model, once with first_break_multiplier=1.0 (off) and once with
the measured value, and prints the per-segment break counts and the day's totals.
This shows the adjustment's concrete effect: it can only lower the first break's
value, so a segment whose marginal first break was already borderline may drop a
break; reported revenue uses the realised retention either way (never inflated).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from kairos.data import ProgramClassifier
from kairos.data.loaders import load_daily_input
from kairos.data.transform import build_segments_from_daily_input
from kairos.model.impact import load_impact_model
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import OptimizerAssumptions, PricingModel

ROOT = Path(__file__).resolve().parents[1]
DAILY = ROOT / "data" / "daily_input" / "Wally_Prime_Reshet_Example_2025-04-27.csv"
COEFF = ROOT / "models" / "tv_break_coefficients.json"


def run(multiplier: float):
    classifier = ProgramClassifier.from_yaml()
    pricing = PricingModel.from_yaml()
    assumptions = OptimizerAssumptions(first_break_multiplier=multiplier)
    impact = load_impact_model(ROOT / "models" / "tv_break_posterior.pkl", assumptions=assumptions)
    daily = load_daily_input(DAILY)
    segs = build_segments_from_daily_input(daily, classifier, pricing,
                                           assumptions=assumptions, impact_model=impact)
    result = optimize_breaks(segs, Guardrails(), revenue_weight=assumptions.revenue_weight)
    counts = {p.segment_id: 0 for p in segs}
    for b in result.placements:
        counts[b.segment_id] = counts.get(b.segment_id, 0) + 1
    return segs, result, counts


def main():
    import json
    meta = json.load(open(COEFF))["metadata"]
    measured = float(meta.get("first_break_multiplier", 1.0))
    print(f"measured first_break_multiplier from JSON = {measured}")
    print(f"gate reason: {meta.get('first_break_reason')}\n")

    segs_off, res_off, counts_off = run(1.0)
    segs_on, res_on, counts_on = run(measured)

    total_off = sum(counts_off.values())
    total_on = sum(counts_on.values())
    print(f"day {segs_off[0].day}, {len(segs_off)} segments")
    print(f"total breaks placed: OFF(1.0)={total_off}  ON({measured})={total_on}  "
          f"delta={total_on - total_off}")
    print(f"objective:           OFF={res_off.objective:.5f}  ON={res_on.objective:.5f}")
    print(f"total revenue:       OFF={res_off.total_revenue:,.0f}  ON={res_on.total_revenue:,.0f}\n")

    print("per-segment break counts where they differ (first break charged extra):")
    any_diff = False
    for s in segs_off:
        a, b = counts_off[s.segment_id], counts_on.get(s.segment_id, 0)
        if a != b:
            any_diff = True
            print(f"  {s.program_type:10s} '{s.program_title[:24]:24s}' "
                  f"breaks {a} -> {b}  (coef={s.impact_coefficient:+.4f})")
    if not any_diff:
        print("  (no count changed on this day; the extra first-break cost did not flip a "
              "marginal break, but every first break's realised retention is now lower)")
    # Show realised retention of the first break for one loaded segment, both modes.
    loaded = next((s for s in segs_on if counts_on.get(s.segment_id, 0) >= 1), None)
    if loaded:
        from kairos.optimize.optimizer import _segment_retention
        off_seg = replace(loaded, first_break_multiplier=1.0)
        print(f"\nexample loaded segment '{loaded.program_title[:30]}':")
        print(f"  retention at 1 break: OFF={_segment_retention(off_seg,1):.4f}  "
              f"ON={_segment_retention(loaded,1):.4f}")


if __name__ == "__main__":
    main()
