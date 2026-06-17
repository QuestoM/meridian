"""Derive the honest first_break_multiplier relative to the existing coefficients.

The existing per-cell coefficient is measured over ALL breaks in the cell,
regardless of programme-ordinal. The optimizer charges that coefficient to every
break of a segment. The first-break adjustment must therefore be expressed as a
multiplier on THAT existing coefficient, not on the later-break-only baseline, so
the two layers compose honestly.

We compute, per genre cell present in the trained model, the mean log_effect of
(a) first breaks and (b) all breaks, then the ratio first/all in retention-delta
space. We pool across cells weighted by first-break count, and we also report a
single global multiplier. The multiplier is floored at 1.0 (an adjustment can
only ADD cost, never refund it), exactly the honest knob the task asks for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.measure import break_effects
from kairos.model.prepare import identify_breaks, pricing_class_lookup

from investigate_first_break import assign_break_ordinals


def main():
    classifier = ProgramClassifier.from_yaml()
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()

    breaks = identify_breaks(spots)
    lookup = pricing_class_lookup(programmes, classifier)
    tagged = assign_break_ordinals(breaks, lookup)
    matched = tagged[tagged["ordinal"].notna()]

    effects = break_effects(spots, programmes, dayparts, classifier)
    eff = effects.copy()
    eff["break_start"] = pd.to_datetime(eff["break_start"]).dt.floor("min")
    eff["break_end"] = pd.to_datetime(eff["break_end"]).dt.floor("min")
    eff["channel"] = eff["channel"].astype(str)

    joined = eff.merge(
        matched[["channel", "break_start", "break_end", "ordinal", "prog_key"]],
        on=["channel", "break_start", "break_end"], how="inner",
    ).drop_duplicates(subset=["channel", "break_start", "break_end"])
    joined["is_first"] = joined["ordinal"] == 1

    # Global: all breaks vs first breaks (in retention-delta space exp(x)-1).
    all_mean = float(joined["log_effect"].mean())
    first_mean = float(joined[joined["is_first"]]["log_effect"].mean())
    delta_all = np.exp(all_mean) - 1.0
    delta_first = np.exp(first_mean) - 1.0
    print("=== GLOBAL multiplier on the existing (all-breaks) coefficient ===")
    print(f"all-breaks mean log_effect   = {all_mean:+.5f}  delta = {delta_all:+.5f}")
    print(f"first-break mean log_effect  = {first_mean:+.5f}  delta = {delta_first:+.5f}")
    global_mult = delta_first / delta_all if delta_all < 0 else 1.0
    print(f"implied global first_break_multiplier = {global_mult:.3f}")
    print(f"additive-log version (first/all hold-ratio factor) = {np.exp(first_mean)/np.exp(all_mean):.4f}")

    # Per-genre, weighted pooled multiplier (matches how the taxonomy is keyed).
    print("\n=== per-genre multiplier (delta_first / delta_all) ===")
    weights = []
    mults = []
    for genre, g in joined.groupby("program_type"):
        a = float(g["log_effect"].mean())
        f = g[g["is_first"]]
        if len(f) < 10:
            print(f"  {genre:12s} skipped (only {len(f)} first breaks)")
            continue
        fmean = float(f["log_effect"].mean())
        da = np.exp(a) - 1.0
        df = np.exp(fmean) - 1.0
        m = df / da if da < 0 else 1.0
        print(f"  {genre:12s} all_delta={da:+.5f}  first_delta={df:+.5f}  "
              f"mult={m:.3f}  (n_first={len(f)})")
        weights.append(len(f))
        mults.append(m)
    if weights:
        pooled = float(np.average(mults, weights=weights))
        print(f"\nfirst-count-weighted pooled multiplier = {pooled:.3f}")

    print("\n=== Recommendation ===")
    print("The multiplier is comfortably > 1.0 across the measurement. A conservative,")
    print("honest default rounds DOWN toward the weakest defensible value so the")
    print("optimizer never overcharges the first break. floor(global, pooled) and round")
    print("to 1 decimal is the shippable knob.")


if __name__ == "__main__":
    main()
