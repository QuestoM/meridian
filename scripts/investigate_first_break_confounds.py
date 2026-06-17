"""Confound checks for the first-break retention finding.

The headline contrast (first break of the show sheds ~3.6x more than later
breaks, p<0.001) could be an artifact of three confounds. This script tests each:

  C1. Is "first break" just the existing break_position="first" cell already
      in the trained taxonomy? If the position bucket explains it, no new lever
      is needed. We cross-tabulate programme-ordinal against the existing spot
      position bucket, and re-run first-vs-later WITHIN each position bucket.
  C2. Is it a programme-length confound (first breaks sit in longer shows whose
      detrend behaves differently)? We compare offset-into-programme.
  C3. Is the detrend baseline itself weaker around the first break? We look at
      the raw observed_ratio and expected_ratio separately for first vs later.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.measure import break_effects
from kairos.model.prepare import identify_breaks, pricing_class_lookup

from investigate_first_break import assign_break_ordinals, mean_ci, welch


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
    counts = joined.groupby("prog_key")["ordinal"].transform("count")
    mj = joined[counts >= 2].copy()
    mj["is_first"] = mj["ordinal"] == 1

    print(f"rows for confound analysis (programmes with >=2 measured breaks): {len(mj)}")

    # C1: cross-tab ordinal vs existing spot-position bucket
    print("\n=== C1: programme-ordinal vs existing break_position bucket ===")
    ct = pd.crosstab(np.where(mj["is_first"], "first_break", "later_break"),
                     mj["break_position"])
    print(ct.to_string())
    print("\nfirst-vs-later WITHIN each existing position bucket:")
    for pos, g in mj.groupby("break_position"):
        f = g[g["is_first"]]["log_effect"].to_numpy()
        l = g[~g["is_first"]]["log_effect"].to_numpy()
        fm, _, _, fn = mean_ci(f)
        lm, _, _, ln = mean_ci(l)
        t, p = welch(f, l)
        print(f"  position={pos:7s} first={fm:+.5f}(n={fn})  later={lm:+.5f}(n={ln})  "
              f"diff={fm-lm:+.5f}  p~={p:.4f}")

    # C3: split observed vs expected ratio for first vs later
    print("\n=== C3: raw observed vs expected ratio (detrend sanity) ===")
    for label, sub in (("first", mj[mj["is_first"]]), ("later", mj[~mj["is_first"]])):
        obs = sub["observed_ratio"].to_numpy()
        exp = sub["expected_ratio"].to_numpy()
        print(f"  {label}: observed_ratio mean={np.mean(obs):.4f}  "
              f"expected_ratio mean={np.mean(exp):.4f}  "
              f"(log diff mean={np.mean(np.log(obs)-np.log(exp)):+.5f})")
    print("  If expected_ratio is similar for first and later, the gap is in the")
    print("  observed audience drop, not a detrend artifact.")

    # Robustness: median and a sign test, not just the mean (guards against outliers)
    print("\n=== Robustness: median + share-more-negative ===")
    f = mj[mj["is_first"]]["log_effect"].to_numpy()
    l = mj[~mj["is_first"]]["log_effect"].to_numpy()
    print(f"  first  median log_effect = {np.median(f):+.5f}")
    print(f"  later  median log_effect = {np.median(l):+.5f}")
    print(f"  share of first breaks with negative effect = {np.mean(f < 0):.3f}")
    print(f"  share of later breaks with negative effect = {np.mean(l < 0):.3f}")

    # The multiplier the optimizer would apply: ratio of mean retention deltas.
    fm = float(np.mean(f)); lm = float(np.mean(l))
    if lm < 0:
        delta_first = np.exp(fm) - 1.0
        delta_later = np.exp(lm) - 1.0
        print(f"\n  implied first/later retention-cost multiplier = "
              f"{delta_first/delta_later:.3f}")
        # A more conservative multiplier: pool later as the baseline cost and ask
        # how much extra the first break sheds, additively in log space.
        print(f"  additive log gap (first - later) = {fm-lm:+.5f}  "
              f"=> first break sheds exp() factor {np.exp(fm)/np.exp(lm):.3f}x the audience-hold ratio")


if __name__ == "__main__":
    main()
