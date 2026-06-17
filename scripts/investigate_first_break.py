"""One-off investigation: can we recover a per-programme break ordinal, and is
the first break's detrended retention shedding materially different from later
breaks? Reuses measure.break_effects (detrending) and prepare primitives. No
optimizer change is made here; this only measures and prints.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.measure import break_effects
from kairos.model.prepare import identify_breaks, pricing_class_lookup


def programme_span_lookup(programmes: pd.DataFrame, classifier: ProgramClassifier):
    """(channel, day) -> ordered list of (start, end, pricing_class) programme spans."""
    return pricing_class_lookup(programmes, classifier)


def assign_break_ordinals(breaks: pd.DataFrame, lookup) -> pd.DataFrame:
    """Tag each break with the programme it falls inside and its 1-based ordinal.

    A break is matched to the programme on the same channel-day whose span
    contains it (start <= break_start and end >= break_end), exactly as
    match_break_to_programme does. Breaks inside the same programme are then
    ordered by break_start and numbered 1, 2, 3, ... A break in no programme
    (Other) gets ordinal NaN and is excluded from the first-vs-later contrast,
    because "first break of the show" is undefined without a show.
    """
    rows = []
    for row in breaks.itertuples(index=False):
        channel = str(getattr(row, "channel"))
        start = pd.Timestamp(getattr(row, "break_start"))
        end = pd.Timestamp(getattr(row, "break_end"))
        day = start.strftime("%Y-%m-%d")
        prog_key = None
        prog_start = None
        for idx, record in enumerate(lookup.get((channel, day), [])):
            s, e = record["start_dt"], record["end_dt"]
            if pd.isna(s) or pd.isna(e):
                continue
            if s <= start and e >= end:
                prog_key = (channel, day, idx)
                prog_start = s
                break
        rows.append({
            "channel": channel,
            "break_start": start.floor("min"),
            "break_end": end.floor("min"),
            "prog_key": prog_key,
            "prog_start": prog_start,
        })
    frame = pd.DataFrame(rows)
    frame["ordinal"] = np.nan
    matched = frame[frame["prog_key"].notna()].copy()
    matched = matched.sort_values(["prog_key", "break_start"])
    matched["ordinal"] = matched.groupby("prog_key").cumcount() + 1
    frame.loc[matched.index, "ordinal"] = matched["ordinal"]
    return frame


def mean_ci(values: np.ndarray):
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(np.mean(values))
    if n == 1:
        return mean, mean, mean, 1
    se = float(np.std(values, ddof=1) / np.sqrt(n))
    return mean, mean - 1.96 * se, mean + 1.96 * se, n


def welch(a: np.ndarray, b: np.ndarray):
    """Welch t-stat and a normal-approx two-sided p-value for difference of means."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (np.mean(a) - np.mean(b)) / se
    # normal approximation for the two-sided p-value
    from math import erfc, sqrt
    p = erfc(abs(t) / sqrt(2.0))
    return float(t), float(p)


def main():
    classifier = ProgramClassifier.from_yaml()
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()

    breaks = identify_breaks(spots)
    print(f"detected breaks: {len(breaks)}")

    lookup = programme_span_lookup(programmes, classifier)
    tagged = assign_break_ordinals(breaks, lookup)

    matched = tagged[tagged["ordinal"].notna()]
    print(f"breaks matched to a programme: {len(matched)} "
          f"({100*len(matched)/max(1,len(tagged)):.1f}%)")
    print(f"unmatched (no containing programme): {len(tagged) - len(matched)}")
    print("ordinal distribution (1=first break of show):")
    print(matched["ordinal"].value_counts().sort_index().head(12).to_string())

    # breaks per programme distribution
    per_prog = matched.groupby("prog_key")["ordinal"].max()
    print(f"\nprogrammes with >=1 matched break: {len(per_prog)}")
    print(f"programmes with >=2 breaks (so 'first vs later' is even defined): "
          f"{int((per_prog >= 2).sum())}")
    print("breaks-per-programme distribution:")
    print(per_prog.value_counts().sort_index().head(12).to_string())

    # Show one example programme with its ordered breaks
    multi = per_prog[per_prog >= 3]
    if len(multi):
        example = multi.index[0]
        ex = matched[matched["prog_key"] == example].sort_values("break_start")
        ch, day, idx = example
        recs = lookup[(ch, day)][idx]
        print(f"\nexample programme: channel={ch} day={day} "
              f"class={recs['pricing_class']} "
              f"span={recs['start_dt']}..{recs['end_dt']}")
        for r in ex.itertuples(index=False):
            print(f"  ordinal={int(r.ordinal)} break {r.break_start} -> {r.break_end}")

    # Now join detrended effects. break_effects keys breaks by (channel, start, end).
    effects = break_effects(spots, programmes, dayparts, classifier)
    print(f"\nmeasurable detrended effects: {len(effects)}")
    eff = effects.copy()
    eff["break_start"] = pd.to_datetime(eff["break_start"]).dt.floor("min")
    eff["break_end"] = pd.to_datetime(eff["break_end"]).dt.floor("min")
    eff["channel"] = eff["channel"].astype(str)

    joined = eff.merge(
        matched[["channel", "break_start", "break_end", "ordinal", "prog_key", "program_type"]]
        if "program_type" in matched.columns else
        matched[["channel", "break_start", "break_end", "ordinal", "prog_key"]],
        on=["channel", "break_start", "break_end"],
        how="inner",
    )
    print(f"effects joined to an ordinal: {len(joined)}")

    # Keep only programmes where first-vs-later is defined (>=2 breaks measured).
    prog_counts = joined.groupby("prog_key")["ordinal"].transform("count")
    multi_joined = joined[prog_counts >= 2].copy()

    first = multi_joined[multi_joined["ordinal"] == 1]["log_effect"].to_numpy()
    later = multi_joined[multi_joined["ordinal"] >= 2]["log_effect"].to_numpy()

    print("\n=== FIRST vs LATER (all matched programmes, detrended log_effect) ===")
    fm, fl, fh, fn = mean_ci(first)
    lm, ll, lh, ln = mean_ci(later)
    print(f"first-break  mean log_effect = {fm:+.5f}  95% CI [{fl:+.5f}, {fh:+.5f}]  n={fn}")
    print(f"later-break  mean log_effect = {lm:+.5f}  95% CI [{ll:+.5f}, {lh:+.5f}]  n={ln}")
    diff = fm - lm
    t, p = welch(first, later)
    print(f"difference (first - later) = {diff:+.5f}  Welch t={t:+.3f}  p~={p:.4f}")
    # translate the means into retention deltas (exp(x)-1) for intuition
    print(f"first as retention delta = {np.exp(fm)-1:+.5f}; "
          f"later = {np.exp(lm)-1:+.5f}; "
          f"ratio of shedding (first/later) = "
          f"{(np.exp(fm)-1)/(np.exp(lm)-1):.3f}" if lm < 0 else "n/a")

    # Within-genre contrast so it is not a genre-mix artifact.
    pt_col = "program_type_x" if "program_type_x" in multi_joined.columns else "program_type"
    if pt_col in multi_joined.columns:
        print("\n=== WITHIN-GENRE (program_type) FIRST vs LATER ===")
        for genre, g in multi_joined.groupby(pt_col):
            gf = g[g["ordinal"] == 1]["log_effect"].to_numpy()
            gl = g[g["ordinal"] >= 2]["log_effect"].to_numpy()
            gfm, _, _, gfn = mean_ci(gf)
            glm, _, _, gln = mean_ci(gl)
            gt, gp = welch(gf, gl)
            print(f"{genre:12s} first={gfm:+.5f} (n={gfn})  later={glm:+.5f} (n={gln})  "
                  f"diff={gfm-glm:+.5f}  p~={gp:.4f}")


if __name__ == "__main__":
    main()
