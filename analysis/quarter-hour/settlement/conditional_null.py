"""Straddle prevalence against a programme-respecting null (owner complication 2).

READ-ONLY over data/. Writes only under analysis/quarter-hour/settlement/.

The uniform-within-hour null in prep_breaks.py ignores that breaks must sit
inside their programme, and programme junctions cluster on round minutes. Here
the null keeps each break's real length L and its real containing programme
span [s, e], draws the break start uniformly on [s, e - L], and computes the
exact probability that the break would cross a quarter-hour (or half-hour)
boundary: the measure of the union over boundaries b in (s, e) of [b - L, b),
intersected with the feasible start interval. Only breaks with a matched
programme (prog_key set) enter. Also reports how programme junctions align
with round quarter hours, which is what separates content constraints from
settlement optics.

Output: conditional_null_results.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_programmes
from kairos.model.prepare import pricing_class_lookup

OUT = Path(__file__).resolve().parent
QH = 900.0
HH = 1800.0
EPOCH = pd.Timestamp("1970-01-01")


def crossing_prob(s: float, e: float, L: float, period: float) -> float:
    """P(break of length L crosses a period boundary | start uniform on [s, e-L])."""
    lo, hi = s, e - L
    if hi <= lo:
        # Degenerate feasible set: the break barely fits; use its actual-span
        # containment: it crosses iff a boundary lies strictly inside [s, s+L).
        b = math.floor(s / period) + 1
        return 1.0 if b * period < s + L else 0.0
    first_b = (math.floor(s / period) + 1) * period
    total = 0.0
    prev_end = lo
    b = first_b
    while b < e:
        seg_lo = max(lo, b - L, prev_end)
        seg_hi = min(hi, b)
        if seg_hi > seg_lo:
            total += seg_hi - seg_lo
            prev_end = seg_hi
        b += period
    return total / (hi - lo)


def main() -> None:
    kb = pd.read_csv(OUT / "breaks_qh.csv")
    kb = kb[kb["prog_key"].notna()].copy()

    programmes = load_programmes()
    classifier = ProgramClassifier.from_yaml()
    lookup = pricing_class_lookup(programmes, classifier)

    spans = {}
    for (channel, day), records in lookup.items():
        for idx, rec in enumerate(records):
            s, e = rec["start_dt"], rec["end_dt"]
            if pd.isna(s) or pd.isna(e):
                continue
            spans[str((channel, day, idx))] = (
                (s - EPOCH).total_seconds(), (e - EPOCH).total_seconds(),
            )

    p_qh, p_hh, keep_rows = [], [], []
    for row in kb.itertuples(index=False):
        span = spans.get(str(row.prog_key))
        if span is None:
            continue
        s, e = span
        L = float(row.break_seconds)
        p_qh.append(crossing_prob(s, e, L, QH))
        p_hh.append(crossing_prob(s, e, L, HH))
        keep_rows.append(row)
    kept = pd.DataFrame(keep_rows)
    kept["p_qh"] = p_qh
    kept["p_hh"] = p_hh

    def table(frame: pd.DataFrame, group_col: str | None, obs_col: str, p_col: str) -> list[dict]:
        groups = frame.groupby(group_col, sort=True) if group_col else [("all", frame)]
        rows = []
        for group, g in groups:
            p = g[p_col].to_numpy()
            obs = int(g[obs_col].sum())
            n = len(g)
            var = float((p * (1 - p)).sum())
            z = (obs - p.sum()) / math.sqrt(var) if var > 0 else float("nan")
            rows.append({
                "group": str(group), "n": n,
                "observed_frac": round(obs / n, 4),
                "expected_conditional_frac": round(float(p.mean()), 4),
                "obs_minus_exp": round(obs / n - float(p.mean()), 4),
                "z": round(float(z), 2),
            })
        return rows

    # Programme-junction alignment with round quarter hours.
    starts = []
    for span in spans.values():
        starts.append(span[0])
    starts = np.array(starts)
    mod = starts % QH
    on_qh = float((np.minimum(mod, QH - mod) <= 60).mean())  # within 1 min of a QH mark
    mod_hh = starts % HH
    on_hh = float((np.minimum(mod_hh, HH - mod_hh) <= 60).mean())

    results = {
        "n_breaks_with_programme": int(len(kept)),
        "n_breaks_dropped_no_span": int(len(kb) - len(kept)),
        "programme_starts_within_1min_of_qh_mark_frac": round(on_qh, 4),
        "programme_starts_within_1min_of_hh_mark_frac": round(on_hh, 4),
        "qh": {
            "overall": table(kept, None, "straddle_qh", "p_qh"),
            "by_channel": table(kept, "channel", "straddle_qh", "p_qh"),
            "by_len_bin": table(kept, "len_bin", "straddle_qh", "p_qh"),
        },
        "hh": {
            "overall": table(kept, None, "straddle_hh", "p_hh"),
            "by_channel": table(kept, "channel", "straddle_hh", "p_hh"),
            "by_len_bin": table(kept, "len_bin", "straddle_hh", "p_hh"),
        },
    }
    with open(OUT / "conditional_null_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
