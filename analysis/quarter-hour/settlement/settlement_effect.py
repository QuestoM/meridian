"""Q3: how much does containment vs straddling move quarter-hour averages?

READ-ONLY over data/. Writes only under analysis/quarter-hour/settlement/.

Settlement proxy (per the refined owner rule in docs/quarter-hour-billing.md:
per SPOT, each spot bills at the average TVR of its own round quarter hour):
we approximate spots by break minutes. For each detected break:

  B      = local content level: mean TVR over minutes within 15 minutes of
           the break span that lie inside NO ad run (min_spots=1 spans).
  billed = mean over the break's minutes of the full quarter-hour-window
           average TVR of the window containing that minute (all 15 minutes
           of the window, content included).
  dilution   = billed / B          (higher = the window average is closer to
                                    content level = better billing optics)
  dip_frac   = 1 - inbreak_tvr / B (true minute-level in-break audience dip)

Observational comparison: dilution for straddled vs contained breaks inside
matched (channel, daypart, length-bin) cells; cells need >= 5 of each. This
is NOT causal: placement is chosen by schedulers, and B itself is estimated
from minutes near the break.

Mechanical bound: for a break of L minutes (L <= 15) with uniform per-minute
dip d_frac, containment depresses its window by d_frac*L/15 of B, while an
even straddle depresses each of two windows by d_frac*L/30, so the maximum
billed-rating gain from an even straddle vs containment is d_frac*L/30 of B.
Computed per length bin from the measured median dip_frac.

Output: settlement_results.json, breaks_settlement.csv.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.loaders import load_dayparts, load_spots
from kairos.model.prepare import identify_breaks

OUT = Path(__file__).resolve().parent
QH_MIN = 15
NEIGH_MIN = 15
MIN_CONTENT_MINUTES = 5
MIN_WINDOW_MINUTES = 13
MIN_CELL_EACH = 5


def minute_tvr_lookup(dayparts: pd.DataFrame) -> dict[tuple[str, int], float]:
    """(channel, epoch-minute) -> TVR. Timebands are 'H:MM' with H in 2..25."""
    frame = dayparts[dayparts["date"].notna() & dayparts["tvr"].notna()].copy()
    parts = frame["timeband"].astype(str).str.split(":", expand=True)
    hours = parts[0].astype(int)
    minutes = parts[1].astype(int)
    ts = frame["date"] + pd.to_timedelta(hours * 60 + minutes, unit="m")
    epoch_min = (ts - pd.Timestamp("1970-01-01")).dt.total_seconds() // 60
    return {
        (ch, int(m)): float(v)
        for ch, m, v in zip(frame["channel"], epoch_min, frame["tvr"])
    }


def ad_minutes_by_channel(spots: pd.DataFrame) -> dict[str, set[int]]:
    """Every floored minute lying inside ANY ad-air run (min_spots=1)."""
    runs = identify_breaks(spots, min_spots=1)
    out: dict[str, set[int]] = {}
    for row in runs.itertuples(index=False):
        ch = str(row.channel)
        s = int(pd.Timestamp(row.break_start).value // 60_000_000_000)
        e = int(pd.Timestamp(row.break_end).value // 60_000_000_000)
        out.setdefault(ch, set()).update(range(s, e + 1))
    return out


def qh_window_avg(ch: str, minute: int, tvr: dict) -> float | None:
    """Full quarter-hour-window average TVR for the window containing minute."""
    w0 = (minute // QH_MIN) * QH_MIN
    vals = [tvr.get((ch, m)) for m in range(w0, w0 + QH_MIN)]
    vals = [v for v in vals if v is not None]
    if len(vals) < MIN_WINDOW_MINUTES:
        return None
    return float(np.mean(vals))


def main() -> None:
    kb = pd.read_csv(OUT / "breaks_qh.csv")
    spots = load_spots()
    dayparts = load_dayparts()
    tvr = minute_tvr_lookup(dayparts)
    ad_min = ad_minutes_by_channel(spots)

    rows = []
    for row in kb.itertuples(index=False):
        ch = str(row.channel)
        s_min = int(row.start_s // 60)
        e_min = int((row.end_s - 1e-6) // 60)
        break_minutes = list(range(s_min, e_min + 1))

        neigh = list(range(s_min - NEIGH_MIN, s_min)) + list(range(e_min + 1, e_min + 1 + NEIGH_MIN))
        ads = ad_min.get(ch, set())
        content_vals = [
            tvr[(ch, m)] for m in neigh if m not in ads and (ch, m) in tvr
        ]
        if len(content_vals) < MIN_CONTENT_MINUTES:
            continue
        B = float(np.mean(content_vals))
        if B <= 0:
            continue

        inbreak_vals = [tvr[(ch, m)] for m in break_minutes if (ch, m) in tvr]
        if not inbreak_vals:
            continue
        inbreak = float(np.mean(inbreak_vals))

        billed_vals = [qh_window_avg(ch, m, tvr) for m in break_minutes]
        billed_vals = [v for v in billed_vals if v is not None]
        if not billed_vals:
            continue
        billed = float(np.mean(billed_vals))

        rows.append({
            "channel": ch, "daypart": row.daypart, "len_bin": row.len_bin,
            "len_min": float(row.len_min), "straddle_qh": int(row.straddle_qh),
            "n_qh_windows": int(row.n_qh_windows),
            "B_content": B, "inbreak_tvr": inbreak, "billed_qh": billed,
            "dilution": billed / B, "dip_frac": 1.0 - inbreak / B,
            "n_content_minutes": len(content_vals),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "breaks_settlement.csv", index=False)

    # Matched-cell comparison: straddled minus contained mean dilution.
    cells = []
    for (ch, dp, lb), g in df.groupby(["channel", "daypart", "len_bin"], sort=True):
        con = g[g["straddle_qh"] == 0]["dilution"]
        strad = g[g["straddle_qh"] == 1]["dilution"]
        if len(con) < MIN_CELL_EACH or len(strad) < MIN_CELL_EACH:
            continue
        cells.append({
            "channel": ch, "daypart": dp, "len_bin": lb,
            "n_contained": int(len(con)), "n_straddled": int(len(strad)),
            "dilution_contained": round(float(con.mean()), 4),
            "dilution_straddled": round(float(strad.mean()), 4),
            "diff_straddled_minus_contained": round(float(strad.mean() - con.mean()), 4),
        })
    cell_df = pd.DataFrame(cells)
    if not cell_df.empty:
        w = cell_df["n_contained"] + cell_df["n_straddled"]
        weighted = float(np.average(cell_df["diff_straddled_minus_contained"], weights=w))
        n_pos = int((cell_df["diff_straddled_minus_contained"] > 0).sum())
    else:
        weighted, n_pos = float("nan"), 0

    naive = {
        "dilution_contained_mean": round(float(df[df["straddle_qh"] == 0]["dilution"].mean()), 4),
        "dilution_straddled_mean": round(float(df[df["straddle_qh"] == 1]["dilution"].mean()), 4),
        "n_contained": int((df["straddle_qh"] == 0).sum()),
        "n_straddled": int((df["straddle_qh"] == 1).sum()),
    }

    # Mechanical bound per length bin from the measured dip.
    bound = []
    for lb, g in df[df["len_min"] <= 15].groupby("len_bin", sort=True):
        d = float(g["dip_frac"].median())
        L = float(g["len_min"].median())
        bound.append({
            "len_bin": lb, "n": int(len(g)),
            "median_dip_frac": round(d, 4), "median_len_min": round(L, 2),
            "max_qh_avg_gain_frac_of_B": round(d * L / 30.0, 5),
            "containment_window_deficit_frac_of_B": round(d * L / 15.0, 5),
        })

    results = {
        "n_breaks_measured": int(len(df)),
        "n_breaks_input": int(len(kb)),
        "naive": naive,
        "matched_cells": cells,
        "matched_weighted_diff_straddled_minus_contained": round(weighted, 5),
        "matched_cells_n": int(len(cell_df)),
        "matched_cells_positive": n_pos,
        "mechanical_bound_by_len_bin": bound,
        "dip_frac_median_all": round(float(df["dip_frac"].median()), 4),
        "dip_frac_by_len_bin": {
            str(lb): round(float(g["dip_frac"].median()), 4)
            for lb, g in df.groupby("len_bin", sort=True)
        },
    }
    with open(OUT / "settlement_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: results[k] for k in [
        "n_breaks_measured", "naive",
        "matched_weighted_diff_straddled_minus_contained",
        "matched_cells_n", "matched_cells_positive",
        "mechanical_bound_by_len_bin", "dip_frac_median_all",
    ]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
