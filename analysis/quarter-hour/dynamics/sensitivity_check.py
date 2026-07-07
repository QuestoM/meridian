"""Sensitivity: does the placement verdict survive the detrend-variant fork?

Rebuilds the exact-L leave/return point profiles under each of the four rel
variants (rel_s smoothed primary, rel_c content-only, rel_g shipped global,
rel_raw undetrended) and reruns the capped/recovered placement simulation.
Writes sensitivity.json with the optimal offset and the best-vs-symmetric and
best-vs-contained-center gaps per variant per L.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from curves_and_placement import (
    L_SET, MIN_CLUSTERS_TRUST, MIN_N_TRUST, build_profile, simulate, sym_offsets,
)

OUT = Path(__file__).resolve().parent
VARIANTS = ["rel_s", "rel_c", "rel_g", "rel_raw"]


def main() -> None:
    traj = pd.read_csv(OUT / "trajectories.csv")
    meta = pd.read_csv(OUT / "breaks_meta.csv")
    df = traj.merge(meta[["break_id", "dur_touched_min", "cluster"]], on="break_id")

    out: dict[str, object] = {}
    for col in VARIANTS:
        sub_all = df[np.isfinite(df[col])]
        per_l: dict[int, object] = {}
        for L in L_SET:
            sub = sub_all[sub_all["dur_touched_min"] == L]
            lv = sub[(sub["phase"] == "leave") & (sub["offset"] < L)]
            lm = lv.groupby("offset")[col].mean().to_dict()
            rv = sub[sub["phase"] == "return"]
            g = rv.groupby("offset").agg(m=(col, "mean"), n=(col, "size"),
                                         ncl=("cluster", "nunique"))
            g = g[(g["ncl"] >= MIN_CLUSTERS_TRUST) & (g["n"] >= MIN_N_TRUST)]
            rm = {int(k): float(v) for k, v in g["m"].items()}
            prof = build_profile({int(k): float(v) for k, v in lm.items()},
                                 rm, L, "recovered", cap=True)
            sim = simulate(prof, L)
            best_i = sim["billed_per_break_minute"].idxmax()
            best = float(sim.loc[best_i, "billed_per_break_minute"])
            sym = float(max(sim[sim["offset"].isin(sym_offsets(L))]["billed_per_break_minute"]))
            center_off = max(0, (15 - L) // 2)
            center = float(sim[sim["offset"] == center_off]["billed_per_break_minute"].iloc[0])
            per_l[int(L)] = {
                "opt_offset": int(sim.loc[best_i, "offset"]),
                "sym_offsets": sym_offsets(L),
                "best_minus_sym": round(best - sym, 5),
                "best_minus_center": round(best - center, 5),
            }
        out[col] = per_l

    with open(OUT / "sensitivity.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
