"""Leave/return curves and quarter-hour placement optimization.

Reads trajectories.csv + breaks_meta.csv (built by prepare_trajectories.py
from the real Nov-2024 minute TVR), computes:

1. Leave curve: mean detrended, pre-normalized audience at each minute offset
   from break start, pooled and per length bin, cluster-bootstrap CIs
   (clusters = programme instances, chan-day fallback).
2. Return curve: same from the first full content minute after break end.
3. Placement: for each exact touched-minute length L, slide the measured
   leave+return profile across start offsets 0..14 inside a quarter-hour
   window and evaluate (a) billed points per break minute (each break minute
   billed at its containing round-quarter-hour window's average) and (b) the
   simple mean of affected window averages. Bootstrap the optimal offset.

Outputs: leave_curve.csv, return_curve.csv, placement.csv, results.json,
fit_run.log (via tee at the call site).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent

REL_COL = "rel_s"          # primary: smoothed-typical-curve detrend baseline
SENS_COLS = ["rel_c", "rel_g", "rel_raw"]
B_BOOT = 800
SEED = 20260707
L_SET = [2, 3, 4, 5, 6, 7]  # exact touched-minute lengths for placement
RETURN_MAX = 10
MIN_CLUSTERS_TRUST = 30    # curve point trusted only with this many clusters
MIN_N_TRUST = 50
WINDOW = 15
TIMELINE = 60              # 4 quarter-hour windows
SIM_START = 15             # break start = SIM_START + offset (window W1)


def load() -> pd.DataFrame:
    traj = pd.read_csv(OUT / "trajectories.csv")
    meta = pd.read_csv(OUT / "breaks_meta.csv")
    df = traj.merge(meta[["break_id", "len_bin", "dur_touched_min", "len_min",
                          "cluster", "channel", "daypart"]], on="break_id")
    df = df[np.isfinite(df[REL_COL])].copy()
    return df


def curve_table(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols)
    out = g.agg(
        mean_rel=(REL_COL, "mean"),
        n=(REL_COL, "size"),
        n_clusters=("cluster", "nunique"),
        **{f"mean_{c}": (c, "mean") for c in SENS_COLS},
    ).reset_index()
    out["trusted"] = (out["n_clusters"] >= MIN_CLUSTERS_TRUST) & (out["n"] >= MIN_N_TRUST)
    return out


def bootstrap_curves(df: pd.DataFrame, keys: pd.Series, rng: np.random.Generator,
                     n_boot: int) -> tuple[np.ndarray, dict[object, int]]:
    """Cluster bootstrap via multiplicity weights. Returns (means matrix
    [n_boot, n_groups], group -> column index)."""
    clusters = df["cluster"].to_numpy()
    uniq, cidx = np.unique(clusters, return_inverse=True)
    ncl = len(uniq)
    gu, gidx = np.unique(keys.to_numpy(), return_inverse=True)
    key_index = {k: j for j, k in enumerate(gu)}
    rel = df[REL_COL].to_numpy()
    means = np.full((n_boot, len(gu)), np.nan)
    for b in range(n_boot):
        draw = rng.integers(0, ncl, ncl)
        wc = np.bincount(draw, minlength=ncl).astype(float)
        w = wc[cidx]
        num = np.bincount(gidx, weights=w * rel, minlength=len(gu))
        den = np.bincount(gidx, weights=w, minlength=len(gu))
        with np.errstate(invalid="ignore"):
            means[b] = np.where(den > 0, num / den, np.nan)
    return means, key_index


def attach_ci(table: pd.DataFrame, key_of_row, means: np.ndarray,
              key_index: dict[object, int]) -> pd.DataFrame:
    lo, hi = [], []
    for _, r in table.iterrows():
        j = key_index.get(key_of_row(r))
        if j is None:
            lo.append(np.nan)
            hi.append(np.nan)
            continue
        col = means[:, j]
        col = col[np.isfinite(col)]
        if len(col) < 100:
            lo.append(np.nan)
            hi.append(np.nan)
        else:
            lo.append(float(np.percentile(col, 2.5)))
            hi.append(float(np.percentile(col, 97.5)))
    table = table.copy()
    table["ci_lo"] = lo
    table["ci_hi"] = hi
    return table


def build_profile(leave_means: dict[int, float], return_means: dict[int, float],
                  L: int, fill: str, cap: bool) -> np.ndarray:
    """Deficit profile: v[0..L-1] = leave curve, v[L..] = return curve then fill.
    fill='recovered' pads with 1.0; fill='persistent' pads with last value.
    cap=True clips every value at 1.0 (deficit-only): above-pre-level audience
    (cliffhanger carry-in, post-break content ramp, selection drift) is not a
    component the scheduler credibly moves by moving the break."""
    prof = np.ones(L + RETURN_MAX)
    for k in range(L):
        prof[k] = leave_means.get(k, prof[k - 1] if k else 1.0)
    last = 1.0
    for r in range(1, RETURN_MAX + 1):
        if r in return_means:
            last = return_means[r]
            prof[L + r - 1] = last
        else:
            prof[L + r - 1] = 1.0 if fill == "recovered" else last
    return np.minimum(prof, 1.0) if cap else prof


def simulate(profile: np.ndarray, L: int) -> pd.DataFrame:
    """Slide break start across offsets 0..14 of window W1; return metrics."""
    recs = []
    for o in range(WINDOW):
        v = np.ones(TIMELINE)
        s = SIM_START + o
        end = min(TIMELINE, s + len(profile))
        v[s:end] = profile[: end - s]
        w_avg = v.reshape(-1, WINDOW).mean(axis=1)
        break_minutes = np.arange(s, min(s + L, TIMELINE))
        wins = break_minutes // WINDOW
        billed = float(w_avg[wins].mean())
        affected = float(w_avg[np.unique(wins)].mean())
        recs.append({"offset": o, "billed_per_break_minute": billed,
                     "mean_affected_windows": affected,
                     "n_affected_windows": int(len(np.unique(wins))),
                     "end_offset_in_window": int((s + L - 1) % WINDOW)})
    return pd.DataFrame(recs)


def profiles_from_frame(df: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Per exact L: leave means {offset: v}, return means {offset: v} (trusted
    points only), and trust bookkeeping."""
    leave, ret, trust = {}, {}, {}
    for L in L_SET:
        sub = df[df["dur_touched_min"] == L]
        lv = sub[(sub["phase"] == "leave")]
        lt = curve_table(lv, ["offset"])
        leave[L] = {int(r["offset"]): float(r["mean_rel"]) for _, r in lt.iterrows()}
        rv = sub[sub["phase"] == "return"]
        rt = curve_table(rv, ["offset"])
        rtr = rt[rt["trusted"]]
        ret[L] = {int(r["offset"]): float(r["mean_rel"]) for _, r in rtr.iterrows()}
        trust[L] = {
            "n_breaks": int(sub["break_id"].nunique()),
            "return_trusted_through": int(rtr["offset"].max()) if len(rtr) else 0,
            "return_offsets_total": int(rt["offset"].max()) if len(rt) else 0,
        }
    return leave, ret, trust


def sym_offsets(L: int) -> list[int]:
    """Start offsets that center the break on the W1/W2 boundary (minute 30 =
    offset 15 boundary). Break covers s..s+L-1; center at boundary means
    s = 30 - L/2, offset = 15 - L/2 (both floor/ceil for odd L)."""
    lo = 15 - (L + 1) // 2
    hi = 15 - L // 2
    return sorted({max(0, min(14, lo)), max(0, min(14, hi))})


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = load()
    results: dict[str, object] = {"rel_col": REL_COL, "b_boot": B_BOOT, "seed": SEED}

    # ---- deliverable 1 + 2: curves per bin and pooled, with cluster CIs ----
    leave_df = df[df["phase"] == "leave"].copy()
    ret_df = df[df["phase"] == "return"].copy()
    pre_df = df[df["phase"] == "pre"].copy()

    pooled_leave = curve_table(leave_df, ["offset"])
    pooled_leave.insert(0, "len_bin", "ALL")
    bin_leave = curve_table(leave_df, ["len_bin", "offset"])
    pooled_ret = curve_table(ret_df, ["offset"])
    pooled_ret.insert(0, "len_bin", "ALL")
    bin_ret = curve_table(ret_df, ["len_bin", "offset"])

    # bootstrap CIs, one pass per phase over combined (bin|ALL, offset) keys
    lv_all = pd.concat([leave_df.assign(len_bin="ALL"), leave_df])
    keys = lv_all["len_bin"].astype(str) + "|" + lv_all["offset"].astype(str)
    m, ki = bootstrap_curves(lv_all, keys, rng, B_BOOT)
    leave_out = pd.concat([pooled_leave, bin_leave])
    leave_out = attach_ci(leave_out, lambda r: f"{r['len_bin']}|{int(r['offset'])}", m, ki)
    leave_out.to_csv(OUT / "leave_curve.csv", index=False)

    rt_all = pd.concat([ret_df.assign(len_bin="ALL"), ret_df])
    keys = rt_all["len_bin"].astype(str) + "|" + rt_all["offset"].astype(str)
    m, ki = bootstrap_curves(rt_all, keys, rng, B_BOOT)
    ret_out = pd.concat([pooled_ret, bin_ret])
    ret_out = attach_ci(ret_out, lambda r: f"{r['len_bin']}|{int(r['offset'])}", m, ki)
    ret_out.to_csv(OUT / "return_curve.csv", index=False)

    # anticipation check
    pre_tab = curve_table(pre_df, ["offset"])
    results["pre_break_check"] = pre_tab.to_dict("records")

    # ---- deliverable 3: placement ----
    leave_p, ret_p, trust = profiles_from_frame(df)
    results["placement_trust"] = trust
    results["mean_len_min_by_touched"] = {
        int(L): round(float(df[df["dur_touched_min"] == L]["len_min"].mean()), 3)
        for L in L_SET
    }

    placement_rows = []
    variants = [("capped", "recovered"), ("capped", "persistent"), ("as_measured", "recovered")]
    point_opt: dict[str, dict[int, dict[str, int]]] = {}
    for cap_name, fill in variants:
        vname = f"{cap_name}|{fill}"
        point_opt[vname] = {}
        for L in L_SET:
            prof = build_profile(leave_p[L], ret_p[L], L, fill, cap=(cap_name == "capped"))
            sim = simulate(prof, L)
            sim.insert(0, "L", L)
            sim.insert(0, "fill", fill)
            sim.insert(0, "cap", cap_name)
            placement_rows.append(sim)
            point_opt[vname][L] = {
                "opt_billed": int(sim.loc[sim["billed_per_break_minute"].idxmax(), "offset"]),
                "opt_affected": int(sim.loc[sim["mean_affected_windows"].idxmax(), "offset"]),
            }
    placement = pd.concat(placement_rows)
    placement.to_csv(OUT / "placement.csv", index=False)
    results["point_optimal_offsets"] = point_opt
    results["symmetric_offsets"] = {int(L): sym_offsets(L) for L in L_SET}

    # gap between optimum and symmetric straddle / fully-contained center
    # (primary variant: capped deficit profile, recovered fill)
    gaps = {}
    for L in L_SET:
        sim = placement[(placement["cap"] == "capped")
                        & (placement["fill"] == "recovered") & (placement["L"] == L)]
        best = sim["billed_per_break_minute"].max()
        sym = max(sim[sim["offset"].isin(sym_offsets(L))]["billed_per_break_minute"])
        center_off = max(0, (15 - L) // 2)
        center = float(sim[sim["offset"] == center_off]["billed_per_break_minute"].iloc[0])
        contained = sim[sim["offset"] <= 15 - L]
        worst_contained = float(contained["billed_per_break_minute"].min()) if len(contained) else np.nan
        gaps[int(L)] = {
            "best": round(float(best), 5),
            "symmetric_straddle": round(float(sym), 5),
            "contained_center": round(center, 5),
            "worst_contained": round(worst_contained, 5),
            "best_minus_sym": round(float(best - sym), 5),
            "best_minus_center": round(float(best - center), 5),
        }
    results["placement_gaps_billed_capped_recovered"] = gaps

    # ---- bootstrap the optimal offset (primary: billed, capped, recovered) ----
    pl_rows = df[(df["dur_touched_min"].isin(L_SET))
                 & (((df["phase"] == "leave") & (df["offset"] < df["dur_touched_min"]))
                    | (df["phase"] == "return"))].copy()
    pl_keys = (pl_rows["dur_touched_min"].astype(int).astype(str) + "|"
               + pl_rows["phase"] + "|" + pl_rows["offset"].astype(str))
    means, key_index = bootstrap_curves(pl_rows, pl_keys, rng, B_BOOT)
    trusted_ret_offsets = {L: set(ret_p[L].keys()) for L in L_SET}
    opt_dist: dict[int, dict[int, int]] = {L: {} for L in L_SET}
    sym_beats = {L: 0 for L in L_SET}
    valid_draws = {L: 0 for L in L_SET}
    for b in range(B_BOOT):
        for L in L_SET:
            lm, rm = {}, {}
            ok = True
            for k in range(L):
                j = key_index.get(f"{L}|leave|{k}")
                v = means[b, j] if j is not None else np.nan
                if not np.isfinite(v):
                    ok = False
                    break
                lm[k] = float(v)
            if not ok:
                continue
            for r in sorted(trusted_ret_offsets[L]):
                j = key_index.get(f"{L}|return|{r}")
                v = means[b, j] if j is not None else np.nan
                if np.isfinite(v):
                    rm[r] = float(v)
            prof = build_profile(lm, rm, L, "recovered", cap=True)
            sim = simulate(prof, L)
            o = int(sim.loc[sim["billed_per_break_minute"].idxmax(), "offset"])
            opt_dist[L][o] = opt_dist[L].get(o, 0) + 1
            valid_draws[L] += 1
            best = float(sim["billed_per_break_minute"].max())
            sym = max(sim[sim["offset"].isin(sym_offsets(L))]["billed_per_break_minute"])
            if best > sym + 1e-12:
                sym_beats[L] += 1
    results["bootstrap_opt_offset_distribution"] = {
        int(L): dict(sorted(opt_dist[L].items())) for L in L_SET}
    results["bootstrap_valid_draws"] = {int(L): valid_draws[L] for L in L_SET}
    results["bootstrap_frac_best_beats_symmetric"] = {
        int(L): round(sym_beats[L] / valid_draws[L], 4) if valid_draws[L] else None
        for L in L_SET}

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
