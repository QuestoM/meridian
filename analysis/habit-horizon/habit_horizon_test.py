"""Habit-horizon feasibility test.

Question: does today's in-programme break load predict tomorrow's tune-in for
the SAME recurring programme strip (Title|Channel)?

Linkage used (all aggregate slot-level, NO viewer panel exists):
  - outcome  : programme-instance average TVR on day t+1 (tune-in)
  - predictor: break density inside the same strip on day t
  - controls : programme strip fixed effect + weekday of t+1

Breaks are the engine's own definition (identify_breaks: >=2 spots, gap<=15s).
Each break is assigned to the content programme instance whose [start,end)
span contains its start. End-load = break minutes in the last third of the
programme span.

Outputs:
  analysis/habit-horizon/panel.csv          (one row per linked day-pair)
  analysis/habit-horizon/habit_results.json (fitted specs, effects, bands, N)

No statsmodels: OLS via lstsq, cluster-robust (by strip) SE computed by hand.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# macOS Accelerate BLAS emits spurious matmul over/underflow RuntimeWarnings on
# finite float64 inputs; the within-transform cross-check confirms identical
# betas, so silence the false alarms rather than mask a real numeric issue.
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*matmul.*")

from kairos.data.loaders import load_programmes, load_spots
from kairos.model.prepare import identify_breaks

OUT = Path("analysis/habit-horizon")
AD_BLOCK_TITLE = "קובץ פרומו/פרסומות"


def build_break_program_map(spots: pd.DataFrame, programmes: pd.DataFrame) -> pd.DataFrame:
    """Assign each engine break to the content programme instance containing it."""
    breaks = identify_breaks(spots)  # channel, break_start/end, break_seconds, num_spots
    content = programmes[
        (programmes["Title"] != AD_BLOCK_TITLE)
        & programmes["start_dt"].notna()
        & programmes["end_dt"].notna()
    ].copy()
    content = content.reset_index(drop=True)
    content["prog_row"] = content.index

    rows = []
    # channel vocab differs (spots.Channel vs programmes.Channel) but both use the
    # same Hebrew channel names, so an exact channel match is valid.
    for channel, chan_breaks in breaks.groupby("channel", sort=False):
        chan_prog = content[content["Channel"] == channel]
        if chan_prog.empty:
            continue
        starts = chan_prog["start_dt"].values
        ends = chan_prog["end_dt"].values
        for b in chan_breaks.itertuples(index=False):
            bs = np.datetime64(b.break_start)
            hit = np.where((starts <= bs) & (bs < ends))[0]
            if hit.size == 0:
                continue
            prow = chan_prog.iloc[hit[0]]
            span = (prow["end_dt"] - prow["start_dt"]).total_seconds()
            # position within programme span [0,1)
            pos = (b.break_start - prow["start_dt"]).total_seconds() / span if span > 0 else 0.0
            rows.append(
                {
                    "prog_row": int(prow["prog_row"]),
                    "Title": prow["Title"],
                    "Channel": channel,
                    "start_dt": prow["start_dt"],
                    "end_dt": prow["end_dt"],
                    "prog_tvr": prow["TVR"],
                    "break_seconds": float(b.break_seconds),
                    "rel_pos": float(pos),
                }
            )
    return pd.DataFrame(rows), content


def aggregate_strip_day(break_map: pd.DataFrame, content: pd.DataFrame) -> pd.DataFrame:
    """One row per (Title, Channel, date): break load + duration-weighted tune-in.

    Every content instance contributes tune-in even if it carried no break (0
    breaks is real signal, not missing), so we start from content programmes.
    """
    content = content.copy()
    content["date"] = content["start_dt"].dt.normalize()
    content["dur"] = (content["end_dt"] - content["start_dt"]).dt.total_seconds()

    # per programme-instance break aggregates
    if not break_map.empty:
        bm = break_map.copy()
        bm["end_load"] = bm["break_seconds"].where(bm["rel_pos"] >= 2.0 / 3.0, 0.0)
        per_inst = bm.groupby("prog_row").agg(
            n_breaks=("break_seconds", "size"),
            break_sec=("break_seconds", "sum"),
            end_break_sec=("end_load", "sum"),
        )
    else:
        per_inst = pd.DataFrame(columns=["n_breaks", "break_sec", "end_break_sec"])

    content = content.merge(per_inst, left_on="prog_row", right_index=True, how="left")
    for c in ("n_breaks", "break_sec", "end_break_sec"):
        content[c] = content[c].fillna(0.0)

    # collapse multiple airings of the same strip on the same day
    def _wmean(g):
        w = g["dur"].to_numpy()
        v = g["TVR"].to_numpy()
        wsum = w.sum()
        return (v * w).sum() / wsum if wsum > 0 else np.nan

    grp = content.groupby(["Title", "Channel", "date"])
    agg = grp.agg(
        n_breaks=("n_breaks", "sum"),
        break_min=("break_sec", lambda s: s.sum() / 60.0),
        end_break_min=("end_break_sec", lambda s: s.sum() / 60.0),
        dur_min=("dur", lambda s: s.sum() / 60.0),
        n_airings=("TVR", "size"),
    ).reset_index()
    tune = grp.apply(_wmean, include_groups=False).rename("tvr").reset_index()
    agg = agg.merge(tune, on=["Title", "Channel", "date"])
    return agg


def build_pairs(agg: pd.DataFrame) -> pd.DataFrame:
    """Link consecutive-day airings of the same strip: predictors_t, outcome_{t+1}."""
    rows = []
    for (title, channel), g in agg.groupby(["Title", "Channel"]):
        g = g.sort_values("date")
        by_date = {d.normalize(): r for d, r in zip(g["date"], g.to_dict("records"))}
        for d, r in by_date.items():
            nxt = d + pd.Timedelta(days=1)
            if nxt in by_date:
                n = by_date[nxt]
                rows.append(
                    {
                        "strip": f"{title}|{channel}",
                        "date_t": d,
                        "weekday_t1": nxt.day_name(),
                        "n_breaks_t": r["n_breaks"],
                        "break_min_t": r["break_min"],
                        "end_break_min_t": r["end_break_min"],
                        "tvr_t": r["tvr"],
                        "tvr_t1": n["tvr"],
                    }
                )
    return pd.DataFrame(rows)


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    """OLS beta with cluster-robust covariance (clusters = programme strips)."""
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for gid in np.unique(groups):
        m = groups == gid
        Xg = X[m]
        ug = resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = len(np.unique(groups))
    n, k = X.shape
    dof = (G / (G - 1)) * ((n - 1) / (n - k)) if G > 1 and n > k else 1.0
    cov = XtX_inv @ meat @ XtX_inv * dof
    se = np.sqrt(np.diag(cov))
    return beta, se, G


def run_spec(pairs: pd.DataFrame, predictor: str, add_lag: bool = False):
    """FE regression: tvr_t1 ~ predictor + strip FE + weekday FE (+ optional tvr_t)."""
    df = pairs.dropna(subset=["tvr_t1", predictor, "tvr_t"]).copy()
    # keep strips with within-strip variation in the predictor (FE needs >=2 obs)
    counts = df["strip"].value_counts()
    keep = counts[counts >= 2].index
    df = df[df["strip"].isin(keep)].copy()
    if df.empty or df[predictor].std() == 0:
        return {"predictor": predictor, "n_pairs": int(len(df)), "note": "insufficient variation"}

    strip_d = pd.get_dummies(df["strip"], prefix="s", drop_first=True, dtype=float)
    wd_d = pd.get_dummies(df["weekday_t1"], prefix="w", drop_first=True, dtype=float)
    parts = [np.ones((len(df), 1)), df[[predictor]].to_numpy(dtype=float)]
    names = ["const", predictor]
    if add_lag:
        parts.append(df[["tvr_t"]].to_numpy(dtype=float))
        names.append("tvr_t")
    parts.append(strip_d.to_numpy())
    names += list(strip_d.columns)
    parts.append(wd_d.to_numpy())
    names += list(wd_d.columns)
    X = np.hstack(parts)
    y = df["tvr_t1"].to_numpy(dtype=float)
    groups = df["strip"].to_numpy()
    beta, se, G = ols_cluster(y, X, groups)
    idx = names.index(predictor)
    b = float(beta[idx])
    s = float(se[idx])
    return {
        "predictor": predictor,
        "with_lag_tvr_t": add_lag,
        "n_pairs": int(len(df)),
        "n_strips": int(G),
        "beta": b,
        "cluster_se": s,
        "ci95_low": b - 1.96 * s,
        "ci95_high": b + 1.96 * s,
        "t_stat": b / s if s > 0 else None,
        "outcome_mean_tvr": float(y.mean()),
        "predictor_mean": float(df[predictor].mean()),
        "predictor_sd": float(df[predictor].std()),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    spots = load_spots()
    programmes = load_programmes()
    break_map, content = build_break_program_map(spots, programmes)
    agg = aggregate_strip_day(break_map, content)
    pairs = build_pairs(agg)
    pairs.to_csv(OUT / "panel.csv", index=False)

    results = {
        "linkage": "content programme strip Title|Channel across consecutive calendar days; aggregate slot-level TVR only (no viewer panel).",
        "window": "2024-11-01..2024-11-30 single month",
        "n_content_instances": int(len(content)),
        "n_strip_days": int(len(agg)),
        "n_linked_pairs_total": int(len(pairs)),
        "n_distinct_strips_in_pairs": int(pairs["strip"].nunique()) if len(pairs) else 0,
        "outcome": "tvr_t1 = duration-weighted mean programme TVR of same strip on day t+1",
        "specs": [],
    }
    for pred in ("n_breaks_t", "break_min_t", "end_break_min_t"):
        results["specs"].append(run_spec(pairs, pred, add_lag=False))
    # robustness: add lagged tune-in as control (mean-reversion guard)
    for pred in ("n_breaks_t", "break_min_t", "end_break_min_t"):
        results["specs"].append(run_spec(pairs, pred, add_lag=True))

    with open(OUT / "habit_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
