"""Convexity of retention cost in break length: fit shed = a + f(length).

Reads breaks_measured.csv and instances.csv produced by prepare_data.py.
Fits several f() shapes with controls (first-break flag, position in
programme, program class, daypart, channel), runs selection-effect
diagnostics, a within-cell equal-total-minutes split estimator, and the
split-vs-consolidate decision delta with cluster-bootstrap uncertainty.

Outputs: fitted_shape.csv, results.json, fit_run printout.
No files outside analysis/convexity/ are written. data/ is untouched.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
RNG = np.random.default_rng(20260707)
B_BOOT = 800

BIN_EDGES = [0.0, 0.75, 1.25, 1.75, 2.25, 2.75, 3.5, 4.5, 6.0, 8.0, 13.0]
GRID = np.round(np.arange(0.5, 10.01, 0.25), 2)


# ---------- OLS with cluster-robust (CR1) covariance ----------

def ols_cluster(X: np.ndarray, y: np.ndarray, clusters: np.ndarray):
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = np.zeros((k, k))
    codes, _ = pd.factorize(clusters)
    for g in range(codes.max() + 1):
        idx = codes == g
        Xg = X[idx]
        ug = u[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    G = codes.max() + 1
    adj = (G / (G - 1)) * ((n - 1) / max(n - k, 1))
    cov = adj * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(cov))
    rss = float(u @ u)
    tss = float(((y - y.mean()) ** 2).sum())
    ll = -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    return {
        "beta": beta, "se": se, "cov": cov, "r2": 1 - rss / tss,
        "aic": 2 * (k + 1) - 2 * ll, "bic": np.log(n) * (k + 1) - 2 * ll,
        "n": n, "k": k, "G": G, "resid": u,
    }


def t_p(beta: float, se: float, dof: int) -> float:
    if se <= 0:
        return float("nan")
    return float(2 * stats.t.sf(abs(beta / se), dof))


# ---------- design matrices ----------

def natural_spline_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis (truncated power, linear beyond boundary)."""
    kK = knots[-1]
    kK1 = knots[-2]

    def d(j_knot: float) -> np.ndarray:
        return (np.clip(x - j_knot, 0, None) ** 3 - np.clip(x - kK, 0, None) ** 3) / (kK - j_knot)

    cols = [x]
    for kj in knots[:-2]:
        cols.append(d(kj) - d(kK1))
    return np.column_stack(cols)


def control_matrix(df: pd.DataFrame):
    parts = [df["first_break"].to_numpy(float).reshape(-1, 1)]
    names = ["first_break"]
    for col, ref in [("break_position", "middle"), ("program_type", "Other"),
                     ("daypart", "afternoon"), ("channel", None)]:
        vals = sorted(df[col].astype(str).unique())
        if ref is None:
            ref = df[col].astype(str).mode()[0]
        for v in vals:
            if v == ref:
                continue
            parts.append((df[col].astype(str) == v).to_numpy(float).reshape(-1, 1))
            names.append(f"{col}={v}")
    return np.hstack(parts), names


def length_features(name: str, x: np.ndarray, knots: np.ndarray, bin_edges):
    if name == "none":
        return np.empty((len(x), 0)), []
    if name == "linear":
        return x.reshape(-1, 1), ["len"]
    if name == "log":
        return np.log(x).reshape(-1, 1), ["log_len"]
    if name == "sqrt":
        return np.sqrt(x).reshape(-1, 1), ["sqrt_len"]
    if name == "quadratic":
        return np.column_stack([x, x ** 2]), ["len", "len_sq"]
    if name == "spline":
        Z = natural_spline_basis(x, knots)
        return Z, [f"ns{i}" for i in range(Z.shape[1])]
    if name == "bins":
        codes = np.digitize(x, bin_edges[1:-1])
        cols, names = [], []
        for b in range(1, len(bin_edges) - 1):  # bin 0 is reference
            cols.append((codes == b).astype(float).reshape(-1, 1))
            names.append(f"bin{b}")
        return np.hstack(cols), names
    raise ValueError(name)


def fit_model(df, name, knots, rep_controls, ctrl_mat, boot=False):
    x = df["len_min"].to_numpy(float)
    y = df["shed"].to_numpy(float)
    F, fnames = length_features(name, x, knots, BIN_EDGES)
    X = np.hstack([np.ones((len(df), 1)), F, ctrl_mat])
    res = ols_cluster(X, y, df["cluster"].to_numpy())
    nf = len(fnames)

    assert np.isfinite(res["beta"]).all(), f"non-finite beta in model {name}"

    def curve(lengths: np.ndarray) -> np.ndarray:
        base = float(res["beta"][0] + rep_controls @ res["beta"][1 + nf:])
        vals = np.full(len(lengths), base)
        if nf:
            Fg, _ = length_features(name, lengths, knots, BIN_EDGES)
            vals = vals + Fg @ res["beta"][1:1 + nf]
        return vals

    c2, c4 = curve(np.array([2.0]))[0], curve(np.array([4.0]))[0]
    out = {
        "model": name, "r2": round(res["r2"], 5), "aic": round(res["aic"], 1),
        "bic": round(res["bic"], 1), "n": res["n"], "clusters": res["G"],
        "intercept": round(float(res["beta"][0]), 5),
        "intercept_se": round(float(res["se"][0]), 5),
        "intercept_p": round(t_p(res["beta"][0], res["se"][0], res["G"] - 1), 5),
        "length_terms": {
            fn: {"beta": round(float(res["beta"][1 + i]), 6),
                 "se": round(float(res["se"][1 + i]), 6),
                 "p": round(t_p(res["beta"][1 + i], res["se"][1 + i], res["G"] - 1), 5)}
            for i, fn in enumerate(fnames)
        },
        "curve_at_2min": round(float(c2), 5),
        "curve_at_4min": round(float(c4), 5),
        "delta_split_4_to_2x2": round(float(2 * c2 - c4), 5),
    }
    return out, curve


def cluster_bootstrap(df, names, knots, rep_controls, b=B_BOOT):
    """Percentile CIs for delta and the curve, resampling prog_key clusters."""
    groups = {c: g.index.to_numpy() for c, g in df.groupby("cluster")}
    keys = np.array(list(groups.keys()), dtype=object)
    deltas = {m: [] for m in names}
    curves = {m: [] for m in names}
    for _ in range(b):
        pick = RNG.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        bs = df.loc[idx].reset_index(drop=True)
        # relabel clusters so repeated draws stay distinct
        reps = np.concatenate([[i] * len(groups[c]) for i, c in enumerate(pick)])
        bs = bs.copy()
        bs["cluster"] = reps
        cm, _ = control_matrix(bs)
        for m in names:
            try:
                out, curve = fit_model(bs, m, knots, rep_controls, cm)
            except np.linalg.LinAlgError:
                continue
            deltas[m].append(out["delta_split_4_to_2x2"])
            curves[m].append(curve(GRID))
    ci = {}
    for m in names:
        arr = np.array(deltas[m])
        cv = np.array(curves[m])
        ci[m] = {
            "delta_ci_lo": round(float(np.percentile(arr, 2.5)), 5),
            "delta_ci_hi": round(float(np.percentile(arr, 97.5)), 5),
            "delta_frac_positive": round(float((arr > 0).mean()), 4),
            "curve_lo": np.percentile(cv, 2.5, axis=0),
            "curve_hi": np.percentile(cv, 97.5, axis=0),
            "n_boot": int(len(arr)),
        }
    return ci


# ---------- selection diagnostics ----------

def cramers_v(tab: np.ndarray) -> float:
    chi2 = stats.chi2_contingency(tab)[0]
    n = tab.sum()
    r, c = tab.shape
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1))))


def selection_checks(df: pd.DataFrame, inst: pd.DataFrame) -> dict:
    out = {}
    for col in ["program_type", "daypart", "break_position", "channel", "first_break"]:
        tab = pd.crosstab(df["break_length"], df[col]).to_numpy()
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        out[f"len_bucket_vs_{col}"] = {
            "chi2": round(float(chi2), 1), "p": float(p), "dof": int(dof),
            "cramers_v": round(cramers_v(tab), 4),
        }
    # instance level: does split count track total minutes / daypart / type
    sub = inst.dropna(subset=["n_breaks", "total_seconds"])
    r = stats.spearmanr(sub["n_breaks"], sub["total_seconds"])
    out["inst_nbreaks_vs_totalminutes_spearman"] = {
        "rho": round(float(r.statistic), 4), "p": float(r.pvalue)}
    for col in ["daypart", "program_type", "channel"]:
        tab = pd.crosstab(sub["n_breaks"].clip(upper=4), sub[col]).to_numpy()
        chi2, p, dof, _ = stats.chi2_contingency(tab)
        out[f"inst_nbreaks_vs_{col}"] = {
            "chi2": round(float(chi2), 1), "p": float(p), "cramers_v": round(cramers_v(tab), 4)}
    return out


# ---------- within-cell equal-total-minutes split estimator ----------

def within_cell_split(inst: pd.DataFrame) -> dict:
    cells = []
    for key, g in inst.groupby(["channel", "program_type", "daypart", "total_minutes_rounded"]):
        if len(g) >= 2 and g["n_breaks"].nunique() >= 2:
            cells.append(g.assign(cell="|".join(str(k) for k in key)))
    if not cells:
        return {"feasible": False}
    d = pd.concat(cells, ignore_index=True)
    d["y_dm"] = d["total_shed"] - d.groupby("cell")["total_shed"].transform("mean")
    d["x_dm"] = d["n_breaks"] - d.groupby("cell")["n_breaks"].transform("mean")
    slope = float((d["x_dm"] * d["y_dm"]).sum() / (d["x_dm"] ** 2).sum())
    cell_keys = d["cell"].unique()
    boots = []
    for _ in range(B_BOOT):
        pick = RNG.choice(cell_keys, size=len(cell_keys), replace=True)
        bs = pd.concat([d[d["cell"] == c] for c in pick], ignore_index=True)
        sxx = (bs["x_dm"] ** 2).sum()
        if sxx > 0:
            boots.append(float((bs["x_dm"] * bs["y_dm"]).sum() / sxx))
    arr = np.array(boots)
    return {
        "feasible": True, "n_cells": int(len(cell_keys)), "n_instances": int(len(d)),
        "per_extra_break_shed": round(slope, 5),
        "ci_lo": round(float(np.percentile(arr, 2.5)), 5),
        "ci_hi": round(float(np.percentile(arr, 97.5)), 5),
        "frac_positive": round(float((arr > 0).mean()), 4),
        "n_boot": int(len(arr)),
    }


def main() -> None:
    df = pd.read_csv(OUT / "breaks_measured.csv")
    inst = pd.read_csv(OUT / "instances.csv")
    results: dict[str, object] = {"n_breaks": int(len(df)), "n_instances_full": int(len(inst))}

    x = df["len_min"].to_numpy(float)
    knots = np.quantile(x, [0.1, 0.35, 0.65, 0.9])
    results["spline_knots"] = [round(float(k), 3) for k in knots]
    codes = np.digitize(x, BIN_EDGES[1:-1])
    results["bin_counts"] = {
        f"({BIN_EDGES[b]},{BIN_EDGES[b + 1]}]": int((codes == b).sum())
        for b in range(len(BIN_EDGES) - 1)
    }

    ctrl_mat, ctrl_names = control_matrix(df)
    results["controls"] = ctrl_names
    # representative program: Other class, afternoon, middle position, not
    # first break, modal channel -> every dummy 0 by reference coding.
    rep_controls = np.zeros(len(ctrl_names))
    results["representative_controls"] = "all reference categories (Other, afternoon, middle, not-first, modal channel)"

    model_names = ["none", "linear", "log", "sqrt", "quadratic", "spline", "bins"]
    fits, curve_fns = {}, {}
    for m in model_names:
        out, curve = fit_model(df, m, knots, rep_controls, ctrl_mat)
        fits[m] = out
        curve_fns[m] = curve
    results["models"] = fits

    boot_models = ["linear", "quadratic", "spline", "bins"]
    ci = cluster_bootstrap(df, boot_models, knots, rep_controls)
    results["bootstrap"] = {
        m: {k: v for k, v in ci[m].items() if not k.startswith("curve")}
        for m in boot_models
    }

    # raw and adjusted bin means for the shape CSV
    df["_bin"] = codes
    raw_bins = df.groupby("_bin").agg(
        len_mid=("len_min", "mean"), shed_mean=("shed", "mean"),
        shed_sem=("shed", lambda s: s.std() / np.sqrt(len(s))), n=("shed", "size"),
    ).reset_index()

    rows = []
    for i, L in enumerate(GRID):
        row = {"len_min": float(L)}
        for m in boot_models:
            row[f"{m}_fit"] = round(float(curve_fns[m](np.array([L]))[0]), 5)
            row[f"{m}_lo"] = round(float(ci[m]["curve_lo"][i]), 5)
            row[f"{m}_hi"] = round(float(ci[m]["curve_hi"][i]), 5)
        rows.append(row)
    shape = pd.DataFrame(rows)
    shape.to_csv(OUT / "fitted_shape.csv", index=False)
    raw_bins.to_csv(OUT / "raw_bin_means.csv", index=False)

    results["selection"] = selection_checks(df, inst)
    results["within_cell_split"] = within_cell_split(inst)

    # convexity headline: 2*s(2) - s(4) per model (positive = splitting costs
    # more audience = consolidate; negative = splitting saves = superlinear)
    results["decision"] = {
        m: {"delta_split_4min_into_2x2min": fits[m]["delta_split_4_to_2x2"],
            **results["bootstrap"].get(m, {})}
        for m in boot_models
    }

    with open(OUT / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
