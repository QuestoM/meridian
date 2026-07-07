"""Habit-horizon regression: does day-t break load predict day-(t+1) tune-in?

Panel = consecutive-day pairs of the same strip (Title+Channel), built by
build_linkage.py. Outcome is the next day's aggregate audience for the same
strip (whole-programme TVR, and start-of-programme TVR as a tune-in-arrival
proxy). This is an ASSOCIATION at the aggregate slot level, not causal and not
individual-viewer habit (no panel data exist). Program fixed effects + weekday
controls; cluster-robust standard errors by strip.

Two specifications per break-load metric:
  A  next tune-in ~ break_load_t + prog_FE + weekday_FE
  B  add today's tune-in (lagged outcome) to isolate break load beyond persistence
     (note: FE + lagged dependent carries Nickell bias at T~29; reported honestly)

Outputs: habit_results.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

np.seterr(all="ignore")  # large FE dummy matmul triggers spurious BLAS warnings
OUT = Path(__file__).resolve().parent


def ols_cluster(y: np.ndarray, X: np.ndarray, groups: np.ndarray):
    """OLS beta with cluster-robust (CR1) covariance clustered on ``groups``."""
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    n, k = X.shape
    uniq = np.unique(groups)
    g = len(uniq)
    meat = np.zeros((k, k))
    for gid in uniq:
        m = groups == gid
        Xg = X[m]
        ug = resid[m]
        s = Xg.T @ ug
        meat += np.outer(s, s)
    dof = (g / (g - 1.0)) * ((n - 1.0) / (n - k))
    cov = dof * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return beta, se, r2, g


def design(df: pd.DataFrame, predictors, controls):
    """Build [intercept, predictors, dummy FE controls] design matrix."""
    parts = [np.ones((len(df), 1))]
    names = ["const"]
    for p in predictors:
        parts.append(df[[p]].to_numpy(float))
        names.append(p)
    for c in controls:
        d = pd.get_dummies(df[c], prefix=c, drop_first=True).astype(float)
        parts.append(d.to_numpy())
        names.extend(list(d.columns))
    return np.hstack(parts), names


def run_spec(df, outcome, predictor, extra_controls):
    sub = df.dropna(subset=[outcome, predictor, "prog_id", "weekday_t1"] + extra_controls).copy()
    # Keep only strips with >=2 within-strip observations (FE needs variation).
    counts = sub.groupby("prog_id")[outcome].transform("count")
    sub = sub[counts >= 2].copy()
    if len(sub) < 30:
        return {"n": int(len(sub)), "status": "insufficient"}
    preds = [predictor] + extra_controls
    X, names = design(sub, preds, ["prog_id", "weekday_t1"])
    y = sub[outcome].to_numpy(float)
    groups = sub["prog_id"].to_numpy()
    beta, se, r2, g = ols_cluster(y, X, groups)
    idx = names.index(predictor)
    b = float(beta[idx]); s = float(se[idx])
    ci = [b - 1.96 * s, b + 1.96 * s]
    t = b / s if s > 0 else float("nan")
    # standardized effect: change in y-SD per 1-SD change in predictor
    std_beta = b * float(sub[predictor].std()) / float(sub[outcome].std())
    return {
        "n": int(len(sub)), "n_strips": int(g),
        "coef": round(b, 6), "se": round(s, 6),
        "ci95": [round(ci[0], 6), round(ci[1], 6)],
        "t": round(t, 3), "std_beta": round(std_beta, 4),
        "r2_full": round(r2, 4),
        "predictor_mean": round(float(sub[predictor].mean()), 4),
        "predictor_sd": round(float(sub[predictor].std()), 4),
        "outcome_mean": round(float(sub[outcome].mean()), 4),
        "outcome_sd": round(float(sub[outcome].std()), 4),
    }


def main():
    df = pd.read_csv(OUT / "panel.csv", parse_dates=["date", "start_dt", "end_dt"])
    metrics = ["ad_min_end", "n_breaks_end", "ad_min", "n_breaks", "breaks_per_hr"]
    results = {"panel_rows": int(len(df)), "specs": {}}
    for outcome, lag in [("prog_tvr_next", "prog_tvr"), ("start_tvr_next", "start_tvr")]:
        results["specs"][outcome] = {}
        for m in metrics:
            a = run_spec(df, outcome, m, [])                 # A: no persistence control
            b = run_spec(df, outcome, m, [lag])              # B: + lagged tune-in
            results["specs"][outcome][m] = {"A_fe_weekday": a, "B_plus_lag": b}
    (OUT / "habit_results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    # console summary of headline end-window metric
    for oc in ("prog_tvr_next", "start_tvr_next"):
        for m in ("ad_min_end", "n_breaks_end"):
            for spec in ("A_fe_weekday", "B_plus_lag"):
                r = results["specs"][oc][m][spec]
                print(oc, m, spec, "coef", r.get("coef"), "ci", r.get("ci95"),
                      "t", r.get("t"), "N", r.get("n"), "strips", r.get("n_strips"))
    print("written", OUT / "habit_results.json")


if __name__ == "__main__":
    main()
