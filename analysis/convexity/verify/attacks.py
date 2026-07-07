"""Adversarial attacks on the convexity verdict. Writes attack_results.json.

Reads breaks_measured.csv / instances.csv (regenerated in this dir, byte
identical to the originals) and keyed_breaks_augmented.csv (full keyed-break
population with gaps, survival flag, titles).

A1 channel confounding: refit within each channel separately.
A2 in-support delta: nonparametric local means at 2min and 4min, no shape.
A3 spacing: shed vs gap to previous break (split-independence assumption).
A4 pipeline: clip survival vs length; refit on uncontaminated full windows;
   duplicate-minute join risk.
A5 selection: title-level variance, within-title split estimator, required
   selection correlation to erase the delta.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fitmod", OUT / "fit_convexity.py")
fitmod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fitmod)

RNG = np.random.default_rng(20260708)
B_LOCAL = 2000
B_FIT = 400


def cluster_boot_stat(df: pd.DataFrame, stat_fn, b: int = B_LOCAL) -> dict:
    groups = {c: g.index.to_numpy() for c, g in df.groupby("cluster")}
    keys = np.array(list(groups.keys()), dtype=object)
    vals = []
    for _ in range(b):
        pick = RNG.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        v = stat_fn(df.loc[idx])
        if v is not None and np.isfinite(v):
            vals.append(float(v))
    arr = np.array(vals)
    return {
        "point": round(float(stat_fn(df)), 5),
        "ci_lo": round(float(np.percentile(arr, 2.5)), 5),
        "ci_hi": round(float(np.percentile(arr, 97.5)), 5),
        "frac_positive": round(float((arr > 0).mean()), 4),
        "n_boot": int(len(arr)),
    }


def fit_with_boot(df: pd.DataFrame, models: list[str], b: int = B_FIT) -> dict:
    x = df["len_min"].to_numpy(float)
    knots = np.quantile(x, [0.1, 0.35, 0.65, 0.9])
    cm, cn = fitmod.control_matrix(df)
    rep = np.zeros(len(cn))
    out = {}
    for m in models:
        fit, _ = fitmod.fit_model(df, m, knots, rep, cm)
        out[m] = {
            "n": fit["n"], "clusters": fit["clusters"],
            "intercept": fit["intercept"], "intercept_se": fit["intercept_se"],
            "intercept_p": fit["intercept_p"],
            "length_terms": fit["length_terms"],
            "curve_at_2min": fit["curve_at_2min"],
            "curve_at_4min": fit["curve_at_4min"],
            "delta": fit["delta_split_4_to_2x2"],
        }
    groups = {c: g.index.to_numpy() for c, g in df.groupby("cluster")}
    keys = np.array(list(groups.keys()), dtype=object)
    deltas = {m: [] for m in models}
    for _ in range(b):
        pick = RNG.choice(keys, size=len(keys), replace=True)
        idx = np.concatenate([groups[c] for c in pick])
        bs = df.loc[idx].reset_index(drop=True)
        reps = np.concatenate([[i] * len(groups[c]) for i, c in enumerate(pick)])
        bs = bs.copy()
        bs["cluster"] = reps
        cmb, _ = fitmod.control_matrix(bs)
        for m in models:
            try:
                fit, _ = fitmod.fit_model(bs, m, knots, rep, cmb)
            except (np.linalg.LinAlgError, AssertionError):
                continue
            deltas[m].append(fit["delta_split_4_to_2x2"])
    for m in models:
        arr = np.array(deltas[m])
        out[m]["delta_ci"] = [round(float(np.percentile(arr, q)), 5) for q in (2.5, 97.5)]
        out[m]["delta_frac_positive"] = round(float((arr > 0).mean()), 4)
        out[m]["n_boot"] = int(len(arr))
    return out


def main() -> None:
    df = pd.read_csv(OUT / "breaks_measured.csv")
    inst = pd.read_csv(OUT / "instances.csv")
    aug = pd.read_csv(OUT / "keyed_breaks_augmented.csv")
    res: dict[str, object] = {}

    # ---------- A1: within-channel fits ----------
    a1 = {}
    for ch, g in df.groupby("channel"):
        g = g.reset_index(drop=True)
        if len(g) < 150:
            a1[ch] = {"n": int(len(g)), "skipped": "too small"}
            continue
        a1[ch] = fit_with_boot(g, ["linear", "quadratic", "bins"])
    res["A1_within_channel"] = a1

    # ---------- A2: nonparametric in-support delta ----------
    w2 = (df["len_min"] >= 1.75) & (df["len_min"] <= 2.25)
    w4 = (df["len_min"] >= 3.5) & (df["len_min"] <= 4.5)
    res["A2_support_n"] = {"n_2min": int(w2.sum()), "n_4min": int(w4.sum())}

    def raw_delta(d):
        m2 = d.loc[(d["len_min"] >= 1.75) & (d["len_min"] <= 2.25), "shed"]
        m4 = d.loc[(d["len_min"] >= 3.5) & (d["len_min"] <= 4.5), "shed"]
        if len(m2) < 5 or len(m4) < 5:
            return None
        return 2 * m2.mean() - m4.mean()

    res["A2_raw_local_delta"] = cluster_boot_stat(df, raw_delta)

    # covariate-adjusted local means: residualize shed on controls only
    cm, cn = fitmod.control_matrix(df)
    X = np.hstack([np.ones((len(df), 1)), cm])
    beta, *_ = np.linalg.lstsq(X, df["shed"].to_numpy(float), rcond=None)
    df["_adj"] = df["shed"].to_numpy(float) - cm @ beta[1:]

    def adj_delta(d):
        m2 = d.loc[(d["len_min"] >= 1.75) & (d["len_min"] <= 2.25), "_adj"]
        m4 = d.loc[(d["len_min"] >= 3.5) & (d["len_min"] <= 4.5), "_adj"]
        if len(m2) < 5 or len(m4) < 5:
            return None
        return 2 * m2.mean() - m4.mean()

    res["A2_adjusted_local_delta"] = cluster_boot_stat(df, adj_delta)

    # per-channel raw local deltas
    a2ch = {}
    for ch, g in df.groupby("channel"):
        g = g.reset_index(drop=True)
        d = raw_delta(g)
        n2 = int(((g["len_min"] >= 1.75) & (g["len_min"] <= 2.25)).sum())
        n4 = int(((g["len_min"] >= 3.5) & (g["len_min"] <= 4.5)).sum())
        if d is None:
            a2ch[ch] = {"n_2min": n2, "n_4min": n4, "skipped": "thin support"}
        else:
            a2ch[ch] = {"n_2min": n2, "n_4min": n4,
                        **cluster_boot_stat(g, raw_delta, b=1000)}
    res["A2_per_channel_raw_delta"] = a2ch

    # kernel version: gaussian kernel means at 2 and 4, bandwidth 0.5 min
    def kern_delta(d, bw=0.5):
        x = d["len_min"].to_numpy(float)
        y = d["shed"].to_numpy(float)
        out = []
        for c in (2.0, 4.0):
            w = np.exp(-0.5 * ((x - c) / bw) ** 2)
            if w.sum() < 5:
                return None
            out.append(float((w * y).sum() / w.sum()))
        return 2 * out[0] - out[1]

    res["A2_kernel_delta_bw0p5"] = cluster_boot_stat(df, kern_delta)

    # ---------- A3: spacing / independence ----------
    m = aug[aug["measured"] == 1].copy()
    m["cluster"] = m["prog_key"].fillna("cd:" + m["channel"].astype(str))
    m = m[m["shed"].notna()].reset_index(drop=True)
    bins = [0, 5, 15, 30, 60, np.inf]
    labels = ["(0,5]", "(5,15]", "(15,30]", "(30,60]", ">60"]
    m["gapprev_bin"] = pd.cut(m["gap_prev_min"], bins=bins, labels=labels)
    tab = m.groupby("gapprev_bin", observed=True)["shed"].agg(["mean", "sem", "size"])
    res["A3_shed_by_gap_prev"] = {
        str(k): {"mean": round(v["mean"], 5), "sem": round(v["sem"], 5),
                 "n": int(v["size"])} for k, v in tab.iterrows()}
    sub = m[m["gap_prev_min"].notna()].copy()
    r = stats.spearmanr(sub["gap_prev_min"], sub["shed"])
    res["A3_spearman_gapprev_shed"] = {"rho": round(float(r.statistic), 4),
                                       "p": float(r.pvalue), "n": int(len(sub))}
    # same-programme close spacing: later break within 15 min of previous
    sp = m[(m["ordinal"].notna()) & (m["ordinal"] > 1) & m["gap_prev_min"].notna()].copy()
    close = sp[sp["gap_prev_min"] <= 15]["shed"]
    far = sp[sp["gap_prev_min"] > 15]["shed"]
    t = stats.ttest_ind(close, far, equal_var=False)
    res["A3_later_breaks_close_vs_far"] = {
        "mean_close_le15": round(float(close.mean()), 5), "n_close": int(len(close)),
        "mean_far_gt15": round(float(far.mean()), 5), "n_far": int(len(far)),
        "welch_t": round(float(t.statistic), 3), "p": round(float(t.pvalue), 4)}

    # ---------- A4: pipeline / clip selection ----------
    aug["len_bin"] = pd.cut(aug["len_min"], bins=fitmod.BIN_EDGES)
    surv = aug.groupby("len_bin", observed=True)["measured"].agg(["mean", "size"])
    res["A4_survival_by_length"] = {
        str(k): {"survival": round(v["mean"], 4), "n_keyed": int(v["size"])}
        for k, v in surv.iterrows()}
    r = stats.spearmanr(aug["len_min"], aug["measured"])
    res["A4_spearman_len_vs_measured"] = {"rho": round(float(r.statistic), 4),
                                          "p": float(r.pvalue)}
    r2 = stats.spearmanr(aug["len_min"], aug["gap_next_min"], nan_policy="omit")
    res["A4_spearman_len_vs_gapnext"] = {"rho": round(float(r2.statistic), 4),
                                         "p": float(r2.pvalue)}

    # refit on uncontaminated breaks only (full 3-min windows on both sides)
    aug["clean"] = (aug["gap_prev_min"].fillna(np.inf) > 3) & (
        aug["gap_next_min"].fillna(np.inf) > 3)
    key = aug[["channel", "start_min", "clean"]].copy()
    dfm = df.merge(key, on=["channel", "start_min"], how="left")
    clean = dfm[dfm["clean"].fillna(False)].reset_index(drop=True)
    res["A4_clean_subset_n"] = int(len(clean))
    res["A4_clean_subset_fits"] = fit_with_boot(clean, ["linear", "bins"])

    # duplicate-minute join risk: measured breaks whose (channel,start_min)
    # had multiple keyed breaks (prepare kept the first arbitrarily)
    dup_keys = aug[aug.duplicated(subset=["channel", "start_min"], keep=False)]
    dk = set(zip(dup_keys["channel"], dup_keys["start_min"]))
    hits = sum((c, s) in dk for c, s in zip(df["channel"], df["start_min"]))
    res["A4_measured_rows_with_dup_minute_join"] = int(hits)

    # shed sign spot-check from raw ratios
    chk = df.assign(recalc=-(np.log(df["observed_ratio"]) - np.log(df["expected_ratio"])))
    res["A4_shed_sign_max_abs_err"] = float(np.max(np.abs(chk["recalc"] - chk["shed"])))

    # ---------- A5: title-level selection ----------
    tm = m[m["title"].fillna("") != ""].copy()
    tstats = tm.groupby("title")["shed"].agg(["mean", "size"])
    big = tstats[tstats["size"] >= 5]
    res["A5_title_effect_sd_ge5"] = {
        "n_titles": int(len(big)), "sd_of_title_means": round(float(big["mean"].std()), 5),
        "overall_break_shed_sd": round(float(m["shed"].std()), 5)}

    # within-title split estimator: same title + same rounded total minutes,
    # different n_breaks (instances mapped to modal measured title)
    pk_title = tm.groupby("prog_key")["title"].agg(
        lambda s: s.mode().iloc[0] if len(s.mode()) else "")
    inst2 = inst.merge(pk_title.rename("title"), on="prog_key", how="left")
    inst2 = inst2[inst2["title"].fillna("") != ""].copy()
    cells = []
    for k, g in inst2.groupby(["title", "total_minutes_rounded"]):
        if len(g) >= 2 and g["n_breaks"].nunique() >= 2:
            cells.append(g.assign(cell=str(k)))
    if cells:
        d = pd.concat(cells, ignore_index=True)
        d["y_dm"] = d["total_shed"] - d.groupby("cell")["total_shed"].transform("mean")
        d["x_dm"] = d["n_breaks"] - d.groupby("cell")["n_breaks"].transform("mean")
        slope = float((d["x_dm"] * d["y_dm"]).sum() / (d["x_dm"] ** 2).sum())
        ck = d["cell"].unique()
        boots = []
        for _ in range(B_LOCAL):
            pick = RNG.choice(ck, size=len(ck), replace=True)
            bs = pd.concat([d[d["cell"] == c] for c in pick], ignore_index=True)
            sxx = (bs["x_dm"] ** 2).sum()
            if sxx > 0:
                boots.append(float((bs["x_dm"] * bs["y_dm"]).sum() / sxx))
        arr = np.array(boots)
        res["A5_within_title_split"] = {
            "n_cells": int(len(ck)), "n_instances": int(len(d)),
            "per_extra_break_shed": round(slope, 5),
            "ci_lo": round(float(np.percentile(arr, 2.5)), 5),
            "ci_hi": round(float(np.percentile(arr, 97.5)), 5),
            "frac_positive": round(float((arr > 0).mean()), 4)}
    else:
        res["A5_within_title_split"] = {"feasible": False}

    # required selection to erase the observational delta: with title effects u,
    # bias(delta) ~= E[u | split] - E[u | consolidated]. To erase +0.03..0.05
    # the split shows must be MORE fragile by that much per break. Compare to
    # the title-mean sd.
    sd_t = float(big["mean"].std())
    res["A5_required_selection"] = {
        "delta_to_erase": [0.03, 0.05],
        "title_mean_sd": round(sd_t, 5),
        "required_gap_in_title_sd_units": [round(0.03 / sd_t, 3), round(0.05 / sd_t, 3)]}

    with open(OUT / "attack_results.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str, ensure_ascii=False)
    print(json.dumps(res, indent=2, default=str, ensure_ascii=False))


if __name__ == "__main__":
    main()
