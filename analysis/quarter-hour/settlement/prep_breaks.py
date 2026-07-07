"""Quarter-hour settlement mechanics in the real Nov-2024 schedule, part 1.

READ-ONLY over data/. Writes only under analysis/quarter-hour/settlement/.

Q1 (straddle prevalence): for every real detected break, does it cross a
round quarter-hour (:00/:15/:30/:45) or half-hour boundary? Observed fraction
vs the expected fraction under uniform-random placement with the SAME length
distribution (a break of length L seconds crosses a QH boundary with
probability min(L/900, 1) when its start is uniform on the quarter-hour
cycle). Per channel and per length bin, with a Poisson-binomial normal
z-score against the uniform null, plus the start-offset-mod-15min histogram.

Q2 (multi-break windows): number of breaks overlapping each occupied
quarter-hour and half-hour window, per channel and daypart, and the fraction
of breaks that share at least one settlement window with another break.

Outputs: breaks_qh.csv, straddle_results.json, windows_results.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_programmes, load_spots
from kairos.model.prepare import keyed_breaks

OUT = Path(__file__).resolve().parent

QH = 900.0
HH = 1800.0

LEN_BINS = [0, 60, 120, 180, 240, 360, 540, 10**9]
LEN_LABELS = ["<1m", "1-2m", "2-3m", "3-4m", "4-6m", "6-9m", "9m+"]

DAYPART_BINS = [-1, 5, 11, 16, 19, 23]
DAYPART_LABELS = ["overnight", "morning", "afternoon", "early_evening", "late_evening"]

EPOCH = pd.Timestamp("1970-01-01")


def daypart_of(hours: pd.Series) -> pd.Series:
    return pd.cut(hours, bins=DAYPART_BINS, labels=DAYPART_LABELS).astype(str)


def epoch_seconds(ts: pd.Series) -> pd.Series:
    return (pd.to_datetime(ts) - EPOCH).dt.total_seconds()


def straddle_table(frame: pd.DataFrame, group_col: str, period: float) -> list[dict]:
    """Observed vs uniform-null boundary-crossing fraction per group.

    The null keeps each break's real length and draws its start uniformly on
    the period cycle: p_i = min(L_i / period, 1). The z-score uses the
    Poisson-binomial normal approximation sum(p_i(1-p_i)).
    """
    col = "straddle_qh" if period == QH else "straddle_hh"
    rows = []
    for group, g in frame.groupby(group_col, sort=True):
        p = np.minimum(g["break_seconds"].to_numpy() / period, 1.0)
        exp_frac = float(p.mean())
        obs = int(g[col].sum())
        n = len(g)
        var = float((p * (1 - p)).sum())
        z = (obs - p.sum()) / math.sqrt(var) if var > 0 else float("nan")
        rows.append({
            "group": str(group), "n": n,
            "observed_frac": round(obs / n, 4),
            "expected_uniform_frac": round(exp_frac, 4),
            "obs_minus_exp": round(obs / n - exp_frac, 4),
            "z": round(float(z), 2),
        })
    return rows


def start_offset_hist(frame: pd.DataFrame) -> dict:
    """Minute-of-quarter histogram of break starts (0 = just after a boundary)."""
    off_min = ((frame["start_s"] % QH) // 60).astype(int)
    counts = off_min.value_counts().sort_index()
    total = int(counts.sum())
    return {
        "counts_by_minute_mod15": {int(k): int(v) for k, v in counts.items()},
        "frac_by_minute_mod15": {int(k): round(float(v) / total, 4) for k, v in counts.items()},
        "n": total,
        "uniform_frac": round(1 / 15, 4),
    }


def window_counts(frame: pd.DataFrame, period: float) -> pd.DataFrame:
    """One row per (channel, window index) with the number of overlapping breaks."""
    recs = []
    idx_lo = np.floor(frame["start_s"].to_numpy() / period).astype(np.int64)
    idx_hi = np.floor((frame["end_s"].to_numpy() - 1e-6) / period).astype(np.int64)
    for (ch, lo, hi) in zip(frame["channel"], idx_lo, idx_hi):
        for w in range(lo, hi + 1):
            recs.append((ch, w))
    win = pd.DataFrame(recs, columns=["channel", "widx"])
    counts = win.groupby(["channel", "widx"]).size().rename("n_breaks").reset_index()
    counts["window_start"] = EPOCH + pd.to_timedelta(counts["widx"] * period, unit="s")
    counts["hour"] = counts["window_start"].dt.hour
    counts["daypart"] = daypart_of(counts["hour"])
    return counts


def dist_summary(counts: pd.DataFrame, keys: list[str]) -> list[dict]:
    rows = []
    for group, g in counts.groupby(keys, sort=True):
        n = len(g)
        dist = g["n_breaks"].value_counts().sort_index()
        rows.append({
            "group": "|".join(str(x) for x in (group if isinstance(group, tuple) else (group,))),
            "occupied_windows": n,
            "mean_breaks_per_occupied_window": round(float(g["n_breaks"].mean()), 3),
            "frac_2plus": round(float((g["n_breaks"] >= 2).mean()), 4),
            "frac_3plus": round(float((g["n_breaks"] >= 3).mean()), 4),
            "dist": {int(k): int(v) for k, v in dist.items()},
        })
    return rows


def shared_window_fracs(frame: pd.DataFrame, counts: pd.DataFrame, period: float) -> dict:
    """Fraction of BREAKS whose settlement window(s) contain another break."""
    multi = set(map(tuple, counts.loc[counts["n_breaks"] >= 2, ["channel", "widx"]].to_numpy()))
    idx_lo = np.floor(frame["start_s"].to_numpy() / period).astype(np.int64)
    idx_hi = np.floor((frame["end_s"].to_numpy() - 1e-6) / period).astype(np.int64)
    shared = np.array([
        any((ch, w) in multi for w in range(lo, hi + 1))
        for ch, lo, hi in zip(frame["channel"], idx_lo, idx_hi)
    ])
    out = {"all": round(float(shared.mean()), 4), "n": int(len(shared))}
    for ch in sorted(frame["channel"].unique()):
        m = (frame["channel"] == ch).to_numpy()
        out[str(ch)] = round(float(shared[m].mean()), 4)
    return out


def main() -> None:
    spots = load_spots()
    programmes = load_programmes()
    classifier = ProgramClassifier.from_yaml()
    kb = keyed_breaks(spots, programmes, classifier)

    kb = kb[kb["break_start"].notna() & kb["break_end"].notna()].copy()
    kb["start_s"] = epoch_seconds(kb["break_start"])
    kb["end_s"] = epoch_seconds(kb["break_end"])
    kb["len_min"] = kb["break_seconds"] / 60.0
    kb["len_bin"] = pd.cut(kb["break_seconds"], bins=LEN_BINS, labels=LEN_LABELS).astype(str)
    kb["hour"] = pd.to_datetime(kb["break_start"]).dt.hour
    kb["daypart"] = daypart_of(kb["hour"])

    qh_lo = np.floor(kb["start_s"] / QH).astype(np.int64)
    qh_hi = np.floor((kb["end_s"] - 1e-6) / QH).astype(np.int64)
    kb["n_qh_windows"] = (qh_hi - qh_lo + 1).astype(int)
    kb["straddle_qh"] = (kb["n_qh_windows"] > 1).astype(int)
    hh_lo = np.floor(kb["start_s"] / HH).astype(np.int64)
    hh_hi = np.floor((kb["end_s"] - 1e-6) / HH).astype(np.int64)
    kb["straddle_hh"] = (hh_hi > hh_lo).astype(int)

    keep = [
        "channel", "break_start", "break_end", "break_seconds", "num_spots",
        "program_type", "break_position", "break_length", "channel_name", "day",
        "ordinal", "prog_key", "start_s", "end_s", "len_min", "len_bin",
        "hour", "daypart", "n_qh_windows", "straddle_qh", "straddle_hh",
    ]
    kb[keep].to_csv(OUT / "breaks_qh.csv", index=False)

    p_all = np.minimum(kb["break_seconds"].to_numpy() / QH, 1.0)
    var_all = float((p_all * (1 - p_all)).sum())
    z_all = float((kb["straddle_qh"].sum() - p_all.sum()) / math.sqrt(var_all))
    straddle = {
        "n_breaks": int(len(kb)),
        "overall": {
            "observed_frac_qh": round(float(kb["straddle_qh"].mean()), 4),
            "expected_uniform_frac_qh": round(float(p_all.mean()), 4),
            "z_qh": round(z_all, 2),
            "observed_frac_hh": round(float(kb["straddle_hh"].mean()), 4),
            "expected_uniform_frac_hh": round(float(np.minimum(kb["break_seconds"] / HH, 1.0).mean()), 4),
        },
        "by_channel_qh": straddle_table(kb, "channel", QH),
        "by_len_bin_qh": straddle_table(kb, "len_bin", QH),
        "by_channel_hh": straddle_table(kb, "channel", HH),
        "by_len_bin_hh": straddle_table(kb, "len_bin", HH),
        "by_channel_len_qh": [
            {"channel": str(ch), "rows": straddle_table(g, "len_bin", QH)}
            for ch, g in kb.groupby("channel", sort=True)
        ],
        "start_offset_hist_all": start_offset_hist(kb),
        "start_offset_hist_by_channel": {
            str(ch): start_offset_hist(g) for ch, g in kb.groupby("channel", sort=True)
        },
    }
    with open(OUT / "straddle_results.json", "w", encoding="utf-8") as f:
        json.dump(straddle, f, indent=2, ensure_ascii=False)

    qh_counts = window_counts(kb, QH)
    hh_counts = window_counts(kb, HH)
    windows = {
        "note": "windows counted only where >= 1 break overlaps; a straddling break counts in each window it touches",
        "qh": {
            "by_channel": dist_summary(qh_counts, ["channel"]),
            "by_daypart": dist_summary(qh_counts, ["daypart"]),
            "by_channel_daypart": dist_summary(qh_counts, ["channel", "daypart"]),
            "break_shares_window_frac": shared_window_fracs(kb, qh_counts, QH),
        },
        "hh": {
            "by_channel": dist_summary(hh_counts, ["channel"]),
            "by_daypart": dist_summary(hh_counts, ["daypart"]),
            "by_channel_daypart": dist_summary(hh_counts, ["channel", "daypart"]),
            "break_shares_window_frac": shared_window_fracs(kb, hh_counts, HH),
        },
    }
    with open(OUT / "windows_results.json", "w", encoding="utf-8") as f:
        json.dump(windows, f, indent=2, ensure_ascii=False)

    print(json.dumps({"n_breaks": len(kb), "straddle_overall": straddle["overall"],
                      "qh_shared": windows["qh"]["break_shares_window_frac"],
                      "hh_shared": windows["hh"]["break_shares_window_frac"]},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
