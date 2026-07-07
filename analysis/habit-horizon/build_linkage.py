"""Build the habit-horizon panel and measure its feasibility.

Question: does today's break load during a programme predict tomorrow's tune-in
for the SAME programme strip (same Title + Channel, next calendar day)?

Linkage is aggregate slot-level only. There is no viewer panel in the data, so
"tune-in" here means the aggregate audience (TVR) of the next day's airing of the
same strip, not any individual return. This script builds the panel and writes
the raw feasibility counts. Regression lives in fit_habit.py.

Outputs (analysis/habit-horizon/):
  panel.csv          one row per linked consecutive-day strip pair
  linkage_report.json  feasibility counts (every number reproducible here)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.loaders import load_programmes, load_spots, load_dayparts
from kairos.model.prepare import identify_breaks

OUT = Path(__file__).resolve().parent
# Programme rows that are ad/promo filler in the EPG, not real programmes.
_FILLER_TOKENS = ("פרומו", "פרסומות")
_END_WINDOW_MIN = 10.0  # "near the end" = breaks starting within last N minutes


def _minute_index(ts: pd.Series) -> pd.Series:
    """Map a datetime to its minute-of-day timeband label 'H:MM' as Dayparts uses."""
    return ts.dt.hour.astype(str) + ":" + ts.dt.minute.astype(str).str.zfill(2)


def build() -> None:
    prog = load_programmes()
    spots = load_spots()
    day = load_dayparts()

    prog = prog[prog["start_dt"].notna() & prog["end_dt"].notna()].copy()
    prog["title"] = prog["Title"].astype(str).str.strip()
    prog["channel"] = prog["Channel"].astype(str).str.strip()
    # Drop EPG ad/promo filler rows: they are not audience-bearing programmes.
    is_filler = prog["title"].apply(lambda t: any(tok in t for tok in _FILLER_TOKENS))
    prog = prog[~is_filler].copy()
    prog["date"] = prog["start_dt"].dt.normalize()
    prog["dur_min"] = (prog["end_dt"] - prog["start_dt"]).dt.total_seconds() / 60.0
    prog = prog[prog["dur_min"] > 0].copy()

    # Detect breaks from spots (engine definition: runs of >=2 spots).
    breaks = identify_breaks(spots)
    breaks = breaks[breaks["break_start"].notna()].copy()
    breaks["channel"] = breaks["channel"].astype(str).str.strip()
    breaks["break_min"] = breaks["break_seconds"] / 60.0

    # Minute-level TVR lookup for start-of-programme tune-in.
    day = day.copy()
    day["channel"] = day["channel"].astype(str).str.strip()
    day_idx = day.set_index(["channel", "date", "timeband"])["tvr"].sort_index()

    # ---- Per-programme-instance break load (day t features) ----
    rows = []
    br_by_ch = {c: g.sort_values("break_start") for c, g in breaks.groupby("channel")}
    for r in prog.itertuples(index=False):
        ch = r.channel
        s, e = r.start_dt, r.end_dt
        g = br_by_ch.get(ch)
        n_breaks = 0
        ad_min = 0.0
        n_end = 0
        ad_min_end = 0.0
        if g is not None and len(g):
            m = (g["break_start"] >= s) & (g["break_start"] < e)
            gin = g[m]
            n_breaks = int(len(gin))
            ad_min = float(gin["break_min"].sum())
            end_cut = e - pd.Timedelta(minutes=_END_WINDOW_MIN)
            gend = gin[gin["break_start"] >= end_cut]
            n_end = int(len(gend))
            ad_min_end = float(gend["break_min"].sum())
        rows.append({
            "channel": ch, "title": r.title, "date": r.date,
            "start_dt": s, "end_dt": e, "dur_min": r.dur_min,
            "prog_tvr": float(r.TVR) if pd.notna(r.TVR) else np.nan,
            "n_breaks": n_breaks, "ad_min": ad_min,
            "n_breaks_end": n_end, "ad_min_end": ad_min_end,
            "breaks_per_hr": n_breaks / (r.dur_min / 60.0),
        })
    inst = pd.DataFrame(rows)

    # start-of-programme TVR: mean minute TVR over first 3 minutes of the airing.
    def start_tvr(row) -> float:
        vals = []
        for k in range(3):
            tb = (row["start_dt"] + pd.Timedelta(minutes=k))
            key = (row["channel"], row["date"], f"{tb.hour}:{tb.minute:02d}")
            try:
                v = day_idx.loc[key]
                if isinstance(v, pd.Series):
                    v = v.iloc[0]
                if pd.notna(v):
                    vals.append(float(v))
            except KeyError:
                pass
        return float(np.mean(vals)) if vals else np.nan
    inst["start_tvr"] = inst.apply(start_tvr, axis=1)

    # If a strip airs twice in one day, keep the first airing to keep pairs clean.
    inst = inst.sort_values(["channel", "title", "date", "start_dt"])
    inst = inst.drop_duplicates(["channel", "title", "date"], keep="first")

    # ---- Link consecutive-day pairs of the same strip ----
    nxt = inst.copy()
    nxt["date"] = nxt["date"] - pd.Timedelta(days=1)  # shift so it joins to day t
    nxt = nxt.rename(columns={
        "prog_tvr": "prog_tvr_next", "start_tvr": "start_tvr_next",
        "n_breaks": "n_breaks_next", "start_dt": "start_dt_next",
        "dur_min": "dur_min_next",
    })[["channel", "title", "date", "prog_tvr_next", "start_tvr_next",
        "n_breaks_next", "start_dt_next", "dur_min_next"]]
    panel = inst.merge(nxt, on=["channel", "title", "date"], how="inner")
    panel["weekday_t1"] = (panel["start_dt_next"]).dt.day_name()
    panel["prog_id"] = panel["channel"] + " | " + panel["title"]

    panel.to_csv(OUT / "panel.csv", index=False)

    # ---- Feasibility counts ----
    strip_days = inst.groupby(["channel", "title"])["date"].nunique()
    recurring = int((strip_days >= 2).sum())
    report = {
        "programme_instances_real": int(len(inst)),
        "distinct_strips": int(inst.groupby(["channel", "title"]).ngroups),
        "strips_recurring_ge2_days": recurring,
        "consecutive_day_pairs": int(len(panel)),
        "pairs_with_prog_tvr_both": int(
            panel[["prog_tvr", "prog_tvr_next"]].notna().all(axis=1).sum()),
        "pairs_with_start_tvr_both": int(
            panel[["start_tvr", "start_tvr_next"]].notna().all(axis=1).sum()),
        "distinct_strips_in_panel": int(panel.groupby("prog_id").ngroups),
        "strips_with_ge3_pairs": int(
            (panel.groupby("prog_id").size() >= 3).sum()),
        "date_range": [str(inst["date"].min().date()), str(inst["date"].max().date())],
        "breaks_detected_total": int(len(breaks)),
        "instances_with_ge1_break": int((inst["n_breaks"] >= 1).sum()),
        "mean_n_breaks": float(inst["n_breaks"].mean()),
        "mean_ad_min": float(inst["ad_min"].mean()),
        "end_window_minutes": _END_WINDOW_MIN,
        "top_strips_by_pairs": panel.groupby("prog_id").size().sort_values(
            ascending=False).head(10).to_dict(),
    }
    (OUT / "linkage_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    build()
