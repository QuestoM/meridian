"""Q4: at what granularity does planned_tvr (the engine's revenue basis) vary?

READ-ONLY over data/. Writes only under analysis/quarter-hour/settlement/.

Two plan sources feed baseline_tvr:

  1. The daily input csv (data/daily_input/Wally_*.csv), column
     'planned_tvr' per spot row. kairos.data.transform.
     build_segments_from_daily_input averages it per programme.
  2. Programmes.xlsx TVR, one value per programme row, used by
     build_segments_from_programmes.

Tests on the daily input:
  A. Is planned_tvr constant within each round quarter hour of spot_time?
  B. Do consecutive-spot value changes bracket a quarter-hour boundary?
  C. Is it constant within a break (break_start group)? Within a programme?
  D. Do spots in the SAME quarter hour but DIFFERENT breaks share the value?

On Programmes.xlsx: rows vs distinct TVR per channel-day (per-programme
granularity check).

Output: planned_tvr_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from kairos.data.loaders import load_daily_input, load_programmes

OUT = Path(__file__).resolve().parent
DAILY = Path(__file__).resolve().parents[3] / "data" / "daily_input"
QH_SEC = 900


def time_to_seconds(text: object) -> float | None:
    parts = str(text).strip().split(":")
    if len(parts) < 2:
        return None
    try:
        h, m = int(parts[0]), int(parts[1])
        s = int(parts[2]) if len(parts) > 2 else 0
        return float(h * 3600 + m * 60 + s)
    except ValueError:
        return None


def main() -> None:
    results: dict[str, object] = {"daily_files": []}

    for path in sorted(DAILY.glob("*.csv")):
        daily = load_daily_input(path)
        frame = daily[daily["planned_tvr"].notna() & daily["spot_time"].notna()].copy()
        frame["sec"] = frame["spot_time"].map(time_to_seconds)
        frame = frame[frame["sec"].notna()].sort_values("sec").reset_index(drop=True)
        frame["qh"] = (frame["sec"] // QH_SEC).astype(int)

        per_qh = frame.groupby("qh")["planned_tvr"].nunique()
        per_break = frame.groupby("break_start")["planned_tvr"].nunique()
        per_prog = frame.groupby("program")["planned_tvr"].nunique()

        changes = []
        vals = frame["planned_tvr"].to_numpy()
        secs = frame["sec"].to_numpy()
        for i in range(1, len(frame)):
            if vals[i] != vals[i - 1]:
                brackets = int(secs[i] // QH_SEC) != int(secs[i - 1] // QH_SEC)
                changes.append({
                    "prev_time": str(frame["spot_time"].iloc[i - 1]),
                    "time": str(frame["spot_time"].iloc[i]),
                    "prev_tvr": float(vals[i - 1]), "tvr": float(vals[i]),
                    "brackets_qh_boundary": brackets,
                    "same_break": bool(
                        frame["break_start"].iloc[i] == frame["break_start"].iloc[i - 1]
                    ),
                })

        # D: same QH, different breaks -> same value?
        cross = []
        for qh, g in frame.groupby("qh"):
            if g["break_start"].nunique() >= 2:
                cross.append({
                    "qh_start": f"{qh * QH_SEC // 3600}:{(qh * QH_SEC % 3600) // 60:02d}",
                    "n_breaks": int(g["break_start"].nunique()),
                    "n_distinct_tvr": int(g["planned_tvr"].nunique()),
                    "values": sorted(set(float(v) for v in g["planned_tvr"])),
                })

        results["daily_files"].append({
            "file": path.name,
            "n_spot_rows": int(len(frame)),
            "n_qh_windows": int(len(per_qh)),
            "qh_windows_with_single_value": int((per_qh == 1).sum()),
            "n_breaks": int(len(per_break)),
            "breaks_with_single_value": int((per_break == 1).sum()),
            "n_programmes": int(len(per_prog)),
            "programmes_with_single_value": int((per_prog == 1).sum()),
            "n_value_changes": len(changes),
            "changes_bracketing_qh_boundary": int(
                sum(c["brackets_qh_boundary"] for c in changes)
            ),
            "changes_within_same_break": int(sum(c["same_break"] for c in changes)),
            "change_points": changes,
            "same_qh_cross_break": cross,
        })

    programmes = load_programmes()
    pr = programmes[programmes["start_dt"].notna()].copy()
    pr["day"] = pr["start_dt"].dt.strftime("%Y-%m-%d")
    g = pr.groupby(["Channel", "day"]).agg(rows=("TVR", "size"), distinct=("TVR", "nunique"))
    results["programmes_xlsx"] = {
        "note": "TVR is one value per programme ROW; distinct/rows < 1 only via repeated values",
        "channel_days": int(len(g)),
        "mean_rows_per_channel_day": round(float(g["rows"].mean()), 1),
        "mean_distinct_tvr_per_channel_day": round(float(g["distinct"].mean()), 1),
        "rows_total": int(g["rows"].sum()),
    }
    results["engine_code_facts"] = {
        "daily_path": "kairos/data/transform.py build_segments_from_daily_input: baseline_tvr = MEAN of planned_tvr over the programme's spot rows (per-programme scalar)",
        "programmes_path": "kairos/data/transform.py build_segments_from_programmes: baseline_tvr = the programme row's TVR (per-programme scalar)",
        "revenue": "kairos/optimize/objective.py break_revenue: cpp * rating_points * duration * premium with rating_points from that per-programme scalar; no window awareness",
    }

    with open(OUT / "planned_tvr_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
