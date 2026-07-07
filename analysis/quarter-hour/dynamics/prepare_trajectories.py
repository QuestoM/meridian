"""Build per-break minute-level detrended trajectories from the Nov-2024 TVR.

READ-ONLY over data/. Writes only under analysis/quarter-hour/dynamics/.

For every real keyed break, this extracts the minute-by-minute audience path:
  - pre-window minutes (-6..-1 relative to the break-start minute), used both
    to normalize each break to its own pre-break level and to check for an
    anticipatory dip;
  - leave minutes (offset 0 = the floor minute of break start, through the
    floor minute of break end);
  - return minutes (offset +1 = the first full content minute after the break
    end minute, through +10), each included only when it lies strictly before
    the next ad-air run on that channel (per-minute contamination clipping,
    same boundary rule as kairos.model.measure).

Detrending follows kairos/model/measure.py: each observed minute TVR is
divided by the channel's typical audience at that broadcast minute. FOUR
variants are carried side by side because the baseline choice was measured to
matter exactly at in-break clock minutes:
  rel_s: SMOOTHED global baseline (circular 15-minute rolling median of the
         typical curve), the primary. It keeps the time-of-day trend but rides
         over the sharp local artifacts both raw baselines carry at habitual
         break minutes (the global curve encodes the break dips themselves;
         the content-only curve at those minutes comes only from atypical,
         lower-audience days: measured bg/bc reaches 1.069 at leave offset 1).
  rel_g: the shipped global baseline (_baseline_levels), sensitivity.
  rel_c: content-only baseline (_content_only_baseline_levels), sensitivity.
  rel_raw: no detrend at all, the raw anchor (drift over a +-10 minute local
         window is small; any detrend-induced shape must stay close to this).
Each break's trajectory is then divided by its own pre-window mean (minutes
-3..-1, at least 2 clean minutes required), so 1.0 = this break's own
pre-break detrended level.

Outputs:
  trajectories.csv   one row per (break, minute offset) with rel_c and rel_g
  breaks_meta.csv    one row per included break (length, bin, cluster, ...)
  prep_summary.json  sample sizes, exclusion ledger, partial-minute fractions
  qh_sharing.json    how often real quarter-hour windows hold 2+ breaks
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.measure import (
    _baseline_levels,
    _broadcast_minute,
    _content_only_baseline_levels,
    _dayparts_frame,
)
from kairos.model.prepare import identify_breaks, keyed_breaks

OUT = Path(__file__).resolve().parent

PRE_NORM_OFFSETS = [-3, -2, -1]        # normalization window
PRE_CHECK_OFFSETS = [-6, -5, -4]       # anticipation check window
RETURN_MAX = 10                        # minutes after content resume
MIN_NORM_MINUTES = 2                   # clean pre minutes required to keep a break

LEN_BIN_EDGES = [0.0, 1.5, 2.5, 3.5, 5.0, 7.0, 13.0]
LEN_BIN_LABELS = ["lt1.5", "1.5-2.5", "2.5-3.5", "3.5-5", "5-7", "7-13"]

DAYPART_BINS = [-1, 5, 11, 16, 19, 23]
DAYPART_LABELS = ["overnight", "morning", "afternoon", "early_evening", "late_evening"]


def smoothed_baseline(base_g: dict[tuple[str, int], float]) -> dict[tuple[str, int], float]:
    """Circular 15-minute rolling MEDIAN of the global typical curve, per
    channel over the 1440 broadcast minutes. Keeps the time-of-day trend but
    removes the localized dips the typical curve inherits from the recurring
    break schedule (and any equally localized artifacts)."""
    out: dict[tuple[str, int], float] = {}
    channels = sorted({ch for ch, _ in base_g})
    for ch in channels:
        series = pd.Series([base_g.get((ch, m), np.nan) for m in range(1440)])
        series = series.interpolate(limit_direction="both")
        pad = 7
        wrapped = pd.concat([series.iloc[-pad:], series, series.iloc[:pad]], ignore_index=True)
        smooth = wrapped.rolling(15, center=True, min_periods=8).median().iloc[pad:pad + 1440]
        for m, v in enumerate(smooth.to_numpy()):
            if np.isfinite(v):
                out[(ch, m)] = float(v)
    return out


def build_run_boundaries(spots: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per channel: sorted arrays of (floor-minute start, floor-minute end) of
    EVERY ad-air run (min_spots=1), the strictest contamination boundary."""
    runs = identify_breaks(spots, min_spots=1)
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel, group in runs.groupby("channel", sort=False):
        starts = pd.to_datetime(group["break_start"]).dt.floor("min").astype("int64").to_numpy() // 60_000_000_000
        ends = pd.to_datetime(group["break_end"]).dt.floor("min").astype("int64").to_numpy() // 60_000_000_000
        order = np.argsort(starts)
        out[str(channel)] = (starts[order], ends[order])
    return out


def neighbour_bounds(bounds: tuple[np.ndarray, np.ndarray], start_min: int, end_min: int) -> tuple[int, int]:
    """(previous run end, next run start) in epoch minutes around a break whose
    own run spans [start_min, end_min]. Runs at the same floored start minute as
    the break itself are treated as the break's own run."""
    starts, ends = bounds
    idx = int(np.searchsorted(starts, start_min, side="left"))
    prev_end = int(ends[idx - 1]) if idx > 0 else -10**12
    # step past any runs that begin inside this break's own span (same run or
    # sub-runs floored into it)
    j = idx
    while j < len(starts) and starts[j] <= end_min:
        j += 1
    next_start = int(starts[j]) if j < len(starts) else 10**12
    return prev_end, next_start


def main() -> None:
    summary: dict[str, object] = {}
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()
    classifier = ProgramClassifier.from_yaml()

    frame = _dayparts_frame(dayparts)
    frame["epoch_min"] = frame["ts"].astype("int64") // 60_000_000_000
    obs = {(str(c), int(m)): float(v) for c, m, v in zip(frame["channel"], frame["epoch_min"], frame["tvr"])}
    base_g = _baseline_levels(frame)
    base_c = _content_only_baseline_levels(frame, spots)
    base_s = smoothed_baseline(base_g)
    summary["baseline_cells_global"] = len(base_g)
    summary["baseline_cells_content_only"] = len(base_c)
    summary["baseline_cells_smoothed"] = len(base_s)

    kb = keyed_breaks(spots, programmes, classifier)
    summary["keyed_breaks"] = int(len(kb))
    bounds = build_run_boundaries(spots)

    ledger = {"no_bounds_channel": 0, "thin_norm_window": 0, "bad_norm_value": 0, "kept": 0}
    rows: list[dict[str, object]] = []
    meta: list[dict[str, object]] = []
    frac0_list: list[float] = []
    frac_last_list: list[float] = []

    for i, row in enumerate(kb.itertuples(index=False)):
        channel = str(getattr(row, "channel"))
        start = pd.Timestamp(getattr(row, "break_start"))
        end = pd.Timestamp(getattr(row, "break_end"))
        s_min = int(start.floor("min").value // 60_000_000_000)
        e_min = int(end.floor("min").value // 60_000_000_000)
        if channel not in bounds:
            ledger["no_bounds_channel"] += 1
            continue
        prev_end, next_start = neighbour_bounds(bounds[channel], s_min, e_min)

        def rel_at(epoch_min: int) -> tuple[float, float, float, float]:
            """(rel_s, rel_c, rel_g, rel_raw) at one minute, un-normalized."""
            ts = pd.Timestamp(epoch_min * 60_000_000_000)
            mod = _broadcast_minute(ts)
            o = obs.get((channel, epoch_min))
            if o is None or o <= 0:
                return np.nan, np.nan, np.nan, np.nan
            vals = []
            for base in (base_s, base_c, base_g):
                b = base.get((channel, mod))
                vals.append((o / b) if (b and b > 0) else np.nan)
            return vals[0], vals[1], vals[2], float(o)

        # normalization window: minutes s_min-3..s_min-1, strictly after prev run end
        norm_vals: list[list[float]] = [[], [], [], []]
        for off in PRE_NORM_OFFSETS:
            m = s_min + off
            if m <= prev_end:
                continue
            for k, v in enumerate(rel_at(m)):
                if np.isfinite(v):
                    norm_vals[k].append(v)
        if min(len(v) for v in norm_vals) < MIN_NORM_MINUTES:
            ledger["thin_norm_window"] += 1
            continue
        norms = [float(np.mean(v)) for v in norm_vals]
        if min(norms) <= 0:
            ledger["bad_norm_value"] += 1
            continue
        norm_s, norm_c, norm_g, norm_raw = norms

        break_id = i
        dur_min = e_min - s_min + 1
        len_min = float(getattr(row, "break_seconds")) / 60.0
        # partial-minute fractions at the two boundary minutes
        frac0 = float(min(end.value, (s_min + 1) * 60_000_000_000) - start.value) / 60_000_000_000
        frac_last = float(end.value - max(start.value, e_min * 60_000_000_000)) / 60_000_000_000
        frac0_list.append(min(1.0, frac0))
        frac_last_list.append(min(1.0, frac_last))

        def emit(phase: str, offset: int, m: int) -> None:
            rs, rc, rg, rr = rel_at(m)
            rows.append({
                "break_id": break_id, "phase": phase, "offset": offset,
                "rel_s": (rs / norm_s) if np.isfinite(rs) else np.nan,
                "rel_c": (rc / norm_c) if np.isfinite(rc) else np.nan,
                "rel_g": (rg / norm_g) if np.isfinite(rg) else np.nan,
                "rel_raw": (rr / norm_raw) if np.isfinite(rr) else np.nan,
            })

        for off in PRE_CHECK_OFFSETS + PRE_NORM_OFFSETS:
            m = s_min + off
            if m > prev_end:
                emit("pre", off, m)
        for off in range(dur_min):
            emit("leave", off, s_min + off)
        for off in range(1, RETURN_MAX + 1):
            m = e_min + off
            if m >= next_start:
                break
            emit("return", off, m)

        hour = int(start.hour)
        prog_key = getattr(row, "prog_key")
        cluster = str(prog_key) if pd.notna(prog_key) else f"cd:{channel}|{start.strftime('%Y-%m-%d')}"
        meta.append({
            "break_id": break_id, "channel": channel,
            "channel_name": getattr(row, "channel_name"),
            "len_min": len_min, "dur_touched_min": dur_min,
            "len_bin": str(pd.cut([len_min], bins=LEN_BIN_EDGES, labels=LEN_BIN_LABELS)[0]),
            "ordinal": getattr(row, "ordinal"), "prog_key": prog_key, "cluster": cluster,
            "hour": hour,
            "daypart": str(pd.cut([hour], bins=DAYPART_BINS, labels=DAYPART_LABELS)[0]),
            "gap_prev_min": s_min - prev_end, "gap_next_min": next_start - e_min,
            "norm_s": norm_s, "norm_c": norm_c, "norm_g": norm_g, "norm_raw": norm_raw,
            "start_minute_in_qh": int(pd.Timestamp(start).floor("min").minute % 15),
        })
        ledger["kept"] += 1

    traj = pd.DataFrame(rows)
    meta_df = pd.DataFrame(meta)
    traj.to_csv(OUT / "trajectories.csv", index=False)
    meta_df.to_csv(OUT / "breaks_meta.csv", index=False)

    summary["exclusion_ledger"] = ledger
    summary["trajectory_rows"] = int(len(traj))
    summary["breaks_kept"] = int(len(meta_df))
    summary["clusters"] = int(meta_df["cluster"].nunique())
    summary["mean_frac_minute0_in_break"] = round(float(np.mean(frac0_list)), 4)
    summary["mean_frac_lastminute_in_break"] = round(float(np.mean(frac_last_list)), 4)
    summary["len_bin_counts"] = meta_df["len_bin"].value_counts().to_dict()
    summary["dur_touched_counts"] = meta_df["dur_touched_min"].value_counts().sort_index().to_dict()
    for col in ("rel_s", "rel_c", "rel_g", "rel_raw"):
        summary[f"{col}_nan_rate_leave"] = round(float(
            traj.loc[traj["phase"] == "leave", col].isna().mean()), 4)

    # Quarter-hour window sharing: how many real QH windows hold 2+ breaks.
    kbq = kb.copy()
    kbq["s_min"] = pd.to_datetime(kbq["break_start"]).dt.floor("min")
    kbq["e_min"] = pd.to_datetime(kbq["break_end"]).dt.floor("min")
    window_break: dict[tuple[str, int], set[int]] = {}
    for j, r in enumerate(kbq.itertuples(index=False)):
        ch = str(getattr(r, "channel"))
        s = int(getattr(r, "s_min").value // 60_000_000_000)
        e = int(getattr(r, "e_min").value // 60_000_000_000)
        for m in range(s, e + 1):
            window_break.setdefault((ch, m // 15), set()).add(j)
    counts = pd.Series([len(v) for v in window_break.values()])
    sharing = {
        "qh_windows_with_any_break": int(len(counts)),
        "windows_with_2plus_breaks": int((counts >= 2).sum()),
        "share_2plus": round(float((counts >= 2).mean()), 4),
        "distribution": counts.value_counts().sort_index().to_dict(),
    }
    # straddle frequency: breaks whose minutes span 2+ QH windows
    kbq["qh_s"] = kbq["s_min"].astype("int64") // 60_000_000_000 // 15
    kbq["qh_e"] = kbq["e_min"].astype("int64") // 60_000_000_000 // 15
    sharing["breaks_straddling_boundary"] = int((kbq["qh_s"] != kbq["qh_e"]).sum())
    sharing["share_straddling"] = round(float((kbq["qh_s"] != kbq["qh_e"]).mean()), 4)
    with open(OUT / "qh_sharing.json", "w", encoding="utf-8") as f:
        json.dump(sharing, f, indent=2, default=str)
    summary["qh_sharing"] = sharing

    with open(OUT / "prep_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
