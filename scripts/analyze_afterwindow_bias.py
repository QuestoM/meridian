"""After-window de-bias verification on the real reference month.

Answers four questions with measured numbers, no opinion:

1. Does the shipped ``models/tv_break_coefficients.json`` equal a recompute
   with the current (clip-carrying) measurement code on the current data?
   If yes, the shipped artifact already contains the after-window fix.
2. What did the clip change? Ablation: re-measure with the adjacent-break
   clip disabled (the pre-fix behavior), and report contamination rates,
   per-cell coefficient shifts, the pooled shift, and the first-break gate
   under both pipelines.
3. Held-out skill before vs after: on CLEAN test breaks (windows untouched
   by clipping, so the target is identical under both pipelines), does a
   model trained on clipped measurements predict better than one trained on
   contaminated measurements?
4. Residual contamination: of the breaks the clipped pipeline still measures,
   how many window minutes overlap ANY aired spot (including single-spot runs
   that the >= 2-spots break detector does not call a break)? And how many
   after-windows cross a programme boundary (a different, non-ad confound)?

Writes the post-clip recompute to
``models/candidates/tv_break_coefficients_afterwindow.json`` with a
verification block, so the lead can adopt or discard with a receipt.
Read-only on shipped artifacts. Run from the repo root:

    PYTHONUTF8=1 python scripts/analyze_afterwindow_bias.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model import measure
from kairos.model.measure import (
    break_effects,
    channel_coefficients,
    first_break_gate,
    write_coefficients_json,
)
from kairos.observability.run_log import checksum_file

SHIPPED = ROOT / "models" / "tv_break_coefficients.json"
CANDIDATE = ROOT / "models" / "candidates" / "tv_break_coefficients_afterwindow.json"

_HOLDOUT_SEED = 42
_HOLDOUT_FRACTION = 0.20


def _key_frame(effects: pd.DataFrame) -> pd.DataFrame:
    out = effects.copy()
    out["key"] = list(zip(out["channel"].astype(str), out["break_start"]))
    return out


def _cell_means(frame: pd.DataFrame) -> dict[str, float]:
    return frame.groupby("channel_name")["log_effect"].mean().to_dict()


def _holdout_metrics(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, float]:
    """RMSE and out-of-sample R2 of cell-mean predictions on the test target."""
    means = _cell_means(train)
    global_mean = float(train["log_effect"].mean())
    y_true = test["log_effect"].to_numpy()
    y_pred = np.array(
        [means.get(str(r.channel_name), global_mean) for r in test.itertuples(index=False)]
    )
    mse_model = float(np.mean((y_true - y_pred) ** 2))
    mse_base = float(np.mean((y_true - global_mean) ** 2))
    return {
        "rmse": float(np.sqrt(mse_model)),
        "r2_vs_train_mean": (1.0 - mse_model / mse_base) if mse_base > 0 else 0.0,
        "n_test": int(len(test)),
    }


def _spot_intervals(spots: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Every aired spot's (air, air+duration) interval per channel, sorted."""
    frame = spots[spots["air_dt"].notna()].copy()
    frame["Duration"] = pd.to_numeric(frame.get("Duration"), errors="coerce").fillna(0.0)
    frame["end_dt"] = frame["air_dt"] + pd.to_timedelta(frame["Duration"], unit="s")
    out: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for channel, group in frame.groupby("Channel", sort=False):
        spans = sorted(
            (row.air_dt, row.end_dt) for row in group.itertuples(index=False)
        )
        out[str(channel)] = spans
    return out


def _minute_overlaps_any(
    spans: list[tuple[pd.Timestamp, pd.Timestamp]], t: pd.Timestamp
) -> bool:
    """True when any (start, end) interval intersects the minute [t, t+1)."""
    import bisect

    minute_end = t + pd.Timedelta(minutes=1)
    idx = bisect.bisect_right(spans, (minute_end, minute_end))
    # Only spans starting before minute_end can overlap; scan a short tail back.
    j = idx - 1
    while j >= 0:
        start, end = spans[j]
        if end <= t:
            # Spans are sorted by start; earlier spans can still be long, so scan
            # a bounded number back. Ad spots are < 5 min, so 32 is generous.
            if idx - j > 32:
                break
            j -= 1
            continue
        if start < minute_end and end > t:
            return True
        j -= 1
    return False


def _window_minutes_for(
    row, spans: list[tuple[pd.Timestamp, pd.Timestamp]], before: int, after: int
) -> tuple[list[pd.Timestamp], list[pd.Timestamp]]:
    """Reproduce break_effects's exact clipped windows for one measured break."""
    start = pd.Timestamp(row.break_start)
    end = pd.Timestamp(row.break_end)
    import bisect

    lo = bisect.bisect_left([s[0] for s in spans], start)
    prev_end = spans[lo - 1][1] if lo > 0 else None
    next_start = spans[lo + 1][0] if lo + 1 < len(spans) else None
    before_ts = [start - pd.Timedelta(minutes=k + 1) for k in range(before)]
    after_ts = [end + pd.Timedelta(minutes=k + 1) for k in range(after)]
    if prev_end is not None:
        before_ts = [t for t in before_ts if t > prev_end]
    if next_start is not None:
        after_ts = [t for t in after_ts if t < next_start]
    return before_ts, after_ts


def main() -> None:
    t0 = time.perf_counter()
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()
    classifier = ProgramClassifier.from_yaml()
    t_load = time.perf_counter() - t0

    # --- clipped (current code) and unclipped (pre-fix ablation) pipelines ----
    t0 = time.perf_counter()
    clipped = break_effects(spots, programmes, dayparts, classifier)
    t_measure = time.perf_counter() - t0

    original_lookup = measure._neighbour_lookup
    try:
        measure._neighbour_lookup = lambda breaks: {}
        unclipped = break_effects(spots, programmes, dayparts, classifier)
    finally:
        measure._neighbour_lookup = original_lookup

    print("=== pipeline sizes ===")
    print(f"data load: {t_load:.1f}s, clipped measurement: {t_measure:.1f}s")
    print(f"breaks measured, clipped (current code): {len(clipped)}")
    print(f"breaks measured, unclipped (pre-fix ablation): {len(unclipped)}")

    # --- 1. shipped artifact == current-code recompute? -----------------------
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    shipped_coeffs = shipped["coefficients"]
    coeffs_clipped = channel_coefficients(clipped)
    max_abs_diff = 0.0
    for name, c in coeffs_clipped.items():
        max_abs_diff = max(max_abs_diff, abs(c.coefficient - float(shipped_coeffs.get(name, np.nan))))
    same_cells = set(coeffs_clipped) == set(shipped_coeffs)
    matches = same_cells and max_abs_diff < 1e-12
    print("\n=== shipped-artifact verification ===")
    print(f"cells match: {same_cells} ({len(coeffs_clipped)} cells)")
    print(f"max |coefficient - shipped|: {max_abs_diff:.3e}")
    print(f"VERDICT: shipped artifact {'IS' if matches else 'IS NOT'} the post-clip recompute")

    # --- 2. contamination accounting ------------------------------------------
    ck = _key_frame(clipped)
    uk = _key_frame(unclipped)
    merged = uk.merge(
        ck[["key", "log_effect"]].rename(columns={"log_effect": "log_effect_clipped"}),
        on="key", how="left",
    )
    dropped = merged["log_effect_clipped"].isna()
    changed = (~dropped) & (
        (merged["log_effect"] - merged["log_effect_clipped"]).abs() > 1e-12
    )
    clean = (~dropped) & (~changed)
    n = len(merged)
    print("\n=== contamination accounting (per unclipped measurement) ===")
    print(f"dropped by clip (window < 1 clean minute): {int(dropped.sum())} ({100*dropped.mean():.1f}%)")
    print(f"window shortened (measurement changed): {int(changed.sum())} ({100*changed.mean():.1f}%)")
    print(f"untouched (clean windows): {int(clean.sum())} ({100*clean.mean():.1f}%)")
    contaminated_frac = (dropped.sum() + changed.sum()) / n
    print(f"total windows overlapping an adjacent break: {100*contaminated_frac:.1f}%")

    bias = merged.loc[changed, "log_effect"] - merged.loc[changed, "log_effect_clipped"]
    print("\nbias in the shortened subset (unclipped minus clipped log_effect):")
    print(f"  mean {bias.mean():+.5f}, median {bias.median():+.5f}, n={len(bias)}")
    pooled_unclipped = float(unclipped["log_effect"].mean())
    pooled_clipped = float(clipped["log_effect"].mean())
    print("\npooled mean log_effect:")
    print(f"  unclipped {pooled_unclipped:+.5f} -> delta {np.exp(pooled_unclipped)-1:+.5f}")
    print(f"  clipped   {pooled_clipped:+.5f} -> delta {np.exp(pooled_clipped)-1:+.5f}")
    print(f"  pooled bias removed (log): {pooled_unclipped - pooled_clipped:+.5f}")

    # --- per-cell coefficient shifts -------------------------------------------
    coeffs_unclipped = channel_coefficients(unclipped)
    rows = []
    for name in sorted(set(coeffs_clipped) | set(coeffs_unclipped)):
        c_new = coeffs_clipped.get(name)
        c_old = coeffs_unclipped.get(name)
        if c_new is None or c_old is None:
            continue
        rows.append((name, c_old.coefficient, c_new.coefficient,
                     c_new.coefficient - c_old.coefficient, c_old.n, c_new.n))
    shift = pd.DataFrame(rows, columns=["cell", "old", "new", "shift", "n_old", "n_new"])
    shift = shift.sort_values("shift", key=lambda s: s.abs(), ascending=False)
    print("\n=== per-cell coefficient shifts (clip minus no-clip), top 10 by |shift| ===")
    for r in shift.head(10).itertuples(index=False):
        print(f"  {r.cell}: {r.old:+.5f} -> {r.new:+.5f} (shift {r.shift:+.5f}, n {r.n_old}->{r.n_new})")
    print(f"mean |shift| across {len(shift)} cells: {shift['shift'].abs().mean():.5f}")
    print(f"mean signed shift: {shift['shift'].mean():+.5f}")

    # --- first-break gate under both pipelines ---------------------------------
    fb_old = first_break_gate(unclipped)
    fb_new = first_break_gate(clipped)
    print("\n=== first-break gate ===")
    print(f"unclipped: multiplier {fb_old['first_break_multiplier']}, active {fb_old['first_break_active']}, p={fb_old['first_break_p_value']:.2e}")
    print(f"clipped:   multiplier {fb_new['first_break_multiplier']}, active {fb_new['first_break_active']}, p={fb_new['first_break_p_value']:.2e}")

    # --- 3. held-out skill on clean test breaks --------------------------------
    clean_keys = merged.loc[clean, "key"].tolist()
    rng = np.random.default_rng(_HOLDOUT_SEED)
    n_test = max(1, int(round(len(clean_keys) * _HOLDOUT_FRACTION)))
    test_keys = set(
        tuple(clean_keys[i]) for i in rng.permutation(len(clean_keys))[:n_test]
    )
    test = ck[ck["key"].isin(test_keys)]
    train_clipped = ck[~ck["key"].isin(test_keys)]
    train_unclipped = uk[~uk["key"].isin(test_keys)]
    m_clip = _holdout_metrics(train_clipped, test)
    m_unclip = _holdout_metrics(train_unclipped, test)
    print("\n=== held-out skill on clean test breaks (identical target both pipelines) ===")
    print(f"n_test={m_clip['n_test']} clean breaks (20% of {len(clean_keys)} clean)")
    print(f"train on UNCLIPPED (contaminated): rmse {m_unclip['rmse']:.5f}, oos R2 {m_unclip['r2_vs_train_mean']:+.5f}")
    print(f"train on CLIPPED (de-biased):      rmse {m_clip['rmse']:.5f}, oos R2 {m_clip['r2_vs_train_mean']:+.5f}")

    # Each pipeline's own-measurement skill (series-gate style), for context.
    def own_split(frame: pd.DataFrame) -> dict[str, float]:
        r = np.random.default_rng(_HOLDOUT_SEED)
        idx = r.permutation(len(frame))
        k = max(1, int(round(len(frame) * _HOLDOUT_FRACTION)))
        te = frame.iloc[idx[:k]]
        tr = frame.iloc[idx[k:]]
        return _holdout_metrics(tr, te)

    o_clip = own_split(clipped)
    o_unclip = own_split(unclipped)
    print("\nown-measurement 80/20 skill (context, targets differ):")
    print(f"unclipped: rmse {o_unclip['rmse']:.5f}, oos R2 {o_unclip['r2_vs_train_mean']:+.5f} (n={o_unclip['n_test']})")
    print(f"clipped:   rmse {o_clip['rmse']:.5f}, oos R2 {o_clip['r2_vs_train_mean']:+.5f} (n={o_clip['n_test']})")

    # --- 4. residual contamination of the surviving windows --------------------
    spot_spans = _spot_intervals(spots)
    detected = measure._neighbour_lookup(
        ck.rename(columns={})[["channel", "break_start", "break_end"]]
    )
    n_breaks_contam = 0
    n_minutes = 0
    n_minutes_contam = 0
    n_after_prog_boundary = 0
    prog_spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    pframe = programmes[programmes["start_dt"].notna()]
    for channel, group in pframe.groupby("Channel", sort=False):
        prog_spans[str(channel)] = sorted(
            (r.start_dt, r.end_dt) for r in group.itertuples(index=False)
            if pd.notna(r.start_dt) and pd.notna(r.end_dt)
        )
    for row in ck.itertuples(index=False):
        channel = str(row.channel)
        spans = detected.get(channel, [])
        before_ts, after_ts = _window_minutes_for(row, spans, 3, 3)
        own = (pd.Timestamp(row.break_start), pd.Timestamp(row.break_end))
        all_spots = spot_spans.get(channel, [])
        hit = False
        for t in before_ts + after_ts:
            n_minutes += 1
            if _minute_overlaps_any(all_spots, t):
                n_minutes_contam += 1
                hit = True
        if hit:
            n_breaks_contam += 1
        # Programme boundary inside the after-window (different confound, not ads).
        starts = [s for s, _ in prog_spans.get(channel, [])]
        for t in after_ts:
            import bisect

            j = bisect.bisect_left(starts, t)
            if j < len(starts) and starts[j] < t + pd.Timedelta(minutes=1):
                n_after_prog_boundary += 1
                break
    print("\n=== residual contamination of surviving (clipped) windows ===")
    print(f"breaks with >=1 window minute overlapping ANY aired spot: {n_breaks_contam} of {len(ck)} ({100*n_breaks_contam/len(ck):.2f}%)")
    print(f"window minutes overlapping spot airtime: {n_minutes_contam} of {n_minutes} ({100*n_minutes_contam/max(1,n_minutes):.2f}%)")
    print(f"after-windows crossing a programme START (non-ad confound): {n_after_prog_boundary} of {len(ck)} ({100*n_after_prog_boundary/len(ck):.2f}%)")

    # --- candidate artifact -----------------------------------------------------
    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "candidate": True,
        "purpose": "after-window de-bias verification recompute (post-clip code, reference data)",
        "verification_matches_shipped": bool(matches),
        "max_abs_coefficient_diff_vs_shipped": max_abs_diff,
        "total_breaks_measured": int(sum(c.n for c in coeffs_clipped.values())),
        "unclipped_breaks": int(len(unclipped)),
        "clip_dropped": int(dropped.sum()),
        "clip_shortened": int(changed.sum()),
        "holdout_clean_test": {
            "n_test": m_clip["n_test"],
            "rmse_train_unclipped": m_unclip["rmse"],
            "rmse_train_clipped": m_clip["rmse"],
            "r2_train_unclipped": m_unclip["r2_vs_train_mean"],
            "r2_train_clipped": m_clip["r2_vs_train_mean"],
        },
        "residual_spot_overlap_breaks_pct": 100 * n_breaks_contam / len(ck),
        "residual_spot_overlap_minutes_pct": 100 * n_minutes_contam / max(1, n_minutes),
        "after_window_programme_start_pct": 100 * n_after_prog_boundary / len(ck),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprints": {
            f"data/reference/{name}": checksum_file(ROOT / "data" / "reference" / name)
            for name in ("Spots.xlsx", "Programmes.xlsx", "Dayparts.xlsx")
        },
    }
    write_coefficients_json(CANDIDATE, coeffs_clipped, metadata=metadata)
    print(f"\nwrote candidate: {CANDIDATE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
