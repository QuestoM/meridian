"""Measure the spot-level window clip (clip_to_all_ad_airtime) on the real month.

The shipped measurement clips windows against detected breaks (runs of >= 2
spots). Single-spot ad runs still leave commercial audience inside 9.44
percent of surviving windows (measured by scripts/analyze_afterwindow_bias.py).
This script measures what extending the clip to ALL ad airtime changes:

  * how many measurements are dropped or shortened by the extra boundaries,
  * that the residual spot-airtime overlap actually goes to zero,
  * per-cell and pooled coefficient shifts,
  * held-out skill on clean test breaks (identical target under both variants),
  * the first-break gate under both variants.

Writes ``models/candidates/tv_break_coefficients_spotclip.json`` with the gate
verdict in its metadata. The default pipeline is unchanged (the flag ships
OFF); adoption is the lead's call on this measured evidence.

    PYTHONUTF8=1 python scripts/measure_spotlevel_clip.py
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.measure import (
    break_effects,
    channel_coefficients,
    first_break_gate,
    write_coefficients_json,
)
from kairos.observability.run_log import checksum_file
from scripts.analyze_afterwindow_bias import (
    _holdout_metrics,
    _key_frame,
    _minute_overlaps_any,
    _spot_intervals,
    _window_minutes_for,
)

CANDIDATE = ROOT / "models" / "candidates" / "tv_break_coefficients_spotclip.json"
_SEED = 42
_FRACTION = 0.20
# Same activation discipline as the series gate: the variant must beat the
# incumbent's held-out RMSE by at least this relative margin to be recommended.
_MIN_RELATIVE_IMPROVEMENT = 0.02


def main() -> None:
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()
    classifier = ProgramClassifier.from_yaml()

    base = break_effects(spots, programmes, dayparts, classifier)
    variant = break_effects(
        spots, programmes, dayparts, classifier, clip_to_all_ad_airtime=True
    )
    print(f"breaks measured, break-level clip (shipped): {len(base)}")
    print(f"breaks measured, spot-level clip (variant): {len(variant)}")

    bk = _key_frame(base)
    vk = _key_frame(variant)
    merged = bk.merge(
        vk[["key", "log_effect"]].rename(columns={"log_effect": "log_effect_variant"}),
        on="key", how="left",
    )
    dropped = merged["log_effect_variant"].isna()
    changed = (~dropped) & (
        (merged["log_effect"] - merged["log_effect_variant"]).abs() > 1e-12
    )
    clean = (~dropped) & (~changed)
    print(f"dropped by spot-level clip: {int(dropped.sum())} ({100*dropped.mean():.2f}%)")
    print(f"shortened (measurement changed): {int(changed.sum())} ({100*changed.mean():.2f}%)")
    print(f"untouched: {int(clean.sum())} ({100*clean.mean():.2f}%)")

    # Residual overlap under the variant: recompute windows against ALL ad runs.
    from kairos.model.prepare import identify_breaks
    from kairos.model.measure import _neighbour_lookup

    all_runs = _neighbour_lookup(identify_breaks(spots, min_spots=1))
    spot_spans = _spot_intervals(spots)
    n_contam = 0
    n_min = n_min_contam = 0
    for row in vk.itertuples(index=False):
        channel = str(row.channel)
        before_ts, after_ts = _window_minutes_for(
            row, all_runs.get(channel, []), 3, 3
        )
        hit = False
        for t in before_ts + after_ts:
            n_min += 1
            if _minute_overlaps_any(spot_spans.get(channel, []), t):
                n_min_contam += 1
                hit = True
        if hit:
            n_contam += 1
    print(f"variant residual: breaks with spot-overlap minutes {n_contam} of {len(vk)} "
          f"({100*n_contam/max(1,len(vk)):.2f}%), minutes {n_min_contam} of {n_min} "
          f"({100*n_min_contam/max(1,n_min):.2f}%)")

    # Coefficient shifts.
    c_base = channel_coefficients(base)
    c_var = channel_coefficients(variant)
    rows = []
    for name in sorted(set(c_base) & set(c_var)):
        rows.append((name, c_base[name].coefficient, c_var[name].coefficient,
                     c_var[name].coefficient - c_base[name].coefficient,
                     c_base[name].n, c_var[name].n))
    shift = pd.DataFrame(rows, columns=["cell", "base", "variant", "shift", "n_base", "n_var"])
    shift = shift.sort_values("shift", key=lambda s: s.abs(), ascending=False)
    print("\ntop 5 per-cell shifts (variant minus base):")
    for r in shift.head(5).itertuples(index=False):
        print(f"  {r.cell}: {r.base:+.5f} -> {r.variant:+.5f} (shift {r.shift:+.5f}, n {r.n_base}->{r.n_var})")
    print(f"mean |shift|: {shift['shift'].abs().mean():.5f}, mean signed: {shift['shift'].mean():+.5f}")
    pooled_base = float(base["log_effect"].mean())
    pooled_var = float(variant["log_effect"].mean())
    print(f"pooled mean log_effect: base {pooled_base:+.5f}, variant {pooled_var:+.5f} "
          f"(shift {pooled_var - pooled_base:+.5f})")

    # Held-out on clean test breaks (identical target under both variants).
    clean_keys = merged.loc[clean, "key"].tolist()
    rng = np.random.default_rng(_SEED)
    n_test = max(1, int(round(len(clean_keys) * _FRACTION)))
    test_keys = set(tuple(clean_keys[i]) for i in rng.permutation(len(clean_keys))[:n_test])
    test = bk[bk["key"].isin(test_keys)]
    m_base = _holdout_metrics(bk[~bk["key"].isin(test_keys)], test)
    m_var = _holdout_metrics(vk[~vk["key"].isin(test_keys)], test)
    print(f"\nheld-out on {m_base['n_test']} clean test breaks:")
    print(f"  base (break-level clip): rmse {m_base['rmse']:.5f}, oos R2 {m_base['r2_vs_train_mean']:+.5f}")
    print(f"  variant (spot-level):    rmse {m_var['rmse']:.5f}, oos R2 {m_var['r2_vs_train_mean']:+.5f}")
    rel = (m_base["rmse"] - m_var["rmse"]) / m_base["rmse"] if m_base["rmse"] > 0 else 0.0
    adopt = rel >= _MIN_RELATIVE_IMPROVEMENT
    reason = (
        f"variant rmse improves base by {100*rel:.2f}% "
        f"(threshold {100*_MIN_RELATIVE_IMPROVEMENT:.0f}%): "
        + ("RECOMMEND ADOPT" if adopt else "keep OFF (within noise)")
    )
    print(f"  gate: {reason}")

    fb_base = first_break_gate(base)
    fb_var = first_break_gate(variant)
    print(f"\nfirst-break gate: base multiplier {fb_base['first_break_multiplier']} "
          f"(p={fb_base['first_break_p_value']:.3f}), variant {fb_var['first_break_multiplier']} "
          f"(p={fb_var['first_break_p_value']:.3f})")

    CANDIDATE.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "candidate": True,
        "purpose": "spot-level window clip (clip_to_all_ad_airtime=True) variant",
        "total_breaks_measured": int(sum(c.n for c in c_var.values())),
        "base_breaks": int(len(base)),
        "dropped_by_spot_clip": int(dropped.sum()),
        "shortened_by_spot_clip": int(changed.sum()),
        "residual_spot_overlap_breaks_pct": 100 * n_contam / max(1, len(vk)),
        "holdout_clean_test": {
            "n_test": m_base["n_test"],
            "rmse_base": m_base["rmse"],
            "rmse_variant": m_var["rmse"],
            "relative_improvement": rel,
            "adopt_recommended": bool(adopt),
            "reason": reason,
        },
        "pooled_log_effect_base": pooled_base,
        "pooled_log_effect_variant": pooled_var,
        "first_break_multiplier_variant": fb_var["first_break_multiplier"],
        "first_break_p_value_variant": fb_var["first_break_p_value"],
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source_fingerprints": {
            f"data/reference/{name}": checksum_file(ROOT / "data" / "reference" / name)
            for name in ("Spots.xlsx", "Programmes.xlsx", "Dayparts.xlsx")
        },
    }
    write_coefficients_json(CANDIDATE, c_var, metadata=metadata)
    print(f"\nwrote candidate: {CANDIDATE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
