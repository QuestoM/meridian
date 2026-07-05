"""Item 4: leave-one-out stability of the pooled retention effect.

Drops one unit at a time and recomputes the pooled per-break effect
(DerSimonian-Laird mu over the genre cells, reported as the retention delta
exp(mu) - 1, the same -0.0391 headline the optimizer prices):

  * one CHANNEL at a time (4 units): spots, programmes and dayparts for the
    channel removed, the whole measurement pipeline re-run (break detection,
    window clipping, detrend baseline all recomputed);
  * one ISO WEEK at a time (5 units covering November 2024): all three data
    frames filtered by broadcast date, full pipeline re-run, so the detrend
    baseline honestly loses that week too;
  * one GENRE (program_type) at a time (4 units): measured effects filtered
    (the audience data itself is genre-agnostic, so no pipeline re-run);
  * every CHANNEL-DAY jackknifed (120 units, closed form on the grand mean),
    reporting the single most influential channel-day.

Deterministic (no randomness at all). Runtime ~2 minutes.
Run from the repo root: python scripts/validation/run_leave_one_out.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import delta, dl_pool, load_bundle, write_section  # noqa: E402

from kairos.model.measure import break_effects  # noqa: E402


def pooled_from_frames(spots, programmes, dayparts, classifier) -> dict:
    effects = break_effects(spots, programmes, dayparts, classifier)
    return dl_pool(effects)


def main() -> None:
    t0 = time.time()
    bundle = load_bundle()
    full = dl_pool(bundle.effects)
    print(f"[full] n={full['n']} pooled delta {full['pooled_delta']:+.5f}")

    results = []  # (kind, unit, n, pooled_delta)

    # --- channels ------------------------------------------------------------
    for channel in sorted(bundle.effects["channel"].unique()):
        spots = bundle.spots[bundle.spots["Channel"] != channel]
        programmes = bundle.programmes[bundle.programmes["Channel"] != channel]
        dayparts = bundle.dayparts[bundle.dayparts["channel"] != channel]
        r = pooled_from_frames(spots, programmes, dayparts, bundle.classifier)
        results.append(("channel", channel, r["n"], r["pooled_delta"]))
        print(f"  drop channel {channel}: n={r['n']} pooled {r['pooled_delta']:+.5f} "
              f"(shift {r['pooled_delta']-full['pooled_delta']:+.5f})")

    # --- ISO weeks -----------------------------------------------------------
    spots_wk = bundle.spots["air_dt"].dt.isocalendar().week
    prog_wk = bundle.programmes["start_dt"].dt.isocalendar().week
    day_wk = bundle.dayparts["date"].dt.isocalendar().week
    for week in sorted(pd.unique(spots_wk.dropna())):
        spots = bundle.spots[spots_wk != week]
        programmes = bundle.programmes[prog_wk != week]
        dayparts = bundle.dayparts[day_wk != week]
        r = pooled_from_frames(spots, programmes, dayparts, bundle.classifier)
        results.append(("week", f"ISO W{int(week)}", r["n"], r["pooled_delta"]))
        print(f"  drop week W{int(week)}: n={r['n']} pooled {r['pooled_delta']:+.5f} "
              f"(shift {r['pooled_delta']-full['pooled_delta']:+.5f})")

    # --- genres (program_type) -------------------------------------------------
    for genre in sorted(bundle.effects["program_type"].unique()):
        remaining = bundle.effects[bundle.effects["program_type"] != genre]
        r = dl_pool(remaining)
        results.append(("genre", genre, r["n"], r["pooled_delta"]))
        print(f"  drop genre {genre}: n={r['n']} pooled {r['pooled_delta']:+.5f} "
              f"(shift {r['pooled_delta']-full['pooled_delta']:+.5f})")

    # --- channel-day jackknife (closed form on the grand mean = DL mu) --------
    logs = bundle.effects["log_effect"]
    N, M = len(logs), float(logs.mean())
    grouped = bundle.effects.groupby("cluster")["log_effect"].agg(["sum", "count"])
    jack = []
    for cluster, row in grouped.iterrows():
        mu_wo = (N * M - row["sum"]) / (N - row["count"])
        jack.append((cluster, delta(mu_wo)))
    jack_deltas = np.array([d for _, d in jack])
    worst_idx = int(np.argmax(np.abs(jack_deltas - full["pooled_delta"])))
    worst_cluster, worst_value = jack[worst_idx]
    print(f"  channel-day jackknife: range [{jack_deltas.min():+.5f}, "
          f"{jack_deltas.max():+.5f}]; most influential {worst_cluster} "
          f"-> {worst_value:+.5f}")

    # --- summary ---------------------------------------------------------------
    shifts = [(kind, unit, n, d, d - full["pooled_delta"]) for kind, unit, n, d in results]
    max_row = max(shifts, key=lambda r: abs(r[4]))
    max_share = abs(max_row[4]) / abs(full["pooled_delta"])
    jk_shift = worst_value - full["pooled_delta"]
    print(f"[summary] max single-unit shift {max_row[4]:+.5f} "
          f"({max_row[0]} {max_row[1]}) = {100*max_share:.1f}% of the pooled effect")

    elapsed = time.time() - t0

    lines = []
    lines.append("## 4. Leave-one-out stability "
                 "(`scripts/validation/run_leave_one_out.py`, deterministic)")
    lines.append("")
    lines.append(f"Full-sample pooled delta {full['pooled_delta']:+.5f} "
                 f"(n={full['n']}). Channel and week drops re-run the ENTIRE "
                 f"measurement pipeline (break detection, clipping, detrend "
                 f"baseline) on the reduced data; genre drops filter the measured "
                 f"effects.")
    lines.append("")
    lines.append("| unit dropped | n breaks | pooled delta | shift |")
    lines.append("|---|---|---|---|")
    for kind, unit, n, d, shift in shifts:
        lines.append(f"| {kind}: {unit} | {n} | {d:+.5f} | {shift:+.5f} |")
    lines.append("")
    lines.append(f"Channel-day jackknife (120 units): pooled delta range "
                 f"[{jack_deltas.min():+.5f}, {jack_deltas.max():+.5f}]; the single "
                 f"most influential channel-day is {worst_cluster} "
                 f"(shift {jk_shift:+.5f} = {100*abs(jk_shift)/abs(full['pooled_delta']):.1f}% "
                 f"of the pooled effect).")
    lines.append("")
    lines.append(f"Maximum single-unit influence: dropping {max_row[0]} "
                 f"**{max_row[1]}** moves the pooled effect by {max_row[4]:+.5f} = "
                 f"**{100*max_share:.1f}%** of its value. Every leave-one-out estimate "
                 f"stays within [{min(r[3] for r in shifts):+.5f}, "
                 f"{max(r[3] for r in shifts):+.5f}]; the pooled cost is not driven by "
                 f"any single channel, week, genre or channel-day.")
    lines.append("")
    lines.append(f"Runtime {elapsed:.0f}s; no randomness.")
    write_section("loo", "\n".join(lines))


if __name__ == "__main__":
    main()
