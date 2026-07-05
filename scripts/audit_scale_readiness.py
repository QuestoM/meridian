"""Two-years scale readiness audit for the retention-model layer.

Times every measurement stage on the real one-month reference data,
extrapolates to 24 months with the honest scaling law of each stage (linear
per-break loops vs the two quadratic span scans), measures the memory of the
minute-lookup structures, and computes the empirical-Bayes shrinkage
trajectory from the SHIPPED artifact's learned variance components, so the
"does the model self-transition from pooled to per-cell as data grows"
question gets a number instead of an opinion.

Also statically confirms which held-out gates re-run at rebuild time (they
must self-activate on richer data with no code change).

    PYTHONUTF8=1 python scripts/audit_scale_readiness.py
"""

from __future__ import annotations

import json
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots

SCALE = 24  # months of data expected vs the current one


def _timed(label: str, fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    return out, (label, dt)


def main() -> None:
    timings: list[tuple[str, float]] = []

    spots, t = _timed("load_spots (xlsx parse)", load_spots)
    timings.append(t)
    programmes, t = _timed("load_programmes (xlsx parse)", load_programmes)
    timings.append(t)
    dayparts, t = _timed("load_dayparts (xlsx parse)", load_dayparts)
    timings.append(t)
    classifier = ProgramClassifier.from_yaml()

    print("=== data volumes (one month) ===")
    print(f"spots rows: {len(spots)}, programmes rows: {len(programmes)}, "
          f"daypart channel-minutes: {len(dayparts)}")

    from kairos.model.prepare import identify_breaks, keyed_breaks

    runs, t = _timed("identify_breaks (iterrows loop)", identify_breaks, spots)
    timings.append(t)
    breaks, t = _timed("keyed_breaks (breaks + programme matching)",
                       keyed_breaks, spots, programmes, classifier)
    timings.append(t)

    from kairos.model import measure

    tracemalloc.start()
    frame = measure._dayparts_frame(dayparts)
    lookup = measure._minute_lookup(frame)
    baseline = measure._baseline_levels(frame)
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"minute-lookup entries: {len(lookup)}, baseline entries: {len(baseline)}")
    print(f"lookup-structures memory: {current/1e6:.0f} MB now -> "
          f"~{SCALE*current/1e6:.0f} MB at {SCALE} months (linear)")

    effects, t = _timed("break_effects (full measurement)",
                        measure.break_effects, spots, programmes, dayparts, classifier)
    timings.append(t)

    from kairos.model.competitor_model import measure_effects_with_competitors

    _, t = _timed("competitor features (attach, incl. re-measure)",
                  lambda: measure_effects_with_competitors(
                      spots=spots, programmes=programmes, dayparts=dayparts,
                      classifier=classifier))
    timings.append(t)

    from kairos.model.series import series_coefficients
    from kairos.model.series_gate import series_holdout_gate

    _, t = _timed("series layer + gate",
                  lambda: (series_coefficients(effects), series_holdout_gate(effects)))
    timings.append(t)

    print("\n=== stage timings, one month measured -> 24 months extrapolated ===")
    for label, dt in timings:
        print(f"  {label}: {dt:.2f}s -> ~{dt*SCALE:.0f}s linear")

    # The two known superlinear scans: per-break linear searches over the
    # channel's programme spans (title matching) and per-minute scans in the
    # window-mean lookups are dict-hash (linear). Count the quadratic term.
    per_channel_progs = programmes.groupby("Channel").size()
    per_channel_breaks = breaks.groupby("channel").size()
    quad_now = int((per_channel_progs * per_channel_breaks).sum())
    print(f"\n_title_for_break span-scan operations: ~{quad_now:.2e} now -> "
          f"~{quad_now*SCALE*SCALE:.2e} at {SCALE} months "
          f"(O(breaks x programmes) per channel, grows {SCALE*SCALE}x). "
          "Checklist: switch to bisect when the 2-year data lands.")

    # EB shrinkage trajectory from the SHIPPED artifact's learned components.
    shipped = json.loads((ROOT / "models" / "tv_break_coefficients.json").read_text("utf-8"))
    meta = shipped["metadata"]
    tau2 = float(meta["between_cell_variance_tau2"])
    s2 = float(meta["pooled_within_variance"])
    n_cells = int(meta["channels"])
    total = int(meta["total_breaks_measured"])
    median_n = total / n_cells
    print("\n=== EB shrinkage trajectory (from the shipped artifact's learned tau2, s2) ===")
    print(f"tau2 {tau2:.3e}, pooled within-variance {s2:.4f}, mean cell n {median_n:.0f}")
    for k in (1, 2, 6, 12, 24):
        n = median_n * k
        sigma2 = s2 / n
        shrink = sigma2 / (sigma2 + tau2)
        print(f"  {k:>2} months (cell n ~{n:,.0f}): {100*shrink:.0f}% of the cell mean "
              f"is pulled to the global mean, {100*(1-shrink):.0f}% is the cell's own data")

    # Gate wiring: which held-out gates re-run at rebuild time.
    compute_src = (ROOT / "scripts" / "compute_measured_coefficients.py").read_text("utf-8")
    print("\n=== held-out gates wired into the rebuild path (compute_measured_coefficients.py) ===")
    for gate, marker in (
        ("series layer (series_holdout_gate)", "series_holdout_gate"),
        ("first-break multiplier (first_break_gate)", "first_break_gate"),
        ("counter-programming covariate (counterprogramming_holdout_gate)",
         "counterprogramming_holdout_gate"),
    ):
        wired = marker in compute_src
        print(f"  {gate}: {'WIRED, re-evaluates each rebuild' if wired else 'NOT wired (candidate script only; lead decides adoption)'}")


if __name__ == "__main__":
    main()
