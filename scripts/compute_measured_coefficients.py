"""Compute the measured per-break retention coefficients and write the JSON.

This is the fast, Meridian-free path to the optimizer's per-channel retention
deltas. It measures the real detrended effect of every break in the reference
data (see :mod:`kairos.model.measure`), pools thin cells, and writes
``models/tv_break_coefficients.json``. Once that file exists,
:func:`kairos.model.impact.load_impact_model` prefers it (source "measured"),
so the optimizer uses measured numbers without needing TensorFlow or Meridian.

Run from the repo root:

    PYTHONUTF8=1 python scripts/compute_measured_coefficients.py
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kairos.data.loaders import DAILY_DIR, REFERENCE_DIR
from kairos.model.measure import (
    compute_measured_coefficients_with_diagnostics,
    write_coefficients_json,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "models" / "tv_break_coefficients.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        action="store_true",
        default=os.environ.get("KAIROS_SERIES_LAYER", "").lower() in ("1", "true", "yes"),
        help="Also emit the additive series-aware layer (genre -> series -> episode pooling).",
    )
    args = parser.parse_args()

    coefficients, diagnostics, series = compute_measured_coefficients_with_diagnostics(
        with_series=args.series,
    )
    if not coefficients:
        raise SystemExit("No breaks measured; refusing to write an empty coefficients file.")

    total_breaks = sum(c.n for c in coefficients.values())
    negative = sum(1 for c in coefficients.values() if c.coefficient < 0)
    metadata = {
        "source_data": str(REFERENCE_DIR.relative_to(ROOT)),
        "daily_input_dir": str(DAILY_DIR.relative_to(ROOT)),
        "channels": len(coefficients),
        "total_breaks_measured": total_breaks,
        "negative_cells": negative,
        "before_after_window_minutes": 3,
        "detrended": True,
        "pooled": True,
        # How the data, not a hand-set constant, set the partial-pooling strength.
        "pooling_method": diagnostics["method"],
        "between_cell_variance_tau2": diagnostics["tau2"],
        "pooled_within_variance": diagnostics["pooled_within_var"],
        "learned_pseudo_count": diagnostics["pseudo_count"],
        "series_layer": bool(series),
        "series_count": len(series),
    }
    written = write_coefficients_json(
        OUTPUT_PATH, coefficients, metadata=metadata, series=series or None,
    )

    print(f"Wrote {len(coefficients)} measured coefficients to {written}")
    print(f"  total breaks measured: {total_breaks}")
    print(f"  negative cells: {negative} of {len(coefficients)}")
    print(f"  pooling: {diagnostics['method']}, tau^2={diagnostics['tau2']:.5g}, "
          f"learned pseudo-count={diagnostics['pseudo_count']}")
    if series:
        print(f"  series layer: {len(series)} (cell, series) records emitted")
    most = sorted(coefficients.values(), key=lambda c: c.coefficient)[:3]
    for c in most:
        print(f"  {c.channel_name}: {c.coefficient:+.4f}  (n={c.n}, ci=[{c.ci_low:+.3f}, {c.ci_high:+.3f}])")


if __name__ == "__main__":
    main()
