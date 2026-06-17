"""Compute the measured per-break retention coefficients and write the JSON.

This is the fast, Meridian-free path to the optimizer's per-channel retention
deltas. It measures the real detrended effect of every break in the reference
data (see :mod:`kairos.model.measure`), pools thin cells, and writes
``models/tv_break_coefficients.json``. Once that file exists,
:func:`kairos.model.impact.load_impact_model` prefers it (source "measured"),
so the optimizer uses measured numbers without needing TensorFlow or Meridian.

Series layer (automatic gate)
------------------------------
Both genre-only and genre+series coefficients are ALWAYS computed. An
automatic held-out gate (:func:`kairos.model.series_gate.series_holdout_gate`)
compares their out-of-sample RMSE on 20 % of breaks withheld from training.
The series block is written to the JSON ONLY when series RMSE beats genre RMSE
by at least 2 % (the ``SERIES_GATE_MIN_RELATIVE_IMPROVEMENT`` constant). When
data is thin or titles are too sparse the gate fails and only the genre layer
is emitted -- which is today's behavior, now automatic and self-explaining.

The gate decision is transparent: ``series_layer_active`` (bool),
``series_gate_holdout`` (genre_rmse, series_rmse, n_test) and
``series_gate_reason`` (one-line explanation) are always written to the JSON
metadata so any reader can audit why the layer was activated or not.

Optional override flags
-----------------------
``--series force-on``  / ``--series force-off`` bypass the gate for debugging.
The default (omitting ``--series``) is the automatic gate.

Run from the repo root:

    PYTHONUTF8=1 python scripts/compute_measured_coefficients.py
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from kairos.data import ProgramClassifier
from kairos.data.loaders import DAILY_DIR, REFERENCE_DIR, load_dayparts, load_programmes, load_spots
from kairos.model.measure import (
    between_cell_variance,
    break_effects,
    channel_coefficients,
    first_break_gate,
    write_coefficients_json,
)
from kairos.model.freshness import COMPUTED_AT_KEY, FINGERPRINTS_KEY
from kairos.model.series import series_coefficients
from kairos.model.series_gate import series_holdout_gate
from kairos.observability.run_log import checksum_file

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "models" / "tv_break_coefficients.json"

# The exact source files the coefficients are measured from. break_effects reads
# spots, programmes and dayparts (load_spots/load_programmes/load_dayparts), which
# resolve to these three reference xlsx files. We fingerprint exactly these so a
# change to any of them is detectable as staleness, and nothing else is claimed.
SOURCE_FILES = (
    REFERENCE_DIR / "Spots.xlsx",
    REFERENCE_DIR / "Programmes.xlsx",
    REFERENCE_DIR / "Dayparts.xlsx",
)

# Sentinel values for the --series override flag.
_FORCE_ON = "force-on"
_FORCE_OFF = "force-off"


def _source_fingerprints() -> dict[str, str]:
    """Map each source file's relative POSIX path to its current sha256.

    Used at write time so the coefficients JSON records exactly what data it was
    computed from. A missing file is skipped here (the compute would already have
    failed to read it), so only files that fed the measurement are fingerprinted.
    """
    prints: dict[str, str] = {}
    for path in SOURCE_FILES:
        digest = checksum_file(path)
        if digest is not None:
            rel = path.relative_to(ROOT).as_posix()
            prints[rel] = digest
    return prints


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--series",
        choices=[_FORCE_ON, _FORCE_OFF],
        default=None,
        help=(
            "Override the automatic series gate. "
            f"'{_FORCE_ON}' always emits the series block; "
            f"'{_FORCE_OFF}' always omits it. "
            "Omit this flag (the default) to let the held-out gate decide. "
            "The env var KAIROS_SERIES_LAYER=force-on/force-off applies the same override."
        ),
    )
    args = parser.parse_args()

    # Honor the env var as well, so CI pipelines can override without editing
    # the command line. Command-line flag takes precedence over the env var.
    env_series = os.environ.get("KAIROS_SERIES_LAYER", "").strip().lower()
    series_override: str | None = args.series
    if series_override is None and env_series in (_FORCE_ON, _FORCE_OFF):
        series_override = env_series
    # Back-compat: the old "1" / "true" env value maps to force-on.
    if series_override is None and env_series in ("1", "true", "yes"):
        series_override = _FORCE_ON

    # Load the reference data and measure every break's detrended log effect.
    spots = load_spots()
    programmes = load_programmes()
    dayparts_frame = load_dayparts()
    classifier = ProgramClassifier.from_yaml()
    effects = break_effects(spots, programmes, dayparts_frame, classifier)

    # Genre-only coefficients (always computed).
    coefficients = channel_coefficients(effects)
    if not coefficients:
        raise SystemExit("No breaks measured; refusing to write an empty coefficients file.")
    diagnostics = between_cell_variance(effects)

    # First-break retention gate: measure whether the show's FIRST interruption
    # sheds more audience than later breaks, and ship a multiplier (> 1.0) only
    # when the contrast is large and significant. Off (1.0) otherwise.
    fb_gate = first_break_gate(effects)

    # Series layer: ALWAYS compute, gate decides whether to emit.
    series_all = series_coefficients(effects)

    # Run the automatic gate (compare genre vs genre+series out-of-sample).
    gate = series_holdout_gate(effects)

    # Decide whether to emit the series block.
    if series_override == _FORCE_ON:
        emit_series = True
        gate_reason_override = f"forced by --series {_FORCE_ON}; gate result: {gate['series_gate_reason']}"
        gate["series_gate_reason"] = gate_reason_override
    elif series_override == _FORCE_OFF:
        emit_series = False
        gate_reason_override = f"forced by --series {_FORCE_OFF}; gate result: {gate['series_gate_reason']}"
        gate["series_gate_reason"] = gate_reason_override
    else:
        emit_series = bool(gate["series_layer_active"])

    series_to_write = series_all if emit_series else {}

    total_breaks = sum(c.n for c in coefficients.values())
    negative = sum(1 for c in coefficients.values() if c.coefficient < 0)
    holdout = gate["series_gate_holdout"]
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
        # Series gate: always present so any reader can audit the decision.
        "series_layer_active": emit_series,
        "series_gate_holdout": {
            "genre_rmse": holdout["genre_rmse"],
            "series_rmse": holdout["series_rmse"],
            "n_test": holdout["n_test"],
        },
        "series_gate_reason": gate["series_gate_reason"],
        # Summary counts (unchanged from the old format).
        "series_layer": emit_series,
        "series_count": len(series_to_write),
        # First-break retention gate: the multiplier the optimizer applies to the
        # show's first break, plus the measured numbers and the gate's reason, so
        # the decision is fully auditable from the JSON.
        "first_break_multiplier": fb_gate["first_break_multiplier"],
        "first_break_active": fb_gate["first_break_active"],
        "first_break_n_first": fb_gate["first_break_n_first"],
        "first_break_n_later": fb_gate["first_break_n_later"],
        "first_break_mean_first": fb_gate["first_break_mean_first"],
        "first_break_mean_later": fb_gate["first_break_mean_later"],
        "first_break_p_value": fb_gate["first_break_p_value"],
        "first_break_reason": fb_gate["first_break_reason"],
    }
    # Freshness stamp: when these coefficients were computed and a sha256 of every
    # source file they were measured from. The freshness checker
    # (kairos.model.freshness) re-hashes those files later and reports stale when
    # the data has changed, so a stale number is detected instead of hidden. The
    # timestamp is generated here at the CLI entry, not inside a pure function, so
    # measure.py stays deterministic for its byte-stable JSON tests.
    metadata[COMPUTED_AT_KEY] = datetime.now(timezone.utc).isoformat()
    metadata[FINGERPRINTS_KEY] = _source_fingerprints()
    written = write_coefficients_json(
        OUTPUT_PATH, coefficients, metadata=metadata,
        series=series_to_write if series_to_write else None,
    )

    print(f"Wrote {len(coefficients)} measured coefficients to {written}")
    print(f"  total breaks measured: {total_breaks}")
    print(f"  negative cells: {negative} of {len(coefficients)}")
    print(f"  pooling: {diagnostics['method']}, tau^2={diagnostics['tau2']:.5g}, "
          f"learned pseudo-count={diagnostics['pseudo_count']}")
    print(f"  series gate: {gate['series_gate_reason']}")
    print(f"  first-break gate: {fb_gate['first_break_reason']}")
    if emit_series:
        print(f"  series layer ACTIVE: {len(series_to_write)} (cell, series) records emitted")
    else:
        print("  series layer INACTIVE (omitted from JSON)")
    most = sorted(coefficients.values(), key=lambda c: c.coefficient)[:3]
    for c in most:
        print(f"  {c.channel_name}: {c.coefficient:+.4f}  (n={c.n}, ci=[{c.ci_low:+.3f}, {c.ci_high:+.3f}])")


if __name__ == "__main__":
    main()
