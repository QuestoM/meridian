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

Counter-programming covariate (automatic gate)
----------------------------------------------
The rival-context covariate (docs/competitor-counterprogramming.md) follows
the same discipline. Every rebuild attaches the competitor features to the
measured breaks and re-runs the deterministic held-out gate
(:func:`kairos.model.competitor_gate.counterprogramming_holdout_gate`, fixed
seed): out-of-sample RMSE WITH vs WITHOUT the covariate. Only when WITH beats
WITHOUT by the required margin do the emitted coefficients become the
competition-adjusted ones and the fitted forward betas ship in the metadata;
otherwise the coefficients are exactly the plain measured ones (today's
verdict on one November). ``counterprogramming_active`` (bool),
``counterprogramming_holdout`` (rmse_without, rmse_with, n_test,
relative_improvement, min_relative_improvement) and
``counterprogramming_reason`` are always written so the decision is auditable
from the JSON. The covariate therefore self-activates the day the data
supports it, with no code change.

Detrend seasonality (evaluate-only verdict)
-------------------------------------------
Every rebuild also runs the season-aware detrend gate
(:func:`kairos.model.detrend_gate.detrend_seasonality_gate`) and records its
verdict in the metadata. This one NEVER self-activates: the baseline mode used
stays "global" (``detrend_baseline_mode`` in the metadata) and switching
:func:`kairos.model.measure.break_effects` to ``month_minute`` is an explicit
owner decision at the multi-year data drop.

Optional override flags
-----------------------
``--series force-on`` / ``--series force-off`` and ``--counterprogramming
force-on`` / ``--counterprogramming force-off`` bypass the respective gate for
debugging. The default (omitting the flag) is the automatic gate. ``--output``
writes the JSON somewhere other than the shipped artifact (used by the
rebuild-equivalence test).

Run from the repo root:

    PYTHONUTF8=1 python scripts/compute_measured_coefficients.py
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

from kairos.data import ProgramClassifier
from kairos.data.loaders import (
    DAILY_DIR,
    REFERENCE_DIR,
    _resolve_reference_path,
    load_dayparts,
    load_programmes,
    load_spots,
)
from kairos.model.competitor_features import (
    EXTENDED_ALL_FEATURES,
    attach_competitor_features,
)
from kairos.model.competitor_gate import counterprogramming_holdout_gate
from kairos.model.competitor_model import (
    adjust_effects_for_forward_competition,
    fit_competitor_betas,
)
from kairos.model.detrend_gate import detrend_seasonality_gate
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
    computed from. Each default xlsx is RESOLVED through the same
    :func:`kairos.data.loaders._resolve_reference_path` the loaders use, so when
    the reference workbooks are replaced by uploaded CSVs (the expected shape of
    the two-year data drop) the fingerprints follow the files that actually fed
    the measurement instead of pointing at absent xlsx and degrading freshness
    to ``unknown``. While the xlsx exist (today) the resolved paths are the xlsx
    and the fingerprints are byte-identical to before. A missing file is skipped
    (the compute would already have failed to read it).
    """
    prints: dict[str, str] = {}
    for default_xlsx in SOURCE_FILES:
        path = _resolve_reference_path(default_xlsx)
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
    parser.add_argument(
        "--counterprogramming",
        choices=[_FORCE_ON, _FORCE_OFF],
        default=None,
        help=(
            "Override the automatic counter-programming gate. "
            f"'{_FORCE_ON}' always emits competition-adjusted coefficients; "
            f"'{_FORCE_OFF}' always emits the plain measured ones. "
            "Omit this flag (the default) to let the held-out gate decide. "
            "The env var KAIROS_COUNTERPROGRAMMING=force-on/force-off applies "
            "the same override."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help=(
            "Where to write the coefficients JSON. Defaults to the shipped "
            "artifact; the rebuild-equivalence test points this at a temp path."
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

    env_cp = os.environ.get("KAIROS_COUNTERPROGRAMMING", "").strip().lower()
    cp_override: str | None = args.counterprogramming
    if cp_override is None and env_cp in (_FORCE_ON, _FORCE_OFF):
        cp_override = env_cp

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

    # Counter-programming gate: attach the rival-context features to the SAME
    # measured breaks and re-measure WITH vs WITHOUT on the deterministic
    # held-out split. Only a passing gate switches the emitted coefficients to
    # the competition-adjusted ones; a failing gate leaves every number above
    # untouched, so today's artifact is byte-equivalent to the pre-gate one.
    # The first-break and series gates stay measured on the raw effects: each
    # optional layer earns its way in against the same plain baseline,
    # independently.
    effects_cp = attach_competitor_features(
        effects, programmes, dayparts_frame, spots, classifier
    )
    cp_gate = counterprogramming_holdout_gate(effects_cp)
    if cp_override == _FORCE_ON:
        cp_active = True
        cp_gate["counterprogramming_reason"] = (
            f"forced by --counterprogramming {_FORCE_ON}; gate result: "
            f"{cp_gate['counterprogramming_reason']}"
        )
    elif cp_override == _FORCE_OFF:
        cp_active = False
        cp_gate["counterprogramming_reason"] = (
            f"forced by --counterprogramming {_FORCE_OFF}; gate result: "
            f"{cp_gate['counterprogramming_reason']}"
        )
    else:
        cp_active = bool(cp_gate["counterprogramming_active"])

    cp_betas_meta: dict[str, dict[str, object]] = {}
    if cp_active:
        # Full-data betas (the gate's are training-split only), applied the
        # same way the candidate artifact applies them: de-confound the log
        # effects by the forward features, then pool as usual.
        betas = fit_competitor_betas(effects_cp, feature_names=EXTENDED_ALL_FEATURES)
        if any(cb.role == "forward" for cb in betas.values()):
            adjusted = adjust_effects_for_forward_competition(effects_cp, betas)
            coefficients = channel_coefficients(adjusted)
            diagnostics = between_cell_variance(adjusted)
            cp_betas_meta = {
                name: {
                    "beta": cb.beta, "se": cb.se, "ci_low": cb.ci_low,
                    "ci_high": cb.ci_high, "role": cb.role,
                    "reference": cb.reference,
                }
                for name, cb in betas.items()
            }
        else:
            cp_active = False
            cp_gate["counterprogramming_reason"] += (
                "; no forward betas could be fitted on the full data, so the "
                "covariate stays off"
            )

    # Detrend seasonality: evaluate-only. The verdict is recorded so the
    # decision is on the table at the multi-year drop, but the baseline mode
    # used above stays "global" regardless (activation is an owner decision).
    dt_gate = detrend_seasonality_gate(dayparts_frame)

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
        # Counter-programming gate: verdict always present (pass/fail, both
        # RMSEs, the relative improvement and its pass threshold) so any
        # reader can audit why the covariate is on or off. The fitted betas
        # are written only when the covariate is ACTIVE, because only then are
        # they load-bearing (they de-confounded these coefficients and they
        # are what a forward adjustment would apply).
        "counterprogramming_active": cp_active,
        "counterprogramming_holdout": cp_gate["counterprogramming_holdout"],
        "counterprogramming_reason": cp_gate["counterprogramming_reason"],
        "counterprogramming_forward_features": cp_gate["forward_features"],
        # Detrend seasonality: the mode these coefficients were measured with,
        # plus the evaluate-only gate verdict for the multi-year drop.
        "detrend_baseline_mode": "global",
        "detrend_seasonality_recommended": dt_gate["detrend_seasonality_recommended"],
        "detrend_seasonality_holdout": dt_gate["detrend_seasonality_holdout"],
        "detrend_seasonality_reason": dt_gate["detrend_seasonality_reason"],
    }
    if cp_betas_meta:
        metadata["counterprogramming_betas"] = cp_betas_meta
    # Freshness stamp: when these coefficients were computed and a sha256 of every
    # source file they were measured from. The freshness checker
    # (kairos.model.freshness) re-hashes those files later and reports stale when
    # the data has changed, so a stale number is detected instead of hidden. The
    # timestamp is generated here at the CLI entry, not inside a pure function, so
    # measure.py stays deterministic for its byte-stable JSON tests.
    metadata[COMPUTED_AT_KEY] = datetime.now(timezone.utc).isoformat()
    metadata[FINGERPRINTS_KEY] = _source_fingerprints()
    written = write_coefficients_json(
        Path(args.output), coefficients, metadata=metadata,
        series=series_to_write if series_to_write else None,
    )

    print(f"Wrote {len(coefficients)} measured coefficients to {written}")
    print(f"  total breaks measured: {total_breaks}")
    print(f"  negative cells: {negative} of {len(coefficients)}")
    print(f"  pooling: {diagnostics['method']}, tau^2={diagnostics['tau2']:.5g}, "
          f"learned pseudo-count={diagnostics['pseudo_count']}")
    print(f"  series gate: {gate['series_gate_reason']}")
    print(f"  first-break gate: {fb_gate['first_break_reason']}")
    print(f"  counter-programming gate: {cp_gate['counterprogramming_reason']}")
    if cp_active:
        print("  counter-programming ACTIVE: coefficients are competition-adjusted "
              f"({len(cp_betas_meta)} betas in metadata)")
    else:
        print("  counter-programming INACTIVE (plain measured coefficients)")
    print(f"  detrend seasonality gate: {dt_gate['detrend_seasonality_reason']} "
          "(baseline mode used: global)")
    if emit_series:
        print(f"  series layer ACTIVE: {len(series_to_write)} (cell, series) records emitted")
    else:
        print("  series layer INACTIVE (omitted from JSON)")
    most = sorted(coefficients.values(), key=lambda c: c.coefficient)[:3]
    for c in most:
        print(f"  {c.channel_name}: {c.coefficient:+.4f}  (n={c.n}, ci=[{c.ci_low:+.3f}, {c.ci_high:+.3f}])")


if __name__ == "__main__":
    main()
