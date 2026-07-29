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

Placebo-drift correction (measured always, applied only when forced on)
------------------------------------------------------------------------
The causal-identification review (docs/model-validation/causal-identification.md)
measured that no-break minutes drift POSITIVE under the shipped null (matched
placebo +0.0151 delta, z = +4.12), so every shipped per-break cost is
understated by its genre's within-show drift. Every rebuild re-measures that
drift (:func:`kairos.model.placebo_correction.measure_placebo_drift`, seeded)
and ALWAYS writes it to the metadata (``placebo_correction_active``,
``placebo_correction`` with pooled_drift / per_genre_drift / n_pseudo,
``placebo_correction_reason``). The correction is applied by
default (it moves the optimizer's per-break charge by roughly a third; the
measured plan and revenue movement are recorded in
docs/model-validation/README.md); ``--placebo-correction force-off`` or
KAIROS_PLACEBO_CORRECTION=force-off disables it for diagnostics. When ON, the
review's fix 2 rides along by construction: effects are re-measured with the
content-only detrend baseline (``baseline_content_only=True``), the drift is
re-measured under that same baseline, and each genre's drift is subtracted
from the raw effects BEFORE the usual DL/EB pooling (measured pair: pooled
about -0.0496; the content-only baseline alone moves AWAY from the causal
value, which is why the two ship together).

Detrend seasonality (evaluate-only verdict)
-------------------------------------------
Every rebuild also runs the season-aware detrend gate
(:func:`kairos.model.detrend_gate.detrend_seasonality_gate`) and records its
verdict in the metadata. This one NEVER self-activates: the baseline mode used
stays "global" (``detrend_baseline_mode`` in the metadata) and switching
:func:`kairos.model.measure.break_effects` to ``month_minute`` is an explicit
deliberate configuration decision at the multi-year data drop.

Interval calibration (measured always, applied only when forced on)
--------------------------------------------------------------------
The uncertainty-calibration review (docs/model-validation/uncertainty-calibration.md)
measured that the shipped plug-in 95% band undercovers (82.6% true coverage at
today's sample size; a ~1.77x widening is needed, self-healing with data)
because tau^2 estimation error is not propagated. Every rebuild runs the
seeded parametric bootstrap (:mod:`kairos.model.interval_calibration`) and
ALWAYS writes the measured verdict (``interval_method``,
``moderated_variances``, ``bootstrap_B``, ``prior_df``,
``width_factor_measured``, ``interval_calibration_reason``).
``--interval-calibration force-on`` replaces ci_low/ci_high with the
calibrated (Laird-Louis mixture) quantiles - the POINT coefficients are
untouched - and adds per-cell ``predictive_low``/``predictive_high`` for a
single future break (separate keys; ci_* semantics never change).
``--moderated-variances force-on`` additionally swaps limma-moderated
per-cell within-variances into the DL weights (this one moves the point
coefficients, which is why it never self-activates). Env overrides:
KAIROS_INTERVAL_CALIBRATION / KAIROS_MODERATED_VARIANCES = force-on/force-off.

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
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

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
from kairos.model.drift_monitor import level_drift
from kairos.model.event_gate import annotate_event_columns, event_layer_gate
from kairos.model.measure import (
    between_cell_variance,
    break_effects,
    channel_coefficients,
    first_break_gate,
    write_coefficients_json,
)
from kairos.model.freshness import COMPUTED_AT_KEY, FINGERPRINTS_KEY
from kairos.model.interval_calibration import (
    DEFAULT_BOOTSTRAP_B,
    DEFAULT_CALIBRATION_SEED,
    apply_calibrated_intervals,
    calibrate_intervals,
    coefficient_map,
    predictive_bands,
)
from kairos.model.placebo_correction import (
    apply_placebo_correction,
    measure_placebo_drift,
)
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
        "--placebo-correction",
        choices=[_FORCE_ON, _FORCE_OFF],
        default=None,
        help=(
            "Override the placebo-drift correction layer. Default (flag omitted) "
            "is OFF: there is no automatic gate, because applying the correction "
            "moves the optimizer's per-break charge by roughly a third and is an "
            "explicit configuration decision. "
            f"'{_FORCE_ON}' emits placebo-corrected coefficients: effects are "
            "re-measured with the content-only detrend baseline and each genre's "
            "measured no-break drift is subtracted before pooling. "
            f"'{_FORCE_OFF}' behaves like the default; either way the measured "
            "drift is always written to the metadata. The env var "
            "KAIROS_PLACEBO_CORRECTION=force-on/force-off applies the same "
            "override."
        ),
    )
    parser.add_argument(
        "--interval-calibration",
        choices=[_FORCE_ON, _FORCE_OFF],
        default=None,
        help=(
            "Override the interval-calibration layer. Default (flag omitted) is "
            "OFF: there is no automatic gate, because the calibrated bands widen "
            "the ci_low the risk_lambda decision prices, an explicit owner "
            "decision. "
            f"'{_FORCE_ON}' replaces ci_low/ci_high with seeded parametric-"
            "bootstrap quantiles that propagate tau^2 estimation error (points "
            "untouched) and adds per-cell predictive_low/predictive_high. "
            f"'{_FORCE_OFF}' behaves like the default; either way the measured "
            "width factor is always written to the metadata. The env var "
            "KAIROS_INTERVAL_CALIBRATION=force-on/force-off applies the same "
            "override."
        ),
    )
    parser.add_argument(
        "--moderated-variances",
        choices=[_FORCE_ON, _FORCE_OFF],
        default=None,
        help=(
            "Override the limma moderated-variance layer (default OFF). "
            f"'{_FORCE_ON}' shrinks each cell's within-variance toward the "
            "learned prior and uses the moderated variances in the DL weights "
            "and intervals; this MOVES the point coefficients, so it never "
            "self-activates. The prior df is always measured and written to "
            "the metadata either way. The env var "
            "KAIROS_MODERATED_VARIANCES=force-on/force-off applies the same "
            "override."
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

    env_pc = os.environ.get("KAIROS_PLACEBO_CORRECTION", "").strip().lower()
    pc_override: str | None = args.placebo_correction
    if pc_override is None and env_pc in (_FORCE_ON, _FORCE_OFF):
        pc_override = env_pc

    env_ic = os.environ.get("KAIROS_INTERVAL_CALIBRATION", "").strip().lower()
    ic_override: str | None = args.interval_calibration
    if ic_override is None and env_ic in (_FORCE_ON, _FORCE_OFF):
        ic_override = env_ic

    env_mv = os.environ.get("KAIROS_MODERATED_VARIANCES", "").strip().lower()
    mv_override: str | None = args.moderated_variances
    if mv_override is None and env_mv in (_FORCE_ON, _FORCE_OFF):
        mv_override = env_mv

    # Load the reference data and measure every break's detrended log effect.
    spots = load_spots()
    programmes = load_programmes()
    dayparts_frame = load_dayparts()
    classifier = ProgramClassifier.from_yaml()
    effects = break_effects(spots, programmes, dayparts_frame, classifier)
    # Calendar-event annotation seam: join the operator's events store onto the
    # measured breaks by date (event_active / event_intensity / event_type).
    # Purely additive columns; the pooling below never reads them, so every
    # emitted coefficient is byte-identical with or without the seam.
    effects = annotate_event_columns(effects)

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

    # Placebo-drift correction: measured on EVERY rebuild, applied only when
    # forced on. OFF (the default) leaves every coefficient above untouched
    # and records the drift that WOULD be subtracted, measured under the
    # shipped baseline. ON re-measures the effects with the content-only
    # baseline (fix 2), re-measures the matched drift under that same
    # baseline, subtracts each genre's drift from the raw effects (fix 1),
    # and re-pools with the same DL/EB machinery; the review measured that
    # fix 2 alone moves AWAY from the causal value, so the pair is atomic.
    # The layer is measured on plain (non-competition-adjusted) effects; when
    # forced on it therefore replaces any competition-adjusted coefficients,
    # and the reason string records that.
    # The placebo-drift correction is applied by default; the measured plan
    # and revenue movement behind that default are recorded in
    # docs/model-validation/README.md. force-off remains for diagnostics.
    pc_active = pc_override != _FORCE_OFF
    if pc_active:
        effects_content = break_effects(
            spots, programmes, dayparts_frame, classifier,
            baseline_content_only=True,
        )
        correction = measure_placebo_drift(
            spots, programmes, dayparts_frame, classifier, effects_content,
            baseline_content_only=True,
        )
        corrected_effects = apply_placebo_correction(effects_content, correction)
        coefficients = channel_coefficients(corrected_effects)
        if not coefficients:
            raise SystemExit(
                "No breaks measured under the content-only baseline; refusing "
                "to write an empty coefficients file."
            )
        diagnostics = between_cell_variance(corrected_effects)
        source = (
            f"forced by --placebo-correction {_FORCE_ON} (or env)"
            if pc_override == _FORCE_ON
            else "active by default"
        )
        pc_reason = (
            f"{source}: content-only "
            "baseline applied and each genre's measured no-break drift "
            f"(pooled {correction.pooled_drift:+.5f} log over "
            f"{correction.n_pseudo} matched pseudo-breaks) subtracted from the "
            "raw effects before pooling"
        )
        if cp_active:
            pc_reason += (
                "; overrides the competition-adjusted coefficients (the placebo "
                "layer is measured on plain effects)"
            )
    else:
        correction = measure_placebo_drift(
            spots, programmes, dayparts_frame, classifier, effects,
        )
        source = (
            f"--placebo-correction {_FORCE_OFF} (or env)"
            if pc_override == _FORCE_OFF
            else "default"
        )
        pc_reason = (
            f"correction left OFF ({source}); drift measured and recorded only "
            f"(pooled {correction.pooled_drift:+.5f} log over "
            f"{correction.n_pseudo} matched pseudo-breaks). Activation is an "
            "explicit decision: the corrected charge moves every "
            "coefficient by its genre's measured drift"
        )
    pc_meta = correction.as_metadata()
    if pc_active:
        # The corrected pooled delta on the optimizer's scale, so the headline
        # of the applied correction is auditable straight from the JSON.
        pc_meta["pooled_corrected_delta"] = float(
            np.exp(float(corrected_effects["log_effect"].mean())) - 1.0
        )

    # Interval calibration (parametric-bootstrap tau^2 propagation) and
    # limma moderated variances: MEASURED on every rebuild so the metadata
    # always reports the honest widening the plug-in band omits, APPLIED only
    # under an explicit force-on. The calibration runs on the SAME effects the
    # emitted coefficients were pooled from (placebo-corrected when that layer
    # is on, else competition-adjusted when that gate is active, else plain).
    if pc_active:
        effects_for_intervals = corrected_effects
    elif cp_active:
        effects_for_intervals = adjusted
    else:
        effects_for_intervals = effects
    # Calibrated intervals are applied by default: the point coefficients are
    # untouched and ci_low/ci_high honestly carry tau^2 estimation error.
    interval_on = ic_override != _FORCE_OFF
    moderated_on = mv_override == _FORCE_ON
    calibration = None
    calibration_error = ""
    try:
        calibration = calibrate_intervals(
            effects_for_intervals,
            bootstrap_b=DEFAULT_BOOTSTRAP_B,
            seed=DEFAULT_CALIBRATION_SEED,
            moderated=moderated_on,
        )
    except ValueError as exc:
        calibration_error = str(exc)

    predictive: dict[str, tuple[float, float]] = {}
    interval_method = "naive"
    width_factor = None
    prior_df = None
    if calibration is None:
        ic_reason = (
            f"unavailable ({calibration_error}); shipped plug-in intervals kept"
        )
    else:
        width_factor = calibration.width_factor()
        if math.isfinite(calibration.prior.df):
            prior_df = float(calibration.prior.df)
        if calibration.moderated:
            # Moderated DL weights legitimately move the point estimates;
            # only an explicit force-on reaches this branch.
            coefficients = coefficient_map(calibration, plugin=not interval_on)
        elif interval_on:
            # Points carried over bit-for-bit; ONLY ci_low/ci_high replaced.
            coefficients = apply_calibrated_intervals(coefficients, calibration)
        if interval_on:
            interval_method = "bootstrap"
            predictive = predictive_bands(calibration)
            ic_source = (
                f"forced by --interval-calibration {_FORCE_ON} (or env)"
                if ic_override == _FORCE_ON
                else "active by default"
            )
            ic_reason = (
                f"{ic_source}: "
                "ci_low/ci_high are seeded parametric-bootstrap mixture "
                f"quantiles (B={calibration.bootstrap_b}, measured width "
                f"factor {width_factor:.2f}x at 95%); predictive_low/"
                "predictive_high added per cell for a single future break"
            )
        else:
            source = (
                f"--interval-calibration {_FORCE_OFF} (or env)"
                if ic_override == _FORCE_OFF
                else "default"
            )
            ic_reason = (
                f"calibration left OFF ({source}); the seeded bootstrap "
                f"measures the plug-in band would need to widen "
                f"{width_factor:.2f}x at 95% to honestly carry tau^2 "
                "estimation error. Activation widens the ci_low that "
                "risk_lambda prices and is an explicit configuration decision"
            )
        if moderated_on:
            if calibration.moderated:
                label = (
                    f"prior df {prior_df:.1f}"
                    if prior_df is not None
                    else "infinite prior df"
                )
                ic_reason += (
                    f"; moderated per-cell variances APPLIED ({label}) - "
                    "DL weights and point coefficients re-weighted"
                )
            else:
                ic_reason += (
                    "; moderated variances requested but unavailable "
                    "(variance prior could not be estimated)"
                )

    # Detrend seasonality: evaluate-only. The verdict is recorded so the
    # decision is on the table at the multi-year drop, but the baseline mode
    # used above stays "global" regardless (activation is a deliberate switch).
    dt_gate = detrend_seasonality_gate(dayparts_frame)

    # Event layer gate: five temporal folds, +2 percent held-out RMSE bar,
    # re-measured on every rebuild against the events store, so the layer
    # self-activates the day history with real event contrast lands. The
    # verdict never alters the coefficients above; it is a recorded decision
    # the events API's model context surfaces (tri-state honest).
    ev_gate = event_layer_gate(effects)

    # Weekly level drift of the measurement base: the binding nonstationarity
    # risk from docs/model-validation/uncertainty-calibration.md, measured from
    # the same plain effects on every rebuild (like the other gates, it never
    # depends on optional layers). Honest absent state under two weeks of data
    # (see kairos.model.drift_monitor).
    drift = level_drift(effects)

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
        # Carries the whole holdout block, including the fold-averaged
        # statistic fields (gate_statistic_method, folds, fold_sd).
        "series_layer_active": emit_series,
        "series_gate_holdout": dict(holdout),
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
        # Placebo-drift correction: the measured no-break drift is ALWAYS
        # recorded (pooled and per genre, with counts and cluster-robust SEs)
        # so any reader can see the correction the causal review calls for;
        # placebo_correction_active says whether it was actually applied to
        # the coefficients above (only under an explicit force-on).
        "placebo_correction_active": pc_active,
        "placebo_correction": pc_meta,
        "placebo_correction_reason": pc_reason,
        # Detrend seasonality: the mode these coefficients were measured with,
        # plus the evaluate-only gate verdict for the multi-year drop.
        "detrend_baseline_mode": "global",
        "detrend_seasonality_recommended": dt_gate["detrend_seasonality_recommended"],
        "detrend_seasonality_holdout": dt_gate["detrend_seasonality_holdout"],
        "detrend_seasonality_reason": dt_gate["detrend_seasonality_reason"],
        # Event layer gate: verdict on/off, reason, fold-mean held-out delta in
        # percent (null when no contrast could be measured) and the measurement
        # timestamp, re-measured on every rebuild from the events store.
        "event_layer_gate": dict(ev_gate),
        # Weekly level drift of the measurement base: weekly_levels,
        # drift_per_week, drift_se, binding and the criterion, measured at
        # rebuild time so the artifact always says whether the level the plan
        # runs on is still the level the coefficients were pooled on.
        "level_drift": drift,
        # Interval calibration: the method actually used for ci_low/ci_high
        # ("naive" plug-in or "bootstrap" mixture quantiles), whether limma
        # moderated variances were APPLIED to the DL weights, and the measured
        # honest widening -- always present so any reader can see the plug-in
        # band's optimism even when the correction is OFF.
        "interval_method": interval_method,
        "moderated_variances": bool(calibration.moderated) if calibration else False,
        "bootstrap_B": calibration.bootstrap_b if calibration else None,
        "interval_seed": calibration.seed if calibration else None,
        "prior_df": prior_df,
        "width_factor_measured": width_factor,
        "interval_calibration_reason": ic_reason,
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
    if predictive:
        # Additive per-cell predictive band (single future break). Injected
        # after the standard writer so the detail schema stays owned by
        # measure.py; every existing reader ignores unknown keys. Only written
        # when the calibration is ON, so the OFF artifact stays byte-stable.
        payload = json.loads(written.read_text(encoding="utf-8"))
        detail_block = payload.get("detail", {})
        for name, (p_lo, p_hi) in predictive.items():
            cell = detail_block.get(name)
            if cell is not None:
                cell["predictive_low"] = p_lo
                cell["predictive_high"] = p_hi
        written.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
    print(f"  placebo correction: {pc_reason}")
    if pc_active:
        print("  placebo correction ACTIVE: drift-corrected, content-only-baseline "
              f"coefficients emitted (pooled corrected delta "
              f"{pc_meta['pooled_corrected_delta']:+.5f})")
    print(f"  detrend seasonality gate: {dt_gate['detrend_seasonality_reason']} "
          "(baseline mode used: global)")
    print(f"  event layer gate: {ev_gate['verdict']} ({ev_gate['reason']})")
    if drift["status"] == "measured":
        print(f"  level drift: {drift['drift_per_week']:+.4f} per week (se {drift['drift_se']:.4f}) "
              f"vs binding threshold {drift['binding_threshold']:.4f} over {drift['n_weeks']} weeks; "
              f"binding={drift['binding']}")
    else:
        print(f"  level drift: not measured ({drift['reason']})")
    print(f"  interval calibration: {ic_reason}")
    if emit_series:
        print(f"  series layer ACTIVE: {len(series_to_write)} (cell, series) records emitted")
    else:
        print("  series layer INACTIVE (omitted from JSON)")
    most = sorted(coefficients.values(), key=lambda c: c.coefficient)[:3]
    for c in most:
        print(f"  {c.channel_name}: {c.coefficient:+.4f}  (n={c.n}, ci=[{c.ci_low:+.3f}, {c.ci_high:+.3f}])")


if __name__ == "__main__":
    main()
