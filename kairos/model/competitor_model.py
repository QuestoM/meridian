"""Estimate how competitive context shifts the per-break retention effect.

Stage 3a of the retention model (see docs/model/retention-model.md). The Stage 2
hierarchical model gives one retention delta per channel cell. But two breaks in
the same cell are not equal: a break opposite a strong rival show, or against a
rival airing the same genre, sheds more because the viewer has somewhere good to
go. This module estimates that competitive sensitivity and uses it to DE-CONFOUND
the per-cell coefficient, so a cell does not look intrinsically worse just because
it happens to air against tougher competition.

How it works, and why it is honest:

  1. Measure the per-break log effects (Stage 1) and attach the competitor-context
     features (:mod:`kairos.model.competitor_features`).
  2. Fit the feature betas by a within-cell (fixed-effects) OLS: regress the
     cell-demeaned log effect on the cell-demeaned features. Demeaning removes each
     cell's own level, so the betas measure the marginal effect of competition
     WITHIN a cell, free of the confound that some cells simply face more
     competition. The training-only feature (rival co-breaking) is included here
     ONLY as a fit-time control, so it cannot bias the forward betas; it is never
     used to adjust a coefficient (see the information boundary below).
  3. De-confound the cell coefficient: subtract the forward betas' contribution
     evaluated against a common reference context, so every cell's reported effect
     is "what this break would shed at the average competitive context". Cells that
     looked bad only because they face strong rivals are pulled toward the typical.
  4. Pool the de-confounded effects with the unchanged Stage 2 hierarchical model.

The information boundary (load-bearing). Only the FORWARD features adjust the
coefficient; the training-only feature is a fit-time control and is stripped from
the adjustment via :func:`kairos.model.competitor_features.assert_forward_only`.
Pure pandas and numpy; no Meridian.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.competitor_features import (
    ALL_FEATURES,
    FORWARD_FEATURES,
    TRAINING_ONLY_FEATURES,
    assert_forward_only,
    attach_competitor_features,
)
from kairos.model.measure import (
    MeasuredCoefficient,
    between_cell_variance,
    break_effects,
    channel_coefficients,
)

logger = logging.getLogger(__name__)

# Below this many usable breaks (rows with finite features and >= 2 cells), the
# regression cannot be estimated reliably, so the betas are reported as empty and
# the coefficients fall back to the plain Stage 2 pooling, unchanged.
_MIN_ROWS_FOR_BETAS = 8


@dataclass(frozen=True)
class CompetitorBeta:
    """One feature's estimated sensitivity, with its provenance.

    ``beta`` is the change in log retention effect per unit of the feature (so a
    negative beta on ``competitor_strength`` means stronger competition sheds more).
    ``se`` is its standard error; ``ci_low``/``ci_high`` the 95% interval. ``role``
    is "forward" (usable to adjust a live decision) or "training_only" (a fit-time
    control that never adjusts a coefficient). ``reference`` is the feature value
    the de-confounding adjusts toward (the data's mean context).
    """

    feature: str
    beta: float
    se: float
    ci_low: float
    ci_high: float
    role: str
    reference: float


def measure_effects_with_competitors(
    *,
    spots: Optional[pd.DataFrame] = None,
    programmes: Optional[pd.DataFrame] = None,
    dayparts: Optional[pd.DataFrame] = None,
    classifier: Optional[ProgramClassifier] = None,
) -> pd.DataFrame:
    """Measure the per-break log effects and attach the competitor-context features.

    Loads the reference data unless frames are supplied. Returns the
    :func:`kairos.model.measure.break_effects` frame extended with the three
    :data:`~kairos.model.competitor_features.ALL_FEATURES` columns.
    """
    spots = load_spots() if spots is None else spots
    programmes = load_programmes() if programmes is None else programmes
    dayparts = load_dayparts() if dayparts is None else dayparts
    classifier = classifier or ProgramClassifier.from_yaml()
    effects = break_effects(spots, programmes, dayparts, classifier)
    if effects.empty:
        for name in ALL_FEATURES:
            effects[name] = pd.Series(dtype=float)
        return effects
    return attach_competitor_features(effects, programmes, dayparts, spots, classifier)


def _within_cell_demean(effects: pd.DataFrame, columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return cell-demeaned ``(X, y)`` for the within (fixed-effects) regression.

    Subtracting each cell's mean from y and from every feature removes the cell
    level, so the slope measures variation WITHIN cells, not across them.
    """
    frame = effects[["channel_name", "log_effect", *columns]].copy()
    grouped = frame.groupby("channel_name")
    y = (frame["log_effect"] - grouped["log_effect"].transform("mean")).to_numpy()
    cols = []
    for name in columns:
        centered = frame[name] - grouped[name].transform("mean")
        cols.append(centered.to_numpy())
    x = np.column_stack(cols) if cols else np.empty((len(frame), 0))
    return x, y


def fit_competitor_betas(effects: pd.DataFrame) -> dict[str, CompetitorBeta]:
    """Fit the within-cell betas for every competitor feature with finite spread.

    Uses a within-cell (fixed-effects) OLS so the betas are free of the confound
    that some cells face more competition. Returns a :class:`CompetitorBeta` per
    feature actually fitted, tagged ``forward`` or ``training_only``. Returns an
    empty dict when there are too few rows, too few cells, or no within-cell feature
    spread to identify any slope (so the caller falls back to plain Stage 2 pooling).
    """
    if effects.empty or len(effects) < _MIN_ROWS_FOR_BETAS:
        return {}
    usable = effects.dropna(subset=["log_effect", *ALL_FEATURES])
    if usable["channel_name"].nunique() < 2 or len(usable) < _MIN_ROWS_FOR_BETAS:
        return {}

    # Keep only features that vary within at least one cell; a feature constant
    # within every cell is wiped out by demeaning and cannot be identified.
    grouped = usable.groupby("channel_name")
    columns = [
        name for name in ALL_FEATURES
        if float(np.nanmax(grouped[name].transform("std").fillna(0.0).to_numpy())) > 1e-12
    ]
    if not columns:
        return {}

    x, y = _within_cell_demean(usable, columns)
    n_cells = usable["channel_name"].nunique()
    dof = len(y) - n_cells - x.shape[1]
    if dof <= 0:
        return {}

    xtx = x.T @ x
    try:
        xtx_inv = np.linalg.inv(xtx)
    except np.linalg.LinAlgError:
        logger.info("Competitor design matrix is singular; skipping beta fit.")
        return {}
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * xtx_inv

    betas: dict[str, CompetitorBeta] = {}
    for idx, name in enumerate(columns):
        b = float(beta[idx])
        se = float(np.sqrt(max(0.0, cov[idx, idx])))
        role = "forward" if name in FORWARD_FEATURES else "training_only"
        reference = float(usable[name].mean())
        betas[name] = CompetitorBeta(
            feature=name, beta=b, se=se,
            ci_low=b - 1.96 * se, ci_high=b + 1.96 * se,
            role=role, reference=reference,
        )
    return betas


def adjust_effects_for_forward_competition(
    effects: pd.DataFrame, betas: dict[str, CompetitorBeta]
) -> pd.DataFrame:
    """Subtract the forward betas' contribution, evaluated against the reference.

    Returns a copy of ``effects`` whose ``log_effect`` is de-confounded: for each
    forward feature, ``beta * (feature - reference)`` is removed, so the residual is
    the effect the break would have at the average competitive context. Enforces the
    information boundary: only :data:`~kairos.model.competitor_features.FORWARD_FEATURES`
    adjust the effect; the training-only beta is never applied.
    """
    forward = {name: cb for name, cb in betas.items() if cb.role == "forward"}
    assert_forward_only(forward.keys())  # belt-and-suspenders: forward path only
    if not forward or effects.empty:
        return effects.copy()
    out = effects.copy()
    adjustment = np.zeros(len(out), dtype=float)
    for name, cb in forward.items():
        adjustment = adjustment + cb.beta * (out[name].to_numpy() - cb.reference)
    out["log_effect"] = out["log_effect"].to_numpy() - adjustment
    return out


def compute_competition_adjusted_coefficients(
    *,
    spots: Optional[pd.DataFrame] = None,
    programmes: Optional[pd.DataFrame] = None,
    dayparts: Optional[pd.DataFrame] = None,
    classifier: Optional[ProgramClassifier] = None,
    shrinkage_k: float = 20.0,
) -> tuple[dict[str, MeasuredCoefficient], dict[str, object]]:
    """Measure, de-confound by competition, and pool, returning coeffs + diagnostics.

    The per-cell coefficients use the SAME contract and JSON shape as Stage 2, so
    nothing downstream changes; they are simply de-confounded from the competitive
    context. The diagnostics carry the fitted betas (forward and training-only,
    each with its posterior) so a reader can see how much, and how certainly,
    competition matters and that the boundary was respected. When the betas cannot
    be estimated, this falls back to the plain Stage 2 coefficients unchanged.
    """
    effects = measure_effects_with_competitors(
        spots=spots, programmes=programmes, dayparts=dayparts, classifier=classifier,
    )
    betas = fit_competitor_betas(effects)
    adjusted = adjust_effects_for_forward_competition(effects, betas)
    coefficients = channel_coefficients(adjusted, shrinkage_k=shrinkage_k)

    diagnostics = dict(between_cell_variance(adjusted))
    diagnostics["competitor_betas"] = {
        name: {
            "beta": cb.beta, "se": cb.se, "ci_low": cb.ci_low, "ci_high": cb.ci_high,
            "role": cb.role, "reference": cb.reference,
        }
        for name, cb in betas.items()
    }
    diagnostics["competition_adjusted"] = bool(
        any(cb.role == "forward" for cb in betas.values())
    )
    diagnostics["forward_features"] = list(FORWARD_FEATURES)
    diagnostics["training_only_features"] = list(TRAINING_ONLY_FEATURES)
    return coefficients, diagnostics
