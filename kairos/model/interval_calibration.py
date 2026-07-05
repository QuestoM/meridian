"""Interval honesty for the measured retention coefficients.

Grounded in the validation panel's measured findings
(docs/model-validation/uncertainty-calibration.md): the shipped per-cell 95%
band undercovers today (true coverage 82.6%, because tau2-hat estimation error
is not propagated; a measured ~1.77x widening is needed, self-healing to
~1.07x at 24x data), tau2-hat is knife-edged (P(=0) = 0.36 when the truth is
the shipped value), and the pooled within-variance is contradicted by a 23x
span of real per-cell variances (67% coverage in the heteroskedastic recovery
scenario).

Three corrections, all applied only through the owner-gated wiring in
scripts/compute_measured_coefficients.py (default OFF, Law 9): a parametric
bootstrap of the DL pipeline (Laird-Louis 1987 style: simulate cell means from
the fitted model B times, re-estimate the hyperparameters each draw with THE
SAME estimators the shipped pipeline uses, and price each cell by the mixture
of conditional posteriors across draws); limma-style moderated per-cell
within-variances (Smyth 2004) inside the DL weights and intervals; and a
per-cell PREDICTIVE band for a single future break (posterior variance plus
within-cell variance, separate keys, never a change to ci_* semantics).

Deterministic given the seed. Pure numpy; scipy is only an optional
normal-CDF fast path, never required.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Mapping, Optional

import numpy as np

from kairos.model.measure import (
    MeasuredCoefficient,
    _cell_stats,
    _dersimonian_laird,
    _pooled_within_variance,
)

if TYPE_CHECKING:  # pragma: no cover - type hint only
    import pandas as pd

# B draws of the full estimator pipeline, fixed seed, so a rebuild reproduces
# run to run; 2000 draws keep the 95% quantile's MC error well under the
# 1.5-2x width effect being corrected.
DEFAULT_BOOTSTRAP_B = 2000
DEFAULT_CALIBRATION_SEED = 20260706
Z95 = 1.96  # the shipped interval multiplier (measure.py hard-codes 1.96)

try:  # pragma: no cover - exercised implicitly wherever scipy is installed
    from scipy.special import ndtr as _ndtr_fast
except ImportError:  # pragma: no cover - fallback path
    _ndtr_fast = None


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    """Vectorized standard normal CDF (scipy fast path, math.erf fallback)."""
    if _ndtr_fast is not None:
        return np.asarray(_ndtr_fast(x), dtype=float)
    flat = np.asarray(x, dtype=float)
    out = np.array([0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in flat.ravel()])
    return out.reshape(flat.shape)


def digamma(x: np.ndarray | float) -> np.ndarray:
    """Vectorized digamma (recurrence + asymptotic; ~1e-10, tested vs scipy)."""
    v = np.asarray(x, dtype=float).copy()
    out = np.zeros_like(v)
    while np.any(v < 6.0):
        mask = v < 6.0
        out[mask] -= 1.0 / v[mask]
        v[mask] += 1.0
    inv2 = 1.0 / (v * v)
    out += (
        np.log(v) - 0.5 / v
        - inv2 * (1.0 / 12.0 - inv2 * (1.0 / 120.0 - inv2 * (1.0 / 252.0 - inv2 / 240.0)))
    )
    return out


def trigamma(x: np.ndarray | float) -> np.ndarray:
    """Vectorized trigamma: upward recurrence to x >= 6 + asymptotic series."""
    v = np.asarray(x, dtype=float).copy()
    out = np.zeros_like(v)
    while np.any(v < 6.0):
        mask = v < 6.0
        out[mask] += 1.0 / (v[mask] * v[mask])
        v[mask] += 1.0
    inv = 1.0 / v
    inv2 = inv * inv
    out += inv * (
        1.0 + 0.5 * inv
        + inv2 * (1.0 / 6.0 - inv2 * (1.0 / 30.0 - inv2 * (1.0 / 42.0 - inv2 / 30.0)))
    )
    return out


def trigamma_inverse(y: float) -> float:
    """Solve trigamma(x) = y, x > 0: bisection on [1/y, 1/y + 1], a valid
    bracket since 1/x < trigamma(x) < 1/(x-1) for x > 1 (trivial for x <= 1).
    """
    if not np.isfinite(y) or y <= 0.0:
        return float("inf")
    lo, hi = 1.0 / y, 1.0 / y + 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if float(trigamma(mid)) > y else (lo, mid)
        if hi - lo <= 1e-12 * max(1.0, lo):
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PriorDf:
    """limma variance prior: s2_c ~ s2_0 F(df_c, d0), moment-estimated.

    ``method``: "moment" (finite prior df), "infinite" (spread fully explained
    by chi-square noise), or "unavailable" (under two usable cells).
    """

    df: float
    s2: float
    n_cells_used: int
    method: str


def estimate_prior_df(dfs: np.ndarray, s2_cells: np.ndarray) -> PriorDf:
    """Estimate the variance prior (Smyth 2004 moment method on log s2_c).

    The spread of e_c = log s2_c - digamma(df_c/2) + log(df_c/2) beyond its
    chi-square part identifies d0 through a trigamma inversion.
    """
    dfs = np.asarray(dfs, dtype=float)
    s2_cells = np.asarray(s2_cells, dtype=float)
    mask = (dfs >= 1.0) & np.isfinite(s2_cells) & (s2_cells > 0.0)
    n_used = int(mask.sum())
    if n_used < 2:
        return PriorDf(df=0.0, s2=float("nan"), n_cells_used=n_used, method="unavailable")
    d = dfs[mask]
    e = np.log(s2_cells[mask]) - digamma(d / 2.0) + np.log(d / 2.0)
    emean = float(np.mean(e))
    evar = float(np.sum((e - emean) ** 2) / (n_used - 1) - np.mean(trigamma(d / 2.0)))
    if evar > 0.0:
        d0 = 2.0 * trigamma_inverse(evar)
        if np.isfinite(d0):
            s2_0 = float(np.exp(emean + float(digamma(d0 / 2.0)) - math.log(d0 / 2.0)))
            return PriorDf(df=float(d0), s2=s2_0, n_cells_used=n_used, method="moment")
    return PriorDf(df=float("inf"), s2=float(np.exp(emean)), n_cells_used=n_used, method="infinite")


def moderate_variances(dfs: np.ndarray, s2_cells: np.ndarray, prior: PriorDf) -> np.ndarray:
    """Moderated variances (d0 s2_0 + df_c s2_c) / (d0 + df_c), elementwise.

    df_c = 0 cells take s2_0 outright; infinite prior df gives every cell
    s2_0; an unavailable prior returns the observed variances unchanged.
    """
    dfs = np.asarray(dfs, dtype=float)
    s2_cells = np.asarray(s2_cells, dtype=float)
    if prior.method == "unavailable":
        return s2_cells.copy()
    if not np.isfinite(prior.df):
        return np.full_like(s2_cells, prior.s2)
    s2_safe = np.where(dfs > 0.0, s2_cells, 0.0)
    return (prior.df * prior.s2 + dfs * s2_safe) / (prior.df + dfs)


def dl_general(means: np.ndarray, var_means: np.ndarray) -> tuple[float, float, float]:
    """General inverse-variance DerSimonian-Laird: (tau2, mu, sum of weights).

    With every ``var_means`` = s_p^2/n_i this reproduces the shipped
    measure._dersimonian_laird (asserted in the tests); with moderated
    variances it drops the homoskedasticity restriction, nothing else.
    """
    w = 1.0 / np.asarray(var_means, dtype=float)
    y = np.asarray(means, dtype=float)
    sw = float(np.sum(w))
    mu = float(np.sum(w * y) / sw)
    q = float(np.sum(w * (y - mu) ** 2))
    c = sw - float(np.sum(w**2)) / sw
    tau2 = max(0.0, (q - (len(y) - 1)) / c) if c > 0 else 0.0
    return tau2, mu, sw


@dataclass(frozen=True)
class _Fit:
    """One hyperparameter fit: DL estimates plus its variance model."""

    tau2: float
    mu: float
    sw: float
    var_means: np.ndarray  # sampling variance of each cell mean
    s2_cell: np.ndarray  # single-break within-cell variance, per cell
    s2_pooled: float
    prior: PriorDf


def _fit(ns: np.ndarray, ybar: np.ndarray, rss: np.ndarray, moderated: bool) -> _Fit:
    """Fit the hierarchy on sufficient statistics, naive or moderated.

    The naive path calls the shipped estimators (numbers bit-identical to a
    channel_coefficients run); the moderated path swaps limma-moderated
    per-cell variances into the same DL machinery. The prior-df diagnostic is
    computed in both modes; it is APPLIED only when ``moderated`` and estimable.
    """
    stats = [("", int(n), float(m), float(r)) for n, m, r in zip(ns, ybar, rss)]
    s2_pooled = _pooled_within_variance(stats)
    if not (len(stats) >= 2 and np.isfinite(s2_pooled) and s2_pooled > 1e-12):
        raise ValueError(
            "interval calibration unavailable: the pooled within-cell variance "
            "cannot be estimated (every cell a single break), so there is no "
            "noise scale to bootstrap from"
        )
    dfs = ns.astype(float) - 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        s2_cells = np.where(dfs > 0.0, rss / np.maximum(dfs, 1.0), np.nan)
    prior = estimate_prior_df(dfs, s2_cells)
    if moderated and prior.method != "unavailable":
        s2_cell = moderate_variances(dfs, s2_cells, prior)
        var_means = s2_cell / ns
        tau2, mu, sw = dl_general(ybar, var_means)
    else:
        tau2, mu, sw = _dersimonian_laird(stats, s2_pooled)
        var_means = s2_pooled / ns
        s2_cell = np.full(len(ns), s2_pooled)
    return _Fit(tau2=tau2, mu=mu, sw=sw, var_means=var_means,
                s2_cell=s2_cell, s2_pooled=float(s2_pooled), prior=prior)


def _posterior(fit: _Fit, ybar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The shipped normal-normal posterior per cell: (theta, posterior variance)."""
    shrink = fit.var_means / (fit.var_means + fit.tau2)
    theta = fit.mu + (1.0 - shrink) * (ybar - fit.mu)
    post_var = (1.0 - shrink) * fit.var_means + (shrink**2) / fit.sw
    return theta, post_var


def _mixture_quantile(
    mix_mean: np.ndarray, mix_sd: np.ndarray, prob: float,
    extra_var: np.ndarray | float = 0.0,
) -> np.ndarray:
    """Per-cell quantile of the equal-weight normal mixture, by bisection.

    ``extra_var`` (scalar or per-cell) is added to every component's variance
    (the coverage studies add a held-out target's own sampling noise there).
    Zero-sd components (a tau2_b = 0 draw) enter the CDF as step functions.
    """
    extra = np.asarray(extra_var, dtype=float)
    sd = np.sqrt(mix_sd**2 + (extra[:, None] if extra.ndim == 1 else extra))
    lo = np.min(mix_mean - 8.0 * sd, axis=1) - 1e-9
    hi = np.max(mix_mean + 8.0 * sd, axis=1) + 1e-9
    positive = sd > 0.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        z = (mid[:, None] - mix_mean) / np.where(positive, sd, 1.0)
        comp = np.where(positive, _normal_cdf(z), (mid[:, None] >= mix_mean).astype(float))
        f = comp.mean(axis=1)
        take_hi = f >= prob
        hi = np.where(take_hi, mid, hi)
        lo = np.where(take_hi, lo, mid)
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class CalibrationResult:
    """The fitted hierarchy plus the bootstrap mixture that calibrates it.

    All in log-effect space (delta-space bounds via expm1 at the edge).
    ``mix_mean``/``mix_sd`` are (cells x B): each cell's conditional posterior
    under each bootstrap re-fit. ``moderated`` = actually APPLIED.
    """

    names: tuple[str, ...]
    ns: np.ndarray
    ybar: np.ndarray
    theta: np.ndarray
    post_sd_plugin: np.ndarray
    mix_mean: np.ndarray
    mix_sd: np.ndarray
    tau2: float
    mu: float
    s2_pooled: float
    s2_cell: np.ndarray
    prior: PriorDf
    moderated: bool
    bootstrap_b: int
    seed: int

    def interval(
        self, level: float = 0.95, extra_var: np.ndarray | float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calibrated central interval (log space) at ``level`` per cell."""
        alpha = (1.0 - level) / 2.0
        lo = _mixture_quantile(self.mix_mean, self.mix_sd, alpha, extra_var)
        hi = _mixture_quantile(self.mix_mean, self.mix_sd, 1.0 - alpha, extra_var)
        return lo, hi

    def plugin_interval(self, level_z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
        """The plug-in (naive) band around theta, for comparison and OFF mode."""
        half = level_z * self.post_sd_plugin
        return self.theta - half, self.theta + half

    def mixture_variance(self) -> np.ndarray:
        """Total variance of the mixture: mean component var + var of centers."""
        return np.mean(self.mix_sd**2, axis=1) + np.var(self.mix_mean, axis=1)

    def width_factor(self, level: float = 0.95) -> float:
        """Measured mean widening of the calibrated band vs the plug-in band."""
        lo, hi = self.interval(level)
        plugin_half = Z95 * self.post_sd_plugin
        ok = plugin_half > 0.0
        return float(np.mean((hi[ok] - lo[ok]) / (2.0 * plugin_half[ok])))

    def predictive_interval(self, z: float = Z95) -> tuple[np.ndarray, np.ndarray]:
        """Band for ONE future break: theta +/- z sqrt(posterior + within var).

        Measured by the panel at 0.938 coverage at nominal 95 on held-out
        breaks; beyond 95% the within-cell tails are non-normal (excess
        kurtosis +7.9) and z-scores are not to be trusted.
        """
        half = z * np.sqrt(self.mixture_variance() + self.s2_cell)
        return self.theta - half, self.theta + half


def calibrate_intervals(
    effects: "pd.DataFrame" = None,
    *,
    stats: Optional[tuple[tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]] = None,
    bootstrap_b: int = DEFAULT_BOOTSTRAP_B,
    seed: int = DEFAULT_CALIBRATION_SEED,
    moderated: bool = False,
) -> CalibrationResult:
    """Run the seeded parametric bootstrap of the DL pipeline.

    Per draw: simulate cell effects from N(mu-hat, tau2-hat), cell means from
    their sampling noise and residual sums of squares from the fitted variance
    model; re-estimate the FULL pipeline on the simulated statistics; form the
    conditional posterior of each REAL cell under the re-estimated
    hyperparameters. The spread of those posteriors across draws is the
    hyperparameter-estimation noise the plug-in interval omits. ``stats``
    (names, ns, ybar, rss) can replace ``effects``.
    """
    if stats is None:
        cell_stats = _cell_stats(effects)
        stats = (
            tuple(s[0] for s in cell_stats),
            [s[1] for s in cell_stats], [s[2] for s in cell_stats],
            [s[3] for s in cell_stats],
        )
    names, ns, ybar, rss = stats
    ns = np.asarray(ns, dtype=int)
    ybar = np.asarray(ybar, dtype=float)
    rss = np.asarray(rss, dtype=float)
    if bootstrap_b < 2:
        raise ValueError("bootstrap_b must be at least 2")

    fit = _fit(ns, ybar, rss, moderated)
    # Honest flag: moderation is APPLIED only when the variance prior could be
    # estimated; _fit falls back to the naive variances otherwise, and the
    # metadata must say what actually happened, not what was requested.
    moderated_applied = bool(moderated and fit.prior.method != "unavailable")
    theta, post_var = _posterior(fit, ybar)
    m = len(ns)
    dfs = ns.astype(float) - 1.0
    df_mask = dfs > 0.0

    rng = np.random.default_rng(seed)
    mix_mean = np.empty((m, bootstrap_b))
    mix_var = np.empty((m, bootstrap_b))
    sd_theta = math.sqrt(fit.tau2) if fit.tau2 > 0 else 0.0
    sd_ybar = np.sqrt(fit.var_means)
    for b in range(bootstrap_b):
        theta_star = fit.mu + rng.normal(size=m) * sd_theta
        ybar_star = theta_star + rng.normal(size=m) * sd_ybar
        rss_star = np.zeros(m)
        if df_mask.any():
            rss_star[df_mask] = rng.chisquare(dfs[df_mask]) * fit.s2_cell[df_mask]
        fit_b = _fit(ns, ybar_star, rss_star, moderated_applied)
        shrink_b = fit_b.var_means / (fit_b.var_means + fit_b.tau2)
        mix_mean[:, b] = fit_b.mu + (1.0 - shrink_b) * (ybar - fit_b.mu)
        mix_var[:, b] = (1.0 - shrink_b) * fit_b.var_means

    return CalibrationResult(
        names=tuple(names), ns=ns, ybar=ybar, theta=theta,
        post_sd_plugin=np.sqrt(np.maximum(0.0, post_var)),
        mix_mean=mix_mean, mix_sd=np.sqrt(np.maximum(0.0, mix_var)),
        tau2=fit.tau2, mu=fit.mu, s2_pooled=fit.s2_pooled, s2_cell=fit.s2_cell,
        prior=fit.prior, moderated=moderated_applied,
        bootstrap_b=int(bootstrap_b), seed=int(seed),
    )


def apply_calibrated_intervals(
    coefficients: Mapping[str, MeasuredCoefficient],
    result: CalibrationResult,
    *,
    level: float = 0.95,
) -> dict[str, MeasuredCoefficient]:
    """Replace ONLY ci_low/ci_high with the calibrated quantiles.

    ``coefficient`` and ``raw_delta`` carry over bit-for-bit, so calibration
    can never move a revenue decision through the point estimate. Cells the
    calibration does not know are returned unchanged.
    """
    lo, hi = result.interval(level)
    by_name = {name: i for i, name in enumerate(result.names)}
    out: dict[str, MeasuredCoefficient] = {}
    for name, coeff in coefficients.items():
        i = by_name.get(name)
        if i is None:
            out[name] = coeff
            continue
        out[name] = replace(
            coeff, ci_low=float(np.expm1(lo[i])), ci_high=float(np.expm1(hi[i]))
        )
    return out


def coefficient_map(
    result: CalibrationResult, *, level: float = 0.95, plugin: bool = False
) -> dict[str, MeasuredCoefficient]:
    """Build a full coefficient map from the calibration's own fit (moderated
    mode: the DL weights, and so the points, legitimately change; the shipped
    non-positive clamp is applied). ``plugin`` = band without the bootstrap.
    """
    if plugin:
        lo, hi = result.plugin_interval()
    else:
        lo, hi = result.interval(level)
    out: dict[str, MeasuredCoefficient] = {}
    for i, name in enumerate(result.names):
        raw_delta = float(np.expm1(result.theta[i]))
        out[name] = MeasuredCoefficient(
            channel_name=name, coefficient=min(0.0, raw_delta), raw_delta=raw_delta,
            n=int(result.ns[i]),
            ci_low=float(np.expm1(lo[i])), ci_high=float(np.expm1(hi[i])),
        )
    return out


def predictive_bands(
    result: CalibrationResult, *, z: float = Z95
) -> dict[str, tuple[float, float]]:
    """Per-cell single-break predictive band in retention-delta space."""
    lo, hi = result.predictive_interval(z)
    return {
        name: (float(np.expm1(lo[i])), float(np.expm1(hi[i])))
        for i, name in enumerate(result.names)
    }
