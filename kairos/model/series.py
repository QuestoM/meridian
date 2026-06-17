"""The optional series-aware retention layer (genre -> series -> episode pooling).

The genre cells in :mod:`kairos.model.measure` key on (program_type x position x
length). The programme TITLE carries extra signal the cell throws away: episodes
of one series should pool, and shows sharing a host or theme are closer to each
other than to an unrelated programme. This module adds that as an ADDITIVE layer
on top of the genre cells, running IN PARALLEL: the genre effect still drives the
base coefficient, and the series is one more feature shrunk toward the genre-cell
mean with the SAME empirical-Bayes (DerSimonian-Laird) machinery the genre layer
uses. It changes nothing about the genre coefficients.

Kept separate from :mod:`kairos.model.measure` for file-size discipline; the
public names (:class:`SeriesCoefficient`, :func:`series_coefficients`,
:func:`read_series_coefficients`) are re-exported from measure for stability.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kairos.data.title_features import canonicalize_series
from kairos.model.measure import (
    CoefficientDetail,
    _dersimonian_laird,
    _pooled_within_variance,
    _use_empirical_bayes,
    confidence_label,
)

logger = logging.getLogger(__name__)

# Default partial-pooling strength, mirroring measure._DEFAULT_SHRINKAGE_K. Used
# only when the within-series spread cannot be estimated (a single airing per
# series), so the series is pooled toward its genre cell with this pseudo-count.
_DEFAULT_SHRINKAGE_K = 20.0


@dataclass(frozen=True)
class SeriesCoefficient:
    """One (genre cell, series) retention delta, shrunk toward the cell mean.

    The series layer runs IN PARALLEL with the genre cell: the genre effect still
    drives the base coefficient, and this is one more feature shrunk toward the
    genre-cell mean (genre -> series -> episode pooling). ``channel_name`` is the
    engine genre cell the series sits inside, ``series_key`` is the canonical key
    from :func:`kairos.data.title_features.canonicalize_series`, ``coefficient`` is
    the non-positive delta the optimizer would consume for a break in this series,
    and ``raw_delta`` keeps a genuine measured gain visible before the clamp.
    """

    channel_name: str
    series_key: str
    coefficient: float
    raw_delta: float
    n: int
    ci_low: float
    ci_high: float


def _shrink_series_within_cell(
    cell_name: str,
    series_stats: list[tuple[str, int, float, float]],
    cell_mean: float,
    shrinkage_k: float,
) -> dict[tuple[str, str], SeriesCoefficient]:
    """Shrink each series mean toward its genre-cell mean and clamp non-positive.

    Reuses the same empirical-Bayes (DerSimonian-Laird) shrinkage the genre cells
    use, but the target is the genre-cell mean rather than the global mean, so a
    series is pulled toward its genre when it is thin or noisy and trusted near its
    own mean when it is rich and the series within the cell genuinely differ. Falls
    back to the fixed pseudo-count toward the cell mean when no within-series spread
    is available.
    """
    out: dict[tuple[str, str], SeriesCoefficient] = {}
    pooled_within = _pooled_within_variance(series_stats)
    use_eb = _use_empirical_bayes(series_stats, pooled_within)
    tau2 = sw = 0.0
    if use_eb:
        tau2, _mu, sw = _dersimonian_laird(series_stats, pooled_within)
    for key, n, mean, _rss in series_stats:
        if use_eb:
            sigma2 = pooled_within / n
            shrink = sigma2 / (sigma2 + tau2)
            theta = cell_mean + (1.0 - shrink) * (mean - cell_mean)
            post_var = (1.0 - shrink) * sigma2 + (shrink ** 2) / sw
            half = 1.96 * float(np.sqrt(max(0.0, post_var)))
        else:
            theta = (n * mean + shrinkage_k * cell_mean) / (n + shrinkage_k)
            half = 0.0
        raw_delta = float(np.exp(theta) - 1.0)
        out[(cell_name, key)] = SeriesCoefficient(
            channel_name=cell_name,
            series_key=key,
            coefficient=min(0.0, raw_delta),
            raw_delta=raw_delta,
            n=int(n),
            ci_low=float(np.exp(theta - half) - 1.0),
            ci_high=float(np.exp(theta + half) - 1.0),
        )
    return out


def series_coefficients(
    effects: pd.DataFrame,
    *,
    shrinkage_k: float = _DEFAULT_SHRINKAGE_K,
) -> dict[tuple[str, str], SeriesCoefficient]:
    """Per-(genre cell, series) retention deltas shrunk toward the genre-cell mean.

    For each existing genre cell (program_type x position x length), the breaks are
    grouped by the canonical series of their programme Title, and each series mean
    is shrunk toward that genre cell's mean with the same hierarchical pooling the
    genre layer uses. This is genre -> series -> episode pooling: episodes of one
    series pool together, and a thin series is pulled toward its genre rather than
    trusted on a handful of airings. Requires a ``title`` column on ``effects``
    (added by :func:`kairos.model.measure.break_effects`); titles that canonicalize
    to an empty key are skipped, so an unmatched break contributes only to its
    genre cell. The genre coefficients are unchanged; this is an additive layer.
    """
    out: dict[tuple[str, str], SeriesCoefficient] = {}
    if effects.empty or "title" not in effects.columns:
        return out
    work = effects.copy()
    work["series_key"] = work["title"].map(canonicalize_series)
    work = work[work["series_key"].astype(bool)]
    if work.empty:
        return out

    for cell_name, cell_group in work.groupby("channel_name"):
        cell_mean = float(cell_group["log_effect"].mean())
        series_stats: list[tuple[str, int, float, float]] = []
        for key, series_group in cell_group.groupby("series_key"):
            logs = series_group["log_effect"].to_numpy()
            n = int(len(logs))
            mean = float(np.mean(logs))
            rss = float(np.sum((logs - mean) ** 2))
            series_stats.append((str(key), n, mean, rss))
        out.update(
            _shrink_series_within_cell(str(cell_name), series_stats, cell_mean, shrinkage_k)
        )
    return out


def read_series_coefficients(path: str | Path) -> dict[tuple[str, str], CoefficientDetail]:
    """Read the additive ``series`` block as ``(cell, series_key) -> detail``.

    Returns an empty dict when the file is missing, unreadable, or carries no
    series block (the back-compat case), so a reader that predates the series
    layer is unaffected. Each record is keyed by its genre cell and canonical
    series key, with the same confidence label the genre cells carry.
    """
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read measured coefficients at %s; ignoring.", source)
        return {}

    block = payload.get("series", {})
    out: dict[tuple[str, str], CoefficientDetail] = {}
    if not isinstance(block, dict):
        return out
    for cell_name, records in block.items():
        if not isinstance(records, list):
            continue
        for raw in records:
            if not isinstance(raw, dict):
                continue
            key = str(raw.get("series_key", ""))
            if not key:
                continue
            coefficient = float(raw.get("coefficient", 0.0))
            n = int(raw.get("n", 0))
            ci_low = float(raw.get("ci_low", coefficient))
            ci_high = float(raw.get("ci_high", coefficient))
            out[(str(cell_name), key)] = CoefficientDetail(
                coefficient=coefficient,
                ci_low=ci_low,
                ci_high=ci_high,
                n=n,
                confidence=confidence_label(n, ci_low, ci_high),
            )
    return out
