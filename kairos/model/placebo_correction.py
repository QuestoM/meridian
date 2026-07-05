"""Placebo-drift correction for the measured per-break retention cost.

The causal-identification review (docs/model-validation/causal-identification.md,
sections 1 and 5) measured that the shipped estimator's implicit null, "hold
your level relative to the daily curve", is not the true no-break
counterfactual: at eligible in-show minutes with NO break, the same
measurement arithmetic reads a positive within-show audience drift (matched
placebo mean +0.0151 delta, z = +4.12 vs zero, 6,141 pseudo-breaks over 121
channel-day clusters). Every shipped per-break cost is therefore understated
by the drift of its genre, and the honest causal cost is the measured effect
MINUS the matched no-break drift (review fix 1: pooled -0.0391 -> about
-0.053 under the shipped baseline, about -0.0496 together with the
content-only baseline of fix 2).

This module productionizes that measurement so the coefficient rebuild
(scripts/compute_measured_coefficients.py, --placebo-correction) can run it as
a gated layer. It ports the review's sampling and arithmetic exactly
(scripts/validation/common.py, run_placebo.py, which stay the frozen read-only
record): pseudo-breaks are sampled matched 1:k to the real measured breaks
(same programme, same floor-minute duration, windows fully inside the
programme and clear of every detected break span), measured with the exact
shipped window/detrend/drop arithmetic, then aggregated to per-genre drift
means with cluster-robust standard errors. The rebuild subtracts each break's
genre drift from its raw log effect BEFORE pooling, so the existing
DerSimonian-Laird / empirical-Bayes machinery pools drift-corrected effects
unchanged.

Law 9: deterministic (numpy ``default_rng`` with the review's seed 42, stable
iteration orders, sorted genre keys; the same data yields a byte-identical
correction object) and OFF by default. Activation is an explicit owner
decision at the rebuild, and fix 2 (``break_effects(...,
baseline_content_only=True)``) must ship WITH fix 1: standalone it moves
coefficients AWAY from the causal value (review section 6, ranked fix 2).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.model.measure import (
    _baseline_levels,
    _broadcast_minute,
    _content_only_baseline_levels,
    _dayparts_frame,
    _minute_lookup,
    _window_mean,
)
from kairos.model.prepare import identify_breaks, pricing_class_lookup

# The review's parameters, kept identical so the productionized correction
# reproduces the panel's published numbers on the same data.
PLACEBO_SEED = 42
PSEUDO_PER_BREAK = 3
_BEFORE_MINUTES = 3
_AFTER_MINUTES = 3
_NS_PER_MINUTE = 60_000_000_000


def _to_min(ts: pd.Timestamp) -> int:
    """Epoch minute (floor) of a timestamp."""
    return int(pd.Timestamp(ts).value // _NS_PER_MINUTE)


def _min_to_ts(minute: int) -> pd.Timestamp:
    return pd.Timestamp(minute * _NS_PER_MINUTE)


def _ceil_min(ts: pd.Timestamp) -> int:
    value = int(pd.Timestamp(ts).value)
    return -(-value // _NS_PER_MINUTE)


@dataclass(frozen=True)
class PlaceboCorrection:
    """The measured no-break drift, per genre and pooled, with provenance.

    ``per_genre_drift`` maps program_type -> mean pseudo-break log effect (log
    scale, the units of ``log_effect``); subtracting it from a raw effect
    removes the within-show audience build the shipped null wrongly treats as
    part of the break's cost. ``pooled_drift`` is the mean over all measured
    pseudo-breaks and is the fallback for a genre with no pseudo sample.
    Standard errors are cluster-robust over channel-day clusters (the review's
    bootstrap clustering, in analytic form so the object is deterministic).
    """

    per_genre_drift: Mapping[str, float]
    per_genre_n: Mapping[str, int]
    per_genre_se: Mapping[str, float]
    pooled_drift: float
    n_pseudo: int
    n_clusters: int
    se: float
    seed: int = PLACEBO_SEED
    baseline: str = "shipped"
    method: str = field(default="matched_pseudo_breaks")

    def as_metadata(self) -> dict[str, object]:
        """A plain, JSON-ready dict with sorted genre keys (byte-stable)."""
        return {
            "pooled_drift": self.pooled_drift,
            "per_genre_drift": {k: self.per_genre_drift[k] for k in sorted(self.per_genre_drift)},
            "per_genre_n": {k: self.per_genre_n[k] for k in sorted(self.per_genre_n)},
            "per_genre_se": {k: self.per_genre_se[k] for k in sorted(self.per_genre_se)},
            "n_pseudo": self.n_pseudo,
            "n_clusters": self.n_clusters,
            "se": self.se,
            "seed": self.seed,
            "baseline": self.baseline,
            "method": self.method,
        }


def _break_span_table(spots: pd.DataFrame) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-channel sorted floor-minute (starts, ends) arrays of detected breaks.

    Detected breaks are runs of >= 2 spots, the machinery's own standard (the
    review's primary "matched" design), so pseudo windows keep the same
    distance from commercial airtime that a surviving real measurement keeps.
    """
    detected = identify_breaks(spots)
    table: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for channel, group in detected.groupby("channel", sort=False):
        pairs = sorted(
            (_to_min(pd.Timestamp(s).floor("min")), _to_min(pd.Timestamp(e).floor("min")))
            for s, e in zip(group["break_start"], group["break_end"])
        )
        starts = np.array([p[0] for p in pairs], dtype=np.int64)
        ends = np.array([p[1] for p in pairs], dtype=np.int64)
        table[str(channel)] = (starts, ends)
    return table


def _overlaps_any(
    spans: dict[str, tuple[np.ndarray, np.ndarray]], channel: str, lo: int, hi: int
) -> bool:
    """True when [lo, hi] (inclusive minutes) intersects any span on the channel.

    Spans are sorted and non-overlapping (maximal spot runs), so it suffices
    to check the last span starting at or before ``hi``.
    """
    entry = spans.get(channel)
    if entry is None:
        return False
    starts, ends = entry
    j = bisect_right(starts, hi)
    if j == 0:
        return False
    return bool(ends[j - 1] >= lo)


def _programme_spans(
    programmes: pd.DataFrame, classifier: ProgramClassifier
) -> dict[tuple, tuple[int, int, str]]:
    """Map prog_key -> (start_min_ceil, end_min_floor, channel).

    The keys are the exact ``(channel, day, programme_index)`` tuples
    :func:`kairos.model.prepare.programme_ordinals` assigns to measured
    breaks, so a break's ``prog_key`` looks up its own show's span directly.
    """
    lookup = pricing_class_lookup(programmes, classifier)
    spans: dict[tuple, tuple[int, int, str]] = {}
    for (channel, day), records in lookup.items():
        for prog_idx, record in enumerate(records):
            start, end = record["start_dt"], record["end_dt"]
            if pd.isna(start) or pd.isna(end) or end <= start:
                continue
            spans[(channel, day, prog_idx)] = (
                _ceil_min(start),
                _to_min(pd.Timestamp(end).floor("min")),
                channel,
            )
    return spans


def _eligible_minutes(
    prog_span: tuple[int, int, str],
    dur_min: int,
    break_spans: dict[str, tuple[np.ndarray, np.ndarray]],
) -> list[int]:
    """All pseudo-break start minutes in the programme with clean windows.

    Eligibility, identical to the review (and to what a surviving real
    measurement effectively enforces): the full extent
    [start - 3, start + dur + 3] lies inside the programme span, and the
    extent intersects NO detected break span on the channel. Data-presence and
    positive-audience rules are applied later by :func:`_measure_effect_at`,
    exactly as the machinery applies them.
    """
    p0, p1, channel = prog_span
    lo = p0 + _BEFORE_MINUTES
    hi = p1 - dur_min - _AFTER_MINUTES
    out: list[int] = []
    for m in range(lo, hi + 1):
        if not _overlaps_any(break_spans, channel, m - _BEFORE_MINUTES, m + dur_min + _AFTER_MINUTES):
            out.append(m)
    return out


def _measure_effect_at(
    observed: dict, baseline: dict, channel: str, s_min: int, e_min: int
) -> Optional[float]:
    """The shipped measurement arithmetic at an arbitrary (pseudo) break span.

    Mirrors :func:`kairos.model.measure.break_effects` exactly for an
    unclipped break: 3-minute before window ending at ``s_min - 1``, 3-minute
    after window starting at ``e_min + 1``, the same ``_window_mean``, the
    same baseline curve keyed by ``_broadcast_minute``, and the same
    positive-audience drop rules. Returns the log effect, or None when the
    machinery would drop the break.
    """
    before_ts = [_min_to_ts(s_min - o) for o in range(1, _BEFORE_MINUTES + 1)]
    after_ts = [_min_to_ts(e_min + o) for o in range(1, _AFTER_MINUTES + 1)]
    obs_before = _window_mean([observed.get((channel, t)) for t in before_ts])
    obs_after = _window_mean([observed.get((channel, t)) for t in after_ts])
    base_before = _window_mean(
        [baseline.get((channel, _broadcast_minute(t))) for t in before_ts])
    base_after = _window_mean(
        [baseline.get((channel, _broadcast_minute(t))) for t in after_ts])
    if not obs_before or obs_before <= 0 or obs_after is None or obs_after <= 0:
        return None
    if not base_before or base_before <= 0 or base_after is None or base_after <= 0:
        return None
    return float(np.log(obs_after / obs_before) - np.log(base_after / base_before))


def sample_pseudo_break_effects(
    spots: pd.DataFrame,
    programmes: pd.DataFrame,
    dayparts: pd.DataFrame,
    classifier: ProgramClassifier,
    effects: pd.DataFrame,
    *,
    baseline_content_only: bool = False,
    pseudo_per_break: int = PSEUDO_PER_BREAK,
    seed: int = PLACEBO_SEED,
) -> pd.DataFrame:
    """The matched pseudo-break sample, measured with the shipped arithmetic.

    For each real measured break in ``effects`` that sits inside a matched
    programme, draw up to ``pseudo_per_break`` pseudo-break start minutes
    (same channel, same programme, same floor-minute duration) uniformly from
    the eligible minutes, without replacement within the programme (two real
    breaks of one show never share a pseudo minute), then measure each with
    the exact shipped arithmetic against the same baseline family the real
    effects were measured with (``baseline_content_only`` selects the
    content-only curve of measure.py, matching the review's clean-baseline
    re-measurement). Returns one row per measured pseudo-break with
    ``program_type`` (the SOURCE break's genre), ``log_effect`` and the
    channel-day ``cluster`` label. Deterministic given ``seed``.
    """
    frame = _dayparts_frame(dayparts)
    observed = _minute_lookup(frame)
    baseline = (
        _content_only_baseline_levels(frame, spots)
        if baseline_content_only
        else _baseline_levels(frame)
    )
    break_spans = _break_span_table(spots)
    prog_spans = _programme_spans(programmes, classifier)

    rng = np.random.default_rng(seed)
    taken: dict[tuple, set[int]] = {}
    rows: list[dict[str, object]] = []
    for row in effects.itertuples(index=False):
        prog_key = getattr(row, "prog_key")
        if prog_key is None or (isinstance(prog_key, float) and np.isnan(prog_key)):
            continue
        span = prog_spans.get(prog_key)
        if span is None:
            continue
        s_min = _to_min(getattr(row, "break_start"))
        e_min = _to_min(getattr(row, "break_end"))
        dur = int(e_min - s_min)
        eligible = _eligible_minutes(span, dur, break_spans)
        used = taken.setdefault(prog_key, set())
        eligible = [m for m in eligible if m not in used]
        if not eligible:
            continue
        n_draw = min(pseudo_per_break, len(eligible))
        picks = rng.choice(len(eligible), size=n_draw, replace=False)
        channel = str(getattr(row, "channel"))
        for pick in np.sort(picks):
            m = eligible[int(pick)]
            used.add(m)
            log_effect = _measure_effect_at(observed, baseline, channel, m, m + dur)
            if log_effect is None:
                continue
            ts = _min_to_ts(m)
            rows.append({
                "channel": channel,
                "program_type": str(getattr(row, "program_type")),
                "pseudo_s_min": m,
                "dur_min": dur,
                "log_effect": log_effect,
                "cluster": channel + "|" + ts.strftime("%Y-%m-%d"),
            })
    return pd.DataFrame(
        rows, columns=["channel", "program_type", "pseudo_s_min", "dur_min",
                       "log_effect", "cluster"],
    )


def _cluster_robust_se(values: np.ndarray, clusters: np.ndarray) -> float:
    """Cluster-robust standard error of a mean (clusters move together).

    The analytic form of the review's channel-day cluster bootstrap: the
    variance of the mean is the sum of squared within-cluster residual sums
    over n^2, with the G/(G-1) small-sample factor. Chosen over a bootstrap so
    the correction object is deterministic without a thousand draws.
    """
    n = int(values.size)
    if n == 0:
        return float("nan")
    resid = values - float(values.mean())
    sums = pd.DataFrame({"r": resid, "c": clusters}).groupby("c")["r"].sum().to_numpy()
    g = int(sums.size)
    if g <= 1:
        if n <= 1:
            return float("nan")
        return float(np.std(values, ddof=1) / np.sqrt(n))
    return float(np.sqrt(np.sum(sums ** 2) * g / (g - 1)) / n)


def measure_placebo_drift(
    spots: pd.DataFrame,
    programmes: pd.DataFrame,
    dayparts: pd.DataFrame,
    classifier: ProgramClassifier,
    effects: pd.DataFrame,
    *,
    baseline_content_only: bool = False,
    pseudo_per_break: int = PSEUDO_PER_BREAK,
    seed: int = PLACEBO_SEED,
) -> PlaceboCorrection:
    """Measure the no-break drift the shipped null absorbs, per genre.

    Runs :func:`sample_pseudo_break_effects` and aggregates: per-genre drift
    means keyed by the SOURCE break's ``program_type`` (the genre whose raw
    effects the correction will be subtracted from), the pooled mean, counts,
    and cluster-robust standard errors. Deterministic: the same inputs yield a
    byte-identical :class:`PlaceboCorrection`.
    """
    pseudo = sample_pseudo_break_effects(
        spots, programmes, dayparts, classifier, effects,
        baseline_content_only=baseline_content_only,
        pseudo_per_break=pseudo_per_break, seed=seed,
    )
    baseline_label = "content_only" if baseline_content_only else "shipped"
    if pseudo.empty:
        return PlaceboCorrection(
            per_genre_drift={}, per_genre_n={}, per_genre_se={},
            pooled_drift=0.0, n_pseudo=0, n_clusters=0, se=float("nan"),
            seed=seed, baseline=baseline_label,
        )
    values = pseudo["log_effect"].to_numpy()
    clusters = pseudo["cluster"].to_numpy()
    per_genre_drift: dict[str, float] = {}
    per_genre_n: dict[str, int] = {}
    per_genre_se: dict[str, float] = {}
    for genre in sorted(pseudo["program_type"].unique()):
        sub = pseudo[pseudo["program_type"] == genre]
        per_genre_drift[genre] = float(sub["log_effect"].mean())
        per_genre_n[genre] = int(len(sub))
        per_genre_se[genre] = _cluster_robust_se(
            sub["log_effect"].to_numpy(), sub["cluster"].to_numpy())
    return PlaceboCorrection(
        per_genre_drift=per_genre_drift,
        per_genre_n=per_genre_n,
        per_genre_se=per_genre_se,
        pooled_drift=float(values.mean()),
        n_pseudo=int(len(pseudo)),
        n_clusters=int(pseudo["cluster"].nunique()),
        se=_cluster_robust_se(values, clusters),
        seed=seed,
        baseline=baseline_label,
    )


def apply_placebo_correction(
    effects: pd.DataFrame, correction: PlaceboCorrection
) -> pd.DataFrame:
    """Subtract each break's genre drift from its raw log effect.

    Returns a copy of ``effects`` whose ``log_effect`` column is the
    drift-corrected causal effect (review fix 1: corrected = raw minus the
    matched no-break drift of the break's genre, in log units, applied to
    every individual effect BEFORE pooling). A genre absent from the
    correction falls back to the pooled drift, so no break is silently left
    uncorrected. ``observed_ratio`` and ``expected_ratio`` keep the raw
    measurement for provenance; the pooling consumes only ``log_effect``.
    The input frame is not mutated.
    """
    out = effects.copy()
    if out.empty:
        return out
    drift = out["program_type"].map(
        lambda genre: correction.per_genre_drift.get(str(genre), correction.pooled_drift)
    )
    out["log_effect"] = out["log_effect"] - drift.astype(float)
    return out
