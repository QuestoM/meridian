"""Measure the real per-break retention effect, detrended and pooled.

This is the honest, data-driven source of the optimizer's per-break
``impact_coefficient``. It replaces the earlier normalization that anchored
Meridian's ``beta_m`` to a declared magnitude. Instead it measures the effect
directly from the minute-level audience curve, removes the time-of-day trend
that would otherwise confound it, and shrinks thin channel cells toward the
global mean so a handful of breaks cannot dictate a large effect.

The three steps, and why each matters:

  1. Measure. For each real break (from the aired-spots log) take the mean
     audience (TVR) in the minutes just before it starts and just after content
     resumes. Their ratio is the audience actually retained across that break.
     The minute-level :func:`kairos.data.loaders.load_dayparts` series makes this
     a direct measurement, not a proxy.
  2. Detrend. Audience rises into prime time regardless of breaks, so a raw
     before/after ratio in the evening looks falsely good. For each break we
     divide its observed ratio by the typical ratio at that broadcast-minute,
     computed from the channel's average audience curve over the month. What
     remains is the break's own marginal effect, not the day's trend. This is the
     correction that stops the optimizer from "learning" that breaks help in prime.
  3. Pool. With 36 channel cells and uneven counts, a cell seen a handful of
     times is shrunk toward the global mean. This is partial pooling: it refuses
     to trust a large difference drawn from a small sample. The shrinkage strength
     is not a hand-set constant; it is learned from the data with a normal-normal
     hierarchical (empirical-Bayes) model. The between-cell variance tau^2 is
     estimated (DerSimonian-Laird), and each cell is pulled toward the global mean
     in proportion to how noisy it is relative to how much cells genuinely differ.
     When cells are alike (small tau^2) the data pools hard; when they truly differ
     (large tau^2) the data trusts each cell. The credible interval is the proper
     hierarchical posterior, which borrows strength and so is tighter than a thin
     cell's own standard error. When the within-cell spread cannot be estimated
     (every cell a single break), it falls back to a fixed pseudo-count.

The result per channel is a retention delta in the optimizer's units (a change on
the [0, 1] retention multiplier, normally <= 0), with the count and a credible
interval kept for transparency. A cell whose measured effect is non-negative is
reported as zero cost: the data shows no shedding there, and the optimizer cannot
consume a positive per-break retention gain. Pure pandas and numpy, so it imports
and unit-tests without Meridian.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Mapping, Optional

if TYPE_CHECKING:  # pragma: no cover - import only for type hints, avoids a cycle
    from kairos.model.series import SeriesCoefficient

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.data.title_features import canonicalize_series
from kairos.model.prepare import keyed_breaks

logger = logging.getLogger(__name__)

# Window sizes, in minutes, for the before and after audience measurement.
_BEFORE_MINUTES = 3
_AFTER_MINUTES = 3
# Minimum window size in minutes. A break whose clipped before- or after-window
# falls below this is dropped rather than measured on contaminated audience.
# One minute is the coarsest granularity in the daypart series.
_MIN_WINDOW_MINUTES = 1
# Broadcast day starts at 02:00 (daypart timebands run hours 2..25).
_BROADCAST_DAY_START_HOUR = 2
_MINUTES_PER_DAY = 1440
# Fallback partial-pooling strength, used ONLY when the within-cell spread cannot
# be estimated (for example a single break per cell), so the hierarchical model
# has no noise scale to learn from. In that degenerate case a cell is shrunk
# toward the global mean as if it carried this many extra observations at that
# mean. In the normal case the shrinkage strength is learned from the data (see
# :func:`channel_coefficients`), and this constant is not used.
_DEFAULT_SHRINKAGE_K = 20.0
# Below this, the pooled within-cell variance is treated as unavailable and the
# fixed-pseudo-count fallback is used instead of the empirical-Bayes path.
_MIN_POOLED_WITHIN_VAR = 1e-12


@dataclass(frozen=True)
class MeasuredCoefficient:
    """One channel's measured per-break retention delta and its provenance.

    ``coefficient`` is the value the optimizer consumes (a retention delta on the
    [0, 1] multiplier, <= 0). ``raw_delta`` is the shrunk measured delta before the
    non-positive clamp, so a genuine measured gain stays visible. ``n`` is the
    number of breaks measured for the cell; ``ci_low``/``ci_high`` bound the delta.
    """

    channel_name: str
    coefficient: float
    raw_delta: float
    n: int
    ci_low: float
    ci_high: float


def _broadcast_minute(timestamp: pd.Timestamp) -> int:
    """Minutes since the 02:00 broadcast-day start (0..1439).

    A time before 02:00 belongs to the previous broadcast day, so 01:30 maps near
    the end of the cycle rather than the start. This keeps the audience curve a
    smooth function of broadcast time across midnight.
    """
    if timestamp.hour >= _BROADCAST_DAY_START_HOUR:
        start = timestamp.normalize() + pd.Timedelta(hours=_BROADCAST_DAY_START_HOUR)
    else:
        start = timestamp.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=_BROADCAST_DAY_START_HOUR)
    return int((timestamp - start).total_seconds() // 60) % _MINUTES_PER_DAY


def _dayparts_frame(dayparts: pd.DataFrame) -> pd.DataFrame:
    """Add a real wall-clock timestamp and broadcast-minute to the daypart rows."""
    frame = dayparts.dropna(subset=["date", "timeband", "tvr"]).copy()
    parts = frame["timeband"].astype(str).str.split(":", expand=True)
    hours = parts[0].astype(int)
    minutes = parts[1].astype(int)
    frame["ts"] = (
        frame["date"].dt.normalize()
        + pd.to_timedelta(hours, unit="h")
        + pd.to_timedelta(minutes, unit="m")
    )
    frame["mod"] = (hours - _BROADCAST_DAY_START_HOUR) * 60 + minutes
    frame["tvr"] = pd.to_numeric(frame["tvr"], errors="coerce")
    return frame.dropna(subset=["tvr"])


def _minute_lookup(frame: pd.DataFrame) -> dict[tuple[str, pd.Timestamp], float]:
    """Map (channel, minute timestamp) -> TVR for the observed audience."""
    return frame.set_index(["channel", "ts"])["tvr"].to_dict()


def _baseline_levels(frame: pd.DataFrame) -> dict[tuple[str, int], float]:
    """Map (channel, broadcast-minute) -> the month's mean TVR at that minute.

    This is the typical audience curve: the level of audience the channel
    normally holds at each broadcast minute, averaged over every day. The break
    minutes themselves never enter the before/after windows, so this curve carries
    the day's trend without the local break dips.
    """
    grouped = frame.groupby(["channel", "mod"])["tvr"].mean()
    return {(str(ch), int(mod)): float(v) for (ch, mod), v in grouped.items()}


def _window_mean(values: list[Optional[float]]) -> Optional[float]:
    clean = [v for v in values if v is not None and v == v]
    return float(np.mean(clean)) if clean else None


def _neighbour_lookup(
    breaks: pd.DataFrame,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Build a per-channel sorted list of (start, end) break spans (floor minutes).

    Used by :func:`break_effects` to clip each break's measurement windows so
    they never cross an adjacent break's audience, which would contaminate the
    before/after comparison. Only the floor-minute timestamps matter; the
    channel key matches the ``channel`` column (raw airing channel, not the
    engine vocabulary name). Deterministic: no clock or randomness.
    """
    lookup: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for channel, group in breaks.groupby("channel", sort=False):
        spans = [
            (
                pd.Timestamp(row.break_start).floor("min"),
                pd.Timestamp(row.break_end).floor("min"),
            )
            for row in group.itertuples(index=False)
        ]
        spans.sort(key=lambda s: s[0])
        lookup[str(channel)] = spans
    return lookup


def _clipped_offsets(
    window_start_ref: pd.Timestamp,
    offsets: list[int],
    *,
    clip_before: Optional[pd.Timestamp],
    clip_after: Optional[pd.Timestamp],
    min_minutes: int,
) -> Optional[list[pd.Timestamp]]:
    """Return the subset of (window_start_ref + offset) timestamps that lie within bounds.

    ``clip_before`` is the earliest timestamp allowed (exclusive lower bound, e.g.
    the previous break's end). ``clip_after`` is the latest allowed (exclusive upper
    bound, e.g. the next break's start). ``offsets`` are in minutes, signed: negative
    for the before-window, positive for the after-window.

    Returns ``None`` when the surviving timestamps number fewer than ``min_minutes``
    so the caller can drop the break rather than measure on contaminated audience.
    """
    ts_list = [window_start_ref + pd.Timedelta(minutes=o) for o in offsets]
    if clip_before is not None:
        ts_list = [t for t in ts_list if t > clip_before]
    if clip_after is not None:
        ts_list = [t for t in ts_list if t < clip_after]
    if len(ts_list) < min_minutes:
        return None
    return ts_list


def _programme_title_lookup(programmes: pd.DataFrame) -> dict[str, list[tuple]]:
    """Map channel -> sorted (start, end, title) spans for title matching.

    Built once and reused for every break, so the series layer can recover the
    programme Title behind each break without re-running break detection. The
    genre cell apparatus is unaffected; this only adds the title feature.
    """
    frame = programmes[programmes["start_dt"].notna()].copy()
    lookup: dict[str, list[tuple]] = {}
    for channel, group in frame.groupby("Channel", sort=False):
        spans = []
        for row in group.itertuples(index=False):
            start = getattr(row, "start_dt")
            end = getattr(row, "end_dt")
            title = getattr(row, "Title")
            if pd.notna(start) and pd.notna(end):
                spans.append((start, end, "" if title is None or pd.isna(title) else str(title)))
        spans.sort(key=lambda s: s[0])
        lookup[str(channel)] = spans
    return lookup


def _title_for_break(
    lookup: dict[str, list[tuple]], channel: str, start: pd.Timestamp, end: pd.Timestamp
) -> str:
    """Return the Title of the programme whose span contains the break, or ""."""
    for s_start, s_end, title in lookup.get(channel, []):
        if s_start <= start and s_end >= end:
            return title
    return ""


def break_effects(
    spots: pd.DataFrame,
    programmes: pd.DataFrame,
    dayparts: pd.DataFrame,
    classifier: ProgramClassifier,
    *,
    before_minutes: int = _BEFORE_MINUTES,
    after_minutes: int = _AFTER_MINUTES,
    min_window_minutes: int = _MIN_WINDOW_MINUTES,
) -> pd.DataFrame:
    """Measure the detrended retention effect of every real break.

    Returns one row per measurable break with its engine channel name and the
    log effect ``log(observed_ratio) - log(expected_ratio)``: positive means the
    break held more audience than the time-of-day trend predicts, negative means
    it shed more. Breaks whose windows have no positive audience are dropped (no
    fabricated rating).

    Window-boundary clipping (after-window boundary fix)
    -----------------------------------------------------
    When adjacent breaks on the same channel are close together, the fixed
    3-minute window extends into the next (or previous) break's air time,
    contaminating the audience measurement with viewers lost to the neighbour
    rather than the current break. To avoid this, each break's after-window is
    clipped so it never crosses the next break's start minute, and its
    before-window never crosses the previous break's end minute. When the
    clipped window shrinks below ``min_window_minutes`` (default 1 minute),
    the break is dropped entirely rather than measured on contaminated audience.
    Unaffected breaks (gap > window) keep the full window unchanged.
    """
    breaks = keyed_breaks(spots, programmes, classifier)
    columns = [
        "channel_name", "program_type", "break_position", "break_length",
        "observed_ratio", "expected_ratio", "log_effect",
        # The airing channel and break span travel with each measured break so the
        # Stage 3 competitor-context extractor can join features without re-running
        # break detection. Harmless to the pooling, which keys only on channel_name.
        "channel", "break_start", "break_end",
        # The programme Title behind each break, recovered from the programme span,
        # so the optional series layer can canonicalize it. Empty when unmatched.
        "title",
        # The break's ordinal WITHIN its programme (1 = the show's first
        # interruption) and the programme key, so the first-break retention gate
        # can contrast first vs later breaks. NaN/None when the break is in no
        # matched programme. Travels with each effect; never enters the pooling.
        "ordinal", "prog_key",
    ]
    if breaks.empty:
        return pd.DataFrame(columns=columns)

    frame = _dayparts_frame(dayparts)
    observed = _minute_lookup(frame)
    baseline = _baseline_levels(frame)
    titles = _programme_title_lookup(programmes)

    # Build per-channel sorted break spans so we can find each break's
    # neighbours without scanning the full frame on every iteration.
    neighbours = _neighbour_lookup(breaks)

    # Signed-minute offsets for the full windows; clipping may shorten them.
    before_offsets_full = [-(k + 1) for k in range(before_minutes)]
    after_offsets_full = [k + 1 for k in range(after_minutes)]

    rows: list[dict[str, object]] = []
    for row in breaks.itertuples(index=False):
        channel = str(getattr(row, "channel"))
        start = pd.Timestamp(getattr(row, "break_start")).floor("min")
        end = pd.Timestamp(getattr(row, "break_end")).floor("min")

        # Find the closest neighbour boundaries on this channel, in floor-minutes.
        # Binary search on the sorted spans list: cheaper than a pandas merge.
        spans = neighbours.get(channel, [])
        prev_end: Optional[pd.Timestamp] = None
        next_start: Optional[pd.Timestamp] = None
        # Locate the position of this break in the sorted list by its start.
        lo, hi = 0, len(spans)
        while lo < hi:
            mid = (lo + hi) // 2
            if spans[mid][0] < start:
                lo = mid + 1
            else:
                hi = mid
        idx = lo  # index of this break (or the first break starting >= start)
        if idx > 0:
            prev_end = spans[idx - 1][1]
        if idx + 1 < len(spans):
            next_start = spans[idx + 1][0]

        # Clip windows: before-window must stay above prev_end; after-window
        # must stay below next_start. Drop the break when the result is too thin.
        before_ts = _clipped_offsets(
            start, before_offsets_full,
            clip_before=prev_end,
            clip_after=None,
            min_minutes=min_window_minutes,
        )
        if before_ts is None:
            continue

        after_ts = _clipped_offsets(
            end, after_offsets_full,
            clip_before=None,
            clip_after=next_start,
            min_minutes=min_window_minutes,
        )
        if after_ts is None:
            continue

        obs_before = _window_mean([observed.get((channel, t)) for t in before_ts])
        obs_after = _window_mean([observed.get((channel, t)) for t in after_ts])
        base_before = _window_mean([baseline.get((channel, _broadcast_minute(t))) for t in before_ts])
        base_after = _window_mean([baseline.get((channel, _broadcast_minute(t))) for t in after_ts])

        if not obs_before or obs_before <= 0 or obs_after is None or obs_after <= 0:
            continue
        if not base_before or base_before <= 0 or base_after is None or base_after <= 0:
            continue

        observed_ratio = obs_after / obs_before
        expected_ratio = base_after / base_before
        rows.append(
            {
                "channel_name": getattr(row, "channel_name"),
                "program_type": getattr(row, "program_type"),
                "break_position": getattr(row, "break_position"),
                "break_length": getattr(row, "break_length"),
                "observed_ratio": observed_ratio,
                "expected_ratio": expected_ratio,
                "log_effect": float(np.log(observed_ratio) - np.log(expected_ratio)),
                "channel": channel,
                "break_start": start,
                "break_end": end,
                "title": _title_for_break(titles, channel, start, end),
                "ordinal": getattr(row, "ordinal"),
                "prog_key": getattr(row, "prog_key"),
            }
        )

    return pd.DataFrame(rows, columns=columns)


# First-break gate. The first interruption of a programme sheds more audience
# than later breaks; we only ship a multiplier when the contrast is large enough
# and significant enough on the data, otherwise it stays 1.0 (off). These are
# explicit, editable knobs, not hidden constants.
_FIRST_BREAK_MIN_BREAKS = 200          # need this many measured breaks in each arm
_FIRST_BREAK_MIN_P = 0.01              # two-sided p-value must beat this
_FIRST_BREAK_MIN_MULTIPLIER = 1.10     # only ship if the lift is at least this
_FIRST_BREAK_MAX_MULTIPLIER = 2.0      # cap so a thin, noisy cell cannot explode it


def _normal_two_sided_p(t: float) -> float:
    """Two-sided p-value for a t-statistic under the normal approximation."""
    from math import erfc, sqrt
    return float(erfc(abs(t) / sqrt(2.0)))


def first_break_gate(effects: pd.DataFrame) -> dict[str, object]:
    """Measure the first-break retention multiplier with an honest gate.

    ``effects`` must carry the per-break ``log_effect`` (the detrended audience
    hold) plus ``channel``, ``break_start`` and ``break_end`` so each break can be
    assigned its ordinal WITHIN its programme. The first break of a programme (the
    show's first interruption) is contrasted with every later break of the same
    programme; the multiplier is ``delta_first / delta_all`` in retention-delta
    space, anchored on the all-breaks coefficient the optimizer already charges.

    The gate ships a value (> 1.0) ONLY when all hold: at least
    :data:`_FIRST_BREAK_MIN_BREAKS` breaks in each arm, a two-sided p-value below
    :data:`_FIRST_BREAK_MIN_P`, and an implied multiplier of at least
    :data:`_FIRST_BREAK_MIN_MULTIPLIER` (capped at
    :data:`_FIRST_BREAK_MAX_MULTIPLIER`). Otherwise it returns 1.0 (off), so a thin
    or noisy month never invents a cost. The decision and its numbers are returned
    for the JSON metadata so any reader can audit it. Deterministic: no clock, no
    randomness, so the byte-stable JSON tests hold.
    """
    out: dict[str, object] = {
        "first_break_multiplier": 1.0,
        "first_break_active": False,
        "first_break_n_first": 0,
        "first_break_n_later": 0,
        "first_break_mean_first": 0.0,
        "first_break_mean_later": 0.0,
        "first_break_p_value": 1.0,
        "first_break_reason": "no break ordinal recoverable",
    }
    needed = {"log_effect", "ordinal"}
    if effects.empty or not needed.issubset(effects.columns):
        return out

    matched = effects[effects["ordinal"].notna()]
    if matched.empty:
        return out
    # Only programmes with >=2 measured breaks define a first-vs-later contrast.
    counts = matched.groupby("prog_key")["ordinal"].transform("count")
    multi = matched[counts >= 2]
    first = multi[multi["ordinal"] == 1]["log_effect"].to_numpy()
    later = multi[multi["ordinal"] >= 2]["log_effect"].to_numpy()

    out["first_break_n_first"] = int(len(first))
    out["first_break_n_later"] = int(len(later))
    if len(first) < 2 or len(later) < 2:
        out["first_break_reason"] = "too few breaks per arm to measure"
        return out

    mean_first = float(np.mean(first))
    mean_later = float(np.mean(later))
    out["first_break_mean_first"] = mean_first
    out["first_break_mean_later"] = mean_later

    var_first = float(np.var(first, ddof=1))
    var_later = float(np.var(later, ddof=1))
    se = float(np.sqrt(var_first / len(first) + var_later / len(later)))
    t = (mean_first - mean_later) / se if se > 0 else 0.0
    p = _normal_two_sided_p(t)
    out["first_break_p_value"] = p

    mean_all = float(np.mean(multi["log_effect"].to_numpy()))
    delta_all = np.exp(mean_all) - 1.0
    delta_first = np.exp(mean_first) - 1.0
    # The multiplier only makes sense when the all-breaks cost is itself negative
    # (the optimizer charges a retention cost there); a non-negative baseline has
    # no cost to scale, so the lever stays off.
    multiplier = (delta_first / delta_all) if delta_all < 0 else 1.0

    enough = len(first) >= _FIRST_BREAK_MIN_BREAKS and len(later) >= _FIRST_BREAK_MIN_BREAKS
    significant = p < _FIRST_BREAK_MIN_P
    sizable = multiplier >= _FIRST_BREAK_MIN_MULTIPLIER
    if enough and significant and sizable:
        capped = min(_FIRST_BREAK_MAX_MULTIPLIER, multiplier)
        out["first_break_multiplier"] = float(round(capped, 3))
        out["first_break_active"] = True
        out["first_break_reason"] = (
            f"first-break shedding {multiplier:.2f}x the all-breaks cost "
            f"(p={p:.4f}, n_first={len(first)}, n_later={len(later)}); "
            f"shipped multiplier {out['first_break_multiplier']}"
        )
    else:
        bits = []
        if not enough:
            bits.append(f"need >={_FIRST_BREAK_MIN_BREAKS} per arm")
        if not significant:
            bits.append(f"p={p:.4f} not < {_FIRST_BREAK_MIN_P}")
        if not sizable:
            bits.append(f"multiplier {multiplier:.2f} < {_FIRST_BREAK_MIN_MULTIPLIER}")
        out["first_break_reason"] = "; ".join(bits) + "; multiplier left at 1.0 (off)"
    return out


def _cell_stats(effects: pd.DataFrame) -> list[tuple[str, int, float, float]]:
    """Per-cell ``(name, n, mean_log_effect, residual_sum_of_squares)``.

    The residual sum of squares is ``sum((x - mean)^2)`` within the cell (0 for a
    single-break cell). It feeds the pooled within-cell variance.
    """
    stats: list[tuple[str, int, float, float]] = []
    for name, group in effects.groupby("channel_name"):
        logs = group["log_effect"].to_numpy()
        n = int(len(logs))
        mean = float(np.mean(logs))
        rss = float(np.sum((logs - mean) ** 2))
        stats.append((str(name), n, mean, rss))
    return stats


def _pooled_within_variance(stats: list[tuple[str, int, float, float]]) -> float:
    """The pooled within-cell variance s_p^2 of a single break's log effect.

    ``sum(rss_i) / (N - m)`` (total residual degrees of freedom). Returns NaN when
    every cell is a single break (``N == m``), so the caller falls back.
    """
    total_n = sum(n for _, n, _, _ in stats)
    n_cells = len(stats)
    df = total_n - n_cells
    if df <= 0:
        return float("nan")
    return sum(rss for _, _, _, rss in stats) / df


def _use_empirical_bayes(stats: list[tuple[str, int, float, float]], pooled_within: float) -> bool:
    """True when the hierarchical model can learn a noise scale from the data."""
    return len(stats) >= 2 and np.isfinite(pooled_within) and pooled_within > _MIN_POOLED_WITHIN_VAR


def _dersimonian_laird(
    stats: list[tuple[str, int, float, float]], pooled_within: float
) -> tuple[float, float, float]:
    """Estimate ``(tau2, mu, sum_weights)`` for the normal-normal hierarchy.

    Each cell mean has sampling variance ``sigma_i^2 = s_p^2 / n_i`` and inverse-
    variance weight ``w_i = n_i / s_p^2``. ``mu`` is the precision-weighted global
    mean (which equals the overall mean of individual breaks here), and ``tau2`` is
    the DerSimonian-Laird method-of-moments estimate of the between-cell variance,
    floored at 0 so a noise-only spread pools completely.
    """
    n_cells = len(stats)
    w = np.array([n / pooled_within for _, n, _, _ in stats])
    y = np.array([mean for _, _, mean, _ in stats])
    sw = float(np.sum(w))
    mu = float(np.sum(w * y) / sw)
    q = float(np.sum(w * (y - mu) ** 2))
    c = sw - float(np.sum(w ** 2)) / sw
    tau2 = max(0.0, (q - (n_cells - 1)) / c) if c > 0 else 0.0
    return tau2, mu, sw


def channel_coefficients(
    effects: pd.DataFrame,
    *,
    shrinkage_k: float = _DEFAULT_SHRINKAGE_K,
) -> dict[str, MeasuredCoefficient]:
    """Pool the per-break log effects into one delta per channel.

    Each cell's mean log effect is shrunk toward the global mean by a strength the
    data sets, not a hand-picked constant: a normal-normal hierarchical model whose
    between-cell variance tau^2 is estimated by empirical Bayes (DerSimonian-Laird).
    A cell measured with little noise, or in a population where cells genuinely
    differ (large tau^2), is trusted near its own mean; a noisy cell, or one in a
    population where cells are alike (small tau^2), is pulled hard toward the global
    mean. The shrunk log effect is converted to a retention delta ``exp(x) - 1`` and
    clamped non-positive for the optimizer. The 95% interval is the proper
    hierarchical posterior (including the global-mean uncertainty), so it borrows
    strength and never claims a thin cell is tighter than its own data allows.

    When the within-cell spread cannot be estimated (every cell a single break),
    it falls back to pooling toward the global mean with ``shrinkage_k``.
    """
    coefficients: dict[str, MeasuredCoefficient] = {}
    if effects.empty:
        return coefficients

    stats = _cell_stats(effects)
    pooled_within = _pooled_within_variance(stats)

    if _use_empirical_bayes(stats, pooled_within):
        tau2, mu, sw = _dersimonian_laird(stats, pooled_within)
        logger.info(
            "Hierarchical pooling: learned tau^2=%.5g, pooled within-var=%.5g over "
            "%d cells (data-set shrinkage, not a fixed pseudo-count).",
            tau2, pooled_within, len(stats),
        )
        for name, n, mean, _rss in stats:
            sigma2 = pooled_within / n
            denom = sigma2 + tau2
            shrink = sigma2 / denom  # fraction pulled to the global mean
            theta = mu + (1.0 - shrink) * (mean - mu)
            # Posterior variance of the cell effect, including uncertainty in the
            # global mean (the second term), so a fully-pooled cell still carries
            # the grand-mean's own error rather than a zero-width interval.
            post_var = (1.0 - shrink) * sigma2 + (shrink ** 2) / sw
            half = 1.96 * float(np.sqrt(max(0.0, post_var)))
            raw_delta = float(np.exp(theta) - 1.0)
            coefficients[name] = MeasuredCoefficient(
                channel_name=name,
                coefficient=min(0.0, raw_delta),
                raw_delta=raw_delta,
                n=n,
                ci_low=float(np.exp(theta - half) - 1.0),
                ci_high=float(np.exp(theta + half) - 1.0),
            )
        return coefficients

    # Fallback: no within-cell spread to learn from, so pool toward the global
    # mean with the fixed pseudo-count and carry the cell's own standard error.
    logger.info(
        "Hierarchical pooling unavailable (pooled within-var=%s, %d cells); "
        "falling back to fixed pseudo-count k=%.1f.",
        pooled_within, len(stats), shrinkage_k,
    )
    grand_mean = float(effects["log_effect"].mean())
    for name, n, mean, rss in stats:
        shrunk = (n * mean + shrinkage_k * grand_mean) / (n + shrinkage_k)
        raw_delta = float(np.exp(shrunk) - 1.0)
        std = float(np.sqrt(rss / (n - 1))) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0.0
        coefficients[name] = MeasuredCoefficient(
            channel_name=name,
            coefficient=min(0.0, raw_delta),
            raw_delta=raw_delta,
            n=n,
            ci_low=float(np.exp(shrunk - 1.96 * se) - 1.0),
            ci_high=float(np.exp(shrunk + 1.96 * se) - 1.0),
        )
    return coefficients


def between_cell_variance(effects: pd.DataFrame) -> dict[str, object]:
    """Diagnostics for the hierarchical pooling, for audit and the JSON metadata.

    Returns the learned between-cell variance ``tau2``, the ``pooled_within_var``
    (s_p^2), the number of ``n_cells`` pooled, the ``method`` actually used
    ("empirical_bayes" or "fixed_pseudo_count"), and the equivalent learned
    ``pseudo_count`` (``s_p^2 / tau2``, the data's answer to the old hand-set k;
    None when tau2 is 0, meaning the data pools completely). Exposed so a reader
    can see how strongly the data, not a constant, set the shrinkage.
    """
    if effects.empty:
        return {"tau2": 0.0, "pooled_within_var": 0.0, "n_cells": 0,
                "method": "empty", "pseudo_count": None}
    stats = _cell_stats(effects)
    pooled_within = _pooled_within_variance(stats)
    if _use_empirical_bayes(stats, pooled_within):
        tau2, _mu, _sw = _dersimonian_laird(stats, pooled_within)
        pseudo = (pooled_within / tau2) if tau2 > 0 else None
        return {"tau2": tau2, "pooled_within_var": pooled_within, "n_cells": len(stats),
                "method": "empirical_bayes", "pseudo_count": pseudo}
    return {"tau2": 0.0,
            "pooled_within_var": float(pooled_within) if np.isfinite(pooled_within) else 0.0,
            "n_cells": len(stats), "method": "fixed_pseudo_count",
            "pseudo_count": _DEFAULT_SHRINKAGE_K}


def compute_measured_coefficients_with_diagnostics(
    *,
    spots: Optional[pd.DataFrame] = None,
    programmes: Optional[pd.DataFrame] = None,
    dayparts: Optional[pd.DataFrame] = None,
    classifier: Optional[ProgramClassifier] = None,
    shrinkage_k: float = _DEFAULT_SHRINKAGE_K,
    with_series: bool = False,
) -> tuple[dict[str, MeasuredCoefficient], dict[str, object],
           dict[tuple[str, str], SeriesCoefficient]]:
    """Measure the coefficients, diagnostics, and optionally the series layer.

    Measures ``break_effects`` once, then returns the per-cell genre coefficients,
    the :func:`between_cell_variance` diagnostics (learned tau^2, pooled within-
    variance, the equivalent learned pseudo-count), and the per-(cell, series)
    layer when ``with_series`` is on (empty otherwise, so existing callers are
    unchanged). The genre layer is identical whether or not the series layer runs.
    """
    spots = load_spots() if spots is None else spots
    programmes = load_programmes() if programmes is None else programmes
    dayparts = load_dayparts() if dayparts is None else dayparts
    classifier = classifier or ProgramClassifier.from_yaml()
    effects = break_effects(spots, programmes, dayparts, classifier)
    coefficients = channel_coefficients(effects, shrinkage_k=shrinkage_k)
    series: dict[tuple[str, str], "SeriesCoefficient"] = {}
    if with_series:
        # Lazy import: series imports measure, so importing it eagerly would form a
        # cycle. It is only needed when the optional series layer is requested.
        from kairos.model.series import series_coefficients

        series = series_coefficients(effects, shrinkage_k=shrinkage_k)
    return coefficients, between_cell_variance(effects), series


def compute_measured_coefficients(
    *,
    spots: Optional[pd.DataFrame] = None,
    programmes: Optional[pd.DataFrame] = None,
    dayparts: Optional[pd.DataFrame] = None,
    classifier: Optional[ProgramClassifier] = None,
    shrinkage_k: float = _DEFAULT_SHRINKAGE_K,
) -> dict[str, MeasuredCoefficient]:
    """Load the reference data (unless frames are supplied) and measure coefficients."""
    coefficients, _diagnostics, _series = compute_measured_coefficients_with_diagnostics(
        spots=spots, programmes=programmes, dayparts=dayparts,
        classifier=classifier, shrinkage_k=shrinkage_k,
    )
    return coefficients


def write_coefficients_json(
    path: str | Path,
    coefficients: Mapping[str, MeasuredCoefficient],
    *,
    metadata: Optional[Mapping[str, object]] = None,
    series: Optional[Mapping[tuple[str, str], SeriesCoefficient]] = None,
) -> Path:
    """Write the measured coefficients (with provenance) as JSON.

    The file carries a flat ``coefficients`` map (channel name -> delta) that
    :func:`kairos.model.impact.load_impact_model` reads, plus per-channel detail
    and any ``metadata`` (data window, method) for audit. When ``series`` is given,
    an additive ``series`` block is written: a list of per-(cell, series) records
    keyed under each genre cell. The block is purely additive, so a reader that
    ignores it still gets the exact same genre coefficients.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {
        "method": "measured_detrended_pooled",
        "metadata": dict(metadata or {}),
        "coefficients": {name: c.coefficient for name, c in coefficients.items()},
        "detail": {name: asdict(c) for name, c in coefficients.items()},
    }
    if series:
        series_block: dict[str, list[dict[str, object]]] = {}
        for (cell_name, _key), record in series.items():
            series_block.setdefault(cell_name, []).append(asdict(record))
        payload["series"] = series_block
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return target


def read_coefficients_json(path: str | Path) -> dict[str, float]:
    """Read the flat channel -> retention-delta map from a coefficients JSON.

    Returns an empty dict when the file is missing or carries no coefficients, so
    the caller can fall back honestly. Kept for back-compat: it deliberately
    returns ONLY the point coefficient and discards the interval and count. Use
    :func:`read_coefficients_detail` to carry the full posterior to the optimizer.
    """
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read measured coefficients at %s; ignoring.", source)
        return {}
    raw = payload.get("coefficients", {})
    return {str(name): float(value) for name, value in raw.items()}


def read_coefficients_metadata(path: str | Path) -> dict[str, object]:
    """Read the ``metadata`` block from a coefficients JSON, or an empty dict.

    The metadata carries provenance (data window, pooling diagnostics, and the
    freshness fields ``computed_at`` and ``source_fingerprints``). Returns an
    empty dict when the file is missing, unreadable, or carries no metadata, so a
    caller can fall back honestly (an empty metadata yields a freshness status of
    ``unknown``, never a false ``fresh``).
    """
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read measured coefficients at %s; ignoring.", source)
        return {}
    metadata = payload.get("metadata", {})
    return dict(metadata) if isinstance(metadata, dict) else {}


# Confidence-label thresholds. A cell is trusted ("high") only when it carries
# enough breaks AND its credible interval is tight; it is "low" when either the
# count is small or the interval is wide. These are explicit, editable knobs, not
# hidden constants, so the operator-facing label is auditable.
_CONFIDENCE_HIGH_MIN_N = 50
_CONFIDENCE_MEDIUM_MIN_N = 15
_CONFIDENCE_HIGH_MAX_HALFWIDTH = 0.02
_CONFIDENCE_MEDIUM_MAX_HALFWIDTH = 0.05


def confidence_label(n: int, ci_low: float, ci_high: float) -> str:
    """Label a cell's retention estimate ``high``/``medium``/``low``.

    The label is the operator-facing "is the model sure here, or guessing"
    signal. It is ``high`` only when the cell carries at least
    :data:`_CONFIDENCE_HIGH_MIN_N` breaks AND the credible interval's half-width is
    at most :data:`_CONFIDENCE_HIGH_MAX_HALFWIDTH`; ``medium`` for a looser bar;
    ``low`` otherwise (a thin cell or a wide interval). Both n and width matter:
    many breaks with a wide interval are still only ``medium``, and a tight
    interval on a handful of breaks (which the fragile normal SE can produce, see
    docs/model/retention-model.md a.3.2) is held to ``medium`` at best by the n
    floor. A non-finite or inverted interval degrades to ``low``, never a false
    ``high``.
    """
    half_width = abs(float(ci_high) - float(ci_low)) / 2.0
    if not (half_width == half_width):  # NaN guard
        return "low"
    if n >= _CONFIDENCE_HIGH_MIN_N and half_width <= _CONFIDENCE_HIGH_MAX_HALFWIDTH:
        return "high"
    if n >= _CONFIDENCE_MEDIUM_MIN_N and half_width <= _CONFIDENCE_MEDIUM_MAX_HALFWIDTH:
        return "medium"
    return "low"


@dataclass(frozen=True)
class CoefficientDetail:
    """The full per-cell retention estimate carried to the optimizer.

    Unlike :func:`read_coefficients_json`, which returns only the point
    ``coefficient``, this keeps the credible interval, the sample count ``n`` and a
    derived ``confidence`` label so the decision can be uncertainty-aware and the
    operator can see where the model is sure versus guessing.
    """

    coefficient: float
    ci_low: float
    ci_high: float
    n: int
    confidence: str


def read_coefficients_detail(path: str | Path) -> dict[str, CoefficientDetail]:
    """Read the full per-channel detail (coefficient + interval + n + confidence).

    Reads the ``detail`` block that :func:`write_coefficients_json` persists, so
    the optimizer can receive not just the point coefficient but its credible
    interval, sample count and a confidence label. Falls back to the flat
    ``coefficients`` map (with a zero-width interval, ``n`` 0 and ``low``
    confidence) for any channel that has no detail, and returns an empty dict when
    the file is missing or unreadable, so the caller can fall back honestly.
    """
    source = Path(path)
    if not source.exists():
        return {}
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("Could not read measured coefficients at %s; ignoring.", source)
        return {}

    detail = payload.get("detail", {})
    flat = payload.get("coefficients", {})
    out: dict[str, CoefficientDetail] = {}
    for name, raw in detail.items():
        if not isinstance(raw, dict):
            continue
        coefficient = float(raw.get("coefficient", flat.get(name, 0.0)))
        n = int(raw.get("n", 0))
        ci_low = float(raw.get("ci_low", coefficient))
        ci_high = float(raw.get("ci_high", coefficient))
        out[str(name)] = CoefficientDetail(
            coefficient=coefficient,
            ci_low=ci_low,
            ci_high=ci_high,
            n=n,
            confidence=confidence_label(n, ci_low, ci_high),
        )
    # Cells present only in the flat map (no detail block) degrade honestly to a
    # point estimate with no interval and low confidence.
    for name, value in flat.items():
        if str(name) not in out:
            point = float(value)
            out[str(name)] = CoefficientDetail(
                coefficient=point, ci_low=point, ci_high=point, n=0, confidence="low",
            )
    return out
