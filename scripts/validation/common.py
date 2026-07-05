"""Shared, deterministic machinery for the causal-identification review.

Read-only with respect to the product code: this module imports
``kairos.model.measure`` / ``kairos.model.prepare`` and reuses their private
helpers so every placebo / selection / inference computation runs the exact
measurement arithmetic under review (same window means, same broadcast-minute
detrend, same drop rules). It never edits product source and never writes
outside ``docs/model-validation``.

Everything is seeded (numpy ``default_rng(42)``) and iteration orders are
deterministic, so every script that imports this module reproduces its output
byte-for-byte on the same data.
"""

from __future__ import annotations

import sys
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from kairos.data.classifier import ProgramClassifier  # noqa: E402
from kairos.data.loaders import load_dayparts, load_programmes, load_spots  # noqa: E402
from kairos.model.measure import (  # noqa: E402
    _baseline_levels,
    _broadcast_minute,
    _dayparts_frame,
    _minute_lookup,
    _window_mean,
    break_effects,
)
from kairos.model.prepare import (  # noqa: E402
    identify_breaks,
    pricing_class_lookup,
    position_bucket,
)

SEED = 42
BEFORE_MINUTES = 3
AFTER_MINUTES = 3
_NS_PER_MIN = 60_000_000_000

DOC_DIR = REPO / "docs" / "model-validation"
DOC_PATH = DOC_DIR / "causal-identification.md"

# The shipped pooled effect under review (delta on the retention multiplier),
# from models/tv_break_coefficients.json / docs/model-card.md.
SHIPPED_POOLED_DELTA = -0.0391


def to_min(ts: pd.Timestamp) -> int:
    """Epoch minute (floor) of a timestamp."""
    return int(pd.Timestamp(ts).value // _NS_PER_MIN)


def min_to_ts(minute: int) -> pd.Timestamp:
    return pd.Timestamp(minute * _NS_PER_MIN)


def ceil_min(ts: pd.Timestamp) -> int:
    value = int(pd.Timestamp(ts).value)
    return -(-value // _NS_PER_MIN)


@dataclass
class Bundle:
    """Everything the validation scripts need, loaded once."""

    spots: pd.DataFrame
    programmes: pd.DataFrame
    dayparts: pd.DataFrame
    classifier: ProgramClassifier
    effects: pd.DataFrame  # real measured per-break effects (the artifact's basis)
    observed: dict  # (channel, minute ts) -> TVR
    baseline: dict  # (channel, broadcast minute) -> month-mean TVR
    break_spans: dict  # channel -> (starts_min asc np.ndarray, ends_min np.ndarray)
    ad_spans: dict  # same, from EVERY ad-air run (min_spots=1)
    prog_spans: dict  # prog_key -> (start_min_ceil, end_min_floor, channel, pricing_class)
    all_prog_records: list  # [(prog_key, channel, start_min_ceil, end_min_floor, pricing_class)]


def _span_table(breaks: pd.DataFrame) -> dict:
    """Per-channel sorted floor-minute (starts, ends) arrays for detected spans."""
    table: dict = {}
    for channel, group in breaks.groupby("channel", sort=False):
        starts = np.array(sorted(to_min(pd.Timestamp(t).floor("min")) for t in group["break_start"]))
        ends_by_start = sorted(
            (to_min(pd.Timestamp(s).floor("min")), to_min(pd.Timestamp(e).floor("min")))
            for s, e in zip(group["break_start"], group["break_end"])
        )
        ends = np.array([e for _s, e in ends_by_start])
        table[str(channel)] = (starts, ends)
    return table


def overlaps_any(spans: dict, channel: str, lo: int, hi: int) -> bool:
    """True when [lo, hi] (inclusive minutes) intersects any span on the channel.

    Spans are sorted and non-overlapping (maximal spot runs), so it suffices to
    check the last span starting at or before ``hi``.
    """
    entry = spans.get(channel)
    if entry is None:
        return False
    starts, ends = entry
    j = bisect_right(starts, hi)
    if j == 0:
        return False
    return bool(ends[j - 1] >= lo)


def load_bundle(verbose: bool = True) -> Bundle:
    spots = load_spots()
    programmes = load_programmes()
    dayparts = load_dayparts()
    classifier = ProgramClassifier.from_yaml()

    effects = break_effects(spots, programmes, dayparts, classifier)
    effects = effects.copy()
    effects["s_min"] = [to_min(t) for t in effects["break_start"]]
    effects["e_min"] = [to_min(t) for t in effects["break_end"]]
    effects["dur_min"] = effects["e_min"] - effects["s_min"]
    effects["cal_day"] = pd.to_datetime(effects["break_start"]).dt.strftime("%Y-%m-%d")
    effects["cluster"] = effects["channel"].astype(str) + "|" + effects["cal_day"]

    frame = _dayparts_frame(dayparts)
    observed = _minute_lookup(frame)
    baseline = _baseline_levels(frame)

    detected = identify_breaks(spots)  # >= 2 spots: the machinery's clip standard
    ad_runs = identify_breaks(spots, min_spots=1)  # every ad-air run (strict variant)
    break_spans = _span_table(detected)
    ad_spans = _span_table(ad_runs)

    lookup = pricing_class_lookup(programmes, classifier)
    prog_spans: dict = {}
    all_prog_records: list = []
    for (channel, day), records in lookup.items():
        for prog_idx, record in enumerate(records):
            start, end = record["start_dt"], record["end_dt"]
            if pd.isna(start) or pd.isna(end) or end <= start:
                continue
            key = (channel, day, prog_idx)
            span = (ceil_min(start), to_min(pd.Timestamp(end).floor("min")), channel,
                    record["pricing_class"])
            prog_spans[key] = span
            all_prog_records.append((key, channel, span[0], span[1], record["pricing_class"]))
    all_prog_records.sort(key=lambda r: (r[1], r[2], r[3]))

    if verbose:
        print(f"[bundle] spots={len(spots)} programmes={len(programmes)} "
              f"dayparts={len(dayparts)} effects={len(effects)} "
              f"detected_breaks={len(detected)} ad_runs={len(ad_runs)} "
              f"prog_spans={len(prog_spans)}")
    return Bundle(
        spots=spots, programmes=programmes, dayparts=dayparts, classifier=classifier,
        effects=effects, observed=observed, baseline=baseline,
        break_spans=break_spans, ad_spans=ad_spans,
        prog_spans=prog_spans, all_prog_records=all_prog_records,
    )


def measure_effect_at(bundle: Bundle, channel: str, s_min: int, e_min: int) -> Optional[dict]:
    """Run the shipped measurement arithmetic at an arbitrary (pseudo) break span.

    Mirrors ``kairos.model.measure.break_effects`` exactly for an unclipped
    break: 3-minute before window ending at ``s_min - 1``, 3-minute after
    window starting at ``e_min + 1``, window means via the same
    ``_window_mean``, detrend via the same ``_baseline_levels`` curve and
    ``_broadcast_minute`` mapping, and the same positive-audience drop rules.
    Returns None when the machinery would drop the break.
    """
    before_ts = [min_to_ts(s_min - o) for o in range(1, BEFORE_MINUTES + 1)]
    after_ts = [min_to_ts(e_min + o) for o in range(1, AFTER_MINUTES + 1)]
    obs_before = _window_mean([bundle.observed.get((channel, t)) for t in before_ts])
    obs_after = _window_mean([bundle.observed.get((channel, t)) for t in after_ts])
    base_before = _window_mean(
        [bundle.baseline.get((channel, _broadcast_minute(t))) for t in before_ts])
    base_after = _window_mean(
        [bundle.baseline.get((channel, _broadcast_minute(t))) for t in after_ts])
    if not obs_before or obs_before <= 0 or obs_after is None or obs_after <= 0:
        return None
    if not base_before or base_before <= 0 or base_after is None or base_after <= 0:
        return None
    observed_ratio = obs_after / obs_before
    expected_ratio = base_after / base_before
    return {
        "observed_ratio": observed_ratio,
        "expected_ratio": expected_ratio,
        "log_effect": float(np.log(observed_ratio) - np.log(expected_ratio)),
        "obs_before": obs_before, "obs_after": obs_after,
        "base_before": base_before, "base_after": base_after,
        "excess_before": float(np.log(obs_before) - np.log(base_before)),
        "excess_after": float(np.log(obs_after) - np.log(base_after)),
    }


def eligible_minutes_for(bundle: Bundle, prog_key, dur_min: int, *,
                         strict: bool = False) -> list:
    """All pseudo-break start minutes inside the programme with clean windows.

    Eligibility (the same rules a real measurement effectively enforces):
      * the full extent [start - 3, start + dur + 3] lies inside the programme
        span, so windows never cross a content junction;
      * the extent intersects NO detected commercial break span on the channel
        (``strict=True`` additionally excludes every single-spot ad run), the
        distance-from-real-breaks rule;
      * data-presence / positive-audience rules are applied later by
        :func:`measure_effect_at`, exactly as the machinery applies them.
    """
    span = bundle.prog_spans.get(prog_key)
    if span is None:
        return []
    p0, p1, channel, _pc = span
    spans = bundle.ad_spans if strict else bundle.break_spans
    lo = p0 + BEFORE_MINUTES
    hi = p1 - dur_min - AFTER_MINUTES
    out = []
    for m in range(lo, hi + 1):
        if not overlaps_any(spans, channel, m - BEFORE_MINUTES, m + dur_min + AFTER_MINUTES):
            out.append(m)
    return out


def sample_matched_pseudo(bundle: Bundle, rng: np.random.Generator, *,
                          k: int = 3, strict: bool = False,
                          effects: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Placebo sample matched 1:k to the real measured breaks.

    For each real measured break that sits inside a matched programme, draw up
    to ``k`` pseudo-break start minutes (same channel, same programme, same
    floor-minute duration) uniformly from the eligible minutes, without
    replacement within the programme (two real breaks of one show never share
    a pseudo minute). The pseudo effect is then measured with the exact
    shipped arithmetic. Deterministic given ``rng``.
    """
    effects = bundle.effects if effects is None else effects
    taken: dict = {}
    rows = []
    for row in effects.itertuples(index=False):
        prog_key = getattr(row, "prog_key")
        if prog_key is None or (isinstance(prog_key, float) and np.isnan(prog_key)):
            continue
        dur = int(getattr(row, "dur_min"))
        eligible = eligible_minutes_for(bundle, prog_key, dur, strict=strict)
        used = taken.setdefault(prog_key, set())
        eligible = [m for m in eligible if m not in used]
        if not eligible:
            continue
        n_draw = min(k, len(eligible))
        picks = rng.choice(len(eligible), size=n_draw, replace=False)
        channel = str(getattr(row, "channel"))
        span = bundle.prog_spans[prog_key]
        prog_len = max(1, span[1] - span[0])
        for pick in np.sort(picks):
            m = eligible[int(pick)]
            used.add(m)
            measured = measure_effect_at(bundle, channel, m, m + dur)
            if measured is None:
                continue
            ts = min_to_ts(m)
            offset_frac = (m - span[0]) / prog_len
            rows.append({
                "channel": channel,
                "prog_key": prog_key,
                "src_s_min": int(getattr(row, "s_min")),
                "pseudo_s_min": m,
                "dur_min": dur,
                "program_type": getattr(row, "program_type"),
                "break_length": getattr(row, "break_length"),
                "pseudo_position": position_bucket(offset_frac),
                "src_position": getattr(row, "break_position"),
                "cal_day": ts.strftime("%Y-%m-%d"),
                **measured,
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["cluster"] = frame["channel"].astype(str) + "|" + frame["cal_day"]
    return frame


def sample_uniform_pseudo(bundle: Bundle, rng: np.random.Generator, *,
                          per_programme: int = 1, strict: bool = False) -> pd.DataFrame:
    """Placebo sample spread uniformly over ALL programmes (design B).

    One eligible pseudo-break per programme (when one exists), with the
    duration drawn from the empirical distribution of real measured break
    durations. Composition therefore follows the EPG, not the break placement
    policy; used as a robustness check on the matched design.
    """
    durations = bundle.effects["dur_min"].to_numpy()
    rows = []
    for prog_key, channel, p0, p1, pricing_class in bundle.all_prog_records:
        dur = int(durations[int(rng.integers(0, len(durations)))])
        eligible = eligible_minutes_for(bundle, prog_key, dur, strict=strict)
        if not eligible:
            continue
        n_draw = min(per_programme, len(eligible))
        picks = rng.choice(len(eligible), size=n_draw, replace=False)
        has_break = overlaps_any(bundle.break_spans, channel, p0, p1)
        for pick in np.sort(picks):
            m = eligible[int(pick)]
            measured = measure_effect_at(bundle, channel, m, m + dur)
            if measured is None:
                continue
            ts = min_to_ts(m)
            rows.append({
                "channel": channel, "prog_key": prog_key,
                "pseudo_s_min": m, "dur_min": dur,
                "program_type": pricing_class,
                "programme_has_break": bool(has_break),
                "cal_day": ts.strftime("%Y-%m-%d"),
                **measured,
            })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["cluster"] = frame["channel"].astype(str) + "|" + frame["cal_day"]
    return frame


# ---------------------------------------------------------------------------
# Cluster-aware inference helpers
# ---------------------------------------------------------------------------

def cluster_bootstrap_mean(frame: pd.DataFrame, rng: np.random.Generator, *,
                           value_col: str = "log_effect", cluster_col: str = "cluster",
                           n_boot: int = 1000) -> np.ndarray:
    """Bootstrap distribution of the mean of ``value_col`` resampling whole clusters."""
    grouped = frame.groupby(cluster_col)[value_col].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy()
    counts = grouped["count"].to_numpy(dtype=float)
    n_clusters = len(grouped)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        total = counts[idx].sum()
        out[b] = sums[idx].sum() / total if total > 0 else np.nan
    return out


def joint_cluster_bootstrap(frames: dict, rng: np.random.Generator, *,
                            value_col: str = "log_effect", cluster_col: str = "cluster",
                            n_boot: int = 1000) -> dict:
    """Jointly bootstrap the means of several frames over the SAME cluster draw.

    Clusters are the union of the frames' cluster labels, so the correlation
    between (say) the real-break mean and the placebo mean on the same
    channel-days is preserved. Returns {name: np.ndarray of bootstrap means}.
    """
    all_clusters = sorted(set().union(*[set(f[cluster_col].unique()) for f in frames.values()]))
    index = {c: i for i, c in enumerate(all_clusters)}
    n_clusters = len(all_clusters)
    stats = {}
    for name, f in frames.items():
        grouped = f.groupby(cluster_col)[value_col].agg(["sum", "count"])
        sums = np.zeros(n_clusters)
        counts = np.zeros(n_clusters)
        for cluster, row in grouped.iterrows():
            sums[index[cluster]] = row["sum"]
            counts[index[cluster]] = row["count"]
        stats[name] = (sums, counts)
    out = {name: np.empty(n_boot) for name in frames}
    for b in range(n_boot):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        for name, (sums, counts) in stats.items():
            total = counts[idx].sum()
            out[name][b] = sums[idx].sum() / total if total > 0 else np.nan
    return out


def percentile_ci(draws: np.ndarray, level: float = 0.95) -> tuple:
    lo = float(np.nanpercentile(draws, 100 * (1 - level) / 2))
    hi = float(np.nanpercentile(draws, 100 * (1 + level) / 2))
    return lo, hi


def dl_pool(effects: pd.DataFrame, value_col: str = "log_effect",
            cell_col: str = "channel_name") -> dict:
    """DerSimonian-Laird pooling summary (the shipped pooling arithmetic).

    Returns mu (precision-weighted global mean of cell means; equals the grand
    mean of individual effects under the machinery's n/s_p^2 weights), tau2,
    pooled_within, the naive independent-breaks SE of mu, and the pooled delta
    exp(mu) - 1 on the optimizer's scale.
    """
    groups = effects.groupby(cell_col)[value_col]
    n = groups.count().to_numpy(dtype=float)
    mean = groups.mean().to_numpy()
    rss = groups.apply(lambda x: float(np.sum((x - x.mean()) ** 2))).to_numpy()
    total_n, n_cells = n.sum(), len(n)
    df = total_n - n_cells
    pooled_within = rss.sum() / df if df > 0 else float("nan")
    w = n / pooled_within
    sw = w.sum()
    mu = float((w * mean).sum() / sw)
    q = float((w * (mean - mu) ** 2).sum())
    c = sw - float((w ** 2).sum()) / sw
    tau2 = max(0.0, (q - (n_cells - 1)) / c) if c > 0 else 0.0
    return {
        "mu": mu, "tau2": tau2, "pooled_within": pooled_within,
        "n": int(total_n), "n_cells": int(n_cells),
        "se_naive": float(np.sqrt(1.0 / sw)),
        "pooled_delta": float(np.exp(mu) - 1.0),
    }


def delta(x: float) -> float:
    """log-effect -> retention-delta scale."""
    return float(np.exp(x) - 1.0)


# ---------------------------------------------------------------------------
# Managed sections of docs/model-validation/causal-identification.md
# ---------------------------------------------------------------------------

_SKELETON = """# Causal identification review: Kairos retention-cost model

Referee-style review of the per-break retention effect shipped in
`models/tv_break_coefficients.json` (pooled delta -0.0391, 2,532 breaks,
November 2024). Every number in the managed sections below is computed from
the real reference data by the seeded, re-runnable scripts named in each
section (`scripts/validation/`). Regenerating a script rewrites only its own
section.

<!-- BEGIN:placebo -->
(placebo section pending: run scripts/validation/run_placebo.py)
<!-- END:placebo -->

<!-- BEGIN:selection -->
(selection section pending: run scripts/validation/run_selection_bias.py)
<!-- END:selection -->

<!-- BEGIN:inference -->
(inference section pending: run scripts/validation/run_inference.py)
<!-- END:inference -->

<!-- BEGIN:loo -->
(leave-one-out section pending: run scripts/validation/run_leave_one_out.py)
<!-- END:loo -->

<!-- BEGIN:verdict -->
(verdict pending)
<!-- END:verdict -->
"""


def write_section(name: str, content: str) -> None:
    """Replace the managed block ``name`` in causal-identification.md."""
    DOC_DIR.mkdir(parents=True, exist_ok=True)
    text = DOC_PATH.read_text(encoding="utf-8") if DOC_PATH.exists() else _SKELETON
    begin, end = f"<!-- BEGIN:{name} -->", f"<!-- END:{name} -->"
    if begin not in text or end not in text:
        text = text.rstrip() + f"\n\n{begin}\n{end}\n"
    head, rest = text.split(begin, 1)
    _, tail = rest.split(end, 1)
    DOC_PATH.write_text(head + begin + "\n" + content.strip() + "\n" + end + tail,
                        encoding="utf-8")
    print(f"[doc] wrote section '{name}' to {DOC_PATH}")
