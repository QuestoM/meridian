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
     times is shrunk toward the global mean by a strength set with a pseudo-count.
     This is partial pooling: it refuses to trust a large difference drawn from a
     small sample.

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
from typing import Mapping, Optional

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import load_dayparts, load_programmes, load_spots
from kairos.model.prepare import keyed_breaks

logger = logging.getLogger(__name__)

# Window sizes, in minutes, for the before and after audience measurement.
_BEFORE_MINUTES = 3
_AFTER_MINUTES = 3
# Broadcast day starts at 02:00 (daypart timebands run hours 2..25).
_BROADCAST_DAY_START_HOUR = 2
_MINUTES_PER_DAY = 1440
# Partial-pooling strength: a cell is shrunk toward the global mean as if it
# carried this many extra observations at that mean. Larger means more shrinkage.
_DEFAULT_SHRINKAGE_K = 20.0


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


def break_effects(
    spots: pd.DataFrame,
    programmes: pd.DataFrame,
    dayparts: pd.DataFrame,
    classifier: ProgramClassifier,
    *,
    before_minutes: int = _BEFORE_MINUTES,
    after_minutes: int = _AFTER_MINUTES,
) -> pd.DataFrame:
    """Measure the detrended retention effect of every real break.

    Returns one row per measurable break with its engine channel name and the
    log effect ``log(observed_ratio) - log(expected_ratio)``: positive means the
    break held more audience than the time-of-day trend predicts, negative means
    it shed more. Breaks whose windows have no positive audience are dropped (no
    fabricated rating).
    """
    breaks = keyed_breaks(spots, programmes, classifier)
    columns = [
        "channel_name", "program_type", "break_position", "break_length",
        "observed_ratio", "expected_ratio", "log_effect",
    ]
    if breaks.empty:
        return pd.DataFrame(columns=columns)

    frame = _dayparts_frame(dayparts)
    observed = _minute_lookup(frame)
    baseline = _baseline_levels(frame)

    before_offsets = [-(k + 1) for k in range(before_minutes)]
    after_offsets = [k + 1 for k in range(after_minutes)]

    rows: list[dict[str, object]] = []
    for row in breaks.itertuples(index=False):
        channel = str(getattr(row, "channel"))
        start = pd.Timestamp(getattr(row, "break_start")).floor("min")
        end = pd.Timestamp(getattr(row, "break_end")).floor("min")

        before_ts = [start + pd.Timedelta(minutes=o) for o in before_offsets]
        after_ts = [end + pd.Timedelta(minutes=o) for o in after_offsets]

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
            }
        )

    return pd.DataFrame(rows, columns=columns)


def channel_coefficients(
    effects: pd.DataFrame,
    *,
    shrinkage_k: float = _DEFAULT_SHRINKAGE_K,
) -> dict[str, MeasuredCoefficient]:
    """Pool the per-break log effects into one delta per channel.

    Each channel's mean log effect is shrunk toward the global mean with a
    pseudo-count of ``shrinkage_k`` (partial pooling), converted to a retention
    delta ``exp(shrunk) - 1``, and clamped to be non-positive for the optimizer.
    A 95% interval is carried from the standard error of the cell's mean.
    """
    coefficients: dict[str, MeasuredCoefficient] = {}
    if effects.empty:
        return coefficients

    grand_mean = float(effects["log_effect"].mean())
    for channel_name, group in effects.groupby("channel_name"):
        logs = group["log_effect"].to_numpy()
        n = int(len(logs))
        cell_mean = float(np.mean(logs))
        shrunk = (n * cell_mean + shrinkage_k * grand_mean) / (n + shrinkage_k)
        raw_delta = float(np.exp(shrunk) - 1.0)

        std = float(np.std(logs, ddof=1)) if n > 1 else 0.0
        se = std / np.sqrt(n) if n > 0 else 0.0
        low_log = shrunk - 1.96 * se
        high_log = shrunk + 1.96 * se
        coefficients[str(channel_name)] = MeasuredCoefficient(
            channel_name=str(channel_name),
            coefficient=min(0.0, raw_delta),
            raw_delta=raw_delta,
            n=n,
            ci_low=float(np.exp(low_log) - 1.0),
            ci_high=float(np.exp(high_log) - 1.0),
        )
    return coefficients


def compute_measured_coefficients(
    *,
    spots: Optional[pd.DataFrame] = None,
    programmes: Optional[pd.DataFrame] = None,
    dayparts: Optional[pd.DataFrame] = None,
    classifier: Optional[ProgramClassifier] = None,
    shrinkage_k: float = _DEFAULT_SHRINKAGE_K,
) -> dict[str, MeasuredCoefficient]:
    """Load the reference data (unless frames are supplied) and measure coefficients."""
    spots = load_spots() if spots is None else spots
    programmes = load_programmes() if programmes is None else programmes
    dayparts = load_dayparts() if dayparts is None else dayparts
    classifier = classifier or ProgramClassifier.from_yaml()
    effects = break_effects(spots, programmes, dayparts, classifier)
    return channel_coefficients(effects, shrinkage_k=shrinkage_k)


def write_coefficients_json(
    path: str | Path,
    coefficients: Mapping[str, MeasuredCoefficient],
    *,
    metadata: Optional[Mapping[str, object]] = None,
) -> Path:
    """Write the measured coefficients (with provenance) as JSON.

    The file carries a flat ``coefficients`` map (channel name -> delta) that
    :func:`kairos.model.impact.load_impact_model` reads, plus per-channel detail
    and any ``metadata`` (data window, method) for audit.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "measured_detrended_pooled",
        "metadata": dict(metadata or {}),
        "coefficients": {name: c.coefficient for name, c in coefficients.items()},
        "detail": {name: asdict(c) for name, c in coefficients.items()},
    }
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
