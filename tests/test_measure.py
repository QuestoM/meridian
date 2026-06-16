"""Tests for the measured, detrended, pooled retention coefficients.

Synthetic minute-level audience curves prove the three properties that make the
measurement honest: the before/after ratio is measured from the curve, the
time-of-day trend is divided out (detrending), and thin cells are shrunk toward
the global mean (pooling). All pure pandas, no Meridian.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from kairos.model.measure import (
    MeasuredCoefficient,
    _broadcast_minute,
    between_cell_variance,
    break_effects,
    channel_coefficients,
    read_coefficients_json,
    write_coefficients_json,
)


# --- broadcast-minute mapping (02:00 start, hours 2..25) --------------------

@pytest.mark.parametrize(
    "stamp, expected",
    [
        ("2024-11-04 02:00:00", 0),
        ("2024-11-04 02:30:00", 30),
        ("2024-11-04 20:00:00", 18 * 60),
        ("2024-11-05 01:30:00", 23 * 60 + 30),  # belongs to the prior broadcast day
    ],
)
def test_broadcast_minute(stamp: str, expected: int) -> None:
    assert _broadcast_minute(pd.Timestamp(stamp)) == expected


# --- synthetic fixtures ------------------------------------------------------

def _programmes() -> pd.DataFrame:
    rows = [("חדשות הערב", "A", "2024-11-04 20:00:00", "2024-11-04 21:00:00", 3600.0)]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "end", "Duration"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    frame["end_dt"] = pd.to_datetime(frame["end"])
    return frame


def _spot(channel: str, start: str, duration: float) -> dict:
    return {"Channel": channel, "air_dt": pd.Timestamp(start), "Duration": duration}


def _dayparts_from_curve(curve: dict[str, float], *, dates=("2024-11-04",)) -> pd.DataFrame:
    """Build a one-channel daypart frame from {timeband: tvr} over the given dates.

    The baseline trajectory is the mean over dates, so supplying several days lets
    a test give the break day a curve that differs from the typical one.
    """
    rows = [
        {"date": pd.Timestamp(date), "timeband": tb, "channel": "A", "tvr": tvr}
        for date in dates
        for tb, tvr in curve.items()
    ]
    return pd.DataFrame(rows)


def _classifier():
    from kairos.data import ProgramClassifier

    return ProgramClassifier.from_yaml()


def _dayparts_mixed(break_after: float) -> pd.DataFrame:
    """One break day plus three normal days, all rising 10 to 11 except the break
    day, whose after-window is ``break_after``. The baseline is the mean over days.
    """
    before = {"20:03": 10.0, "20:04": 10.0, "20:05": 10.0}
    normal_after = {"20:08": 11.0, "20:09": 11.0, "20:10": 11.0}
    break_after_curve = {"20:08": break_after, "20:09": break_after, "20:10": break_after}
    rows = []
    for date in ("2024-11-05", "2024-11-06", "2024-11-07"):
        for tb, tvr in {**before, **normal_after}.items():
            rows.append({"date": pd.Timestamp(date), "timeband": tb, "channel": "A", "tvr": tvr})
    for tb, tvr in {**before, **break_after_curve}.items():
        rows.append({"date": pd.Timestamp("2024-11-04"), "timeband": tb, "channel": "A", "tvr": tvr})
    return pd.DataFrame(rows)


def _break_spots() -> pd.DataFrame:
    return pd.DataFrame([
        _spot("A", "2024-11-04 20:06:00", 30),
        _spot("A", "2024-11-04 20:06:30", 30),  # ends 20:07 -> two-spot break
    ])


def test_break_effect_detrends_the_day_curve() -> None:
    # The break day rises 10 to 11 exactly like every other day, so once the
    # day curve is divided out the detrended effect must be ~0, not +10%.
    effects = break_effects(_break_spots(), _programmes(), _dayparts_mixed(11.0), _classifier())
    assert len(effects) == 1
    assert effects.iloc[0]["log_effect"] == pytest.approx(0.0, abs=1e-9)


def test_break_effect_captures_real_shedding_above_trend() -> None:
    # The break day's after-window is only 9 while the norm at that minute is ~11,
    # so the detrended effect must be negative (real shedding below the trend).
    effects = break_effects(_break_spots(), _programmes(), _dayparts_mixed(9.0), _classifier())
    assert effects.iloc[0]["log_effect"] < 0.0


# --- pooling and clamping ----------------------------------------------------

def _effects(rows: dict[str, tuple[float, int]]) -> pd.DataFrame:
    """Build an effects frame from {channel_name: (log_effect, n)}."""
    frames = []
    for name, (log_effect, n) in rows.items():
        frames.append(pd.DataFrame({"channel_name": [name] * n, "log_effect": [log_effect] * n}))
    return pd.concat(frames, ignore_index=True)


def test_channel_coefficients_shrink_thin_noisy_cell_toward_mean() -> None:
    # 50 breaks clearly shedding, 3 noisy breaks apparently gaining. With a global
    # mean that is negative, the thin cell is pulled negative (partial pooling),
    # not trusted at its raw +20%.
    coefficients = channel_coefficients(
        _effects({"News_first_short": (-0.105, 50), "Other_last_long": (0.182, 3)}),
        shrinkage_k=20.0,
    )
    shedding = coefficients["News_first_short"]
    thin = coefficients["Other_last_long"]
    assert shedding.coefficient < -0.05 and shedding.n == 50
    # The thin cell's raw +18% is shrunk well below itself toward the negative mean.
    assert thin.raw_delta < 0.182
    assert thin.coefficient <= 0.0


def test_channel_coefficients_clamp_genuinely_positive_to_zero() -> None:
    # When the data genuinely shows no shedding (a positive shrunk delta survives),
    # the optimizer coefficient is clamped to zero while raw_delta stays positive.
    coefficients = channel_coefficients(
        _effects({"PrimeShow2_last_short": (0.04, 40), "PrimeShow2_middle_short": (0.04, 40)}),
        shrinkage_k=20.0,
    )
    cell = coefficients["PrimeShow2_last_short"]
    assert cell.raw_delta > 0.0
    assert cell.coefficient == 0.0


def test_channel_coefficients_empty_is_empty() -> None:
    assert channel_coefficients(pd.DataFrame(columns=["channel_name", "log_effect"])) == {}


# --- empirical-Bayes (learned) shrinkage -------------------------------------

def _varied_effects(rows: dict[str, tuple[float, int, float]]) -> pd.DataFrame:
    """Build an effects frame with real within-cell spread.

    ``{channel_name: (mean, n, spread)}``. The n values are symmetric around the
    mean (so the cell mean is exactly ``mean``) but carry a genuine spread, which
    is what lets the hierarchical model estimate a within-cell noise scale instead
    of falling back to the fixed pseudo-count.
    """
    import numpy as np

    frames = []
    for name, (mean, n, spread) in rows.items():
        offsets = spread * np.linspace(-1.0, 1.0, n)
        values = (float(mean) + offsets).tolist()
        frames.append(pd.DataFrame({"channel_name": [name] * n, "log_effect": values}))
    return pd.concat(frames, ignore_index=True)


def test_empirical_bayes_shrinks_thin_cell_more_than_rich_cell() -> None:
    # Two cells share the SAME apparent gain (+0.15) but differ in sample size.
    # A negative-heavy anchor sets the global mean below zero. The thin cell, being
    # measured less precisely, is pulled harder toward that negative mean than the
    # rich cell: precision-weighted shrinkage the fixed pseudo-count cannot express
    # (it would shrink both by count alone, but here the learned strength reacts to
    # how noisy each cell's mean is). Proves the data, not a constant, sets it.
    effects = _varied_effects({
        "News_first_short": (-0.10, 100, 0.03),
        "Other_last_long": (0.15, 4, 0.05),     # thin, same mean as the rich cell
        "Other_middle_long": (0.15, 80, 0.05),  # rich, same mean as the thin cell
    })
    coefficients = channel_coefficients(effects)
    thin = coefficients["Other_last_long"]
    rich = coefficients["Other_middle_long"]
    assert thin.raw_delta < rich.raw_delta  # thin pulled further toward the mean
    assert thin.coefficient <= 0.0 and rich.coefficient <= 0.0
    # The estimator ran the hierarchical path, not the fixed-k fallback.
    diagnostics = between_cell_variance(effects)
    assert diagnostics["method"] == "empirical_bayes"
    assert diagnostics["tau2"] > 0.0
    assert diagnostics["pooled_within_var"] > 0.0


def test_between_cell_variance_separated_cells_learn_weak_shrinkage() -> None:
    # When cells genuinely differ a lot (means far apart, little within-cell noise)
    # the learned strength is SMALL: the data says trust each cell, so the
    # equivalent pseudo-count is well below the old hand-set 20.
    effects = _varied_effects({
        "News_first_short": (-0.10, 60, 0.04),
        "Other_last_long": (0.20, 8, 0.04),
    })
    diagnostics = between_cell_variance(effects)
    assert diagnostics["method"] == "empirical_bayes"
    assert diagnostics["tau2"] > 0.0
    assert diagnostics["pseudo_count"] is not None
    assert diagnostics["pseudo_count"] < 20.0


def test_between_cell_variance_alike_cells_pool_completely() -> None:
    # When cells are statistically indistinguishable (same mean, only noise apart)
    # the between-cell variance is estimated at 0, so the data pools completely:
    # tau2 is 0 and the equivalent pseudo-count is infinite (reported as None).
    effects = _varied_effects({
        "PrimeShow2_last_short": (0.05, 40, 0.06),
        "PrimeShow2_middle_short": (0.05, 40, 0.06),
        "PrimeShow2_first_short": (0.05, 40, 0.06),
    })
    diagnostics = between_cell_variance(effects)
    assert diagnostics["method"] == "empirical_bayes"
    assert diagnostics["tau2"] == 0.0
    assert diagnostics["pseudo_count"] is None


def test_between_cell_variance_empty_frame() -> None:
    diagnostics = between_cell_variance(pd.DataFrame(columns=["channel_name", "log_effect"]))
    assert diagnostics["method"] == "empty"
    assert diagnostics["n_cells"] == 0


def test_channel_coefficients_falls_back_when_no_within_spread() -> None:
    # The synthetic fixtures give every break in a cell an identical effect, so the
    # within-cell variance is zero and the hierarchical model has no noise scale to
    # learn. The estimator must fall back to the fixed pseudo-count rather than
    # divide by zero, and the diagnostics must report that honestly.
    effects = _effects({"News_first_short": (-0.105, 50), "Other_last_long": (0.182, 3)})
    diagnostics = between_cell_variance(effects)
    assert diagnostics["method"] == "fixed_pseudo_count"
    assert diagnostics["pseudo_count"] == 20.0


# --- JSON round trip ---------------------------------------------------------

def test_write_and_read_coefficients_json(tmp_path) -> None:
    coefficients = {
        "News_first_short": MeasuredCoefficient(
            channel_name="News_first_short", coefficient=-0.05, raw_delta=-0.05,
            n=120, ci_low=-0.08, ci_high=-0.02,
        ),
    }
    path = tmp_path / "coeffs.json"
    write_coefficients_json(path, coefficients, metadata={"detrended": True})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["coefficients"]["News_first_short"] == -0.05
    assert payload["detail"]["News_first_short"]["n"] == 120
    assert payload["metadata"]["detrended"] is True
    # The flat reader returns just the channel -> delta map.
    assert read_coefficients_json(path) == {"News_first_short": -0.05}


def test_read_missing_json_is_empty(tmp_path) -> None:
    assert read_coefficients_json(tmp_path / "nope.json") == {}
