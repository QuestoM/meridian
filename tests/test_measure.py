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
