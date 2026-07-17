"""Season-aware detrend baseline: synthetic two-season proofs.

The shipped detrend divides each break's before/after ratio by the channel's
whole-window typical curve ("global"). On a multi-year window that smears
winter into summer, so measure.break_effects now accepts
baseline_seasonality="month_minute": one curve per calendar month of the
broadcast day with a minimum-sample fallback to global. Nothing shipped uses
it (default "global", proven byte-identical here); activation is owner-gated
via the evaluate-only kairos.model.detrend_gate.

Hand-computed expectations on a two-season fixture (winter level 10, summer
level 2): the month_minute baseline recovers each season's own level, the
fallback fires below the minimum sample count, the measured log effect under
each mode matches arithmetic done by hand, and the held-out gate recommends
month_minute exactly when the months genuinely differ.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from kairos.data import ProgramClassifier
from kairos.model.detrend_gate import detrend_seasonality_gate
from kairos.model.measure import (
    BASELINE_SEASONALITY_MODES,
    _baseline_at,
    _broadcast_month,
    _dayparts_frame,
    _seasonal_baseline_levels,
    break_effects,
)

_MOD_2000 = 18 * 60  # broadcast minute of 20:00 (02:00 day start)


def _classifier() -> ProgramClassifier:
    return ProgramClassifier.from_yaml()


# --- the seasonal baseline map -------------------------------------------------

def _flat_two_season_rows(n_winter: int, n_summer: int) -> pd.DataFrame:
    rows = []
    for day in range(1, n_winter + 1):
        rows.append({"date": pd.Timestamp(f"2024-01-{day:02d}"), "timeband": "20:00",
                     "channel": "A", "tvr": 10.0})
    for day in range(1, n_summer + 1):
        rows.append({"date": pd.Timestamp(f"2024-07-{day:02d}"), "timeband": "20:00",
                     "channel": "A", "tvr": 2.0})
    return pd.DataFrame(rows)


def test_month_minute_recovers_winter_and_summer_levels() -> None:
    frame = _dayparts_frame(_flat_two_season_rows(10, 10))
    seasonal = _seasonal_baseline_levels(frame, min_samples=8)
    # By hand: January holds 10.0, July holds 2.0 at broadcast minute 1080.
    assert seasonal[("A", 1, _MOD_2000)] == pytest.approx(10.0)
    assert seasonal[("A", 7, _MOD_2000)] == pytest.approx(2.0)


def test_month_minute_omits_cells_below_min_samples() -> None:
    # January observed on 3 days only (< 8): no January cell is emitted, so
    # the lookup falls back to the global curve for winter minutes.
    frame = _dayparts_frame(_flat_two_season_rows(3, 10))
    seasonal = _seasonal_baseline_levels(frame, min_samples=8)
    assert ("A", 1, _MOD_2000) not in seasonal
    assert seasonal[("A", 7, _MOD_2000)] == pytest.approx(2.0)
    global_levels = {("A", _MOD_2000): 6.0}
    fell_back = _baseline_at(global_levels, seasonal, "A", pd.Timestamp("2024-01-02 20:00"))
    assert fell_back == pytest.approx(6.0)  # the fallback fired
    summer = _baseline_at(global_levels, seasonal, "A", pd.Timestamp("2024-07-02 20:00"))
    assert summer == pytest.approx(2.0)  # the seasonal cell answered


def test_broadcast_month_keeps_post_midnight_with_its_evening() -> None:
    assert _broadcast_month(pd.Timestamp("2024-12-01 01:30")) == 11
    assert _broadcast_month(pd.Timestamp("2024-12-01 02:00")) == 12


# --- break_effects under each mode, hand-computed --------------------------------

def _two_season_measurement_fixture(n_jan_normal: int):
    """Spots, programmes and dayparts for one January break in a two-season world.

    January days hold 10.0 all evening; the break day (2024-01-20) drops to
    9.0 in the after-window (real shedding). July days hold 2.0 throughout.
    """
    spots = pd.DataFrame([
        {"Channel": "A", "air_dt": pd.Timestamp("2024-01-20 20:06:00"), "Duration": 30.0},
        {"Channel": "A", "air_dt": pd.Timestamp("2024-01-20 20:06:30"), "Duration": 30.0},
    ])
    programmes = pd.DataFrame(
        [("News", "A", "2024-01-20 20:00:00", "2024-01-20 21:00:00", 3600.0)],
        columns=["Title", "Channel", "start", "end", "Duration"],
    )
    programmes["start_dt"] = pd.to_datetime(programmes["start"])
    programmes["end_dt"] = pd.to_datetime(programmes["end"])

    before = {"20:03": 10.0, "20:04": 10.0, "20:05": 10.0}
    after_normal = {"20:08": 10.0, "20:09": 10.0, "20:10": 10.0}
    after_break = {"20:08": 9.0, "20:09": 9.0, "20:10": 9.0}
    rows = []
    for day in range(1, n_jan_normal + 1):
        for tb, tvr in {**before, **after_normal}.items():
            rows.append({"date": pd.Timestamp(f"2024-01-{day:02d}"), "timeband": tb,
                         "channel": "A", "tvr": tvr})
    for tb, tvr in {**before, **after_break}.items():
        rows.append({"date": pd.Timestamp("2024-01-20"), "timeband": tb,
                     "channel": "A", "tvr": tvr})
    for day in range(1, 9):
        for tb in list(before) + list(after_normal):
            rows.append({"date": pd.Timestamp(f"2024-07-{day:02d}"), "timeband": tb,
                         "channel": "A", "tvr": 2.0})
    return spots, programmes, pd.DataFrame(rows)


def test_month_minute_log_effect_matches_hand_arithmetic() -> None:
    # 8 normal January days + the break day = 9 winter samples (>= 8), and 8
    # July days at level 2.0 that ONLY the global mode smears in.
    spots, programmes, dayparts = _two_season_measurement_fixture(8)

    effects_mm = break_effects(spots, programmes, dayparts, _classifier(),
                               baseline_seasonality="month_minute")
    assert len(effects_mm) == 1
    # By hand: observed ratio 9/10. January baseline: before 10.0 (nine days
    # at 10), after (8*10 + 9)/9 = 89/9, so expected ratio 89/90 and
    # log_effect = ln(9/10) - ln(89/90) = ln(81/89).
    assert effects_mm.iloc[0]["observed_ratio"] == pytest.approx(0.9)
    assert effects_mm.iloc[0]["expected_ratio"] == pytest.approx(89.0 / 90.0)
    assert effects_mm.iloc[0]["log_effect"] == pytest.approx(math.log(81.0 / 89.0))

    effects_global = break_effects(spots, programmes, dayparts, _classifier())
    # By hand: the global curve averages winter and summer. Before level
    # (9*10 + 8*2)/17 = 106/17; after (8*10 + 9 + 8*2)/17 = 105/17; expected
    # ratio 105/106 and log_effect = ln(9/10) - ln(105/106).
    assert effects_global.iloc[0]["expected_ratio"] == pytest.approx(105.0 / 106.0)
    assert effects_global.iloc[0]["log_effect"] == pytest.approx(
        math.log(0.9) - math.log(105.0 / 106.0)
    )
    # The smear is real: the two modes disagree on this fixture.
    assert effects_mm.iloc[0]["log_effect"] != effects_global.iloc[0]["log_effect"]

    # Explicit "global" equals the default (the shipped path is untouched).
    effects_explicit = break_effects(spots, programmes, dayparts, _classifier(),
                                     baseline_seasonality="global")
    assert effects_explicit.iloc[0]["log_effect"] == effects_global.iloc[0]["log_effect"]


def test_month_minute_falls_back_to_global_below_min_samples() -> None:
    # Only 3 normal January days + the break day = 4 winter samples (< 8):
    # every January cell is omitted, the fallback fires for every window
    # minute, and month_minute reproduces the global measurement exactly.
    spots, programmes, dayparts = _two_season_measurement_fixture(3)
    effects_mm = break_effects(spots, programmes, dayparts, _classifier(),
                               baseline_seasonality="month_minute")
    effects_global = break_effects(spots, programmes, dayparts, _classifier())
    assert len(effects_mm) == len(effects_global) == 1
    assert effects_mm.iloc[0]["log_effect"] == effects_global.iloc[0]["log_effect"]
    assert effects_mm.iloc[0]["expected_ratio"] == effects_global.iloc[0]["expected_ratio"]


def test_unknown_mode_raises() -> None:
    spots, programmes, dayparts = _two_season_measurement_fixture(3)
    assert BASELINE_SEASONALITY_MODES == ("global", "month_minute")
    with pytest.raises(ValueError, match="baseline_seasonality"):
        break_effects(spots, programmes, dayparts, _classifier(),
                      baseline_seasonality="weekly")


# --- the evaluate-only held-out gate ---------------------------------------------

def test_gate_recommends_month_minute_on_genuine_seasonality() -> None:
    # 15 winter days at 10.0 and 15 summer days at 2.0: on held-out days the
    # month curve predicts exactly, the global curve predicts a smeared mean.
    result = detrend_seasonality_gate(_flat_two_season_rows(15, 15))
    hold = result["detrend_seasonality_holdout"]
    assert result["detrend_seasonality_recommended"] is True
    assert hold["rmse_month_minute"] < hold["rmse_global"]
    assert hold["rmse_month_minute"] == pytest.approx(0.0, abs=1e-12)
    assert hold["relative_improvement"] == pytest.approx(1.0)
    assert hold["min_relative_improvement"] == 0.02
    assert hold["n_test_days"] == 6  # 20 percent of 30 days
    assert "recommended" in result["detrend_seasonality_reason"]


def test_gate_declines_when_months_are_alike() -> None:
    # Both months carry the identical 9/11 alternation: no seasonal structure,
    # so month_minute cannot beat global by the 2 percent bar.
    rows = []
    for month in (1, 7):
        for day in range(1, 15):
            rows.append({
                "date": pd.Timestamp(f"2024-{month:02d}-{day:02d}"),
                "timeband": "20:00", "channel": "A",
                "tvr": 9.0 if day % 2 == 0 else 11.0,
            })
    result = detrend_seasonality_gate(pd.DataFrame(rows))
    assert result["detrend_seasonality_recommended"] is False
    assert "mode stays global" in result["detrend_seasonality_reason"]


def test_gate_abstains_on_too_few_days() -> None:
    result = detrend_seasonality_gate(_flat_two_season_rows(2, 2))
    assert result["detrend_seasonality_recommended"] is False
    assert "too few held-out days" in result["detrend_seasonality_reason"]


def test_gate_reports_honestly_on_empty_dayparts() -> None:
    empty = pd.DataFrame(columns=["date", "timeband", "channel", "tvr"])
    result = detrend_seasonality_gate(empty)
    assert result["detrend_seasonality_recommended"] is False
    assert "cannot be evaluated" in result["detrend_seasonality_reason"]
