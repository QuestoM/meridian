"""Tests for the spot-level window clip (clip_to_all_ad_airtime).

The shipped measurement clips before/after windows against detected breaks
(runs of >= 2 spots). Single-spot ad runs are not breaks, but their airtime is
still commercial audience; the opt-in spot-level clip bounds the windows
against EVERY ad run. These tests pin the hand-computed arithmetic of both
variants on synthetic curves, and the default's byte-stable equivalence when
no single-spot runs exist. Pure pandas, no Meridian.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.model.measure import break_effects
from kairos.model.prepare import identify_breaks


def _classifier():
    from kairos.data import ProgramClassifier

    return ProgramClassifier.from_yaml()


def _programmes() -> pd.DataFrame:
    rows = [("חדשות הערב", "A", "2024-11-04 20:00:00", "2024-11-04 21:00:00", 3600.0)]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "end", "Duration"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    frame["end_dt"] = pd.to_datetime(frame["end"])
    return frame


def _spot(channel: str, start: str, duration: float) -> dict:
    return {"Channel": channel, "air_dt": pd.Timestamp(start), "Duration": duration}


def _break_plus_single_spot(single_at: str) -> pd.DataFrame:
    """A two-spot break 20:06:00-20:07:00 plus one lone spot (not a break)."""
    return pd.DataFrame([
        _spot("A", "2024-11-04 20:06:00", 30),
        _spot("A", "2024-11-04 20:06:30", 30),
        _spot("A", single_at, 30),
    ])


def _dayparts_with_dip() -> pd.DataFrame:
    """Before-window flat 10; after-window 11 except a dip to 5 at 20:09 on the
    break day only (the single spot's airtime). Three normal days keep the
    baseline's 20:09 at (11*3 + 5)/4 = 9.5.
    """
    rows = []
    curve_normal = {"20:03": 10.0, "20:04": 10.0, "20:05": 10.0,
                    "20:08": 11.0, "20:09": 11.0, "20:10": 11.0}
    curve_break_day = dict(curve_normal, **{"20:09": 5.0})
    for date in ("2024-11-05", "2024-11-06", "2024-11-07"):
        for tb, tvr in curve_normal.items():
            rows.append({"date": pd.Timestamp(date), "timeband": tb, "channel": "A", "tvr": tvr})
    for tb, tvr in curve_break_day.items():
        rows.append({"date": pd.Timestamp("2024-11-04"), "timeband": tb, "channel": "A", "tvr": tvr})
    return pd.DataFrame(rows)


def test_identify_breaks_min_spots_one_includes_single_runs() -> None:
    spots = _break_plus_single_spot("2024-11-04 20:09:10")
    assert len(identify_breaks(spots)) == 1  # default: the lone spot is not a break
    runs = identify_breaks(spots, min_spots=1)
    assert len(runs) == 2  # every ad-air run, including the single spot
    assert runs["num_spots"].tolist() == [2, 1]


def test_default_measures_through_single_spot_contamination() -> None:
    # Hand-computed: after-window minutes 20:08, 20:09, 20:10.
    # obs_after = mean(11, 5, 11) = 9.0; obs_before = 10.
    # baseline after = mean(11, 9.5, 11) = 10.5; baseline before = 10.
    # log_effect = ln(0.9) - ln(1.05) = ln(6/7) = -0.15415...
    effects = break_effects(
        _break_plus_single_spot("2024-11-04 20:09:10"),
        _programmes(), _dayparts_with_dip(), _classifier(),
    )
    assert len(effects) == 1
    assert effects.iloc[0]["log_effect"] == pytest.approx(-0.1541507, abs=1e-6)


def test_spot_clip_bounds_window_before_the_single_spot() -> None:
    # With clip_to_all_ad_airtime the lone spot at 20:09 becomes a clip
    # boundary: the after-window is [20:08] only.
    # obs_after = 11, baseline after = 11; obs_before = 10, baseline = 10.
    # log_effect = ln(1.1) - ln(1.1) = 0 exactly: contamination removed.
    effects = break_effects(
        _break_plus_single_spot("2024-11-04 20:09:10"),
        _programmes(), _dayparts_with_dip(), _classifier(),
        clip_to_all_ad_airtime=True,
    )
    assert len(effects) == 1
    assert effects.iloc[0]["log_effect"] == pytest.approx(0.0, abs=1e-9)


def test_spot_clip_drops_break_when_no_clean_minute_survives() -> None:
    # The lone spot airs at 20:08:10 (floor 20:08), the first after-window
    # minute, so the clipped after-window is empty and the break must be
    # DROPPED, never measured on contaminated audience.
    effects = break_effects(
        _break_plus_single_spot("2024-11-04 20:08:10"),
        _programmes(), _dayparts_with_dip(), _classifier(),
        clip_to_all_ad_airtime=True,
    )
    assert effects.empty
    # The default (break-level clip) still measures it: the lone spot is not
    # a detected break, so nothing bounds the window. This is the documented
    # residual bias the flag exists to remove.
    default = break_effects(
        _break_plus_single_spot("2024-11-04 20:08:10"),
        _programmes(), _dayparts_with_dip(), _classifier(),
    )
    assert len(default) == 1


def test_variants_identical_when_no_single_spot_runs_exist() -> None:
    # With only full breaks on air, the spot-level clip changes nothing:
    # both variants measure the same breaks to the same values.
    spots = pd.DataFrame([
        _spot("A", "2024-11-04 20:06:00", 30),
        _spot("A", "2024-11-04 20:06:30", 30),
    ])
    base = break_effects(spots, _programmes(), _dayparts_with_dip(), _classifier())
    variant = break_effects(
        spots, _programmes(), _dayparts_with_dip(), _classifier(),
        clip_to_all_ad_airtime=True,
    )
    assert len(base) == len(variant) == 1
    assert base.iloc[0]["log_effect"] == variant.iloc[0]["log_effect"]
