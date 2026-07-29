"""Competitor lineup: airing collapse, EB title means, overlap math, null-vs-zero.

Every numeric assertion here is hand-computed from the synthetic fixture and
written out in a comment, so the module's conventions (airing collapse rules,
EB shrinkage toward the channel mean, change-point windows, overlap-weighted
query pressure, null for unknown dates versus 0.0 for known-empty lineups) are
pinned by arithmetic, not by trusting the implementation. The final test is a
real-data smoke on the historical window.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.model.competitor_lineup import (
    EB_PRIOR_AIRINGS,
    LINEUP_COLUMNS,
    SECONDS_PER_DAY,
    collapse_airings,
    lineup_frame,
    pressure_for_window,
    title_strengths,
)

_OWN = "קשת 12"
_RIVAL = "רשת 13"
_RIVAL2 = "כאן 11"


def _spot(channel: str, title: str, when: str, duration: float, tvr: float) -> dict:
    return {
        "Channel": channel,
        "Title": title,
        "air_dt": pd.Timestamp(when),
        "Duration": duration,
        "TVR": tvr,
    }


# --- airing collapse -----------------------------------------------------------

def _collapse_fixture() -> pd.DataFrame:
    return pd.DataFrame(
        [
            # Title X: two spots 15 minutes apart -> ONE airing 20:00:00-20:15:30.
            _spot(_RIVAL, "X", "2024-11-04 20:00:00", 30.0, 2.0),
            _spot(_RIVAL, "X", "2024-11-04 20:15:00", 30.0, 4.0),
            # Title Y at 20:30 -> the title change splits a new airing.
            _spot(_RIVAL, "Y", "2024-11-04 20:30:00", 30.0, 1.0),
            # Title X again at 22:00 -> a separate airing (title changed back).
            _spot(_RIVAL, "X", "2024-11-04 22:00:00", 30.0, 6.0),
        ]
    )


def test_airing_collapse_splits_on_title_change_and_gap() -> None:
    airings = collapse_airings(_collapse_fixture())
    assert list(airings["title"]) == ["X", "Y", "X"]
    first = airings.iloc[0]
    assert first["start_dt"] == pd.Timestamp("2024-11-04 20:00:00")
    # End is the last spot's clock plus its duration: 20:15:00 + 30s.
    assert first["end_dt"] == pd.Timestamp("2024-11-04 20:15:30")
    # Mean of the airing's spot TVRs: (2.0 + 4.0) / 2 = 3.0.
    assert first["mean_tvr"] == pytest.approx(3.0)
    assert int(first["spot_count"]) == 2


def test_airing_collapse_splits_same_title_across_a_large_gap() -> None:
    frame = pd.DataFrame(
        [
            _spot(_RIVAL, "X", "2024-11-04 10:00:00", 30.0, 2.0),
            # 119.5 minutes after the previous spot's end: over the 60-minute
            # gap rule, so the same title starts a NEW airing.
            _spot(_RIVAL, "X", "2024-11-04 12:00:00", 30.0, 4.0),
        ]
    )
    airings = collapse_airings(frame)
    assert len(airings) == 2
    assert list(airings["mean_tvr"]) == pytest.approx([2.0, 4.0])


def test_airing_collapse_crosses_midnight_as_one_airing() -> None:
    frame = pd.DataFrame(
        [
            _spot(_RIVAL, "N", "2024-11-04 23:50:00", 30.0, 2.0),
            _spot(_RIVAL, "N", "2024-11-05 00:10:00", 30.0, 2.0),
        ]
    )
    airings = collapse_airings(frame)
    assert len(airings) == 1
    assert airings.iloc[0]["end_dt"] == pd.Timestamp("2024-11-05 00:10:30")


# --- EB title strengths --------------------------------------------------------

def test_eb_strength_shrinks_a_one_airing_title_toward_the_channel_mean() -> None:
    frame = pd.DataFrame(
        [
            # Title A: two airings (2 hours apart), each mean TVR 4.0.
            _spot(_RIVAL, "A", "2024-11-04 18:00:00", 30.0, 4.0),
            _spot(_RIVAL, "A", "2024-11-04 21:00:00", 30.0, 4.0),
            # Title B: one airing, mean TVR 1.0.
            _spot(_RIVAL, "B", "2024-11-04 19:30:00", 30.0, 1.0),
        ]
    )
    strengths = title_strengths(collapse_airings(frame)).set_index("title")
    assert EB_PRIOR_AIRINGS == 5.0
    # Airing means on the channel: 4.0, 4.0, 1.0 -> channel mean 3.0.
    # A: n=2 airings -> (2*4.0 + 5*3.0) / 7 = 23/7.
    assert strengths.loc["A", "strength"] == pytest.approx(23.0 / 7.0)
    # B: n=1 airing -> (1*1.0 + 5*3.0) / 6 = 16/6: the single airing cannot
    # spike (or crater) the title far from the channel mean.
    assert strengths.loc["B", "strength"] == pytest.approx(16.0 / 6.0)


# --- lineup windows and overlap weighting --------------------------------------

def _lineup_fixture() -> pd.DataFrame:
    """Rival airs A 20:00:00-20:10:00 then B 20:10:00-20:20:00; owned airs too."""
    return pd.DataFrame(
        [
            _spot(_RIVAL, "A", "2024-11-04 20:00:00", 30.0, 4.0),
            _spot(_RIVAL, "A", "2024-11-04 20:09:30", 30.0, 4.0),
            _spot(_RIVAL, "B", "2024-11-04 20:10:00", 30.0, 1.0),
            _spot(_RIVAL, "B", "2024-11-04 20:19:30", 30.0, 1.0),
            # The owned channel's own huge programme must contribute NOTHING.
            _spot(_OWN, "OWN-HIT", "2024-11-04 20:00:00", 30.0, 25.0),
            _spot(_OWN, "OWN-HIT", "2024-11-04 20:19:30", 30.0, 25.0),
        ]
    )


def test_lineup_frame_columns_and_owned_channel_exclusion() -> None:
    frame = lineup_frame(["2024-11-04"], _OWN, spots=_lineup_fixture(), epg=None)
    assert tuple(frame.columns) == LINEUP_COLUMNS
    assert frame.attrs["coverage"] == {"2024-11-04": "history"}
    joined = ";".join(frame["competitor_titles"])
    assert "OWN-HIT" not in joined
    assert "A" in joined and "B" in joined
    # The windows partition the whole day contiguously.
    assert int(frame.iloc[0]["start_seconds"]) == 0
    assert int(frame.iloc[-1]["end_seconds"]) == SECONDS_PER_DAY
    starts = list(frame["start_seconds"])[1:]
    ends = list(frame["end_seconds"])[:-1]
    assert starts == ends


def test_overlap_weighted_pressure_hand_math() -> None:
    frame = lineup_frame(["2024-11-04"], _OWN, spots=_lineup_fixture(), epg=None)
    # Rival airing means: A 4.0 (one airing), B 1.0 (one airing); channel mean
    # (4.0 + 1.0) / 2 = 2.5. EB strengths: A = (1*4.0 + 5*2.5)/6 = 2.75,
    # B = (1*1.0 + 5*2.5)/6 = 2.25.
    a_window = pressure_for_window(frame, "2024-11-04", 72_000, 72_600)  # 20:00-20:10
    assert a_window["competitor_pressure"] == pytest.approx(2.75)
    assert a_window["competitor_titles"] == ["A"]
    # Query 20:05-20:15 straddles the change point at 20:10: 300 seconds over
    # A's window and 300 over B's -> (2.75*300 + 2.25*300) / 600 = 2.5.
    straddle = pressure_for_window(frame, "2024-11-04", 72_300, 72_900)
    assert straddle["competitor_pressure"] == pytest.approx(2.5)
    assert straddle["competitor_titles"] == ["A", "B"]
    assert straddle["covered_seconds"] == pytest.approx(600.0)


def test_midnight_overhang_reaches_the_next_covered_day() -> None:
    frame_spots = pd.DataFrame(
        [
            # One airing crossing midnight: 2024-11-04 23:50 -> 11-05 00:10:30.
            _spot(_RIVAL, "N", "2024-11-04 23:50:00", 30.0, 2.0),
            _spot(_RIVAL, "N", "2024-11-05 00:10:00", 30.0, 2.0),
            # A second airing STARTING on 11-05, which makes 11-05 history-covered.
            _spot(_RIVAL, "M", "2024-11-05 20:00:00", 30.0, 2.0),
        ]
    )
    frame = lineup_frame(["2024-11-05"], _OWN, spots=frame_spots, epg=None)
    # The overhang occupies [0, 630) seconds of 11-05 (00:10:30 = 630s).
    head = pressure_for_window(frame, "2024-11-05", 0, 630)
    assert head["competitor_titles"] == ["N"]
    assert head["competitor_pressure"] is not None and head["competitor_pressure"] > 0.0


# --- null versus zero ----------------------------------------------------------

def test_unknown_date_is_null_never_zero() -> None:
    frame = lineup_frame(["2024-12-10"], _OWN, spots=_lineup_fixture(), epg=None)
    assert frame.attrs["coverage"] == {"2024-12-10": "unknown"}
    assert len(frame) == 1
    assert pd.isna(frame.iloc[0]["competitor_pressure"])
    assert pressure_for_window(frame, "2024-12-10", 0, 3600)["competitor_pressure"] is None


def test_known_empty_forward_lineup_is_zero() -> None:
    # The forward EPG covers 12-08 through 12-12 but lists nothing on 12-10:
    # a KNOWN-empty lineup, honestly 0.0 (unlike the unknown-date null above).
    epg = pd.DataFrame(
        [
            {"Channel": _RIVAL, "Title": "A",
             "start_dt": pd.Timestamp("2024-12-08 20:00:00"),
             "end_dt": pd.Timestamp("2024-12-08 21:00:00")},
            {"Channel": _RIVAL, "Title": "A",
             "start_dt": pd.Timestamp("2024-12-12 20:00:00"),
             "end_dt": pd.Timestamp("2024-12-12 21:00:00")},
        ]
    )
    frame = lineup_frame(["2024-12-10"], _OWN, spots=_lineup_fixture(), epg=epg)
    assert frame.attrs["coverage"] == {"2024-12-10": "forward_epg"}
    assert len(frame) == 1
    assert frame.iloc[0]["competitor_pressure"] == 0.0
    assert frame.iloc[0]["competitor_titles"] == ""


def test_forward_date_resolves_titles_through_the_epg() -> None:
    epg = pd.DataFrame(
        [
            # Title A is known from history -> its EB strength (2.75) applies.
            {"Channel": _RIVAL, "Title": "A",
             "start_dt": pd.Timestamp("2024-12-08 20:00:00"),
             "end_dt": pd.Timestamp("2024-12-08 21:00:00")},
            # Title Z has no history -> honest 0.0 contribution, still listed.
            {"Channel": _RIVAL2, "Title": "Z",
             "start_dt": pd.Timestamp("2024-12-08 20:00:00"),
             "end_dt": pd.Timestamp("2024-12-08 21:00:00")},
        ]
    )
    frame = lineup_frame(["2024-12-08"], _OWN, spots=_lineup_fixture(), epg=epg)
    assert frame.attrs["coverage"] == {"2024-12-08": "forward_epg"}
    prime = pressure_for_window(frame, "2024-12-08", 72_000, 75_600)  # 20:00-21:00
    assert prime["competitor_pressure"] == pytest.approx(2.75)
    assert prime["competitor_titles"] == ["A", "Z"]


# --- real-data smoke -----------------------------------------------------------

def test_real_data_smoke_historical_window() -> None:
    try:
        from kairos.data.loaders import load_spots

        spots = load_spots()
    except FileNotFoundError:
        pytest.skip("reference spots data absent on this machine")
    dates = ["2024-11-04", "2024-11-05", "2024-11-06", "2026-08-03"]
    frame = lineup_frame(dates, _OWN, spots=spots, epg=None)
    assert not frame.empty
    coverage = frame.attrs["coverage"]
    # The November dates sit inside the spots history and resolve from it;
    # the forward date has no EPG here and must be honestly unknown.
    for day in dates[:3]:
        assert coverage[day] == "history"
        day_rows = frame[frame["date"] == day]
        assert int(day_rows.iloc[0]["start_seconds"]) == 0
        assert int(day_rows.iloc[-1]["end_seconds"]) == SECONDS_PER_DAY
        assert day_rows["competitor_pressure"].notna().all()
        assert (day_rows["competitor_pressure"] > 0.0).any()
    assert coverage["2026-08-03"] == "unknown"
    forward = frame[frame["date"] == "2026-08-03"]
    assert len(forward) == 1
    assert pd.isna(forward.iloc[0]["competitor_pressure"])
