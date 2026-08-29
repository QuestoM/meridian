"""The pods the channel aired, counted the way the channel counted them.

Every other break count in this engine is inferred by grouping spots that fall
within some seconds of each other, and that inference is load-bearing enough to
move the revenue figure by millions when the threshold moves. The as-run log did
not need to be guessed at: it carries the broadcaster's own block numbering, and
nothing read it until this module.

What these tests pin:

- the reconstruction round-trips against the size the channel itself reported,
  so the ground truth is checked rather than trusted,
- concurrent channels do not contaminate each other, which is the one mistake
  that makes every pod look shorter and more numerous than it was,
- unsold airtime (promos, sponsorships, public-service announcements) is not
  counted as a commercial pod, because the channel never numbered it,
- and a source without the block columns answers empty instead of raising,
  because an older export is a real state and not a programming error.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.data import as_aired


def _spot(channel, when, duration, position, block_size, spot_type="פרסומת", tvr=1.0):
    return {
        "Channel": channel,
        "air_dt": pd.Timestamp(when),
        "Duration": duration,
        "Pos. Block 1": position,
        "Spots Block 1": block_size,
        "Spot type": spot_type,
        "TVR": tvr,
    }


@pytest.fixture()
def two_channels() -> pd.DataFrame:
    """Two broadcasters airing pods at overlapping times, as they really do."""
    rows = [
        # Channel A: a three-spot pod, then a two-spot pod.
        _spot("A", "2024-11-01 08:00:00", 30, 1, 3),
        _spot("A", "2024-11-01 08:00:30", 20, 2, 3),
        _spot("A", "2024-11-01 08:00:50", 10, 3, 3),
        _spot("A", "2024-11-01 09:00:00", 45, 1, 2),
        _spot("A", "2024-11-01 09:00:45", 15, 2, 2),
        # Channel B interleaves in time with A's first pod.
        _spot("B", "2024-11-01 08:00:10", 25, 1, 2),
        _spot("B", "2024-11-01 08:00:35", 35, 2, 2),
    ]
    return pd.DataFrame(rows)


def test_the_reconstruction_agrees_with_the_size_the_channel_reported(two_channels):
    """The channel states each block's size on every row of that block. Counting
    the rows we grouped must reproduce it, or the ground truth is not ground."""
    for channel in ("A", "B"):
        report = as_aired.reconstruction_agreement(two_channels, channel=channel)
        assert report["agreement"] == 1.0, f"{channel} did not round-trip: {report}"


def test_concurrent_channels_do_not_split_each_other_s_pods(two_channels):
    """Sorting by time alone interleaves broadcasters, and the position counter
    then restarts constantly. Measured on the real month that mistake reported
    1,387 pods on the operator channel instead of 829, at a median of 84 seconds
    instead of 181 -- every pod shorter and more numerous than it truly was,
    which is the exact error this module exists to remove."""
    both = as_aired.identify_aired_pods(two_channels)
    assert len(both) == 3, f"expected 2 pods on A and 1 on B, got {len(both)}"
    a_pods = both[both["channel"] == "A"]
    assert list(a_pods["spots"]) == [3, 2]
    assert list(a_pods["seconds"]) == [60, 60]

    # And the answer must not depend on whether the caller filtered first.
    alone = as_aired.identify_aired_pods(two_channels, channel="A")
    assert len(alone) == len(a_pods)
    assert list(alone["seconds"]) == list(a_pods["seconds"])


def test_airtime_the_channel_never_numbered_is_not_a_commercial_pod():
    """Promos, sponsorships and PSAs carry no block number. Counting them as
    sold pods is what makes an inferred break count explode."""
    rows = [
        _spot("A", "2024-11-01 08:00:00", 30, 1, 2),
        _spot("A", "2024-11-01 08:00:30", 30, 2, 2),
        _spot("A", "2024-11-01 08:01:00", 30, 0, 0, spot_type="פרומו"),
        _spot("A", "2024-11-01 08:01:30", 30, 0, 0, spot_type="חסות"),
        _spot("A", "2024-11-01 08:02:00", 30, 0, 0, spot_type="תשדיר שרות"),
    ]
    pods = as_aired.identify_aired_pods(pd.DataFrame(rows))
    assert len(pods) == 1
    assert int(pods.iloc[0]["spots"]) == 2
    assert int(pods.iloc[0]["seconds"]) == 60, "unsold airtime must not inflate the pod"


def test_a_counter_that_never_restarts_at_one_still_separates_pods():
    """A log that begins mid-pod never shows position 1. The counter going
    backwards is the other signal a pod ended, and missing it would merge two
    real pods into one long fiction."""
    rows = [
        _spot("A", "2024-11-01 08:00:00", 30, 4, 5),
        _spot("A", "2024-11-01 08:00:30", 30, 5, 5),
        _spot("A", "2024-11-01 09:00:00", 30, 2, 3),
        _spot("A", "2024-11-01 09:00:30", 30, 3, 3),
    ]
    pods = as_aired.identify_aired_pods(pd.DataFrame(rows))
    assert len(pods) == 2


def test_a_source_without_the_block_columns_answers_empty_rather_than_raising():
    """An older export is a real state. It must read as "no ground truth here",
    never as a crash and never as a fabricated count."""
    bare = pd.DataFrame([{
        "Channel": "A", "air_dt": pd.Timestamp("2024-11-01 08:00:00"), "Duration": 30,
    }])
    pods = as_aired.identify_aired_pods(bare)
    assert pods.empty
    assert list(pods.columns) == ["channel", "day", "hour", "start", "seconds", "spots", "tvr"]
    assert as_aired.reconstruction_agreement(bare)["agreement"] is None
    assert as_aired.identify_aired_pods(pd.DataFrame()).empty


def test_the_hourly_load_is_what_a_cap_is_measured_against(two_channels):
    """A regulatory cap counts seconds in a clock hour, and any plan-versus-
    reality comparison has to hold this fixed or it measures the volume
    assumption instead of the schedule."""
    load = as_aired.hourly_ad_load(as_aired.identify_aired_pods(two_channels))
    a_eight = load[(load["channel"] == "A") & (load["hour"] == 8)]
    assert len(a_eight) == 1
    assert int(a_eight.iloc[0]["seconds"]) == 60
    assert int(a_eight.iloc[0]["pods"]) == 1
    assert as_aired.hourly_ad_load(pd.DataFrame()).empty
