"""P11: the pacing board's arithmetic, and the four ways it refuses to state a pace.

Every fixture here is written as the two stores actually hold it, so a test that
passes is a statement about the shipped readers and not about a mock. The bar the
assertions encode is JS-6's: a three-state verdict, a published numeric trigger,
and never a figure the ledger did not carry.

The last test in this file is the guard on the whole class. It takes the same
campaign twice, once with every elapsed day sourced and once with one elapsed day
missing, and asserts the verdict changes. A board that counted a missing day as
zero would return the same confident ratio for both, which is the exact defect
this piece exists to make impossible.
"""

from __future__ import annotations

from datetime import date

import pytest

from kairos_api import pacing_alerts_api_board as board
from kairos_api import pacing_alerts_api_words as words

AS_OF = date(2025, 4, 30)


def _campaign(**overrides):
    record = {
        "campaign_id": "CMP_T1",
        "name": "campaign under test",
        "advertiser": "מפרסם",
        "agency_id": "AGY_01",
        "channel": "רשת 13",
        "status": "active",
        "starts_on": "2025-04-27",
        "ends_on": "2025-05-03",
        "is_demo": False,
        "demo": {},
        "flights": [],
        "commitment": {
            "budget_ils": 70000.0,
            "rating_goal_points": 70.0,
            "rating_goal_audience": "all_viewers",
            "rating_goal_measurable": True,
        },
    }
    record.update(overrides)
    return record


def _day(when: str, state: str, rating: float = 0.0, spend: float = 0.0):
    known = state in ("aired", "scheduled")
    return {
        "broadcast_date": when,
        "air_state": state,
        "channel": "רשת 13",
        "spots": 1 if known else None,
        "seconds": 30.0 if known else None,
        "rating_points_planned": rating if known else None,
        "spend_ils": spend if known else None,
        "spots_dropped_by_rule": 0 if known else None,
        "dropped_rule_id": "",
        "figures_basis": "traffic file" if known else "",
        "source_file": "Wally_Prime_Reshet_Example_2025-04-27.csv" if known else "",
        "note": "",
        "is_demo": False,
    }


def _full_elapsed(rating_per_day: float = 10.0):
    """The four elapsed days of the flight, every one of them sourced."""
    return [_day(when, "aired", rating_per_day, 7000.0)
            for when in ("2025-04-27", "2025-04-28", "2025-04-29", "2025-04-30")]


def _unsourced_remaining():
    return [_day(when, "unknown") for when in ("2025-05-01", "2025-05-02", "2025-05-03")]


def test_a_complete_elapsed_window_states_a_pace_against_the_published_reference() -> None:
    row = board.campaign_row(_campaign(), _full_elapsed() + _unsourced_remaining(), AS_OF)
    rating = row["rating"]
    # Four of seven flight days counted, so an even share of the goal is 40.0.
    assert rating["counted"]["days_counted"] == 4
    assert rating["counted"]["days_in_flight"] == 7
    assert rating["reference"]["expected_through_counted_day"] == 40.0
    assert rating["counted"]["through_counted_day"] == 40.0
    assert rating["pace"]["verdict"] == words.ON_PACE
    assert rating["pace"]["ratio"] == 1.0
    assert row["headline"]["verdict"] == words.ON_PACE


@pytest.mark.parametrize(
    "per_day,expected",
    [(10.0, words.ON_PACE), (9.3, words.AT_RISK), (8.0, words.BEHIND)],
)
def test_the_two_published_triggers_are_the_only_thing_that_decides_the_verdict(per_day, expected) -> None:
    row = board.campaign_row(_campaign(), _full_elapsed(per_day) + _unsourced_remaining(), AS_OF)
    ratio = row["rating"]["pace"]["ratio"]
    assert row["rating"]["pace"]["verdict"] == expected
    if expected == words.ON_PACE:
        assert ratio >= words.ON_PACE_RATIO
    elif expected == words.AT_RISK:
        assert words.AT_RISK_RATIO <= ratio < words.ON_PACE_RATIO
    else:
        assert ratio < words.AT_RISK_RATIO


def test_an_unsourced_day_inside_the_flight_is_never_counted_as_zero() -> None:
    days = _full_elapsed() + _unsourced_remaining()
    row = board.campaign_row(_campaign(), days, AS_OF)
    # The three remaining days carry no source. They are listed, and the counted
    # figure covers only the four days that do.
    assert row["flight"]["unsourced_remaining_days"] == ["2025-05-01", "2025-05-02", "2025-05-03"]
    assert row["rating"]["counted"]["booked_total"] == 40.0
    assert row["rating"]["forward"]["state"] == words.NOT_BOOKED_YET
    assert row["rating"]["forward"]["remaining_to_goal"] == 30.0
    assert row["rating"]["forward"]["reason_en"]
    assert row["rating"]["forward"]["path_forward_en"]


def test_a_flight_whose_remaining_days_are_all_sourced_and_short_is_measured_not_projected() -> None:
    days = _full_elapsed() + [_day(when, "scheduled", 5.0, 3000.0)
                              for when in ("2025-05-01", "2025-05-02", "2025-05-03")]
    row = board.campaign_row(_campaign(), days, AS_OF)
    assert row["rating"]["counted"]["booked_total"] == 55.0
    assert row["rating"]["forward"]["state"] == words.SHORT_CERTAIN
    assert row["rating"]["forward"]["remaining_to_goal"] == 15.0
    assert row["rating"]["forward"]["unsourced_remaining_days"] == []


def test_a_flight_booked_past_its_goal_says_so() -> None:
    days = _full_elapsed(20.0) + [_day(when, "scheduled", 20.0, 3000.0)
                                  for when in ("2025-05-01", "2025-05-02", "2025-05-03")]
    row = board.campaign_row(_campaign(), days, AS_OF)
    assert row["rating"]["forward"]["state"] == words.COVERED
    assert row["rating"]["forward"]["remaining_to_goal"] == 0.0


def test_a_campaign_with_no_goal_carries_a_reason_and_no_ratio() -> None:
    terms = {"budget_ils": None, "rating_goal_points": None,
             "rating_goal_audience": "", "rating_goal_measurable": False}
    row = board.campaign_row(_campaign(commitment=terms), _full_elapsed(), AS_OF)
    assert row["headline"]["verdict"] == words.UNKNOWN
    assert row["headline"]["ratio"] is None
    assert row["headline"]["code"] == "no_goal"
    assert row["headline"]["reason_he"]
    assert row["headline"]["path_forward_he"]


def test_a_goal_on_an_audience_with_no_panel_is_unknown_rather_than_measured_on_another_base() -> None:
    terms = {"budget_ils": None, "rating_goal_points": 70.0,
             "rating_goal_audience": "adults_25_54", "rating_goal_measurable": False}
    row = board.campaign_row(_campaign(commitment=terms), _full_elapsed(), AS_OF)
    assert row["rating"]["pace"]["verdict"] == words.UNKNOWN
    assert row["rating"]["pace"]["code"] == "unmeasurable"
    assert row["rating"]["pace"]["ratio"] is None


def test_a_campaign_with_no_flight_dates_has_no_flight_to_pace_across() -> None:
    row = board.campaign_row(_campaign(starts_on="", ends_on=""), [], AS_OF)
    assert row["flight"] is None
    assert row["headline"]["code"] == "no_flight_dates"
    assert row["rating"] is None


def test_a_flight_that_has_not_started_is_not_behind() -> None:
    row = board.campaign_row(_campaign(), [], date(2025, 4, 20))
    assert row["headline"]["verdict"] == words.UNKNOWN
    assert row["headline"]["code"] == "not_started"


def test_the_board_is_ordered_worst_first() -> None:
    rows = board.build_rows(
        [
            _campaign(campaign_id="A"),
            _campaign(campaign_id="B"),
            _campaign(campaign_id="C"),
        ],
        {
            "A": _full_elapsed(10.0) + _unsourced_remaining(),
            "B": _full_elapsed(8.0) + _unsourced_remaining(),
            "C": _full_elapsed(9.3) + _unsourced_remaining(),
        },
        AS_OF,
    )
    assert [row["campaign_id"] for row in rows] == ["B", "C", "A"]
    assert board.counts(rows) == {
        words.BEHIND: 1, words.AT_RISK: 1, words.ON_PACE: 1, words.UNKNOWN: 0,
        "total": 3, "demo": 0,
    }


def test_a_hole_in_the_elapsed_window_removes_the_verdict_rather_than_moving_it() -> None:
    """The guard on the class. A missing elapsed day must not be counted as zero.

    Both boards below hold the same three sourced days. The first knows those are
    all the elapsed days there are; the second has a fourth elapsed day with no
    source at all. A reader who was handed one ratio for both cases would have been
    told a campaign was behind when what is true is that nobody counted a day of it.
    """
    sourced_three = [_day(when, "aired", 10.0, 7000.0)
                     for when in ("2025-04-27", "2025-04-28", "2025-04-29")]

    complete = board.campaign_row(_campaign(ends_on="2025-04-29", starts_on="2025-04-27"),
                                  sourced_three, date(2025, 4, 29))
    holed = board.campaign_row(_campaign(), sourced_three + _unsourced_remaining(), AS_OF)

    assert complete["rating"]["pace"]["ratio"] is not None
    assert complete["rating"]["pace"]["verdict"] in (words.ON_PACE, words.AT_RISK, words.BEHIND)

    assert holed["rating"]["pace"]["verdict"] == words.UNKNOWN
    assert holed["rating"]["pace"]["ratio"] is None
    assert holed["rating"]["pace"]["code"] == "gap_in_elapsed"
    assert holed["rating"]["pace"]["unsourced_elapsed_days"] == ["2025-04-30"]
    # And the honest counted figure survives: the three sourced days are still summed.
    assert holed["rating"]["counted"]["through_counted_day"] == 30.0


def test_a_flight_with_no_sourced_day_at_all_names_the_missing_feed_not_the_missing_days() -> None:
    """No source anywhere and a hole in a counted window are two different routes.

    One sends the reader to the feed that does not exist. The other sends them to
    the named days that were not counted. A single reason for both would send
    somebody to upload a file they already uploaded.
    """
    nothing = board.campaign_row(_campaign(), [_day(when, "unknown") for when in
                                               ("2025-04-27", "2025-04-28", "2025-04-29", "2025-04-30")], AS_OF)
    holed = board.campaign_row(_campaign(), _full_elapsed()[:3] + [_day("2025-04-30", "unknown")], AS_OF)
    assert nothing["rating"]["pace"]["code"] == "no_source"
    assert holed["rating"]["pace"]["code"] == "gap_in_elapsed"
    assert holed["rating"]["pace"]["unsourced_elapsed_days"] == ["2025-04-30"]
    # Neither states a pace, and neither counts an uncounted day as zero.
    assert nothing["rating"]["pace"]["ratio"] is None
    assert nothing["rating"]["counted"]["through_counted_day"] == 0.0
    assert holed["rating"]["counted"]["through_counted_day"] == 30.0
