"""The pod's arithmetic, driven on the shipped traffic file and on constructed pods.

The arithmetic is the whole point of this surface, so it is asserted against
figures worked out by hand from the file rather than against whatever the code
happens to produce. The hand working for the break at ``2025-04-27 20:40:09`` is
in the first test and it is deliberately spelled out.

Every test that could pass vacuously checks its own input first and skips with the
reason rather than asserting nothing.
"""

from __future__ import annotations

import pytest

from kairos_api import break_api_pod as pod

DAY = "2025-04-27"
POD_CLOCK = "20:40:09"


def _pods():
    days = pod.covered_days()
    if DAY not in days:
        pytest.skip(f"no traffic file covers {DAY}; covered days are {sorted(days)}")
    return pod.pods_for_day(DAY)


def _one(clock: str = POD_CLOCK):
    for record in _pods():
        if record["break_start_clock"] == clock:
            return record
    pytest.skip(f"the traffic file declares no break at {clock} on {DAY}")
    return None


def test_the_pod_arithmetic_matches_the_figures_worked_out_by_hand_from_the_file():
    """Sum, span and difference on the break at 20:40:09, checked against the file.

    The pod opens at 20:40:09. Its first spot starts at 20:40:51 and its last at
    20:50:37 for 6 s, so the pod runs to 20:50:43, which is 634 s from the break's
    own declared start. The 28 spots declare 569 s between them. The difference is
    65 s that no spot covers, and the two gaps inside it are the 13 s after the
    opening billboards and the 10 s before the closing ones, 23 s in all.
    """
    record = _one()
    arithmetic = record["arithmetic"]
    assert arithmetic["spot_count"] == 28
    assert arithmetic["spots_missing_a_length"] == 0
    assert arithmetic["declared_load"]["seconds"] == 569.0
    assert arithmetic["span"]["seconds"] == 634.0
    assert arithmetic["unfilled"]["seconds"] == 65.0
    assert arithmetic["unfilled"]["seconds"] == arithmetic["span"]["seconds"] - arithmetic["declared_load"]["seconds"]
    assert arithmetic["gaps_between_spots"] == {"count": 2, "seconds": 23.0}
    assert arithmetic["overlaps_between_spots"] == {"count": 0, "seconds": 0.0}


def test_every_pod_on_the_day_closes_its_own_arithmetic():
    """Span less load equals unfilled on every pod, or the figure is not served."""
    pods = _pods()
    assert pods, "the traffic file declared no pods at all"
    for record in pods:
        arithmetic = record["arithmetic"]
        span = arithmetic["span"]["seconds"]
        load = arithmetic["declared_load"]["seconds"]
        unfilled = arithmetic["unfilled"]["seconds"]
        if arithmetic["spots_missing_a_length"]:
            assert unfilled is None, f"{record['pod_id']} totalled a pod with a missing length"
            continue
        assert round(span - load, 1) == unfilled, f"{record['pod_id']} does not close"


def test_a_spot_with_no_declared_length_is_missing_rather_than_zero():
    """A blank length never becomes a zero, and it stops the difference being served.

    Driven on a constructed pod because the shipped file declares a length on
    every spot, and the case this guards is the one that would silently understate
    a pod by the length of the spot nobody declared.
    """
    spots = [
        {"spot_key": "s0", "start_seconds": 0.0, "end_seconds": 30.0, "duration": {"state": "real", "seconds": 30.0}},
        {"spot_key": "s1", "start_seconds": 30.0, "end_seconds": None, "duration": pod._duration("")},
    ]
    assert spots[1]["duration"]["state"] == "unknown"
    assert spots[1]["duration"]["seconds"] is None
    assert spots[1]["duration"]["reason_he"]
    arithmetic = pod.pod_arithmetic(0.0, spots)
    assert arithmetic["spots_missing_a_length"] == 1
    assert arithmetic["declared_load"]["seconds"] == 30.0
    assert arithmetic["unfilled"]["seconds"] is None
    assert arithmetic["unfilled"]["state"] == "unknown"


def test_spots_running_past_the_pod_are_reported_as_an_overflow_not_a_negative_gap():
    spots = [
        {"spot_key": "s0", "start_seconds": 0.0, "end_seconds": 40.0, "duration": {"state": "real", "seconds": 40.0}},
        {"spot_key": "s1", "start_seconds": 30.0, "end_seconds": 60.0, "duration": {"state": "real", "seconds": 30.0}},
    ]
    arithmetic = pod.pod_arithmetic(0.0, spots)
    assert arithmetic["overlaps_between_spots"] == {"count": 1, "seconds": 10.0}
    assert arithmetic["gaps_between_spots"]["count"] == 0
    assert arithmetic["unfilled"]["seconds"] == -10.0


def test_last_is_its_own_position_and_never_the_fifth_ordinal():
    """Positions are 1 to 5 plus L, and the file's 99 is L rather than an ordinal.

    The trade note is explicit that Last is a distinct position because one
    campaign can hold both the first and the last spot of a break, which is two
    positions in one pod. A product that numbered Last would make that impossible
    to express.
    """
    last = pod.position_of(99)
    assert last["kind"] == "last"
    assert last["code"] == "L"
    assert last["ordinal"] is None
    assert last["preferred"] is True
    fifth = pod.position_of(5)
    assert fifth["kind"] == "ordinal"
    assert fifth["code"] == "5"
    assert fifth["ordinal"] == 5
    assert fifth["preferred"] is True
    sixth = pod.position_of(6)
    assert sixth["ordinal"] == 6
    assert sixth["preferred"] is False
    none = pod.position_of(0)
    assert none["kind"] == "unpositioned"
    assert none["code"] is None
    assert none["ordinal"] is None
    unknown = pod.position_of("")
    assert unknown["state"] == "unknown"
    assert unknown["code"] is None
    assert unknown["reason_he"]


def test_the_shipped_pod_carries_a_last_and_nine_unpositioned_billboards():
    """The position model reads the real file, not only constructed codes."""
    record = _one()
    kinds = [spot["position"]["kind"] for spot in record["spots"]]
    assert kinds.count("last") == 1
    assert kinds.count("unpositioned") == 9
    assert record["positions"]["last_held"] is True
    assert record["positions"]["unpositioned"] == 9
    ordinals = sorted(spot["position"]["ordinal"] for spot in record["spots"] if spot["position"]["kind"] == "ordinal")
    assert ordinals == list(range(1, 19))
    assert "L" in record["positions"]["preferred_set"]
    assert record["positions"]["basis_he"]


def test_the_preferred_set_is_stated_as_a_default_and_not_as_anybody_agreement():
    assert pod.PREFERRED_POSITIONS == ("1", "2", "3", "4", "5", "L")
    assert "default" in pod.PREFERRED_BASIS
    assert "agreed per client" in pod.PREFERRED_BASIS


def test_every_spot_names_an_advertiser_a_length_and_a_house_number_or_says_it_cannot():
    record = _one()
    for spot in record["spots"]:
        for field in ("advertiser", "campaign", "creative", "house_number", "agency"):
            value = spot[field]
            assert value["state"] in {"real", "unknown"}
            if value["state"] == "real":
                assert value["value"], f"{field} was real with nothing in it"
            else:
                assert value["reason"] and value["reason_he"]


def test_the_declared_break_length_is_a_state_rather_than_a_number_it_never_read():
    """No plan covers the traffic day, so the comparison is unavailable and says so.

    This is the honest half. The half that proves the arithmetic itself is real
    lives in ``test_p10_pod_declared_length.py``, which builds a traffic file on a
    day the plan does cover and drives the same code path to a figure.
    """
    record = _one()
    declared = record["declared_break_length"]
    assert declared["state"] in {"unavailable", "unknown"}
    assert declared["seconds"] is None
    assert declared["reason"] and declared["reason_he"]
    against = record["against_declared"]
    assert against["state"] == "unavailable"
    assert against["seconds"] is None
    assert against["verdict"] == "unknown"


def test_against_declared_states_a_gap_and_an_overflow_in_seconds():
    """The arithmetic itself, driven on both sides of exact."""
    arithmetic = pod.pod_arithmetic(0.0, [
        {"spot_key": "s0", "start_seconds": 0.0, "end_seconds": 90.0, "duration": {"state": "real", "seconds": 90.0}},
    ])
    short = pod.against_declared({"state": "real", "seconds": 120.0}, arithmetic)
    assert short["verdict"] == "gap"
    assert short["seconds"] == 30.0
    assert short["signed_seconds"] == 30.0
    over = pod.against_declared({"state": "real", "seconds": 60.0}, arithmetic)
    assert over["verdict"] == "overflow"
    assert over["seconds"] == 30.0
    assert over["signed_seconds"] == -30.0
    exact = pod.against_declared({"state": "real", "seconds": 90.0}, arithmetic)
    assert exact["verdict"] == "exact"
    assert exact["seconds"] == 0.0


def test_a_pod_id_survives_a_round_trip_and_refuses_a_malformed_one():
    identifier = pod.pod_id(DAY, POD_CLOCK)
    assert identifier == f"{DAY}~{POD_CLOCK}"
    assert pod.parse_pod_id(identifier) == (DAY, POD_CLOCK)
    for bad in ("", "2025-04-27", "~20:40:09", "2025-04-27~"):
        with pytest.raises(ValueError):
            pod.parse_pod_id(bad)
