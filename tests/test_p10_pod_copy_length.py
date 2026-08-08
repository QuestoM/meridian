"""The copy version's own declared length against the booked duration.

JS-7's own done condition: any ad whose booked duration disagrees with its copy
must be impossible to miss. Pinned against the exact break the gap named,
``2025-04-27~22:53:49``: seven spots, and exactly one of them disagrees, spot 3,
copy ``מחליפה - סרט מלא 35'`` booked at 36 s while its own name reads 35. The
other six never falsely disagree, including the two whose copy is a bare number
with no mark (``34``, ``20``, never read as a length at all) and the one whose
copy embeds a number that is not a length (``סרט 15 ימי מכירות``, a count of
sale days on a 14 s spot, at both ``22:53:49`` and ``22:59:40``).
"""

from __future__ import annotations

import pytest

from kairos_api import break_api_pod as pod
from kairos_api.break_api_pod_spots import copy_declared_seconds, copy_length_check

DAY = "2025-04-27"


def _pods():
    days = pod.covered_days()
    if DAY not in days:
        pytest.skip(f"no traffic file covers {DAY}; covered days are {sorted(days)}")
    return pod.pods_for_day(DAY)


def _one(clock: str):
    for record in _pods():
        if record["break_start_clock"] == clock:
            return record
    pytest.skip(f"the traffic file declares no break at {clock} on {DAY}")
    return None


def test_the_parse_rule_reads_a_mark_and_never_a_bare_number():
    assert copy_declared_seconds('מרצדס סטאר ליס 28"') == 28.0
    assert copy_declared_seconds('סרט ראשי 35"') == 35.0
    assert copy_declared_seconds("מחליפה - סרט מלא 35'") == 35.0
    assert copy_declared_seconds("יוגה 10 שניות") == 10.0
    assert copy_declared_seconds("סלייס - 10 שניות") == 10.0
    assert copy_declared_seconds("34") is None
    assert copy_declared_seconds("20") is None
    assert copy_declared_seconds("סרט 15 ימי מכירות") is None
    assert copy_declared_seconds("") is None
    assert copy_declared_seconds(None) is None


def test_the_tri_state_is_exactly_three_states_and_never_a_fourth():
    agrees = copy_length_check({"value": '28"'}, {"seconds": 28.0})
    assert agrees["state"] == "agrees"
    disagrees = copy_length_check({"value": "35'"}, {"seconds": 36.0})
    assert disagrees["state"] == "disagrees"
    assert disagrees["difference_seconds"] == 1.0
    none_state = copy_length_check({"value": "34"}, {"seconds": 34.0})
    assert none_state["state"] == "none"
    assert none_state["reason"] and none_state["reason_he"]
    states = {agrees["state"], disagrees["state"], none_state["state"]}
    assert states == {"agrees", "disagrees", "none"}


def test_the_2253_pod_returns_exactly_one_disagreement_the_named_one():
    record = _one("22:53:49")
    assert record["arithmetic"]["spot_count"] == 7
    disagreeing = [spot for spot in record["spots"] if spot["copy_length"]["state"] == "disagrees"]
    assert len(disagreeing) == 1
    only = disagreeing[0]
    assert only["house_number"]["value"] == "CID178977"
    assert only["advertiser"]["value"] == 'קופ"ח מאוחדת'
    assert only["copy_length"]["copy_seconds"] == 35.0
    assert only["copy_length"]["booked_seconds"] == 36.0
    assert record["copy_length_disagreements"] == 1


def test_the_2259_pod_reads_zero_disagreements_from_the_sale_days_spot():
    record = _one("22:59:40")
    sale_days = [spot for spot in record["spots"] if "ימי מכירות" in (spot["creative"].get("value") or "")]
    assert sale_days, "the fixture no longer carries the sale-days copy this test pins"
    for spot in sale_days:
        assert spot["duration"]["seconds"] == 14.0
        assert spot["copy_length"]["state"] == "none"
    disagreeing = [spot for spot in record["spots"] if spot["copy_length"]["state"] == "disagrees"]
    assert disagreeing == []
    assert record["copy_length_disagreements"] == 0
