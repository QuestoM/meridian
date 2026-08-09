"""The goal-based order: the seam into the engine, and the order the product accepts.

Four things are held here, one per job story.

**A trader books a goal rather than a spot list.** An order that states a
rating-point goal and books no lines is COMPLETE, and the product says so in
words rather than rendering its absent spot list as missing data.

**The optimizer places against that goal.** The seam is exercised in both of its
halves: the ranking weight that folds beside advertiser demand, inventory and
pacing, and the objective wrapper that the refiner and the exact DP tier cannot
optimise back out because they climb it too.

**The channel can say whether it will deliver.** Every figure it cannot derive
comes back as unknown carrying the reason, and never as zero.

**The golden must not move.** The seam is proved inert on the data that is
actually on disk, arithmetically rather than by a flag, so no shipped plan can
change until a real goal-based order is booked.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from kairos.optimize import goal_seam
from kairos.optimize.demand import WEIGHT_CAP, WEIGHT_FLOOR
from kairos.optimize.optimizer import ProgramSegment
from kairos_api import campaigns_commitment, campaigns_goal_order, campaigns_goal_words

ROOT = Path(__file__).resolve().parents[1]
TODAY = date(2025, 5, 1)


def _segment(segment_id: str, tvr: float, *, cpp: float = 100.0, day: str = "2025-05-01",
             channel: str = "רשת 13") -> ProgramSegment:
    return ProgramSegment(
        segment_id=segment_id, channel=channel, day=day,
        start_seconds=20 * 3600.0, duration_seconds=3600.0,
        program_type="Drama", baseline_tvr=tvr, cpp=cpp,
    )


def _order(points: float = 100.0, *, audience: str = "all_viewers",
           starts: str = "2025-05-01", ends: str = "2025-05-05",
           channel: str = "רשת 13") -> goal_seam.GoalOrder:
    return goal_seam.GoalOrder(
        campaign_id="CMP_T1", channel=channel, audience=audience,
        goal_points=points, starts_on=starts, ends_on=ends,
    )


# ---------------------------------------------------------------------------
# The order itself: a goal with no spot list is a complete order
# ---------------------------------------------------------------------------


def test_goal_with_no_spot_list_is_a_complete_order() -> None:
    block = campaigns_goal_order.order_block({"rating_goal_points": 65.0}, [])
    assert block["kind"] == campaigns_goal_words.GOAL_BASED
    assert block["is_complete"] is True
    assert block["carries_spot_list"] is False
    assert block["no_spot_list_en"] and block["no_spot_list_he"]


def test_booked_lines_make_it_a_spot_list_order() -> None:
    flights = [{"goal_kind": "spots", "goal_value": 40}]
    block = campaigns_goal_order.order_block({"rating_goal_points": 65.0}, flights)
    assert block["kind"] == campaigns_goal_words.SPOT_LIST
    assert block["carries_spot_list"] is True


def test_a_flight_stating_an_outcome_is_not_a_spot_list() -> None:
    """A flight booked in GRP states the same outcome the goal does."""
    flights = [{"goal_kind": "grp", "goal_value": 40}]
    block = campaigns_goal_order.order_block({"rating_goal_points": 65.0}, flights)
    assert block["kind"] == campaigns_goal_words.GOAL_BASED


def test_neither_a_goal_nor_a_line_is_named_and_not_rendered_blank() -> None:
    block = campaigns_goal_order.order_block({}, [])
    assert block["kind"] == campaigns_goal_words.NOT_AN_ORDER_YET
    assert block["is_complete"] is False
    assert block["path_forward_en"] and block["path_forward_he"]


def test_every_stored_campaign_carries_an_order_kind() -> None:
    from kairos_api import campaigns_api_store as store

    records = store.campaigns_with_flights(store.load_frame())
    assert records, "the campaigns store is empty, so this piece has nothing to read"
    for record in records:
        assert record["order"]["kind"] in campaigns_goal_words.ORDER_KIND_VALUES


# ---------------------------------------------------------------------------
# Reading the store, and the demo boundary
# ---------------------------------------------------------------------------


def test_demo_rows_never_reach_the_engine() -> None:
    """A seeded row is not a booking, so the default load excludes it."""
    real = goal_seam.load_goal_orders()
    seeded = goal_seam.load_goal_orders(include_demo=True)
    assert all(not order.is_demo for order in real)
    assert len(seeded) >= len(real)


def test_the_seam_is_inert_on_the_data_on_disk() -> None:
    """Nothing on disk is a real goal-based order, so the plan cannot move."""
    orders = goal_seam.load_goal_orders()
    state = goal_seam.seam_state(orders)
    assert state["is_identity"] is True
    segments = [_segment("a", 4.0), _segment("b", 1.0)]
    weights = goal_seam.build_goal_weights(segments, orders, TODAY)
    assert weights == {"a": 1.0, "b": 1.0}


def test_all_viewers_matches_the_commercial_vocabulary() -> None:
    """The engine holds the audience name as a string; it must not drift."""
    assert goal_seam.ALL_VIEWERS == campaigns_commitment.ALL_VIEWERS


# ---------------------------------------------------------------------------
# What the delivery ledger can honestly say
# ---------------------------------------------------------------------------


def test_an_unsourced_day_is_unknown_and_not_zero(tmp_path: Path) -> None:
    path = tmp_path / "delivery.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["campaign_id", "air_state", "rating_points_planned"])
        writer.writerow(["CMP_T1", "aired", "8.7"])
        writer.writerow(["CMP_T1", "unknown", ""])
        writer.writerow(["CMP_T1", "scheduled", "5.0"])
    record = goal_seam.load_delivered_points(path)["CMP_T1"]
    assert record.points_counted == pytest.approx(8.7)
    assert record.days_counted == 1
    assert record.days_unknown == 2
    assert record.complete is False


def test_a_booked_day_is_not_a_delivered_day(tmp_path: Path) -> None:
    """A scheduled day carries a rating and still counts as unknown."""
    path = tmp_path / "delivery.csv"
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["campaign_id", "air_state", "rating_points_planned"])
        writer.writerow(["CMP_T1", "scheduled", "12.0"])
    record = goal_seam.load_delivered_points(path)["CMP_T1"]
    assert record.points_counted == 0.0
    assert record.days_unknown == 1


def test_the_remainder_carries_its_basis() -> None:
    order = _order(100.0)
    full = goal_seam.DeliveredPoints("CMP_T1", 30.0, 3, 0, 3)
    partial = goal_seam.DeliveredPoints("CMP_T1", 30.0, 1, 2, 3)
    assert goal_seam.unmet_points(order, full, TODAY) == (70.0, goal_seam.BASIS_MEASURED)
    assert goal_seam.unmet_points(order, partial, TODAY) == (70.0, goal_seam.BASIS_GAP_IN_ELAPSED)
    assert goal_seam.unmet_points(order, None, TODAY) == (100.0, goal_seam.BASIS_NO_SOURCE)


def test_an_unmeasurable_audience_is_unknown_and_steers_nothing() -> None:
    order = _order(100.0, audience="women_25_54")
    remainder, basis = goal_seam.unmet_points(order, None, TODAY)
    assert remainder is None
    assert basis == goal_seam.BASIS_UNMEASURABLE
    segments = [_segment("a", 4.0), _segment("b", 1.0)]
    assert goal_seam.build_goal_weights(segments, [order], TODAY) == {"a": 1.0, "b": 1.0}


def test_a_flight_with_no_dates_is_unknown() -> None:
    order = _order(100.0, starts="", ends="")
    assert goal_seam.unmet_points(order, None, TODAY)[1] == goal_seam.BASIS_NO_FLIGHT_DATES
    assert goal_seam.days_left(order, TODAY) is None


# ---------------------------------------------------------------------------
# The ranking half of the seam
# ---------------------------------------------------------------------------


def test_a_goal_prefers_rating_efficiency_over_the_day_mean() -> None:
    segments = [_segment("high", 6.0), _segment("low", 2.0)]
    weights = goal_seam.build_goal_weights(segments, [_order(4.0)], TODAY, delivered_of={})
    assert weights["high"] > 1.0 > weights["low"]
    assert goal_seam.GOAL_U_MIN <= weights["low"]
    assert weights["high"] <= goal_seam.GOAL_U_MAX


def test_a_flat_rating_day_is_an_exact_identity() -> None:
    """With no rating-efficiency difference there is nothing for a goal to prefer."""
    segments = [_segment("a", 3.0), _segment("b", 3.0), _segment("c", 3.0)]
    weights = goal_seam.build_goal_weights(segments, [_order(90.0)], TODAY, delivered_of={})
    assert weights == {"a": 1.0, "b": 1.0, "c": 1.0}


def test_another_channel_is_never_steered() -> None:
    """The competitor boundary: an order reaches its own channel and no other."""
    segments = [_segment("mine", 6.0), _segment("theirs", 2.0, channel="קשת 12")]
    weights = goal_seam.build_goal_weights(segments, [_order(4.0)], TODAY, delivered_of={})
    assert weights["theirs"] == 1.0


def test_an_order_with_no_channel_steers_nothing() -> None:
    segments = [_segment("a", 6.0), _segment("b", 2.0)]
    weights = goal_seam.build_goal_weights(
        segments, [_order(4.0, channel="")], TODAY, delivered_of={},
    )
    assert weights == {"a": 1.0, "b": 1.0}


def test_a_finished_flight_steers_nothing() -> None:
    order = _order(100.0, starts="2025-04-01", ends="2025-04-10")
    segments = [_segment("a", 6.0), _segment("b", 2.0)]
    assert goal_seam.days_left(order, TODAY) == 0
    assert goal_seam.build_goal_weights(segments, [order], TODAY, delivered_of={}) == {
        "a": 1.0, "b": 1.0,
    }


def test_the_fold_stays_inside_the_engine_bounds() -> None:
    segments = [_segment("high", 20.0), _segment("low", 0.1)]
    base = {"high": WEIGHT_CAP, "low": WEIGHT_FLOOR}
    folded = goal_seam.fold_into_demand_weights(
        base, segments, TODAY, orders=[_order(10.0)], delivered_of={},
    )
    for value in folded.values():
        assert WEIGHT_FLOOR <= value <= WEIGHT_CAP


def test_the_fold_with_no_reference_date_changes_nothing() -> None:
    segments = [_segment("high", 6.0), _segment("low", 2.0)]
    base = {"high": 1.0, "low": 1.0}
    assert goal_seam.fold_into_demand_weights(
        base, segments, None, orders=[_order(4.0)],
    ) == base


def test_the_fold_with_no_orders_changes_nothing() -> None:
    segments = [_segment("high", 6.0), _segment("low", 2.0)]
    base = {"high": 1.3, "low": 0.7}
    assert goal_seam.fold_into_demand_weights(base, segments, TODAY, orders=[]) == base


# ---------------------------------------------------------------------------
# The objective half of the seam, which the refiner cannot optimise back out
# ---------------------------------------------------------------------------


def _net(segment: ProgramSegment, count: int) -> float:
    """A stand-in net: revenue only, so the goal term is the sole difference."""
    units = segment.break_length_seconds / segment.unit_seconds
    return count * segment.baseline_tvr * segment.cpp * units


def test_the_objective_wrapper_is_the_same_object_when_nothing_is_booked() -> None:
    segments = [_segment("a", 4.0)]
    assert goal_seam.goal_adjusted_net(_net, segments, [], TODAY) is _net
    assert goal_seam.goal_adjusted_net(_net, segments, [_order(4.0)], None) is _net


def test_the_objective_wrapper_values_points_and_not_shekels() -> None:
    """Equal revenue, unequal rating: the goal breaks the tie toward the rating."""
    cheap = _segment("cheap", 8.0, cpp=50.0)
    dear = _segment("dear", 2.0, cpp=200.0)
    assert _net(cheap, 1) == pytest.approx(_net(dear, 1))
    wrapped = goal_seam.goal_adjusted_net(
        _net, [cheap, dear], [_order(10.0)], TODAY, delivered_of={},
    )
    assert wrapped(cheap, 1) > wrapped(dear, 1)


def test_the_objective_wrapper_is_zero_at_zero_shadow() -> None:
    segments = [_segment("a", 8.0), _segment("b", 2.0)]
    wrapped = goal_seam.goal_adjusted_net(
        _net, segments, [_order(10.0)], TODAY, delivered_of={}, shadow=0.0,
    )
    for segment in segments:
        for count in range(4):
            assert wrapped(segment, count) == pytest.approx(_net(segment, count))


def test_the_objective_wrapper_adds_nothing_to_an_empty_segment() -> None:
    segments = [_segment("a", 8.0), _segment("b", 2.0)]
    wrapped = goal_seam.goal_adjusted_net(
        _net, segments, [_order(10.0)], TODAY, delivered_of={},
    )
    assert wrapped(segments[0], 0) == pytest.approx(_net(segments[0], 0))


def test_a_met_goal_stops_adding_value() -> None:
    segments = [_segment("a", 8.0), _segment("b", 2.0)]
    met = {"CMP_T1": goal_seam.DeliveredPoints("CMP_T1", 100.0, 4, 0, 4)}
    wrapped = goal_seam.goal_adjusted_net(
        _net, segments, [_order(100.0)], TODAY, delivered_of=met,
    )
    assert wrapped is _net


# ---------------------------------------------------------------------------
# What the channel says before the flight starts
# ---------------------------------------------------------------------------


def test_pressure_never_exceeds_the_whole_day() -> None:
    orders = [_order(1000.0), _order(1000.0), _order(1000.0)]
    assert goal_seam.day_pressure(10.0, orders, TODAY, "רשת 13", TODAY, {}) == 1.0


def test_feasibility_states_the_share_of_supply() -> None:
    order = _order(100.0, starts="2025-05-01", ends="2025-05-05")
    verdict = goal_seam.goal_feasibility(order, None, TODAY, 100.0)
    assert verdict.days_left == 5
    assert verdict.required_per_day == pytest.approx(20.0)
    assert verdict.share_of_supply == pytest.approx(0.2)
    assert verdict.state == goal_seam.FITS


def test_a_goal_beyond_the_supply_says_so() -> None:
    order = _order(1000.0, starts="2025-05-01", ends="2025-05-05")
    verdict = goal_seam.goal_feasibility(order, None, TODAY, 100.0)
    assert verdict.state == goal_seam.EXCEEDS_SUPPLY
    assert verdict.share_of_supply > 1.0


def test_an_unknown_supply_invents_no_share() -> None:
    order = _order(100.0)
    verdict = goal_seam.goal_feasibility(order, None, TODAY, None)
    assert verdict.state == goal_seam.UNKNOWN
    assert verdict.share_of_supply is None
    assert verdict.supply_per_day is None
    assert verdict.required_per_day == pytest.approx(20.0)


def test_the_read_publishes_the_refusal_the_product_already_owns() -> None:
    from kairos_api import pacing_alerts_api_words as pacing_words

    order = _order(100.0, audience="women_25_54")
    read = campaigns_goal_order.goal_order_read(order, today=TODAY, delivered=None)
    assert read["state"] == campaigns_goal_words.UNKNOWN
    assert read["unavailable"]["reason_en"] == pacing_words.UNMEASURABLE_EN
    assert read["unavailable"]["reason_he"] == pacing_words.UNMEASURABLE_HE


def test_the_read_never_promises_a_delivery() -> None:
    order = _order(100.0)
    read = campaigns_goal_order.goal_order_read(
        order, today=TODAY, delivered=None, supply_per_day=100.0,
    )
    assert read["not_a_promise_en"] == campaigns_goal_words.NOT_A_PROMISE_EN
    assert read["supply_basis_he"] == campaigns_goal_words.SUPPLY_BASIS_HE


def test_a_ceiling_remainder_is_never_reported_as_measured() -> None:
    order = _order(100.0)
    partial = goal_seam.DeliveredPoints("CMP_T1", 30.0, 1, 2, 3)
    read = campaigns_goal_order.goal_order_read(
        order, today=TODAY, delivered=partial, supply_per_day=100.0,
    )
    assert read["basis_en"] == campaigns_goal_words.BASIS_CEILING_EN
    assert read["delivered"]["is_a_floor"] is True


def test_the_remaining_days_are_the_days_the_lean_is_spread_over() -> None:
    order = _order(100.0, starts="2025-05-01", ends="2025-05-05")
    days = goal_seam.remaining_days(order, TODAY)
    assert days == ["2025-05-01", "2025-05-02", "2025-05-03", "2025-05-04", "2025-05-05"]
    assert len(days) == goal_seam.days_left(order, TODAY)


def test_the_supply_read_is_unknown_when_the_plan_does_not_hold_those_days() -> None:
    assert campaigns_goal_order.expected_supply_per_day("רשת 13", ["1999-01-01"]) is None
    assert campaigns_goal_order.expected_supply_per_day("", ["2024-11-04"]) is None


def test_the_whole_read_runs_on_the_data_on_disk() -> None:
    payload = campaigns_goal_order.goal_orders_read(today=TODAY)
    assert payload["seam"]["is_identity"] is True
    assert payload["seam"]["inert_en"]
    assert payload["orders"] == []
    assert payload["vocabularies"]["order_kinds"]
