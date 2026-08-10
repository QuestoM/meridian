"""The goal-based order as the product reports it: what kind it is, and what it will take.

Written beside :mod:`kairos_api.campaigns_api_store`, which persists the goal,
and it does not persist anything itself. Two questions live here.

**What kind of order is this.** An order that states a rating-point goal and
books no lines is a goal-based order, and it is COMPLETE. That is the whole
point of it: the agency states the outcome and the channel owns the placement.
Every surface that treats a missing spot list as missing data on such an order
has misread it, so the kind is published on the record rather than inferred.

**What will this goal take, and on what basis.** The honest answer is a SUPPLY
verdict. The weekly plan holds break counts and not per-campaign lines, so the
product cannot say how many rating points this one order will receive. It can
say what the goal still needs on each broadcast day it has left, what the channel
expects to have on those days, and whether the first fits inside the second. Any
figure it cannot derive is reported as unknown with the reason, never as zero.

The refusals are read from :mod:`kairos_api.pacing_alerts_api_words`, which
already publishes the product's sentence for an unmeasurable audience, a flight
with no dates and a gap in the elapsed days. The sentences that are new to the
goal-based order come from :mod:`kairos_api.campaigns_goal_words`. Nothing here
composes a sentence of its own.

The engine side of the seam is :mod:`kairos.optimize.goal_seam`, which is the one
door a booked goal walks through into the placement engine. Both sides publish
the contract; it is written out in ``docs/goal-based-order-design.md``.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from kairos.optimize import goal_seam
from kairos_api import campaigns_goal_words as words
from kairos_api import pacing_alerts_api_words as pacing_words

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"

# A booked line is a flight that names spots or seconds. A flight naming rating
# points, GRP or impressions is another way of stating the same outcome goal, so
# it does not turn a goal-based order into a spot-list one.
LINE_GOAL_KINDS = frozenset({"spots", "seconds"})


def _positive(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def carries_booked_lines(flights: Optional[Iterable[Mapping[str, Any]]]) -> bool:
    """Whether any flight on this campaign books lines rather than an outcome."""
    for flight in flights or ():
        kind = str(flight.get("goal_kind", "") or "").strip().lower()
        if kind in LINE_GOAL_KINDS and _positive(flight.get("goal_value")) is not None:
            return True
    return False


def order_kind(
    commitment: Optional[Mapping[str, Any]],
    flights: Optional[Iterable[Mapping[str, Any]]],
) -> str:
    """Which of the three kinds of order this campaign is.

    A campaign that books lines is a spot-list order whatever else it carries,
    because those lines are what the channel is accountable for. A campaign that
    books no lines and states a rating-point goal is a goal-based order. A
    campaign that does neither is not an order yet, which is a state the product
    names rather than a blank it renders.
    """
    if carries_booked_lines(flights):
        return words.SPOT_LIST
    if _positive((commitment or {}).get("rating_goal_points")) is not None:
        return words.GOAL_BASED
    return words.NOT_AN_ORDER_YET


def order_block(
    commitment: Optional[Mapping[str, Any]],
    flights: Optional[Iterable[Mapping[str, Any]]],
) -> dict[str, Any]:
    """How the product reports this campaign's order kind and its completeness.

    A goal-based order is complete with no spot list, and the block says so in
    both languages so no surface can render its absent lines as missing data. A
    campaign that is not an order yet carries the one path forward that makes it
    one, and no invented figure.
    """
    kind = order_kind(commitment, flights)
    entry = words.order_kind_entry(kind)
    block: dict[str, Any] = {
        "kind": kind,
        "label_en": entry["label_en"],
        "label_he": entry["label_he"],
        "meaning_en": entry["meaning_en"],
        "meaning_he": entry["meaning_he"],
        "carries_spot_list": kind == words.SPOT_LIST,
        "is_complete": kind != words.NOT_AN_ORDER_YET,
    }
    if kind == words.GOAL_BASED:
        block["no_spot_list_en"] = words.NO_SPOT_LIST_EN
        block["no_spot_list_he"] = words.NO_SPOT_LIST_HE
        block["means_what_it_says_en"] = words.MEANS_WHAT_IT_SAYS_EN
        block["means_what_it_says_he"] = words.MEANS_WHAT_IT_SAYS_HE
    if kind == words.NOT_AN_ORDER_YET:
        block["path_forward_en"] = words.COMPLETE_PATH_EN
        block["path_forward_he"] = words.COMPLETE_PATH_HE
    return block


def expected_supply_per_day(
    channel: str,
    days: Sequence[str],
    *,
    path: Optional[str | Path] = None,
) -> Optional[float]:
    """The channel's mean expected rating on those broadcast days, or None.

    Read from the weekly plan export, summing each segment's ``baseline_tvr``
    once per segment and averaging over the days that the plan actually holds.
    Days the plan does not reach contribute nothing and are not counted in the
    average, so a window half outside the plan reports the supply of the half
    inside it rather than a figure diluted by days nobody planned.

    Returns ``None`` when the plan holds none of those days, which is an unknown
    supply and must be reported as one.
    """
    target = Path(path) if path is not None else DEFAULT_PLAN_PATH
    wanted = {str(day).strip() for day in days if str(day).strip()}
    if not wanted or not target.exists():
        return None
    channel_text = str(channel or "").strip()
    if not channel_text:
        return None
    seen: set[tuple[str, str]] = set()
    per_day: dict[str, float] = {}
    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return None
        for row in reader:
            if str(row.get("channel", "") or "").strip() != channel_text:
                continue
            day = str(row.get("date", "") or "").strip()
            if day not in wanted:
                continue
            segment_id = str(row.get("segment_id", "") or "").strip()
            key = (day, segment_id)
            if segment_id and key in seen:
                continue
            seen.add(key)
            try:
                tvr = float(row.get("baseline_tvr") or 0.0)
            except (TypeError, ValueError):
                continue
            per_day[day] = per_day.get(day, 0.0) + max(0.0, tvr)
    if not per_day:
        return None
    return sum(per_day.values()) / len(per_day)


def goal_order_read(
    order: goal_seam.GoalOrder,
    *,
    today: Optional[date] = None,
    delivered: Optional[goal_seam.DeliveredPoints] = None,
    supply_per_day: Optional[float] = None,
    plan_path: Optional[str | Path] = None,
) -> dict[str, Any]:
    """What this goal will take from the channel, and what that answer rests on.

    The read is the pre-flight commitment check a goal-based order needs: before
    the flight starts, the channel states whether the goal fits inside its own
    expected rating, and states honestly when it cannot. It promises no delivery,
    because the product has no per-campaign allocation to promise from, and it
    says that in the payload rather than leaving a reader to assume otherwise.

    ``today`` defaults to the calendar day, and every caller that needs a frozen
    answer passes its own reference date instead. ``supply_per_day`` is read from
    the weekly plan when it is not supplied.
    """
    reference = today or date.today()
    if delivered is None:
        delivered = goal_seam.load_delivered_points().get(order.campaign_id)
    days = goal_seam.remaining_days(order, reference)
    if supply_per_day is None and days:
        supply_per_day = expected_supply_per_day(order.channel, days, path=plan_path)
    verdict = goal_seam.goal_feasibility(order, delivered, reference, supply_per_day)
    entry = words.feasibility_entry(verdict.state)

    payload: dict[str, Any] = {
        "campaign_id": order.campaign_id,
        "channel": order.channel,
        "audience": order.audience,
        "goal_points": order.goal_points,
        "starts_on": order.starts_on,
        "ends_on": order.ends_on,
        "state": verdict.state,
        "label_en": entry["label_en"],
        "label_he": entry["label_he"],
        "meaning_en": entry["meaning_en"],
        "meaning_he": entry["meaning_he"],
        "unmet_points": verdict.unmet_points,
        "days_left": verdict.days_left,
        "required_per_day": verdict.required_per_day,
        "supply_per_day": verdict.supply_per_day,
        "share_of_supply": verdict.share_of_supply,
        "supply_basis_en": words.SUPPLY_BASIS_EN,
        "supply_basis_he": words.SUPPLY_BASIS_HE,
        "not_a_promise_en": words.NOT_A_PROMISE_EN,
        "not_a_promise_he": words.NOT_A_PROMISE_HE,
    }
    payload.update(words.basis_words(verdict.basis))
    refusal = pacing_words.reason(verdict.basis)
    if refusal.get("code"):
        payload["unavailable"] = refusal
    if verdict.state == words.EXCEEDS_SUPPLY:
        payload["path_forward_en"] = words.EXCEEDS_PATH_EN
        payload["path_forward_he"] = words.EXCEEDS_PATH_HE
    if delivered is not None:
        payload["delivered"] = {
            "points_counted": delivered.points_counted,
            "days_counted": delivered.days_counted,
            "days_unknown": delivered.days_unknown,
            "is_a_floor": not delivered.complete,
        }
    return payload


def goal_orders_read(
    *,
    today: Optional[date] = None,
    include_demo: bool = False,
    plan_path: Optional[str | Path] = None,
    channel: str = "",
) -> dict[str, Any]:
    """Every goal-based order the engine can see, and what the seam is doing with them.

    ``include_demo`` is false by default and matches the seam exactly, so this
    read answers the question a planner actually asks, which is what is steering
    the plan rather than what is stored. A demo row is not a booking.
    """
    reference = today or date.today()
    orders = goal_seam.load_goal_orders(include_demo=include_demo)
    if channel:
        orders = [order for order in orders if order.channel == channel]
    delivered_of = goal_seam.load_delivered_points()
    state = goal_seam.seam_state(orders)
    return {
        "as_of": reference.isoformat(),
        "scope": {"channel": channel, "scoped": bool(channel)},
        "seam": {**state, **words.seam_words(bool(state["is_identity"]))},
        "orders": [
            goal_order_read(
                order,
                today=reference,
                delivered=delivered_of.get(order.campaign_id),
                plan_path=plan_path,
            )
            for order in orders
        ],
        "vocabularies": words.vocabularies(),
    }
