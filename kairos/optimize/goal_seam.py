"""The goal-based order seam: the one door a booked goal walks through into the optimizer.

The commercial layer has modelled the goal for a long time. A campaign carries a
rating-point target and the audience it counts against, the pacing board reads it
as a denominator, the delivery ledger settles against it and the make-good ledger
raises a shortfall from it. None of that ever reached the engine that decides
where breaks go. The optimizer maximises revenue net of retention and had never
been told a goal exists.

This module is that missing door, and only that door. It holds no placement
logic, no objective and no greedy step. Its persistence half lives in
:mod:`kairos.optimize.goal_seam_store`, re-exported here so callers import one.

Why a goal moves placement at all
---------------------------------
Revenue is ``cpp * rating_points``. A rating-point goal is ``rating_points``
alone, and those are not the same ordering. A large audience in a cheap daypart
is efficient for a points goal and inefficient for revenue. That divergence is the whole mechanism.

The two halves of the seam
--------------------------
:func:`build_goal_weights` and :func:`fold_into_demand_weights` are the RANKING
half, folding beside advertiser demand, inventory awareness and delivery pacing.
Measured caveat, and it applies to that whole class rather than to the goal: the
F1 refiner and the exact DP tier do not read demand weights, so a bias the greedy
took on is optimised back out wherever they can improve the true objective.

:func:`goal_adjusted_net` is the OBJECTIVE half and the half that survives,
because the greedy, the refiner and the DP tier all climb the scalar it returns.
Measured over 30 real operator-channel days, a goal worth a quarter of a day's
rating moved 53 of 2540 segments and raised delivered goal points by 0.3754
percent. ``docs/goal-based-order-design.md`` carries the contract in full.

Honesty contract
----------------
Every path is an identity no-op until real data lands, and the identity is
arithmetic rather than a flag:

  * No goal orders, from any cause, leaves every weight at exactly 1.0, and
    :func:`goal_adjusted_net` returns the SAME function object.
  * A goal whose audience this product holds no panel for contributes nothing.
  * An order carrying no channel, or a channel that is not the segment's, steers
    nothing. That is the competitor boundary inside this seam.
  * A day whose in-scope segments all carry the same rating is an exact identity.
  * A demo row is not a booking. All 51 stored goals are demo rows, so the seam
    ships provably inert.

Nothing here reads a clock or calls random, and nothing here changes reported
revenue: both halves touch ranking or the objective scalar only.

Weight formula, over the segments of one channel on one day::

    supply    = sum of baseline_tvr
    pressure  = clamp(sum over orders of (unmet / days_left) / supply, 0, 1)
    weight(s) = clamp(1 + K * pressure * (tvr(s) / (supply / count) - 1), U_MIN, U_MAX)

The ``(ratio - 1)`` term is what makes the lean differential; a uniform
multiplier on a day changes no ranking and would be a silent no-op.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Optional, Sequence

from kairos.optimize.optimizer import ProgramSegment

# The persistence half of this seam, re-exported so a caller imports one module.
from kairos.optimize.goal_seam_store import (  # noqa: F401
    ALL_VIEWERS,
    BASIS_BOOKED,
    BASIS_GAP_IN_ELAPSED,
    BASIS_MEASURED,
    BASIS_NO_FLIGHT_DATES,
    BASIS_NO_GOAL,
    BASIS_NO_SOURCE,
    BASIS_UNMEASURABLE,
    DEFAULT_CAMPAIGNS_PATH,
    DEFAULT_DELIVERY_PATH,
    DeliveredPoints,
    GoalOrder,
    days_left,
    load_delivered_points,
    load_goal_orders,
    remaining_days,
    unmet_points,
)
from kairos.optimize.goal_seam_store import _parse_iso

# How hard a unit of goal pressure leans, and the bounds the lean lives in. K is
# the strength; U_MAX caps a boost so one large goal cannot own a whole day;
# U_MIN floors the penalty so a low-rating segment is de-prioritised for a points
# goal and never forbidden, which would suppress revenue by the back door.
GOAL_K = 1.0
GOAL_U_MAX = 2.0
GOAL_U_MIN = 0.5

# How much a committed rating point is worth in the objective, as a multiple of
# the day's own mean CPP. 1.0 says a point the channel has promised is worth one
# point of ordinary airtime on top of what that airtime already earns, which is
# what a make-good actually costs to settle. Zero disables the objective half of
# the seam and leaves the net untouched.
GOAL_SHADOW = 1.0

# Feasibility verdicts for the pre-flight question "will the channel deliver
# this". They are about SUPPLY, not about a promise: the product has no
# per-campaign spot allocation in the weekly plan, so it can state whether the
# goal fits inside the channel's expected rating and must not state a delivery it
# cannot derive.
FITS = "fits"
TIGHT = "tight"
EXCEEDS_SUPPLY = "exceeds_supply"
UNKNOWN = "unknown"

# Where the tight band sits, as a share of the channel's expected rating on the
# flight's remaining days. Published so a surface can quote it.
TIGHT_SHARE = 0.5

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True)
class GoalFeasibility:
    """Whether a booked goal fits inside the channel's expected rating supply.

    ``required_per_day`` is the points the order still has to place, spread over
    the broadcast days it has left. ``supply_per_day`` is the channel's own
    expected rating on a day of that window. ``share_of_supply`` is the first
    divided by the second, and it is the number a trader and a planner can both
    read: a goal at 0.2 wants a fifth of everything the channel has that day.

    Every figure is ``None`` when the state is ``unknown``, never zero.
    """

    campaign_id: str
    state: str
    basis: str
    required_per_day: Optional[float] = None
    supply_per_day: Optional[float] = None
    share_of_supply: Optional[float] = None
    days_left: Optional[int] = None
    unmet_points: Optional[float] = None


def goal_feasibility(
    order: GoalOrder,
    delivered: Optional[DeliveredPoints],
    today: date,
    supply_per_day: Optional[float],
) -> GoalFeasibility:
    """Whether the goal fits inside the channel's expected rating on the days it has left.

    This is the pre-flight answer to "will you deliver this", stated in the only
    currency the product can honestly derive it in. The product holds no
    per-campaign spot allocation in the weekly plan, so it cannot say how many
    points THIS campaign will get. It can say what the goal needs per day and
    what the channel has per day, and whether the first fits inside the second.

    ``supply_per_day`` of ``None`` or zero means the channel's expected rating on
    those days is not known, so the verdict is ``unknown`` and no share is
    invented.
    """
    remainder, basis = unmet_points(order, delivered, today)
    if remainder is None:
        return GoalFeasibility(order.campaign_id, UNKNOWN, basis)
    left = days_left(order, today) or 0
    if left <= 0:
        return GoalFeasibility(
            order.campaign_id, UNKNOWN, basis, days_left=0, unmet_points=remainder,
        )
    required = remainder / left
    if supply_per_day is None or supply_per_day <= 0:
        return GoalFeasibility(
            order.campaign_id, UNKNOWN, basis,
            required_per_day=round(required, 4), days_left=left, unmet_points=remainder,
        )
    share = required / supply_per_day
    if share > 1.0:
        state = EXCEEDS_SUPPLY
    elif share >= TIGHT_SHARE:
        state = TIGHT
    else:
        state = FITS
    return GoalFeasibility(
        campaign_id=order.campaign_id,
        state=state,
        basis=basis,
        required_per_day=round(required, 4),
        supply_per_day=round(float(supply_per_day), 4),
        share_of_supply=round(share, 4),
        days_left=left,
        unmet_points=remainder,
    )


def _order_covers_day(order: GoalOrder, day: date, channel: str) -> bool:
    """Whether one order wants placement on this channel on this day.

    The channel test is the competitor boundary in this seam. An order carrying
    no channel matches nothing, because a booking made while no operator channel
    was configured cannot be assumed to be about the channel in front of us.
    """
    if not order.channel or order.channel.strip() != str(channel or "").strip():
        return False
    starts = _parse_iso(order.starts_on)
    ends = _parse_iso(order.ends_on)
    if starts is None or ends is None:
        return False
    return starts <= day <= ends


def day_pressure(
    supply: float,
    orders: Sequence[GoalOrder],
    day: date,
    channel: str,
    today: date,
    delivered_of: Mapping[str, DeliveredPoints],
) -> float:
    """The share of one day's whole expected rating the booked goals still need.

    Summed over every order whose window and channel cover this day, and clamped
    into ``[0, 1]``: three orders each wanting a fifth of the day read as
    three-fifths of pressure, and no set of orders can read as more than the
    whole day. An order whose remainder cannot be stated contributes nothing,
    because an unknown may not be spent as though it were a number.
    """
    if supply <= 0.0:
        return 0.0
    pressure = 0.0
    for order in orders:
        if not _order_covers_day(order, day, channel):
            continue
        remainder, _ = unmet_points(order, delivered_of.get(order.campaign_id), today)
        if remainder is None or remainder <= 0.0:
            continue
        left = days_left(order, today) or 0
        if left <= 0:
            continue
        pressure += (remainder / left) / supply
    return _clamp(pressure, 0.0, 1.0)


def build_goal_weights(
    segments: Iterable[ProgramSegment],
    orders: Optional[Sequence[GoalOrder]],
    today: date,
    *,
    delivered_of: Optional[Mapping[str, DeliveredPoints]] = None,
    k: float = GOAL_K,
    u_max: float = GOAL_U_MAX,
    u_min: float = GOAL_U_MIN,
) -> dict[str, float]:
    """Per-segment goal-based placement weights by segment_id, 1.0 when nothing is booked.

    The signal is TWO-SIDED, like delivery pacing: for a rating-point goal a
    segment above its day's mean expected rating is preferred and one below it is
    de-prioritised, in proportion to how much of the day's whole rating the
    booked goals still need. It touches ranking only and is never charged, so no
    campaign can be billed more for being leaned toward.

    ``orders`` of ``None`` or ``[]`` returns every weight at 1.0, a pure identity
    no-op. So does a day whose in-scope segments all carry the same expected
    rating, and so does a goal whose audience carries no panel.
    """
    seg_list = list(segments)
    weights: dict[str, float] = {seg.segment_id: 1.0 for seg in seg_list}
    if not orders:
        return weights
    delivered_of = delivered_of or {}
    groups: dict[tuple[str, str], list[ProgramSegment]] = defaultdict(list)
    for segment in seg_list:
        groups[(str(segment.day), str(segment.channel))].append(segment)
    for (day_text, channel), group in groups.items():
        day = _parse_iso(day_text)
        if day is None:
            continue
        supply = sum(max(0.0, float(seg.baseline_tvr)) for seg in group)
        if supply <= 0.0:
            continue
        mean_tvr = supply / len(group)
        if mean_tvr <= 0.0:
            continue
        pressure = day_pressure(supply, orders, day, channel, today, delivered_of)
        if pressure <= 0.0:
            continue
        for segment in group:
            ratio = max(0.0, float(segment.baseline_tvr)) / mean_tvr
            weights[segment.segment_id] = _clamp(
                1.0 + k * pressure * (ratio - 1.0), u_min, u_max
            )
    return weights


def fold_into_demand_weights(
    weights: Mapping[str, float],
    segments: Iterable[ProgramSegment],
    today: Optional[date],
    *,
    orders: Optional[Sequence[GoalOrder]] = None,
    delivered_of: Optional[Mapping[str, DeliveredPoints]] = None,
    k: float = GOAL_K,
    u_max: float = GOAL_U_MAX,
    u_min: float = GOAL_U_MIN,
) -> dict[str, float]:
    """Multiply the goal lean onto an already-folded demand weight map.

    This is the whole call site. :func:`kairos.optimize.demand.build_demand_weights`
    folds advertiser demand, inventory awareness and delivery pacing and clamps
    the product into ``[WEIGHT_FLOOR, WEIGHT_CAP]``; this applies the goal signal
    on top and re-clamps into the same global bounds, so the goal composes with
    the other three exactly as they compose with each other and no signal can
    push a slot outside the bounds the engine already guarantees.

    ``today`` of ``None`` returns the map unchanged, because a goal's remainder
    and its remaining days are both dated and the honest answer with no reference
    date is to steer nothing. ``orders`` of ``None`` loads the real, non-demo
    goal orders from the campaigns store; pass an explicit list to steer with a
    caller's own set, and an empty list to prove the identity.
    """
    from kairos.optimize.demand import WEIGHT_CAP, WEIGHT_FLOOR

    base = dict(weights)
    if today is None:
        return base
    if orders is None:
        orders = load_goal_orders()
    if not orders:
        return base
    seg_list = list(segments)
    if delivered_of is None:
        delivered_of = load_delivered_points()
    lean = build_goal_weights(
        seg_list, orders, today,
        delivered_of=delivered_of, k=k, u_max=u_max, u_min=u_min,
    )
    for segment_id, factor in lean.items():
        if factor == 1.0:
            continue
        base[segment_id] = _clamp(
            base.get(segment_id, 1.0) * factor, WEIGHT_FLOOR, WEIGHT_CAP
        )
    return base


def goal_adjusted_net(
    net_of: Callable[[ProgramSegment, int], float],
    segments: Iterable[ProgramSegment],
    orders: Optional[Sequence[GoalOrder]],
    today: Optional[date],
    *,
    delivered_of: Optional[Mapping[str, DeliveredPoints]] = None,
    shadow: float = GOAL_SHADOW,
) -> Callable[[ProgramSegment, int], float]:
    """Wrap the per-segment net so the engine's OBJECTIVE can see a committed goal.

    The weight map is a ranking bias and the refiner does not read it, so a
    placement bias the greedy took on can be optimised straight back out. This is
    the other half of the seam and the half that survives: it returns a function
    of exactly the shape :func:`kairos.optimize.revenue_net.segment_net_revenue`
    has, so it threads into the greedy step, the F1 refiner and the exact DP tier
    through the one ``net_of`` parameter they already share, and every one of
    them climbs the same adjusted scalar.

    What it adds, and why it is not a fabricated shekel::

        adjusted(segment, k) = net(segment, k)
                             + shadow * pressure(day) * price(day) * points(segment, k)

    ``points`` is the rating the segment's ``k`` breaks carry, in the same
    thirty-second units revenue is quoted in. ``price`` is the day's own
    points-weighted mean CPP, read from the segments in front of it rather than
    from a rate nobody supplied, and it stands for what the channel would have to
    give away to make good a point it committed to and missed. ``pressure`` is
    the share of the day's rating the booked goals still need, so the term
    vanishes the moment the goals are met and the objective is the untouched net
    again.

    The term is proportional to POINTS while the net is proportional to SHEKELS,
    which is the whole reason it moves anything: a large audience in a cheap
    daypart is worth more to a points goal than to a revenue plan, and this is
    where the product finally says so.

    It changes the objective and never the reported revenue. Revenue is built
    from :func:`~kairos.optimize._segment_math._segment_revenue` in
    ``_build_result`` and does not pass through here, so no campaign is charged a
    shekel more for having been leaned toward.

    With no orders, no reference date or zero pressure this returns ``net_of``
    itself, so the identity is the same object and not merely the same numbers.
    """
    rows = list(orders or [])
    if today is None or not rows:
        return net_of
    delivered_of = delivered_of or {}
    pressure_of: dict[tuple[str, str], float] = {}
    price_of: dict[tuple[str, str], float] = {}
    groups: dict[tuple[str, str], list[ProgramSegment]] = defaultdict(list)
    for segment in segments:
        groups[(str(segment.day), str(segment.channel))].append(segment)
    for key, group in groups.items():
        day = _parse_iso(key[0])
        if day is None:
            continue
        supply = sum(max(0.0, float(seg.baseline_tvr)) for seg in group)
        pressure = day_pressure(supply, rows, day, key[1], today, delivered_of)
        if pressure <= 0.0:
            continue
        priced = sum(
            max(0.0, float(seg.baseline_tvr)) * max(0.0, float(seg.cpp)) for seg in group
        )
        if priced <= 0.0:
            continue
        pressure_of[key] = pressure
        price_of[key] = priced / supply
    if not pressure_of:
        return net_of

    def adjusted(segment: ProgramSegment, count: int) -> float:
        base = net_of(segment, count)
        key = (str(segment.day), str(segment.channel))
        pressure = pressure_of.get(key, 0.0)
        price = price_of.get(key, 0.0)
        if pressure <= 0.0 or price <= 0.0 or count <= 0:
            return base
        unit = float(segment.unit_seconds) or 1.0
        units = float(segment.break_length_seconds) / unit
        points = count * max(0.0, float(segment.baseline_tvr)) * units
        return base + shadow * pressure * price * points

    return adjusted


def seam_state(orders: Optional[Sequence[GoalOrder]]) -> dict[str, object]:
    """What this seam can currently see, for a surface or a report to quote.

    Says how many goal orders reached the engine and how many of them carry an
    audience the product can count against, so nobody has to infer from a flat
    weight map whether the seam is wired or merely quiet.
    """
    rows = list(orders or [])
    return {
        "orders": len(rows),
        "measurable_orders": sum(1 for order in rows if order.measurable),
        "goal_points": round(sum(order.goal_points for order in rows), 4),
        "is_identity": len(rows) == 0,
    }
