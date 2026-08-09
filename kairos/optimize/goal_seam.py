"""The goal-based order seam: the one door a booked goal walks through into the optimizer.

The commercial layer has modelled the goal for a long time. A campaign carries a
rating-point target and the audience it counts against, the pacing board reads it
as a denominator, the delivery ledger settles against it and the make-good ledger
raises a shortfall from it. None of that ever reached the engine that decides
where breaks go. The optimizer maximises revenue net of retention and has never
been told a goal exists.

This module is that missing door, and only that door. It holds no placement
logic, no objective and no greedy step. It reads what was booked, reads what the
delivery ledger can honestly say was delivered, and returns one per-segment
weight map in exactly the shape :func:`kairos.optimize.demand.build_demand_weights`
already folds. The engine keeps every decision it had; it simply gains one more
signal, on the same footing as advertiser demand, inventory awareness and
delivery pacing.

Why a goal moves placement at all
---------------------------------
Revenue is ``cpp * rating_points``. A rating-point goal is ``rating_points``
alone. Those two orderings are not the same ordering. A segment with a large
audience in a cheap daypart is efficient for a points goal and inefficient for
revenue; an expensive prime segment is the reverse. So a channel that has
committed to points leans toward rating per second of inventory, while a channel
optimising revenue leans toward shekels per second. The seam expresses exactly
that divergence and nothing else.

Honesty contract
----------------
Every path here is an identity no-op until real data lands, and the identity is
arithmetic rather than a flag:

  * No goal orders, from any cause, gives every weight exactly 1.0.
  * A goal whose audience this product holds no panel for contributes no
    pressure, because progress against it is unknown and an unknown may not be
    spent as though it were zero.
  * A day on which every in-scope segment carries the same expected rating gives
    every weight exactly 1.0, because there is no rating-efficiency difference
    for a points goal to prefer.
  * A demo row is not a booking. :func:`load_goal_orders` excludes demo rows by
    default, so a seeded campaign can never steer a real plan.

Measured on this tree: of the 52 campaign rows on disk, 51 carry a rating goal
and all 51 are demo rows. So ``load_goal_orders()`` returns an empty list today
and this seam is provably inert on the real plan, the same way the pacing signal
is inert while ``campaign_flights.csv`` is header only.

Purity
------
``today`` is supplied by the caller. This module never reads a clock and never
calls random, so the same inputs always produce the same weights.

Weight formula
--------------
For a broadcast day, over the segments of one channel::

    supply      = sum of baseline_tvr over the day's in-scope segments
    mean_tvr    = supply / count
    unmet_i     = the goal points order i still has to place (see unmet_points)
    days_left_i = broadcast days from today, inclusive, to the flight end
    pressure    = clamp(sum_i (unmet_i / days_left_i) / supply, 0, 1)
    weight(s)   = clamp(1 + K * pressure * (tvr(s) / mean_tvr - 1), U_MIN, U_MAX)

``pressure`` is the share of the day's whole expected rating that the booked
goals still need, so it is a demand fact rather than a delivery guess. The
``(ratio - 1)`` term is what makes the lean differential: a uniform multiplier on
every segment of a day changes no ranking and would be a silent no-op.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence

from kairos.optimize.optimizer import ProgramSegment

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGNS_PATH = ROOT / "data" / "campaigns.csv"
DEFAULT_DELIVERY_PATH = ROOT / "data" / "campaign_delivery.csv"

# The one audience this product's ratings are the base for. Held here as a
# string rather than imported from kairos_api, because the engine must not depend
# on the API layer; kairos_api.campaigns_commitment.ALL_VIEWERS is the same value
# and tests/test_goal_based_order.py asserts the two never drift apart.
ALL_VIEWERS = "all_viewers"

# How hard a unit of goal pressure leans, and the bounds the lean lives in. K is
# the strength; U_MAX caps a boost so one large goal cannot own a whole day;
# U_MIN floors the penalty so a low-rating segment is de-prioritised for a points
# goal and never forbidden, which would suppress revenue by the back door.
GOAL_K = 1.0
GOAL_U_MAX = 2.0
GOAL_U_MIN = 0.5

# Basis codes. The first four name a state the pacing words module already has a
# published bilingual refusal for, so a surface renders the product's own
# sentence rather than one this seam invented. MEASURED and BOOKED are not
# refusals: they say what the number rests on.
BASIS_MEASURED = "measured"
BASIS_BOOKED = "booked"
BASIS_NO_GOAL = "no_goal"
BASIS_UNMEASURABLE = "unmeasurable"
BASIS_NO_FLIGHT_DATES = "no_flight_dates"
BASIS_GAP_IN_ELAPSED = "gap_in_elapsed"
BASIS_NO_SOURCE = "no_source"

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

_TRUE_WORDS = frozenset({"true", "yes", "1", "y"})


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _is_true(value: object) -> bool:
    return str(value or "").strip().lower() in _TRUE_WORDS


def _parse_iso(value: object) -> Optional[date]:
    """Parse a YYYY-MM-DD (or leading ISO datetime) into a date, or None."""
    text = str(value or "").strip()
    if not text:
        return None
    head = text.split(" ")[0].split("T")[0]
    parts = head.split("-")
    if len(parts) == 3 and all(part.isdigit() for part in parts):
        try:
            return date(int(parts[0]), int(parts[1]), int(parts[2]))
        except ValueError:
            return None
    return None


def _to_float(value: object) -> Optional[float]:
    text = str(value if value is not None else "").strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class GoalOrder:
    """One goal-based order: points against an audience, on one channel, in a window.

    This is the whole order. There is no spot list, by design, because that is
    what a goal-based order means: the agency states the outcome and the channel
    owns the placement. Anything that needs a spot list must say so rather than
    invent one.

    ``channel`` comes from the campaign row, which the store fills from settings
    through ``channel_scope.operator_channel``. An order carrying no channel
    cannot be scoped to the operator's own inventory, so it steers nothing.
    """

    campaign_id: str
    channel: str
    audience: str
    goal_points: float
    starts_on: str
    ends_on: str
    status: str = "active"
    is_demo: bool = False
    priority: str = ""
    pacing_mode: str = ""

    @property
    def measurable(self) -> bool:
        """Whether this product can count rating points against this audience."""
        return self.audience.strip() == ALL_VIEWERS


@dataclass(frozen=True)
class DeliveredPoints:
    """What the delivery ledger can honestly say one campaign has delivered.

    ``points_counted`` is a FLOOR whenever ``days_unknown`` is above zero: those
    broadcast days carry no per-spot source, so what aired on them is unknown and
    is not zero. A reader that treats the floor as a total has understated the
    remainder, which is the safe direction for a lean and the wrong direction for
    a claim, so the basis travels with the number everywhere.
    """

    campaign_id: str
    points_counted: float
    days_counted: int
    days_unknown: int
    days_total: int

    @property
    def complete(self) -> bool:
        return self.days_total > 0 and self.days_unknown == 0


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


def load_goal_orders(
    path: Optional[str | Path] = None,
    *,
    include_demo: bool = False,
) -> list[GoalOrder]:
    """Read goal-based orders from the campaigns store.

    A campaign row is a goal-based order when it carries a positive
    ``rating_goal_points``. Rows without one are ordinary campaigns and are not
    this seam's business. Ended campaigns are skipped, because a closed order
    cannot want future placement.

    Demo rows are excluded unless ``include_demo`` is set. A seeded row is never
    a booking, so it must never steer a real plan. Every one of the 51 rows on
    disk carrying a rating goal is a demo row, which is why this function returns
    an empty list today and the seam is provably inert.
    """
    target = Path(path) if path is not None else DEFAULT_CAMPAIGNS_PATH
    if not target.exists():
        return []
    out: list[GoalOrder] = []
    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        for row in reader:
            if str(row.get("record_type", "") or "").strip() != "campaign":
                continue
            campaign_id = str(row.get("campaign_id", "") or "").strip()
            points = _to_float(row.get("rating_goal_points"))
            if not campaign_id or points is None or points <= 0:
                continue
            status = str(row.get("status", "") or "").strip() or "active"
            if status != "active":
                continue
            demo = _is_true(row.get("is_demo"))
            if demo and not include_demo:
                continue
            out.append(GoalOrder(
                campaign_id=campaign_id,
                channel=str(row.get("channel", "") or "").strip(),
                audience=str(row.get("rating_goal_audience", "") or "").strip(),
                goal_points=points,
                starts_on=str(row.get("starts_on", "") or "").strip(),
                ends_on=str(row.get("ends_on", "") or "").strip(),
                status=status,
                is_demo=demo,
                priority=str(row.get("priority", "") or "").strip(),
                pacing_mode=str(row.get("pacing_mode", "") or "").strip(),
            ))
    return out


def load_delivered_points(
    path: Optional[str | Path] = None,
) -> dict[str, DeliveredPoints]:
    """Sum the delivery ledger's counted rating points per campaign, honestly.

    A ledger day counts toward ``points_counted`` only when its ``air_state`` is
    ``aired`` and it carries a rating figure. Every other day is counted as
    unknown, including a day marked ``scheduled``, because a booked day is not a
    delivered day. The returned record therefore states a floor and says how many
    days it could not see.
    """
    target = Path(path) if path is not None else DEFAULT_DELIVERY_PATH
    if not target.exists():
        return {}
    counted: dict[str, float] = defaultdict(float)
    known: dict[str, int] = defaultdict(int)
    unknown: dict[str, int] = defaultdict(int)
    with open(target, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return {}
        for row in reader:
            campaign_id = str(row.get("campaign_id", "") or "").strip()
            if not campaign_id:
                continue
            state = str(row.get("air_state", "") or "").strip().lower()
            points = _to_float(row.get("rating_points_planned"))
            if state == "aired" and points is not None:
                counted[campaign_id] += points
                known[campaign_id] += 1
            else:
                unknown[campaign_id] += 1
    ids = set(counted) | set(known) | set(unknown)
    return {
        campaign_id: DeliveredPoints(
            campaign_id=campaign_id,
            points_counted=round(counted.get(campaign_id, 0.0), 4),
            days_counted=known.get(campaign_id, 0),
            days_unknown=unknown.get(campaign_id, 0),
            days_total=known.get(campaign_id, 0) + unknown.get(campaign_id, 0),
        )
        for campaign_id in ids
    }


def days_left(order: GoalOrder, today: date) -> Optional[int]:
    """Broadcast days from ``today`` to the flight end, inclusive, or None.

    Before the flight starts the whole window is still ahead, so the count runs
    from the start date. After the flight ends there is nothing left to place and
    the answer is zero. A window with no dates has no answer at all.
    """
    starts = _parse_iso(order.starts_on)
    ends = _parse_iso(order.ends_on)
    if starts is None or ends is None:
        return None
    if today > ends:
        return 0
    first = starts if today < starts else today
    return max(0, (ends - first).days + 1)


def remaining_days(order: GoalOrder, today: date) -> list[str]:
    """The ISO broadcast days this order still has to place into, in order.

    The same window :func:`days_left` counts, spelled out, so a caller that needs
    to ask another source about those days (the plan's expected rating, for one)
    asks about exactly the days the lean is spread over and not a day more.
    """
    starts = _parse_iso(order.starts_on)
    ends = _parse_iso(order.ends_on)
    if starts is None or ends is None or today > ends:
        return []
    first = starts if today < starts else today
    span = (ends - first).days + 1
    return [(first + timedelta(days=index)).isoformat() for index in range(max(0, span))]


def unmet_points(
    order: GoalOrder,
    delivered: Optional[DeliveredPoints],
    today: date,
) -> tuple[Optional[float], str]:
    """The points this order still has to place, and what that number rests on.

    Returns ``(None, basis)`` whenever the remainder cannot be stated: an
    audience with no panel behind it, or a flight with no dates. Otherwise the
    remainder is the goal less what the ledger counted, and the basis says
    whether the ledger saw every elapsed day (``measured``), saw some of them
    (``gap_in_elapsed``, so the remainder is a ceiling), or has nothing on this
    campaign at all (``no_source``, so the remainder is the booked goal itself).

    The remainder is never negative and is never rounded up to the goal to
    flatter a lean.
    """
    if order.goal_points <= 0:
        return None, BASIS_NO_GOAL
    if not order.measurable:
        return None, BASIS_UNMEASURABLE
    if days_left(order, today) is None:
        return None, BASIS_NO_FLIGHT_DATES
    if delivered is None or delivered.days_total == 0:
        return order.goal_points, BASIS_NO_SOURCE
    remainder = max(0.0, order.goal_points - delivered.points_counted)
    if delivered.complete:
        return remainder, BASIS_MEASURED
    return remainder, BASIS_GAP_IN_ELAPSED


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
        pressure = _clamp(pressure, 0.0, 1.0)
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
