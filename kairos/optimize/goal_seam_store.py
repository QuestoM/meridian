"""Reading a booked goal, and reading what the delivery ledger can honestly say.

The persistence half of :mod:`kairos.optimize.goal_seam`, split out so that
module stays under the project line limit and kept separate on purpose: this file
knows two CSV files and nothing about placement, while the seam knows placement
and nothing about a file. Everything here is re-exported from the seam under the
names it always had, so a caller imports one module.

Two stores are read and neither is written.

**The campaigns store.** A campaign row carrying a positive
``rating_goal_points`` is a goal-based order. Ended campaigns are skipped, and
demo rows are skipped unless a caller asks for them, because a seeded row is not
a booking and must never steer a real plan.

**The delivery ledger.** A broadcast day counts only when it says it aired and
carries a rating. Every other day, including a day that is merely booked, is
counted as unknown. So what a campaign has delivered is a FLOOR and what it has
left to place is a CEILING, and the basis travels with the number wherever it is
shown. Measured on this tree, 306 of the ledger's 368 days carry no per-spot
source, so that is the ordinary case and not the edge one.

Nothing here reads a clock. Every date is the caller's.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGNS_PATH = ROOT / "data" / "campaigns.csv"
DEFAULT_DELIVERY_PATH = ROOT / "data" / "campaign_delivery.csv"

# The one audience this product's ratings are the base for. Held here as a
# string rather than imported from kairos_api, because the engine must not depend
# on the API layer; kairos_api.campaigns_commitment.ALL_VIEWERS is the same value
# and tests/test_goal_based_order.py asserts the two never drift apart.
ALL_VIEWERS = "all_viewers"

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

_TRUE_WORDS = frozenset({"true", "yes", "1", "y"})


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


