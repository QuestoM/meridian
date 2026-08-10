"""Wire booked rating goals into the shared day optimizer.

The goal seam deliberately owns no placement logic.  This adapter prepares its
ranking and objective inputs for the one day-level call site used by live plans,
simulations and weekly exports.  With no real, applicable order it preserves the
caller's objective mode and supplies no custom objective, making the integration
an exact operational no-op on the shipped demo-only store.
"""

from __future__ import annotations

from datetime import date
from typing import Callable, Mapping, Optional, Sequence

from kairos.optimize.goal_seam import (
    DeliveredPoints,
    GoalOrder,
    fold_into_demand_weights,
    goal_adjusted_net,
    load_delivered_points,
    load_goal_orders,
)
from kairos.optimize.optimizer import (
    OBJECTIVE_BLEND,
    OBJECTIVE_REVENUE_NET,
    ProgramSegment,
)
from kairos.optimize.revenue_net import segment_net_revenue

NetOf = Callable[[ProgramSegment, int], float]


def prepare_goal_inputs(
    segments: Sequence[ProgramSegment],
    demand_weights: Mapping[str, float],
    today: Optional[date],
    objective_mode: str,
    *,
    orders: Optional[Sequence[GoalOrder]] = None,
    delivered_of: Optional[Mapping[str, DeliveredPoints]] = None,
) -> tuple[dict[str, float], str, Optional[NetOf]]:
    """Return goal-folded weights, effective mode and an optional net function.

    ``orders=None`` means read the real-order store.  An explicit empty sequence
    means the caller already read it and found no real bookings.  Delivery is
    loaded only after a real order exists, avoiding needless I/O on today's
    demo-only data.

    A goal whose pressure can be valued switches the ordinary blend path to the
    net objective so the greedy step, F1 refiner and DP tier all optimise the same
    commitment-aware scalar.  Explicitly invalid modes are left untouched for the
    optimizer to reject rather than being hidden by the adapter.
    """
    rows = list(load_goal_orders() if orders is None else orders)
    base = dict(demand_weights)
    if today is None or not rows:
        return base, objective_mode, None

    delivery = dict(load_delivered_points() if delivered_of is None else delivered_of)
    folded = fold_into_demand_weights(
        base,
        segments,
        today,
        orders=rows,
        delivered_of=delivery,
    )
    adjusted = goal_adjusted_net(
        segment_net_revenue,
        segments,
        rows,
        today,
        delivered_of=delivery,
    )
    if adjusted is segment_net_revenue:
        return folded, objective_mode, None
    if objective_mode not in (OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET):
        return folded, objective_mode, None
    return folded, OBJECTIVE_REVENUE_NET, adjusted
