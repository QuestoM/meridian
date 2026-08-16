"""The revenue-versus-retention frontier and the net-focused comparison beside it.

Frozen helper of :mod:`kairos_api.plan_read`, split out under the 450-line law.
Each point is a real optimizer run, so the sweep is computed in ONE background
thread guarded by ONE lock and ONE state dict, and a request never blocks on it.
Three owners read this module: Today (the overview chart), Plan (the optimizer
net comparison) and Kai (the capability and simulate tools), so it belongs to
none of them.

Moved verbatim from dashboard_api.py with the leading underscore dropped. The old
names keep resolving from :mod:`kairos_api.dashboard_api` and
:mod:`kairos_api.server`, against these same objects, including the single
lru_cache instances, the lock and the state dict.
"""

from __future__ import annotations

import copy
import logging
import threading
from functools import lru_cache
from typing import Any

from kairos.optimize.inventory import InventoryInputError, load_inventory
from kairos_api.core import (
    KairosSettings,
    _pacing_call_kwargs,
    _plan_segment_index,
    _safe_number,
    run_scenario,
)
from kairos_api.plan_read_scope import (
    frontier_data_signature,
    owned_representative_day,
    parse_frontier_scope,
)

logger = logging.getLogger(__name__)


@lru_cache(maxsize=32)
def frontier_points_cached(
    signature: tuple[tuple[str, int], ...],
    channel: str,
    day: str | None,
    saved_floor: float,
    max_breaks_per_hour: int,
    risk_lambda: float,
    revenue_weight: int,
) -> tuple[dict[str, Any], ...]:
    """Trace the genuine revenue-vs-retention Pareto frontier for one owned scope.

    The frontier sweeps the RETENTION FLOOR at the saved revenue weight, and each
    point is the REFINED optimum (``refine=True``), not a greedy approximation.
    This is a deliberate correctness choice backed by measurement: at a fixed
    floor the refined optimum is nearly invariant to the revenue weight above a
    low threshold (once retention already clears the floor, the weight barely
    moves the plan), so a revenue-weight sweep collapses onto a single point, and
    any spread it appears to show is an artifact of the weaker greedy optimizer
    leaving a different amount of revenue on the table at each weight. The
    retention floor is the binding lever: tightening it sheds the lowest-value
    breaks, trading revenue for retention, which is the real tradeoff the
    operator is choosing between. Cached on the data-file ``signature`` plus the
    guardrail inputs so the sweep runs once and is reused across requests.
    """
    del signature  # part of the cache key only
    anchor = round(float(saved_floor), 4)
    floors = sorted({0.72, 0.80, 0.85, 0.90, 0.93, 0.97, anchor})
    pacing = _pacing_call_kwargs()
    points: list[dict[str, Any]] = []
    for floor in floors:
        try:
            payload = run_scenario(
                revenue_weight=revenue_weight,
                retention_floor=floor,
                max_breaks_per_hour=max_breaks_per_hour,
                risk_lambda=risk_lambda,
                channel=channel,
                day=day,
                refine=True,
                require_usable_inventory=True,
                **pacing,
            )
        except Exception:
            logger.exception("frontier scenario failed at retention_floor=%s", floor)
            continue
        summary = payload.get("summary", {})
        retention = summary.get("average_retention")
        revenue = summary.get("projected_revenue")
        if retention is None or revenue is None:
            continue
        points.append(
            {
                "retention": round(_safe_number(retention), 1),
                "revenue": round(_safe_number(revenue), 2),
                "retention_floor": round(float(floor), 4),
                "num_breaks": int(_safe_number(summary.get("total_breaks", 0))),
                "selected": abs(float(floor) - anchor) < 1e-9,
            }
        )
    points.sort(key=lambda point: point["retention"])
    return tuple(points)


# Label id of the additive net-focused scenario point served beside the frontier
# sweep (the whole-schedule optimum under objective_mode='revenue_net').
NET_POINT_ID = "net_focused"


def scenario_plan_money(
    payload: dict[str, Any], segments: list[Any], risk_lambda: float
) -> dict[str, Any]:
    """Price one run_scenario plan in ILS on the engine's own plan money model.

    Joins the payload's per-segment break counts back to the rebuilt
    ProgramSegment objects and prices the plan with
    :func:`kairos.optimize.revenue_net.plan_revenue_net`: gross is the runner's
    own projected revenue, the retention cost is the per-break audience loss
    priced at the same CPP, and net is their difference. This is the SAME
    per-break cost model the committed plan's /api/yield-per-second money uses,
    so the comparison and the committed story share one basis. Coefficients are
    first risk-adjusted exactly as the optimizer decided
    (:func:`kairos.optimize._segment_math._risk_adjusted_coefficient`, an exact
    identity at risk_lambda 0). Returns the money block, or an honest
    ``{"available": False, "reason": ...}`` when the plan cannot be priced;
    nothing is proxied.
    """
    from dataclasses import replace as _dataclass_replace
    from types import SimpleNamespace

    from kairos.optimize._segment_math import _risk_adjusted_coefficient
    from kairos.optimize.revenue_net import plan_revenue_net

    summary = payload.get("summary", {})
    plan_rows = payload.get("segments", [])
    adjusted = [
        _dataclass_replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda))
        for s in segments
    ]
    shim = SimpleNamespace(
        segments=[
            SimpleNamespace(
                segment_id=str(row.get("segment_id", "")),
                num_breaks=int(_safe_number(row.get("num_breaks", 0))),
                revenue=float(_safe_number(row.get("revenue", 0.0))),
            )
            for row in plan_rows
        ],
        total_revenue=float(_safe_number(summary.get("projected_revenue", 0.0))),
    )
    money = plan_revenue_net(shim, segments=adjusted)
    if not money.get("available"):
        return {"available": False, "reason": str(money.get("reason") or "plan money unavailable")}
    if int(money.get("priced_segments") or 0) < len(plan_rows):
        return {
            "available": False,
            "reason": (
                "scenario plan and rebuilt segments no longer join; "
                "retention cannot be priced honestly"
            ),
        }
    return {
        "available": True,
        "gross": money["revenue_ils"],
        "retention_cost": money["retention_cost_ils"],
        "net": money["revenue_net_ils"],
        "breaks": int(_safe_number(summary.get("total_breaks", 0))),
    }


def net_bundle_failure(channel: str, day: str | None, reason: str) -> dict[str, Any]:
    """Honest empty net bundle: no point, no money, the reason named."""
    return {
        "channel": channel,
        "day": day,
        "net_point": None,
        "comparison_available": False,
        "reason": reason,
        "current": None,
        "net_focused": None,
    }


@lru_cache(maxsize=32)
def frontier_net_bundle_cached(
    signature: tuple[tuple[str, int], ...],
    channel: str,
    day: str | None,
    saved_floor: float,
    max_breaks_per_hour: int,
    risk_lambda: float,
    revenue_weight: int,
    objective_mode: str,
) -> dict[str, Any]:
    """One net-focused whole-schedule scenario beside the sweep, with money.

    Runs the SAME scenario runner as the frontier sweep on the same owned scope,
    refined, under the saved guardrails: once at the operator's saved
    ``objective_mode`` (the 'current' side, the saved decision re-evaluated so a
    money block exists on the sweep's own anchor basis) and once under
    ``objective_mode='revenue_net'`` (the net-focused side). When the saved mode
    already is ``revenue_net`` the single run serves both sides. Each plan is
    priced with the per-break retention-cost model
    (:func:`scenario_plan_money`), so the operator sees gross, the
    model-estimated retention cost, and net on one shared basis. Cached beside
    the point sweep on the same data signature and computed in the same single
    background thread, never inline in a request.
    """
    del signature  # part of the cache key only
    if day is None:
        return net_bundle_failure(
            channel, day, "owned channel has no dated programmes to scope the comparison"
        )
    pacing = _pacing_call_kwargs()

    def _run(mode: str) -> dict[str, Any]:
        return run_scenario(
            revenue_weight=revenue_weight,
            retention_floor=saved_floor,
            max_breaks_per_hour=max_breaks_per_hour,
            risk_lambda=risk_lambda,
            channel=channel,
            day=day,
            objective_mode=mode,
            require_usable_inventory=True,
            **pacing,
        )

    try:
        net_payload = _run("revenue_net")
        current_payload = (
            net_payload if objective_mode == "revenue_net" else _run(objective_mode)
        )
    except Exception:
        logger.exception("net-focused frontier scenario failed")
        return net_bundle_failure(
            channel, day, "net-focused scenario run failed; see the server log"
        )

    summary = net_payload.get("summary", {})
    net_point: dict[str, Any] | None = None
    if summary.get("average_retention") is not None and summary.get("projected_revenue") is not None:
        # Same fields as a frontier sweep point, plus the label id. 'selected' is
        # honest: True only when the saved objective_mode IS revenue_net.
        net_point = {
            "retention": round(_safe_number(summary.get("average_retention")), 1),
            "revenue": round(_safe_number(summary.get("projected_revenue")), 2),
            "retention_floor": round(float(saved_floor), 4),
            "num_breaks": int(_safe_number(summary.get("total_breaks", 0))),
            "selected": objective_mode == "revenue_net",
            "id": NET_POINT_ID,
        }

    try:
        segments = list(_plan_segment_index(((channel, str(day)),), pacing["settings"]).values())
        money_current = scenario_plan_money(current_payload, segments, risk_lambda)
        money_net = scenario_plan_money(net_payload, segments, risk_lambda)
    except Exception:
        logger.exception("net-focused plan pricing failed")
        bundle = net_bundle_failure(
            channel, day, "plan money pricing failed; see the server log"
        )
        bundle["net_point"] = net_point
        return bundle

    available = bool(money_current.get("available") and money_net.get("available"))
    reason = None
    if not available:
        reason = str(
            (money_current if not money_current.get("available") else money_net).get("reason")
            or "comparison money unavailable"
        )
    return {
        "channel": channel,
        "day": day,
        "net_point": net_point,
        "comparison_available": available,
        "reason": reason,
        "current": money_current if money_current.get("available") else None,
        "net_focused": money_net if money_net.get("available") else None,
    }


# The frontier is a real optimizer sweep and is too slow to trace inline on a cold
# cache, so it is computed in a background thread. These guard the single in-flight
# computation and its result so /api/overview never blocks on the sweep.
frontier_bg_lock = threading.Lock()
frontier_bg_state: dict[str, Any] = {
    "key": None, "status": "idle", "points": (), "net_bundle": None,
}


def frontier_async(settings: KairosSettings, scope: str | None = None) -> tuple[list[dict[str, Any]], str]:
    """Return ``(points, status)`` for the revenue-vs-retention frontier without
    ever blocking the request.

    Each point is an ACTUAL optimization (:func:`kairos.service.run_scenario`),
    not a synthetic offset off one summary: the curve is the genuine Pareto
    trade-off the engine produces as it shifts from retention-first
    (revenue_weight 0) to revenue-first (revenue_weight 100), under the saved
    retention floor, hourly break cap and risk aversion. The point matching the
    saved revenue_weight is marked ``selected``.

    The frontier forecasts the operator's OWNED channel inventory only. Revenue is
    never projected for a competitor channel: competitor programming informs the
    churn/retention model, not the revenue projection (the competitor-information
    boundary). The curve is scoped to ``settings.operator_channel`` on its busiest
    broadcast day (see :func:`owned_representative_day`); a ``day:<date>`` scope
    narrows it to another day within the owned channel, and a ``channel:<id>`` scope
    is accepted only when it equals the owned channel, so the forecast can never be
    redirected to a competitor.

    Status is one of: ``no_channel`` (no owned channel set yet, points empty: the
    dashboard prompts the operator to pick their channel), ``computing`` (a
    background sweep is in flight, points empty: an honest "forecast is being
    computed" state, never a fabricated curve), ``unavailable`` (a present
    inventory source cannot produce a usable slot, so no sweep is attempted), or
    ``ready`` (points populated from the finished sweep). The sweep itself is cached
    on the data-file signature plus the guardrails, so it runs once and is reused
    across requests and weights.
    """
    points, _net_bundle, status = frontier_state(settings, scope)
    return points, status


def frontier_state(
    settings: KairosSettings, scope: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any] | None, str]:
    """``(points, net_bundle, status)`` for the frontier machinery, never blocking.

    The single shared engine behind :func:`frontier_async` (whose points/status
    contract is unchanged), the overview's additive ``net_point``, and the
    ``/api/optimizer/net-comparison`` endpoint. ONE background thread computes
    the point sweep and the net-focused bundle together under one key, so their
    statuses can never disagree and no second background machine exists. The key
    extends the sweep key with the saved ``objective_mode`` (the 'current' side
    of the bundle is evaluated at it), so a mode edit honestly re-enters
    ``computing``; the sweep points themselves are cached without the mode and
    are byte-identical across that transition. ``net_bundle`` is ``None`` until
    status is ``ready``; a ready bundle may still report
    ``comparison_available: False`` with a reason, never invented numbers.
    """
    owned = str(settings.operator_channel or "").strip()
    if not owned:
        return [], None, "no_channel"
    try:
        # This check deliberately runs before consulting the background cache:
        # cached money from a formerly valid source must not survive after the
        # inventory file becomes present-but-unusable. A missing file remains the
        # explicitly neutral signal and does not raise.
        load_inventory(require_usable=True)
    except InventoryInputError as exc:
        return [], net_bundle_failure(owned, None, str(exc)), "unavailable"
    signature = frontier_data_signature()
    scope_kwargs = parse_frontier_scope(scope, settings)
    effective_day = scope_kwargs["day"] or owned_representative_day(signature, owned)
    key = (
        signature,
        owned,
        effective_day,
        float(settings.min_retention_floor),
        int(settings.max_breaks_per_hour),
        float(settings.risk_lambda),
        int(settings.revenue_weight),
        str(getattr(settings, "objective_mode", "blend") or "blend"),
    )
    with frontier_bg_lock:
        state = frontier_bg_state
        if state["key"] == key and state["status"] == "ready":
            return (
                [dict(point) for point in state["points"]],
                copy.deepcopy(state.get("net_bundle")),
                "ready",
            )
        if state["key"] == key and state["status"] == "computing":
            return [], None, "computing"
        # New (or stale) key: start a fresh single in-flight computation.
        state["key"] = key
        state["status"] = "computing"
        state["points"] = ()
        state["net_bundle"] = None

    def _compute() -> None:
        try:
            points = frontier_points_cached(*key[:7])
        except Exception:
            logger.exception("frontier background compute failed")
            points = ()
        try:
            net_bundle = frontier_net_bundle_cached(*key)
        except Exception:
            logger.exception("frontier net-focused bundle compute failed")
            net_bundle = net_bundle_failure(
                key[1], key[2], "net-focused computation failed; see the server log"
            )
        with frontier_bg_lock:
            if frontier_bg_state["key"] == key:
                frontier_bg_state["points"] = points
                frontier_bg_state["net_bundle"] = net_bundle
                frontier_bg_state["status"] = "ready"

    threading.Thread(target=_compute, name="kairos-frontier", daemon=True).start()
    return [], None, "computing"
