"""Plan, week, compare: the lever resolution and the money on each A/B leg.

Split out of ``scenario_compare_api`` under the 450-line law, named by the
helper rule. It holds the part of the comparison that is arithmetic rather than
routing: which levers each leg actually ran under, the gross, retention cost and
net on one shared basis, the honest statement that two legs produced the same
plan when they did, and the one assembly of the response body that both the
week-scoped route and the single-day route return.

Nothing here runs the optimizer. The caller runs both legs and hands the
payloads in, so this module can be exercised on a recorded payload without an
engine.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from kairos_api.scenario_compare_levers import LEVER_FIELDS, ScenarioLevers

logger = logging.getLogger(__name__)


def _scenario_summary(payload: dict[str, Any], levers: dict[str, Any]) -> dict[str, Any]:
    """Pull the comparable fields from a run_scenario payload.

    ``objective`` is the optimizer's convex-blend score (a weighted blend of
    revenue and retention, NOT a literal revenue-minus-cost subtraction), so it is
    reported under its own name and never relabeled as revenue_net. The net
    figure is a separate, genuinely computed quantity added by :func:`_priced`.
    ``levers`` is the resolved lever set this leg actually ran under, so the
    comparison prints what it compared rather than what was asked for.
    """
    summary = payload.get("summary", {})
    return {
        "revenue_weight": payload.get("controls", {}).get("revenue_weight"),
        "projected_revenue": summary.get("projected_revenue"),
        "average_retention": summary.get("average_retention"),
        "total_breaks": summary.get("total_breaks"),
        "total_ad_seconds": summary.get("total_ad_seconds"),
        "objective": summary.get("objective"),
        "compliant": summary.get("compliant"),
        "channel": payload.get("channel"),
        "day": payload.get("day"),
        "levers": dict(levers),
    }


def _priced(
    summary: dict[str, Any], payload: dict[str, Any], segments: list[Any], risk_lambda: float
) -> dict[str, Any]:
    """Add gross, retention cost and net to one leg's summary, or say why not.

    The money comes from the frozen read layer's own pricer, which joins the
    scenario's per-segment break counts back to the engine's ProgramSegment
    objects and values the retention loss at the same CPP the revenue was priced
    at. Gross minus retention cost equals net by construction, so the three
    figures cannot disagree. When the plan and the rebuilt segments no longer
    join, the pricer refuses, and this refusal is carried verbatim rather than
    replaced by a zero.
    """
    from kairos_api.plan_read_frontier import scenario_plan_money

    if not segments:
        summary["money_available"] = False
        summary["money_reason"] = (
            "the engine's segments for this channel-day could not be rebuilt, "
            "so retention cost cannot be priced"
        )
        return summary
    try:
        money = scenario_plan_money(payload, segments, risk_lambda)
    except Exception as exc:  # pragma: no cover - data/environment dependent
        logger.exception("scenario money pricing failed")
        summary["money_available"] = False
        summary["money_reason"] = f"plan money pricing failed: {str(exc)[:160]}"
        return summary
    if not money.get("available"):
        summary["money_available"] = False
        summary["money_reason"] = str(money.get("reason") or "plan money unavailable")
        return summary
    summary["money_available"] = True
    summary["gross"] = money["gross"]
    summary["retention_cost"] = money["retention_cost"]
    summary["revenue_net"] = money["net"]
    return summary


def _delta(a: dict[str, Any], b: dict[str, Any], key: str) -> Optional[float]:
    """b - a for a numeric summary field, or None when either side is missing."""
    av, bv = a.get(key), b.get(key)
    if av is None or bv is None:
        return None
    return round(float(bv) - float(av), 4)


def _resolve_levers(
    leg: Optional[ScenarioLevers], weight: int, shared: dict[str, Any]
) -> dict[str, Any]:
    """One leg's effective levers: per-leg value, then shared, then saved."""
    values = dict(shared)
    values["revenue_weight"] = weight
    if leg is not None:
        for field in LEVER_FIELDS:
            supplied = getattr(leg, field, None)
            if supplied is not None:
                values[field] = supplied
    return values


_WEEK_MONEY_BASIS = (
    "Expected revenue is the optimizer's own projection for every broadcast day in the plan's "
    "own week, added up. Retention cost is the audience those breaks are modelled to lose, "
    "priced at the same CPP. Net is expected revenue minus that cost, on the same per-break "
    "basis the committed plan's yield-per-second money uses."
)

_DAY_MONEY_BASIS = (
    "Expected revenue is the optimizer's own projection for one representative broadcast day, "
    "not the week. Retention cost is the audience that day's breaks are modelled to lose, "
    "priced at the same CPP. Net is expected revenue minus that cost, on the same per-break "
    "basis the committed plan's yield-per-second money uses."
)

_OBJECTIVE_NOTE = (
    "Objective is the optimizer's convex-blend score and is reported under its own name. "
    "It is not revenue minus retention cost; that subtraction is the net figure beside it."
)

_WEEK_OBJECTIVE_NOTE = (
    "Objective is the optimizer's convex-blend score, normalised inside one broadcast day, so "
    "the figure over a week is the mean of its days and never a sum. It is not revenue minus "
    "retention cost; that subtraction is the net figure beside it."
)


def compare_body(
    result: dict[str, Any], guardrails: dict[str, Any], by_day: Optional[list[dict[str, Any]]] = None
) -> dict[str, Any]:
    """The response both comparison routes return, assembled once.

    ``result`` carries the two priced legs and the scope they were run on. The
    week-scoped run and the single-day run reach this with the same shape, so the
    deltas, the sameness note and the basis sentences have one implementation and
    the two windows cannot drift apart in what they claim.
    """
    a = result["a"]
    b = result["b"]
    scope = result.get("scope") or {}
    week = str(scope.get("mode") or "day") == "week"
    money_available = bool(a.get("money_available") and b.get("money_available"))
    return {
        "available": True,
        "guardrails": dict(guardrails),
        "a": a,
        "b": b,
        "delta": {
            "revenue": _delta(a, b, "projected_revenue"),
            "retention": _delta(a, b, "average_retention"),
            "breaks": _delta(a, b, "total_breaks"),
            "ad_seconds": _delta(a, b, "total_ad_seconds"),
            "objective": _delta(a, b, "objective"),
            "gross": _delta(a, b, "gross") if money_available else None,
            "retention_cost": _delta(a, b, "retention_cost") if money_available else None,
            "revenue_net": _delta(a, b, "revenue_net") if money_available else None,
        },
        "money_available": money_available,
        "money_reason": None if money_available else (a.get("money_reason") or b.get("money_reason")),
        "money_basis": _WEEK_MONEY_BASIS if week else _DAY_MONEY_BASIS,
        "objective_note": _WEEK_OBJECTIVE_NOTE if week else _OBJECTIVE_NOTE,
        "sameness": _identical_note(a, b),
        "scope": scope,
        "by_day": result.get("by_day") if by_day is None else by_day,
    }


def _identical_note(a: dict[str, Any], b: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Whether the two legs produced the same plan, and which levers were equal.

    Two scenarios that differ only in the revenue weight normally do produce the
    same plan, because at a fixed retention floor the refined optimum is nearly
    invariant to the weight. Saying so is the honest answer; printing a delta of
    zero and leaving the planner to infer it is not.
    """
    compared = ("projected_revenue", "average_retention", "total_breaks", "total_ad_seconds")
    if any(a.get(key) != b.get(key) for key in compared):
        return None
    if a.get("revenue_net") != b.get("revenue_net"):
        return None
    differing = [field for field in LEVER_FIELDS if a["levers"].get(field) != b["levers"].get(field)]
    return {
        "identical": True,
        "levers_that_differ": differing,
        "levers_that_match": [field for field in LEVER_FIELDS if field not in differing],
    }
