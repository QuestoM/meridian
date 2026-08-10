"""Monetize viewer-retention damage in ILS, so revenue can be reported NET.

The optimizer's headline objective is a unitless convex blend of normalised
revenue and a retention share (see :mod:`kairos.optimize.optimizer`). That blend
answers "which schedule balances money against audience" but it is not itself a
currency figure, so the product could not state, in shekels, what audience shed
by ad breaks actually costs. This module supplies that missing money number from
inputs that already exist, with no fabricated audience.

The economic model
------------------
Every quantity below is real: nothing invents a rating or a price.

  * ``baseline_tvr`` is the real mean planned break rating of the programme
    (:mod:`kairos.data.transform` builds it from the plan; a programme with no
    planned rating is a zero-value segment, never a guessed one).
  * ``retention_share`` is the audience share the segment keeps once it carries
    its breaks, in [0, 1], from the measured impact coefficient
    (:func:`kairos.optimize._segment_math._segment_retention`).
  * ``base_rate`` is the effective price per rating-point-second in ILS, the
    channel CPP times the segment premium (``cpp * premium``). Israeli TV ad
    pricing is Cost Per rating Point, so revenue scales with rating points, the
    seconds a break runs, and this rate.
  * ``ad_seconds`` is the break time the segment carries (its breaks summed).

The lost audience a break sheds, valued at the same rate the delivered spots
earn, is the retention cost:

    lost_tvr           = baseline_tvr * (1 - retention_share)
    retention_cost_ils = base_rate * lost_tvr * (ad_seconds / unit_seconds)
    revenue_ils        = base_rate * (baseline_tvr * retention_share)
                                   * (ad_seconds / unit_seconds)
    revenue_net_ils    = revenue_ils - retention_cost_ils

``revenue_ils`` is exactly the revenue the optimizer already reports (it is
already valued at the smaller, retained audience). ``retention_cost_ils`` is the
ad revenue foregone because the breaks pushed audience away, priced at the same
CPP. ``revenue_net_ils`` charges that damage against the delivered revenue, which
is the honest "what did the interruptions cost us" figure the product needs. When
``retention_share == 1`` (no audience lost) the cost is zero and net equals
revenue, as it must.

Honesty (Law 9)
---------------
Every returned figure carries a ``basis`` block: the formula string, the named
inputs, and ``source: 'modeled'`` (this is a model output, not a measured
receipt). When a required per-segment input is missing (no rating, no rate), the
framework returns ``available: false`` naming the exact missing input rather than
proxying a number. This module never mutates the optimizer's objective; the
optimizer keeps its convex blend unless a caller explicitly opts into the net
objective mode.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from kairos.optimize._segment_math import _segment_retention
from kairos.optimize._types import ProgramSegment

_EPSILON = 1e-9

# The single formula string shipped with every number, so a reader can trace the
# figure back to its inputs without leaving the payload.
RETENTION_COST_FORMULA = (
    "retention_cost_ils = base_rate * baseline_tvr * (1 - retention_share) "
    "* (ad_seconds / unit_seconds); "
    "revenue_net_ils = revenue_ils - retention_cost_ils"
)

# The named inputs and where each one comes from, shipped in the basis block.
_INPUT_SOURCES: dict[str, str] = {
    "baseline_tvr": "real mean planned break rating per segment (kairos.data.transform)",
    "retention_share": "measured retention share at the chosen break count (kairos.optimize._segment_math)",
    "base_rate": "effective ILS price per rating-point-second, cpp * premium (kairos.optimize.pricing)",
    "ad_seconds": "break seconds the segment carries (num_breaks * break_length)",
    "unit_seconds": "seconds the CPP is quoted per (1.0 for the per-second real-data rate)",
}


def _basis(inputs: Sequence[str]) -> dict[str, Any]:
    """The disclosure block shipped with every monetized number.

    ``formula`` is the literal expression, ``inputs`` maps each named input to its
    real source, and ``source`` is ``'modeled'`` because this is a model output,
    never a measured receipt. Only the inputs actually used are disclosed.
    """
    return {
        "formula": RETENTION_COST_FORMULA,
        "inputs": {name: _INPUT_SOURCES[name] for name in inputs if name in _INPUT_SOURCES},
        "source": "modeled",
    }


def retention_cost_ils(
    *,
    baseline_tvr: float,
    retention_share: float,
    base_rate: float,
    ad_seconds: float,
    unit_seconds: float = 1.0,
) -> float:
    """The ad revenue foregone (ILS) because breaks shed audience on one segment.

    ``lost_tvr = baseline_tvr * (1 - retention_share)`` is the rating points the
    breaks pushed away; valued at ``base_rate`` (ILS per rating-point-second) over
    the ``ad_seconds`` the segment runs, that is the retention cost. A segment that
    keeps its whole audience (``retention_share == 1``) or carries no ad seconds
    costs zero. ``retention_share`` is clamped into [0, 1] defensively; the raw
    inputs are validated as non-negative so a bad feed raises rather than returns a
    silently wrong cost.
    """
    if baseline_tvr < 0 or base_rate < 0 or ad_seconds < 0:
        raise ValueError("baseline_tvr, base_rate and ad_seconds must be non-negative")
    if unit_seconds <= 0:
        raise ValueError("unit_seconds must be positive")
    share = min(1.0, max(0.0, retention_share))
    lost_tvr = baseline_tvr * (1.0 - share)
    return base_rate * lost_tvr * (ad_seconds / unit_seconds)


def require_monetizable(segments: Sequence[ProgramSegment]) -> None:
    """Raise unless at least one segment carries a rating and rate to value.

    Net-mode monetization needs a real ``baseline_tvr`` and a positive rate
    (``cpp * premium``) somewhere in the day; with none there is no lost audience
    to price, so the honest response is to refuse rather than return an empty plan
    for a hidden reason.
    """
    if not any(s.baseline_tvr > 0 and (s.cpp * s.premium) > 0 for s in segments):
        raise ValueError(
            "objective_mode='revenue_net' requires a per-segment audience to value: "
            "no segment carries a positive rating and rate (baseline_tvr * cpp * premium)."
        )


def segment_retention_cost_ils(segment: ProgramSegment, k: int) -> float:
    """One segment's retention cost (ILS) at ``k`` breaks, summed per break.

    Each break j (1..k) is valued at the retention that holds once j breaks are
    present, exactly as the optimizer values that break's revenue
    (:func:`kairos.optimize._segment_math._marginal_revenue`). Its lost audience is
    ``baseline_tvr * (1 - retention(j))``, priced over that break's seconds. Summing
    per break keeps revenue and cost consistent: revenue(j) + cost(j) is the gross
    at full audience for that break, so the segment's revenue plus this cost equals
    its gross potential exactly. A different retention at each break is why this is
    a sum, not the final share applied once.
    """
    if k <= 0:
        return 0.0
    base_rate = segment.cpp * segment.premium
    unit_seconds = segment.unit_seconds if segment.unit_seconds > 0 else 1.0
    total = 0.0
    for j in range(1, k + 1):
        total += retention_cost_ils(
            baseline_tvr=segment.baseline_tvr,
            retention_share=_segment_retention(segment, j),
            base_rate=base_rate,
            ad_seconds=segment.break_length_seconds,
            unit_seconds=unit_seconds,
        )
    return total


def segment_net_revenue(segment: ProgramSegment, k: int) -> float:
    """One segment's revenue net of retention cost (ILS) at ``k`` breaks.

    Revenue is the ad revenue at the retained audience (what the blend path already
    reports, valued per break); the retention cost is the ad revenue foregone as the
    breaks shed audience, valued per break the same way
    (:func:`segment_retention_cost_ils`) so the two reconcile exactly. The net is
    revenue minus that cost, so the net-mode greedy step values a break by the change
    in this quantity. Both are zero at ``k == 0``. Imported lazily by
    :func:`kairos.optimize.optimizer.optimize_breaks` in net mode.
    """
    if k <= 0:
        return 0.0
    from kairos.optimize._segment_math import _segment_revenue

    return _segment_revenue(segment, k) - segment_retention_cost_ils(segment, k)


def _unavailable(reason: str) -> dict[str, Any]:
    """Honest empty result: no number, the exact missing input named."""
    return {
        "available": False,
        "reason": reason,
        "revenue_ils": None,
        "retention_cost_ils": None,
        "revenue_net_ils": None,
        "basis": _basis(()),
    }


def plan_revenue_net(
    plan: Any,
    *,
    segments: Optional[Sequence[ProgramSegment]] = None,
) -> dict[str, Any]:
    """Revenue, retention cost and net (ILS) for a whole optimizer plan.

    ``plan`` is an :class:`~kairos.optimize.optimizer.OptimizationResult`. Its
    ``segments`` carry each programme's break count and reported revenue; the
    matching :class:`~kairos.optimize._types.ProgramSegment` objects (passed as
    ``segments``, the same list handed to the optimizer) carry the real
    ``baseline_tvr``, ``cpp`` and ``premium`` the plan does not echo. When
    ``segments`` is omitted the function still reports revenue from the plan but
    reports the retention cost as unavailable, naming the missing input, because
    the per-segment audience is not on the plan alone.

    ``segments`` must be on the plan's DECISION basis: when the plan was decided
    with ``risk_lambda > 0`` the optimizer priced every break at the
    risk-adjusted retention, so pass segments whose ``impact_coefficient`` has
    been through the same
    :func:`kairos.optimize._segment_math._risk_adjusted_coefficient` pre-pass
    (an exact identity at ``risk_lambda == 0``). Pricing a risk-adjusted plan
    with unadjusted segments understates the retention cost, materially at high
    risk aversion. The same applies to the saved schedule: ``predicted_revenue``
    in the weekly CSV (the COLUMNS block in :mod:`kairos.export.schedule`) is
    decision-basis under ``risk_lambda > 0``, alongside its ``retention_used``
    column.

    Returns ``{available, revenue_ils, retention_cost_ils, revenue_net_ils,
    basis}``; ``basis`` names the formula, the inputs and ``source: 'modeled'``.
    """
    plan_segments = list(getattr(plan, "segments", []) or [])
    revenue_ils = float(getattr(plan, "total_revenue", 0.0) or 0.0)
    if segments is None:
        return {
            "available": False,
            "reason": (
                "Per-segment audience (baseline_tvr) is not carried on the plan alone; "
                "pass the ProgramSegment list the optimizer used."
            ),
            "revenue_ils": round(revenue_ils, 2),
            "retention_cost_ils": None,
            "revenue_net_ils": None,
            "basis": _basis(()),
        }

    by_id = {s.segment_id: s for s in segments}
    total_cost = 0.0
    recomputed_revenue = 0.0
    priced = 0
    for sp in plan_segments:
        segment = by_id.get(sp.segment_id)
        if segment is None:
            continue
        # Cost is summed per break at each break's own retention, the same way the
        # plan values that break's revenue, so revenue and cost reconcile exactly.
        total_cost += segment_retention_cost_ils(segment, sp.num_breaks)
        recomputed_revenue += float(sp.revenue)
        priced += 1

    if priced == 0:
        return _unavailable(
            "No plan segment matched a ProgramSegment; cannot value retention loss."
        )
    # Prefer the plan's own reported total when present, so the net reconciles to
    # the exact revenue the optimizer produced; fall back to the reconstruction
    # (identical by construction) when the plan reports no total.
    revenue = revenue_ils if revenue_ils > _EPSILON else recomputed_revenue
    return {
        "available": True,
        "revenue_ils": round(revenue, 2),
        "retention_cost_ils": round(total_cost, 2),
        "revenue_net_ils": round(revenue - total_cost, 2),
        "priced_segments": priced,
        "basis": _basis(
            ("baseline_tvr", "retention_share", "base_rate", "ad_seconds", "unit_seconds")
        ),
    }


def frame_revenue_net(
    frame: Any,
    *,
    revenue_col: str = "predicted_revenue",
    baseline_tvr_col: str = "baseline_tvr",
    rate_col: str = "base_rate",
    ad_seconds_col: str = "total_break_time",
    unit_seconds: float = 1.0,
) -> dict[str, Any]:
    """Revenue, retention cost and net (ILS) from a saved weekly-schedule frame.

    Exact when the frame carries the segment's ``baseline_tvr``. The delivered
    revenue already encodes the per-break-weighted retained audience, so the cost is
    the gross potential minus that delivered revenue:

        gross_potential    = base_rate * baseline_tvr * (ad_seconds / unit_seconds)
        retention_cost_ils = gross_potential - revenue

    which equals the live per-break sum exactly (revenue is the sum of per-break
    revenue, gross is the same seconds at full audience). No audience is invented and
    no per-break retention path is needed.

    IMPORTANT (Law 9): the saved CSV persists ``retention_used`` (the FINAL retention
    share), not ``baseline_tvr`` or the per-break audience path. Recovering
    ``baseline_tvr`` from the final share alone treats the average-across-breaks
    retention as the final one, which overstates the cost materially (measured near
    40 percent on real days). So this refuses honestly when ``baseline_tvr`` is
    absent, naming it as the missing input, rather than shipping a biased proxy. The
    unlock is a single persisted column: write ``baseline_tvr`` on each schedule row.
    """
    columns = list(getattr(frame, "columns", []))
    for col in (revenue_col, rate_col, ad_seconds_col):
        if col not in columns:
            return _unavailable(
                f"Saved weekly schedule is missing the '{col}' column needed to "
                "value retention loss; recompute the schedule to populate it."
            )
    if baseline_tvr_col not in columns:
        return _unavailable(
            f"Saved weekly schedule does not persist '{baseline_tvr_col}', the "
            "per-segment audience needed to value retention loss exactly. The saved "
            "'retention_used' is the final share, not the per-break audience, so it "
            "cannot recover the cost without bias. Unlock: write baseline_tvr per row."
        )

    import pandas as pd  # local: keeps the pure-plan path import-light

    revenue = pd.to_numeric(frame[revenue_col], errors="coerce").fillna(0.0)
    baseline_tvr = pd.to_numeric(frame[baseline_tvr_col], errors="coerce")
    rate = pd.to_numeric(frame[rate_col], errors="coerce")
    ad_seconds = pd.to_numeric(frame[ad_seconds_col], errors="coerce").fillna(0.0)

    # A row is monetizable only with a known rating and rate and real ad seconds.
    priced = (
        rate.notna() & baseline_tvr.notna() & (rate > 0) & (ad_seconds > 0) & (revenue > 0)
    )
    if not bool(priced.any()):
        return _unavailable(
            "No saved row carries the rating, rate and ad-seconds needed to value "
            "retention loss (schedule has no monetizable breaks)."
        )

    gross_potential = rate * baseline_tvr * (ad_seconds / unit_seconds)
    # Cost is gross minus delivered revenue; clip at zero so a rounding wobble on a
    # near-full-retention row never reads as a negative cost.
    retention_cost = (gross_potential - revenue).clip(lower=0.0)

    total_revenue = float(revenue[priced].sum())
    total_cost = float(retention_cost[priced].sum())
    return {
        "available": True,
        "revenue_ils": round(total_revenue, 2),
        "retention_cost_ils": round(total_cost, 2),
        "revenue_net_ils": round(total_revenue - total_cost, 2),
        "priced_rows": int(priced.sum()),
        "basis": _basis(
            ("baseline_tvr", "retention_share", "base_rate", "ad_seconds", "unit_seconds")
        ),
    }


def compare_objectives(
    segments: Sequence[ProgramSegment],
    *,
    guardrails: Any = None,
    revenue_weight: float = 0.5,
    risk_lambda: float = 0.0,
) -> dict[str, Any]:
    """Run one channel-day under the blend objective vs the net objective, honestly.

    Runs the genuine optimizer twice on the same segments through the same entry
    point (:func:`kairos.optimize.optimizer.optimize_breaks`): once in the default
    ``blend`` mode (the convex blend that ships everywhere) and once in
    ``revenue_net`` mode (maximise ILS net directly). Each plan is then measured
    with :func:`plan_revenue_net`, so both legs report revenue, retention cost and
    net on the same honest money model. Nothing here is synthesized; the two legs
    differ only in what the optimizer maximised.

    Both legs run pure greedy (``refine=False``) so this narrow diagnostic holds
    the search tier fixed while it isolates the objective. The production
    optimizer now supports both blend and net objectives through the greedy, F1
    and exact-DP tiers; this helper deliberately disables both refiners on BOTH
    legs rather than comparing two fully refined plans.

    Both legs are priced on the plans' DECISION basis: the segments are put
    through the same risk-adjustment pre-pass the optimizer applied
    (:func:`kairos.optimize._segment_math._risk_adjusted_coefficient`, exactly as
    the dashboard's scenario pricing does) before :func:`plan_revenue_net` values
    the retention loss, so under ``risk_lambda > 0`` the reported cost matches
    what the optimizer actually decided with instead of understating it with the
    unadjusted point coefficients. At ``risk_lambda == 0`` this is an exact
    identity.

    Returns ``{blend, net, delta}`` where each side carries the plan's revenue,
    retention cost, net, break count and objective, and ``delta`` is net minus
    blend on each money field, so a caller can see exactly what the net objective
    bought or gave up in shekels.
    """
    from dataclasses import replace

    from kairos.optimize.optimizer import optimize_breaks
    from kairos.optimize._segment_math import _risk_adjusted_coefficient

    seg_list = list(segments)
    # Adjusted once, used by both legs: the decision basis both plans were made on.
    adjusted = [
        replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda))
        for s in seg_list
    ]

    def _run(mode: str) -> dict[str, Any]:
        result = optimize_breaks(
            seg_list,
            guardrails,
            revenue_weight=revenue_weight,
            risk_lambda=risk_lambda,
            refine=False,
            objective_mode=mode,
        )
        money = plan_revenue_net(result, segments=adjusted)
        return {
            "objective_mode": mode,
            "revenue_ils": money.get("revenue_ils"),
            "retention_cost_ils": money.get("retention_cost_ils"),
            "revenue_net_ils": money.get("revenue_net_ils"),
            "total_breaks": result.total_breaks,
            "objective": round(result.objective, 6),
            "compliant": result.is_compliant,
            "basis": money.get("basis"),
        }

    blend = _run("blend")
    net = _run("revenue_net")

    def _diff(key: str) -> Optional[float]:
        a, b = blend.get(key), net.get(key)
        if a is None or b is None:
            return None
        return round(float(b) - float(a), 2)

    return {
        "blend": blend,
        "net": net,
        "delta": {
            "revenue_ils": _diff("revenue_ils"),
            "retention_cost_ils": _diff("retention_cost_ils"),
            "revenue_net_ils": _diff("revenue_net_ils"),
            "total_breaks": (net["total_breaks"] - blend["total_breaks"]),
        },
        "note": (
            "Both legs are the same optimizer on the same segments; only the "
            "maximised objective differs. Figures are modeled (basis on each side)."
        ),
    }
