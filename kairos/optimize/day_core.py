"""The single break-optimization core: one channel-day, every seam assembled once.

There is exactly one engine primitive, :func:`kairos.optimize.optimizer.optimize_breaks`.
Historically it was wrapped three times with the surrounding seams (demand fold,
constraint resolution, guardrails, weights) re-assembled independently by the live
day plan, the scenario/frontier slider, and the weekly CSV export. That let the
three paths drift apart even though they answer the same question.

This module holds the shared assembly. :func:`_optimize_one_day` takes the built
segments for one channel-day plus the resolved inputs, folds the demand signal,
resolves the operator's stored constraints, and calls the primitive. "One day vs
the whole week" is now only a loop boundary: the weekly export loads the demand
resources once and calls this core in a loop, while the live paths load-then-call
once. The demand resources (advertiser engine, inventory pool, campaigns, pacing
reference date) are passed in so a 120-day export reads each source file once, not
once per day.

The demand fold and the constraint resolution themselves live in
:mod:`kairos.service`; this core calls them (via a lazy import that avoids the
service -> day_core import cycle) so there is a single implementation of each. The
optimize call is taken as a parameter so the caller supplies the function bound in
its own module, which keeps the export's demand-assembly equivalence test able to
observe the demand weights the fold produced.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Sequence

from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import OptimizationResult, optimize_breaks
from kairos.optimize.overrides import OverrideSet


def _optimize_one_day(
    segments: list,
    *,
    guardrails: Guardrails,
    revenue_weight: float,
    risk_lambda: float,
    demand_engine: Optional[Any] = None,
    inventory_pool: Optional[Mapping[Any, Any]] = None,
    campaigns: Optional[Sequence[Any]] = None,
    pacing_today: Optional[Any] = None,
    pacing_knobs: Optional[Mapping[str, Any]] = None,
    constraints: Optional[Sequence[Any]] = None,
    overrides: Optional[OverrideSet] = None,
    placement_pins: Optional[Mapping[str, Any]] = None,
    operator_channel: str = "",
    refine: bool = True,
    optimize_fn: Callable[..., OptimizationResult] = optimize_breaks,
) -> OptimizationResult:
    """Fold demand, resolve constraints and place breaks for one channel-day.

    ``segments`` are the already-built programme segments for a single channel-day
    (the caller owns segment construction because the daily-plan, programmes and
    export paths build them differently). ``guardrails``, ``revenue_weight`` (on the
    engine's [0, 1] scale) and ``risk_lambda`` are the decided policy for this run.

    ``demand_engine``, ``inventory_pool`` and ``campaigns`` are the pre-loaded
    demand resources. A single-day caller may leave them ``None``, in which case the
    demand fold loads them itself; the weekly export loads them ONCE before its loop
    and passes them in, so the fold never re-reads a file per day. ``pacing_today``
    and ``pacing_knobs`` steer the delivery-pacing signal; both are inert until real
    campaign rows land.

    ``constraints`` is the operator's pre-loaded placement-constraint list, resolved
    against these segments and merged on top of ``overrides`` / ``placement_pins``.
    ``operator_channel`` scopes those constraints to the operator's own channel.

    ``optimize_fn`` is the break-placement primitive to call; it defaults to
    :func:`~kairos.optimize.optimizer.optimize_breaks`. Callers pass the reference
    bound in their own module so a test that patches the optimizer there still sees
    the call. ``refine`` is forwarded unchanged (the frontier's ``refine=False`` is a
    deliberate performance choice, not incoherence).
    """
    # Imported lazily so this core can be imported by kairos.service at module load
    # without the reverse import cycle (service imports _optimize_one_day at top).
    from kairos.service import _assemble_demand_weights, _constraint_inputs

    demand_weights = _assemble_demand_weights(
        segments,
        today=pacing_today,
        pacing_knobs=pacing_knobs,
        engine=demand_engine,
        inventory_pool=inventory_pool,
        campaigns=campaigns,
    )
    merged_pins, merged_overrides = _constraint_inputs(
        segments, constraints, overrides, placement_pins,
        operator_channel=operator_channel,
    )
    return optimize_fn(
        segments,
        guardrails,
        revenue_weight=revenue_weight,
        risk_lambda=risk_lambda,
        overrides=merged_overrides,
        placement_pins=merged_pins,
        demand_weights=demand_weights,
        refine=refine,
    )
