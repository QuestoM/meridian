"""What the demand-weight steer actually reaches, measured 2026-08-09.

Three placement levers fold into one ``demand_weights`` map: advertiser demand,
inventory awareness and delivery pacing (kairos.service._assemble_demand_weights).
The map is read in exactly ONE place, the greedy loop's ranking comparison
(``kairos/optimize/optimizer.py``, the ``demand_weights is not None`` branch).
The two refiner tiers that run AFTER greedy on the shipped default -- the F1
refiner (``optimize_group``) and the exact DP tier (``apply_dp_tier``) -- are not
given the map and have no parameter to receive it. They climb the reported
objective from greedy's warm start, so a plan greedy shaped by weight is climbed
back toward the unweighted optimum.

Measured on the operator's 30 real channel-days with weights spanning 0.77..2.00:
``refine=False`` moved 754 segment break-counts across 30 of 30 days;
``refine=True`` (the shipped default) moved 2, on 1 of 30 days. Shipping the
weighted plan instead of the refined one would have cost 5,987,373.30 ILS across
those 30 days, -14.62%.

These tests PIN that behaviour rather than bless it. They fail the moment anyone
threads the weights into either refiner tier -- which is the intended alarm,
because that change moves real money and must be measured, not assumed.

This is the first caller of :mod:`tests.lever_probe`, the reusable guard for THE
INERT LEVER defect class. Instances two and three (the hourly ad-minutes cap, the
inventory loader's silent discard) are named in that module and can be added by
anyone who writes a ``run``, a ``binds`` and a settings range.
"""

from __future__ import annotations

import inspect

from kairos.optimize import dp_refine, refiner
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks
from tests.lever_probe import assert_lever_bites, assert_lever_is_inert, probe_lever


def _day(count: int = 24) -> list[ProgramSegment]:
    """One synthetic channel-day whose DAILY CAP BINDS, as every real day does.

    Sizing matters. A weight only changes the plan when something scarce forces a
    choice: 24 segments x max 4 breaks is 96 candidate breaks against the default
    ``max_daily_ad_seconds`` of 9600s / 120s = 80, so 16 breaks must lose. Below
    the cap greedy simply gives every segment its maximum and the ranking order is
    irrelevant -- which is itself part of the finding, and why an undersized
    fixture would make these tests pass vacuously.

    Varied tvr/cpp so the segments are genuinely rankable: a weight map has real
    ordering work to do, rather than breaking arbitrary ties.
    """
    return [
        ProgramSegment(
            segment_id=f"s{i:02d}",
            channel="ch1",
            day="2026-01-05",
            start_seconds=i * 3600.0,
            duration_seconds=3600.0,
            program_type="Drama",
            baseline_tvr=4.0 + (i % 5) * 1.5,
            cpp=800.0 + (i % 4) * 150.0,
            impact_coefficient=-0.012,
            max_breaks=4,
            break_length_seconds=120.0,
        )
        for i in range(count)
    ]


def _spanning(segments: list[ProgramSegment]) -> dict[str, float]:
    """Weights spanning 0.77..2.00, the range a configured lever really produces."""
    span = [0.77, 2.00, 1.10, 0.85, 1.75, 0.95, 1.40, 0.80, 1.90, 1.05, 0.90, 1.60]
    return {s.segment_id: span[i % len(span)] for i, s in enumerate(segments)}


def _binding(counts: dict[str, int]) -> bool:
    """Did the daily cap actually bind? 96 candidate breaks, 80 allowed."""
    return sum(counts.values()) < 96


def _counts(result) -> dict[str, int]:
    return {p.segment_id: p.num_breaks for p in result.segments}


def _plan(segments, weights, *, refine: bool) -> dict[str, int]:
    return _counts(optimize_breaks(
        segments, Guardrails(), revenue_weight=0.6,
        demand_weights=weights, refine=refine,
    ))


# ---------------------------------------------------------------------------
# The structural fact: neither refiner tier can even receive the weights.
# ---------------------------------------------------------------------------


def test_neither_refiner_tier_accepts_demand_weights() -> None:
    """The F1 refiner and the exact DP tier have no demand_weights parameter.

    Fails if someone adds one, which is the point: adding it is a money change.
    """
    assert "demand_weights" not in inspect.signature(refiner.optimize_group).parameters
    assert "demand_weights" not in inspect.signature(dp_refine.apply_dp_tier).parameters


def test_demand_weights_are_not_passed_to_either_tier() -> None:
    """On a real refine=True run, no tier is handed the weight map."""
    seen: list[tuple[str, tuple[str, ...]]] = []
    original_group, original_dp = refiner.optimize_group, dp_refine.apply_dp_tier

    def spy_group(*args, **kwargs):
        seen.append(("F1", tuple(sorted(kwargs))))
        return original_group(*args, **kwargs)

    def spy_dp(*args, **kwargs):
        seen.append(("DP", tuple(sorted(kwargs))))
        return original_dp(*args, **kwargs)

    refiner.optimize_group = spy_group
    dp_refine.apply_dp_tier = spy_dp
    try:
        segments = _day()
        optimize_breaks(
            segments, Guardrails(), revenue_weight=0.6,
            demand_weights=_spanning(segments), refine=True,
        )
    finally:
        refiner.optimize_group = original_group
        dp_refine.apply_dp_tier = original_dp

    assert seen, "no refiner tier ran; this test would pass vacuously"
    for tier, kwargs in seen:
        assert "demand_weights" not in kwargs, f"{tier} tier was handed the weight map"


# ---------------------------------------------------------------------------
# The behavioural consequence: the steer survives greedy and not refinement.
# ---------------------------------------------------------------------------


def _probe(*, refine: bool):
    """The demand-weight lever across its range, at one refinement setting."""
    segments = _day()
    return probe_lever(
        name=f"demand weights (refine={refine})",
        run=lambda weights: _plan(segments, weights, refine=refine),
        settings=[{s.segment_id: 1.0 for s in segments}, _spanning(segments)],
        binds=_binding,
    )


def test_demand_weights_move_the_plan_when_refinement_is_off() -> None:
    """Without refinement the steer is real: the weighted plan differs.

    This is the control. If it ever stops biting, the weight map has stopped
    reaching greedy too and the lever is inert end to end.
    """
    assert_lever_bites(_probe(refine=False))


def test_refinement_erases_the_demand_steer_on_the_shipped_default() -> None:
    """With refine=True (shipped) the same weights change nothing.

    Measured, not assumed: the refiner tiers climb the reported objective from
    greedy's warm start and never read the weights, so the weighted and unweighted
    plans converge. If this fails because the plans now DIFFER, someone has wired
    the steer into a refiner tier; re-measure the revenue before adopting it (the
    30-day operator measurement above put the price at -14.62%).
    """
    assert_lever_is_inert(
        _probe(refine=True),
        because=(
            "the F1 refiner and the exact DP tier never receive demand_weights "
            "(optimizer.py call sites), so they climb the reported objective back "
            "out of any bias greedy took on"
        ),
    )
