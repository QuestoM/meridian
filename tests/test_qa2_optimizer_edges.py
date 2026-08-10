"""QA2 optimizer-edge regressions: finiteness, refine guard, DP budgets, pins, pricing basis.

Locks the fix wave on the optimizer engine: non-finite segment inputs raise a
clear ValueError and non-finite CI bounds degrade to None; an undersized
revenue_scale can no longer let refinement ship a plan the reported (clamped)
objective rates below pure greedy (the revert is labeled in result.notes); the
exact DP falls back honestly when its deterministic per-group work budget is
exhausted instead of stalling; the DP's per-segment count feasibility scans
every k so a positive coefficient reaches above-floor counts and a below-floor
middle k is never proposed; placement pins obey the count-pin max_breaks
contract and every rejected placement locks the segment at 0 breaks (one
documented semantic, per-segment and channel-day backout alike); the DP tier's
coverage counters ride the result; compare_objectives prices both legs on the
plans' risk-adjusted decision basis; and operator regex patterns are
length-capped with compile failures answered as a clean no-match.

Everything here is synthetic and in-process; no server, no real-data loads.
"""

from __future__ import annotations

import math
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.optimize.dp_refine import (  # noqa: E402
    DEFAULT_MAX_STATES,
    DEFAULT_MAX_WORK_UNITS,
    _allowed_break_counts,
    dp_refine_group,
)
from kairos.optimize.guardrails import Guardrails  # noqa: E402
from kairos.optimize.optimizer import (  # noqa: E402
    PlacementPin,
    ProgramSegment,
    optimize_breaks,
)
from kairos.optimize.overrides import Override, OverrideSet  # noqa: E402
from kairos.optimize.predicate import evaluate_predicate  # noqa: E402
from kairos.optimize.revenue_net import compare_objectives, plan_revenue_net  # noqa: E402
from kairos.optimize._segment_math import _risk_adjusted_coefficient, _segment_revenue  # noqa: E402

GR = Guardrails()


def make_segment(**overrides) -> ProgramSegment:
    base = dict(
        segment_id="s1",
        channel="C",
        day="2024-11-01",
        start_seconds=0.0,
        duration_seconds=3600.0,
        program_type="Movie",
        baseline_tvr=10.0,
        cpp=1000.0,
        impact_coefficient=-0.05,
        retention_baseline=1.0,
        premium=1.0,
        max_breaks=4,
        break_length_seconds=120.0,
        unit_seconds=1.0,
    )
    base.update(overrides)
    return ProgramSegment(**base)


# --------------------------------------------------------------------------- #
# optedge-1: finiteness is validated, never fabricated through clamps.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "field,value",
    [
        ("impact_coefficient", float("nan")),
        ("baseline_tvr", float("inf")),
        ("cpp", float("nan")),
        ("premium", float("inf")),
        ("start_seconds", float("nan")),
        ("duration_seconds", float("inf")),
        ("break_length_seconds", float("nan")),
        ("unit_seconds", float("inf")),
        ("first_break_multiplier", float("inf")),
    ],
)
def test_non_finite_segment_input_raises_named_valueerror(field, value) -> None:
    segment = make_segment(**{field: value})
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        optimize_breaks([segment], GR)


def test_nan_retention_baseline_raises_finiteness_error_first() -> None:
    segment = make_segment(retention_baseline=float("nan"))
    with pytest.raises(ValueError, match="retention_baseline must be finite"):
        segment.validate()


def test_non_finite_ci_bounds_degrade_to_point_only() -> None:
    segment = make_segment(impact_ci_low=float("nan"), impact_ci_high=-0.01)
    assert segment.impact_ci_low is None and segment.impact_ci_high is None
    # A replace() copy stays clean too (post-init runs on construction).
    copied = replace(make_segment(), impact_ci_low=-0.06, impact_ci_high=float("inf"))
    assert copied.impact_ci_low is None and copied.impact_ci_high is None
    # Finite bounds survive untouched.
    fine = make_segment(impact_ci_low=-0.06, impact_ci_high=-0.01)
    assert fine.impact_ci_low == -0.06 and fine.impact_ci_high == -0.01


# --------------------------------------------------------------------------- #
# optedge-2: refinement can never land below greedy on the reported objective.
# --------------------------------------------------------------------------- #
def _four_disjoint_segments() -> list[ProgramSegment]:
    return [
        make_segment(segment_id=f"s{i}", start_seconds=i * 7200.0, duration_seconds=3600.0)
        for i in range(4)
    ]


def test_undersized_scale_refine_never_below_greedy_and_is_labeled() -> None:
    """The audit case: revenue_scale far below real revenue made the unclamped
    refiner gates adopt counts the clamped reported objective rates below pure
    greedy (measured 0.985 -> 0.910). The final guard reverts and says so."""
    segs = _four_disjoint_segments()
    greedy = optimize_breaks(segs, GR, revenue_weight=0.6, revenue_scale=1000.0, refine=False)
    refined = optimize_breaks(segs, GR, revenue_weight=0.6, revenue_scale=1000.0, refine=True)
    assert refined.objective >= greedy.objective - 1e-12, (
        f"refine landed below greedy on the reported objective: "
        f"{refined.objective} < {greedy.objective}"
    )
    # The guard actually fired here (the gates did try to adopt worse counts),
    # the plan is exactly the greedy plan, and the switch is labeled, not silent.
    assert refined.total_breaks == greedy.total_breaks
    assert refined.objective == greedy.objective
    assert len(refined.notes) == 1 and "reverted" in refined.notes[0]


def test_default_scale_keeps_refinement_and_carries_no_notes() -> None:
    segs = _four_disjoint_segments()
    result = optimize_breaks(segs, GR, revenue_weight=0.6)
    assert result.notes == ()
    greedy = optimize_breaks(segs, GR, revenue_weight=0.6, refine=False)
    assert result.objective >= greedy.objective - 1e-12


# --------------------------------------------------------------------------- #
# optedge-3: DP budgets fall back honestly instead of stalling.
# --------------------------------------------------------------------------- #
_LOOSE = Guardrails(
    max_ad_seconds_per_hour=1e9,
    max_breaks_per_hour=1000,
    min_break_spacing_seconds=1.0,
    min_retention_floor=0.0,
    max_daily_ad_seconds=1e9,
    protected_max_ad_seconds_per_hour=1e9,
    gold_breaks_max_per_day=1000,
)


def _pathological_group(n: int = 8) -> list[ProgramSegment]:
    """n long overlapping segments: every one stays open across the whole sweep,
    so the joint state table grows like 5^stage and only a budget can stop it."""
    return [
        make_segment(
            segment_id=f"p{i}",
            start_seconds=i * 3700.0,
            duration_seconds=40000.0,
            impact_coefficient=-0.001,
        )
        for i in range(n)
    ]


def _dp_call(group, guardrails, **kwargs):
    scale = max(sum(_segment_revenue(s, s.max_breaks) for s in group), 1e-9)
    total_tvr = sum(s.baseline_tvr for s in group)
    zero = {s.segment_id: 0 for s in group}
    return dp_refine_group(
        group, zero, guardrails, revenue_weight=0.6, revenue_scale=scale,
        total_tvr=total_tvr, objective_mode="blend", **kwargs,
    )


def test_state_budget_exhaustion_falls_back_labeled_not_hang() -> None:
    group = _pathological_group()
    start = time.perf_counter()
    out = _dp_call(group, _LOOSE, max_states=1000)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0, f"budget fallback must be fast, took {elapsed:.2f}s"
    assert out.fell_back
    assert out.reason_code == "state_budget"
    assert "exceeds the 1000 budget" in out.reason
    assert out.counts == {s.segment_id: 0 for s in group}  # input kept, never-worse


def test_work_budget_exhaustion_falls_back_labeled_and_deterministic() -> None:
    group = _pathological_group()
    start = time.perf_counter()
    out = _dp_call(group, _LOOSE, max_work_units=1)
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"budget fallback must be fast, took {elapsed:.2f}s"
    assert out.fell_back
    assert out.reason_code == "work_budget"
    assert "transition count" in out.reason
    assert out.counts == {s.segment_id: 0 for s in group}


def test_default_budgets_clear_an_ordinary_day() -> None:
    """The budgets are backstops: a small ordinary group stays on the exact path
    (the real corpus peaks at 40,485 states / 0.70s, well inside the defaults)."""
    assert DEFAULT_MAX_STATES == 200_000
    assert DEFAULT_MAX_WORK_UNITS == 5_000_000
    group = [
        make_segment(segment_id="a", start_seconds=0.0),
        make_segment(segment_id="b", start_seconds=1800.0),
    ]
    out = _dp_call(group, GR)
    assert not out.fell_back and out.reason == "" and out.reason_code == ""


# --------------------------------------------------------------------------- #
# optedge-4: allowed counts scan every k (positive coefficients included).
# --------------------------------------------------------------------------- #
def _rising_retention_segment(max_breaks: int) -> ProgramSegment:
    # retention(k) = 0.6 + 0.1k: k=1 -> 0.70 (below the 0.72 floor),
    # k=2 -> 0.80 and k=3 -> 0.90 (above it).
    return make_segment(
        segment_id="p",
        duration_seconds=7200.0,
        retention_baseline=0.6,
        impact_coefficient=0.1,
        max_breaks=max_breaks,
    )


def test_allowed_counts_reach_above_floor_k_beyond_a_below_floor_one() -> None:
    seg = _rising_retention_segment(max_breaks=3)
    assert _allowed_break_counts([seg], GR) == [[0, 2, 3]]
    out = _dp_call([seg], GR)
    assert not out.fell_back
    # The old first-failure break capped this segment at 0; the DP now reaches
    # the feasible, above-floor k=3 (max revenue on a compliant retention).
    assert out.counts == {"p": 3}


def test_below_floor_middle_k_is_never_proposed() -> None:
    seg = _rising_retention_segment(max_breaks=1)  # only k=1 exists and it breaches
    assert _allowed_break_counts([seg], GR) == [[0]]
    out = _dp_call([seg], GR)
    assert not out.fell_back
    assert out.counts == {"p": 0}


# --------------------------------------------------------------------------- #
# optedge-5: placement pins obey the count-pin max_breaks contract.
# --------------------------------------------------------------------------- #
def _pins(n: int, spacing: float = 700.0) -> list[PlacementPin]:
    return [
        PlacementPin(offset_seconds=200.0 + i * spacing, duration_seconds=120.0)
        for i in range(n)
    ]


def test_placement_pins_above_max_breaks_are_rejected_and_locked_to_zero() -> None:
    segment = make_segment(max_breaks=2, impact_coefficient=0.0)
    result = optimize_breaks(
        [segment], GR, revenue_weight=1.0, placement_pins={"s1": _pins(3)},
    )
    rejected = [r for r in result.rejected_overrides if r.segment_id == "s1"]
    assert len(rejected) == 1
    assert rejected[0].kind == "placement"
    assert rejected[0].requested == 3
    assert rejected[0].reason == "pinned count 3 exceeds max_breaks 2"
    # The unified rejected-placement semantic: locked at 0 breaks, even though
    # revenue-only free optimization would fill the segment to max_breaks.
    assert result.segments[0].num_breaks == 0
    assert result.placements == ()


def test_count_pin_above_max_breaks_uses_the_same_wording() -> None:
    segment = make_segment(max_breaks=4, impact_coefficient=0.0)
    overrides = OverrideSet(overrides=[
        Override(override_id="o1", scope="segment", target_id="s1", kind="pin", value="5"),
    ])
    result = optimize_breaks([segment], GR, revenue_weight=1.0, overrides=overrides)
    rejected = [r for r in result.rejected_overrides if r.segment_id == "s1"]
    assert len(rejected) == 1
    assert rejected[0].kind == "pin"
    assert rejected[0].reason == "pinned count 5 exceeds max_breaks 4"


def test_placement_pins_at_exactly_max_breaks_are_honored() -> None:
    segment = make_segment(max_breaks=2, impact_coefficient=0.0)
    result = optimize_breaks(
        [segment], GR, revenue_weight=1.0, placement_pins={"s1": _pins(2)},
    )
    assert not result.rejected_overrides
    assert result.segments[0].num_breaks == 2


# --------------------------------------------------------------------------- #
# math-overflow-5: the channel-day backout applies the same lock-to-zero.
# --------------------------------------------------------------------------- #
def test_group_level_placement_backout_locks_the_segment_to_zero() -> None:
    a = make_segment(segment_id="a", start_seconds=0.0, impact_coefficient=0.0)
    b = make_segment(segment_id="b", start_seconds=3600.0, impact_coefficient=0.0)
    # Each pin set is valid on its own; together the last break of "a" and the
    # break of "b" sit 180s apart, breaching the 420s channel-day spacing.
    pins = {
        "a": [
            PlacementPin(offset_seconds=2800.0, duration_seconds=120.0),
            PlacementPin(offset_seconds=3400.0, duration_seconds=120.0),
        ],
        "b": [PlacementPin(offset_seconds=100.0, duration_seconds=120.0)],
    }
    result = optimize_breaks([a, b], GR, revenue_weight=1.0, placement_pins=pins)
    rejected = [r for r in result.rejected_overrides if r.segment_id == "a"]
    assert len(rejected) == 1 and rejected[0].kind == "placement"
    assert "channel-day" in rejected[0].reason
    by_id = {p.segment_id: p for p in result.segments}
    # The backed-out segment is LOCKED at 0 (previously it was freed to
    # max_breaks, contradicting the per-segment rejection semantic).
    assert by_id["a"].num_breaks == 0
    # The surviving pinned segment keeps its exact pin.
    assert by_id["b"].num_breaks == 1
    assert result.is_compliant


# --------------------------------------------------------------------------- #
# optedge-6: DP coverage counters ride the result.
# --------------------------------------------------------------------------- #
def test_dp_stats_report_coverage_and_fallback_histogram() -> None:
    segs = [make_segment(segment_id="a"), make_segment(segment_id="b", start_seconds=7200.0)]
    result = optimize_breaks(segs, GR, revenue_weight=0.6)
    stats = result.dp_stats
    assert stats is not None
    assert stats["groups_total"] == 1
    assert stats["groups_exact"] + sum(stats["fallback_reasons"].values()) == 1
    for key in ("groups_adopted", "groups_not_better", "groups_noncompliant"):
        assert key in stats

    # A pinned group is a labeled fallback, histogrammed under its stable code.
    pinned = optimize_breaks(
        segs, GR, revenue_weight=0.6,
        placement_pins={"a": [PlacementPin(offset_seconds=600.0, duration_seconds=120.0)]},
    )
    assert pinned.dp_stats["fallback_reasons"] == {"placement_pins": 1}
    assert pinned.dp_stats["groups_exact"] == 0


def test_dp_stats_absent_when_the_tier_does_not_run() -> None:
    segs = [make_segment()]
    assert optimize_breaks(segs, GR, dp_refine=False).dp_stats is None
    assert optimize_breaks(segs, GR, refine=False).dp_stats is None


# --------------------------------------------------------------------------- #
# math-compare-objectives-risk-mix: pricing on the decision basis.
# --------------------------------------------------------------------------- #
def _uncertain_segments() -> list[ProgramSegment]:
    return [
        make_segment(
            segment_id=f"u{i}", start_seconds=i * 7200.0,
            impact_coefficient=-0.02, impact_ci_low=-0.08, impact_ci_high=-0.01,
            max_breaks=3,
        )
        for i in range(2)
    ]


def test_compare_objectives_prices_with_risk_adjusted_segments() -> None:
    segs = _uncertain_segments()
    risk = 1.0
    report = compare_objectives(segs, guardrails=GR, revenue_weight=0.6, risk_lambda=risk)

    # Reproduce the blend leg exactly and price it on the decision basis.
    result = optimize_breaks(
        segs, GR, revenue_weight=0.6, risk_lambda=risk, refine=False, objective_mode="blend",
    )
    assert result.total_breaks > 0, "the leg must place breaks for the cost to bind"
    adjusted = [
        replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk)) for s in segs
    ]
    expected = plan_revenue_net(result, segments=adjusted)
    assert report["blend"]["retention_cost_ils"] == expected["retention_cost_ils"]
    assert report["blend"]["revenue_net_ils"] == expected["revenue_net_ils"]

    # The old mixed basis (unadjusted points) understates the cost; the report
    # must sit on the adjusted, larger figure.
    understated = plan_revenue_net(result, segments=segs)
    assert expected["retention_cost_ils"] > understated["retention_cost_ils"]
    assert report["blend"]["retention_cost_ils"] > understated["retention_cost_ils"]


def test_compare_objectives_at_zero_risk_is_the_identity_basis() -> None:
    segs = _uncertain_segments()
    report = compare_objectives(segs, guardrails=GR, revenue_weight=0.6, risk_lambda=0.0)
    result = optimize_breaks(
        segs, GR, revenue_weight=0.6, risk_lambda=0.0, refine=False, objective_mode="blend",
    )
    expected = plan_revenue_net(result, segments=segs)
    assert report["blend"]["retention_cost_ils"] == expected["retention_cost_ils"]


# --------------------------------------------------------------------------- #
# sec-predicate-regex: pattern length cap and clean compile-failure handling.
# --------------------------------------------------------------------------- #
def _predicate_segment(title: str) -> SimpleNamespace:
    return SimpleNamespace(
        program_title=title, program_type="Drama", day="2024-11-01",
        start_seconds=0.0, channel="C",
    )


def _regex_group(pattern: str) -> dict:
    return {
        "combinator": "and",
        "conditions": [{"field": "programme", "operator": "regex", "value": pattern}],
    }


def test_regex_pattern_above_cap_is_a_clean_no_match() -> None:
    seg = _predicate_segment("a" * 600)
    assert evaluate_predicate(_regex_group("a" * 513), seg) is False


def test_regex_pattern_at_cap_still_matches() -> None:
    seg = _predicate_segment("a" * 600)
    assert evaluate_predicate(_regex_group("a" * 512), seg) is True


def test_malformed_regex_is_a_clean_no_match_not_an_exception() -> None:
    seg = _predicate_segment("Evening News")
    assert evaluate_predicate(_regex_group("(unclosed"), seg) is False
    # A pattern the re engine rejects beyond re.error (huge but under the length
    # cap once expanded by counted repetition) is also a clean no-match.
    assert evaluate_predicate(_regex_group("(a{100}){100}" * 30), seg) is False


def test_reported_objective_stays_finite_on_a_normal_run() -> None:
    """Belt-and-braces for the explicit raises: a clean run reports finite money."""
    result = optimize_breaks(_four_disjoint_segments(), GR, revenue_weight=0.6)
    assert math.isfinite(result.objective) and math.isfinite(result.total_revenue)
