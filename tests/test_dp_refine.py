"""Tests for the exact DP refiner tier (kairos/optimize/dp_refine.py).

Five gates, each checkable:

  * oracle: the DP's objective equals an exhaustive brute-force enumeration on
    small synthetic channel-days, over several seeds, both objective modes, and
    with the risk_lambda pre-pass exercised;
  * never-worse: on real channel-days through the shipped optimizer, the DP tier
    on never scores below the greedy+F1 plan and always stays compliant;
  * flag-off byte identity: with the DP tier disabled the result is identical to
    the pre-change (no-DP) path, to the cent, on a real day;
  * preconditions: overrides, pins, gold-forcing, non-finite input, mixed break
    lengths, and an exceeded depth guard each trip the honest silent fallback;
  * depth-13 day: the real Kan 11 2024-11-09 (open depth 13) runs on the exact
    path without tripping the default guard, inside a stated time bound.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import replace
from itertools import product

import pytest

from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.model.impact import load_impact_model
from kairos.optimize.dp_refine import (
    DEFAULT_MAX_OPEN_DEPTH,
    OBJECTIVE_BLEND,
    OBJECTIVE_REVENUE_NET,
    dp_refine_group,
)
from kairos.optimize.guardrails import Guardrails, evaluate, is_compliant
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings
from kairos.optimize.revenue_net import segment_net_revenue
from kairos.optimize._segment_math import (
    _group_breaks,
    _group_objective_contribution,
    _risk_adjusted_coefficient,
    _segment_revenue,
)
from kairos.optimize._types import PlacementPin, ProgramSegment
from kairos.service import (
    DEFAULT_IMPACT_MODEL_PATH,
    _apply_first_break_multiplier,
    _build_classifier,
)

GR = Guardrails()
_TOL = 1e-6

# Real channel-days used by the never-worse and identity gates: the largest day,
# the depth-13 day, and two more spread across channels. Hebrew channel strings are
# the exact values the data carries (never romanized).
DEPTH13_DAY = ("כאן 11", "2024-11-09")
LARGEST_DAY = ("קשת 12", "2024-11-22")
REAL_DAYS = [LARGEST_DAY, DEPTH13_DAY, ("רשת 13", "2024-11-03"), ("קשת 12", "2024-11-30")]


# ---------------------------------------------------------------------------
# Real-data loading (module-scoped so the impact model loads once)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def loader():
    assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
    pricing = pricing_from_settings(None, None)
    classifier = _build_classifier()
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    programmes = load_programmes()

    def build(channel, day):
        return build_segments_from_programmes(
            programmes, classifier, pricing,
            assumptions=assumptions, impact_model=impact_model,
            channel=channel, day=day,
        )

    return build


def _prep(segs, *, risk_lambda):
    """Risk pre-pass and global normalisers, mirroring optimize_breaks exactly."""
    adj = [replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda))
           for s in segs]
    scale = max(sum(_segment_revenue(s, s.max_breaks) for s in adj), 1e-9)
    total_tvr = sum(s.baseline_tvr for s in adj)
    return adj, scale, total_tvr


def _score(group, counts, scale, total_tvr, *, mode, revenue_weight):
    """The shipped per-group objective the adoption gate measures."""
    if mode == OBJECTIVE_REVENUE_NET:
        return sum(segment_net_revenue(s, counts[s.segment_id]) for s in group)
    contribution, _, _ = _group_objective_contribution(
        group, counts, revenue_weight=revenue_weight,
        revenue_scale=scale, total_tvr=total_tvr,
    )
    return contribution


# ---------------------------------------------------------------------------
# (a) Oracle gate: DP == brute force on small synthetic channel-days
# ---------------------------------------------------------------------------
_PROGRAM_TYPES = ["Movie", "Series", "Sport", "News", "Children", "Entertainment"]


def _guardrail_profile(rng):
    base = dict(
        max_ad_seconds_per_hour=720.0, max_breaks_per_hour=4,
        min_break_spacing_seconds=420.0, min_retention_floor=0.72,
        max_daily_ad_seconds=9600.0, protected_max_ad_seconds_per_hour=480.0,
        gold_breaks_max_per_day=3,
    )
    kind = rng.choice(["loose", "tight_daily", "tight_gold", "tight_hour", "protected"])
    if kind == "tight_daily":
        base["max_daily_ad_seconds"] = rng.choice([360.0, 480.0, 600.0, 720.0])
    elif kind == "tight_gold":
        base["gold_breaks_max_per_day"] = rng.choice([1, 2])
    elif kind == "tight_hour":
        base["max_breaks_per_hour"] = rng.choice([1, 2, 3])
        base["max_ad_seconds_per_hour"] = rng.choice([240.0, 360.0, 480.0])
    elif kind == "protected":
        base["protected_max_ad_seconds_per_hour"] = rng.choice([120.0, 240.0])
    return Guardrails(**base), kind


def _make_segment(rng, idx, kind):
    ptype = rng.choice(_PROGRAM_TYPES)
    if kind == "protected" and rng.random() < 0.5:
        ptype = rng.choice(["News", "Children", "Kids"])
    coeff = -rng.uniform(0.0, 0.09)
    seg = ProgramSegment(
        segment_id=f"s{idx}", channel="C", day="2024-11-01",
        start_seconds=rng.uniform(0, 4 * 3600.0),
        duration_seconds=rng.uniform(900.0, 4500.0), program_type=ptype,
        baseline_tvr=rng.uniform(0.5, 12.0), cpp=rng.uniform(500.0, 5000.0),
        impact_coefficient=coeff, retention_baseline=rng.uniform(0.9, 1.0),
        premium=rng.choice([1.0, 1.0, 1.15, 1.3]),
        is_gold=(kind == "tight_gold" and rng.random() < 0.6),
        max_breaks=rng.randint(1, 4), break_length_seconds=120.0,
    )
    if rng.random() < 0.3:
        lo = coeff - rng.uniform(0.01, 0.05)
        seg = replace(seg, impact_ci_low=lo, impact_ci_high=min(0.0, coeff + 0.01))
    return seg


def _brute_best(group, scale, total_tvr, guardrails, *, mode, revenue_weight):
    best = -math.inf
    for vec in product(*[range(s.max_breaks + 1) for s in group]):
        counts = {s.segment_id: k for s, k in zip(group, vec)}
        if not is_compliant(_group_breaks(group, counts), guardrails):
            continue
        val = _score(group, counts, scale, total_tvr, mode=mode, revenue_weight=revenue_weight)
        best = max(best, val)
    return best


@pytest.mark.parametrize("seed", [1, 7, 42, 2024, 20260707])
def test_oracle_dp_matches_brute_force(seed):
    """The DP's objective equals the exhaustive optimum, both modes, seeded."""
    rng = random.Random(seed)
    checked = 0
    for _ in range(40):
        guardrails, kind = _guardrail_profile(rng)
        n = rng.randint(3, 6)
        segs = [_make_segment(rng, i, kind) for i in range(n)]
        mode = rng.choice([OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET])
        revenue_weight = rng.uniform(0.0, 1.0)
        risk_lambda = rng.choice([0.0, 0.0, 0.5, 1.0])
        adj, scale, total_tvr = _prep(segs, risk_lambda=risk_lambda)
        zero = {s.segment_id: 0 for s in adj}
        outcome = dp_refine_group(
            adj, zero, guardrails, revenue_weight=revenue_weight,
            revenue_scale=scale, total_tvr=total_tvr, objective_mode=mode,
            net_of=segment_net_revenue,
        )
        if outcome.fell_back:
            continue
        brute = _brute_best(adj, scale, total_tvr, guardrails,
                            mode=mode, revenue_weight=revenue_weight)
        assert outcome.dp_objective == pytest.approx(brute, abs=_TOL, rel=_TOL), (
            f"seed={seed} mode={mode} dp={outcome.dp_objective} brute={brute}"
        )
        # The reconstructed counts must score to the same optimum and be compliant.
        replayed = _score(adj, outcome.counts, scale, total_tvr,
                          mode=mode, revenue_weight=revenue_weight)
        assert replayed == pytest.approx(brute, abs=_TOL, rel=_TOL)
        assert is_compliant(_group_breaks(adj, outcome.counts), guardrails)
        checked += 1
    assert checked >= 20, f"too few instances stayed on the exact path ({checked})"


# ---------------------------------------------------------------------------
# (b) Never-worse on real channel-days through the shipped optimizer
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("channel,day", REAL_DAYS)
@pytest.mark.parametrize("mode", [OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET])
def test_never_worse_on_real_days(loader, channel, day, mode):
    """DP tier on scores >= greedy+F1 and ships a compliant plan, both modes."""
    segs = loader(channel, day)
    assert segs, f"no segments for {channel} {day}"
    off = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                          refine=True, dp_refine=False, objective_mode=mode)
    on = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                         refine=True, dp_refine=True, objective_mode=mode)
    assert on.objective >= off.objective - _TOL, (
        f"DP tier regressed {channel} {day} {mode}: {on.objective} < {off.objective}"
    )
    assert not evaluate(
        _group_breaks(list(segs), {p.segment_id: p.num_breaks for p in on.segments}), GR
    ), "DP-adopted plan is not compliant"


def test_dp_tier_strictly_improves_a_known_day(loader):
    """A guard against a no-op tier: the largest day must actually improve on.

    Measured on the true net-ILS money basis (the reported ``objective`` field is
    always the blend scalar regardless of mode), so the improvement is stated in
    real shekels, not a normalised contribution.
    """
    segs = loader(*LARGEST_DAY)
    by_id = {s.segment_id: s for s in segs}

    def money(res):
        return sum(segment_net_revenue(by_id[p.segment_id], p.num_breaks) for p in res.segments)

    off = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                          refine=True, dp_refine=False, objective_mode=OBJECTIVE_REVENUE_NET)
    on = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                         refine=True, dp_refine=True, objective_mode=OBJECTIVE_REVENUE_NET)
    assert money(on) > money(off) + 1000.0, "DP tier did not improve the largest day"


# ---------------------------------------------------------------------------
# (c) Flag-off byte identity vs the pre-change (no-DP) path
# ---------------------------------------------------------------------------
def test_flag_off_is_byte_identical(loader, monkeypatch):
    """dp_refine=False equals the DP-disabled path to the cent on a real day.

    The DP tier is the only thing the flag gates, so disabling the flag must
    reproduce the pre-change result exactly. Proven by comparing dp_refine=False
    against dp_refine=True with apply_dp_tier neutralised to a no-op (the code the
    engine ran before this tier existed).
    """
    segs = loader(*LARGEST_DAY)

    def _noop(*args, **kwargs):
        return None

    monkeypatch.setattr("kairos.optimize.dp_refine.apply_dp_tier", _noop)
    reference = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                               refine=True, dp_refine=True, objective_mode=OBJECTIVE_BLEND)
    monkeypatch.undo()
    off = optimize_breaks(segs, GR, revenue_weight=0.6, risk_lambda=0.0,
                          refine=True, dp_refine=False, objective_mode=OBJECTIVE_BLEND)

    assert off.objective == reference.objective
    assert off.total_revenue == reference.total_revenue
    assert off.aggregate_retention == reference.aggregate_retention
    ref_counts = {p.segment_id: p.num_breaks for p in reference.segments}
    off_counts = {p.segment_id: p.num_breaks for p in off.segments}
    assert off_counts == ref_counts
    assert len(off.placements) == len(reference.placements)


# ---------------------------------------------------------------------------
# (d) Every precondition trips the honest silent fallback
# ---------------------------------------------------------------------------
def _two_overlapping_segments():
    a = ProgramSegment(
        segment_id="a", channel="C", day="2024-11-01", start_seconds=0.0,
        duration_seconds=3600.0, program_type="Movie", baseline_tvr=5.0, cpp=2000.0,
        impact_coefficient=-0.03, max_breaks=3, break_length_seconds=120.0,
    )
    b = ProgramSegment(
        segment_id="b", channel="C", day="2024-11-01", start_seconds=1800.0,
        duration_seconds=3600.0, program_type="Series", baseline_tvr=4.0, cpp=1800.0,
        impact_coefficient=-0.03, max_breaks=3, break_length_seconds=120.0,
    )
    return [a, b]


def _call(group, **kwargs):
    scale = max(sum(_segment_revenue(s, s.max_breaks) for s in group), 1e-9)
    total_tvr = sum(s.baseline_tvr for s in group)
    zero = {s.segment_id: 0 for s in group}
    return dp_refine_group(
        group, zero, GR, revenue_weight=0.6, revenue_scale=scale,
        total_tvr=total_tvr, objective_mode=OBJECTIVE_BLEND, **kwargs,
    )


def test_precondition_placement_pins_falls_back():
    group = _two_overlapping_segments()
    out = _call(group, placements={"a": [PlacementPin(offset_seconds=600.0, duration_seconds=120.0)]})
    assert out.fell_back and "placement pins" in out.reason
    assert out.counts == {"a": 0, "b": 0}  # the input counts, untouched


def test_precondition_override_floor_falls_back():
    group = _two_overlapping_segments()
    out = _call(group, floors={"a": 1})
    assert out.fell_back and "override floor" in out.reason


def test_precondition_override_cap_falls_back():
    group = _two_overlapping_segments()
    out = _call(group, caps={"a": 0})
    assert out.fell_back and "override cap" in out.reason


def test_precondition_gold_forcing_falls_back():
    group = _two_overlapping_segments()
    out = _call(group, gold_by_id={"a": True})
    assert out.fell_back and "gold-forcing" in out.reason


def test_precondition_non_finite_input_falls_back():
    group = _two_overlapping_segments()
    group[0] = replace(group[0], impact_coefficient=float("nan"))
    out = _call(group)
    assert out.fell_back and "non-finite" in out.reason


def test_precondition_mixed_break_lengths_falls_back():
    group = _two_overlapping_segments()
    group[1] = replace(group[1], break_length_seconds=90.0)
    out = _call(group)
    assert out.fell_back and "break length" in out.reason


def test_precondition_depth_guard_falls_back():
    group = _two_overlapping_segments()  # overlapping -> open depth >= 1
    out = _call(group, max_open_depth=0)
    assert out.fell_back and "open depth" in out.reason


def test_covered_group_does_not_fall_back():
    """The free path clears every precondition and runs the exact DP."""
    group = _two_overlapping_segments()
    out = _call(group)
    assert not out.fell_back and out.reason == ""
    assert out.max_open_depth >= 1 and out.peak_states >= 1


# ---------------------------------------------------------------------------
# (e) The depth-13 real day runs on the exact path within a time bound
# ---------------------------------------------------------------------------
def test_depth13_day_runs_on_exact_path(loader):
    """Kan 11 2024-11-09 reaches open depth 13, does NOT trip the default guard,
    and completes well under a conservative time bound on both objective modes."""
    segs = loader(*DEPTH13_DAY)
    assert segs, "no segments for the depth-13 day"
    adj, scale, total_tvr = _prep(segs, risk_lambda=0.0)
    zero = {s.segment_id: 0 for s in adj}
    assert DEFAULT_MAX_OPEN_DEPTH >= 14  # guard clears the measured depth of 13
    for mode in (OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET):
        start = time.perf_counter()
        out = dp_refine_group(
            adj, zero, GR, revenue_weight=0.6, revenue_scale=scale,
            total_tvr=total_tvr, objective_mode=mode, net_of=segment_net_revenue,
        )
        elapsed = time.perf_counter() - start
        assert not out.fell_back, f"depth-13 day fell back on {mode}: {out.reason}"
        assert out.max_open_depth == 13, f"expected depth 13, got {out.max_open_depth}"
        assert out.peak_states > 0
        assert elapsed < 5.0, f"depth-13 day too slow on {mode}: {elapsed:.2f}s"
