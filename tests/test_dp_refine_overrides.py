"""The exact DP tier under operator overrides (floors, caps, gold marks).

A single gold-marked segment used to take the exact tier off the WHOLE channel-day:
:func:`_blocking_constraint` returned on the first marked segment and the caller
fell back for the entire group. Measured on רשת 13 2024-11-03 (91 segments) that
one mark cost 122,886.51 ILS, 7.48 percent of the day's revenue, for a retention
gain of 0.0138 of a point. Floors and caps forfeited the same day the same way.

Three gates here, each of which fails on that old behaviour:

  * constrained oracle: with random floors, caps and gold marks the DP's objective
    still equals an exhaustive brute force that enumerates the same bounded ranges
    and checks compliance on the gold-marked reconstruction. On the old code every
    constrained instance fell back, so nothing reached the oracle at all;
  * the real day: the gold mark no longer appears in the fallback histogram and no
    longer costs revenue;
  * the daily gold budget is charged off the segment-or-override union the
    guardrail counts, so a BINDING gold cap still binds. Unblocking without this
    would be worse than the defect: the tier would propose over-gilded plans.

Why exactly these three constraints and not placement pins: a pin carries its own
per-break duration, and the sweep's daily ad-seconds budget is an integer count of
uniform-length breaks, so pins remain a labeled fallback (asserted below).
"""
from __future__ import annotations

import math
import random
from dataclasses import replace
from itertools import product

import pytest

from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.model.impact import load_impact_model
from kairos.optimize.dp_refine import (
    OBJECTIVE_BLEND,
    OBJECTIVE_REVENUE_NET,
    dp_refine_group,
)
from kairos.optimize.guardrails import Guardrails, evaluate, is_compliant
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.overrides import Override, OverrideSet
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings
from kairos.optimize.revenue_net import segment_net_revenue
from kairos.optimize._segment_math import (
    _group_breaks,
    _group_objective_contribution,
    _risk_adjusted_coefficient,
    _segment_break_objects,
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

# The day the defect was measured on. Hebrew channel strings are the exact values
# the data carries (never romanized).
GOLD_DAY = ("רשת 13", "2024-11-03")


# ---------------------------------------------------------------------------
# Synthetic instances for the constrained oracle
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
        is_gold=(kind == "tight_gold" and rng.random() < 0.4),
        max_breaks=rng.randint(1, 4), break_length_seconds=120.0,
    )
    if rng.random() < 0.3:
        seg = replace(seg, impact_ci_low=coeff - rng.uniform(0.01, 0.05),
                      impact_ci_high=min(0.0, coeff + 0.01))
    return seg


def _random_constraints(rng, group):
    """A random operator constraint set: floors, caps, gold marks.

    Deliberately harsher than production, where
    :func:`~kairos.optimize._override_logic._reject_infeasible_floors` backs an
    impossible floor out before the DP ever sees it. An instance with no
    constraint at all is skipped by the caller (it would test the free path).
    """
    floors: dict[str, int] = {}
    caps: dict[str, int] = {}
    gold: dict[str, bool] = {}
    for s in group:
        roll = rng.random()
        if roll < 0.20:
            floors[s.segment_id] = rng.randint(1, s.max_breaks)
        elif roll < 0.35:
            caps[s.segment_id] = rng.randint(0, s.max_breaks)
        if rng.random() < 0.35:
            gold[s.segment_id] = True
    return floors, caps, gold


def _prep(segs, *, risk_lambda):
    """Risk pre-pass and global normalisers, mirroring optimize_breaks exactly."""
    adj = [replace(s, impact_coefficient=_risk_adjusted_coefficient(s, risk_lambda))
           for s in segs]
    scale = max(sum(_segment_revenue(s, s.max_breaks) for s in adj), 1e-9)
    return adj, scale, sum(s.baseline_tvr for s in adj)


def _score(group, counts, scale, total_tvr, *, mode, revenue_weight):
    if mode == OBJECTIVE_REVENUE_NET:
        return sum(segment_net_revenue(s, counts[s.segment_id]) for s in group)
    contribution, _, _ = _group_objective_contribution(
        group, counts, revenue_weight=revenue_weight,
        revenue_scale=scale, total_tvr=total_tvr,
    )
    return contribution


def _brute_best(group, scale, total_tvr, guardrails, *, mode, revenue_weight,
                floors, caps, gold):
    """Exhaustive optimum over the OVERRIDE-BOUNDED ranges, gold-aware compliance."""
    best = -math.inf
    ranges = [
        range(max(0, floors.get(s.segment_id, 0)),
              min(s.max_breaks, caps.get(s.segment_id, s.max_breaks)) + 1)
        for s in group
    ]
    for vec in product(*ranges):
        counts = {s.segment_id: k for s, k in zip(group, vec)}
        if not is_compliant(_group_breaks(group, counts, gold), guardrails):
            continue
        best = max(best, _score(group, counts, scale, total_tvr,
                                mode=mode, revenue_weight=revenue_weight))
    return best


@pytest.mark.parametrize("seed", [1, 7, 42, 2024, 20260707])
def test_constrained_oracle_dp_matches_brute_force(seed):
    """With floors, caps and gold marks the DP still equals the exhaustive optimum.

    Also asserts the two things unblocking could have broken quietly: the proposed
    counts stay inside the operator's bounds, and the gold-marked reconstruction
    passes the engine's own compliance check. A fallback is allowed only when brute
    force found nothing feasible either, so the tier never trades coverage for
    exactness.
    """
    rng = random.Random(seed)
    checked = 0
    for _ in range(40):
        guardrails, kind = _guardrail_profile(rng)
        segs = [_make_segment(rng, i, kind) for i in range(rng.randint(3, 6))]
        mode = rng.choice([OBJECTIVE_BLEND, OBJECTIVE_REVENUE_NET])
        revenue_weight = rng.uniform(0.0, 1.0)
        adj, scale, total_tvr = _prep(segs, risk_lambda=rng.choice([0.0, 0.0, 0.5, 1.0]))
        floors, caps, gold = _random_constraints(rng, adj)
        if not floors and not caps and not gold:
            continue

        outcome = dp_refine_group(
            adj, {s.segment_id: 0 for s in adj}, guardrails,
            revenue_weight=revenue_weight, revenue_scale=scale, total_tvr=total_tvr,
            objective_mode=mode, net_of=segment_net_revenue,
            floors=floors, caps=caps, gold_by_id=gold,
        )
        brute = _brute_best(adj, scale, total_tvr, guardrails, mode=mode,
                            revenue_weight=revenue_weight,
                            floors=floors, caps=caps, gold=gold)
        if outcome.fell_back:
            assert brute == -math.inf, (
                f"seed={seed} fell back ({outcome.reason_code}) on an instance brute "
                f"force solved to {brute}"
            )
            continue
        assert outcome.dp_objective == pytest.approx(brute, abs=_TOL, rel=_TOL), (
            f"seed={seed} mode={mode} dp={outcome.dp_objective} brute={brute}"
        )
        replayed = _score(adj, outcome.counts, scale, total_tvr,
                          mode=mode, revenue_weight=revenue_weight)
        assert replayed == pytest.approx(brute, abs=_TOL, rel=_TOL)
        assert is_compliant(_group_breaks(adj, outcome.counts, gold), guardrails)
        for sid, k in outcome.counts.items():
            assert k >= floors.get(sid, 0), f"override floor breached on {sid}"
            assert k <= caps.get(sid, math.inf), f"override cap breached on {sid}"
        checked += 1
    assert checked >= 20, f"too few constrained instances stayed on the exact path ({checked})"


def test_placement_pins_remain_a_labeled_fallback():
    """Pins carry their own durations, which the count-based daily budget cannot hold."""
    group = [
        ProgramSegment(
            segment_id="a", channel="C", day="2024-11-01", start_seconds=0.0,
            duration_seconds=3600.0, program_type="Movie", baseline_tvr=5.0, cpp=2000.0,
            impact_coefficient=-0.03, max_breaks=3, break_length_seconds=120.0,
        ),
    ]
    scale = max(sum(_segment_revenue(s, s.max_breaks) for s in group), 1e-9)
    out = dp_refine_group(
        group, {"a": 0}, GR, revenue_weight=0.6, revenue_scale=scale,
        total_tvr=sum(s.baseline_tvr for s in group), objective_mode=OBJECTIVE_BLEND,
        placements={"a": [PlacementPin(offset_seconds=600.0, duration_seconds=90.0)]},
    )
    assert out.fell_back and out.reason_code == "placement_pins"


# ---------------------------------------------------------------------------
# The real channel-day the defect was measured on
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def gold_day_segments():
    assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
    segs = build_segments_from_programmes(
        load_programmes(), _build_classifier(), pricing_from_settings(None, None),
        assumptions=assumptions,
        impact_model=load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions),
        channel=GOLD_DAY[0], day=GOLD_DAY[1],
    )
    assert segs, f"no segments for {GOLD_DAY}"
    return segs


def _plan(segs, overrides, guardrails=GR):
    return optimize_breaks(segs, guardrails, revenue_weight=0.6, risk_lambda=0.0,
                           refine=True, dp_refine=True, overrides=overrides)


def _gold_override(segment_id, override_id="ux-gold-1"):
    return OverrideSet([Override(
        override_id=override_id, scope="segment", target_id=segment_id, kind="gold")])


def test_a_gold_mark_no_longer_forfeits_the_channel_day(gold_day_segments):
    """The money gate. One gold mark must not move the exact tier off the day.

    On the old behaviour this day came back with ``fallback_reasons ==
    {'gold_forcing': 1}``, ``groups_exact == 0`` and 122,886.51 ILS less revenue.
    The mark here is non-binding (one gold break against a cap of three), so the
    honest expectation is that revenue is IDENTICAL to the unmarked day.
    """
    reference = _plan(gold_day_segments, None)
    assert reference.dp_stats["groups_exact"] == 1

    target = gold_day_segments[0].segment_id
    marked = _plan(gold_day_segments, _gold_override(target))

    assert "gold_forcing" not in marked.dp_stats["fallback_reasons"]
    assert marked.dp_stats["groups_exact"] == 1
    assert marked.dp_stats["fallback_reasons"] == {}
    assert marked.total_revenue == pytest.approx(reference.total_revenue, rel=1e-9)

    counts = {p.segment_id: p.num_breaks for p in marked.segments}
    assert not evaluate(_group_breaks(list(gold_day_segments), counts, {target: True}), GR)


def test_the_daily_gold_budget_is_charged_off_the_override(gold_day_segments):
    """Two gold marks against a cap of one: the DP must charge the union, not is_gold.

    Reading gold off the frozen segment (which is False for an override-marked
    segment) would let the sweep propose an over-gilded plan; the belt-and-braces
    check would then reject it and the day would silently lose the exact tier again.
    So assert BOTH that the day stays exact and that the emitted plan holds the cap.
    """
    tight = replace(GR, gold_breaks_max_per_day=1)
    first, second = (s.segment_id for s in gold_day_segments[:2])
    overrides = OverrideSet([
        Override(override_id="g1", scope="segment", target_id=first, kind="gold"),
        Override(override_id="g2", scope="segment", target_id=second, kind="gold"),
    ])
    plan = _plan(gold_day_segments, overrides, tight)

    assert plan.dp_stats["fallback_reasons"] == {}
    assert plan.dp_stats["groups_exact"] == 1
    assert plan.dp_stats["groups_noncompliant"] == 0

    counts = {p.segment_id: p.num_breaks for p in plan.segments}
    marks = {first: True, second: True}
    breaks = _group_breaks(list(gold_day_segments), counts, marks)
    assert sum(1 for b in breaks if b.is_gold) <= tight.gold_breaks_max_per_day
    assert not evaluate(breaks, tight)


def test_an_override_floor_no_longer_forfeits_the_channel_day(gold_day_segments):
    """The same forfeit, through the floor blocker: a forced minimum stays exact.

    The target is chosen as the first segment long enough to actually hold two
    breaks. Most of this day's programmes are not: at 464 seconds the first one
    would put its two breaks 155 seconds apart, and the engine rightly backs that
    floor out (``_reject_infeasible_floors``) before any tier sees it, which would
    make the floor invisible rather than honored.
    """
    target = next(
        s.segment_id for s in gold_day_segments
        if is_compliant(_segment_break_objects(s, 2), GR)
    )
    overrides = OverrideSet([Override(
        override_id="f1", scope="segment", target_id=target, kind="force", value="2")])
    plan = _plan(gold_day_segments, overrides)

    assert plan.rejected_overrides == (), "the floor must bind, not be backed out"
    assert "override_floor" not in plan.dp_stats["fallback_reasons"]
    assert plan.dp_stats["groups_exact"] == 1
    counts = {p.segment_id: p.num_breaks for p in plan.segments}
    assert counts[target] >= 2
    assert not evaluate(_group_breaks(list(gold_day_segments), counts), GR)


def test_an_override_cap_no_longer_forfeits_the_channel_day(gold_day_segments):
    """The same forfeit, through the cap blocker: a forbid stays exact."""
    target = gold_day_segments[0].segment_id
    overrides = OverrideSet([Override(
        override_id="c1", scope="segment", target_id=target, kind="forbid")])
    plan = _plan(gold_day_segments, overrides)

    assert "override_cap" not in plan.dp_stats["fallback_reasons"]
    assert plan.dp_stats["groups_exact"] == 1
    counts = {p.segment_id: p.num_breaks for p in plan.segments}
    assert counts[target] == 0
    assert not evaluate(_group_breaks(list(gold_day_segments), counts), GR)
