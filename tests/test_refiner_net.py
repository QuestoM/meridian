"""Tests for the net-ILS mode of the F1 per-channel-day refiner.

The refiner recovers the value greedy leaves on the table per channel-day. It
used to climb only the convex-blend contribution, so ``objective_mode='revenue_net'``
shipped pure greedy and could stop 3.9M ILS/week short of the (refined) blend plan
on the very net-ILS metric net mode maximises. The refiner now climbs whichever
objective the run selected; these tests prove the net path:

  * BLEND BYTE-IDENTITY: the shipped weekly-schedule golden is unchanged, so the
    default blend path did not move by a byte when the net objective was threaded
    through the same climb (:func:`test_blend_golden_is_byte_identical`).
  * NET NEVER-REGRESS: on real channel-days, per group, the net-refined total is
    >= the net-greedy total, and at least one group is strictly better
    (:func:`test_net_refined_beats_net_greedy_per_group_on_real_data`).
  * NET RECOVERY: on a hand-computed fixture where greedy is provably net-
    suboptimal (a spacing collision makes the better split unreachable one break
    at a time), the refiner recovers the known net optimum
    (:func:`test_net_refiner_recovers_hand_computed_optimum`).
"""
from __future__ import annotations

import sys
from datetime import date
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.data.loaders import load_programmes  # noqa: E402
from kairos.data.transform import build_segments_from_programmes  # noqa: E402
from kairos.export.schedule import (  # noqa: E402
    DEFAULT_IMPACT_MODEL_PATH,
    _build_classifier,
)
from kairos.optimize.guardrails import Guardrails, is_compliant  # noqa: E402
from kairos.optimize._segment_math import _group_breaks  # noqa: E402
from kairos.optimize.optimizer import (  # noqa: E402
    _EPSILON,
    ProgramSegment,
    optimize_breaks,
)
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings  # noqa: E402
from kairos.optimize.refiner import _MAX_EXACT_COMBOS, optimize_group  # noqa: E402
from kairos.optimize.revenue_net import segment_net_revenue  # noqa: E402
from kairos.model.impact import load_impact_model  # noqa: E402
from kairos.service import guardrails_from_settings  # noqa: E402

import tests.golden_weekly_schedule as golden  # noqa: E402

SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"


def _net_total(segments, counts: dict[str, int]) -> float:
    """The group's pure-ILS net (sum of each segment's net at its count)."""
    return sum(segment_net_revenue(s, counts[s.segment_id]) for s in segments)


# --------------------------------------------------------------------------- #
# (a) BLEND BYTE-IDENTITY
# --------------------------------------------------------------------------- #
def test_blend_golden_is_byte_identical() -> None:
    """The default blend weekly schedule must be byte-identical after net wiring.

    The net objective is threaded through the SAME refiner climb the blend path
    uses; this reruns the committed golden-master (the exact ``POST
    /api/recompute-schedule`` path) and asserts both the full-CSV hash and the
    per-channel-day aggregate hash still match to the byte. If the net threading
    perturbed the blend path at all, a single per-day total would move and both
    hashes would flip.
    """
    _, _, csv_sha, agg_sha, problems = golden.evaluate()
    assert not problems, "blend golden drifted:\n" + "\n".join(problems)
    assert csv_sha == golden.GOLDEN_CSV_SHA256
    assert agg_sha == golden.GOLDEN_AGG_SHA256


# --------------------------------------------------------------------------- #
# (b) NET NEVER-REGRESS ON REAL DATA
# --------------------------------------------------------------------------- #
def _real_channel_days(limit: int) -> list[tuple[str, str, list]]:
    """Build a few real channel-days from the committed corpus.

    Uses the SAME classifier, pricing, assumptions and impact model the shipped
    export builds segments with, so the net values are what the optimizer decides
    against. Returns ``(channel, day, segments)`` for the first ``limit``
    non-empty channel-days, and skips days too small to hold a coordinated move.
    """
    import json

    settings_map = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    assumptions = OptimizerAssumptions()
    pricing = pricing_from_settings(settings_map, None)
    classifier = _build_classifier()
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    programmes = load_programmes()
    channels = sorted(set(programmes["Channel"].dropna().astype(str)))
    days = sorted(set(programmes["start_dt"].dropna().dt.strftime("%Y-%m-%d")))
    out: list[tuple[str, str, list]] = []
    for day in days:
        for channel in channels:
            segments = build_segments_from_programmes(
                programmes, classifier, pricing, assumptions=assumptions,
                impact_model=impact_model, channel=channel, day=day,
            )
            if len(segments) >= 3:
                out.append((channel, day, segments))
            if len(out) >= limit:
                return out
    return out


def test_net_refined_beats_net_greedy_per_group_on_real_data() -> None:
    """Per real channel-day: net-refined >= net-greedy, at least one strictly up.

    For each of a handful of real channel-days, allocate the same segments under
    ``objective_mode='revenue_net'`` with ``refine=False`` (pure greedy) and with
    ``refine=True`` (net-mode refiner), evaluate both plans on the pure-ILS net,
    and assert the refined net is never below greedy (the never-regress invariant)
    and that at least one channel-day is strictly improved (the refiner is doing
    real work in net mode, not a no-op).
    """
    settings_map = __import__("json").loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    guardrails = guardrails_from_settings(settings_map)
    weight = settings_map["revenue_weight"] / 100.0
    risk_lambda = settings_map["risk_lambda"]

    channel_days = _real_channel_days(limit=6)
    assert channel_days, "no real channel-days built from the corpus"

    any_strictly_better = False
    for channel, day, segments in channel_days:
        greedy = optimize_breaks(
            segments, guardrails, revenue_weight=weight, risk_lambda=risk_lambda,
            refine=False, objective_mode="revenue_net",
        )
        refined = optimize_breaks(
            segments, guardrails, revenue_weight=weight, risk_lambda=risk_lambda,
            refine=True, objective_mode="revenue_net",
        )
        greedy_counts = {s.segment_id: s.num_breaks for s in greedy.segments}
        refined_counts = {s.segment_id: s.num_breaks for s in refined.segments}
        greedy_net = _net_total(segments, greedy_counts)
        refined_net = _net_total(segments, refined_counts)

        assert refined.is_compliant, f"{channel} {day}: refined plan not compliant"
        assert refined_net >= greedy_net - _EPSILON, (
            f"{channel} {day}: net refined {refined_net} regressed below greedy {greedy_net}"
        )
        if refined_net > greedy_net + _EPSILON:
            any_strictly_better = True

    assert any_strictly_better, (
        "net refiner never strictly beat net greedy on any sampled real "
        "channel-day; expected at least one recovered group"
    )


# --------------------------------------------------------------------------- #
# (c) NET RECOVERY OF A HAND-COMPUTED OPTIMUM
# --------------------------------------------------------------------------- #
def _fixture_segment(sid: str, start: float, cpp: float, coeff: float, tvr: float) -> ProgramSegment:
    return ProgramSegment(
        segment_id=sid, channel="C", day="Mon", start_seconds=start,
        duration_seconds=3600.0, program_type="Drama", baseline_tvr=tvr, cpp=cpp,
        impact_coefficient=coeff, retention_baseline=1.0, premium=1.0, is_gold=False,
        max_breaks=4, break_length_seconds=120.0,
    )


def _net_suboptimal_fixture() -> tuple[list[ProgramSegment], Guardrails]:
    """Three programmes in one hour where NET greedy is provably stuck.

    All three share the 20:00 hour under a 2-breaks-per-hour cap and the default
    420s spacing. Hand arithmetic (unit CPP, 120s breaks, 3600s programmes):

        a: net(1)=58,880  net(2)=112,640
        b: net(1)=60,480  net(2)=109,440
        c: net(1)=60,480  net(2)=114,240

    Compliant two-break allocations under the cap:
        {b:2} -> 109,440   {c:2} -> 114,240 (the net optimum)
    The higher raw split {b:1, c:1} = 120,960 is INFEASIBLE: b's and c's single
    breaks land at their segment centres 130s apart, inside the 420s spacing floor,
    which is exactly the re-spacing non-monotonicity the refiner exists for.

    Greedy takes b's first break (marginal 60,480, and b sorts before c on the
    tie), then adds b's second (48,960, feasible) because adding c's first would
    make the spacing-infeasible {b:1, c:1}. So greedy lands at {b:2}=109,440. The
    net optimum {c:2}=114,240 needs BOTH of b's breaks dropped and BOTH of c's
    added at once, unreachable one break at a time. The refiner (this tiny group
    takes the exact-enumeration tier) recovers it: +4,800 net.
    """
    segments = [
        _fixture_segment("a", 20 * 3600.0, cpp=2000.0, coeff=-0.04, tvr=8.0),
        _fixture_segment("b", 20 * 3600.0 + 250.0, cpp=1800.0, coeff=-0.08, tvr=10.0),
        _fixture_segment("c", 20 * 3600.0 + 500.0, cpp=1200.0, coeff=-0.05, tvr=14.0),
    ]
    return segments, Guardrails(max_breaks_per_hour=2)


def _brute_force_net_optimum(segments, guardrails) -> tuple[dict[str, int], float]:
    """Independent exhaustive net optimum over the full compliant break box."""
    ranges = [range(0, s.max_breaks + 1) for s in segments]
    best_counts, best = None, float("-inf")
    for vector in product(*ranges):
        counts = {s.segment_id: k for s, k in zip(segments, vector)}
        if not is_compliant(_group_breaks(segments, counts, {}), guardrails):
            continue
        value = _net_total(segments, counts)
        if value > best + _EPSILON:
            best, best_counts = value, counts
    return best_counts, best


def test_net_refiner_recovers_hand_computed_optimum() -> None:
    segments, guardrails = _net_suboptimal_fixture()

    # This group is on the exact-enumeration tier, so the recovery is provable.
    combos = 1
    for s in segments:
        combos *= s.max_breaks + 1
    assert combos <= _MAX_EXACT_COMBOS

    # The hand-computed optimum, cross-checked by an independent brute force.
    expected_counts = {"a": 0, "b": 0, "c": 2}
    expected_net = 114240.0
    brute_counts, brute_net = _brute_force_net_optimum(segments, guardrails)
    assert brute_counts == expected_counts
    assert abs(brute_net - expected_net) < 1e-6

    # Net greedy is genuinely stuck short of the optimum here.
    greedy = optimize_breaks(segments, guardrails, revenue_weight=0.6, refine=False, objective_mode="revenue_net")
    greedy_counts = {s.segment_id: s.num_breaks for s in greedy.segments}
    greedy_net = _net_total(segments, greedy_counts)
    assert greedy_counts == {"a": 0, "b": 2, "c": 0}
    assert abs(greedy_net - 109440.0) < 1e-6
    assert greedy_net < expected_net - _EPSILON  # greedy is provably suboptimal

    # The net-mode refiner recovers the known optimum, end to end and via the
    # group primitive directly.
    refined = optimize_breaks(segments, guardrails, revenue_weight=0.6, refine=True, objective_mode="revenue_net")
    refined_counts = {s.segment_id: s.num_breaks for s in refined.segments}
    assert refined.is_compliant
    assert refined_counts == expected_counts
    assert abs(_net_total(segments, refined_counts) - expected_net) < 1e-6

    floors = {s.segment_id: 0 for s in segments}
    caps = {s.segment_id: s.max_breaks for s in segments}
    seed = dict(greedy_counts)  # seed from the stuck greedy allocation
    group_counts = optimize_group(
        segments, seed, floors, caps, {}, guardrails,
        revenue_weight=0.6, revenue_scale=1.0, total_tvr=sum(s.baseline_tvr for s in segments),
        net_of=segment_net_revenue,
    )
    assert group_counts == expected_counts
