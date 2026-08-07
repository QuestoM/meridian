"""Standing guard on the decision-robustness review machinery.

Fast, seeded, one channel-day, tiny K. Asserts three things the review
(docs/model-validation/decision-robustness.md) depends on:

  1. the coefficient draws are deterministic for a seed and sign-plausible,
  2. the shipped decision process replayed in memory still reproduces the saved
     weekly CSV's plan on the owned channel's representative day (the review's
     premise that live engine and saved plan agree; if this fails after a
     deliberate settings/coefficients change, re-run the review scripts), and
  3. the shipped plan repriced under drawn coefficients stays inside a sane
     revenue-net band around its point valuation (the review measured a
     P10-P90 swing of about +/-2.5 percent at K=200; the +/-10 percent band
     here is four times that, so it trips only on real breakage).

Marked ``realdata`` (repo convention: reads the real reference data), runs in
about 15 seconds. Never writes output/weekly_break_schedule.csv or any product
file; every plan is built in memory via kairos.service.run_scenario.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "validation"))

import decision_uncertainty_lib as lib  # noqa: E402

pytestmark = pytest.mark.realdata

K = 3
SEED = 42
NET_BAND = 0.10          # shipped plan repriced under a draw: net within +/-10% of point
BREAKS_BAND = 0.15       # re-optimized total breaks within +/-15% of the shipped total


@pytest.fixture(scope="module")
def ctx():
    return lib.load_context()


@pytest.fixture(scope="module")
def draws(ctx):
    return lib.draw_coefficient_vectors(ctx.detail, K, seed=SEED)


def test_draws_are_seeded_and_sign_plausible(ctx, draws):
    again = lib.draw_coefficient_vectors(ctx.detail, K, seed=SEED)
    assert draws == again, "same seed must reproduce identical coefficient draws"
    assert len(draws) == K and all(len(d) == len(ctx.cells) for d in draws)
    for draw in draws:
        for cell, value in draw.items():
            assert -1.0 <= value <= 0.0, f"{cell} drawn outside the sign-plausible range: {value}"


def test_replay_reproduces_the_shipped_plan(ctx):
    """The in-memory replay of the shipped process equals the saved CSV plan.

    This is the review's premise (live engine == plan of record), and the
    premise is measured here rather than assumed. The saved plan is a live
    artifact: any recompute rewrites it under the settings that were on disk at
    that moment, so a plan written under one objective weight and read back
    under another cannot be reproduced by anything, and reading that as an
    engine fault sends a repair after code that is correct. The product already
    measures exactly this (kairos.export.schedule_freshness), so its verdict
    decides. Fresh, or unstamped as in a fresh checkout: the replay must
    reproduce the plan. Stale: the inputs moved after the plan was written, so
    the review numbers are stale, and the answer is to recompute the plan and
    re-run scripts/validation/decision_sensitivity.py and friends.

    Staleness is its own loud failure here and it is never an excuse for a count
    difference. An earlier version skipped when the counts differed AND the plan
    was stale, which was measured to be an accommodation on three grounds. The
    same commit that added the skip moved revenue_weight from 60 to 35 and
    min_retention_floor from 0.72 to 0.78, so it shipped the escape hatch for
    the staleness it had just caused. The counts differ even when freshness
    reports fresh with nothing changed, so staleness never explained the
    divergence, it only hid it whenever some input happened to have moved. And
    any one of twelve input groups going stale, several of which do not drive
    weekly break counts at all, excused any difference including a real
    optimizer regression that merely coincided with an unrelated edit. A skip is
    also invisible without -rs. The repairer's real concern is kept in full: a
    stale plan still fails with the recompute instruction rather than sending a
    repair after correct code, because the message says which inputs moved.
    """
    counts, payload = lib.reoptimize(ctx)
    assert not payload["violations"], "shipped replay must be guardrail-compliant"
    freshness = lib.shipped_plan_freshness()
    assert freshness["status"] != "stale", (
        f"the saved plan was written {freshness['computed_at']} and its "
        f"{', '.join(freshness['changed'])} inputs have moved since, so it is not this "
        "process's plan; recompute it and re-run scripts/validation/decision_sensitivity.py "
        "before reading this suite"
    )
    assert counts == ctx.shipped_counts


def test_shipped_plan_repriced_under_draws_stays_in_band(ctx, draws):
    point = {c: float(ctx.detail[c]["coefficient"]) for c in ctx.cells}
    point_segments = lib.build_segments(ctx, lib.make_impact_model(point, ctx.detail))
    lib.verify_segment_mapping(point_segments, point)
    at_point = lib.evaluate_counts(point_segments, ctx.shipped_counts)
    assert at_point["retention_cost_ils"] > 0
    assert at_point["gross_ils"] > at_point["net_ils"] > 0

    for draw in draws:
        segments = lib.build_segments(ctx, lib.make_impact_model(draw, ctx.detail))
        repriced = lib.evaluate_counts(segments, ctx.shipped_counts)
        ratio = repriced["net_ils"] / at_point["net_ils"]
        assert 1 - NET_BAND <= ratio <= 1 + NET_BAND, (
            f"shipped plan repriced at a plausible coefficient draw moved "
            f"{ratio - 1:+.1%}, outside the +/-{NET_BAND:.0%} sanity band"
        )
        assert repriced["retention_cost_ils"] > 0


def test_reoptimization_under_a_draw_runs_and_is_consistent(ctx, draws):
    draw = draws[0]
    model = lib.make_impact_model(draw, ctx.detail, degenerate_ci=True)
    counts, payload = lib.reoptimize(ctx, model, refine=True)
    assert not payload["violations"], "re-optimized plan must be guardrail-compliant"

    segments = lib.build_segments(ctx, model)
    lib.verify_segment_mapping(segments, draw)
    evaluation = lib.evaluate_counts(segments, counts)
    # The harness segments must reprice the service's own plan to the cent;
    # any daylight means the harness drifted from kairos.service internals.
    lib.assert_revenue_consistency(payload, evaluation)

    shipped_total = sum(ctx.shipped_counts.values())
    assert abs(evaluation["breaks"] - shipped_total) <= BREAKS_BAND * shipped_total, (
        f"re-optimized break volume {evaluation['breaks']:.0f} strays more than "
        f"{BREAKS_BAND:.0%} from the shipped {shipped_total}"
    )
