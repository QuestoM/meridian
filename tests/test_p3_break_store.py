"""The break entity: its identity, its day, and whether its money is the plan's.

The claim these tests hold is the one everything else on the day board rests on:
a break's projected revenue is the optimizer's own credit to that break, so the
day's breaks sum back to the day's revenue exactly, and the cheap score of the
saved arrangement equals the expensive optimization it came from.
"""

from __future__ import annotations

import pytest

from kairos_api import break_store

pytestmark = pytest.mark.realdata


def declare_operator_channel():
    """Give these tests an operator channel when the saved settings carry none.

    ``data/kairos_settings.json`` is a shared mutable file that the running
    product and every other suite writes. Measured during this wave: it was
    rewritten with an empty ``operator_channel`` while these tests were running,
    and four boundary assertions turned into silent skips. A test that cannot
    fail proves nothing, so when settings carry no channel this declares one from
    the saved plan and returns the patcher to undo. The real read path is
    exercised either way, because this replaces the settings, never the answer.
    """
    from kairos_api.core import _load_settings

    settings = _load_settings()
    if str(settings.operator_channel or "").strip():
        return None
    from kairos_api.core import _load_break_schedule

    schedule = _load_break_schedule()
    if schedule.empty or "channel" not in schedule.columns:
        return None
    channel = sorted({str(value).strip() for value in schedule["channel"].tolist() if str(value).strip()})[0]
    declared = settings.model_copy(update={"operator_channel": channel})
    patch = pytest.MonkeyPatch()
    import kairos_api.core as core
    import kairos_api.day_api as day_api

    patch.setattr(core, "_load_settings", lambda: declared)
    patch.setattr(day_api, "_load_settings", lambda: declared)
    return patch


@pytest.fixture(scope="module", autouse=True)
def owned_channel():
    patch = declare_operator_channel()
    yield
    if patch is not None:
        patch.undo()


@pytest.fixture(scope="module")
def plan():
    days = break_store.plan_days()
    if not days:
        pytest.skip("no saved plan on the operator channel, so there is no day to build")
    return break_store.day_plan(days[0])


def test_a_break_id_round_trips_through_its_two_parts():
    value = break_store.break_id("2024-11-01|רשת 13|008", 2)
    assert value == "2024-11-01|רשת 13|008~2"
    assert break_store.parse_break_id(value) == ("2024-11-01|רשת 13|008", 2)


@pytest.mark.parametrize("bad", ["", "no-ordinal", "segment~0", "segment~x", "~1"])
def test_a_malformed_break_id_is_refused_rather_than_guessed(bad):
    with pytest.raises(ValueError):
        break_store.parse_break_id(bad)


def test_the_days_offered_are_only_the_operator_channels_own(plan):
    days = break_store.plan_days()
    assert days == sorted(days)
    assert plan.day in days
    assert plan.channel == break_store.operator_channel()


def test_every_segment_and_every_break_is_on_the_owned_channel(plan):
    """The competitor boundary, held by construction rather than by a filter."""
    owned = break_store.operator_channel()
    assert {segment.channel for segment in plan.segments} == {owned}
    assert {placement.channel for placement in plan.result.placements} == {owned}
    assert {record["channel"] for record in break_store.break_records(plan)} == {owned}


def test_the_breaks_sum_back_to_the_day_within_display_rounding(plan):
    """The column adds up, and the bound it adds up to is stated rather than hoped.

    Each break's figure is rounded to the agora for display, so the sum of the
    rounded figures can differ from the day's own total by at most half an agora
    per break. Measured on the reference day that is 80 breaks and a drift of
    0.021 ILS against a bound of 0.40. The board therefore prints the engine's own
    day total in the footer rather than the column sum, so the two can never
    contradict each other on screen.
    """
    records = break_store.break_records(plan)
    assert records, "the day carries no breaks, so there is nothing to sum"
    total = sum(record["projected_revenue"] for record in records)
    bound = 0.005 * len(records)
    assert abs(total - plan.result.total_revenue) <= bound

    unrounded = sum(float(placement.revenue) for placement in plan.result.placements)
    assert unrounded == pytest.approx(plan.result.total_revenue, abs=1e-6)


def test_one_break_record_exists_for_every_placement_the_optimizer_made(plan):
    records = break_store.break_records(plan)
    assert len(records) == len(plan.result.placements) == plan.result.total_breaks
    assert len({record["break_id"] for record in records}) == len(records)


def test_the_ordinals_within_a_segment_run_from_one_and_follow_the_clock(plan):
    grouped: dict[str, list[dict]] = {}
    for record in break_store.break_records(plan):
        grouped.setdefault(record["segment_id"], []).append(record)
    for rows in grouped.values():
        assert [row["ordinal"] for row in rows] == list(range(1, len(rows) + 1))
        starts = [row["start_seconds"] for row in rows]
        assert starts == sorted(starts)
        assert all(row["breaks_in_segment"] == len(rows) for row in rows)


def test_the_cheap_score_of_the_saved_arrangement_is_the_optimizers_own_answer(plan):
    """The whole live-money design rests on this equality, so it is asserted."""
    from kairos.optimize.evaluate import score

    counts, pins = break_store.arrangement(plan)
    evaluation = score(plan.basis, counts, revenue_weight=plan.revenue_weight, placements=pins)
    assert evaluation.revenue == pytest.approx(plan.result.total_revenue, abs=1e-6)
    assert evaluation.retention == pytest.approx(plan.result.aggregate_retention, abs=1e-12)
    assert evaluation.objective == pytest.approx(plan.result.objective, abs=1e-12)


def test_a_second_read_of_the_same_day_is_the_same_object(plan):
    """The cache is what keeps opening a day cheap after the first build.

    Read twice with nothing between them rather than against the module fixture.
    The day is keyed on a fingerprint of everything that can change it, and one of
    those inputs is ``data/kairos_settings.json``, which the product rewrites
    while merely serving a read: the wave-zero handover records exactly that and
    the file belongs to another piece. Measured during this wave on one identical
    eight file command, three runs failed here and three passed, and the
    fingerprint moved inside a 33 s batch, 918843994992144633 to
    -2465893684241373906. A rebuild after an input moved is the cache obeying its
    own contract rather than breaking it, so the assertion is on the property the
    cache actually promises.
    """
    first = break_store.day_plan(plan.day)
    assert break_store.day_plan(plan.day) is first


def test_an_unknown_day_is_a_lookup_failure_and_not_an_empty_board():
    with pytest.raises(LookupError):
        break_store.day_plan("1999-01-01")
    with pytest.raises(LookupError):
        break_store.day_plan("")


def test_the_delivered_coverage_read_names_the_dates_it_actually_has():
    from kairos.export.spots_coverage import daily_input_days

    covered = daily_input_days()
    plan_dates = set(break_store.plan_days())
    assert covered, "the shipped daily file should be readable"
    assert not (covered & plan_dates), (
        "if a daily file ever covers a planned day, delivered money stops being "
        "an empty state and this test is the place that says so"
    )
