"""P2: the comparison's wait, taken before the planner asks for it.

Step three is fourteen real optimizations over the plan's own week. Measured on
this tree with the engine in process: 12.61 s cold end to end, 0.9 s of refined
optimizer per day per leg, and 22 ms warm with a byte-identical body. Two
cheaper routes were measured and refused, and both refusals are recorded in
``use-compare-prepare.js`` so nobody re-derives them: threads do not help
because the work is GIL bound (six day runs, 4.18 s sequential against 4.20 s
across six workers), and reading leg A off the committed plan would halve the
work but nothing on disk records which levers produced the saved plan.

What is left is to spend the wait while the planner is still setting the
scenarios up. This locks the contract of that mechanism, which is a frontend
one, in the same way the piece already locks the panel's copy: against the
source, because the assertions are about what the code is wired to do rather
than about a figure. The figures themselves are asserted in
``test_p2_compare_net.py`` against the payload builders.
"""

from __future__ import annotations

from pathlib import Path

import pytest

WEEK = Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src" / "plan" / "week"


@pytest.fixture(scope="module")
def prepare_source() -> str:
    return (WEEK / "use-compare-prepare.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def surface_source() -> str:
    return (WEEK / "use-plan-surface.js").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def panel_source() -> str:
    return (WEEK / "ComparePanel.jsx").read_text(encoding="utf-8")


def test_the_preparation_and_the_comparison_send_the_same_body(prepare_source, surface_source):
    """One builder for both bodies.

    A preparation that differs from the comparison by one field prepares a week
    the comparison then runs again from scratch, and the planner waits the full
    twelve seconds while the panel says the work is done. So the body has exactly
    one implementation and both callers use it.
    """
    assert "export function compareRequestBody(" in prepare_source
    assert "compareRequestBody(legs.current.legA, legs.current.legB)" in prepare_source
    assert "compareRequestBody(legA, legB)" in surface_source
    # And the old inline body is gone, so there is nothing left to drift.
    assert "weight_a: Math.round(" not in surface_source


def test_the_preparation_runs_only_while_the_step_is_open_and_nothing_else_is_running(
    prepare_source, surface_source
):
    assert "if (!enabled || busy || !key)" in prepare_source
    assert "busy: compareState === 'running'" in surface_source
    assert "prepareCompare: section === 'compare'" in (
        (WEEK / "PlanWeek.jsx").read_text(encoding="utf-8")
    )


def test_a_settled_comparison_is_not_prepared_again(prepare_source, surface_source):
    """The planner's own comparison already computed that week, so preparing it
    a second time would spend the machine on an answer that is already held."""
    assert "settledKey" in prepare_source
    assert "ready.current.add(settledKey)" in prepare_source
    assert "settledKey: comparedKey" in surface_source
    assert "setComparedKey" in surface_source


def test_changing_a_lever_abandons_the_preparation_in_flight(prepare_source):
    """A preparation for levers nobody is looking at any more is waste, and a
    stream left open is worse than waste."""
    assert "abort.current?.abort()" in prepare_source
    assert "window.clearTimeout(timer)" in prepare_source
    assert "SETTLE_MS = 900" in prepare_source


def test_a_failed_preparation_is_silent_and_never_reaches_a_figure(prepare_source):
    """It is an optimization, not an answer. Nothing it produces is displayed,
    and a failure is the comparison's to report when it is asked for."""
    assert ".catch(() => {" in prepare_source
    assert "if (live) setPhase('idle');" in prepare_source
    # The hook hands back a phase and a key. No payload, no money, no day rows.
    assert "return { phase, key };" in prepare_source


def test_the_panel_says_which_of_the_two_states_it_is_in(panel_source):
    """Spending a machine's time silently is not honest about what it costs."""
    assert "Both scenarios are being computed in the background while you set them up." in panel_source
    assert "שני התרחישים מחושבים ברקע בזמן שאתם מכווננים אותם." in panel_source
    assert "Both scenarios are already computed for these settings, so the comparison returns without another wait." in panel_source
    assert "שני התרחישים כבר חושבו בהגדרות האלה, ולכן ההשוואה תחזור בלי המתנה נוספת." in panel_source
    # And the run's real cost still prints afterwards, so the claim is checkable
    # against what the server says it did.
    assert "runCostLine" in panel_source
    assert "computed now" in panel_source


def test_the_key_covers_every_lever_a_run_stands_on(prepare_source):
    """A key that omits a lever would call a different week prepared."""
    for field in ("revenue_weight", "retention_floor", "max_breaks_per_hour", "risk_lambda", "objective_mode"):
        assert field in prepare_source
    assert "export function compareKey(" in prepare_source


def test_every_file_the_preparation_touches_is_inside_the_size_law():
    for name in ("use-compare-prepare.js", "use-plan-surface.js", "ComparePanel.jsx", "PlanWeek.jsx"):
        lines = (WEEK / name).read_text(encoding="utf-8").splitlines()
        assert len(lines) <= 450, f"{name} is {len(lines)} lines"
