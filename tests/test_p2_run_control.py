"""P2: the plan's own state line, and the one act its control performs.

Two defects a blind critic measured on this destination, each pinned here.

The state row carried a button with the run panel's own label whose click only
scrolled to that panel: measured, two visible buttons with identical words, and
zero calls to the job route in two minutes of waiting. That is the dead end the
baseline recorded on the frozen top bar, reproduced inside the new destination.

And the state itself was the one thing this destination still took from the
shell, while only one of its four entrances is handed a refresh handler. Measured
straight after a finished run: the header read out of date while the same route,
read at the same moment, answered fresh.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
WEEK = SRC / "plan" / "week"
ENTRANCES = ("OptimizerWorkspace.jsx", "SchedulePage.jsx", "InventoryPage.jsx", "ForecastsPage.jsx")


@pytest.fixture()
def client():
    from kairos_api.server import app

    return TestClient(app)


def _text(name: str) -> str:
    return (WEEK / name).read_text(encoding="utf-8")


def _body(text: str, opening: str) -> str:
    """The source from a declaration up to the next one at the same indent."""
    assert opening in text, opening
    after = text.split(opening, 1)[1]
    end = after.find("\n  const ")
    return after if end == -1 else after[:end]


def _word(key: str) -> tuple[str, str]:
    source = (SRC / "vocabulary.js").read_text(encoding="utf-8")
    match = re.search(rf"'{re.escape(key)}':\s*\{{\s*en:\s*'([^']*)',\s*he:\s*'([^']*)',", source)
    assert match, key
    return match.group(1), match.group(2)


def test_the_state_row_control_runs_the_plan_rather_than_pointing_at_it():
    """The click that measured zero requests now starts the run itself."""
    header = _text("PlanWeekHeader.jsx")
    assert "onClick={onRun}" in header
    # The navigation-only handler that carried a run's words is gone.
    assert "onClick={() => onGo('run')}" not in header


def test_one_function_stands_behind_every_door_that_runs_the_plan():
    """The state row, the palette row and the R key call the same thing, so no
    control can carry a run's words while doing something else."""
    page = _text("PlanWeek.jsx")
    run_now = _body(page, "const runNow = useCallback(() => {")
    assert "go('run');" in run_now
    assert "surface.runPlan();" in run_now

    assert "onRun={runNow}" in page, "the state row's control"
    assert "      runNow,\n" in page, "the command list"

    commands = _text("plan-week-commands.js")
    assert "runNow," in commands.split(")")[0], "planCommands takes it"
    run_row = commands.split("id: 'run-plan'", 1)[1].split("},", 1)[0]
    assert "run: runNow," in run_row
    # Nothing rebuilds the act a second time inside the list.
    assert "surface.runPlan()" not in commands


def test_the_state_row_control_does_not_borrow_the_run_panels_words():
    """Two controls that do the same thing may sit on one screen. Two controls
    carrying the same string, one of which did nothing, is what was measured, so
    the words are pinned apart as well as the behaviour."""
    header = _text("PlanWeekHeader.jsx")
    panel = _text("RunPanel.jsx")
    assert "words.runShort" in header
    assert "words.run}" not in header.replace("words.runShort", "")
    assert "words.run}" in panel

    model = _text("plan-week-model.js")
    assert "runShort: word('action.run_plan', locale)," in model
    assert "run: word('action.run_weekly_plan', locale)," in model

    short_en, short_he = _word("action.run_plan")
    long_en, long_he = _word("action.run_weekly_plan")
    assert short_en != long_en, (short_en, long_en)
    assert short_he != long_he, (short_he, long_he)
    # Both still speak the canonical verb rather than a retired one.
    for label in (short_en, long_en):
        assert label.lower().startswith("run"), label
    for label in (short_he, long_he):
        assert "הרצ" in label or "הריצו" in label, label


def test_the_destination_reads_the_plan_state_itself():
    """Beside its settings, its plan versions, its progress and its yield."""
    surface = _text("use-plan-surface.js")
    assert "import { usePlanFreshness } from './use-plan-freshness';" in surface
    assert "reload: reloadFreshness," in surface
    for field in ("freshness,", "freshnessState,", "freshnessError,"):
        assert field in surface, field

    hook = _text("use-plan-freshness.js")
    assert "readPlanFreshness()" in hook
    api = _text("plan-week-api.js")
    assert "export async function readPlanFreshness()" in api
    assert "schedule_freshness" in api

    page = _text("PlanWeek.jsx")
    assert "freshness={surface.freshness}" in page
    assert "overview" not in page, "the shell's copy is not read here any more"


def test_no_entrance_forwards_a_state_the_destination_now_owns():
    """A destination that behaved differently by the door somebody came through
    would not be one destination."""
    for name in ENTRANCES:
        assert "overview" not in _text(name), name


def test_the_plan_state_is_read_again_after_every_act_that_can_move_it():
    surface = _text("use-plan-surface.js")
    for act, opening in (
        ("run", "const runPlan = useCallback(async () => {"),
        ("save", "const saveObjective = useCallback(async () => {"),
        ("restore", "const restore = useCallback(async (version) => {"),
    ):
        body = _body(surface, opening)
        assert "reloadFreshness();" in body, act
        # And the dependency is declared, or the closure would call a stale one.
        assert "reloadFreshness]" in body or "reloadFreshness," in body.split("}, [")[-1], act


def test_a_read_still_in_flight_is_not_drawn_as_a_verdict():
    """"No run stamp was found" is a claim about the plan. While nobody has asked
    the server yet it would be a false one, so loading has its own words."""
    header = _text("PlanWeekHeader.jsx")
    assert "const reading = freshnessState === 'loading';" in header
    assert "if (reading) return pageText(locale, 'Reading the plan state from the server.', 'קורא את מצב התוכנית מהשרת.');" in header
    assert "const status = reading || error ? '' : String(freshness?.status || '').toLowerCase();" in header

    panel = _text("RunPanel.jsx")
    assert "const reading = freshnessState === 'loading';" in panel
    assert "const status = reading ? '' : String(freshness?.status || '').toLowerCase();" in panel


def test_a_failed_read_names_its_reason_and_prints_no_state_word():
    header = _text("PlanWeekHeader.jsx")
    assert "freshnessState === 'unavailable'" in header
    assert "The plan state could not be read: ${error}" in header
    assert "לא ניתן היה לקרוא את מצב התוכנית: ${error}" in header

    hook = _text("use-plan-freshness.js")
    assert "setState('unavailable');" in hook
    assert "setVerdict(null);" in hook, "an unreadable state carries no verdict"


def test_the_verdict_the_surface_reads_is_really_on_the_route_it_reads(client):
    """The read is only honest if the route answers with one of the three states
    and never with a fabricated fresh."""
    response = client.get("/api/overview")
    assert response.status_code == 200
    verdict = response.json().get("schedule_freshness")
    assert isinstance(verdict, dict), "the surface reads this key"
    assert verdict.get("status") in {"fresh", "stale", "unknown"}, verdict
    assert "computed_at" in verdict
    assert isinstance(verdict.get("changed"), list)
    if verdict["status"] == "stale":
        assert verdict["changed"], "a stale plan names what changed"
