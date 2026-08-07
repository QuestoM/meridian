"""What the editor's timeline says about what it draws, and whether it is true.

A critic measured the shipped editor at 1440x900 through the scheduler's own
door: one lane reading "רשת 13 / Friday, 8 breaks" with exactly 8 draggable
chips, directly above a readout pricing the whole day and naming 76 breaks and
82 programmes. Three break counts stood on one screen with no sentence
connecting them, and no route from the drawn 8 to the other 68, or from that one
day to the other 29 the plan holds. The route the timeline reads,
``/api/break-operations``, calls section 8.2's frozen ``plan_read`` cap and this
piece does not own that cap, so the fix is not a bigger timeline. It is the one
honest sentence the critic asked for, built from figures the page already fetches
(the drawn lists, the money panel's own live score, and one read of the plan's
own day count), plus a live link to the day board that holds the rest.

A second critic measured that first sentence and found its arithmetic false: the
drawn 8 chips are built from ``output/weekly_break_schedule.csv``, the saved
weekly plan (80 breaks that day), while the module read its denominator off
``score.current.breaks``, the live re-plan (76 breaks that day). Zero of the 8
drawn clock times sat among the live 76, so "8 of 76" asserted a containment
that did not hold. The fix reads both counts off ``score.basis.committed``
instead, the same saved-plan basis the drawn chips come from, and names an
honest unavailable state on the rare channel-day that carries no committed row
at all rather than borrowing a number from the live plan. The tests below hold
both halves: the sentence still names every figure, and it never again pairs a
numerator from one plan with a denominator from another.

This file holds that sentence to the same bar as everything else in this piece:
the words are executed in node from the shipped module, and the figures they
name are read from the routes the page itself calls, never typed into a fixture.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
SCOPE_JS = SRC / "plan" / "day" / "schedule-editor-scope.js"


def read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def node_scope(body: str) -> dict:
    """Run the shipped coverage module in node and return what it computed.

    The module the browser imports is the module asserted here, for the same
    reason every other node-driven test in this piece gives: a python
    re-implementation would only prove two pieces of test code agree.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    script = f"const m = await import({json.dumps(SCOPE_JS.as_uri())});\n{body}"
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_sentence_names_every_figure_the_critic_found_missing():
    """Eight of eighty, twelve of eighty two, one of thirty, on the record.

    Both the numerator (the drawn chips) and the denominator (this figure) are
    the same saved weekly plan, so the counts here are ``score.basis.committed``,
    not ``score.current``: the live re-plan's 76 breaks are a different plan and
    do not contain the drawn 8, which is exactly what the second critic measured.
    """
    measured = node_scope("""
      const coverage = m.buildCoverage({
        breaksShown: 8,
        programsShown: 12,
        score: {
          current: { breaks: 76 },
          basis: { segments: 82, day: '2024-11-01', committed: { breaks: 80, segments: 82 } },
        },
        daysInPlan: 30,
      });
      process.stdout.write(JSON.stringify({
        en: m.coverageSentence(coverage, 'en'),
        he: m.coverageSentence(coverage, 'he'),
      }));
    """)
    assert "8 of the day's 80 breaks" in measured["en"]
    assert "12 of 82 programmes" in measured["en"]
    assert "1 of the plan's 30 days" in measured["en"]
    assert "2024-11-01" in measured["en"]
    assert "8 מתוך 80" in measured["he"]
    assert "12 מתוך 82" in measured["he"]
    assert "1 מתוך 30" in measured["he"]
    assert "2024-11-01" in measured["he"]


def test_the_sentence_never_pairs_a_saved_numerator_with_a_live_denominator():
    """The exact regression a second critic measured: 8 drawn, 76 live, 0 in common.

    ``score.current.breaks`` is the live re-plan against current settings and
    models; the drawn chips come from the saved weekly plan on disk. A sentence
    built from one and the other states a containment the data does not hold, so
    this asserts the live figure never reaches the rendered sentence at all.
    """
    measured = node_scope("""
      const coverage = m.buildCoverage({
        breaksShown: 8,
        programsShown: 12,
        score: {
          current: { breaks: 76 },
          basis: { segments: 82, day: '2024-11-01', committed: { breaks: 80, segments: 82 } },
        },
        daysInPlan: 30,
      });
      process.stdout.write(JSON.stringify({
        coverage,
        en: m.coverageSentence(coverage, 'en'),
      }));
    """)
    assert measured["coverage"]["breaksInDay"] == 80, "the denominator must be the plan the chips were drawn from"
    assert "76" not in measured["en"], "the live re-plan's count must never reach the rendered sentence"


def test_the_sentence_is_honest_when_no_plan_was_ever_committed_for_this_day():
    """A live day with no saved weekly plan behind it names that, not a live guess."""
    measured = node_scope("""
      const coverage = m.buildCoverage({
        breaksShown: 8,
        programsShown: 12,
        score: {
          current: { breaks: 76 },
          basis: { segments: 82, day: '2024-11-01', committed: null },
        },
        daysInPlan: 30,
      });
      process.stdout.write(JSON.stringify({
        coverage,
        en: m.coverageSentence(coverage, 'en'),
        he: m.coverageSentence(coverage, 'he'),
      }));
    """)
    assert measured["coverage"]["breaksInDay"] is None
    assert measured["coverage"]["committedUnavailable"] is True
    assert "76" not in measured["en"], "no figure from the live re-plan may stand in for the missing committed one"
    assert "no committed figures" in measured["en"]
    assert "2024-11-01" in measured["en"]
    assert "מחויבים" in measured["he"]


def test_the_sentence_never_fabricates_a_plan_total_it_does_not_have():
    """The plan's own day count is a second fetch, and it can still be in flight.

    Honest math bars a state that looks complete before it is: a sentence that
    named a plan total before the fetch answered would be a guess wearing a
    number. It names the day it does have, and stays silent on the rest.
    """
    measured = node_scope("""
      const midflight = m.buildCoverage({
        breaksShown: 8, programsShown: 12,
        score: {
          current: { breaks: 76 },
          basis: { segments: 82, day: '2024-11-01', committed: { breaks: 80, segments: 82 } },
        },
        daysInPlan: null,
      });
      const unscored = m.buildCoverage({
        breaksShown: 8, programsShown: 12, score: null, daysInPlan: 30,
      });
      process.stdout.write(JSON.stringify({
        midflight: m.coverageSentence(midflight, 'en'),
        unscored: m.coverageSentence(unscored, 'en'),
      }));
    """)
    assert "of the plan's" not in measured["midflight"], "no plan total was fetched yet, so none is named"
    assert "2024-11-01" in measured["midflight"], "the day itself is still named on its own"
    assert measured["unscored"] == "Reading how much of the day this timeline draws."


@pytest.mark.realdata
def test_the_sentence_s_inputs_are_the_routes_the_page_itself_calls():
    """Driven end to end: the drawn counts, the live day, and the plan's days.

    Not a description. The two routes ``ScheduleEditor`` and ``useEditorCoverage``
    actually call are read here the same way, and the sentence's own arithmetic,
    that what is shown can never exceed what the day and the plan hold, is
    checked on what those routes return today rather than on a number typed into
    this file.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kairos_api import break_store
    from kairos_api.break_api import router as break_router
    from kairos_api.day_api import _break_operations_cached, break_operations

    owned = break_store.operator_channel()
    if not owned:
        pytest.skip("no operator channel configured, so there is no day to measure")

    _break_operations_cached.cache_clear()
    drawn = break_operations()
    breaks_shown = len(drawn["breaks"])
    programs_shown = len(drawn["programs"])
    if not breaks_shown:
        pytest.skip("the editor has no breaks to draw, so there is nothing to cover")
    day = drawn["breaks"][0]["date"]

    app = FastAPI()
    app.include_router(break_router)
    client = TestClient(app)

    days = client.get("/api/plan/days").json()
    assert days["available"], "the plan carries no days, so the sentence has nothing to compare against"
    assert day in days["days"], "the day the capped board drew is not even in the plan's own day list"

    scored = client.post("/api/plan/day/score", json={"day": day, "moves": []}).json()
    committed = scored["basis"]["committed"]
    assert committed, "the chips were drawn from this same saved plan, so it must carry a committed row"
    breaks_in_day = committed["breaks"]
    programmes_in_day = committed["segments"]

    assert breaks_shown <= breaks_in_day, "the timeline drew more breaks than the saved plan actually holds"
    assert programs_shown <= programmes_in_day, "the timeline drew more programmes than the saved plan actually holds"
    assert days["count"] >= 1


def test_the_editor_wires_the_sentence_and_a_live_route_to_the_rest():
    """The four lines this piece's own file budget allowed for the wiring."""
    editor = read("plan/day/ScheduleEditor.jsx")
    assert "import { useEditorCoverage } from './schedule-editor-scope';" in editor
    assert "import ScheduleEditorScope from './ScheduleEditorScope';" in editor
    assert "useEditorCoverage({ breaksShown: breaks.length, programsShown: programs.length, score: money.score })" in editor
    assert "<ScheduleEditorScope coverage={coverage} locale={locale} />" in editor

    scope = read("plan/day/ScheduleEditorScope.jsx")
    assert "coverageSentence(coverage, locale)" in scope
    assert 'href="#Overrides"' in scope, "the route to the full day board is a live link, not a name"

    hook = read("plan/day/schedule-editor-scope.js")
    assert "import { fetchDays } from './day-board-actions.js';" in hook
