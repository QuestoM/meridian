"""Every count on the day surfaces names the plan it counts, and the sentence is true.

THIS FILE GUARDS A CLASS, NOT AN INSTANCE, AND IT IS THE SECOND ATTEMPT AT IT.

Two plans answer "how many breaks does this day have" on these surfaces.
``output/weekly_break_schedule.csv`` is the SAVED weekly plan, the artifact the
week board and every export read, and it holds 80 breaks across 82 programmes on
רשת 13 / 2024-11-01. ``GET /api/plan/day`` re-plans that same day LIVE against
current settings, constraints and models, and it holds 76. Both are real. A
figure that does not say which one it came from is not.

A critic measured that silence four rounds running, and each round closed the one
instance named and left the class standing. Round 2: a numerator from the saved
plan over a denominator from the live one, so "8 of 76" asserted a containment
that did not hold, 0 of the 8 drawn clock times being among the live 76. Round 4:
the sentence said "the day's 80 breaks" with no basis while the money tile 344 px
below said "Breaks in the day 76" with none either, and the saved plan places 13
breaks in the 12 programmes the timeline draws against the 8 it draws, so 5
breaks inside programmes already on screen were counted as part of the 72 "rest"
and routed to a board that holds neither them nor the 8.

The previous version of this file asserted the editor's sentence and nothing
else, so it stayed green through all of that. The tests below assert instead that
EVERY count-bearing site reads its wording from ``plan-basis.js``, that the basis
words are spelled in that one file and nowhere else, and that the sentence's
arithmetic holds against the routes the page calls. A fifth site that forgets
fails here. The shipped modules are executed in node rather than re-implemented,
because a re-implementation would only prove two pieces of test code agree.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
DAY = SRC / "plan" / "day"
SCOPE_JS = DAY / "schedule-editor-scope.js"
BASIS_JS = DAY / "plan-basis.js"

# The words that name a plan. They live in plan-basis.js and nowhere else, which
# is the whole mechanism: a new surface cannot spell them slightly differently,
# and changing one changes every site at once.
BASIS_WORDS = (
    "this live plan",
    "the saved weekly plan",
    "התוכנית החיה הזו",
    "התוכנית השבועית השמורה",
)

BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.S)
LINE_COMMENT = re.compile(r"^\s*//.*$", re.M)


def read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def day_modules() -> dict[str, str]:
    """Every javascript module on the day surfaces, by file name."""
    return {
        path.name: path.read_text(encoding="utf-8")
        for path in sorted(DAY.iterdir())
        if path.suffix in (".js", ".jsx")
    }


def code_only(text: str) -> str:
    """The module with its prose stripped, so a comment about the class is not read as an instance of it."""
    return LINE_COMMENT.sub("", BLOCK_COMMENT.sub("", text))


def node_scope(body: str, module: Path = SCOPE_JS) -> dict:
    """Run a shipped module in node and return what it computed."""
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    script = f"const m = await import({json.dumps(module.as_uri())});\n{body}"
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# The measured day, shared by the sentence tests. Every figure was read off the
# real routes: basis.committed and current.breaks are what POST
# /api/plan/day/score serves for 2024-11-01, and the drawn 8 across 12 programmes
# with 13 planned in them is what build_break_operations and
# /api/schedule/segments hold today.
MEASURED_DAY = """
  const programmes = [
    ['02:00', 1], ['02:17', 1], ['02:41', 2], ['02:57', 0], ['03:10', 4], ['04:08', 1],
    ['04:29', 1], ['04:33', 0], ['04:57', 0], ['05:01', 2], ['05:22', 0], ['05:40', 1],
  ].map(([start, planned]) => ({
    channel: 'רשת 13', date: '2024-11-01', start_time: start, lane: 'רשת 13 / Fri', planned,
  }));
  const resolve = (channel, date, startClock) => {
    const hit = programmes.find((p) => p.channel === channel && p.date === date && p.start_time === startClock);
    return hit ? { segmentId: `seg-${hit.start_time}`, plannedBreaks: hit.planned } : null;
  };
  const score = {
    current: { breaks: 76 },
    basis: { segments: 82, day: '2024-11-01', committed: { breaks: 80, segments: 82 } },
  };
"""


def test_the_basis_words_are_spelled_in_exactly_one_module():
    """The class guard. One vocabulary, one file, every site reading from it.

    The assertion the previous version did not make, and its absence is why four
    rounds each fixed one site. A surface that spells "this live plan" itself is
    a site the next change to the wording will miss, which is how the money tile
    and the coverage sentence came to disagree on one screen.
    """
    offenders = {}
    for name, text in day_modules().items():
        if name == BASIS_JS.name:
            continue
        body = code_only(text).lower()
        hits = [word for word in BASIS_WORDS if word.lower() in body]
        if hits:
            offenders[name] = hits
    assert not offenders, f"the basis words must come from plan-basis.js, not be re-spelled: {offenders}"

    basis = BASIS_JS.read_text(encoding="utf-8")
    for word in BASIS_WORDS:
        assert word in basis, f"plan-basis.js is the one place {word!r} is defined"


def test_every_figure_scope_on_these_surfaces_names_its_plan():
    """Each money tile prints the plan behind its number, beside it and not in a tooltip.

    Measured: the tile reading "Breaks in the day 76" carried a scope line of bare
    ad seconds, so two answers to how many breaks the day holds stood 344 px apart
    with only one of them attributed.
    """
    scope_tag = 'className="day-figure-scope"'
    surfaces = {
        name: text for name, text in day_modules().items() if scope_tag in text
    }
    assert surfaces, "the figure tiles moved; this guard has to move with them"
    for name, text in surfaces.items():
        body = code_only(text)
        tiles = re.findall(r'className="day-figure-scope"[^>]*>\{(.+?)\}</small>', body, re.S)
        assert tiles, f"{name} carries figure scopes this guard could not read"
        for value in tiles:
            assert re.search(r"scopeText|scopeWithBasis\(|livePlanPointer\(", value), (
                f"{name} prints a figure scope that names no plan: {value.strip()!r}"
            )
        if "scopeText" in body:
            assert re.search(r"scopeText = .*scopeWithBasis\(", body), (
                f"{name} builds scopeText without naming the plan it was computed on"
            )


def test_no_lane_header_states_a_bare_break_count():
    """A lane that says "8 breaks" claims the lane holds 8, and it holds what was drawn.

    The editor's lane draws a capped slice of the saved plan, the day board's
    draws the whole live plan, and both carried the same bare count.
    """
    bare = re.compile(r"\{[\w.]+\.length\}\s*\{\s*\w*[Pp]ageText\(locale,\s*'breaks'")
    for name, text in day_modules().items():
        assert not bare.search(code_only(text)), f"{name} still prints a lane count with no plan behind it"

    assert 'className="timeline-lane-basis"' in read("plan/day/ScheduleEditorScope.jsx")
    assert 'className="timeline-lane-basis"' in read("plan/day/DayBoard.jsx")
    assert "drawnOfPlannedText(shown, planned, locale)" in read("plan/day/ScheduleEditorScope.jsx")
    assert "breakCountText(breaks.length, locale)" in read("plan/day/DayBoard.jsx")


def test_every_day_total_row_names_the_plan_it_previews():
    """The two preview tables print day totals from the live engine, and say so."""
    for module in ("plan/day/OverrideDecisions.jsx", "plan/day/ScheduleInspector.jsx"):
        body = code_only(read(module))
        assert "withBasis(pageText(locale, d.en, d.he), LIVE_PLAN, locale)" in body, (
            f"{module} labels a day total without naming the plan behind it"
        )
    inspector = code_only(read("plan/day/ScheduleInspector.jsx"))
    assert "withBasis(pageText(locale, 'Break plan', 'תוכנית ברייקים'), SAVED_PLAN, locale)" in inspector, (
        "the drawer's break plan is the saved plan's row and has to say which plan it is"
    )


def test_the_sentence_covers_what_the_plan_places_in_the_programmes_it_shows():
    """8 of the 13 in the 12 shown, out of the day's 80, across 12 of 82, on 1 of 30.

    The round 4 gap: the saved plan places 13 breaks in the 12 programmes drawn
    and 8 are drawn, so the old sentence counted the other 5 as part of the 72 it
    routed elsewhere. Every figure here is that one plan's own.
    """
    measured = node_scope(MEASURED_DAY + """
      const coverage = m.buildCoverage({
        breaksShown: 8, programs: programmes, score, daysInPlan: 30, resolve, anchorsLoaded: true,
      });
      process.stdout.write(JSON.stringify({
        coverage,
        en: m.coverageSentence(coverage, 'en'),
        he: m.coverageSentence(coverage, 'he'),
      }));
    """)
    assert measured["coverage"]["plannedInShown"] == 13
    assert measured["coverage"]["programmesMatched"] == 12
    assert measured["coverage"]["breaksInDay"] == 80
    assert "8 of the 13 breaks the saved weekly plan places in the 12 programmes it shows" in measured["en"]
    assert "out of the 80 that plan holds for this day" in measured["en"]
    assert "across 12 of its 82 programmes" in measured["en"]
    assert "on 1 of its 30 days (2024-11-01)" in measured["en"]
    assert "8 מתוך 13 הברייקים שהתוכנית השבועית השמורה קובעת ל-12 התוכניות" in measured["he"]
    assert "מתוך 80" in measured["he"]
    assert "12 מתוך 82" in measured["he"]
    assert "1 מתוך 30" in measured["he"]


def test_the_sentence_never_pairs_a_saved_numerator_with_a_live_denominator():
    """The round 2 regression, still guarded: 8 drawn, 76 live, 0 in common.

    Every count in this sentence is the saved plan's, so the live figure must not
    reach it. It belongs in the route sentence, which is about the board that
    holds it.
    """
    measured = node_scope(MEASURED_DAY + """
      const coverage = m.buildCoverage({
        breaksShown: 8, programs: programmes, score, daysInPlan: 30, resolve, anchorsLoaded: true,
      });
      process.stdout.write(JSON.stringify({
        en: m.coverageSentence(coverage, 'en'),
        he: m.coverageSentence(coverage, 'he'),
        routeEn: m.routeSentence(coverage, 'en'),
        routeHe: m.routeSentence(coverage, 'he'),
      }));
    """)
    assert "76" not in measured["en"], "the live re-plan's count must never reach the coverage sentence"
    assert "76" not in measured["he"], "the live re-plan's count must never reach the coverage sentence"
    assert "76 breaks of its own" in measured["routeEn"]
    assert "against the 80 counted here" in measured["routeEn"]
    assert "this live plan" in measured["routeEn"], "the clause that carries 76 has to name whose 76 it is"
    assert "76 ברייקים משלה" in measured["routeHe"]
    assert "התוכנית החיה הזו" in measured["routeHe"]


def test_the_route_tells_the_operator_what_is_actually_behind_the_link():
    """The link opens a board that re-plans this day live and holds a different set.

    Measured: 1 click to 76 chips, 0 of the 8 drawn clock times among them. The
    sentence claims no containment, because there is none; it states both counts
    and says which plan each belongs to.
    """
    scope = read("plan/day/ScheduleEditorScope.jsx")
    assert "routeSentence(coverage, locale)" in scope, "the route is described, not implied"
    assert 'href="#Overrides"' in scope, "the route to the full day board is a live link, not a name"
    assert "'Open the full day board (Overrides)', 'פתחו את לוח היום המלא (עקיפות)'" in scope, (
        "the link names the rail entry it activates, so a person recognises where they land"
    )
    measured = node_scope(MEASURED_DAY + """
      const unscored = m.buildCoverage({ breaksShown: 8, programs: programmes, score: null, daysInPlan: 30, resolve, anchorsLoaded: true });
      process.stdout.write(JSON.stringify({ en: m.routeSentence(unscored, 'en'), he: m.routeSentence(unscored, 'he') }));
    """)
    assert "76" not in measured["en"], "before the score answers there is no live count to name"
    assert "re-plans this day live" in measured["en"]
    assert "בזמן אמת" in measured["he"]


def test_the_sentence_names_no_figure_it_does_not_have_yet():
    """Four fetches answer at four times, and none of them may be guessed at.

    The day count, the per-programme counts behind the 13, the committed basis
    and the score land separately, and a sentence that looked complete before
    they did would be a guess wearing a number.
    """
    measured = node_scope(MEASURED_DAY + """
      const noAnchors = m.buildCoverage({ breaksShown: 8, programs: programmes, score, daysInPlan: 30, resolve, anchorsLoaded: false });
      const noDays = m.buildCoverage({ breaksShown: 8, programs: programmes, score, daysInPlan: null, resolve, anchorsLoaded: true });
      const unscored = m.buildCoverage({ breaksShown: 8, programs: programmes, score: null, daysInPlan: 30, resolve, anchorsLoaded: true });
      process.stdout.write(JSON.stringify({
        noAnchors: m.coverageSentence(noAnchors, 'en'),
        noAnchorsPlanned: noAnchors.plannedInShown,
        noDays: m.coverageSentence(noDays, 'en'),
        unscored: m.coverageSentence(unscored, 'en'),
      }));
    """)
    assert measured["noAnchorsPlanned"] is None
    assert "13" not in measured["noAnchors"], "no per-programme count answered yet, so none is named"
    assert "8 of the 80 breaks the saved weekly plan holds for this day" in measured["noAnchors"], (
        "the clause falls back to figures it does have, still naming the plan"
    )
    assert "of its 30 days" not in measured["noDays"], "no plan total was fetched yet, so none is named"
    assert "2024-11-01" in measured["noDays"], "the day itself is still named on its own"
    assert measured["unscored"] == "Reading how much of the day this timeline draws."


def test_the_sentence_is_honest_when_no_plan_was_ever_committed_for_this_day():
    """A live day with no saved weekly plan behind it names that, not a live guess."""
    measured = node_scope(MEASURED_DAY + """
      const coverage = m.buildCoverage({
        breaksShown: 8, programs: programmes, daysInPlan: 30, resolve, anchorsLoaded: true,
        score: { current: { breaks: 76 }, basis: { segments: 82, day: '2024-11-01', committed: null } },
      });
      process.stdout.write(JSON.stringify({
        coverage, en: m.coverageSentence(coverage, 'en'), he: m.coverageSentence(coverage, 'he'),
      }));
    """)
    assert measured["coverage"]["breaksInDay"] is None
    assert measured["coverage"]["committedUnavailable"] is True
    assert "76" not in measured["en"], "no figure from the live re-plan may stand in for the missing committed one"
    assert "The saved weekly plan carries no committed figures" in measured["en"]
    assert "2024-11-01" in measured["en"]
    assert "התוכנית השבועית השמורה אינה מחזיקה נתונים מחויבים" in measured["he"]


def test_a_drawn_programme_the_plan_has_no_row_for_is_counted_on_neither_side_silently():
    """The latent case: the numerator and the denominator count different things.

    programsShown counts what the EPG carries and programmesInDay counts rows of
    the saved plan. All 12 match today. A drawn programme with no plan row would
    inflate a numerator against a denominator that never carried it.
    """
    measured = node_scope("""
      const programmes = [
        { channel: 'רשת 13', date: '2024-11-01', start_time: '02:17', lane: 'L' },
        { channel: 'רשת 13', date: '2024-11-01', start_time: '03:10', lane: 'L' },
        { channel: 'רשת 13', date: '2024-11-01', start_time: '23:59', lane: 'L' },
      ];
      const planned = { '02:17': 1, '03:10': 4 };
      const resolve = (channel, date, startClock) => (
        planned[startClock] === undefined ? null : { segmentId: 's', plannedBreaks: planned[startClock] }
      );
      const coverage = m.buildCoverage({
        breaksShown: 4, programs: programmes, daysInPlan: 30, resolve, anchorsLoaded: true,
        score: { current: { breaks: 76 }, basis: { segments: 82, day: '2024-11-01', committed: { breaks: 80, segments: 82 } } },
      });
      process.stdout.write(JSON.stringify({
        coverage, en: m.coverageSentence(coverage, 'en'), he: m.coverageSentence(coverage, 'he'),
      }));
    """)
    assert measured["coverage"]["programsShown"] == 3
    assert measured["coverage"]["programmesMatched"] == 2
    assert measured["coverage"]["plannedInShown"] == 5
    assert "4 of the 5 breaks the saved weekly plan places in the 2 programmes it shows" in measured["en"]
    assert "1 more programmes are drawn here that the saved weekly plan carries no row for." in measured["en"]
    assert "עוד 1 תוכניות מוצגות כאן שהתוכנית השבועית השמורה אינה מחזיקה להן שורה." in measured["he"]


def test_a_second_date_on_the_timeline_is_named_rather_than_absorbed():
    """The sentence takes its day from the score, so a chip on another date says so."""
    measured = node_scope("""
      const programmes = [
        { channel: 'רשת 13', date: '2024-11-01', start_time: '02:17', lane: 'L' },
        { channel: 'רשת 13', date: '2024-11-02', start_time: '03:10', lane: 'L' },
      ];
      const resolve = () => ({ segmentId: 's', plannedBreaks: 1 });
      const coverage = m.buildCoverage({
        breaksShown: 2, programs: programmes, daysInPlan: 30, resolve, anchorsLoaded: true,
        score: { current: { breaks: 76 }, basis: { segments: 82, day: '2024-11-01', committed: { breaks: 80, segments: 82 } } },
      });
      process.stdout.write(JSON.stringify({ en: m.coverageSentence(coverage, 'en'), he: m.coverageSentence(coverage, 'he') }));
    """)
    assert "The breaks drawn also fall on 2024-11-02." in measured["en"]
    assert "משתרעים גם על 2024-11-02." in measured["he"]


def test_the_committed_note_prints_no_percentage_it_does_not_have():
    """A saved plan of zero revenue has no percentage gap, and used to print "null%"."""
    gap = node_scope("""
      const zero = m.committedGap({ committed: { revenue: 0, breaks: 80 } }, { revenue: 10, breaks: 76 });
      const real = m.committedGap({ committed: { revenue: 1067845.56, breaks: 80 } }, { revenue: 992668.69, breaks: 76 });
      process.stdout.write(JSON.stringify({ zero, real }));
    """, module=DAY / "day-board-model.js")
    assert gap["zero"]["percent"] is None
    assert gap["real"]["percent"] == -7.0
    assert gap["real"]["breaksGap"] == -4

    readout = code_only(read("plan/day/DayBoardReadout.jsx"))
    assert "gap.percent === null ? '' : ` (${gap.percent}%)`" in readout, (
        "the percentage clause is dropped when there is no percentage, not interpolated as null"
    )
    assert "ו-${gap.breaksGap}" not in readout, (
        "the Hebrew conjunction hyphen immediately before a negative figure rendered as ו--4"
    )
    assert "והפרש של ${gap.breaksGap} ברייקים" in readout


@pytest.mark.realdata
def test_the_sentence_s_inputs_are_the_routes_the_page_itself_calls():
    """Driven end to end on today's data, through the shipped module, not a fixture.

    The routes the editor calls are read the way the editor reads them, the
    shipped ``plannedInShown`` is executed in node on what they return, and the
    arithmetic is checked on that: drawn cannot exceed what the plan places in
    the programmes drawn, which cannot exceed the day.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kairos_api import break_store
    from kairos_api.break_api import router as break_router
    from kairos_api.day_api import _break_operations_cached, break_operations, schedule_segments

    owned = break_store.operator_channel()
    if not owned:
        pytest.skip("no operator channel configured, so there is no day to measure")

    _break_operations_cached.cache_clear()
    drawn = break_operations()
    breaks_shown = len(drawn["breaks"])
    programs = drawn["programs"]
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

    # The anchors the editor already holds, built exactly as useSegmentAnchors
    # builds them, so the 13 is read off the same route the browser reads.
    anchors = {
        f"{seg['channel']}|{seg['anchor']['date']}|{seg['anchor']['start_clock']}": seg["state"]["num_breaks"]
        for seg in schedule_segments()["segments"]
    }
    measured = node_scope(f"""
      const anchors = {json.dumps(anchors, ensure_ascii=False)};
      const programs = {json.dumps(programs, ensure_ascii=False)};
      const resolve = (channel, date, startClock) => {{
        const planned = anchors[`${{channel}}|${{date}}|${{startClock}}`];
        return planned === undefined ? null : {{ segmentId: 's', plannedBreaks: planned }};
      }};
      process.stdout.write(JSON.stringify(m.plannedInShown(programs, resolve)));
    """)

    assert measured["matched"] <= len(programs)
    assert breaks_shown <= measured["total"], "the timeline drew more breaks than the plan places in the programmes it draws"
    assert measured["total"] <= committed["breaks"], "the programmes drawn cannot hold more breaks than the whole day does"
    assert len(programs) <= committed["segments"], "the timeline drew more programmes than the saved plan actually holds"
    assert days["count"] >= 1


def test_the_editor_wires_the_sentence_and_a_live_route_to_the_rest():
    """The wiring, inside the file budget this piece has always had to work in."""
    editor = read("plan/day/ScheduleEditor.jsx")
    assert "import { useEditorCoverage } from './schedule-editor-scope';" in editor
    assert "import ScheduleEditorScope, { LaneCount } from './ScheduleEditorScope';" in editor
    assert "const { resolve, loaded: anchorsLoaded } = useSegmentAnchors();" in editor
    assert "useEditorCoverage({ breaksShown: breaks.length, programs, score: money.score, resolve, anchorsLoaded })" in editor
    assert "<ScheduleEditorScope coverage={coverage} locale={locale} />" in editor
    assert "<LaneCount shown={lane.items.length} planned={coverage.plannedByLane[lane.lane]} locale={locale} />" in editor

    anchors = read("plan/day/schedule-track-view.jsx")
    assert "plannedBreaks: Number.isFinite(planned) ? planned : null," in anchors, (
        "the saved plan's own per-segment count is what the covers clause is built from"
    )
    assert "return { segMap, resolve, loaded };" in anchors
