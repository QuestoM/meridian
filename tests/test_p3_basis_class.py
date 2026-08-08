"""No count on the day surfaces may hide which of the two plans it counts.

THIS FILE GUARDS A CLASS. It is the guard four rounds of a build-critique loop
did not have, and its absence is why each round closed exactly the instance the
critic named and the same defect reappeared at the next site.

Two plans answer "how many breaks does this day have".
``output/weekly_break_schedule.csv`` is the SAVED weekly plan, the artifact the
week board and every export read, and it holds 80 breaks across 82 programmes on
רשת 13 / 2024-11-01. ``GET /api/plan/day`` re-plans that same day LIVE against
current settings, constraints and models, and it holds 76. Both are real. A
figure that does not say which one it came from is not.

Measured on the shipped build: the editor's coverage sentence said "the day's 80
breaks" and the money tile 344 px below it said "Breaks in the day 76" over a
scope line of bare ad seconds. Two answers to one question, on one screen,
neither attributed.

So the wording lives in one module, ``src/plan/day/plan-basis.js``, and the
tests here assert that every site reads from it and that the words are spelled
nowhere else. A new surface that re-spells them is a failure here rather than a
fifth round. ``test_p3_editor_coverage.py`` holds the arithmetic of the sentence
those words appear in.
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


def test_the_basis_words_are_spelled_in_exactly_one_module():
    """One vocabulary, one file, every site reading from it.

    A surface that spells "this live plan" itself is a site the next change to
    the wording will miss, which is how the money tile and the coverage sentence
    came to disagree on one screen.
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
    surfaces = {name: text for name, text in day_modules().items() if scope_tag in text}
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


def test_the_committed_note_prints_no_percentage_it_does_not_have():
    """A saved plan of zero revenue has no percentage gap, and used to print "null%"."""
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    module = (DAY / "day-board-model.js").as_uri()
    script = (
        f"const m = await import({json.dumps(module)});\n"
        "const zero = m.committedGap({ committed: { revenue: 0, breaks: 80 } }, { revenue: 10, breaks: 76 });\n"
        "const real = m.committedGap({ committed: { revenue: 1067845.56, breaks: 80 } }, { revenue: 992668.69, breaks: 76 });\n"
        "process.stdout.write(JSON.stringify({ zero, real }));"
    )
    result = subprocess.run(
        # shell/bidi and shell/dates are real shell primitives the shipped module
        # imports; this loader hook resolves them to the real compiled files.
        ["node", "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    gap = json.loads(result.stdout)
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
