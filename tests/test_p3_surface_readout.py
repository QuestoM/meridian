"""What the day and break surfaces print, and where the controls land.

The second half of the mechanics this destination is graded on. The first half,
in ``test_p3_direct_manipulation.py``, is the gesture and its inverse: drag,
select, type, nudge, undo, and the scope a saved move binds. This half is what a
person reads once the gesture is done, which is where the measured defects were
of a different kind: a footer that totalled the whole day under a filtered
column, a filter that emptied a table and left the figure standing, a Save button
below the fold, and a record that could not be opened from the drawer that named
it.

Split from that file under the 450-line law. Nothing moved in behaviour: every
assertion below is the one that was there, at its new address.
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
BOARD_MODEL_JS = SRC / "plan" / "day" / "day-board-model.js"


def read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def node_board_model(body: str) -> dict:
    """Run the shipped board model in node and return what it computed."""
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    script = f"const m = await import({json.dumps(BOARD_MODEL_JS.as_uri())});\n{body}"
    result = subprocess.run(
        # shell/bidi and shell/dates are real shell primitives the shipped module
        # imports; this loader hook resolves them to the real compiled files.
        ["node", "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_day_board_releases_the_shared_scroll_floor_so_the_money_stays_on_screen():
    """The shared container reserves 440 px for a stack of lanes. This has one.

    Measured at 1440 x 823 on רשת 13 / 2024-11-01 before the override: the scroll
    box was 440 px tall around 148 px of content, so the hour strip began at
    869 px, the money readout at 927 px and the save, undo and discard buttons at
    1043 px, every one of them below the fold on a day that fits the screen.
    After it: 148 px, 577 px, 635 px and 751 px, all of them on screen.
    """
    timeline = (SRC / "shell" / "styles-timeline.css").read_text(encoding="utf-8")
    assert "min-height: 440px;" in timeline, "the shared floor moved, so this override wants re-measuring"
    css = (SRC / "plan" / "day" / "day-board.css").read_text(encoding="utf-8")
    block = css.split(".day-board .timeline-scroll {")[1].split("}")[0]
    assert "min-height: 0;" in block


def test_the_controls_stay_on_screen_on_the_day_a_person_needs_them_most():
    """A failing day must not push its own undo off the bottom of the screen.

    Same viewport, same day, one break lengthened until three checks fail.
    Measured with the verdicts stacked one per line: the save, undo and discard
    row began at 819 px in an 823 px viewport. Side by side: 786 px, with all
    three verdicts still rendered in full, each with its scope and its observed
    against its limit.
    """
    css = (SRC / "plan" / "day" / "day-readout.css").read_text(encoding="utf-8")
    block = css.split(".day-violations {")[1].split("}")[0]
    assert "flex-wrap: wrap;" in block, "the verdicts share a line and wrap on a narrow window"
    item = css.split(".day-violations li {")[1].split("}")[0]
    # Each verdict is still set off as its own box, and the shape that does it
    # moved. It used to be a one-sided rule down the inline-start edge; that is
    # now banned outright by docs/ux-gauntlet/design-rules.md section 1, because a
    # rule on one edge reads as an unfinished frame and lands on the opposite edge
    # under right-to-left. A full border in the same colour says the same thing in
    # both languages, and the inline padding that came with it keeps the text off
    # that border. A correction, not a rename: do not put the one-sided rule back.
    assert "border: 1px solid var(--red);" in item, (
        "each verdict keeps a box of its own, in the state colour for a breach"
    )
    assert "border-inline-start" not in item and "border-left" not in item and "border-right" not in item, (
        "a one-sided accent bar is banned, and a physical side does not mirror at all"
    )
    assert "padding-inline-start:" in item, "the text sits off its own border"
    assert "padding-left:" not in item and "padding-right:" not in item, (
        "padding stated as a physical side stays on the same edge when Hebrew flips the box"
    )
    readout = read("plan/day/DayBoardReadout.jsx")
    assert "{violations.map((violation, index) => (" in readout, "every failed check is still rendered"
    assert "{formatNumber(violation.observed, locale)} / {formatNumber(violation.limit, locale)}" in readout


def test_the_drawer_keeps_its_place_in_the_set_it_was_opened_from():
    """The list-position device: a counter and two arrows, plus the keyboard."""
    inspector = read("plan/break/BreakInspector.jsx")
    assert "break-inspector-walk" in inspector
    assert "{index + 1} / {set.length}" in inspector
    assert "ArrowDown" in inspector and "ArrowUp" in inspector
    assert "disabled={index === 0}" in inspector
    day_page = read("plan/day/DayPage.jsx")
    assert "siblings={breakIds}" in day_page
    board = read("plan/break/BreakBoard.jsx")
    assert "siblings={rows.map((row) => row.break_id)}" in board


def test_a_programme_on_the_day_board_opens_its_own_record():
    """The one move the surface this destination replaces still won on.

    Measured in a browser on רשת 13 / 2024-11-01 before this closed: the board
    drew 82 ``.timeline-program-band`` elements, none of them carrying
    ``role="button"`` or a tabindex, and a click on one opened nothing. The
    shipped editor opened the programme inspector in 63 ms from a click on a
    break chip, showing that segment's class, break plan and its own economics.
    So a scheduler could reach a programme's record from the timeline that
    morning and not from the board that replaced it.
    """
    timeline = read("plan/day/DayBoardTimeline.jsx")
    assert "clickable={Boolean(onOpenProgramme) && widthPx >= 44}" in timeline
    assert "onOpen={() => onOpenProgramme(programme)}" in timeline
    assert 'className="day-short-programmes"' in timeline
    assert "onClick={() => onOpenProgramme(programme)}" in timeline, (
        "programmes too narrow to carry a legible band remain directly reachable"
    )
    band = read("plan/day/schedule-track-view.jsx")
    assert 'role="button"' in band and "tabIndex={0}" in band, "a band marked clickable is a real control"
    page = read("plan/day/DayPage.jsx")
    assert "import ScheduleInspector from './ScheduleInspector';" in page
    assert "onOpenProgramme={onOpenProgramme}" in page
    assert "segmentId={openProgramme.segmentId}" in page
    assert "channel={openProgramme.channel}" in page and "day={openProgramme.day}" in page
    assert "setOpenBreak(null);" in page, "one record is open at a time"


def test_an_hour_bar_is_a_control_and_not_a_box_with_a_title_on_it():
    """25 inert divs carrying a title attribute reached neither key nor reader."""
    strip = read("plan/day/DayBoardReadout.jsx").split("export function HourStrip")[1]
    assert "<Pressable" in strip and 'type="button"' in strip
    assert "../../studio/dom-controls" in read("plan/day/DayBoardReadout.jsx")
    assert "aria-label=" in strip, "the load and the limit have to be in the accessible name"
    assert "aria-pressed=" in strip
    assert "onOpenHour" in strip
    board = read("plan/day/DayBoard.jsx")
    assert "firstBreakInHour(breaks, programmes, liveOf, hour)" in board
    css = (SRC / "plan" / "day" / "day-readout.css").read_text(encoding="utf-8")
    assert ".day-hour:focus-visible" in css, "a control that cannot be seen focused is not keyboard reachable"


def test_the_hour_resolves_to_the_first_break_the_plan_puts_in_it():
    """Pointing at an hour means the break inside it, earliest first."""
    measured = node_board_model("""
      const programmes = new Map([['s', { segment_id: 's', start_seconds: 3600, duration_seconds: 7200 }]]);
      const breaks = [
        { break_id: 's~1', segment_id: 's', offset_seconds: 3000, duration_seconds: 120, is_gold: false },
        { break_id: 's~2', segment_id: 's', offset_seconds: 600, duration_seconds: 120, is_gold: false },
        { break_id: 's~3', segment_id: 's', offset_seconds: 6600, duration_seconds: 120, is_gold: false },
      ];
      const liveOf = (item) => m.liveBreak(item, {});
      process.stdout.write(JSON.stringify({
        first: m.firstBreakInHour(breaks, programmes, liveOf, 1),
        second: m.firstBreakInHour(breaks, programmes, liveOf, 2),
        empty: m.firstBreakInHour(breaks, programmes, liveOf, 9),
      }));
    """)
    assert measured["first"] == "s~2", "the earliest break in the hour, not the first in the list"
    assert measured["second"] == "s~3"
    assert measured["empty"] is None, "an hour the plan puts no break in changes nothing"


def test_no_record_named_on_the_break_drawer_is_a_dead_end():
    """The drawer's own module docstring promises three, so three are checked.

    Measured before this closed: the drawer's controls were exactly three, both
    walk arrows and Close, and the programme title ``משחקי השף עונה 7 ש.ח`` was
    plain text with no ``dd a``, ``dd button`` or ``dd [role=button]`` anywhere
    on it.
    """
    inspector = read("plan/break/BreakInspector.jsx")
    assert "import ScheduleInspector, { confidenceLabel } from '../day/ScheduleInspector';" in inspector
    assert "onClick={() => setProgrammeOpen(true)}" in inspector
    assert "segmentId={detail.programme.segment_id}" in inspector
    assert "channel={detail.identity.channel}" in inspector and "day={detail.identity.day}" in inspector
    assert "aria-expanded={hourOpen}" in inspector, "the hour opens the breaks that make its load"
    assert "onClick={() => onNavigate(row.break_id)}" in inspector
    assert "aria-expanded={pinOpen}" in inspector, "the restriction opens the record that carries it"
    assert "if (programmeOpen) setProgrammeOpen(false);" in inspector, "escape closes the top record, not the stack"
    assert "if (programmeOpen) return;" in inspector, "the arrows stop at the record on top"


def test_the_drawer_never_prints_an_engine_word_where_a_person_reads_a_level():
    """An otherwise fully Hebrew drawer read ``רמת ביטחון: medium``."""
    inspector = read("plan/break/BreakInspector.jsx")
    assert "{confidenceLabel(detail.retention.confidence, locale)}" in inspector
    assert "<dd>{detail.retention.confidence}</dd>" not in inspector
    editor = read("plan/day/ScheduleInspector.jsx")
    assert "export function confidenceLabel" in editor, "one translation of one engine vocabulary, not two"
    for word in ("גבוהה", "בינונית", "נמוכה"):
        assert word in editor


def test_money_on_this_surface_is_printed_exactly_and_never_compacted():
    """The day total and its own column footer have to be the same number.

    The shared formatter compacts at 100,000, so a day worth 1,062,669.88 would
    print as 1.06M in the footer while the column above it printed every break to
    the shekel. This surface prints both exactly.
    """
    model = read("plan/day/day-board-model.js")
    assert "export function exactCurrency" in model
    body = model.split("export function exactCurrency")[1].split("export function")[0]
    assert "notation" not in body, "an exact figure never asks for compact notation"
    assert "maximumFractionDigits: 0" in body
    for path in ("plan/day/DayBoardReadout.jsx", "plan/break/BreakBoard.jsx", "plan/break/BreakInspector.jsx"):
        text = read(path)
        assert "formatCurrency(" not in text, f"{path} still compacts a figure at 100,000"
        assert "exactCurrency(" in text


def node_break_model(body: str) -> dict:
    """Run the shipped break board money rules in node, same reason as above."""
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    module = SRC / "plan" / "break" / "break-board-model.js"
    script = f"const m = await import({json.dumps(module.as_uri())});\n{body}"
    result = subprocess.run(
        # shell/bidi and shell/dates are real shell primitives the shipped module
        # imports; this loader hook resolves them to the real compiled files.
        ["node", "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# The three real credits behind the three gold rows in the reproduction below,
# read from GET /api/plan/day on רשת 13 / 2024-11-01 with 001~1 marked gold. The
# mark moves the plan, so the day itself is 1,028,205.58 while it is in force.
GOLD_TRIO = [
    {"break_id": "001~1", "is_gold": True, "projected_revenue": 10711.71},
    {"break_id": "001~2", "is_gold": True, "projected_revenue": 10162.61},
    {"break_id": "001~3", "is_gold": True, "projected_revenue": 9613.52},
]


def test_the_column_footer_totals_the_rows_above_it_and_never_the_whole_day():
    """The label says these breaks, so the figure has to be these breaks.

    Reproduced in a browser on רשת 13 / 2024-11-01 at 1440 x 823 before this
    closed: mark 001~1 gold from the day board, which marks the three breaks of
    its programme, then open the break board and press the ברייקי זהב filter.
    Three rows remain, printing 10,712, 10,163 and 9,614, and the line under them
    printed 1,028,206 ILS under the label סכום על הברייקים האלה, the whole day,
    33.7 times the column it claimed to total, while the note under the table
    still read that the breaks sum back to the day.

    The defect was one expression, so the rule it broke is now a module a test
    can execute rather than an expression inside JSX. Driven here on the three
    credits the route actually served.
    """
    measured = node_break_model("""
      const day = [
        { break_id: 'a', is_gold: false, projected_revenue: 100.4 },
        { break_id: 'b', is_gold: true, projected_revenue: 10711.71 },
        { break_id: 'c', is_gold: true, projected_revenue: 10162.61 },
        { break_id: 'd', is_gold: true, projected_revenue: 9613.52 },
      ];
      const gold = m.visibleRows(day, true);
      const every = m.visibleRows(day, false);
      process.stdout.write(JSON.stringify({
        goldCount: gold.length,
        goldSum: m.sumRevenue(gold),
        everyCount: every.length,
        everySum: m.sumRevenue(every),
        share: m.shareOfDay(m.sumRevenue(gold), 1028205.58),
        noDay: m.shareOfDay(500, 0),
        emptySum: m.sumRevenue(m.visibleRows([{ break_id: 'a', is_gold: false, projected_revenue: 100.4 }], true)),
      }));
    """)
    assert measured["goldCount"] == 3
    assert round(measured["goldSum"], 2) == 30487.84, "the sum of the three rows on screen, and nothing else"
    assert round(measured["everySum"], 2) == 30588.24
    assert measured["goldSum"] != measured["everySum"], "a filtered column can never total the unfiltered one"
    assert round(measured["share"], 3) == 2.965, "the share of the day is computed, not asserted in words"
    assert measured["noDay"] is None, "a day with no revenue has an unknown share, never a zero one"
    assert measured["emptySum"] == 0, "an emptied filter totals nothing, because nothing is displayed"

    board = read("plan/break/BreakBoard.jsx")
    assert "const rows = useMemo(() => visibleRows(all, goldOnly), [all, goldOnly]);" in board
    assert "const shown = useMemo(() => sumRevenue(rows), [rows]);" in board
    tfoot = board.split("<tfoot>")[1].split("</tfoot>")[0]
    assert "exactCurrency(shown, locale)" in tfoot, "the totalled figure is the sum of the rows"
    assert "board.totals.revenue" not in tfoot, "the day is not what a row-sum label is allowed to print"
    day_line = tfoot.split("break-foot-day")[1]
    assert "The whole day, every break" in day_line and "כל היום, כל הברייקים" in day_line
    assert "exactCurrency(dayRevenue, locale)" in day_line, "the day is printed, under its own name, when filtered"


def test_the_note_under_the_table_describes_the_column_that_is_on_screen():
    """The sentence moved with the filter, because before it did not.

    It also names the rounding rather than leaving it to be found: the route
    serves each credit to the agora and this board prints whole shekels, so the
    printed column adds by eye to 30,489 while its three credits add to 30,487.84.
    Measured across all thirty planned days: the day and its breaks added up agree
    to the shekel on twenty nine and are one shekel apart on 2024-11-22, and
    hand-adding a printed column of eighty rows can be out by up to five.
    """
    measured = node_break_model("""
      const args = { shownCount: 3, total: 79, portion: '3%', locale: 'he' };
      process.stdout.write(JSON.stringify({
        openHe: m.basisSentence({ ...args, goldOnly: false }),
        goldHe: m.basisSentence({ ...args, goldOnly: true }),
        goldEn: m.basisSentence({ ...args, goldOnly: true, locale: 'en' }),
        shareless: m.basisSentence({ ...args, goldOnly: true, portion: null }),
        roundingHe: m.roundingSentence('he'),
        roundingEn: m.roundingSentence('en'),
      }));
    """)
    assert "הסינון כבוי" in measured["openHe"] and "הכנסת היום" in measured["openHe"]
    assert "3 ברייקים מתוך 79" in measured["goldHe"], "the filtered note counts the subset it describes"
    assert "3%" in measured["goldHe"] and "השורה השנייה" in measured["goldHe"]
    assert "הסינון כבוי" not in measured["goldHe"], "the unfiltered claim must not survive the filter"
    assert "3 breaks of 79" in measured["goldEn"]
    assert "%" not in measured["shareless"], "an uncomputable share is left unsaid, never printed as zero"
    assert "מעוגלת לשקל" in measured["roundingHe"] and "rounded to the whole shekel" in measured["roundingEn"]
    board = read("plan/break/BreakBoard.jsx")
    assert "break-board-rounding" in board, "the footnote is rendered, not only exported"


def test_a_filter_that_empties_the_table_says_what_is_missing_and_where_it_is_made():
    """Zero rows and a figure under them is the defect in its purest form.

    Reproduced before this closed, same day and same viewport with no break
    marked: the ברייקי זהב filter left no rows, no empty state at all, and the
    footer still printed 1,062,670 ILS under סכום על הברייקים האלה. Measured
    after it: the table and its foot are gone, the panel names the mark that is
    missing, names the day board, the G key and the Gold break button as the
    place it is made, prints the served ceiling of 3 a day, keeps the day's own
    80 breaks and 1,062,670 ILS on screen under a label that says the day, and
    offers the way back out of the filter.
    """
    board = read("plan/break/BreakBoard.jsx")
    assert "{board && rows.length === 0 && (" in board, "no rows means no table and no foot"
    assert "{board && rows.length > 0 && (" in board
    empty = board.split("export function EmptyBoard")[1]
    assert "No break in this day is marked gold" in empty and "אין ביום הזה ברייק שמסומן כברייק זהב" in empty
    assert "press G" in empty and "והקישו G" in empty, "the empty state names the act, not only the absence"
    assert "gold.max_per_day" in empty, "the ceiling is the served one"
    assert "gold.enabled === false" in empty, "gold switched off is a different missing thing and says so"
    assert "board.totals.breaks" in empty and "board.totals.revenue" in empty
    assert "The day itself holds" in empty and "היום עצמו מחזיק" in empty, "the day figure carries a day label"
    assert "board.basis.channel" in empty and "board.basis.day" in empty, "and its own scope"
    assert "Show every break in the day" in empty and "הצגת כל הברייקים ביום" in empty


def test_the_verdict_and_the_four_acts_are_the_first_thing_in_the_panel():
    """Save must not need a scroll after an edit, and it did.

    Measured at 1440 x 823 on רשת 13 / 2024-11-01, one ArrowRight on the first
    break so exactly one edit is pending: the panel opened at 714 px, the money at
    727 px, the compliance verdict at 839 px and all four buttons at 917 px, in an
    823 px viewport. Both the verdict and every control were under the fold, on
    the row a scheduler reaches for after every single edit.

    After it, same viewport and same keystroke: the commit row at 727 px, the four
    buttons at 727 px to 755 px, the verdict at 732 px to 750 px and the four money
    tiles at 763 px to 821 px, all on screen. Driven further with one break
    lengthened until a check fails: the buttons and the verdict do not move at all,
    727 px and 732 px, because what grows now sits below them.

    Re-measured in the narrowest state the surface has, English with the assistant
    dock open, which leaves the panel 730 px wide: the verdict is 169 px and the
    four buttons 513 px, so the row holds one line, and the panel opens at 712 px
    with the acts at 721 px to 749 px, the verdict at 726 px to 744 px and the
    money at 757 px to 815 px, every one of them above 823 px. At the old button
    padding the same row came to 726 px against 704 px of panel, wrapped, and put
    the money tiles at 845 px, which is why the padding below is asserted with the
    rest of the geometry.
    """
    readout = read("plan/day/DayBoardReadout.jsx")
    head = readout.split('<div className="day-readout-head">')[1].rsplit("{stranded}", 1)[0]
    assert "day-verdict is-ok" in head and "day-verdict is-bad" in head, "the verdict is in the commit row"
    for control in ("onUndo", "onDiscard", "onCheck", "onSave"):
        assert control in head, f"{control} moved out of the row that must stay on screen"
    body = readout.rsplit("{stranded}", 1)[1]
    assert "day-readout-actions" not in body, "the acts are rendered once, and at the top"
    assert '<div className="day-readout-figures">' in body, "the figures follow the row, not precede it"
    assert readout.index('className="day-readout-head"') < readout.index('className="day-readout-figures"')
    assert "{violations.length > 0 && (" in readout, "the detail that grows is below the acts"
    css = (SRC / "plan" / "day" / "day-readout.css").read_text(encoding="utf-8")
    block = css.split(".day-readout-head {")[1].split("}")[0]
    assert "justify-content: space-between;" in block
    assert "flex-wrap: wrap;" in block, "it wraps rather than pushing a control off a narrow window"
    assert "margin-inline-start: auto;" in css.split(".day-readout-head .day-readout-actions {")[1].split("}")[0]
    action = css.split("\n.day-action {")[1].split("}")[0]
    assert "padding: var(--space-1) var(--space-2);" in action, "a fatter button wraps the row and drops the money below the fold"


def test_the_replan_review_button_counts_one_change_in_hebrew_as_one():
    """The server action is a reviewed re-plan, and its singular still reads naturally."""
    readout = read("plan/day/DayBoardReadout.jsx")
    assert "editCount === 1 ? 'בדיקת שינוי אחד ותכנון מחדש'" in readout
    assert 'id="day-board-server-replan"' in readout


def test_a_zero_gap_is_positive_zero_and_never_prints_as_minus_zero():
    measured = node_board_model("""
      const gap = m.committedGap(
        { committed: { revenue: 100, breaks: 2 } },
        { revenue: 99.999, breaks: 2 },
      );
      process.stdout.write(JSON.stringify({ gap, negativeZero: Object.is(gap.revenueGap, -0) }));
    """)
    assert measured["gap"]["revenueGap"] == 0
    assert measured["negativeZero"] is False


def test_gold_cannot_replan_away_a_pending_placement_edit():
    writes = read("plan/day/day-board-writes.js")
    board = read("plan/day/DayBoard.jsx")
    assert "if (pendingEditCount > 0)" in writes
    assert "Save or discard the pending placement changes" in writes
    assert "edited.length !== Object.keys(edits).length" in writes
    assert "No placement was saved" in writes
    assert "pendingEditCount: Object.keys(edits).length" in board


def test_no_money_figure_on_this_surface_renders_without_its_scope():
    readout = read("plan/day/DayBoardReadout.jsx")
    assert "day-figure-scope" in readout
    assert readout.count("day-figure-scope") >= 4, "every figure tile prints its own scope"
    board = read("plan/break/BreakBoard.jsx")
    assert "board.basis.channel" in board and "board.basis.day" in board


def test_every_display_string_is_one_source_line():
    """No display string is hard wrapped across lines, on any file this piece owns."""
    offenders = []
    for path in sorted(list((SRC / "plan" / "day").glob("*.js*")) + list((SRC / "plan" / "break").glob("*.js*"))):
        lines = path.read_text(encoding="utf-8").splitlines()
        for index, line in enumerate(lines):
            stripped = line.strip()
            # A display string that opens on one line and does not close on it.
            if re.match(r"^'[^']*$", stripped) or re.match(r'^"[^"]*$', stripped):
                offenders.append(f"{path.name}:{index + 1}")
    assert not offenders, offenders
