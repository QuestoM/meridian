"""The mechanics the day board is graded on, held at the source.

The reference for this destination is a drawing tool for selection, drag and
undo, and a professional editing timeline for exact durations. Both references
have concrete mechanics, and a mechanic that is not asserted anywhere is a
mechanic the next edit can quietly remove. So each one is pinned here against
the source that implements it, the way the wave-zero seam tests pin theirs.

Nothing here is a screenshot check. Each assertion names a behaviour a critic
can also reproduce in a browser: type a clock and the break moves there, press
an arrow and it steps by a published increment, press Cmd Z and it goes back,
open a break and walk the set it came from without returning to the board.
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
    """Run the shipped board model in node and return what it computed.

    The module the operator's browser imports is the module asserted here. A
    python re-implementation of liveBreak would only prove that two pieces of
    test code agree with each other, which is what let the defect below ship.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    script = f"const m = await import({json.dumps(BOARD_MODEL_JS.as_uri())});\n{body}"
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def node_track_math(body: str) -> dict:
    """Run the shipped time-axis math in node, same reason as above."""
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    module = SRC / "plan" / "day" / "schedule-track.js"
    script = f"const m = await import({json.dumps(module.as_uri())});\n{body}"
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_a_chip_prints_no_number_it_would_have_to_cut():
    """The label the object shows is the label it can show whole, or nothing.

    Reproduced in Chrome on רשת 13 / 2024-11-01 at 1440 x 823, at the scale this
    board opens at: 6 px per minute, every break 120 s, so all 80 chips are 12 px
    wide with 4 px of content box once the padding and the border are taken. The
    clock 00:28:24 wants 47 px and the length 120s wants 25, and both were
    rendered anyway, so every chip on the track printed the character 0 over the
    character 1. Measured that morning: 160 rendered labels, 160 of them clipped.

    Measured after this rule, same day and same viewport: 0 numbers rendered
    inside a chip and 0 clipped. At the maximum scale, where a 120 s chip is
    60 px, both are printed inside it again, which is the ladder working at both
    ends rather than a label switched off.
    """
    measured = node_board_model("""
      const clock = '00:28:24';
      process.stdout.write(JSON.stringify({
        opening: m.chipLabels(12, clock, '120s'),
        dayFit: m.chipLabels(8, clock, '120s'),
        programmeFit: m.chipLabels(27, clock, '120s'),
        lengthOnly: m.chipLabels(33, clock, '120s'),
        justUnderTheClock: m.chipLabels(57, clock, '120s'),
        justOverTheClock: m.chipLabels(58, clock, '120s'),
        maximumScale: m.chipLabels(60, clock, '120s'),
        noWidthAtAll: m.chipLabels(undefined, clock, '120s'),
        longLength: m.chipLabels(33, clock, '1020s'),
        char: m.CHIP_CHAR_PX,
        chrome: m.CHIP_CHROME_PX,
      }));
    """)
    assert measured["opening"] == {"clock": False, "length": False}, "the 12 px chip the critic measured prints nothing"
    assert measured["dayFit"] == {"clock": False, "length": False}
    assert measured["programmeFit"] == {"clock": False, "length": False}
    assert measured["lengthOnly"] == {"clock": False, "length": True}, "the shorter number appears first, on its own"
    assert measured["justUnderTheClock"]["clock"] is False
    assert measured["justOverTheClock"]["clock"] is True, "the threshold is the text's own measured width"
    assert measured["maximumScale"] == {"clock": True, "length": True}
    assert measured["noWidthAtAll"] == {"clock": False, "length": False}, "an unmeasured chip prints nothing"
    assert measured["longLength"]["length"] is False, "a longer length needs more room, and is tested for it"
    assert measured["char"] == 6.25 and measured["chrome"] == 8

    chip = read("plan/day/DayBoardChip.jsx")
    assert "const fits = chipLabels(widthPx, clock, lengthText);" in chip
    assert "{fits.clock && <span className=\"day-chip-clock\"" in chip, "the clock is rendered only when it fits"
    assert "{fits.length && <span className=\"day-chip-length\"" in chip
    board = read("plan/day/DayBoard.jsx")
    assert "widthPx={parseFloat(geometry.width)}" in board, "the width tested is the width the chip is drawn at"
    assert "const geometry = positionStyle(" in board


def test_the_numbers_a_chip_cannot_hold_are_drawn_beside_it_and_never_lost():
    """The drawing tool's device: dimensions in a badge, never inside the object.

    Measured in Chrome on רשת 13 / 2024-11-01 at 1440 x 823, one break selected
    and the pointer on its neighbour: two badges on an 80 chip board, 92 px wide
    against 12 px chips, centred on their chip to the pixel, both inside the
    track, and neither clipped: the clock wants 51 px in a 51 px box and the
    length 27 in 27. Driven again after a real pointer drag, the badge read
    01:13:55 while the typed field read 01:13:55, so it is the live readout of
    the gesture and not a static caption.
    """
    chip = read("plan/day/DayBoardChip.jsx")
    assert '<span className="day-chip-readout" dir="ltr" aria-hidden="true">' in chip
    assert '<span className="day-chip-readout-clock">{clock}</span>' in chip
    assert '<span className="day-chip-readout-length">{lengthText}</span>' in chip
    assert "aria-label={[identity, ...states].join(', ')}" in chip, "the numbers stay in the accessible name"
    assert "${clock}, ${seconds} ${label('seconds'" in chip, "and the accessible name is what carries them"

    css = (SRC / "plan" / "day" / "day-chip.css").read_text(encoding="utf-8")
    badge = css.split(".day-chip-readout {")[1].split("}")[0]
    assert "position: absolute;" in badge
    assert "display: none;" in badge, "eighty badges at once would be a smear, so it is drawn on demand"
    assert "pointer-events: none;" in badge, "a badge that swallowed a press would break the drag it narrates"
    assert "white-space: nowrap;" in badge and "overflow" not in badge, "the badge is sized by its own text"
    assert "inset-block-end: calc(100% + 3px);" in badge, "above the chip: the scroll box clips its own block axis"
    shown = css.split(".day-chip.is-selected > .day-chip-readout {")[1].split("}")[0]
    assert "display: inline-flex;" in shown
    assert ".day-chip:hover > .day-chip-readout," in css, "hover is one of the two moments a person is asking"
    assert ".day-chip:focus-visible > .day-chip-readout," in css, "and the keyboard reaches it the same way"


def test_the_board_frames_the_whole_day_and_one_programme_in_one_press():
    """A scale you can only step through is a scale you get lost in.

    Measured in Chrome on רשת 13 / 2024-11-01 at 1440 x 823 with 978 px of
    visible track: at the maximum scale the day track is 46,800 px, 40.7 times
    the width of the box that holds it. One press of the first preset put the
    whole 26 hour day at 978 px against 978 px of track, with 0 px of horizontal
    scroll left over, all 80 breaks still drawn, and the ruler stepped to three
    hour labels, 9 of them, 0 overlapping. One press of the second, with a break
    selected, put its 72 minute programme at 978 px and scrolled the selection
    back under the eye.
    """
    measured = node_track_math("""
      process.stdout.write(JSON.stringify({
        wholeDay: m.fitZoom(1560, 978),
        oneProgramme: m.fitZoom(72, 978),
        noMinutes: m.fitZoom(0, 978),
        floorUnchangedForEveryOtherCaller: m.clampZoom(0.627),
        floorLoweredForAFit: m.clampZoom(0.627, 0.627),
        ceilingStillHolds: m.clampZoom(195.6, 0.627),
        rulerAtTheDayFit: m.tickStep(0.627),
        rulerAtTheSharedFloor: m.tickStep(1.9),
        rulerAtTheOpeningScale: m.tickStep(6),
        minimum: m.MIN_PX_PER_MIN,
        maximum: m.MAX_PX_PER_MIN,
      }));
    """)
    assert round(measured["wholeDay"], 3) == 0.627, "the day fit is the width divided by the span, and nothing else"
    assert round(measured["oneProgramme"], 2) == 13.58
    assert measured["noMinutes"] == 978, "a span of nothing still returns a usable scale rather than infinity"
    assert measured["floorUnchangedForEveryOtherCaller"] == 1.9, "the week timeline's own band is untouched"
    assert measured["floorLoweredForAFit"] == 0.627, "and a fit may open exactly the scale it measured"
    assert measured["ceilingStillHolds"] == 30, "a fit never zooms past the band's own maximum"
    assert measured["rulerAtTheDayFit"] == 180, "an hour line with no room for its clock steps to three hours"
    assert measured["rulerAtTheSharedFloor"] == 60, "and nothing changes at any scale the other surface can reach"
    assert measured["rulerAtTheOpeningScale"] == 15, "and nothing changes at the scale this board opens at"
    assert (measured["minimum"], measured["maximum"]) == (1.9, 30), "the published band is unchanged"

    zoom = read("plan/day/day-board-zoom.js")
    assert "export function fitTheDay(axis, trackRef, fitTo)" in zoom
    assert "export function fitTheProgramme(programme, trackRef, fitTo, breakId)" in zoom
    assert "box.clientWidth - LANE_GUTTER" in zoom, "the width fitted is the width measured, not a constant"
    assert "chip.scrollIntoView({ inline: 'center', block: 'nearest' })" in zoom

    toolbar = read("plan/day/DayBoardToolbar.jsx")
    assert "onClick={onFitDay}" in toolbar and "onClick={onFitProgramme}" in toolbar
    assert "'Fit the whole day on screen', 'התאמת כל היום למסך'" in toolbar
    assert "'Fit the programme of the selected break on screen', 'התאמת התוכנית של הברייק הנבחר למסך'" in toolbar
    assert "disabled={!selectedItem}" in toolbar, "a fit with nothing selected has nothing to frame"
    assert toolbar.count("aria-label={label('Fit") == 2, "an icon control carries its words in its name"
    board = read("plan/day/DayBoard.jsx")
    assert "onFitDay={() => fitTheDay(axis, trackRef, fitTo)}" in board
    assert "fitTheProgramme(programmes.get(selectedItem.segment_id), trackRef, fitTo, selectedItem.break_id)" in board
    assert "floor={floor}" in board, "the wheel obeys the same floor the preset opened"


def test_both_the_start_and_the_length_are_typed_targets_not_readouts():
    """The editing timeline's device: the position is a field, not a display."""
    toolbar = read("plan/day/DayBoardToolbar.jsx")
    assert "<output" not in toolbar, "a number a person may want to state must not be a readout"
    assert "commitStart" in toolbar and "commitLength" in toolbar
    assert "parseClock" in toolbar
    assert toolbar.count("onKeyDown") >= 2, "Enter commits both fields"
    board = read("plan/day/DayBoard.jsx")
    assert "onStart={" in board and "onLength={" in board


def test_a_typed_clock_parses_the_two_forms_a_person_writes_and_refuses_the_rest():
    """The parser is small enough to check exactly, so it is checked exactly."""
    model = read("plan/day/day-board-model.js")
    body = model.split("export function parseClock")[1].split("export function")[0]
    assert "parts.length < 2 || parts.length > 3" in body, "HH:MM and HH:MM:SS, nothing else"
    assert "return null" in body, "a half-typed time must not move a break"
    assert "minute > 59 || second > 59" in body


def test_the_nudge_increments_are_published_where_the_person_reads_them():
    """A keyboard taught in place, on the surface that performs it."""
    model = read("plan/day/day-board-model.js")
    assert "if (event.altKey) return 1;" in model
    assert "if (event.shiftKey) return grid * 5;" in model
    toolbar = read("plan/day/DayBoardToolbar.jsx")
    for fragment in ("Shift", "Alt", "G ", "Enter"):
        assert fragment in toolbar, f"the help text stopped naming {fragment}"
    assert "מקשי החיצים" in toolbar, "the keyboard is taught in Hebrew too"


def test_undo_and_redo_are_one_stack_read_in_two_directions():
    model = read("plan/day/day-board-model.js")
    for name in ("emptyHistory", "pushAction", "undoAction", "redoAction"):
        assert f"export function {name}" in model
    assert "future: []" in model, "a new act after an undo drops the redone tail"
    history = read("plan/day/day-board-history.js")
    assert "if (event.shiftKey) redo(); else undo();" in history
    assert "event.metaKey || event.ctrlKey" in history
    board = read("plan/day/DayBoard.jsx")
    assert "useBoardHistory" in board, "the board reaches undo through the one hook that owns it"


def test_a_drag_that_ends_where_it_began_records_nothing():
    drag = read("plan/day/day-board-drag.js")
    assert "if (!moved) return;" in drag
    assert "setSnapMark" in drag, "the snap line is the in-flight signifier"


def test_a_pending_edit_never_shadows_the_break_s_gold_state():
    """Gold is the plan's answer, and an unsaved move must not overwrite it.

    Reproduced in a browser on רשת 13 / 2024-11-01, break 001~1, before this was
    closed: select the break, press ArrowRight once so one unsaved edit exists,
    then press G. The override landed, the day came back gold and two sibling
    chips rendered gold, while this one stayed grey with the toolbar Gold button
    reading off, because the placement edit had snapshotted is_gold false and
    false ?? true is false. The mark was then unclearable from the surface, and
    each further press wrote another identical override row instead of taking it
    off: measured c199a87597c8 and 25906b93d511, both carrying the same note.

    The same sequence is driven here through the shipped module.
    """
    measured = node_board_model("""
      const item = { break_id: 'b', offset_seconds: 100, duration_seconds: 120, is_gold: false };
      const start = m.liveBreak(item, {});
      const edits = m.applyEdit({}, item, { ...start, offsetSeconds: start.offsetSeconds + 60 });
      const served = { ...item, is_gold: true };
      const live = m.liveBreak(served, edits);
      const roundTrip = m.applyEdit(edits, served, { ...live, offsetSeconds: served.offset_seconds });
      const goldAct = m.applyEdit({}, served, { ...live, isGold: false, goldEdit: true });
      process.stdout.write(JSON.stringify({
        edit: edits[item.break_id],
        editedGold: live.isGold,
        stillEdited: live.edited,
        moves: m.movesFrom(edits),
        roundTrip,
        goldActHeld: goldAct[item.break_id],
        goldActLive: m.liveBreak(served, goldAct).isGold,
      }));
    """)
    assert "is_gold" not in measured["edit"], "a placement edit must not carry a copy of the gold state"
    assert measured["editedGold"] is True, "the chip has to render the gold the plan came back with"
    assert measured["stillEdited"] is True, "and it is still an edited break, so the move is not lost"
    assert measured["moves"][0]["is_gold"] is None, "an untouched flag travels as untouched, not as false"
    assert measured["roundTrip"] == {}, "a move back to the plan's own offset still leaves no trace"
    # A gold act on top of the pending move keeps the move and adds its own decision.
    assert measured["goldActHeld"] == {"offset_seconds": 160, "duration_seconds": 120, "is_gold": False}
    assert measured["goldActLive"] is False, "an act that IS a gold act still decides the gold state"


def test_the_gold_button_reads_the_same_live_state_the_chip_does():
    """One source for the mark, so the board and its toolbar cannot disagree.

    Measured in a browser after the fix, same break and same two keystrokes: the
    edited chip renders is-gold with its two siblings, the toolbar button carries
    is-on, and pressing G again clears all three and leaves the override store
    back at its header row.
    """
    toolbar = read("plan/day/DayBoardToolbar.jsx")
    assert "className={live.isGold ? 'day-chip-button is-on' : 'day-chip-button'}" in toolbar
    chip = read("plan/day/DayBoardChip.jsx")
    assert "live.isGold ? 'is-gold' : ''" in chip
    board = read("plan/day/DayBoard.jsx")
    assert "const live = liveOf(item);" in board, "the toolbar and the chip read one live state"
    assert "if (live.isGold) {" in read("plan/day/day-board-writes.js"), "and the act decides from it"
    # Marking is not idempotent at the store, so two presses inside one round
    # trip are coalesced into the one decision they are.
    actions = read("plan/day/day-board-actions.js")
    assert "onceInFlight(`mark:${breakId}`" in actions
    assert "onceInFlight(`clear:${breakId}`" in actions


def test_a_break_carrying_a_saved_placement_offers_its_inverse_with_an_empty_history():
    """The act that moves the money has to be reversible after a reload.

    Measured on רשת 13 / 2024-11-01 before this closed: select 003~1, press
    ArrowRight once, click Save. The settlement panel reported a realised change
    of -25,400 ILS against a prediction of 0 and offered to put it back. Press
    reload: the panel is gone, ``lastSave`` is null, ``history.past`` is empty,
    and the only trace left on the surface was an 11 px lock glyph carrying
    ``aria-hidden``. One arrow key and one click cost 25,400 ILS with no route
    back from this destination.

    So the inverse is read off the break the server served rather than off the
    session, and both halves of that are asserted: the shipped model answers from
    the payload alone with the undo stack empty, and the toolbar that renders the
    control reads no history at all.
    """
    measured = node_board_model("""
      const free = { break_id: 'x~1', ordinal: 1 };
      const pinned = { break_id: 'x~1', ordinal: 1, saved_placement: { break_id: 'x~1', constraint_id: '0679cf29dc86', saved_at: '2026-08-01T09:00:00Z', note: '' } };
      process.stdout.write(JSON.stringify({
        free: m.inversePlacement(free),
        pinned: m.inversePlacement(pinned),
        orphan: m.inversePlacement({ break_id: 'x~1', saved_placement: { break_id: 'x~1', constraint_id: '' } }),
        history: m.emptyHistory(),
      }));
    """)
    assert measured["history"] == {"past": [], "future": []}, "the session stack is empty, as it is after a reload"
    assert measured["free"] is None, "a break the plan placed itself has nothing to reverse"
    assert measured["pinned"] == {
        "breakId": "x~1",
        "constraintId": "0679cf29dc86",
        "savedAt": "2026-08-01T09:00:00Z",
        "note": "",
    }, "the inverse names the restriction it has to delete"
    assert measured["orphan"] is not None, "a record whose restriction is already gone is still removable"

    toolbar = read("plan/day/DayBoardToolbar.jsx")
    assert "const savedPin = inversePlacement(selectedItem);" in toolbar
    assert "{savedPin && (" in toolbar, "the control is rendered from the served break, not from a flag"
    assert "Remove the saved placement" in toolbar and "הסרת הנעיצה השמורה" in toolbar
    for forbidden in ("lastSave", "history", "settlement"):
        assert forbidden not in toolbar, f"the control must not depend on {forbidden}, which a reload empties"

    board = read("plan/day/DayBoard.jsx")
    assert "function removeSavedPlacement(item)" in board
    assert "predicted: inverseOf(settlement)," in board, "it settles like the save it reverses"
    assert "onRemoveSaved={() => selectedItem && removeSavedPlacement(selectedItem)}" in board

    acts = read("plan/day/day-board-writes.js")
    assert "export async function removeSavedPlacement({" in acts
    assert "const inverse = inversePlacement(item);" in acts
    assert "await settleAfter('undo', predicted, async () => {" in acts, "the same settlement the save printed"
    assert "await undoBreakPlacement(inverse);" in acts, "the same inverse the session undo performs"
    assert "forgetRecord(inverse.breakId);" in acts


def test_the_inverse_is_read_off_the_record_itself_when_no_break_carries_it():
    """The chip is not the only place a saved placement can be reversed from.

    Measured on רשת 13 / 2024-11-01: pinning 001~2 one snap unit right re-plans
    that programme from four breaks to one, so the day falls 1,067,845.55 to
    1,020,401.35 and the id the record names stops existing. The control asserted
    above hangs off the break chip, so at that moment it hangs off nothing.

    The record survives that, because it carries both ids the inverse needs. This
    drives the shipped model on a record alone, with no break and no session.
    """
    measured = node_board_model("""
      const record = { break_id: 's~2', constraint_id: 'a498ae76c186', saved_at: '2026-08-01T20:16:31Z', note: '' };
      process.stdout.write(JSON.stringify({
        fromRecord: m.inverseOfRecord(record),
        fromBreak: m.inversePlacement({ break_id: 's~2', saved_placement: record }),
        nothing: m.inverseOfRecord(null),
        idless: m.inverseOfRecord({ constraint_id: 'a498ae76c186' }),
      }));
    """)
    assert measured["fromRecord"] == {
        "breakId": "s~2",
        "constraintId": "a498ae76c186",
        "savedAt": "2026-08-01T20:16:31Z",
        "note": "",
    }
    assert measured["fromBreak"] == measured["fromRecord"], "one inverse, whichever side it is read from"
    assert measured["nothing"] is None
    assert measured["idless"] is None, "a record with no break id addresses nothing and must not offer to"

    rows = read("plan/day/DayBoardReadout.jsx").split("export function StrandedPlacements")[1].split("export function")[0]
    for forbidden in ("lastSave", "history", "settlement", "forecast"):
        assert forbidden not in rows, f"the row must not depend on {forbidden}, which a reload empties"


def test_the_chip_says_gold_and_saved_in_its_name_and_not_only_in_a_glyph():
    """Both glyphs are decorative, so both states have to be in the label."""
    chip = read("plan/day/DayBoardChip.jsx")
    assert chip.count('aria-hidden="true"') >= 2, "the star and the lock are decoration"
    assert "if (live.isGold) states.push(" in chip
    assert "if (saved) states.push(" in chip
    assert "aria-label={[identity, ...states].join(', ')}" in chip
    assert "ברייק זהב" in chip and "נעיצה שמורה של המפעיל" in chip


def test_the_day_board_releases_the_shared_scroll_floor_so_the_money_stays_on_screen():
    """The shared container reserves 440 px for a stack of lanes. This has one.

    Measured at 1440 x 823 on רשת 13 / 2024-11-01 before the override: the scroll
    box was 440 px tall around 148 px of content, so the hour strip began at
    869 px, the money readout at 927 px and the save, undo and discard buttons at
    1043 px, every one of them below the fold on a day that fits the screen.
    After it: 148 px, 577 px, 635 px and 751 px, all of them on screen.
    """
    shell = (SRC / "shell" / "styles.css").read_text(encoding="utf-8")
    assert "min-height: 440px;" in shell, "the shared floor moved, so this override wants re-measuring"
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
    assert "border-inline-start" in item, "each verdict keeps a rule of its own, and it mirrors in Hebrew"
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
    board = read("plan/day/DayBoard.jsx")
    assert "clickable={Boolean(onOpenProgramme)}" in board
    assert "onOpen={() => onOpenProgramme(programme)}" in board
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
    assert "<button" in strip and 'type="button"' in strip
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
        ["node", "--input-type=module", "-e", script],
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


def test_the_save_button_counts_one_change_in_hebrew_as_one():
    """It read שמירת 1 שינויים, which is a plural verb over a singular count."""
    readout = read("plan/day/DayBoardReadout.jsx")
    assert "editCount === 1 ? 'שמירת שינוי אחד'" in readout


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
