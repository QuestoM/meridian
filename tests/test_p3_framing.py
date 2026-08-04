"""The scale the board draws at: what a chip can print, and the two framings.

A drawing tool's own mechanics for a canvas you can zoom: an object shows its own
exact numbers, and one press frames the whole drawing or the selection. Both were
measured false at the scale this board opens at, and both are held here against
the shipped modules rather than against a screenshot.

Split from ``test_p3_direct_manipulation.py`` under the 450-line law. Nothing moved
in behaviour: every assertion below is the one that was there, at its new address.
"""

from __future__ import annotations

import json
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
    python re-implementation would only prove that two pieces of test code agree
    with each other, which is what let the defects below ship.
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


