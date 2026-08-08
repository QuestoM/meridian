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
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
SRC = APP / "src"
BOARD_MODEL_JS = SRC / "plan" / "day" / "day-board-model.js"
CHIP = SRC / "plan" / "day" / "DayBoardChip.jsx"

# Isolation has one home, tv-break-dashboard/src/shell/bidi.jsx. A figure inside
# the chip paints as an inline run carrying bidi-figure and no dir attribute,
# and that is a correction rather than a rename: a dir attribute fixes the run's
# internal order, which is wanted, and also re-anchors that element's own
# alignment, which is the defect. Do not put a dir back on a printed run here.
FIGURE_CLASS = "bidi-figure"

# One break, at the two scales the ladder is measured at. 1704 seconds is
# 00:28:24, the clock the critic measured, and 120 seconds is every break on the
# day the defect was reproduced on.
CHIP_CLOCK = "00:28:24"
CHIP_LENGTH = "120s"
CHIP_START_SECONDS = 1704
CHIP_PROGRAMME = "חדשות הערב"


def read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


# One node run: bundle the shipped chip where it lives so its own imports resolve
# as they do in the browser bundle, render it with React at each width, and
# report the markup a person would see. The same device the P4 render files use.
CHIP_RENDER = """
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';

const [entry, outDir, casesFile, outFile] = process.argv.slice(2);
const require_ = createRequire('APP_PACKAGE');
const MAP = {};
for (const bare of ['react', 'react/jsx-runtime', 'react-dom/server', 'lucide-react', 'rolldown']) {
  MAP[bare] = pathToFileURL(require_.resolve(bare)).href;
}
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) {
      return { url: MAP[specifier], shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

const { build } = await import('rolldown');
await build({
  input: entry,
  external: ['react', 'react-dom', 'react/jsx-runtime', 'lucide-react'],
  output: { dir: outDir, format: 'esm', entryFileNames: 'surface.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  logLevel: 'silent',
  plugins: [{
    name: 'chip-stylesheet',
    // The bundler the product builds with refuses to bundle css, and the
    // stylesheet is asserted separately below, so the import resolves to an
    // empty module under a virtual id that carries no css extension.
    resolveId(source) {
      return source.endsWith('.css') ? { id: `\\0stylesheet-${source}.mjs` } : null;
    },
    load(id) {
      return id.startsWith('\\0stylesheet-') ? 'export default {};' : null;
    },
  }],
});

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const surface = await import(pathToFileURL(`${outDir}/surface.mjs`).href);
const cases = JSON.parse(fs.readFileSync(casesFile, 'utf8'));

const noop = () => {};
const rendered = {};
for (const [name, props] of Object.entries(cases)) {
  rendered[name] = renderToStaticMarkup(React.createElement(surface.DayBoardChip, {
    locale: 'he',
    style: {},
    selected: true,
    edited: false,
    saved: false,
    onSelect: noop,
    onMovePointerDown: noop,
    onResizePointerDown: noop,
    onKeyDown: noop,
    onOpen: noop,
    ...props,
  }));
}
fs.writeFileSync(outFile, JSON.stringify(rendered), 'utf8');
"""


def _chip_node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped chip cannot be rendered here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True, text=True, check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so react cannot be resolved from the app")
    if not (APP / "node_modules" / "react-dom").is_dir():
        pytest.skip("the dashboard's node_modules is not installed, so nothing can be rendered")
    if not (APP / "node_modules" / "rolldown").is_dir():
        pytest.skip("the bundler the product builds with is not installed")
    return found


def render_chip(tmp_path: Path, widths: dict) -> dict:
    """Render the shipped chip at each width and return the markup.

    The number a person reads off a chip is produced by the component, so it is
    read off the component's own output rather than off its source. A source
    assertion describes how the rule was written and breaks on every correct
    refactor; this one describes what is on screen.
    """
    node = _chip_node()
    work = tmp_path / "chip"
    work.mkdir(parents=True, exist_ok=True)
    entry = work / "entry.mjs"
    entry.write_text(f"export {{ default as DayBoardChip }} from '{CHIP.as_posix()}';\n", encoding="utf-8")
    script = work / "render.mjs"
    script.write_text(CHIP_RENDER.replace("APP_PACKAGE", (APP / "package.json").as_posix()), encoding="utf-8")
    cases = work / "cases.json"
    cases.write_text(
        json.dumps(
            {
                name: {
                    "item": {
                        "break_id": "BRK_0003", "ordinal": 3, "breaks_in_segment": 5,
                        "programme": CHIP_PROGRAMME,
                    },
                    "live": {"durationSeconds": 120, "isGold": False},
                    "startSeconds": CHIP_START_SECONDS,
                    "widthPx": width,
                }
                for name, width in widths.items()
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    out = work / "out.json"
    result = subprocess.run(
        # shell/bidi and shell/dates are real shell primitives the shipped chip
        # imports; this loader hook resolves them to the real modules.
        [node, "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         str(script), str(entry), str(work / "bundle"), str(cases), str(out)],
        capture_output=True, text=True, check=False, cwd=str(work),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


def body_of(markup: str) -> str:
    """What the chip itself prints, without the badge drawn beside it."""
    return markup.split('<span class="day-chip-body">')[1].split('<span class="day-chip-readout"')[0]


def badge_of(markup: str) -> str:
    """The badge the numbers a chip cannot hold are drawn in."""
    return markup.split('<span class="day-chip-readout"')[1].split('<i class="day-chip-resize"')[0]


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
        # shell/bidi and shell/dates are real shell primitives the shipped module
        # imports; this loader hook resolves them to the real compiled files.
        ["node", "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         "--input-type=module", "-e", script],
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
        # shell/bidi and shell/dates are real shell primitives the shipped module
        # imports; this loader hook resolves them to the real compiled files.
        ["node", "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_a_chip_prints_no_number_it_would_have_to_cut(tmp_path):
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

    # And the rule, on the rendered chip rather than in its source. The source
    # form of this said which JSX line carried which class, so it broke the day
    # the figures moved onto the shell primitive while the product stayed right.
    # Rendered, the three rungs of the ladder are visible as what is on screen.
    chip = read("plan/day/DayBoardChip.jsx")
    assert "const fits = chipLabels(widthPx, clock, lengthText);" in chip
    painted = render_chip(tmp_path, {"opening": 12, "lengthOnly": 33, "maximum": 60})
    opening = body_of(painted["opening"])
    assert CHIP_CLOCK not in opening and CHIP_LENGTH not in opening, (
        f"the 12 px chip the critic measured prints a number it must cut: {opening!r}"
    )
    length_only = body_of(painted["lengthOnly"])
    assert CHIP_LENGTH in length_only and CHIP_CLOCK not in length_only
    widest = body_of(painted["maximum"])
    assert CHIP_CLOCK in widest and CHIP_LENGTH in widest, "at the maximum scale both are inside the chip again"
    # Isolation moved to the shell primitive, so a printed figure carries the
    # class and no dir. A dir on this inline run would re-anchor it and the chip
    # would stop lining up with the track it sits on.
    assert f'<span class="{FIGURE_CLASS} day-chip-clock">{CHIP_CLOCK}</span>' in widest
    assert f'<span class="{FIGURE_CLASS} day-chip-length">{CHIP_LENGTH}</span>' in widest
    for tag in re.findall(r"<[a-z]+[^>]*>", painted["maximum"]):
        assert "dir=" not in tag, f"{tag} re-anchors its own alignment on a right-to-left board"

    board = read("plan/day/DayBoard.jsx")
    assert "widthPx={parseFloat(geometry.width)}" in board, "the width tested is the width the chip is drawn at"
    assert "const geometry = positionStyle(" in board


def test_the_numbers_a_chip_cannot_hold_are_drawn_beside_it_and_never_lost(tmp_path):
    """The drawing tool's device: dimensions in a badge, never inside the object.

    Measured in Chrome on רשת 13 / 2024-11-01 at 1440 x 823, one break selected
    and the pointer on its neighbour: two badges on an 80 chip board, 92 px wide
    against 12 px chips, centred on their chip to the pixel, both inside the
    track, and neither clipped: the clock wants 51 px in a 51 px box and the
    length 27 in 27. Driven again after a real pointer drag, the badge read
    01:13:55 while the typed field read 01:13:55, so it is the live readout of
    the gesture and not a static caption.
    """
    # Read off the rendered chip, not off its source. The badge is drawn at every
    # scale, including the one where the chip itself can print nothing, which is
    # the whole point of it: the numbers are moved, never lost.
    painted = render_chip(tmp_path, {"opening": 12, "maximum": 60})
    for name in ("opening", "maximum"):
        badge = badge_of(painted[name])
        assert f'<span class="day-chip-readout-clock">{CHIP_CLOCK}</span>' in badge
        assert f'<span class="day-chip-readout-length">{CHIP_LENGTH}</span>' in badge
        assert 'aria-hidden="true"' in painted[name].split('<span class="day-chip-readout"')[1][:40], (
            "the badge repeats what the accessible name already carries, so it is hidden from it"
        )
        # Isolation moved to the shell primitive and the badge states no
        # direction of its own. A dir here would set the badge's base direction
        # and re-anchor it, which on a Hebrew board puts the two numbers in the
        # order an English reader expects rather than the order this reader does.
        assert "dir=" not in painted[name].split('<span class="day-chip-readout"')[1].split(">")[0]
        # The numbers a person can no longer see printed are still spoken.
        name_attr = re.search(r'aria-label="([^"]*)"', painted[name]).group(1)
        assert CHIP_CLOCK in name_attr and "120 שניות" in name_attr, (
            f"the accessible name lost the numbers the chip stopped printing: {name_attr!r}"
        )

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


