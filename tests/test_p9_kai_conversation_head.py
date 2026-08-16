"""P9: the conversation head may not cut the strings it prints.

The defect a blind critic measured at every viewport he opened. The head of
Kai's conversation panel carries three lines: what the panel is, what it does
with the conversation, and who is acting. The shared ``panel-head`` from
``styles.css`` is a fixed 50 px row, which is the right shape for a one-line
title, and those three stack to 58 px. With the row fixed and its content
centred, the first and the last line paint outside the box and are cut by it.

Measured in a real browser before the fix, in the dock and on the full page, at
1280, 1500, 1920 and 2560 wide, and at the minimum and the default dock width:
``שיחה`` painted 9 px above the head's top border and ``מבצע: auth-disabled``
6 px below its bottom border. Eight of the twelve configurations cut a string,
and the four that did not were the widest dock, where the two lower strings fit
on one line.

``kai-conversation-head.css`` makes the shared 50 px a floor instead of a
ceiling, so nothing moves where the title already fits on one line and nothing
is cut where it does not. This file measures that in the same headless browser
the bidi tests use, with the shipped strings, the shipped markup and the shipped
stylesheets, at five widths spanning the dock's own clamp. It carries its
control: the same document without that one stylesheet must still cut, so a pass
here can never be vacuous.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
KAI = DASHBOARD / "src" / "kai"
TOKENS = DASHBOARD / "src" / "tokens.css"
INDEX = (DASHBOARD / "src" / "index.jsx").read_text(encoding="utf-8")
# The shared head moved out of the foundation sheet during the CSS ownership
# split. Load the relative global sheets in exactly index.jsx order so this
# browser proof follows future ownership moves without reconstructing the old
# monolith. Package font sheets do not affect the head box contract.
GLOBAL_SHEETS = tuple(
    DASHBOARD / "src" / specifier[2:]
    for specifier in re.findall(r"import '([^']+\.css)';", INDEX)
    if specifier.startswith("./")
)
CONSOLE_SHEET = KAI / "assistant-console.css"
HEAD_SHEET = KAI / "kai-conversation-head.css"
PANEL = (KAI / "AssistantPanel.jsx").read_text(encoding="utf-8")
DOCK = (KAI / "AssistantDock.jsx").read_text(encoding="utf-8")
PROBE = Path(__file__).with_name("test_p9_paint_probe.mjs")

# The dock clamps its own width, so the head has to hold from the narrowest
# dock to the widest. These are the two ends and three widths between them,
# taken as the panel's own width rather than the dock's, which is 24 px wider
# for the body padding on both sides.
WIDTHS = (296, 340, 396, 500, 616)
# The panel width at the dock's default, where the defect was measured in the
# running product.
DEFAULT_WIDTH = 396
# This document is the head alone, with none of the page chrome that narrows it
# in the product, so its text wraps later than the product's does and the third
# line appears further down. This is where it stacks the same three lines the
# running dock stacks at its default width, and where the control reproduces
# the measured defect in full rather than by its first millimetre.
STACKED_WIDTH = 260

# The account name the head prints beside its label. Its length is what the
# measurement depends on, and this is the name the product shows when auth is
# disabled, which is how the defect was measured.
ACTING_USER = "auth-disabled"


def _head_strings() -> list[tuple[str, str]]:
    """The display strings the shipped head prints, read out of the shipped
    source so this file follows the copy rather than repeating it."""
    start = PANEL.index('<div className="panel-head">')
    end = PANEL.index('<div className="asst-thread"', start)
    return re.findall(r"pageText\(locale, '([^']*)', '([^']*)'\)", PANEL[start:end])


def _document(locale: str, width: int, with_fix: bool) -> str:
    strings = _head_strings()
    assert len(strings) >= 3, f"the head's display strings did not parse: {strings}"
    index = 1 if locale == "he" else 0
    title, subtitle, acting = strings[0][index], strings[1][index], strings[2][index]
    clear = strings[3][index] if len(strings) > 3 else ""
    shell = "rtl" if locale == "he" else "ltr"
    sheets = [*GLOBAL_SHEETS, CONSOLE_SHEET] + ([HEAD_SHEET] if with_fix else [])
    links = "\n".join(f'<link rel="stylesheet" href="file://{sheet}">' for sheet in sheets)
    # The acting-user line, byte-for-byte what AssistantPanel.jsx renders: a
    # plain <span class="asst-user"> (no dir="auto") wrapping a <b> (no
    # dir="ltr") around <Name>, which is shell/bidi.jsx's plain
    # <span class="bidi-name">. AssistantPanel.jsx dropped both dir attributes
    # in favour of unicode-bidi: plaintext on .bidi-name, so a control that
    # still hand-adds them is not measuring the shipped markup.
    return f"""<!doctype html><meta charset="utf-8">
{links}
<body style="margin:0"><div dir="{shell}" style="width:{width}px">
<section class="page-panel asst-chat">
<div class="panel-head" id="head">
<div>
<h2>{title}</h2>
<span>{subtitle}</span>
<span class="asst-user">{acting}: <b><span class="bidi-name">{ACTING_USER}</span></b></span>
</div>
<button type="button" class="asst-clear-btn"><span style="width:13px;height:13px;display:inline-block"></span>{clear}</button>
</div>
</section>
</div></body>"""


# Every string in the head, and how far it paints outside the head's own box.
PAINT = """(() => {
  const head = document.getElementById('head');
  const box = head.getBoundingClientRect();
  const out = [];
  for (const node of head.querySelectorAll('h2, span, b')) {
    const r = node.getBoundingClientRect();
    if (r.height === 0) continue;
    out.push({ text: (node.textContent || '').trim().slice(0, 44),
               above: Math.round(box.top - r.top), below: Math.round(r.bottom - box.bottom) });
  }
  return { height: Math.round(box.height), cut: out.filter((p) => p.above > 0 || p.below > 0) };
})()"""


def _paint(locale: str, width: int, with_fix: bool, tmp_path: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the paint cannot be measured")
    name = f"head-{locale}-{width}-{'fixed' if with_fix else 'control'}"
    document = tmp_path / f"{name}.html"
    document.write_text(_document(locale, width, with_fix), encoding="utf-8")
    expression = tmp_path / "paint.js"
    expression.write_text(PAINT, encoding="utf-8")
    # --import wires the shell resolver hook. The probe itself never imports a
    # shell primitive, but the flag costs nothing and keeps every node
    # invocation in this file resolving the same way.
    done = subprocess.run([node, "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
                          str(PROBE), str(document), str(expression)],
                          capture_output=True, text=True, timeout=180)
    if done.returncode == 2 and "no chrome" in done.stderr:
        pytest.skip("no Chrome on this machine, so the paint cannot be measured")
    assert done.returncode == 0, f"the probe failed: {done.stderr[-800:]}"
    return json.loads(done.stdout)


@pytest.mark.parametrize("locale", ["he", "en"])
def test_no_string_in_the_conversation_head_paints_outside_it(locale, tmp_path) -> None:
    """The bar, in both readings and across the dock's whole width clamp. A
    string that paints above or below the head's border is a string the border
    cuts, which is what the operator sees."""
    for width in (STACKED_WIDTH,) + WIDTHS:
        painted = _paint(locale, width, True, tmp_path)
        assert painted["cut"] == [], (
            f"the head cuts a string at {width}px in {locale}: {painted['cut']}")


def test_the_control_cuts_one_string_above_and_one_below(tmp_path) -> None:
    """The measurement can fail. Without the one stylesheet that lifts the
    fixed height, the same head at the width where its three lines stack cuts
    the title above the border and the acting user below it, which is the
    defect exactly as it was measured in the running product."""
    painted = _paint("he", STACKED_WIDTH, False, tmp_path)
    assert painted["height"] == 50, "the control is not the shared fixed row any more"
    assert len(painted["cut"]) >= 2, f"the control stopped cutting: {painted}"
    assert any(part["above"] > 0 for part in painted["cut"]), "nothing painted above the head"
    assert any(part["below"] > 0 for part in painted["cut"]), "nothing painted below the head"


def test_at_the_default_dock_width_the_control_cuts_and_the_fix_does_not(tmp_path) -> None:
    """The two states side by side at one width, so the difference is the one
    stylesheet and nothing else."""
    control = _paint("he", DEFAULT_WIDTH, False, tmp_path)
    fixed = _paint("he", DEFAULT_WIDTH, True, tmp_path)
    assert control["cut"], "the control cuts nothing, so this width proves nothing"
    assert fixed["cut"] == [], f"the head still cuts with the fix in place: {fixed['cut']}"
    assert fixed["height"] > control["height"], "the head did not grow to hold its own lines"


def test_the_fix_is_scoped_to_this_head_and_reaches_every_screen() -> None:
    """It adjusts one head, not the shared row every other panel uses, and it
    is imported where the shell already loads a module on every screen."""
    rule = HEAD_SHEET.read_text(encoding="utf-8")
    selectors = [line.split("{")[0].strip() for line in rule.splitlines()
                 if "{" in line and not line.strip().startswith("/")]
    assert selectors == [".asst-chat > .panel-head"], f"the fix is not scoped: {selectors}"
    assert "min-height: 50px" in rule, "the shared row's height is no longer the floor"
    assert "import './kai-conversation-head.css';" in DOCK
