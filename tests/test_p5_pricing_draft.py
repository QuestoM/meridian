"""P5: discarding a rate-card edit puts the saved figure back in the box.

A blind critic measured this on the shipped surface: the base-price box was bound
with ``defaultValue={state.base.value}`` and a remount key made of the same saved
value, so pressing "ביטול העריכה" cleared the draft, left the key unchanged, and
left the discarded 80 sitting in the box. The same screen then read base 80 in
the card and ``מחיר בסיס 60.00`` in the price tester two columns away, while the
server held 60. Three figures, one rate card, and the one the operator was
looking at was the only one that was not real.

The rule this file pins is that a draft-bound control shows the draft while one
exists and the saved card the moment it does not. Two halves, both measured:

* the shipped helpers, run in node through ``test_p5_draft_probe.mjs``, over the
  exact sequence a person performs on the surface;
* the binding, read out of the shipped JSX, because a correct helper that a box
  does not consult is what the defect was.

Measured in a real browser on the running app after the fix, at
``/?rules=rate_card#Settings``: the box read 60 on load, 80 after the edit, and
60 again after the discard, with the price tester holding 60.00 throughout and
the effect panel closing. That measurement needs an API and a build, so it is
recorded here rather than run here.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RULES = ROOT / "tv-break-dashboard" / "src" / "rules"
SHELL = ROOT / "tv-break-dashboard" / "src" / "shell"
MANAGER = RULES / "PricingManager.jsx"
EVENTS = RULES / "PricingEventsLayer.jsx"
PROBE = Path(__file__).with_name("test_p5_draft_probe.mjs")

# rules-lib.js now reads a calendar day through shell/dates.js, and rules-bidi.js
# gets its isolate from shell/bidi.jsx. Both are new since the probe was written,
# and the probe copies a hand-listed trio of rules/*.js files into its own temp
# tree, so neither specifier resolves and every test behind the probe errors
# before an assertion runs (see test_p4_rollup_tristate.py for the first fix in
# this class, and the commit that shipped it for the shape of the other four).
#
# dates.js is plain JavaScript with no react in it, so the real file is copied
# in verbatim below rather than restated as a stub: a date these tests read is
# formatted exactly the way the product formats it, and a future change to that
# file shows up here instead of going stale under a hand-written double. Its one
# relative import has no extension, which the bundler resolves and node does
# not, so that one line is rewritten to add it; nothing else about the file
# changes.
#
# This file used to carry its own loader hook and its own stub of the bidi
# primitive, run through node's deprecated ``--experimental-loader``. It was the
# last caller of that flag in the repository.
#
# What replaced it is the point. ``tests/js/shell-resolver.mjs`` is ONE hook that
# every browser probe here already uses, and it resolves the shell primitives to
# the REAL modules compiled with the bundler's own transform. This file was
# resolving them to a stub instead: bidi's ``isolate`` was replaced by the
# identity function, on the reasoning that these assertions never read what it
# produces.
#
# That reasoning was sound and the arrangement was still worse, for a reason
# worth keeping. A probe asserting against a fake primitive proves nothing about
# what ships, and the moment the fake and the real one diverge the probe goes on
# passing. There is no reason left to accept that here, because the shared hook
# already solves the problem the stub was working around: bidi.jsx imports react
# and returns JSX, and the hook compiles it and places the copy where a bare
# specifier can still find node_modules.
#
# So this file now runs the same primitives the dashboard runs, through the same
# hook as its forty-odd siblings, and there is no second place that has to learn
# about a shell primitive when one is added.

@pytest.fixture(scope="module")
def draft() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on this machine, so the shipped helpers cannot be run")

    result = subprocess.run(
        [node, "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"), str(PROBE)],
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_a_discarded_edit_leaves_the_saved_figure_in_the_box(draft):
    """The measured defect, in the helpers the box is bound through."""
    assert draft["on_load"] == 60
    assert draft["after_edit"] == 80
    assert draft["after_discard"] == 60, (
        "discarding the draft has to put the saved rate card back in the box"
    )


def test_typing_the_saved_figure_back_is_a_revert_and_not_a_no_op(draft):
    """Without this the draft kept the earlier edit while the box showed the
    saved value, and the effect panel priced a figure on nobody's screen."""
    assert draft["after_typing_the_saved_figure_back"] == {"shown": 60, "draft": None}
    assert draft["premium_after_edit"] == 1.65
    assert draft["premium_after_revert"] == {"shown": 1.15, "draft": None}


def test_reverting_one_premium_leaves_the_others_staged(draft):
    assert draft["sibling_survives"]["draft"] == {"premiums": {"program_type": {"Reality": 2}}}
    assert draft["sibling_survives"]["reality"] == 2


def test_a_staged_zero_is_a_value_and_not_an_absence(draft):
    """The promo ad-type multiplier is 0.00, so the overlay cannot be written
    with a falsy test or that layer becomes uneditable."""
    assert draft["staged_zero"] == 0


def test_the_activation_switch_holds_the_click_until_it_is_discarded(draft):
    assert draft["switch_after_click"] is True
    assert draft["switch_after_discard"] is False


# ---------------------------------------------------------------------------
# The binding. A correct helper the box does not consult is the defect itself.

# Not `[^>]*`: an onBlur handler carries an arrow, and an arrow carries a `>`.
INPUT = re.compile(r"<(?:input|InputControl)\b.*?/>", re.DOTALL)


def _inputs(source: str) -> list[str]:
    return INPUT.findall(source)


def _attribute(element: str, name: str) -> str | None:
    match = re.search(rf"\b{name}=\{{(.*?)\}}\s*\n", element, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(rf"\b{name}=\{{([^\n]*?)\}}", element)
    return match.group(1).strip() if match else None


def test_every_value_box_on_the_rate_card_shows_the_draft_and_remounts_on_it():
    source = MANAGER.read_text(encoding="utf-8")
    boxes = [element for element in _inputs(source) if "defaultValue=" in element]
    assert len(boxes) == 2, f"the rate card has {len(boxes)} value boxes, expected the base and the premiums"

    overlays = set()
    for element in boxes:
        shown = _attribute(element, "defaultValue")
        key = _attribute(element, "key")
        assert shown and key, f"a value box is missing its value or its key: {element}"
        assert shown in key, (
            f"the box shows {shown} and remounts on {key}, so a change to the value it shows "
            "does not remount it and the box keeps whatever was typed"
        )
        overlays.add(shown)

    for name in overlays:
        definition = re.search(rf"\bconst {re.escape(name)} = ([^\n]+)", source)
        assert definition, (
            f"the box shows {name}, which is not a local this component computes, so it "
            "cannot be consulting the draft"
        )
        assert "draftValueAt(pending" in definition.group(1), (
            f"{name} does not consult the draft, so a discarded edit stays in the box"
        )

    # The exact shape the critic measured, named so it cannot come back quietly.
    assert "defaultValue={state.base.value}" not in source
    assert "key={`base-${state.base.value}`}" not in source


def test_every_switch_on_the_rate_card_shows_the_draft_too():
    manager = MANAGER.read_text(encoding="utf-8")
    switches = [element for element in _inputs(manager) if 'type="checkbox"' in element]
    assert switches, "the rate card has no activation switch"
    for element in switches:
        checked = _attribute(element, "checked")
        assert checked and "draftValueAt(pending" in checked, (
            f"a switch is bound to the saved card alone, so it snaps back when clicked: {checked}"
        )

    events = EVENTS.read_text(encoding="utf-8")
    layer_switch = [element for element in _inputs(events) if 'type="checkbox"' in element]
    assert len(layer_switch) == 1
    checked = _attribute(layer_switch[0], "checked")
    assert "shownEnabled" in checked, checked
    definition = re.search(r"const shownEnabled = ([^\n]+)", events)
    assert definition and "stagedEnabled" in definition.group(1)
