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
import os
import re
import shutil
import subprocess
import tempfile
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
# bidi.jsx cannot be copied the same way: it opens with `import React from
# 'react'` and returns JSX, and this harness has no react runtime the way
# test_p4_rollup_tristate.py's fake-React probe does. dates.js only calls
# `isolate` to keep a formatted run from being reordered inside a Hebrew line,
# which this file's assertions never read, so the stub is the identity
# function, the same shape test_p4's fix used for the same reason: a correct
# isolate the probe cannot see is not worth a fake react tree to produce it.
BIDI_STUB = "export function isolate(value) { return value; }\n"

# node's static ESM resolution cannot be redirected from the CLI the way a
# bundler alias can, so a real loader hook does it: anything ending in
# shell/dates(.js) or shell/bidi(.js) is sent to the two files this fixture
# writes into its own support directory instead of the paths the probe's copy
# never created. The two support files are passed in by URL through the
# environment because the loader hook runs in its own thread and a literal
# path baked into the hook source would have to survive being embedded in a
# JS string; an env var sidesteps that entirely.
LOADER_HOOK = """
const DATES_URL = process.env.P5_DATES_URL;
const BIDI_URL = process.env.P5_BIDI_URL;

export function resolve(specifier, context, nextResolve) {
  if (specifier.endsWith('shell/dates.js') || specifier.endsWith('shell/dates')) {
    return { url: DATES_URL, shortCircuit: true };
  }
  if (specifier.endsWith('shell/bidi.js') || specifier.endsWith('shell/bidi')) {
    return { url: BIDI_URL, shortCircuit: true };
  }
  return nextResolve(specifier, context);
}
"""


@pytest.fixture(scope="module")
def draft() -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on this machine, so the shipped helpers cannot be run")

    with tempfile.TemporaryDirectory() as support_dir:
        support = Path(support_dir)
        dates_source = (SHELL / "dates.js").read_text(encoding="utf-8")
        (support / "dates.js").write_text(
            dates_source.replace("from './bidi'", "from './bidi.js'"), encoding="utf-8",
        )
        (support / "bidi.js").write_text(BIDI_STUB, encoding="utf-8")
        hook_path = support / "resolve-shell.mjs"
        hook_path.write_text(LOADER_HOOK, encoding="utf-8")

        env = dict(os.environ)
        env["P5_DATES_URL"] = (support / "dates.js").as_uri()
        env["P5_BIDI_URL"] = (support / "bidi.js").as_uri()

        result = subprocess.run(
            [node, f"--experimental-loader={hook_path}", str(PROBE)],
            capture_output=True, text=True, timeout=120, check=False, env=env,
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
INPUT = re.compile(r"<input\b.*?/>", re.DOTALL)


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
