"""P5: the money pair on Rules paints before then after, measured in a browser.

Both of this destination's "state the effect before it is saved" cards print a
value that moved as ``before -> after``. The string was always right. The paint
was not.

This market's money is formatted by ``Intl`` as ``he-IL``, which puts U+200F in
front of the digits and another in front of the shekel sign, so each side is a
strong right-to-left run and the arrow between them is a neutral. The
bidirectional algorithm resolves a neutral between two right-to-left runs as
right-to-left, welds all three into one run and paints it from the right.
``dir="ltr"`` on the container does not stop it: an attribute sets a base
direction, it does not isolate a run.

Measured in a real browser with ``Range.getBoundingClientRect`` on the shipped
Hebrew rate card before the fix, the before value's box started at x=783 and the
after value's at x=636, so the after value painted 147 px to the LEFT of the
before value. On the restriction card, measured on the same element with the
isolate marks stripped, the gap was 98 px. A revenue owner raising the base rate
read the plan falling while the signed delta directly above said it rose.

So these tests do not assert the string. They lay the shipped string out in a
real browser and compare where the two runs actually landed, and they carry the
control that makes that assertion mean something: the same string without the
isolate marks must still paint backwards. If it ever stops doing so, the world
changed and this file should say so out loud rather than pass quietly.
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
HELPER = RULES / "rules-bidi.js"
PROBE = Path(__file__).with_name("test_p5_paint_probe.mjs")
TOKENS = ROOT / "tv-break-dashboard" / "src" / "tokens.css"
SHEET = RULES / "rules-restrictions.css"

FIRST_STRONG_ISOLATE = "⁨"
POP_DIRECTIONAL_ISOLATE = "⁩"

# Every expression on this destination that prints a pair, and the file that
# prints it. A new one belongs here on the day it is written.
PAIR_SITES = {
    "RestrictionEffect.jsx": 2,
    "RateCardEffect.jsx": 2,
    "LicencePage.jsx": 1,
}


def _node() -> str:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so the shipped helper cannot be executed here")
    return node


def _run_node(script: str) -> dict:
    # --import wires the shell resolver hook, so the copied rules-bidi module's
    # own `../shell/bidi` import finds the real primitive rather than nothing.
    done = subprocess.run(
        [_node(), "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
    return json.loads(done.stdout)


def _shipped_pairs(tmp_path: Path) -> dict:
    """The pair strings the destination really prints, from the real module.

    The helper is imported rather than reimplemented, so a change to it changes
    what these tests measure instead of silently diverging from it.
    """
    module = tmp_path / "rules-bidi.mjs"
    module.write_text(HELPER.read_text(encoding="utf-8"), encoding="utf-8")
    script = (
        f"import {{ valuePair }} from {json.dumps(str(module))};"
        "const money = (v) => new Intl.NumberFormat('he-IL',"
        " {style:'currency',currency:'ILS',maximumFractionDigits:0,minimumFractionDigits:0}).format(v);"
        "const rate = (v) => new Intl.NumberFormat('he-IL',"
        " {style:'currency',currency:'ILS',maximumFractionDigits:4,minimumFractionDigits:4}).format(v);"
        "process.stdout.write(JSON.stringify({"
        " money: valuePair(money(11155641), money(11083016)),"
        " rate: valuePair(rate(156.8068), rate(190.0689)),"
        " breaks: valuePair(3, 2),"
        " sides: [money(11155641), money(11083016)]}));"
    )
    return _run_node(script)


# The element each pair really ships in, read off the running product. The rate
# pair sits in the large bold delta element and the break counts sit in a change
# row, so laying all three out under one class would measure a context the
# product does not have and would miss a direction rule added to either of the
# other two.
CARRIER = {
    "money": "rules-figure-pair",
    "rate": "rules-figure-delta",
    "breaks": "rules-change-breaks",
    "stripped": "rules-figure-pair",
}

MEASURE = r"""
(() => {
  const out = {};
  document.querySelectorAll('span.rules-figure-pair, span.rules-figure-delta, span.rules-change-breaks').forEach((span) => {
    const node = span.firstChild;
    const text = node.textContent;
    const arrow = text.indexOf('→');
    const rect = (from, to) => {
      const range = document.createRange();
      range.setStart(node, from);
      range.setEnd(node, to);
      const box = range.getBoundingClientRect();
      return { x: Math.round(box.x), right: Math.round(box.right) };
    };
    const digits = (part, offset) => {
      const found = /[0-9][0-9.,]*/.exec(part);
      return found ? rect(offset + found.index, offset + found.index + found[0].length) : null;
    };
    out[span.id] = {
      logical: text,
      before: digits(text.slice(0, arrow), 0),
      after: digits(text.slice(arrow + 1), arrow + 1),
      arrow: rect(arrow, arrow + 1),
    };
  });
  return out;
})()
"""


def _paint(tmp_path: Path, spans: dict) -> dict:
    """Lay the given strings out in a browser and report where they landed."""
    if not PROBE.exists():
        pytest.skip("the paint probe is missing")
    body = "\n".join(
        f'<span class="{CARRIER.get(name, "rules-figure-pair")}" dir="ltr" id="{name}">{text}</span>'
        for name, text in spans.items()
    )
    document = tmp_path / "pair.html"
    document.write_text(
        "<!doctype html><html dir=\"rtl\" lang=\"he\"><head><meta charset=\"utf-8\"><style>"
        + TOKENS.read_text(encoding="utf-8")
        + SHEET.read_text(encoding="utf-8")
        + "span{display:block}</style></head><body><div class=\"rules-figure\">"
        + body
        + "</div></body></html>",
        encoding="utf-8",
    )
    expression = tmp_path / "measure.js"
    expression.write_text(MEASURE, encoding="utf-8")
    # --import wires the shell resolver hook. The probe itself never imports a
    # shell primitive, but the flag costs nothing and keeps every node
    # invocation in this file resolving the same way.
    done = subprocess.run(
        [_node(), "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         str(PROBE), str(document), str(expression)],
        capture_output=True, text=True,
    )
    if done.returncode == 2:
        pytest.skip("no chrome on this machine, so the paint cannot be measured here")
    assert done.returncode == 0, f"the probe failed: {done.stderr[:600]}"
    return json.loads(done.stdout)


def test_the_helper_wraps_each_side_in_a_first_strong_isolate(tmp_path):
    """First strong, not left to right: a Hebrew figure keeps its own direction."""
    pairs = _shipped_pairs(tmp_path)
    before, after = pairs["sides"]
    expected = (
        f"{FIRST_STRONG_ISOLATE}{before}{POP_DIRECTIONAL_ISOLATE}"
        f" → "
        f"{FIRST_STRONG_ISOLATE}{after}{POP_DIRECTIONAL_ISOLATE}"
    )
    assert pairs["money"] == expected
    assert pairs["breaks"] == f"{FIRST_STRONG_ISOLATE}3{POP_DIRECTIONAL_ISOLATE} → {FIRST_STRONG_ISOLATE}2{POP_DIRECTIONAL_ISOLATE}"
    source = HELPER.read_text(encoding="utf-8")
    assert "'\\u2068'" in source and "'\\u2069'" in source, "the marks are not written as escapes"
    assert "⁦" not in source and "⁧" not in source, "a directional isolate other than first-strong"


@pytest.mark.parametrize("figure", ["money", "rate", "breaks"])
def test_the_shipped_pair_paints_before_then_after(tmp_path, figure):
    """The measurement the string cannot make: which run is further left."""
    pairs = _shipped_pairs(tmp_path)
    painted = _paint(tmp_path, {figure: pairs[figure]})[figure]
    before, after, arrow = painted["before"], painted["after"], painted["arrow"]
    assert before and after, f"no digits found in {painted['logical']!r}"
    assert before["right"] <= arrow["x"], (
        f"the before value does not sit left of the arrow: {before} then {arrow}"
    )
    assert arrow["right"] <= after["x"], (
        f"the after value does not sit right of the arrow: {arrow} then {after}"
    )


def test_the_same_string_without_the_isolates_still_paints_backwards(tmp_path):
    """The control. Without it the assertion above could be measuring nothing."""
    pairs = _shipped_pairs(tmp_path)
    stripped = re.sub(f"[{FIRST_STRONG_ISOLATE}{POP_DIRECTIONAL_ISOLATE}]", "", pairs["money"])
    painted = _paint(tmp_path, {"stripped": stripped})["stripped"]
    before, after = painted["before"], painted["after"]
    assert before and after
    assert after["x"] < before["x"], (
        "the unisolated pair now paints in order, so this browser no longer reproduces the "
        f"bug these isolates exist to fix: {painted}"
    )


def test_every_pair_this_destination_prints_goes_through_the_helper(tmp_path):
    """One definition. A second copy is how two lines drift apart later."""
    for name, count in PAIR_SITES.items():
        source = (RULES / name).read_text(encoding="utf-8")
        assert source.count("valuePair(") == count, f"{name} no longer prints {count} pairs"
    hand_rolled = [
        path.name
        for path in sorted(RULES.iterdir())
        if path.suffix in {".js", ".jsx"}
        and path.name != "rules-bidi.js"
        and re.search(r"\}\s*→\s*\$\{", path.read_text(encoding="utf-8"))
    ]
    assert hand_rolled == [], f"{hand_rolled} build a pair by hand instead of calling valuePair"
    carriers = [
        path.name
        for path in sorted(RULES.iterdir())
        if path.suffix in {".js", ".jsx"} and "\\u2068" in path.read_text(encoding="utf-8")
    ]
    assert carriers == ["rules-bidi.js"], f"the isolate marks are also written by {carriers}"


def test_the_signed_delta_beside_the_pair_is_isolated_too():
    """One card, one currency, one rendering. The delta sits directly above the pair."""
    for name in ("RestrictionEffect.jsx", "RateCardEffect.jsx"):
        source = (RULES / name).read_text(encoding="utf-8")
        deltas = re.findall(r"rules-figure-delta[^\n]*\n\s*\{([^}]+)\}", source)
        assert deltas, f"{name} no longer prints a signed delta, so this guard is stale"
        for expression in deltas:
            assert "isolate(" in expression or "valuePair(" in expression, (
                f"{name} prints {expression.strip()} without isolating it"
            )


def test_every_standalone_money_figure_on_this_destination_is_isolated_too():
    """The rate card's own heading figure sits above the pair and is the same currency.

    Measured on the running product before this was closed: the worth of a second
    painted the shekel sign against the last digit with its space stranded, while
    the pair three centimetres below painted the sign in front of the digits. Both
    were legible; they were not the same rendering, on one screen, of one number.
    """
    for name, marker in (("WorthOfASecond.jsx", "rules-worth-value"),):
        source = (RULES / name).read_text(encoding="utf-8")
        printed = re.findall(rf"{marker}[^\n]*\{{([^}}]+)\}}", source)
        assert printed, f"{name} no longer prints {marker}, so this guard is stale"
        for expression in printed:
            assert "isolate(" in expression, f"{name} prints {expression.strip()} without isolating it"


def test_the_money_formatter_is_the_one_these_measurements_were_taken_with():
    """If money stops being he-IL currency, the strings measured above are not the strings shipped."""
    source = (RULES / "rules-lib.js").read_text(encoding="utf-8")
    assert "'he-IL'" in source
    assert "style: 'currency'" in source
    assert "currency: 'ILS'" in source
    assert "export { isolate, valuePair } from './rules-bidi';" in source
