"""P9: a value inside a sentence is a run of its own, in both readings.

Measured on the shipped English approval card before this file existed. The
money's provenance line is the one sentence that stops a one-representative-day
figure being read as a weekly one, and it sits directly above four money rows at
the moment the operator presses Approve. It interpolated the channel name (a
Hebrew run) and the day (an ISO date) into one ``dir="auto"`` paragraph, so the
bidirectional algorithm resolved the comma, the brackets and the digits next to
the Hebrew run as part of it. Measured in a real browser at a 420 px dock with
per-character client rects, the second visual line read

    2024-11-11 ,13), not the weekly total.

with the date printed before the "13" it does not belong to and the comma on the
wrong side of both. The same sentence in Hebrew was correct, which is exactly
why a Hebrew-only reading never caught it.

The rule this file enforces is the one the effect view already stated for its
figures three lines above the defect: one isolate per value, never one around
the pair. So no display string in ``src/kai`` interpolates a value that can be
the other script, in either of the two sinks this destination has. On a card the
value is composed beside the sentence in its own ``bdi``. In a notice, which
takes two plain strings and hands them to the shell's toast, no element can go,
so the value carries the isolate characters instead. The only expressions still
allowed inside a template are counts, listed below and checked to be counts.

Both halves are here: the source rule, which a reviewer can run anywhere, and
the paint, laid out in a real headless browser with the shipped strings and the
shipped stylesheet. Each paint test carries its control: the same characters
without the isolate must still scramble. If one ever stops doing so, the world
changed and this file should say so out loud rather than pass quietly.

Measured after the fix, same browser, same dock width, both readings:

    English  shipped  (13 תשר, 2024-11-11), not the weekly total.
    English  control  2024-11-11 ,13), not the weekly total.
    Hebrew   shipped  , )2024-11-11 ,13 תשר( דחא גציימ ץורע-םוי לע הצר היצלומיסה

And the notice, with the owned channel's name as an uploaded file name, where
the run being read is the file name alone:

    English  alone    csv.13 תשר
    English  shipped  csv.13 תשר
    English  control  13 תשר.csv

(the glyph dumps are visual order, left to right, so Hebrew reads backwards.)
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KAI = ROOT / "tv-break-dashboard" / "src" / "kai"
TOKENS = ROOT / "tv-break-dashboard" / "src" / "tokens.css"
SHEET = KAI / "assistant-console.css"
PROBE = Path(__file__).with_name("test_p9_paint_probe.mjs")
PROPOSALS = ROOT / "data" / "assistant" / "proposals.json"
SETTINGS = ROOT / "data" / "kairos_settings.json"

# Every expression still interpolated into a bilingual display string in this
# destination. Each one is a count or a measured number: a digit run carries no
# script, so it cannot take a neighbouring character into itself. Anything else
# is composed beside the sentence in its own isolate, so a new name, title,
# file, date or error message added to a sentence fails this file until it is.
COUNTS = {
    "restoreResult.files.length",
    "count",
    "thread.length",
    "itemIds.length",
    "removedCount",
    "appliedCount",
    "failedCount",
    "rows.length - shown.length",
    "bulk.changeCount",
    "selectedIds.length",
    "results.length",
    "deadline",
    "file.change_count",
    "file.bytes_now",
    "file.bytes_after_restore",
    "file.changes_omitted",
    "done ? done.restored : 0",
    "rows",
    "sheets",
}

# What makes an allowlisted expression a count rather than a promise that it is
# one. Every entry above has to look like a quantity in its own name.
COUNTLIKE = re.compile(
    r"length|count|bytes|restored|omitted|deadline|^rows$|^sheets$", re.IGNORECASE)


def _read_template(source: str, start: int) -> tuple[str, int]:
    """The template literal opening at ``start``, and the index just past it."""
    index = start + 1
    depth = 0
    while index < len(source):
        char = source[index]
        if char == "\\":
            index += 2
            continue
        if depth == 0 and char == "`":
            return source[start + 1:index], index + 1
        if char == "$" and source[index + 1:index + 2] == "{":
            depth += 1
            index += 2
            continue
        if depth and char == "{":
            depth += 1
        elif depth and char == "}":
            depth -= 1
        elif depth and char == "`":
            _, index = _read_template(source, index)
            continue
        index += 1
    return source[start + 1:], index


def _skip_quoted(source: str, start: int) -> int:
    quote = source[start]
    index = start + 1
    while index < len(source):
        if source[index] == "\\":
            index += 2
            continue
        if source[index] == quote:
            return index + 1
        index += 1
    return index


# The two sinks a bilingual display string in this destination goes into. One
# renders a node, so a value beside it carries a bdi element. The other takes
# two plain strings and hands them to the shell's toast and activity feed, where
# an element cannot go, so a value carries the isolate characters instead. The
# rule is the same rule; only the spelling of the isolate differs.
SINKS = ("pageText(", "notify(")


def _sink_templates(source: str, sink: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    at = source.find(sink)
    while at >= 0:
        line = source.count("\n", 0, at) + 1
        index = at + len(sink)
        depth = 1
        while index < len(source) and depth:
            char = source[index]
            if char == "`":
                literal, index = _read_template(source, index)
                found.append((line, literal))
                continue
            if char in "'\"":
                index = _skip_quoted(source, index)
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            index += 1
        at = source.find(sink, at + 1)
    return found


def _display_templates(source: str) -> list[tuple[int, str]]:
    """Every template literal passed to a display sink, with its line."""
    found: list[tuple[int, str]] = []
    for sink in SINKS:
        found.extend(_sink_templates(source, sink))
    return sorted(found)


def _interpolations(literal: str) -> list[str]:
    """Every ${...} expression in a template literal, innermost included."""
    found: list[str] = []
    index = 0
    while index < len(literal):
        if literal[index] == "$" and literal[index + 1:index + 2] == "{":
            depth = 1
            cursor = index + 2
            while cursor < len(literal) and depth:
                if literal[cursor] == "{":
                    depth += 1
                elif literal[cursor] == "}":
                    depth -= 1
                cursor += 1
            body = literal[index + 2:cursor - 1]
            found.append(body.strip())
            found.extend(_interpolations(body))
            index = cursor
            continue
        index += 1
    return found


def unisolated_interpolations() -> list[tuple[str, int, str]]:
    """Every value interpolated into a display string bare: not a count, and
    not already carrying its own isolate."""
    offenders: list[tuple[str, int, str]] = []
    for path in sorted(KAI.glob("*.js")) + sorted(KAI.glob("*.jsx")):
        source = path.read_text(encoding="utf-8")
        for line, literal in _display_templates(source):
            for expression in _interpolations(literal):
                if expression in COUNTS or expression.startswith("isolate("):
                    continue
                offenders.append((path.name, line, expression))
    return offenders


def test_no_display_string_interpolates_a_value_that_can_be_the_other_script() -> None:
    assert unisolated_interpolations() == []


def test_the_allowlist_holds_only_counts_and_no_stale_entry() -> None:
    """An allowlist that can grow to hold a name is not an allowlist."""
    live = {expression for path in sorted(KAI.glob("*.js")) + sorted(KAI.glob("*.jsx"))
            for _, literal in _display_templates(path.read_text(encoding="utf-8"))
            for expression in _interpolations(literal)}
    for expression in COUNTS:
        assert COUNTLIKE.search(expression), f"not a count: {expression!r}"
        assert expression in live, f"stale allowlist entry: {expression!r}"


def test_the_scanner_sees_an_interpolated_name_when_one_is_put_back() -> None:
    """The measurement that would have caught the defect, run as the test: the
    exact pre-fix line is fed to the scanner and it has to name it."""
    before = ("      {pageText(locale, `The simulation runs one representative "
              "channel-day (${channel}, ${day}), not the weekly total.`, `x (${channel})`)}")
    templates = _display_templates(before)
    assert [expression for _, literal in templates for expression in _interpolations(literal)] == [
        "channel", "day", "channel"]
    assert not any(expression in COUNTS for _, literal in templates
                   for expression in _interpolations(literal))


def test_the_two_sentences_the_defect_was_measured_on_are_composed() -> None:
    """Source half of the paint below: each value sits in its own isolate."""
    effect = (KAI / "AssistantEffectView.jsx").read_text(encoding="utf-8")
    assert '<bdi dir="auto">{channel}</bdi>' in effect
    assert '<bdi dir="ltr">{day}</bdi>' in effect
    panel = (KAI / "AssistantPanel.jsx").read_text(encoding="utf-8")
    assert '<bdi dir="auto">{page.label}</bdi>' in panel
    assert '<bdi dir="auto">{entityLabel}</bdi>' in panel


# --- the paint ---------------------------------------------------------------
BASIS = re.compile(r"const BASIS_(EN|HE) = \{\n\s*lead: '([^']*)',\n\s*mid: '([^']*)',\n\s*tail: '([^']*)',")


def _readings() -> dict[str, tuple[str, str, str]]:
    source = (KAI / "AssistantEffectView.jsx").read_text(encoding="utf-8")
    found = {match[0]: (match[1], match[2], match[3]) for match in BASIS.findall(source)}
    assert set(found) == {"EN", "HE"}, "the shipped basis strings did not parse"
    return found


def _measured_basis() -> tuple[str, str]:
    """The channel and day a real stored proposal was simulated on."""
    if not PROPOSALS.exists():
        pytest.skip("no stored proposals on this machine, so there is no real basis to lay out")
    for batch in json.loads(PROPOSALS.read_text(encoding="utf-8")).get("batches", []):
        for item in batch.get("items", []):
            basis = item.get("effect_basis")
            if basis and basis.get("channel") and basis.get("day"):
                return str(basis["channel"]), str(basis["day"])
    pytest.skip("no stored proposal carries an effect basis")


def _document(locale: str, channel: str, day: str) -> str:
    """The shipped strings in the shipped markup under the shipped stylesheet,
    laid out wide enough for one line: the property under test is the order the
    runs paint in, and a line break is measured on the dock itself."""
    lead, mid, tail = _readings()["HE" if locale == "he" else "EN"]
    shell = "rtl" if locale == "he" else "ltr"
    return f"""<!doctype html><meta charset="utf-8">
<link rel="stylesheet" href="file://{TOKENS}">
<link rel="stylesheet" href="file://{SHEET}">
<body style="margin:0"><div dir="{shell}" style="width:800px">
<p class="asst-effect-basis" id="shipped" dir="auto">{lead}<bdi dir="auto">{channel}</bdi>{mid}<bdi dir="ltr">{day}</bdi>{tail}</p>
<p class="asst-effect-basis" id="control" dir="auto">{lead}{channel}{mid}{day}{tail}</p>
</div></body>"""


PAINT = """(() => {
  const box = (r) => ({ left: Math.round(r.left * 10) / 10, right: Math.round(r.right * 10) / 10, top: Math.round(r.top) });
  const shipped = document.getElementById('shipped');
  const control = document.getElementById('control');
  const bdis = shipped.querySelectorAll('bdi');
  const node = control.firstChild;
  const lead = shipped.firstChild.textContent.length;
  const mid = shipped.childNodes[2].textContent.length;
  const channel = bdis[0].textContent.length;
  const day = bdis[1].textContent.length;
  const range = (start, length) => {
    const r = document.createRange();
    r.setStart(node, start);
    r.setEnd(node, start + length);
    return box(r.getBoundingClientRect());
  };
  return {
    shipped: { channel: box(bdis[0].getBoundingClientRect()), day: box(bdis[1].getBoundingClientRect()) },
    control: { channel: range(lead, channel), day: range(lead + channel + mid, day) },
  };
})()"""


def _paint(locale: str, tmp_path: Path) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the paint cannot be measured")
    channel, day = _measured_basis()
    document = tmp_path / f"basis-{locale}.html"
    document.write_text(_document(locale, channel, day), encoding="utf-8")
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


def test_the_english_reading_paints_the_name_then_the_date(tmp_path) -> None:
    """The measured defect: with the header set to English the date landed
    inside the channel name. Each value now paints as one block, in the reading
    order of the sentence, and the control proves the measurement can fail."""
    painted = _paint("en", tmp_path)
    shipped, control = painted["shipped"], painted["control"]
    assert shipped["channel"]["top"] == shipped["day"]["top"], "the pair split across lines"
    assert shipped["day"]["left"] >= shipped["channel"]["right"], (
        f"the date does not follow the name: {shipped}")
    # The control is the same characters as one interpolated string. It has to
    # still put the date on the wrong side of the name.
    assert control["day"]["right"] <= control["channel"]["left"], (
        f"the pre-fix form no longer scrambles, so this test proves nothing: {control}")


def test_the_hebrew_reading_paints_the_name_then_the_date(tmp_path) -> None:
    """Right to left, so the date paints to the left of the name it follows."""
    painted = _paint("he", tmp_path)
    shipped = painted["shipped"]
    assert shipped["channel"]["top"] == shipped["day"]["top"], "the pair split across lines"
    assert shipped["day"]["right"] <= shipped["channel"]["left"], (
        f"the date does not follow the name: {shipped}")


def test_the_name_and_the_date_are_never_split_across_lines() -> None:
    """Measured in a 420 px dock: the line broke inside the channel name and
    printed רשת at the end of one line and 13 at the start of the next."""
    rule = SHEET.read_text(encoding="utf-8")
    assert ".asst-effect-basis bdi { white-space: nowrap; }" in rule


# --- the other sink ----------------------------------------------------------
# notify takes two plain strings, so the isolate there is the characters and not
# an element. Same rule, same proof: the value has to read inside the sentence
# exactly as it reads on its own, and the bare form has to still fail.
NOTICE = re.compile(
    r"notify\(`([^`]*)\$\{isolate\(filename\)\}([^`]*)`, `([^`]*)\$\{isolate\(filename\)\}([^`]*)`\)")
ISOLATED = "\u2068{value}\u2069"

RUNS = """(() => {
  const order = (id) => {
    const el = document.getElementById(id);
    const node = el.firstChild;
    const start = Number(el.dataset.start);
    const items = [];
    for (let i = start; i < start + Number(el.dataset.len); i += 1) {
      const r = document.createRange();
      r.setStart(node, i);
      r.setEnd(node, i + 1);
      const b = r.getBoundingClientRect();
      if (b.width === 0) continue;
      items.push({ ch: node.data[i], x: b.left });
    }
    items.sort((a, b) => a.x - b.x);
    return items.map((i) => i.ch).join('');
  };
  return { alone: order('alone'), shipped: order('shipped'), control: order('control') };
})()"""


def _notice_document(lead: str, tail: str, value: str, shell: str) -> str:
    shipped = lead + ISOLATED.format(value=value) + tail
    control = lead + value + tail
    return f"""<!doctype html><meta charset="utf-8">
<body style="margin:0;font:12px sans-serif"><div dir="{shell}" style="width:520px">
<p id="alone" dir="auto" data-start="0" data-len="{len(value)}">{value}</p>
<p id="shipped" dir="auto" data-start="{len(lead) + 1}" data-len="{len(value)}">{shipped}</p>
<p id="control" dir="auto" data-start="{len(lead)}" data-len="{len(value)}">{control}</p>
</div></body>"""


def _runs(reading: str, tmp_path: Path) -> dict:
    """The shipped upload notice, laid out with a Hebrew file name in it."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the paint cannot be measured")
    found = NOTICE.search((KAI / "AssistantUpload.jsx").read_text(encoding="utf-8"))
    assert found, "the shipped upload notice did not parse"
    lead, tail = (found.group(1), found.group(2)) if reading == "en" else (found.group(3), found.group(4))
    channel = json.loads(SETTINGS.read_text(encoding="utf-8")).get("operator_channel", "")
    assert channel, "the owned channel comes from settings, not from this file"
    document = tmp_path / f"notice-{reading}.html"
    document.write_text(_notice_document(lead, tail, f"{channel}.csv", "ltr" if reading == "en" else "rtl"), encoding="utf-8")
    expression = tmp_path / "runs.js"
    expression.write_text(RUNS, encoding="utf-8")
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


def test_the_english_notice_paints_the_file_name_as_itself(tmp_path) -> None:
    """Measured before the isolate: the notice printed the Hebrew file name with
    its extension torn off and moved past it, as "Uploaded 13 תשר.csv." The
    isolated form has to read the same inside the sentence as it does alone."""
    painted = _runs("en", tmp_path)
    assert painted["shipped"] == painted["alone"], f"the name reads differently inside the sentence: {painted}"
    assert painted["control"] != painted["alone"], (
        f"the bare form no longer scrambles, so this test proves nothing: {painted}")


def test_the_hebrew_notice_paints_the_file_name_as_itself(tmp_path) -> None:
    """The reading that was always correct, held so a later change cannot break
    it while fixing the other one."""
    painted = _runs("he", tmp_path)
    assert painted["shipped"] == painted["alone"], f"the name reads differently inside the sentence: {painted}"
    assert SETTINGS.exists(), "the owned channel comes from settings, not from this file"
