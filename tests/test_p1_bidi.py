"""The operator's channel name is data, and data does not obey a sentence.

The name is Hebrew in this market and Today ships an English toggle, so on the
English screen a Hebrew run sits inside an English line. Left unisolated, the
bidirectional algorithm resolves the separator and the digits next to that run
as part of it. Measured in the browser with ``Range.getBoundingClientRect`` on
the shipped English Today screen, the money figure's own basis line

    "רשת 13 · 1 Nov 2024 to 7 Nov 2024 (7 days) · ..."

reached the reader as

    "1 · רשת 13 Nov 2024 to 7 Nov 2024 (7 days) · ..."

with the window's first day torn off its date and printed before the separator,
and the yield panel's scope line put "2024-11-01" in the middle of the channel
name the same way. That is the destination's doctrine broken at the one place it
matters most: the basis travels with the figure, so a wrong basis is a wrong
figure.

The fix is one pure helper and three call sites. This file holds both ends of
it: the helper really emits the isolate pair, and no line in the destination
interpolates the channel without it. The rendered proof is the browser
measurement above, which no test runner in this repository can take.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TODAY = ROOT / "tv-break-dashboard" / "src" / "today"
# The helper moved and this file had to follow it.
#
# It was src/today/today-bidi.js, one destination's own copy. On 2026-08-08 the
# isolation primitive was consolidated into src/shell/bidi.jsx as the single home
# for the whole product, and the per-destination copy was deleted. Today's call
# sites now import isolate from ../shell/bidi and nothing in this destination
# writes the characters at all.
#
# The point this file was written to defend is unchanged and is now defended
# over a wider area: ONE definition, because a second copy is how two lines drift
# apart later. So the home test below asserts the marks live in the primitive and
# nowhere in today/, which is strictly stronger than asserting they live in one
# named file inside today/.
HELPER = ROOT / "tv-break-dashboard" / "src" / "shell" / "bidi.jsx"

FIRST_STRONG_ISOLATE = "⁨"
POP_DIRECTIONAL_ISOLATE = "⁩"

# Every expression in this destination that puts the channel name into a line of
# text, and the file that prints it. A new one belongs here on the day it is
# written, which is the point of listing them rather than globbing.
CHANNEL_SITES = {
    "TodayMoney.jsx": ["scope.channel"],
    "MoneyWaterfall.jsx": ["scopeChannel"],
}

CHANNEL_NAME_SITES = {
    "TransmissionRibbon.jsx": ["programs[0]?.channel || today?.channel"],
}


def _source(name: str) -> str:
    return (TODAY / name).read_text(encoding="utf-8")


def test_the_helper_emits_the_first_strong_isolate_pair_and_nothing_else():
    """First strong, not left to right: the name's direction is the name's own."""
    source = HELPER.read_text(encoding="utf-8")
    assert "'\\u2068'" in source, "the opening mark is not the first-strong isolate"
    assert "'\\u2069'" in source, "the closing mark is not the pop-directional isolate"
    assert "⁦" not in source and "⁧" not in source


def test_the_helper_wraps_a_real_name_and_stays_silent_on_an_absent_one(tmp_path):
    """Executed, not read: the same bytes, imported and called by node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so the helper cannot be executed here")
    # The helper is JSX now, and node cannot parse JSX. It is still the same
    # bytes that ship: the file is compiled with the bundler's own transform,
    # the one the product builds with, rather than hand-copied or reduced to the
    # function under test. A stub of isolate would prove nothing, since the whole
    # claim of this test is that the shipped definition really emits the pair.
    # The compiled copy goes INSIDE the dashboard, not into pytest's tmp_path.
    # It still imports react at the top, and an ES module resolves a bare
    # specifier from the importing FILE's own location rather than from the
    # working directory, so a copy anywhere else cannot find node_modules.
    dashboard = ROOT / "tv-break-dashboard"
    module = dashboard / f".bidi-probe-{tmp_path.name}.mjs"
    compile_script = (
        "import { readFileSync, writeFileSync } from 'node:fs';"
        "const { transformWithOxc } = await import('vite');"
        f"const src = readFileSync({json.dumps(str(HELPER))}, 'utf8');"
        f"const out = await transformWithOxc(src, {json.dumps(str(HELPER))});"
        f"writeFileSync({json.dumps(str(module))}, out.code);"
    )
    compiled = subprocess.run(
        [node, "--input-type=module", "-e", compile_script],
        capture_output=True, text=True, cwd=str(dashboard), check=False,
    )
    if compiled.returncode != 0:
        pytest.skip(f"the dashboard's bundler is not installed here: {compiled.stderr[:200]}")
    script = (
        f"import {{ isolate }} from {json.dumps(str(module))};"
        "const out = {channel: isolate('רשת 13'), latin: isolate('Channel 13'),"
        " empty: isolate(''), missing: isolate(null), padded: isolate('  רשת 13  ')};"
        "process.stdout.write(JSON.stringify(out));"
    )
    try:
        done = subprocess.run(
            [node, "--input-type=module", "-e", script],
            capture_output=True, text=True, cwd=str(dashboard), check=True,
        )
    finally:
        module.unlink(missing_ok=True)
    out = json.loads(done.stdout)
    assert out["channel"] == f"{FIRST_STRONG_ISOLATE}רשת 13{POP_DIRECTIONAL_ISOLATE}"
    assert out["latin"] == f"{FIRST_STRONG_ISOLATE}Channel 13{POP_DIRECTIONAL_ISOLATE}"
    assert out["padded"] == out["channel"], "padding must not end up inside the isolate"
    assert out["empty"] == "" and out["missing"] == "", "an absent name gets no marks at all"


@pytest.mark.parametrize("name", sorted(CHANNEL_SITES))
def test_every_channel_the_destination_prints_is_isolated_first(name):
    source = _source(name)
    for expression in CHANNEL_SITES[name]:
        printed = re.findall(rf"[^a-zA-Z0-9_.]({re.escape(expression)})\s*[}})]", source)
        assert printed, f"{name} no longer prints {expression}, so this guard is stale"
        bare = re.findall(rf"\$\{{\s*{re.escape(expression)}\s*}}", source)
        assert not bare, f"{name} interpolates {expression} without isolating it"
        assert f"isolate({expression})" in source, f"{name} does not isolate {expression}"


@pytest.mark.parametrize("name", sorted(CHANNEL_NAME_SITES))
def test_every_channel_printed_as_a_name_uses_the_shared_bidi_primitive(name):
    source = _source(name)
    for expression in CHANNEL_NAME_SITES[name]:
        assert f"<Name>{{{expression}}}</Name>" in source, (
            f"{name} prints {expression} without the shared first-strong Name primitive"
        )


def test_the_helper_is_the_only_source_of_the_marks_in_this_destination():
    """One definition. A second copy is how two lines drift apart later."""
    carriers = [
        path.name
        for path in sorted(TODAY.iterdir())
        if path.suffix in {".js", ".jsx"} and "\\u2068" in path.read_text(encoding="utf-8")
    ]
    assert carriers == [], (
        f"the isolate marks are written inside today/ by {carriers}. They belong in "
        f"{HELPER.name} and nowhere else; import isolate from ../shell/bidi instead of "
        "restating the characters."
    )
    assert "\\u2068" in HELPER.read_text(encoding="utf-8"), (
        f"{HELPER.name} no longer defines the isolate pair, so this guard is stale"
    )


def test_the_downloaded_file_never_carries_the_marks():
    """A CSV is read by a machine, and an invisible mark inside a cell is a bug."""
    export = _source("today-export.js")
    assert "isolate(" not in export
    assert "\\u2068" not in export
