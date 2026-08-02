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
HELPER = TODAY / "today-bidi.js"

FIRST_STRONG_ISOLATE = "⁨"
POP_DIRECTIONAL_ISOLATE = "⁩"

# Every expression in this destination that puts the channel name into a line of
# text, and the file that prints it. A new one belongs here on the day it is
# written, which is the point of listing them rather than globbing.
CHANNEL_SITES = {
    "TodayMoney.jsx": ["scope.channel"],
    "OverviewPage.jsx": ["channel"],
    "MoneyWaterfall.jsx": ["scopeChannel"],
}


def _source(name: str) -> str:
    return (TODAY / name).read_text(encoding="utf-8")


def test_the_helper_emits_the_first_strong_isolate_pair_and_nothing_else():
    """First strong, not left to right: the name's direction is the name's own."""
    source = _source("today-bidi.js")
    assert "'\\u2068'" in source, "the opening mark is not the first-strong isolate"
    assert "'\\u2069'" in source, "the closing mark is not the pop-directional isolate"
    assert "⁦" not in source and "⁧" not in source


def test_the_helper_wraps_a_real_name_and_stays_silent_on_an_absent_one(tmp_path):
    """Executed, not read: the same bytes, imported and called by node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not installed, so the helper cannot be executed here")
    module = tmp_path / "today-bidi.mjs"
    module.write_text(HELPER.read_text(encoding="utf-8"), encoding="utf-8")
    script = (
        f"import {{ isolate }} from {json.dumps(str(module))};"
        "const out = {channel: isolate('רשת 13'), latin: isolate('Channel 13'),"
        " empty: isolate(''), missing: isolate(null), padded: isolate('  רשת 13  ')};"
        "process.stdout.write(JSON.stringify(out));"
    )
    done = subprocess.run(
        [node, "--input-type=module", "-e", script],
        capture_output=True, text=True, check=True,
    )
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


def test_the_helper_is_the_only_source_of_the_marks_in_this_destination():
    """One definition. A second copy is how two lines drift apart later."""
    carriers = [
        path.name
        for path in sorted(TODAY.iterdir())
        if path.suffix in {".js", ".jsx"} and "\\u2068" in path.read_text(encoding="utf-8")
    ]
    assert carriers == ["today-bidi.js"], f"the isolate marks are also written by {carriers}"


def test_the_downloaded_file_never_carries_the_marks():
    """A CSV is read by a machine, and an invisible mark inside a cell is a bug."""
    export = _source("today-export.js")
    assert "isolate(" not in export
    assert "\\u2068" not in export
