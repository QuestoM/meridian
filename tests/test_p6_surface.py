"""P6 Sources, the surface itself: the words, the laws and the file sizes.

Every assertion here is over the shipped source of the Sources destination, so
it fails on the tree rather than in a browser somebody has to be looking at.
The Hebrew half of this surface is not observable from the API alone: half the
words live in the destination's own copy table, and a screen an Israeli
broadcaster reads is not a screen anyone can grade in English.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "tv-break-dashboard" / "src" / "sources"
COPY = SOURCES / "sources-copy.js"

# The four words the vocabulary rule retires, in both languages. The critic's
# check is a grep over the whole frontend; this is that grep, on this tree.
RETIRED = ("recompute", "rebuild", "חישוב מחדש", "בנייה מחדש")

# The words the copy law forbids in product text.
FORBIDDEN_MARKS = ("—", "–", "!")

HEBREW = re.compile(r"[֐-׿]")
PAIR = re.compile(r"\{ ?en: '((?:[^'\\]|\\.)*)', he: '((?:[^'\\]|\\.)*)' ?\}")


def _sources_files(suffixes=(".jsx", ".js")) -> list[Path]:
    return sorted(path for path in SOURCES.iterdir() if path.suffix in suffixes)


def test_every_word_this_destination_renders_has_both_languages() -> None:
    text = COPY.read_text(encoding="utf-8")
    pairs = PAIR.findall(text)
    assert len(pairs) >= 60, f"the copy table shrank to {len(pairs)} pairs"
    for english, hebrew in pairs:
        assert english.strip(), "an English string is empty"
        assert hebrew.strip(), f"'{english}' has no Hebrew"
        assert HEBREW.search(hebrew), f"'{english}' has a Hebrew slot with no Hebrew in it"


def test_no_retired_word_reaches_this_destination() -> None:
    for path in _sources_files((".jsx", ".js", ".css")):
        body = path.read_text(encoding="utf-8").lower()
        for word in RETIRED:
            assert word.lower() not in body, f"{path.name} carries the retired word {word}"


def _display_strings(path: Path) -> list[str]:
    """Every string this file can put on a screen.

    The copy table's pairs, plus any quoted string carrying Hebrew, which is
    the only other place product text lives in this tree. Operators like ``!==``
    are code and are not read by anybody, so they are not swept.
    """
    body = path.read_text(encoding="utf-8")
    strings = [side for pair in PAIR.findall(body) for side in pair]
    strings.extend(match for match in re.findall(r"'([^'\n]*)'", body) if HEBREW.search(match))
    strings.extend(match for match in re.findall(r">([^<>{}\n]+)<", body) if match.strip())
    return strings


def test_the_copy_carries_no_em_dash_no_emoji_and_no_exclamation() -> None:
    for path in _sources_files():
        for string in _display_strings(path):
            for mark in FORBIDDEN_MARKS:
                assert mark not in string, f"{path.name} carries {mark!r} in {string!r}"
            assert not any(0x2100 < ord(character) < 0xE01F0 for character in string), (
                f"{path.name} carries a symbol outside the text planes in {string!r}"
            )


def test_the_operator_is_never_called_a_user_in_hebrew() -> None:
    """The vocabulary rule is on the noun, and the critic's check is a grep, so
    the verb form is avoided too rather than argued about."""
    for path in _sources_files():
        assert "משתמש" not in path.read_text(encoding="utf-8"), f"{path.name} says משתמש"


def test_no_display_string_is_hard_wrapped_across_source_lines() -> None:
    """One display string per source line. A sentence split across two lines is
    a sentence a translator and a reviewer both read wrong."""
    for path in _sources_files():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.count("'") % 2 == 1 and not line.rstrip().endswith(("'", "',", "';", "'}")):
                assert "\\" not in line[-2:], f"{path.name}:{number} continues a string onto the next line"


def test_no_file_in_this_destination_is_over_the_size_law() -> None:
    for path in _sources_files((".jsx", ".js", ".css")):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= 450, f"{path.name} is {lines} lines"


def test_no_file_on_this_row_is_over_the_size_law() -> None:
    """The same law over the whole row, not only the half a browser renders.

    Measured before this test existed: ``tests/test_p6_sources.py`` had reached
    542 lines while every frontend file was inside the cap, so the check that
    was here passed over a file that was not. A test file is a source file, and
    the two backend modules on this row sit one line under the cap, which is
    exactly where a law stops holding by itself.
    """
    backend = sorted(
        path
        for pattern in ("uploads*.py", "downloads_api*.py", "exporters.py")
        for path in (ROOT / "kairos_api").glob(pattern)
    )
    tests = sorted((ROOT / "tests").glob("test_p6_*.py"))
    # A sweep over an empty list passes without looking at anything, which is
    # the failure this test exists to catch one level up.
    assert len(backend) >= 12, f"the backend sweep found {len(backend)} modules"
    assert len(tests) >= 7, f"the test sweep found {len(tests)} files"
    for path in backend + tests:
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= 450, f"{path.name} is {lines} lines"


def test_the_destination_defines_no_design_token() -> None:
    """Tokens live in one file and this destination is not it."""
    for path in sorted(SOURCES.glob("*.css")):
        body = path.read_text(encoding="utf-8")
        assert not re.search(r"^\s*--[a-z-]+:", body, re.MULTILINE), f"{path.name} defines a token"


def test_every_state_and_every_role_the_server_can_send_has_a_word() -> None:
    from kairos_api import uploads_status

    text = COPY.read_text(encoding="utf-8")
    for state in uploads_status.STATES:
        assert re.search(rf"\b{state}: \{{", text), f"the surface has no word for the {state} state"
    for role in ("input", "plan", "model"):
        assert re.search(rf"\b{role}: \{{", text), f"the surface has no word for the {role} role"


def test_the_paths_that_read_a_server_sentence_never_write_their_own() -> None:
    """A consequence, a remedy and a note are computed by the server, so the
    surface renders the words it was sent. A local copy is how two screens come
    to disagree about what an upload will do."""
    for name in ("SourceCard.jsx", "SourceFilesView.jsx", "RowsDrawer.jsx", "ReportRowsDrawer.jsx"):
        body = (SOURCES / name).read_text(encoding="utf-8")
        for field in ("consequence", "remedy", "note"):
            if f"input.{field}" in body or f"row.{field}" in body or f"report.{field}" in body:
                assert "serverText(" in body, f"{name} renders a {field} without the server's own words"


def test_the_two_drawers_open_on_the_side_the_page_reads_from() -> None:
    """The drawer is placed with logical properties, so it opens from the
    reading edge in Hebrew and from the other one in English without a second
    rule."""
    body = (SOURCES / "sources-tables.css").read_text(encoding="utf-8")
    block = body[body.index(".rows-drawer {"): body.index("}", body.index(".rows-drawer {"))]
    assert "inset-inline-end" in block, "the drawer is pinned to a physical side"
    assert "right:" not in block and "left:" not in block, "the drawer uses a physical side"
