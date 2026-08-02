"""P8 History, the laws: what a surface can be read for.

Split out of ``tests/test_p8_history_frontend.py`` when that file reached 491
lines against the 450-line law. Its own cap test globs ``src/history`` and so
could not see itself, which is exactly the blind spot this file closes: the cap
here is measured over the piece's own test files as well as over its source.

The laws a surface can be read for are the Israeli week Sunday-first with the
weekend on Friday and Saturday, one timezone shared with the server, the
canonical Hebrew vocabulary with the retired words absent, no em-dash and no
exclamation mark in copy, logical properties rather than left and right, only
declared design tokens, and one display string per source line.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
HISTORY = SRC / "history"

LINE_CAP = 450

# The words section 4.8 retires from both activities, in both languages.
RETIRED_WORDS = ("recompute", "rebuild", "חישוב מחדש", "בנייה מחדש")


def _files(*suffixes: str) -> list[Path]:
    return sorted(path for path in HISTORY.rglob("*") if path.suffix in suffixes)


def _read(name: str) -> str:
    return (HISTORY / name).read_text(encoding="utf-8")


def _tree_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _files(".jsx", ".js"))


def test_the_israeli_week_is_sunday_first_with_a_friday_saturday_weekend() -> None:
    labels = _read("history-labels.js")
    order = re.findall(r"\['(\w+)', '([^']+)'\]", labels)
    weekdays = [pair for pair in order if pair[0] in (
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")]
    assert [day for day, _ in weekdays] == [
        "Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    assert weekdays[0][1] == "ראשון" and weekdays[6][1] == "שבת"
    assert "export const WEEKEND_DAYS = [5, 6];" in labels


def test_the_surface_reads_timestamps_in_the_same_zone_the_server_files_them_in() -> None:
    """The day a row is grouped under and the clock printed on it must come from
    one zone, or a change made after midnight in Tel Aviv reads as yesterday."""
    labels = _read("history-labels.js")
    assert "export const BROADCAST_ZONE = 'Asia/Jerusalem';" in labels
    assert labels.count("timeZone: BROADCAST_ZONE") >= 2
    assert "toLocaleDateString('en-CA', { timeZone: BROADCAST_ZONE })" in labels
    server = (ROOT / "kairos_api" / "history_api_timeline.py").read_text(encoding="utf-8")
    assert 'ZoneInfo("Asia/Jerusalem")' in server


def test_the_canonical_hebrew_words_are_used_and_the_retired_ones_are_absent() -> None:
    text = _tree_text()
    for retired in RETIRED_WORDS:
        assert retired not in text, f"the retired word {retired} is on a History surface"
    assert "משתמש" not in text, "the operator is מפעיל, never משתמש"
    assert "נקודת שחזור" in text
    assert "הכנסה צפויה" in _read("history-labels.js")


def test_no_em_dash_no_emoji_and_no_exclamation_mark_in_the_copy() -> None:
    """Read the display strings themselves rather than the whole file, so a
    boolean negation in code cannot be mistaken for an exclamation mark in copy."""
    emoji = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
    for path in _files(".jsx", ".js", ".css"):
        text = path.read_text(encoding="utf-8")
        assert "—" not in text and "–" not in text, f"{path.name} carries a dash that is not a hyphen"
        assert not emoji.search(text), f"{path.name} carries an emoji"
        for literal in re.findall(r"'([^'\\\n]*)'", text):
            assert "!" not in literal, f"{path.name} carries an exclamation mark in copy: {literal}"


def test_the_stylesheets_use_logical_properties_and_only_declared_tokens() -> None:
    tokens = (SRC / "tokens.css").read_text(encoding="utf-8")
    declared = set(re.findall(r"(--[a-z0-9-]+):", tokens))
    for path in _files(".css"):
        text = path.read_text(encoding="utf-8")
        used = set(re.findall(r"var\((--[a-z0-9-]+)\)", text))
        missing = sorted(used - declared)
        assert missing == [], f"{path.name} reads tokens that tokens.css does not define: {missing}"
        for physical in ("margin-left:", "margin-right:", "padding-left:", "padding-right:",
                         "border-left:", "border-right:", "left:", "right:"):
            assert physical not in text, f"{path.name} uses {physical}, which does not flip in RTL"


def test_no_history_source_file_is_over_the_cap() -> None:
    oversize = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in _files(".jsx", ".js", ".css")
        if len(path.read_text(encoding="utf-8").splitlines()) > LINE_CAP
    }
    assert oversize == {}


def test_no_test_file_this_piece_owns_is_over_the_cap_either() -> None:
    """The cap is a law about files, not about source files. The measurement that
    caught this piece was a critic counting the lines of a test, so the count is
    taken here rather than left to a glob that cannot see itself."""
    owned = sorted((ROOT / "tests").glob("test_p8_*.py"))
    assert len(owned) >= 4, "the piece's own test files are found by their reserved prefix"
    oversize = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in owned
        if len(path.read_text(encoding="utf-8").splitlines()) > LINE_CAP
    }
    assert oversize == {}


def test_every_display_string_sits_on_one_source_line() -> None:
    """A display string split across source lines cannot be read, translated or
    grepped as one string. Every pageText pair opens and closes on its own line."""
    template = re.compile(r"`[^`]*`")
    for path in _files(".jsx", ".js"):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if "pageText(" not in line:
                continue
            code = template.sub("", line)
            assert code.count("'") % 2 == 0, (
                f"{path.name}:{number} leaves a display string open at the line end")
            assert line.count("`") % 2 == 0, (
                f"{path.name}:{number} leaves a template string open at the line end")
