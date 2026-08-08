"""A Hebrew screen never prints a Latin unit letter for seconds.

RULED 2026-08-09, after measuring rather than choosing. The product printed
seconds three ways for one unit:

    שניות     the full word, in twenty places
    שנ'       the standard Hebrew abbreviation, in two
    s         a bare Latin letter, in eight

The third is the one that is simply wrong, and it was on the compact surfaces a
person reads most: the pod's copy-check badge, its elapsed and length columns,
the run button on three destinations, and the breaks-times-seconds figure on two
Today panels. A Hebrew reader saw `מריץ 12s` and `4 × 30s`.

The ruling is NOT "always spell it out". A column is not a sentence, and Hebrew
has an accepted abbreviation, so a compact place uses `שנ'` and prose keeps
`שניות`. What is forbidden is a fourth form, and a Latin letter standing in for
a Hebrew one.

This guards the forbidden half, which is the half a person can get wrong by
accident. It is a class check and not a site check: it sweeps the whole tree,
because the original defect was reported at one site and found at eight.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"

# A number, or an interpolation, immediately followed by a bare Latin s, inside a
# template literal that also carries Hebrew. The Hebrew is what makes it a Hebrew
# string rather than an English one, which is allowed to say 12s all it likes.
LATIN_SECONDS = re.compile(r"[`'\"][^`'\"\n]*[֐-׿][^`'\"\n]*?(?:\}|\d)s(?![a-zA-Z])")

# CSS lives elsewhere and a duration in a transition is not a sentence.
SKIP_SUFFIXES = {".css", ".json", ".md"}


def _sources() -> list[Path]:
    return sorted(
        path
        for path in SRC.rglob("*")
        if path.is_file() and path.suffix not in SKIP_SUFFIXES and path.suffix in {".js", ".jsx"}
    )


def test_no_hebrew_string_prints_a_latin_s_for_seconds():
    offenders: list[str] = []
    for path in _sources():
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if LATIN_SECONDS.search(line):
                offenders.append(f"{path.relative_to(SRC)}:{number}: {line.strip()[:110]}")
    assert offenders == [], (
        "a Hebrew string prints a Latin s where the unit belongs:\n  "
        + "\n  ".join(offenders)
        + "\nHebrew says שנ' where a column needs the short form and שניות in prose. "
        "A Latin unit letter on a Hebrew screen is not an abbreviation, it is another "
        "language's."
    )


def test_the_check_bites(tmp_path, monkeypatch):
    """A guard that has never failed has never been shown to work."""
    folder = tmp_path / "src"
    folder.mkdir()
    (folder / "bait.jsx").write_text(
        "const label = `מריץ ${elapsed}s`;\n", encoding="utf-8"
    )
    monkeypatch.setattr(Path, "rglob", Path.rglob)
    monkeypatch.setitem(globals(), "SRC", folder)
    offenders = [
        line
        for line in (folder / "bait.jsx").read_text(encoding="utf-8").splitlines()
        if LATIN_SECONDS.search(line)
    ]
    assert offenders, "the pattern does not match the very shape it was written for"


@pytest.mark.parametrize("sample,matches", [
    ("`מריץ ${elapsed}s`", True),
    ("`4 × ${n}s`", False),          # an English-only string may say 12s
    ("`מריץ ${elapsed} שנ'`", False),
    ("`שניות: ${n}`", False),
])
def test_the_pattern_tells_a_hebrew_string_from_an_english_one(sample, matches):
    assert bool(LATIN_SECONDS.search(sample)) is matches
