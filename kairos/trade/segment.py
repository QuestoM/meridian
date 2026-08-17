"""Deterministic clause segmentation for Hebrew commercial agreements.

Pure text -> Clause list (engine-design §4 stage 3): no I/O, no model. The
input is the ingest stage's normalised page text (bidi controls already
stripped, furniture already removed); the output is every line of the
document landed in exactly one Clause — body text in ``Clause.text``,
section headings preserved on ``Clause.heading``.

Boundary signals, in priority order: chapter headings (פרק ב'), appendix
headings and appendix section heads (נספח א' סעיף 2 -> "appA-2"), the
signature block (ולראיה באו), preamble blocks (בין/ובין parties, הואיל
recitals), then the document's own clause numbering.

Numbered heads are where extracted text lies: pdftotext's bidi handling
lets dates (01.02.2026), decimal table cells (1.15 alone on a line) and
wrapped cross-references (סעיף 2.4) start a line looking exactly like a
clause head. Three guards keep them out, each earned against the corpus:
components with leading zeros are dates; a number with no letters after it
on the line is a table cell; and accepted heads must strictly increase
(document numbering never goes backwards, references usually do).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from .documents import Clause
from .ingest import PageText

# Hebrew alphabet in traditional order; appendix letters map to Latin by
# position (א -> appA, ב -> appB, ...).
_HEBREW_ORDER = "אבגדהוזחטיכלמנסעפצקרשת"

_CHAPTER = re.compile(r"^פרק\s+([א-ת])[\"'׳]?\s*(?:[—–-]|$)")
_APPENDIX_HEADING = re.compile(r"^נספח\s+([א-ת])['׳]?\s*[—–-]")
_APPENDIX_SECTION = re.compile(r"^נספח\s+([א-ת])['׳]?\s*סעיף\s*:?\s*(\d{1,3})")
_APPENDIX_WHOLE = re.compile(r"^נספח\s+([א-ת])['׳]?\s*:")
_SIGNATURE = re.compile(r"^ולראיה\s+באו")
_PARTIES = re.compile(r"^(ו?בין)\s*:")
_RECITALS = re.compile(r"^הואיל\b")
# Dotted head: the number may be preceded by a bidi-displaced quote and is
# usually glued straight onto the Hebrew text (" 1.2נקודת רייטינג").
_DOTTED_HEAD = re.compile(r"^[\"']?\s*(\d{1,3}(?:\.\d{1,3})+)(?=\D|$)")
_BARE_HEAD = re.compile(r"^(\d{1,3})[.)](?=\s|[א-ת\"'])")
# pdftotext mirrors a flat clause head inside an RTL line: the LTR run "1."
# comes back as ".1" — dot before digit — once the bidi controls are
# stripped. Dotted heads (1.1) survive because a digits-dot-digits run is
# extracted as one atomic number, so only the bare form needs the mirrored
# pattern; the same strictly-increasing acceptance keeps decimals out.
_BARE_HEAD_REVERSED = re.compile(r"^\.(\d{1,3})(?=\s|[א-ת\"'])")
_SAIF_HEAD = re.compile(r"^סעיף\s+:?\s*(\d{1,3})(?![\d.])")

_PIPE_ROW = re.compile(r"^\|.+\|$")
HEBREW_WORD = re.compile(r"[֐-׿]{3,}")

# Table-run heuristic: this many consecutive short lines, enough of them
# digit-bearing, is a rendered table (labels and cells extract line by line).
_TABLE_RUN_MIN = 4
_TABLE_CELL_MAX_LEN = 35
_TABLE_RUN_MIN_DIGIT_LINES = 3


def hebrew_word_bag(text: str) -> set[str]:
    """Hebrew words (>= 3 letters) as a set: the unit of text comparison.

    Bidi rendering reorders mixed-direction runs but individual Hebrew words
    survive intact, so bag membership is the honest resemblance measure
    between extracted and source text (scripts/trade_corpus_render.py
    learned this the hard way).
    """
    return set(HEBREW_WORD.findall(text))


@dataclass
class _Block:
    clause_id: str
    heading: Optional[str]
    lines: list[tuple[int, str]] = field(default_factory=list)


class _Segmenter:
    def __init__(self) -> None:
        self.blocks: list[_Block] = []
        self.current: Optional[_Block] = None
        self.heading: Optional[str] = None
        self.appendix: Optional[str] = None  # Latin letter once inside a נספח
        self.in_preamble = True
        self.parties_open = False
        self.last_dotted: Optional[tuple[int, ...]] = None
        self.last_bare: Optional[int] = None
        self.pre_count = 0
        self.synth_count = 0
        self.used_ids: set[str] = set()

    # -- block management ---------------------------------------------------

    def _open(self, clause_id: str, page: int, line: str) -> None:
        if clause_id in self.used_ids:
            n = 2
            while f"{clause_id}-{n}" in self.used_ids:
                n += 1
            clause_id = f"{clause_id}-{n}"
        self.used_ids.add(clause_id)
        self.current = _Block(clause_id=clause_id, heading=self.heading)
        self.current.lines.append((page, line))
        self.blocks.append(self.current)

    def _attach(self, page: int, line: str) -> None:
        if self.current is None:
            self._open(self._synthetic_id(), page, line)
        else:
            self.current.lines.append((page, line))

    def _synthetic_id(self) -> str:
        if self.in_preamble:
            self.pre_count += 1
            return f"pre-{self.pre_count}"
        self.synth_count += 1
        return f"c-{self.synth_count:03d}"

    def _open_pre(self, page: int, line: str) -> None:
        self.pre_count += 1
        self._open(f"pre-{self.pre_count}", page, line)

    # -- numbered-head acceptance ------------------------------------------

    def _dotted_ok(self, raw: str, line_rest: str) -> Optional[tuple[int, ...]]:
        parts = raw.split(".")
        if any(len(p) > 1 and p[0] == "0" for p in parts):
            return None  # leading-zero component: a date, never a clause number
        if not re.search(r"[A-Za-zא-ת]", line_rest):
            return None  # number with no text after it: a table cell
        cand = tuple(int(p) for p in parts)
        if self.last_dotted is None:
            return cand if cand[0] <= 3 else None
        if cand > self.last_dotted and cand[0] - self.last_dotted[0] <= 3:
            return cand
        return None

    def _bare_ok(self, n: int) -> bool:
        # Bare "5." heads belong to documents that never use dotted numbering;
        # in a dotted document a bare number at line start is wrap debris.
        if self.last_dotted is not None:
            return False
        expected = 1 if self.last_bare is None else self.last_bare + 1
        return n == expected

    # -- the line dispatcher ------------------------------------------------

    def feed(self, page: int, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return

        m = _CHAPTER.match(stripped)
        if m:
            self.current = None
            self.heading = stripped
            self.appendix = None
            self.in_preamble = False
            return

        m = _APPENDIX_HEADING.match(stripped)
        if m:
            self.current = None
            self.heading = stripped
            self.appendix = self._latin(m.group(1))
            self.in_preamble = False
            return

        m = _APPENDIX_SECTION.match(stripped)
        if m:
            letter = self._latin(m.group(1))
            self.appendix = letter
            self.in_preamble = False
            self._open(f"app{letter}-{int(m.group(2))}", page, stripped)
            return

        m = _APPENDIX_WHOLE.match(stripped)
        if m:
            letter = self._latin(m.group(1))
            self.appendix = letter
            self.in_preamble = False
            self._open(f"app{letter}-1", page, stripped)
            return

        if _SIGNATURE.match(stripped):
            self._open("sig-1", page, stripped)
            return

        if self.in_preamble:
            m = _PARTIES.match(stripped)
            if m:
                if m.group(1) == "ובין" and self.parties_open:
                    self._attach(page, stripped)
                else:
                    self._open_pre(page, stripped)
                    self.parties_open = True
                return
            if _RECITALS.match(stripped):
                self._open_pre(page, stripped)
                self.parties_open = False
                return

        if self.appendix is None:
            m = _DOTTED_HEAD.match(stripped)
            if m:
                cand = self._dotted_ok(m.group(1), stripped[m.end():])
                if cand is not None:
                    self.last_dotted = cand
                    self.in_preamble = False
                    self.parties_open = False
                    self._open(m.group(1), page, stripped)
                    return
            m = (_BARE_HEAD.match(stripped) or _BARE_HEAD_REVERSED.match(stripped)
                 or _SAIF_HEAD.match(stripped))
            if m and self._bare_ok(int(m.group(1))):
                self.last_bare = int(m.group(1))
                self.in_preamble = False
                self.parties_open = False
                self._open(str(int(m.group(1))), page, stripped)
                return

        self._attach(page, stripped)

    @staticmethod
    def _latin(hebrew_letter: str) -> str:
        idx = _HEBREW_ORDER.find(hebrew_letter)
        if idx < 0 or idx > 25:
            return "X"
        return chr(ord("A") + idx)


def _looks_tabular(lines: Sequence[str]) -> bool:
    if any(_PIPE_ROW.match(ln) for ln in lines):
        return True
    run: list[str] = []
    for ln in list(lines) + [""]:
        if ln and len(ln) <= _TABLE_CELL_MAX_LEN:
            run.append(ln)
            continue
        if len(run) >= _TABLE_RUN_MIN:
            digit_lines = sum(1 for cell in run if any(ch.isdigit() for ch in cell))
            if digit_lines >= _TABLE_RUN_MIN_DIGIT_LINES:
                return True
        run = []
    return False


def segment_pages(pages: Sequence[PageText]) -> list[Clause]:
    """Segment normalised page text into clauses. Pure and deterministic.

    Every non-blank input line lands in exactly one clause: as clause text,
    or — for chapter/appendix heading lines — as the ``heading`` carried by
    the clauses that follow them. Ids reuse the document's own numbering
    ("2.1", "appA-2"); blocks before the numbering starts are "pre-N", the
    signature block is "sig-1", anything else synthetic is "c-NNN".
    """
    seg = _Segmenter()
    for pt in pages:
        for line in pt.text.split("\n"):
            seg.feed(pt.number, line)

    clauses: list[Clause] = []
    for block in seg.blocks:
        if not block.lines:
            continue
        text = "\n".join(ln for _, ln in block.lines)
        page_numbers = tuple(sorted({p for p, _ in block.lines}))
        clauses.append(
            Clause(
                clause_id=block.clause_id,
                text=text,
                pages=page_numbers,
                heading=block.heading,
                is_table=_looks_tabular([ln for _, ln in block.lines]),
            )
        )
    return clauses
