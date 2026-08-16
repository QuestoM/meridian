"""Ingest stage: PDF / image files -> routed, normalised page text.

The first pipeline stage (engine-design §4 stages 1-2). A document arrives as
a PDF (digital or scanned) or as page images; this module hashes it, detects
the route, and for the digital route produces per-page text that is already
safe to segment: bidi control characters stripped, whitespace runs collapsed
(line breaks kept — the segmenter is line-driven), repeating page headers /
footers removed and recorded.

Route detection is deliberately distrustful: a scanned PDF often carries a
junk OCR text layer, so "pdftotext returned something" is not evidence of a
digital document. We require substantial real letters per page AND a low
garbage ratio before trusting the text layer; otherwise the document takes
the scanned-vision route and ships as page images.

Everything here is deterministic subprocess work (poppler's pdftotext /
pdftoppm / pdfinfo); no model is ever called from this module.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

PDFTOTEXT = "/opt/homebrew/bin/pdftotext"
PDFTOPPM = "/opt/homebrew/bin/pdftoppm"
PDFINFO = "/opt/homebrew/bin/pdfinfo"

ROUTE_DIGITAL = "digital"
ROUTE_SCANNED = "scanned-vision"

# pdftotext seeds Hebrew output with directional embedding/isolate controls
# and left/right marks; every downstream comparison lies unless they go.
_BIDI_CONTROLS = re.compile("[‪-‮⁦-⁩‎‏­]")

# Route thresholds: mean real letters per page a digital text layer must
# clear, and the garbage fraction above which a text layer is junk.
MIN_LETTERS_PER_PAGE = 200
MAX_GARBAGE_RATIO = 0.20

# A top/bottom line repeating on at least this fraction of pages is page
# furniture (running header/footer), not document text.
FURNITURE_MIN_PAGE_FRACTION = 0.6
_FURNITURE_EDGE_LINES = 2  # how many lines at each page edge we inspect
_PAGE_NUMBER_LINE = re.compile(r"^\s*(?:-\s*)?\d{1,4}(?:\s*-)?\s*$")

_SUBPROCESS_TIMEOUT = 120


class IngestError(RuntimeError):
    """A document could not be ingested; the message says exactly why."""


@dataclass(frozen=True)
class PageText:
    """One page of normalised extracted text, 1-based."""

    number: int
    text: str


@dataclass
class IngestResult:
    """What the ingest stage hands to segmentation (or to the vision route)."""

    document_sha256: str
    page_count: int
    route: str  # ROUTE_DIGITAL | ROUTE_SCANNED
    pages: list[PageText] = field(default_factory=list)
    page_images: list[Path] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


def strip_bidi_controls(text: str) -> str:
    """Remove directional formatting characters (and soft hyphens)."""
    return _BIDI_CONTROLS.sub("", text)


def normalize_page_text(text: str) -> str:
    """Bidi cleanup + whitespace normalisation for one page.

    Horizontal whitespace runs collapse to a single space and line edges are
    trimmed, but line breaks survive: the segmenter reads the page as lines.
    Blank-line runs collapse to one blank line.
    """
    text = strip_bidi_controls(text)
    lines = [re.sub(r"[ \t ]+", " ", ln).strip() for ln in text.split("\n")]
    out: list[str] = []
    for ln in lines:
        if ln == "" and (not out or out[-1] == ""):
            continue
        out.append(ln)
    while out and out[-1] == "":
        out.pop()
    return "\n".join(out)


def _run(cmd: Sequence[str], what: str) -> subprocess.CompletedProcess:
    try:
        result = subprocess.run(
            list(cmd), capture_output=True, text=False, timeout=_SUBPROCESS_TIMEOUT
        )
    except FileNotFoundError as exc:
        raise IngestError(f"{what}: {cmd[0]} is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise IngestError(f"{what}: timed out after {_SUBPROCESS_TIMEOUT}s") from exc
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", "replace")[-500:]
        raise IngestError(f"{what}: exit {result.returncode}: {stderr}")
    return result


def _pdf_page_count(path: Path) -> int:
    result = _run([PDFINFO, str(path)], f"pdfinfo {path.name}")
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        if line.startswith("Pages:"):
            return int(line.split(":", 1)[1].strip())
    raise IngestError(f"pdfinfo {path.name}: no page count in output")


def _extract_raw_pages(path: Path, page_count: int) -> list[str]:
    """Whole-document pdftotext, split on the form feeds it emits per page."""
    result = _run([PDFTOTEXT, str(path), "-"], f"pdftotext {path.name}")
    raw = result.stdout.decode("utf-8", "replace")
    parts = raw.split("\f")
    if parts and parts[-1].strip() == "":
        parts = parts[:-1]
    if len(parts) < page_count:
        parts += [""] * (page_count - len(parts))
    return parts[:page_count]


def _letters(text: str) -> int:
    return sum(1 for ch in text if ch.isalpha())


def _garbage_ratio(text: str) -> float:
    """Fraction of non-whitespace characters that are junk: replacement
    characters, control/unassigned codepoints. A honest text layer sits
    near zero; OCR debris does not."""
    visible = [ch for ch in text if not ch.isspace()]
    if not visible:
        return 0.0
    junk = sum(
        1
        for ch in visible
        if ch == "�" or (ord(ch) < 32) or (0x80 <= ord(ch) < 0xA0)
    )
    return junk / len(visible)


def detect_route(raw_pages: Sequence[str]) -> tuple[str, dict[str, float]]:
    """Digital only when the text layer is both substantial and clean."""
    if not raw_pages:
        return ROUTE_SCANNED, {"mean_letters_per_page": 0.0, "garbage_ratio": 0.0}
    stripped = [strip_bidi_controls(p) for p in raw_pages]
    mean_letters = sum(_letters(p) for p in stripped) / len(stripped)
    whole = "".join(stripped)
    garbage = _garbage_ratio(whole)
    route = (
        ROUTE_DIGITAL
        if mean_letters > MIN_LETTERS_PER_PAGE and garbage < MAX_GARBAGE_RATIO
        else ROUTE_SCANNED
    )
    return route, {
        "mean_letters_per_page": round(mean_letters, 1),
        "garbage_ratio": round(garbage, 4),
    }


def _edge_lines(page_lines: list[str], top: bool) -> list[tuple[int, str]]:
    """(index, squeezed line) for the first/last non-empty lines of a page."""
    indexed = [(i, ln) for i, ln in enumerate(page_lines) if ln.strip()]
    picked = indexed[:_FURNITURE_EDGE_LINES] if top else indexed[-_FURNITURE_EDGE_LINES:]
    return [(i, re.sub(r"\s+", " ", ln).strip()) for i, ln in picked]


def strip_page_furniture(
    pages: list[list[str]],
) -> tuple[list[list[str]], dict[str, Any]]:
    """Remove running headers/footers and bare page-number lines.

    A line counts as furniture when its squeezed text repeats at a page edge
    on >= FURNITURE_MIN_PAGE_FRACTION of pages (needs >= 3 pages to mean
    anything), or when it is a bare page number sitting at a page edge on
    that fraction of pages. Removals are returned in stats, never silent.
    """
    n = len(pages)
    stats: dict[str, Any] = {"furniture_removed": [], "page_number_lines_removed": 0}
    if n < 3:
        return pages, stats
    threshold = max(2, int(n * FURNITURE_MIN_PAGE_FRACTION + 0.999))

    repeat_counts: dict[tuple[bool, str], int] = {}
    number_edge_pages = {True: 0, False: 0}
    for page_lines in pages:
        for top in (True, False):
            seen_here: set[str] = set()
            for _, squeezed in _edge_lines(page_lines, top):
                if _PAGE_NUMBER_LINE.match(squeezed):
                    number_edge_pages[top] += 1
                elif squeezed not in seen_here:
                    repeat_counts[(top, squeezed)] = repeat_counts.get((top, squeezed), 0) + 1
                    seen_here.add(squeezed)

    furniture = {key for key, count in repeat_counts.items() if count >= threshold}
    strip_numbers = {top for top, cnt in number_edge_pages.items() if cnt >= threshold}

    cleaned: list[list[str]] = []
    for page_lines in pages:
        drop: set[int] = set()
        for top in (True, False):
            for idx, squeezed in _edge_lines(page_lines, top):
                if (top, squeezed) in furniture:
                    drop.add(idx)
                elif top in strip_numbers and _PAGE_NUMBER_LINE.match(squeezed):
                    drop.add(idx)
                    stats["page_number_lines_removed"] += 1
        cleaned.append([ln for i, ln in enumerate(page_lines) if i not in drop])
    stats["furniture_removed"] = sorted({text for _, text in furniture})
    return cleaned, stats


def ingest_pdf(path: Path | str, workdir: Optional[Path] = None) -> IngestResult:
    """Ingest one PDF: hash, route, and either page text or page images.

    ``workdir`` receives the rasterised pages on the scanned route; when not
    given, a sibling directory ``<stem>-pages`` next to the PDF is used.
    """
    pdf = Path(path)
    if not pdf.is_file():
        raise IngestError(f"no such PDF: {pdf}")
    digest = hashlib.sha256(pdf.read_bytes()).hexdigest()
    page_count = _pdf_page_count(pdf)
    raw_pages = _extract_raw_pages(pdf, page_count)
    route, route_stats = detect_route(raw_pages)

    if route == ROUTE_SCANNED:
        out_dir = workdir if workdir is not None else pdf.parent / f"{pdf.stem}-pages"
        images = _rasterize(pdf, Path(out_dir))
        return IngestResult(
            document_sha256=digest,
            page_count=page_count,
            route=route,
            page_images=images,
            stats={"route": route_stats},
        )

    page_lines = [normalize_page_text(p).split("\n") for p in raw_pages]
    cleaned, furniture_stats = strip_page_furniture(page_lines)
    pages = [
        PageText(number=i + 1, text="\n".join(lines).strip("\n"))
        for i, lines in enumerate(cleaned)
    ]
    stats = {"route": route_stats, **furniture_stats}
    return IngestResult(
        document_sha256=digest,
        page_count=page_count,
        route=route,
        pages=pages,
        stats=stats,
    )


def _rasterize(pdf: Path, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("page-*.png"):
        old.unlink()
    _run(
        [PDFTOPPM, "-png", "-gray", "-r", "150", str(pdf), str(out_dir / "page")],
        f"pdftoppm {pdf.name}",
    )
    images = sorted(out_dir.glob("page-*.png"))
    if not images:
        raise IngestError(f"pdftoppm produced no pages for {pdf.name}")
    return images


def ingest_images(paths: Sequence[Path | str]) -> IngestResult:
    """Direct image uploads: one page per image, ordered by filename."""
    files = [Path(p) for p in paths]
    missing = [str(p) for p in files if not p.is_file()]
    if missing:
        raise IngestError(f"missing image files: {missing}")
    if not files:
        raise IngestError("no image files given")
    ordered = sorted(files, key=lambda p: p.name)
    digest = hashlib.sha256()
    for img in ordered:
        digest.update(img.read_bytes())
    return IngestResult(
        document_sha256=digest.hexdigest(),
        page_count=len(ordered),
        route=ROUTE_SCANNED,
        page_images=ordered,
        stats={"source": "images"},
    )
