"""Render corpus agreements to the PDFs (and scan PNGs) the pipeline ingests.

The corpus truth cites (clause, page); this renderer typesets each source.md
into an RTL Hebrew PDF whose pagination MATCHES render.json exactly, then
verifies that claim with pdftotext page by page. A corpus PDF whose real page
boundaries drift from the truth would make every citation-fidelity number a
lie, so --verify is not optional: rendering ends by verifying.

Digital route:  HTML -> Chrome --headless --print-to-pdf -> document.pdf
Scanned route:  the same PDF -> pdftoppm grayscale 130dpi + slight skew ->
                pages/page-N.png (document.pdf is still kept as the source
                of truth for humans; the pipeline ingests the PNGs).

Usage:
  python scripts/trade_corpus_render.py            # all corpus documents
  python scripts/trade_corpus_render.py <doc-id>   # one document
"""

from __future__ import annotations

import hashlib
import html
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kairos.trade.corpus import corpus_root, parse_source_clauses  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
html { direction: rtl; }
body {
  font-family: "Noto Sans Hebrew", "Arial Hebrew", "David", sans-serif;
  font-size: 11pt; line-height: 1.55; color: #111; margin: 0;
}
h1 { font-size: 15pt; text-align: center; margin: 0 0 14pt; }
h2 { font-size: 12pt; margin: 14pt 0 6pt; }
p.clause { margin: 0 0 7pt; text-align: justify; }
section.page { break-after: page; }
section.page:last-child { break-after: auto; }
table { border-collapse: collapse; margin: 6pt auto; direction: rtl; }
td, th { border: 1px solid #444; padding: 3pt 10pt; font-size: 10.5pt; }
.sig { margin-top: 18pt; white-space: pre-wrap; }
"""

_MD_TABLE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")


def _render_clause_text(text: str) -> str:
    """Clause text -> HTML: markdown tables become real tables, the rest
    becomes paragraphs. Escaping first; layout only after."""
    lines = text.splitlines()
    out: list[str] = []
    table: list[list[str]] = []

    def flush_table() -> None:
        nonlocal table
        if not table:
            return
        rows = [r for r in table if not all(set(c) <= {"-", ":", " "} for c in r)]
        cells = "".join(
            "<tr>" + "".join(f"<td>{html.escape(c.strip())}</td>" for c in row) + "</tr>"
            for row in rows
        )
        out.append(f"<table>{cells}</table>")
        table = []

    for line in lines:
        m = _MD_TABLE_ROW.match(line)
        if m:
            table.append(m.group(1).split("|"))
            continue
        flush_table()
        if line.strip():
            out.append(f'<p class="clause">{html.escape(line.strip())}</p>')
    flush_table()
    return "\n".join(out)


def build_html(doc_dir: Path) -> tuple[str, dict[str, int]]:
    """Compose the print HTML; returns (html, clause->page map)."""
    source = (doc_dir / "source.md").read_text(encoding="utf-8")
    render = json.loads((doc_dir / "render.json").read_text(encoding="utf-8"))
    clauses = parse_source_clauses(source)

    page_of: dict[str, int] = {}
    for page_str, ids in render["pages"].items():
        for cid in ids:
            page_of[cid] = int(page_str)

    # Section headings from the raw markdown, attached to the first clause
    # that follows them, so the print keeps the document's visible structure.
    heading_before: dict[str, str] = {}
    title = ""
    pending: list[str] = []
    for chunk in re.split(r"(<!--\s*clause:[^>]+-->)", source):
        m = re.match(r"<!--\s*clause:([^\s>]+)\s*-->", chunk)
        if m:
            cid = m.group(1)
            if pending:
                heading_before[cid] = pending[-1]
                pending = []
            continue
        for line in chunk.splitlines():
            if line.startswith("# ") and not title:
                title = line[2:].strip()
            elif line.startswith("## "):
                pending.append(line[3:].strip())

    pages: dict[int, list[str]] = {}
    for cid in clauses:
        if cid not in page_of:
            raise SystemExit(f"{doc_dir.name}: clause {cid} missing from render.json")
        pages.setdefault(page_of[cid], []).append(cid)

    body: list[str] = []
    for page_no in sorted(pages):
        parts: list[str] = []
        if page_no == 1 and title:
            parts.append(f"<h1>{html.escape(title)}</h1>")
        for cid in pages[page_no]:
            if cid in heading_before:
                parts.append(f"<h2>{html.escape(heading_before[cid])}</h2>")
            klass = "sig" if cid.startswith("sig") else "clause-block"
            parts.append(
                f'<div class="{klass}" data-clause="{html.escape(cid)}">'
                f"{_render_clause_text(clauses[cid])}</div>"
            )
        body.append(f'<section class="page">{"".join(parts)}</section>')

    doc = (
        f'<!doctype html><html lang="he" dir="rtl"><head><meta charset="utf-8">'
        f"<style>{CSS}</style></head><body>{''.join(body)}</body></html>"
    )
    return doc, page_of


def print_pdf(html_text: str, out_pdf: Path) -> None:
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "doc.html"
        src.write_text(html_text, encoding="utf-8")
        result = subprocess.run(
            [
                CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                f"--print-to-pdf={out_pdf}", str(src),
            ],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0 or not out_pdf.exists():
            raise SystemExit(f"chrome print failed: {result.stderr[-800:]}")


def _letters_only(text: str) -> str:
    """Comparison form for extracted-vs-source text: letters and digits only.

    pdftotext seeds Hebrew output with bidi control characters (U+202A-E,
    U+2066-69, U+200E/F) and reorders punctuation runs, so any comparison
    that keeps punctuation or spacing lies. The same lesson applies to the
    ingestion normaliser.
    """
    return "".join(ch for ch in text if ch.isalnum())


def verify_pagination(doc_dir: Path, pdf: Path, page_of: dict[str, int]) -> list[str]:
    """Every clause must have a distinctive text run ON ITS DECLARED PAGE."""
    source = (doc_dir / "source.md").read_text(encoding="utf-8")
    clauses = parse_source_clauses(source)
    problems: list[str] = []
    n_pages = max(page_of.values())
    page_texts: dict[int, str] = {}
    for p in range(1, n_pages + 1):
        res = subprocess.run(
            ["pdftotext", "-f", str(p), "-l", str(p), str(pdf), "-"],
            capture_output=True, text=True, timeout=60,
        )
        page_texts[p] = _letters_only(res.stdout)
    for cid, text in clauses.items():
        # Bidi reorders mixed Hebrew/Latin/digit runs, so contiguous probes
        # lie; individual Hebrew words survive intact, so membership of the
        # clause's word bag on the declared page is the honest check.
        words = re.findall(r"[֐-׿]{3,}", text)
        target = page_of[cid]
        if not words:
            digits = re.findall(r"\d{3,}", text)
            page = page_texts.get(target, "")
            if digits and not all(d in page for d in digits):
                problems.append(
                    f"{doc_dir.name}: clause {cid} digits not on declared "
                    f"page {target}"
                )
            continue
        page = page_texts.get(target, "")
        hits = sum(1 for w in words if w in page)
        if hits / len(words) < 0.8:
            where = [
                p for p, t in page_texts.items()
                if sum(1 for w in words if w in t) / len(words) >= 0.8
            ]
            problems.append(
                f"{doc_dir.name}: clause {cid} only {hits}/{len(words)} words "
                f"on declared page {target} (matches {where or 'no page'})"
            )
    return problems


def rasterize(pdf: Path, out_dir: Path) -> list[Path]:
    """Scanned route: grayscale 130dpi pages with a slight skew."""
    out_dir.mkdir(exist_ok=True)
    for old in out_dir.glob("page-*.png"):
        old.unlink()
    subprocess.run(
        ["pdftoppm", "-png", "-gray", "-r", "130", str(pdf), str(out_dir / "page")],
        check=True, timeout=120,
    )
    from PIL import Image  # noqa: PLC0415

    pages = sorted(out_dir.glob("page-*.png"))
    for img_path in pages:
        img = Image.open(img_path).convert("L").rotate(
            0.5, expand=False, fillcolor=245
        )
        img.save(img_path)
    return pages


def render_document(doc_dir: Path) -> dict:
    html_text, page_of = build_html(doc_dir)
    pdf = doc_dir / "document.pdf"
    print_pdf(html_text, pdf)
    problems = verify_pagination(doc_dir, pdf, page_of)
    if problems:
        raise SystemExit("PAGINATION DRIFT:\n" + "\n".join(problems))
    render = json.loads((doc_dir / "render.json").read_text(encoding="utf-8"))
    entry = {
        "document_id": render["document_id"],
        "pdf": pdf.name,
        "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
        "pages": max(page_of.values()),
        "ingest_route": render.get("ingest_route", "digital"),
    }
    if entry["ingest_route"] == "scanned-vision":
        pages = rasterize(pdf, doc_dir / "pages")
        entry["page_images"] = [p.name for p in pages]
    (doc_dir / "render-manifest.json").write_text(
        json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return entry


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    root = corpus_root()
    rendered = []
    for doc_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if only and doc_dir.name != only:
            continue
        rendered.append(render_document(doc_dir))
        print(f"rendered {doc_dir.name}: {rendered[-1]['pages']} pages "
              f"({rendered[-1]['ingest_route']})")
    if not rendered:
        raise SystemExit(f"no corpus documents matched {only!r}")


if __name__ == "__main__":
    main()
