"""Ingest + segmentation round-trip against the trade corpus ground truth.

The corpus is the measure: each rendered document.pdf is ingested and
segmented, and the result is scored against the clause boundaries,
pagination and numbering that kairos.trade.corpus declares. Bars, per the
stage contract: clause count within ±10 percent, >= 90 percent of truth
clauses matched by Hebrew word bag on the right page, matched clauses on
exactly the truth's pages, >= 80 percent of dotted clause ids recovered
verbatim. Failures print the per-document scoreboard so a regression names
its numbers.

Corpus authors add documents concurrently: only the flagship is REQUIRED
to be rendered; other documents are tested when their PDF exists and is
newer than source.md, and skipped by name otherwise.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from kairos.trade.corpus import corpus_root, load_corpus_document
from kairos.trade.ingest import (
    IngestError,
    PageText,
    ROUTE_DIGITAL,
    ROUTE_SCANNED,
    detect_route,
    ingest_images,
    ingest_pdf,
    normalize_page_text,
    strip_bidi_controls,
    strip_page_furniture,
)
from kairos.trade.segment import hebrew_word_bag, segment_pages

FLAGSHIP = "heb-annual-framework-2026"
_DOTTED_ID = re.compile(r"\d+(\.\d+)+")


def _corpus_dirs() -> list[Path]:
    root = corpus_root()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir())


def _renderable(doc_dir: Path) -> str | None:
    """None when the round-trip can run; otherwise the skip/fail reason."""
    pdf = doc_dir / "document.pdf"
    if not pdf.exists():
        return "document.pdf not rendered yet (scripts/trade_corpus_render.py)"
    if pdf.stat().st_mtime < (doc_dir / "source.md").stat().st_mtime:
        return "document.pdf older than source.md — re-render before trusting"
    return None


@pytest.mark.parametrize("doc_name", [p.name for p in _corpus_dirs()])
def test_corpus_round_trip(doc_name: str) -> None:
    doc_dir = corpus_root() / doc_name
    truth = load_corpus_document(doc_dir)
    if truth.ingest_route != ROUTE_DIGITAL:
        pytest.skip(f"{doc_name}: route {truth.ingest_route}, not segmented here")
    reason = _renderable(doc_dir)
    if reason is not None:
        if doc_name == FLAGSHIP:
            pytest.fail(f"{doc_name} is the required corpus document: {reason}")
        pytest.skip(f"{doc_name}: {reason}")

    result = ingest_pdf(doc_dir / "document.pdf")
    assert result.route == ROUTE_DIGITAL, (
        f"{doc_name}: route detection said {result.route} for a digital PDF "
        f"(stats {result.stats.get('route')})"
    )
    mine = segment_pages(result.pages)

    by_page: dict[int, list] = {}
    for clause in mine:
        for page in clause.pages:
            by_page.setdefault(page, []).append(clause)

    matched = 0
    pagination_ok = 0
    unmatched: list[str] = []
    pagination_bad: list[str] = []
    for t in truth.clauses:
        want = hebrew_word_bag(t.text)
        best_score, best_clause = 0.0, None
        for page in t.pages:
            for cand in by_page.get(page, []):
                score = (
                    len(want & hebrew_word_bag(cand.text)) / len(want) if want else 1.0
                )
                if score > best_score or best_clause is None:
                    best_score, best_clause = score, cand
        if best_clause is not None and best_score >= 0.70:
            matched += 1
            if set(best_clause.pages) == set(t.pages):
                pagination_ok += 1
            else:
                pagination_bad.append(
                    f"{t.clause_id}: truth pages {t.pages} vs {best_clause.pages}"
                )
        else:
            unmatched.append(f"{t.clause_id} (best {best_score:.2f})")

    dotted = [c.clause_id for c in truth.clauses if _DOTTED_ID.fullmatch(c.clause_id)]
    mine_ids = {c.clause_id for c in mine}
    ids_recovered = sum(1 for cid in dotted if cid in mine_ids)

    board = (
        f"[{doc_name}] clauses truth={len(truth.clauses)} found={len(mine)} | "
        f"match {matched}/{len(truth.clauses)} "
        f"({100 * matched / len(truth.clauses):.0f}%) | "
        f"id recovery {ids_recovered}/{len(dotted)} | "
        f"pagination {pagination_ok}/{matched}"
    )
    print(board)

    low, high = 0.9 * len(truth.clauses), 1.1 * len(truth.clauses)
    assert low <= len(mine) <= high, (
        f"{board} — clause count outside ±10%: {len(mine)} vs {len(truth.clauses)}"
    )
    assert matched >= 0.90 * len(truth.clauses), (
        f"{board} — unmatched truth clauses: {unmatched}"
    )
    assert not pagination_bad, f"{board} — pagination drift: {pagination_bad}"
    assert ids_recovered >= 0.80 * len(dotted), (
        f"{board} — dotted ids lost: {sorted(set(dotted) - mine_ids)}"
    )


def test_flagship_appendix_table_detected() -> None:
    doc_dir = corpus_root() / FLAGSHIP
    reason = _renderable(doc_dir)
    if reason is not None:
        pytest.fail(f"{FLAGSHIP} is required: {reason}")
    result = ingest_pdf(doc_dir / "document.pdf")
    clauses = {c.clause_id: c for c in segment_pages(result.pages)}
    assert "appA-1" in clauses, f"appA-1 not segmented; got {sorted(clauses)}"
    assert clauses["appA-1"].is_table, "the CPP rate table must carry is_table"
    assert clauses["appA-1"].pages == (9,)
    body_tables = [cid for cid, c in clauses.items() if c.is_table]
    assert body_tables == ["appA-1"], (
        f"only the CPP table is tabular in the flagship, got {body_tables}"
    )


# -- unit: normalisation ----------------------------------------------------


def test_bidi_controls_stripped() -> None:
    dirty = "‫ סעיף ‪1.1‬ ראשון‬‏⁦x⁩"
    assert strip_bidi_controls(dirty) == " סעיף 1.1 ראשוןx"
    assert normalize_page_text(dirty) == "סעיף 1.1 ראשוןx"


def test_normalize_keeps_line_breaks_collapses_runs() -> None:
    raw = "שורה   ראשונה\t\tעם רווחים\n\n\n\nשורה שנייה  \n"
    assert normalize_page_text(raw) == "שורה ראשונה עם רווחים\n\nשורה שנייה"


def test_header_footer_furniture_stripped_and_recorded() -> None:
    pages = [
        ["רשת 13 — סודי ביותר", f"תוכן עמוד מספר {i} עם טקסט", str(i)]
        for i in range(1, 6)
    ]
    cleaned, stats = strip_page_furniture(pages)
    for lines in cleaned:
        assert len(lines) == 1 and lines[0].startswith("תוכן עמוד")
    assert stats["furniture_removed"] == ["רשת 13 — סודי ביותר"]
    assert stats["page_number_lines_removed"] == 5


def test_furniture_needs_repetition_across_pages() -> None:
    pages = [["כותרת חד פעמית", "גוף המסמך הראשון"], ["גוף המסמך השני"]]
    cleaned, stats = strip_page_furniture(pages)
    assert cleaned == pages  # under 3 pages: nothing is furniture
    assert stats["furniture_removed"] == []


# -- unit: route detection --------------------------------------------------


def test_route_detection_heuristics() -> None:
    hebrew = "הסוכנות מתחייבת להיקף רכישות כולל של ארבעה עשר מיליון שקלים " * 10
    route, stats = detect_route([hebrew] * 4)
    assert route == ROUTE_DIGITAL and stats["mean_letters_per_page"] > 200

    route, _ = detect_route(["אב", "", "12 34"])  # near-empty text layer
    assert route == ROUTE_SCANNED

    junk = ("אבג" + "\x01\x02\x03�" * 40) * 30  # substantial but garbage-heavy
    route, stats = detect_route([junk] * 3)
    assert route == ROUTE_SCANNED and stats["garbage_ratio"] > 0.2

    assert detect_route([])[0] == ROUTE_SCANNED


def test_image_only_pdf_routes_to_scanned(tmp_path: Path) -> None:
    src = corpus_root() / FLAGSHIP / "document.pdf"
    if not src.exists():
        pytest.fail(f"{FLAGSHIP} must be rendered for the scanned-route test")
    subprocess.run(
        ["/opt/homebrew/bin/pdftoppm", "-png", "-gray", "-r", "80",
         "-f", "1", "-l", "1", str(src), str(tmp_path / "raster")],
        check=True, timeout=120,
    )
    png = next(tmp_path.glob("raster*.png"))
    scanned_pdf = tmp_path / "scanned.pdf"
    Image.open(png).save(scanned_pdf)  # image-only PDF: no text layer at all

    result = ingest_pdf(scanned_pdf, workdir=tmp_path / "out")
    assert result.route == ROUTE_SCANNED, result.stats
    assert result.page_count == 1
    assert [p.name for p in result.page_images] == ["page-1.png"]
    assert result.pages == []


def test_ingest_images_orders_by_filename(tmp_path: Path) -> None:
    for name in ("page-3.png", "page-1.png", "page-2.png"):
        Image.new("L", (40, 60), 240).save(tmp_path / name)
    result = ingest_images(
        [tmp_path / "page-3.png", tmp_path / "page-1.png", tmp_path / "page-2.png"]
    )
    assert result.route == ROUTE_SCANNED
    assert result.page_count == 3
    assert [p.name for p in result.page_images] == [
        "page-1.png", "page-2.png", "page-3.png",
    ]
    assert len(result.document_sha256) == 64


def test_ingest_errors_are_loud(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="no such PDF"):
        ingest_pdf(tmp_path / "missing.pdf")
    with pytest.raises(IngestError, match="no image files"):
        ingest_images([])
    with pytest.raises(IngestError, match="missing image files"):
        ingest_images([tmp_path / "nope.png"])


# -- unit: segmentation on synthetic text -----------------------------------


def _page(number: int, *lines: str) -> PageText:
    return PageText(number=number, text="\n".join(lines))


def test_preamble_chapters_and_numbering() -> None:
    clauses = segment_pages([
        _page(
            1,
            "הסכם מסגרת לדוגמה",
            "הסכם שנערך ונחתם בתל אביב ביום 01.06.2026",
            'בין :רשת 13 בע"מ (להלן: "הערוץ")',
            'ובין :סוכנות הדגמה בע"מ (להלן: "הסוכנות")',
            "הואיל והצדדים מעוניינים בכך; הוסכם כדלקמן:",
            "פרק א' — הגדרות",
            "1.1בהסכם זה למונחים המשמעות שלצידם והוראות נוספות",
            "1.2תוקפו של ההסכם שנה אחת מיום החתימה",
        ),
        _page(
            2,
            "פרק ב' — תמורה",
            "2.1התמורה תשולם בתנאי שוטף פלוס שישים",
            "והמשך ישיר של אותו סעיף בשורה נגררת",
            "ולראיה באו הצדדים על החתום:",
            "הערוץ: ____ הסוכנות: ____",
        ),
    ])
    ids = [c.clause_id for c in clauses]
    assert ids == ["pre-1", "pre-2", "pre-3", "1.1", "1.2", "2.1", "sig-1"]
    by_id = {c.clause_id: c for c in clauses}
    assert "ובין" in by_id["pre-2"].text  # ובין continues the parties block
    assert by_id["1.1"].heading == "פרק א' — הגדרות"
    assert by_id["2.1"].heading == "פרק ב' — תמורה"
    assert "נגררת" in by_id["2.1"].text  # stray line attaches to previous clause
    assert by_id["2.1"].pages == (2,)
    assert by_id["sig-1"].text.endswith("____")


def test_false_heads_rejected() -> None:
    clauses = segment_pages([
        _page(
            1,
            "1.1סעיף ראשון עם תוכן רגיל בעברית",
            "01.02.2026ועד סוף התקופה הנדונה",  # date wrap: leading-zero component
            "1.15",  # bare decimal cell: no letters after the number
            "2.1סעיף שני שמתחיל כדין אחרי הראשון",
            "1.1כאמור לעיל ההפניה חוזרת אחורה",  # backwards reference at line start
        ),
    ])
    assert [c.clause_id for c in clauses] == ["1.1", "2.1"]
    by_id = {c.clause_id: c for c in clauses}
    assert "01.02.2026" in by_id["1.1"].text and "1.15" in by_id["1.1"].text
    assert "ההפניה" in by_id["2.1"].text


def test_appendix_ids_and_dotted_disabled_inside() -> None:
    clauses = segment_pages([
        _page(
            1,
            "1.1גוף ההסכם קובע הוראה כלשהי לצורך הדוגמה",
            "נספח ב' — הוראות נוספות",
            "נספח ב' סעיף 1: הוראה ראשונה בנספח",
            "נספח ב' סעיף :2הוראה שנייה עם נקודתיים שזזו",
            "2.1שורה שנגררה לתחילת שורה בתוך הנספח",
            "נספח ג' :נספח שלם ללא סעיפים פנימיים",
        ),
    ])
    ids = [c.clause_id for c in clauses]
    assert ids == ["1.1", "appB-1", "appB-2", "appC-1"]
    by_id = {c.clause_id: c for c in clauses}
    assert by_id["appB-1"].heading == "נספח ב' — הוראות נוספות"
    assert "שנגררה" in by_id["appB-2"].text  # dotted heads never fire in a נספח


def test_bare_numbering_document() -> None:
    clauses = segment_pages([
        _page(
            1,
            "הסכם פרסום קצר",
            "1. תוקף ההסכם שישה חודשים מיום החתימה",
            "2. המחיר לתשדיר בודד קבוע וסופי",
            "30. שניות הוא אורך התשדיר המוסכם",  # 30 != 3: sequence rejects
            "3. תנאי התשלום שוטף שלושים",
        ),
    ])
    ids = [c.clause_id for c in clauses]
    assert ids == ["pre-1", "1", "2", "3"]
    by_id = {c.clause_id: c for c in clauses}
    assert "30. שניות" in by_id["2"].text


def test_pipe_table_marked_is_table() -> None:
    clauses = segment_pages([
        _page(
            1,
            "1.1התעריפים יהיו כמפורט בטבלה שלהלן:",
            "| רצועה | מחיר |",
            "|---|---|",
            "| פריים | 1,450 |",
            "1.2סעיף רגיל שאינו טבלה כלל ועיקר",
        ),
    ])
    by_id = {c.clause_id: c for c in clauses}
    assert by_id["1.1"].is_table
    assert not by_id["1.2"].is_table


def test_clause_spanning_pages_carries_both() -> None:
    clauses = segment_pages([
        _page(1, "1.1סעיף שמתחיל בעמוד הראשון ונמשך"),
        _page(2, "ההמשך של אותו סעיף בעמוד השני", "1.2סעיף חדש בעמוד השני"),
    ])
    by_id = {c.clause_id: c for c in clauses}
    assert by_id["1.1"].pages == (1, 2)
    assert by_id["1.2"].pages == (2,)


def test_segment_empty_input() -> None:
    assert segment_pages([]) == []
