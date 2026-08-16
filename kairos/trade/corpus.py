"""Corpus loader: ground-truth agreements as first-class extractions.

A corpus document is a directory holding:

- ``source.md``   — the Hebrew agreement text; clause boundaries are marked
  with ``<!-- clause:ID -->`` comments and run until the next marker.
  Markdown headings between clauses are section furniture, not clauses.
- ``truth.json``  — the expected extraction: instances (with clause-level
  citations), a disposition per clause, and expected conflicts.
- ``render.json`` — pagination (clause → page) and rendering hints, so the
  PDF the pipeline ingests and the truth agree about pages.

The loader assembles these into the same DocumentExtraction shape the live
pipeline emits, and VALIDATES the ground truth as strictly as any pipeline
output — a corpus whose own truth fails the completeness machinery would
measure nothing. Quote fidelity is enforced here too: every ground-truth
citation quote must literally appear in its clause's text.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from .documents import (
    Citation,
    Clause,
    ClauseDisposition,
    DocumentExtraction,
    TermInstance,
)
from . import taxonomy_schemas

_CLAUSE_MARK = re.compile(r"<!--\s*clause:([^\s>]+)\s*-->")


def parse_source_clauses(source_text: str) -> dict[str, str]:
    """Split marked source text into {clause_id: clause_text}, order-preserving."""
    parts = _CLAUSE_MARK.split(source_text)
    # parts = [prefix, id1, text1, id2, text2, ...]
    clauses: dict[str, str] = {}
    for i in range(1, len(parts) - 1, 2):
        clause_id = parts[i].strip()
        body = parts[i + 1]
        # Strip markdown headings that belong to the NEXT section, and edges.
        lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith("#")]
        text = "\n".join(lines).strip()
        if clause_id in clauses:
            raise ValueError(f"duplicate clause id {clause_id!r} in source")
        if not text:
            raise ValueError(f"clause {clause_id!r} has no text")
        clauses[clause_id] = text
    if not clauses:
        raise ValueError("source contains no clause markers")
    return clauses


def _page_map(render: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    by_clause: dict[str, list[int]] = {}
    for page_str, clause_ids in render["pages"].items():
        page = int(page_str)
        for cid in clause_ids:
            by_clause.setdefault(cid, []).append(page)
    return {cid: tuple(sorted(pages)) for cid, pages in by_clause.items()}


def load_corpus_document(directory: Path | str) -> DocumentExtraction:
    """Load one corpus directory into a validated DocumentExtraction."""
    root = Path(directory)
    source_text = (root / "source.md").read_text(encoding="utf-8")
    truth = json.loads((root / "truth.json").read_text(encoding="utf-8"))
    render = json.loads((root / "render.json").read_text(encoding="utf-8"))

    clause_texts = parse_source_clauses(source_text)
    pages = _page_map(render)

    missing_pages = sorted(set(clause_texts) - set(pages))
    if missing_pages:
        raise ValueError(f"render.json assigns no page to clauses: {missing_pages}")
    ghost_pages = sorted(set(pages) - set(clause_texts))
    if ghost_pages:
        raise ValueError(f"render.json paginates unknown clauses: {ghost_pages}")

    tables = set(render.get("tables", ()))
    clauses = [
        Clause(
            clause_id=cid,
            text=text,
            pages=pages[cid],
            is_table=cid in tables,
        )
        for cid, text in clause_texts.items()
    ]

    document_id = truth["document_id"]
    instances: list[TermInstance] = []
    for raw in truth["instances"]:
        citations = []
        for cite in raw["cite"]:
            cid = cite["clause_id"]
            quote = cite["quote"]
            clause_text = clause_texts.get(cid)
            if clause_text is None:
                raise ValueError(
                    f"instance {raw['instance_id']} cites unknown clause {cid!r}"
                )
            if quote not in clause_text:
                raise ValueError(
                    f"instance {raw['instance_id']}: quote not found verbatim in "
                    f"clause {cid}: {quote!r}"
                )
            citations.append(
                Citation(
                    document_id=document_id,
                    page=pages[cid][0],
                    clause_id=cid,
                    quote=quote,
                )
            )
        instances.append(
            TermInstance(
                instance_id=raw["instance_id"],
                term_id=raw["term_id"],
                params=raw.get("params", {}),
                citations=citations,
                confidence=raw.get("confidence", "high"),
                scope=raw.get("scope", {}),
                window=raw.get("window", {}),
                missing=list(raw.get("missing", [])),
                notes=raw.get("notes", ""),
            )
        )

    dispositions = [
        ClauseDisposition(
            clause_id=cid,
            disposition=d["disposition"],
            instance_ids=tuple(d.get("instance_ids", ())),
            irrelevant_class=d.get("irrelevant_class"),
            reason=d.get("reason", ""),
        )
        for cid, d in truth["dispositions"].items()
    ]

    doc = DocumentExtraction(
        document_id=document_id,
        clauses=clauses,
        instances=instances,
        dispositions=dispositions,
        ingest_route=render.get("ingest_route", "digital"),
        stats={"corpus": True, "expected_conflicts": truth.get("expected_conflicts", [])},
    )
    doc.validate()
    _validate_truth_params(doc)
    return doc


def _validate_truth_params(doc: DocumentExtraction) -> None:
    """Ground-truth params must satisfy their term schemas' REQUIRED fields,
    except fields the truth explicitly lists as missing (the honest-incomplete
    path is itself corpus-tested)."""
    for inst in doc.instances:
        schema = taxonomy_schemas.schema_for(inst.term_id)
        required = schema.get("required", [])
        for field_name in required:
            if field_name in inst.missing:
                continue
            if field_name not in inst.params:
                raise ValueError(
                    f"ground truth {inst.instance_id} ({inst.term_id}) lacks "
                    f"required param {field_name!r} and does not declare it "
                    "missing"
                )


def corpus_root() -> Path:
    return Path(__file__).resolve().parents[2] / "tests" / "trade_corpus" / "agreements"


def load_all() -> dict[str, DocumentExtraction]:
    root = corpus_root()
    out: dict[str, DocumentExtraction] = {}
    for directory in sorted(p for p in root.iterdir() if p.is_dir()):
        doc = load_corpus_document(directory)
        out[doc.document_id] = doc
    return out
