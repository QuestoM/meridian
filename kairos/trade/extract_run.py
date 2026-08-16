"""The extraction runner: clauses in, a complete honest proposal out.

``extract_document`` is the assembly core: it classifies every clause,
parameterises every (clause × term), resolves cross-references, merges
pointer-and-content pairs into one instance, detects conflicts through the
precedence algebra, and returns a validated DocumentExtraction in which every
clause is accounted for. A model failure on one unit of work becomes an
honest artifact (an incomplete instance or an unmapped clause naming the
failure) — the run never dies of one clause and never drops one.

``run_pdf`` wraps it with ingest + segmentation (+ vision transcription for
the scanned route) for the live path.
"""

from __future__ import annotations

import re
import time
from typing import Any, Callable, Optional

# The two reference postures (see the merge loop): a pointer defers to the
# referenced clause's content; an override contradicts it deliberately.
_POINTER_RE = re.compile(r"כמפורט|מפורטים|מפורט בנספח|המפורט|יפורטו|בהתאם לנספח|כאמור בנספח")
_OVERRIDE_RE = re.compile(r"על אף|חרף|בסתירה ל")

from . import precedence
from .documents import (
    Clause,
    Citation,
    ClauseDisposition,
    DocumentExtraction,
    TermInstance,
    UNMAPPED,
)
from .extract_stages import (
    CallFn,
    classify_clauses,
    parameterise,
    referenced_clause_ids,
    transcribe_page,
)
from . import taxonomy, taxonomy_schemas


def _clause_head_citation(document_id: str, clause: Clause) -> Citation:
    head = clause.text.strip().splitlines()[0][:80] or clause.text[:80]
    return Citation(
        document_id=document_id, page=clause.pages[0],
        clause_id=clause.clause_id, quote=head,
    )


def _definitions_note(instances: list[TermInstance]) -> str:
    lines: list[str] = []
    for inst in instances:
        if inst.term_id != "definitions":
            continue
        for entry in inst.params.get("entries", []):
            term = str(entry.get("term", "")).strip()
            definition = str(entry.get("definition", "")).strip()
            if term and definition:
                lines.append(f"- {term}: {definition}")
    return "\n".join(lines)


def extract_document(
    document_id: str,
    clauses: list[Clause],
    call: CallFn,
    *,
    ingest_route: str = "digital",
    agreement_id: str = "",
    progress: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> DocumentExtraction:
    started = time.monotonic()

    def _progress(stage: str, detail: dict[str, Any]) -> None:
        if progress is not None:
            progress(stage, detail)

    by_id = {c.clause_id: c for c in clauses}
    order = [c.clause_id for c in clauses]

    # ---- classify -------------------------------------------------------
    _progress("classify", {"clauses": len(clauses)})
    labels = classify_clauses(clauses, call)

    # ---- group pointer clauses with the content they reference ----------
    refs: dict[str, list[str]] = {
        cid: [r for r in referenced_clause_ids(by_id[cid].text, order) if r != cid]
        for cid in order
    }
    # merged_into[a] = b means clause a's instance for a shared label lives on b.
    # A POINTER ("כמפורט בנספח א'") merges: one term, stated once, referenced
    # once. An OVERRIDE ("על אף האמור בסעיף 2.2") must NOT merge — it is a
    # second, contradicting instance the precedence algebra exists to judge.
    merged_into: dict[tuple[str, str], str] = {}
    for cid in order:
        text = by_id[cid].text
        if _OVERRIDE_RE.search(text) or not _POINTER_RE.search(text):
            continue
        my_terms = {l for l in labels.get(cid, {}).get("labels", [])
                    if l in taxonomy.TERMS}
        for ref in refs.get(cid, []):
            ref_terms = {l for l in labels.get(ref, {}).get("labels", [])
                         if l in taxonomy.TERMS}
            for term in my_terms & ref_terms:
                # The referenced clause holds the content; the referring
                # clause becomes a co-citation of the same instance.
                merged_into[(cid, term)] = ref

    # ---- parameterise ---------------------------------------------------
    instances: list[TermInstance] = []
    clause_instances: dict[str, list[str]] = {cid: [] for cid in order}
    failures: list[dict[str, str]] = []
    counter = 0

    def _neighbours(cid: str) -> list[Clause]:
        index = order.index(cid)
        out = []
        if index > 0:
            out.append(by_id[order[index - 1]])
        if index + 1 < len(order):
            out.append(by_id[order[index + 1]])
        return out

    # Definitions and meta first, so later calls get the document's own words.
    def _priority(item: tuple[str, str]) -> int:
        return 0 if taxonomy.get(item[1]).family == "A" else 1

    work: list[tuple[str, str]] = []
    for cid in order:
        for label in labels.get(cid, {}).get("labels", []):
            if label in taxonomy.TERMS and (cid, label) not in merged_into:
                work.append((cid, label))
    work.sort(key=_priority)

    definitions_note = ""
    for cid, term_id in work:
        clause = by_id[cid]
        counter += 1
        instance_id = f"x-{counter:03d}"
        referenced = [by_id[r] for r in refs.get(cid, []) if r in by_id]
        co_citers = [
            other for (other, term), target in merged_into.items()
            if target == cid and term == term_id
        ]
        _progress("parameterise", {"clause": cid, "term": term_id})
        try:
            result = parameterise(
                clause, term_id, call,
                neighbours=_neighbours(cid) + [by_id[c] for c in co_citers],
                referenced=referenced,
                definitions_note=definitions_note,
            )
            citations = [
                Citation(document_id=document_id, page=by_id[q["clause_id"]].pages[0],
                         clause_id=q["clause_id"], quote=q["quote"])
                for q in result["quotes"]
                if q["clause_id"] in by_id
            ] or [_clause_head_citation(document_id, clause)]
            for co in co_citers:
                citations.append(_clause_head_citation(document_id, by_id[co]))
            inst = TermInstance(
                instance_id=instance_id,
                term_id=term_id,
                params=result["params"],
                citations=citations,
                confidence=result["confidence"],
                scope=result["scope"],
                window=result["window"],
                missing=result["missing"],
                notes=result["notes"],
            )
        except Exception as exc:  # noqa: BLE001 - one failure, one honest artifact
            failures.append({"clause_id": cid, "term_id": term_id,
                             "error": type(exc).__name__})
            required = list(taxonomy_schemas.schema_for(term_id).get("required", []))
            inst = TermInstance(
                instance_id=instance_id,
                term_id=term_id,
                params={},
                citations=[_clause_head_citation(document_id, clause)],
                confidence="low",
                missing=required,
                notes=f"החילוץ נכשל ({type(exc).__name__}); נדרשת השלמה ידנית",
            )
        instances.append(inst)
        clause_instances[cid].append(instance_id)
        for co in co_citers:
            clause_instances[co].append(instance_id)
        if term_id == "definitions":
            definitions_note = _definitions_note(instances)

    # ---- dispositions ---------------------------------------------------
    dispositions: list[ClauseDisposition] = []
    for cid in order:
        entry = labels.get(cid, {"labels": [UNMAPPED], "note": ""})
        note = str(entry.get("note", ""))
        if clause_instances[cid]:
            dispositions.append(ClauseDisposition(
                clause_id=cid, disposition="mapped",
                instance_ids=tuple(clause_instances[cid]),
            ))
            continue
        irrelevant = [l for l in entry["labels"] if l.startswith("irrelevant:")]
        if irrelevant:
            key = irrelevant[0].split(":", 1)[1]
            dispositions.append(ClauseDisposition(
                clause_id=cid, disposition="irrelevant",
                irrelevant_class=key,
                reason=note or taxonomy.IRRELEVANT_CLASSES.get(key, key),
            ))
            continue
        dispositions.append(ClauseDisposition(
            clause_id=cid, disposition="unmapped",
            reason=note or "בעל תוכן מסחרי שלא סווג לאף מונח נתמך",
        ))

    # ---- conflicts through the precedence algebra ----------------------
    candidates = []
    for inst in instances:
        region = precedence.region_of_clause(inst.citations[0].clause_id)
        candidates.append(precedence.Candidate(
            instance_id=inst.instance_id,
            term_id=inst.term_id,
            params=inst.params,
            scope=inst.scope,
            window=inst.window,
            provenance=precedence.Provenance(
                agreement_id=agreement_id or document_id,
                level="advertiser",
                document_id=document_id,
                region=region,
                effective_date="",
            ),
        ))
    edges = precedence.edges_from_precedence_instances(
        [{"term_id": i.term_id, "params": i.params} for i in instances],
        agreement_id=agreement_id or document_id,
        document_id=document_id,
    )
    conflicts = [c.to_payload() for c in precedence.detect_and_resolve(candidates, edges)]

    extraction = DocumentExtraction(
        document_id=document_id,
        clauses=clauses,
        instances=instances,
        dispositions=dispositions,
        ingest_route=ingest_route,
        stats={
            "elapsed_seconds": round(time.monotonic() - started, 2),
            "work_units": len(work),
            "failures": failures,
            "conflicts": conflicts,
        },
    )
    extraction.validate()
    return extraction


# ---------------------------------------------------------------- live wrapper

def run_pdf(
    path: Any,
    caller: Any,
    *,
    document_id: str,
    agreement_id: str = "",
    progress: Optional[Callable[[str, dict[str, Any]], None]] = None,
) -> DocumentExtraction:
    """Ingest a PDF (or image set) and extract it end to end.

    ``caller`` is a StageCaller (extract_provider). The scanned route
    transcribes each page through the vision tier first and then follows the
    same segmentation path, so the completeness contract holds identically.
    """
    from . import ingest as ingest_mod
    from . import segment as segment_mod
    from .extract_provider import image_block

    result = ingest_mod.ingest_pdf(path)
    if progress is not None:
        progress("ingest", {"route": result.route, "pages": result.page_count})
    if result.route == "scanned-vision":
        pages = []
        for i, image_path in enumerate(result.page_images, start=1):
            transcribed = transcribe_page(
                image_block(open(image_path, "rb").read()), i, caller.call)
            pages.append(ingest_mod.PageText(number=i, text=str(transcribed.get("text", ""))))
            if progress is not None:
                progress("transcribe", {"page": i})
    else:
        pages = result.pages
    clauses = segment_mod.segment_pages(pages)
    if progress is not None:
        progress("segment", {"clauses": len(clauses)})
    extraction = extract_document(
        document_id, clauses, caller.call,
        ingest_route=result.route,
        agreement_id=agreement_id,
        progress=progress,
    )
    extraction.stats["provider"] = caller.stats.to_payload()
    extraction.stats["document_sha256"] = result.document_sha256
    return extraction
