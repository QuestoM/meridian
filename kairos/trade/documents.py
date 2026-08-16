"""Document and extraction shapes for the trade engine.

The completeness guarantee lives in these types: a SourceDocument is split
into Clauses, every Clause ends the pipeline with exactly one Disposition
(mapped / irrelevant-with-reason / unmapped-flagged), and every extracted
TermInstance carries a Citation back to document, page, clause and exact
text. Nothing here talks to a model provider; the pipeline modules do.

These shapes are also the ground-truth format of the extraction test corpus
(tests/trade_corpus/), so measured accuracy and the live pipeline can never
drift apart on what a "correct extraction" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Mapping, Optional

from . import taxonomy

CONFIDENCE_LEVELS = ("high", "medium", "low")

# The one reserved non-taxonomy label: the model understood the clause to be
# commercial but could not place it. It is a refusal, surfaced loudly, and it
# blocks approval until a human disposes of it.
UNMAPPED = "unmapped"

DISPOSITIONS = ("mapped", "irrelevant", "unmapped")


@dataclass(frozen=True)
class Clause:
    """One segmented clause of a source document.

    ``clause_id`` keeps the document's own numbering when it has one
    (e.g. "5.2.1", "נספח א 3"); synthetic ids are sequential "c-041" forms.
    ``pages`` is every page the clause touches, 1-based.
    """

    clause_id: str
    text: str
    pages: tuple[int, ...]
    heading: Optional[str] = None
    is_table: bool = False

    def __post_init__(self) -> None:
        if not self.clause_id:
            raise ValueError("clause_id must be non-empty")
        if not self.pages:
            raise ValueError(f"clause {self.clause_id} names no pages")


@dataclass(frozen=True)
class Citation:
    """Where an extracted value came from: document, page, clause, exact text."""

    document_id: str
    page: int
    clause_id: str
    quote: str

    def __post_init__(self) -> None:
        if not self.quote.strip():
            raise ValueError(
                f"citation for {self.document_id}/{self.clause_id} has an empty quote"
            )


@dataclass
class TermInstance:
    """One extracted commercial term, still a PROPOSAL until a human approves.

    ``params`` follows taxonomy_schemas.SCHEMAS[term_id]. ``missing`` lists
    required schema fields the document did not yield — an incomplete
    instance is shown honestly, never dropped and never defaulted.
    ``scope``/``window`` are the universal envelope blocks.
    """

    instance_id: str
    term_id: str
    params: dict[str, Any]
    citations: list[Citation]
    confidence: str
    scope: dict[str, Any] = field(default_factory=dict)
    window: dict[str, Any] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        taxonomy.get(self.term_id)  # raises on unknown term
        if self.confidence not in CONFIDENCE_LEVELS:
            raise ValueError(
                f"instance {self.instance_id}: confidence {self.confidence!r} "
                f"not in {CONFIDENCE_LEVELS}"
            )
        if not self.citations:
            raise ValueError(
                f"instance {self.instance_id} ({self.term_id}) carries no citation; "
                "every extracted term must point back to its origin"
            )

    @property
    def incomplete(self) -> bool:
        return bool(self.missing)


@dataclass
class ClauseDisposition:
    """The completeness verdict for one clause. Exactly one per clause.

    - mapped: ``instance_ids`` name the TermInstances built from it.
    - irrelevant: ``irrelevant_class`` is a key of taxonomy.IRRELEVANT_CLASSES
      and ``reason`` says why in a sentence.
    - unmapped: understood-but-not-placed; ``reason`` says what the clause
      appears to do. Loud at review, blocks approval until disposed.
    """

    clause_id: str
    disposition: str
    instance_ids: tuple[str, ...] = ()
    irrelevant_class: Optional[str] = None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.disposition not in DISPOSITIONS:
            raise ValueError(
                f"clause {self.clause_id}: disposition {self.disposition!r} "
                f"not in {DISPOSITIONS}"
            )
        if self.disposition == "mapped" and not self.instance_ids:
            raise ValueError(
                f"clause {self.clause_id} is 'mapped' but names no instances"
            )
        if self.disposition == "irrelevant":
            if self.irrelevant_class not in taxonomy.IRRELEVANT_CLASSES:
                raise ValueError(
                    f"clause {self.clause_id}: irrelevant class "
                    f"{self.irrelevant_class!r} is not in the closed list"
                )
            if not self.reason.strip():
                raise ValueError(
                    f"clause {self.clause_id} marked irrelevant without a reason"
                )
        if self.disposition == "unmapped" and not self.reason.strip():
            raise ValueError(
                f"clause {self.clause_id} is unmapped without saying what it "
                "appears to do; silence is the one unacceptable outcome"
            )


@dataclass
class Coverage:
    """Whole-document completeness state, derived — never hand-set."""

    total_clauses: int
    mapped: int
    irrelevant: int
    unmapped: int

    @property
    def accounted(self) -> int:
        return self.mapped + self.irrelevant + self.unmapped

    @property
    def complete(self) -> bool:
        """Every clause carries a disposition. NOT the approval gate by
        itself: approval additionally requires zero unmapped clauses and a
        human having seen every mapped instance."""
        return self.accounted == self.total_clauses


@dataclass
class DocumentExtraction:
    """The pipeline's full output for one document: the proposal a reviewer sees."""

    document_id: str
    clauses: list[Clause]
    instances: list[TermInstance]
    dispositions: list[ClauseDisposition]
    source_language: str = "he"
    ingest_route: str = "digital"  # digital | scanned-vision
    stats: dict[str, Any] = field(default_factory=dict)

    def coverage(self) -> Coverage:
        by_clause = {d.clause_id: d for d in self.dispositions}
        missing = [c.clause_id for c in self.clauses if c.clause_id not in by_clause]
        if missing:
            raise ValueError(
                f"document {self.document_id}: clauses with NO disposition at "
                f"all: {missing}; every clause must be accounted for"
            )
        extra = set(by_clause) - {c.clause_id for c in self.clauses}
        if extra:
            raise ValueError(
                f"document {self.document_id}: dispositions for unknown "
                f"clauses: {sorted(extra)}"
            )
        counts = {"mapped": 0, "irrelevant": 0, "unmapped": 0}
        for d in self.dispositions:
            counts[d.disposition] += 1
        return Coverage(
            total_clauses=len(self.clauses),
            mapped=counts["mapped"],
            irrelevant=counts["irrelevant"],
            unmapped=counts["unmapped"],
        )

    def validate(self) -> None:
        """Structural integrity beyond per-object checks: citations point at
        real clauses/pages, mapped instances exist, no orphan instances."""
        clause_by_id = {c.clause_id: c for c in self.clauses}
        instance_ids = {i.instance_id for i in self.instances}
        if len(instance_ids) != len(self.instances):
            raise ValueError(f"document {self.document_id}: duplicate instance ids")
        for inst in self.instances:
            for cit in inst.citations:
                clause = clause_by_id.get(cit.clause_id)
                if clause is None:
                    raise ValueError(
                        f"instance {inst.instance_id} cites unknown clause "
                        f"{cit.clause_id!r}"
                    )
                if cit.page not in clause.pages:
                    raise ValueError(
                        f"instance {inst.instance_id} cites page {cit.page} for "
                        f"clause {cit.clause_id}, which spans pages {clause.pages}"
                    )
        referenced: set[str] = set()
        for d in self.dispositions:
            for iid in d.instance_ids:
                if iid not in instance_ids:
                    raise ValueError(
                        f"clause {d.clause_id} maps to unknown instance {iid!r}"
                    )
                referenced.add(iid)
        orphans = instance_ids - referenced
        if orphans:
            raise ValueError(
                f"document {self.document_id}: instances not reachable from any "
                f"clause disposition: {sorted(orphans)}"
            )
        self.coverage()  # raises on undispositioned clauses

    def to_payload(self) -> dict[str, Any]:
        """JSON-safe form for storage and the review API."""
        payload = asdict(self)
        cov = self.coverage()
        payload["coverage"] = asdict(cov)
        payload["coverage"]["complete"] = cov.complete
        return payload


def instances_by_term(
    instances: Iterable[TermInstance],
) -> dict[str, list[TermInstance]]:
    out: dict[str, list[TermInstance]] = {}
    for inst in instances:
        out.setdefault(inst.term_id, []).append(inst)
    return out


def extraction_from_payload(payload: Mapping[str, Any]) -> DocumentExtraction:
    """Inverse of to_payload, validating on the way in."""
    doc = DocumentExtraction(
        document_id=payload["document_id"],
        clauses=[
            Clause(
                clause_id=c["clause_id"],
                text=c["text"],
                pages=tuple(c["pages"]),
                heading=c.get("heading"),
                is_table=bool(c.get("is_table", False)),
            )
            for c in payload["clauses"]
        ],
        instances=[
            TermInstance(
                instance_id=i["instance_id"],
                term_id=i["term_id"],
                params=dict(i["params"]),
                citations=[Citation(**c) for c in i["citations"]],
                confidence=i["confidence"],
                scope=dict(i.get("scope", {})),
                window=dict(i.get("window", {})),
                missing=list(i.get("missing", [])),
                notes=i.get("notes", ""),
            )
            for i in payload["instances"]
        ],
        dispositions=[
            ClauseDisposition(
                clause_id=d["clause_id"],
                disposition=d["disposition"],
                instance_ids=tuple(d.get("instance_ids", ())),
                irrelevant_class=d.get("irrelevant_class"),
                reason=d.get("reason", ""),
            )
            for d in payload["dispositions"]
        ],
        source_language=payload.get("source_language", "he"),
        ingest_route=payload.get("ingest_route", "digital"),
        stats=dict(payload.get("stats", {})),
    )
    doc.validate()
    return doc
