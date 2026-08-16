"""The completeness guarantee's data layer.

These tests pin the property the whole trade engine sells: every clause is
accounted for, every extracted term points at its origin, and the structures
refuse silently-dropped work at construction time rather than at review time.
"""

import pytest

from kairos.trade.documents import (
    Citation,
    Clause,
    ClauseDisposition,
    DocumentExtraction,
    TermInstance,
    extraction_from_payload,
)


def _clause(cid="1.1", pages=(1,), text="המפרסם מתחייב לתקציב שנתי"):
    return Clause(clause_id=cid, text=text, pages=tuple(pages))


def _citation(cid="1.1", page=1):
    return Citation(document_id="doc-1", page=page, clause_id=cid, quote="תקציב שנתי בסך 12 מיליון")


def _instance(iid="i-1", cid="1.1", term="budget-commitment"):
    return TermInstance(
        instance_id=iid,
        term_id=term,
        params={"amount": {"amount": 12_000_000, "basis": "gross"}, "period": "year"},
        citations=[_citation(cid=cid)],
        confidence="high",
    )


def _doc(clauses, instances, dispositions):
    return DocumentExtraction(
        document_id="doc-1",
        clauses=clauses,
        instances=instances,
        dispositions=dispositions,
    )


def test_a_complete_document_validates_and_reports_coverage():
    doc = _doc(
        [_clause("1.1"), _clause("9.9", text="חתימות הצדדים")],
        [_instance()],
        [
            ClauseDisposition("1.1", "mapped", instance_ids=("i-1",)),
            ClauseDisposition(
                "9.9", "irrelevant", irrelevant_class="signature-block",
                reason="בלוק חתימות בלבד",
            ),
        ],
    )
    doc.validate()
    cov = doc.coverage()
    assert (cov.total_clauses, cov.mapped, cov.irrelevant, cov.unmapped) == (2, 1, 1, 0)
    assert cov.complete


def test_a_clause_without_a_disposition_fails_loudly():
    doc = _doc([_clause("1.1"), _clause("2.2")], [_instance()],
               [ClauseDisposition("1.1", "mapped", instance_ids=("i-1",))])
    with pytest.raises(ValueError, match="NO disposition"):
        doc.coverage()


def test_an_instance_reachable_from_no_clause_is_an_error():
    doc = _doc(
        [_clause("1.1")],
        [_instance(), _instance(iid="i-orphan")],
        [ClauseDisposition("1.1", "mapped", instance_ids=("i-1",))],
    )
    with pytest.raises(ValueError, match="not reachable"):
        doc.validate()


def test_unmapped_requires_saying_what_the_clause_appears_to_do():
    with pytest.raises(ValueError, match="silence"):
        ClauseDisposition("3.4", "unmapped", reason="  ")


def test_irrelevant_requires_a_closed_class_and_a_reason():
    with pytest.raises(ValueError, match="closed list"):
        ClauseDisposition("9.9", "irrelevant", irrelevant_class="whatever", reason="x")
    with pytest.raises(ValueError, match="without a reason"):
        ClauseDisposition("9.9", "irrelevant", irrelevant_class="severability", reason="")


def test_an_instance_without_citations_cannot_exist():
    with pytest.raises(ValueError, match="origin"):
        TermInstance(
            instance_id="i-2",
            term_id="budget-commitment",
            params={},
            citations=[],
            confidence="high",
        )


def test_a_citation_must_land_on_a_page_the_clause_spans():
    doc = _doc(
        [_clause("1.1", pages=(3, 4))],
        [TermInstance(
            instance_id="i-1",
            term_id="budget-commitment",
            params={},
            citations=[_citation(page=7)],
            confidence="low",
        )],
        [ClauseDisposition("1.1", "mapped", instance_ids=("i-1",))],
    )
    with pytest.raises(ValueError, match="spans pages"):
        doc.validate()


def test_unknown_term_ids_are_refused_at_construction():
    with pytest.raises(KeyError):
        _instance(term="no-such-term")


def test_payload_round_trip_preserves_everything():
    doc = _doc(
        [_clause("1.1"), _clause("5.2", text="סעיף לא מזוהה")],
        [_instance()],
        [
            ClauseDisposition("1.1", "mapped", instance_ids=("i-1",)),
            ClauseDisposition("5.2", "unmapped", reason="נראה כסעיף שיפוי שאינו נתמך"),
        ],
    )
    payload = doc.to_payload()
    assert payload["coverage"]["complete"] is True
    assert payload["coverage"]["unmapped"] == 1
    back = extraction_from_payload(payload)
    assert back.document_id == doc.document_id
    assert [c.clause_id for c in back.clauses] == ["1.1", "5.2"]
    assert back.instances[0].params["period"] == "year"
    assert back.instances[0].citations[0].quote.startswith("תקציב")
