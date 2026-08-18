"""The agreement lifecycle: create → attach → extract → review → approve.

The property under test is the mission's hard rule: nothing is approvable
while any clause is unseen, any proposal undecided, any unmapped clause
unacknowledged, or any conflict open — and the gate is server truth that
names its blockers, not a UI state.
"""

import pytest

from kairos_api import trade_review, trade_store


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv(trade_store.AGREEMENTS_DIR_ENV, str(tmp_path / "agreements"))
    return tmp_path


def _extraction_payload(doc_id="doc-x"):
    return {
        "document_id": doc_id,
        "clauses": [
            {"clause_id": "1.1", "text": "המפרסם מתחייב לתקציב שנתי בסך 5,000,000 ₪ במונחי מחירון", "pages": [1]},
            {"clause_id": "2.1", "text": "עמלת הסוכנות 15% מהנטו", "pages": [1]},
            {"clause_id": "3.1", "text": "הסוכנות רשאית למכור מלאי לצד שלישי", "pages": [2]},
            {"clause_id": "9.9", "text": "ולראיה באו הצדדים על החתום", "pages": [2]},
        ],
        "instances": [
            {
                "instance_id": "i-budget",
                "term_id": "budget-commitment",
                "params": {"amount": {"amount": 5_000_000, "basis": "ratecard"}, "period": "year"},
                "citations": [{"document_id": doc_id, "page": 1, "clause_id": "1.1", "quote": "5,000,000 ₪"}],
                "confidence": "high",
            },
            {
                "instance_id": "i-commission",
                "term_id": "agency-commission",
                "params": {"percent": 15, "base": "net_of_discount", "form": "invoice_deduction"},
                "citations": [{"document_id": doc_id, "page": 1, "clause_id": "2.1", "quote": "עמלת הסוכנות 15%"}],
                "confidence": "medium",
            },
        ],
        "dispositions": [
            {"clause_id": "1.1", "disposition": "mapped", "instance_ids": ["i-budget"]},
            {"clause_id": "2.1", "disposition": "mapped", "instance_ids": ["i-commission"]},
            {"clause_id": "3.1", "disposition": "unmapped", "reason": "נראה כזכות מכירה משנית שאין לה מונח נתמך"},
            {"clause_id": "9.9", "disposition": "irrelevant", "irrelevant_class": "signature-block", "reason": "חתימות"},
        ],
    }


def _agreement_in_review(actor="dana"):
    head = trade_store.create(
        title="הסכם בדיקה", level="advertiser", actor=actor,
        window={"starts_on": "2026-01-01", "ends_on": "2026-12-31"},
    )
    aid = head["agreement_id"]
    doc = trade_store.attach_document(
        aid, filename="deal.pdf", payload=b"%PDF-1.4 fake", actor=actor
    )
    trade_store.save_extraction(aid, doc["document_id"], _extraction_payload(doc["document_id"]), actor)
    trade_store.set_status(aid, trade_store.IN_REVIEW, actor)
    return aid, doc["document_id"]


def test_gate_names_every_blocker_and_approval_refuses(store):
    aid, doc_id = _agreement_in_review()
    gate = trade_review.agreement_gate(aid)
    kinds = {b["kind"] for b in gate["blockers"]}
    assert kinds == {"clauses_unseen", "instances_undecided", "unmapped_unacknowledged"}
    assert not gate["ready"]
    with pytest.raises(ValueError, match="completeness gate"):
        trade_review.approve(aid, actor="dana")


def test_full_review_path_reaches_an_approved_version(store):
    aid, doc_id = _agreement_in_review()
    trade_review.mark_clauses_seen(aid, doc_id, ["1.1", "2.1", "3.1", "9.9"], "dana")
    trade_review.decide_instance(aid, doc_id, "i-budget", "confirmed", "dana")
    trade_review.decide_instance(
        aid, doc_id, "i-commission", "edited", "dana",
        edited_params={"percent": 12, "base": "net_of_discount", "form": "invoice_deduction"},
    )
    trade_review.acknowledge_unmapped(
        aid, doc_id, "3.1", "dana", note="זכות מכירה משנית — מנוהל מחוץ למערכת, באחריות סמנכ\"ל מסחר"
    )
    added = trade_review.add_reviewer_instance(
        aid, doc_id,
        term_id="payment-terms",
        params={"terms": "שוטף + 45"},
        actor="dana",
        clause_id="2.1",
        quote="עמלת הסוכנות",
        note="נוסף ידנית לצורך בדיקה",
    )
    gate = trade_review.agreement_gate(aid)
    assert gate["ready"], gate["blockers"]

    manifest = trade_review.approve(aid, actor="dana", note="אישור ראשון")
    assert manifest["counts"]["approved_terms"] == 3  # budget + edited commission + added
    assert manifest["counts"]["acknowledged_unsupported"] == 1

    head = trade_store.load_head(aid)
    assert head["status"] == trade_store.APPROVED
    assert head["current_version_id"] == manifest["version_id"]

    termset = trade_store.load_termset(aid, manifest["version_id"])
    by_id = {i["instance_id"]: i for i in termset["instances"]}
    # The reviewer's edit wins in the termset, and the extraction's original
    # value stays visible beside it.
    assert by_id["i-commission"]["params"]["percent"] == 12
    assert by_id["i-commission"]["review"]["extracted_params"]["percent"] == 15
    assert by_id[added["instance_id"]]["review"]["state"] == "reviewer_added"
    assert termset["acknowledged_unsupported"][0]["clause_id"] == "3.1"


def test_rejection_requires_a_reason_and_excludes_from_the_termset(store):
    aid, doc_id = _agreement_in_review()
    trade_review.mark_clauses_seen(aid, doc_id, ["1.1", "2.1", "3.1", "9.9"], "dana")
    with pytest.raises(ValueError, match="reason"):
        trade_review.decide_instance(aid, doc_id, "i-budget", "rejected", "dana")
    trade_review.decide_instance(aid, doc_id, "i-budget", "rejected", "dana", reason="סעיף בוטל בנספח")
    trade_review.decide_instance(aid, doc_id, "i-commission", "confirmed", "dana")
    trade_review.acknowledge_unmapped(aid, doc_id, "3.1", "dana", note="לא נתמך, מטופל ידנית")
    manifest = trade_review.approve(aid, actor="dana")
    termset = trade_store.load_termset(aid, manifest["version_id"])
    ids = {i["instance_id"] for i in termset["instances"]}
    assert "i-budget" not in ids
    assert termset["rejected"][0]["reason"] == "סעיף בוטל בנספח"


def test_reviewer_added_term_must_cite_or_declare_not_in_document(store):
    aid, doc_id = _agreement_in_review()
    with pytest.raises(ValueError, match="not_in_document"):
        trade_review.add_reviewer_instance(
            aid, doc_id, term_id="payment-terms", params={"terms": "שוטף+30"},
            actor="dana",
        )
    with pytest.raises(ValueError, match="verbatim"):
        trade_review.add_reviewer_instance(
            aid, doc_id, term_id="payment-terms", params={"terms": "שוטף+30"},
            actor="dana", clause_id="2.1", quote="טקסט שאינו שם",
        )
    record = trade_review.add_reviewer_instance(
        aid, doc_id, term_id="payment-terms", params={"terms": "שוטף+30"},
        actor="dana", not_in_document=True, note="סוכם בעל פה בפגישה",
    )
    assert record["not_in_document"] is True


def test_status_machine_refuses_illegal_moves_and_documents_freeze(store):
    aid, doc_id = _agreement_in_review()
    with pytest.raises(ValueError, match="cannot become"):
        trade_store.set_status(aid, trade_store.SUPERSEDED, "dana")
    trade_review.mark_clauses_seen(aid, doc_id, ["1.1", "2.1", "3.1", "9.9"], "dana")
    trade_review.decide_instance(aid, doc_id, "i-budget", "confirmed", "dana")
    trade_review.decide_instance(aid, doc_id, "i-commission", "confirmed", "dana")
    trade_review.acknowledge_unmapped(aid, doc_id, "3.1", "dana", note="מנוהל ידנית")
    trade_review.approve(aid, actor="dana")
    # An amendment arriving on an APPROVED agreement is legal - that is how
    # appendices and mid-flight changes arrive in this market - and it sends
    # the agreement back to review while the approved version keeps governing.
    late = trade_store.attach_document(
        aid, filename="amendment.pdf", payload=b"%PDF-1.4 amendment", actor="dana",
    )
    head = trade_store.load_head(aid)
    assert head["status"] == trade_store.IN_REVIEW
    assert head["current_version_id"], "the approved version keeps governing until superseded"
    assert any(d["document_id"] == late["document_id"] for d in head["documents"])


def test_non_pdf_uploads_are_refused_by_content_not_filename(store):
    head = trade_store.create(
        title="הסכם בדיקה", level="advertiser", actor="dana",
        window={"starts_on": "2026-01-01", "ends_on": "2026-12-31"},
    )
    aid = head["agreement_id"]
    # An xlsx renamed to .pdf is still an xlsx: refusal reads the bytes.
    xlsx_magic = b"PK\x03\x04 not a pdf at all"
    with pytest.raises(ValueError, match="PDF"):
        trade_store.attach_document(aid, filename="deal.pdf", payload=xlsx_magic, actor="dana")
    with pytest.raises(ValueError, match="PDF"):
        trade_store.attach_document(aid, filename="deal.xlsx", payload=xlsx_magic, actor="dana")
    assert trade_store.load_head(aid).get("documents", []) == []


def test_a_new_extraction_resets_review_and_archives_the_old(store):
    aid, doc_id = _agreement_in_review()
    trade_review.mark_clauses_seen(aid, doc_id, ["1.1"], "dana")
    trade_store.save_extraction(aid, doc_id, _extraction_payload(doc_id), "dana")
    review = trade_store.load_review(aid, doc_id)
    assert review["clauses_seen"] == {}
    archived = list(
        (trade_store.agreements_root() / aid / "review").glob("*.superseded.json")
    )
    assert len(archived) == 1


def test_unknown_edit_parameters_are_refused(store):
    aid, doc_id = _agreement_in_review()
    trade_review.mark_clauses_seen(aid, doc_id, ["1.1", "2.1", "3.1", "9.9"], "dana")
    with pytest.raises(ValueError, match="does not take parameters"):
        trade_review.decide_instance(
            aid, doc_id, "i-commission", "edited", "dana",
            edited_params={"percent": 12, "made_up_field": True},
        )


def _with_an_empty_reading(doc_id="doc-x"):
    """The same document, plus a reading that carries the shape of a term and
    nothing in it — the thing that used to sit in the list a person approves."""
    payload = _extraction_payload(doc_id)
    payload["clauses"].append(
        {"clause_id": "4.1", "text": "ההנחות ייקבעו בהמשך בהתאם להיקף שיסוכם", "pages": [2]})
    payload["instances"].append({
        "instance_id": "i-hollow",
        "term_id": "volume-discount-ladder",
        "params": {"tiers": [{"threshold": 0, "discount_percent": 0}],
                   "basis": "unstated", "mechanics": "unstated", "period": "campaign"},
        "citations": [{"document_id": doc_id, "page": 2, "clause_id": "4.1",
                       "quote": "ההנחות ייקבעו בהמשך"}],
        "confidence": "low",
    })
    payload["dispositions"].append(
        {"clause_id": "4.1", "disposition": "mapped", "instance_ids": ["i-hollow"]})
    return payload


def test_a_reading_with_no_values_does_not_hold_the_gate_shut(store):
    """It is on the screen, in its own list, and it blocks nothing.

    The completeness guarantee is untouched by this: the clause still carries a
    disposition and is still counted, because what moved is which LIST a term
    sits in, never whether the clause was accounted for.
    """
    actor = "dana"
    aid, doc = _agreement_in_review(actor)
    trade_store.save_extraction(aid, doc, _with_an_empty_reading(doc), actor)

    gate = trade_review.document_gate(aid, doc)
    assert gate["instances_interpretive"] == 1
    undecided = next(b for b in gate["blockers"] if b["kind"] == "instances_undecided")
    assert "i-hollow" not in undecided["ids"], "an empty reading is holding the gate shut"
    assert set(undecided["ids"]) == {"i-budget", "i-commission"}
    # every clause is still accounted for, which is the guarantee that matters
    assert gate["clauses_total"] == 5
    assert sum(gate["dispositions"].values()) == 5


def test_promoting_an_interpretation_makes_it_a_proposal_that_must_be_decided(store):
    actor = "dana"
    aid, doc = _agreement_in_review(actor)
    trade_store.save_extraction(aid, doc, _with_an_empty_reading(doc), actor)

    before = trade_review.document_gate(aid, doc)
    assert "i-hollow" not in next(
        b for b in before["blockers"] if b["kind"] == "instances_undecided")["ids"]

    trade_review.promote_instance(aid, doc, "i-hollow", actor)

    after = trade_review.document_gate(aid, doc)
    assert after["instances_interpretive"] == 0
    assert "i-hollow" in next(
        b for b in after["blockers"] if b["kind"] == "instances_undecided")["ids"], (
        "a promoted reading still does not block approval, so promoting it did nothing")
    # and the standings the screen reads agree with the gate
    extraction = trade_store.load_extraction(aid, doc)
    review = trade_store.load_review(aid, doc)
    assert trade_review.standings(extraction, review)["i-hollow"]["standing"] == "confident"


def test_promoting_a_term_that_already_carries_values_is_refused(store):
    actor = "dana"
    aid, doc = _agreement_in_review(actor)
    with pytest.raises(ValueError, match="already a proposal"):
        trade_review.promote_instance(aid, doc, "i-budget", actor)
