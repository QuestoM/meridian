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
    with pytest.raises(ValueError, match="editable"):
        trade_store.attach_document(aid, filename="late.pdf", payload=b"x", actor="dana")


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
