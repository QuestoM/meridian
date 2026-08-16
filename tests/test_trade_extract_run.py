"""The extraction runner against a simulated-perfect and a failing model.

The fake ``call`` answers from the corpus flagship's own ground truth, so the
assembly logic (grouping, merging, dispositions, conflicts) is tested exactly
as it will run live — only the model is simulated. The failure fakes prove
the honesty paths: one failed unit becomes an incomplete instance or an
unmapped clause naming the failure, never a dropped one.
"""

import pytest

from kairos.trade import corpus, extract_run
from kairos.trade.documents import UNMAPPED


@pytest.fixture(scope="module")
def flagship():
    return corpus.load_all()["heb-annual-framework-2026"]


def _truth_fake_call(flagship):
    """A call fake that answers classify/parameterise from ground truth."""
    truth_labels: dict[str, list[str]] = {}
    for d in flagship.dispositions:
        if d.disposition == "mapped":
            terms = []
            for iid in d.instance_ids:
                inst = next(i for i in flagship.instances if i.instance_id == iid)
                terms.append(inst.term_id)
            truth_labels[d.clause_id] = sorted(set(terms))
        elif d.disposition == "irrelevant":
            truth_labels[d.clause_id] = [f"irrelevant:{d.irrelevant_class}"]
        else:
            truth_labels[d.clause_id] = [UNMAPPED]

    by_clause_term = {}
    for inst in flagship.instances:
        for cit in inst.citations:
            by_clause_term[(cit.clause_id, inst.term_id)] = inst

    def call(*, stage, tier, system, content, tool_name, tool_schema):
        if stage == "classify":
            import re
            ids = re.findall(r'<clause id="([^"]+)"', content)
            return {"classifications": [
                {"clause_id": cid,
                 "labels": truth_labels.get(cid, [UNMAPPED]),
                 "note": "סעיף ללא מונח נתמך" if truth_labels.get(cid) == [UNMAPPED] else ""}
                for cid in ids
            ]}
        if stage == "parameterise":
            import re
            m = re.search(r'<הסעיף id="([^"]+)"', content)
            term = re.search(r"המונח לחילוץ: ([a-z0-9-]+)", system + content)
            term_id = term.group(1) if term else ""
            # find the source term instruction inside content first line
            first = re.search(r"המונח לחילוץ: ([a-z0-9-]+)", content)
            if first:
                term_id = first.group(1)
            cid = m.group(1)
            inst = by_clause_term.get((cid, term_id))
            if inst is None:
                return {"params": {}, "quotes": [content[:20]], "confidence": "low",
                        "missing": [], "notes": "אין אמת ידועה"}
            quotes = [c.quote for c in inst.citations if c.clause_id == cid] or [
                c.quote for c in inst.citations
            ]
            return {
                "params": inst.params,
                "scope": inst.scope,
                "window": inst.window,
                "quotes": quotes,
                "confidence": "high",
                "missing": inst.missing,
                "notes": inst.notes,
            }
        raise AssertionError(f"unexpected stage {stage}")

    return call


def test_perfect_model_assembles_a_complete_valid_extraction(flagship):
    result = extract_run.extract_document(
        flagship.document_id, flagship.clauses, _truth_fake_call(flagship),
        agreement_id="agr-flag",
    )
    cov = result.coverage()
    assert cov.complete
    assert cov.total_clauses == 50
    assert cov.unmapped == 0
    terms_found = {i.term_id for i in result.instances}
    terms_truth = {i.term_id for i in flagship.instances}
    assert terms_truth <= terms_found
    # The planted ladder contradiction is detected and auto-resolved through
    # the appendix-precedence edge extracted from the document itself.
    conflicts = result.stats["conflicts"]
    ladder = [c for c in conflicts if c["term_id"] == "volume-discount-ladder"]
    assert len(ladder) == 1
    assert ladder[0]["resolution"] == "resolved_by_rule"
    winner = next(i for i in result.instances
                  if i.instance_id == ladder[0]["winner"])
    assert winner.citations[0].clause_id == "appA-3"


def test_pointer_and_content_clauses_share_one_instance(flagship):
    result = extract_run.extract_document(
        flagship.document_id, flagship.clauses, _truth_fake_call(flagship),
    )
    d31 = next(d for d in result.dispositions if d.clause_id == "3.1")
    dappA1 = next(d for d in result.dispositions if d.clause_id == "appA-1")
    shared = set(d31.instance_ids) & set(dappA1.instance_ids)
    assert shared, "the CPP pointer (3.1) and its table (appA-1) must share an instance"
    inst = next(i for i in result.instances if i.instance_id in shared)
    assert inst.term_id == "cpp-daypart-table"
    cited = {c.clause_id for c in inst.citations}
    assert {"3.1", "appA-1"} <= cited


def test_a_parameterise_failure_becomes_an_honest_incomplete_instance(flagship):
    truth_call = _truth_fake_call(flagship)

    def failing_call(**kwargs):
        if kwargs["stage"] == "parameterise" and "budget-commitment" in kwargs["content"]:
            raise RuntimeError("boom")
        return truth_call(**kwargs)

    result = extract_run.extract_document(
        flagship.document_id, flagship.clauses, failing_call,
    )
    assert result.coverage().complete
    budget = [i for i in result.instances if i.term_id == "budget-commitment"]
    assert budget and budget[0].incomplete
    assert "נכשל" in budget[0].notes
    assert result.stats["failures"] and result.stats["failures"][0]["term_id"] == "budget-commitment"


def test_a_clause_the_classifier_never_answers_lands_unmapped_loudly(flagship):
    truth_call = _truth_fake_call(flagship)

    def forgetful_call(**kwargs):
        result = truth_call(**kwargs)
        if kwargs["stage"] == "classify":
            result = {"classifications": [
                c for c in result["classifications"] if c["clause_id"] != "8.2"
            ]}
        return result

    result = extract_run.extract_document(
        flagship.document_id, flagship.clauses, forgetful_call,
    )
    assert result.coverage().complete
    d82 = next(d for d in result.dispositions if d.clause_id == "8.2")
    assert d82.disposition == "unmapped"
    assert "בדיקה אנושית" in d82.reason
