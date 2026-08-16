"""The precedence algebra: deterministic where the documents are, open where
they are not, and never a silent coin-flip on money."""

from kairos.trade import corpus, precedence
from kairos.trade.precedence import Candidate, Edge, Provenance


def _cand(iid, term, params, *, scope=None, window=None, level="advertiser",
          region="body", effective="2026-01-01", agreement="agr-x", doc="doc-x"):
    return Candidate(
        instance_id=iid, term_id=term, params=params, scope=scope or {},
        window=window or {},
        provenance=Provenance(
            agreement_id=agreement, level=level, document_id=doc,
            region=region, effective_date=effective,
        ),
    )


def test_the_flagship_planted_conflict_resolves_to_the_appendix():
    doc = corpus.load_all()["heb-annual-framework-2026"]
    by_id = {i.instance_id: i for i in doc.instances}
    body = by_id["gt-ladder-body"]
    appendix = by_id["gt-ladder-appendix"]
    prec = by_id["gt-precedence-appendix"]

    edges = precedence.edges_from_precedence_instances(
        [{"term_id": prec.term_id, "params": prec.params}],
        agreement_id="agr-flagship", document_id=doc.document_id,
    )
    assert edges, "the flagship's 10.1 must interpret into an edge"

    candidates = [
        _cand("gt-ladder-body", body.term_id, body.params,
              region=precedence.region_of_clause(body.citations[0].clause_id),
              doc=doc.document_id, agreement="agr-flagship"),
        _cand("gt-ladder-appendix", appendix.term_id, appendix.params,
              region=precedence.region_of_clause(appendix.citations[0].clause_id),
              doc=doc.document_id, agreement="agr-flagship"),
    ]
    conflicts = precedence.detect_and_resolve(candidates, edges)
    assert len(conflicts) == 1
    verdict = conflicts[0]
    assert verdict.resolution == "resolved_by_rule"
    assert verdict.winner == "gt-ladder-appendix"
    assert verdict.rule == "explicit_precedence"
    assert "סעיף עדיפות" in verdict.explanation_he


def test_later_document_beats_earlier_when_no_edge_decides():
    a = _cand("old", "agency-commission",
              {"percent": 15, "base": "gross", "form": "invoice_deduction"},
              effective="2026-01-01")
    b = _cand("new", "agency-commission",
              {"percent": 12, "base": "gross", "form": "invoice_deduction"},
              region="amendment", effective="2026-06-01")
    (verdict,) = precedence.detect_and_resolve([a, b])
    assert verdict.winner == "new"
    assert verdict.rule == "later_document"


def test_campaign_level_beats_agency_framework():
    a = _cand("framework", "cash-discount",
              {"percent": 1.5, "qualifying_terms": "10 ימים"},
              level="agency_framework")
    b = _cand("campaign", "cash-discount",
              {"percent": 2.0, "qualifying_terms": "10 ימים"},
              level="campaign")
    (verdict,) = precedence.detect_and_resolve([a, b])
    assert verdict.winner == "campaign"
    assert verdict.rule == "specific_level"


def test_specific_scope_beats_general():
    a = _cand("general", "frequency-caps", {"unit": "hour", "cap": 3})
    b = _cand("scoped", "frequency-caps", {"unit": "hour", "cap": 2},
              scope={"programmes": ["מאסטר קלאס"]})
    (verdict,) = precedence.detect_and_resolve([a, b])
    assert verdict.winner == "scoped"
    assert verdict.rule == "specific_scope"


def test_constraints_fall_back_to_the_stricter_side():
    a = _cand("loose", "competitive-separation",
              {"separation_unit": "same_break", "separation_quantity": 0, "hard": True})
    b = _cand("strict", "competitive-separation",
              {"separation_unit": "minutes", "separation_quantity": 10, "hard": True})
    (verdict,) = precedence.detect_and_resolve([a, b])
    assert verdict.winner == "strict"
    assert verdict.rule == "safer_constraint"


def test_money_with_no_deciding_rule_stays_open():
    a = _cand("x", "agency-commission",
              {"percent": 15, "base": "gross", "form": "invoice_deduction"})
    b = _cand("y", "agency-commission",
              {"percent": 12, "base": "gross", "form": "invoice_deduction"})
    (verdict,) = precedence.detect_and_resolve([a, b])
    assert verdict.resolution == "open"
    assert verdict.winner is None
    assert "הכרעה אנושית" in verdict.explanation_he


def test_disjoint_scopes_and_windows_do_not_conflict():
    a = _cand("summer", "seasonal-coefficients",
              {"rows": [{"period_label": "יולי", "coefficient": 0.9}]},
              window={"from": "2026-07-01", "to": "2026-07-31"})
    b = _cand("winter", "seasonal-coefficients",
              {"rows": [{"period_label": "דצמבר", "coefficient": 1.2}]},
              window={"from": "2026-12-01", "to": "2026-12-31"})
    assert precedence.detect_and_resolve([a, b]) == []
    c = _cand("brand-a", "frequency-caps", {"unit": "day", "cap": 4},
              scope={"brands": ["A"]})
    d = _cand("brand-b", "frequency-caps", {"unit": "day", "cap": 2},
              scope={"brands": ["B"]})
    assert precedence.detect_and_resolve([c, d]) == []


def test_identical_duplicates_are_not_conflicts():
    a = _cand("one", "frequency-caps", {"unit": "hour", "cap": 2})
    b = _cand("two", "frequency-caps", {"unit": "hour", "cap": 2})
    assert precedence.detect_and_resolve([a, b]) == []
