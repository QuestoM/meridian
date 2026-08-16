"""Every corpus agreement's ground truth must survive the same validation as
a live pipeline output: full clause coverage, real citations with verbatim
quotes, schema-complete params (or honestly-declared missing fields). A
corpus whose truth is broken measures nothing."""

import pytest

from kairos.trade import corpus


@pytest.fixture(scope="module")
def all_docs():
    docs = corpus.load_all()
    assert docs, "the corpus directory holds no agreements"
    return docs


def test_every_corpus_document_loads_and_validates(all_docs):
    for doc in all_docs.values():
        cov = doc.coverage()
        assert cov.complete, f"{doc.document_id} leaves clauses unaccounted"


def test_flagship_framework_has_the_planted_shape(all_docs):
    doc = all_docs["heb-annual-framework-2026"]
    cov = doc.coverage()
    assert cov.total_clauses == 50
    assert cov.unmapped == 0
    terms = {i.term_id for i in doc.instances}
    # The families the flagship deliberately exercises.
    for expected in [
        "volume-discount-ladder",
        "trp-delivery-guarantee",
        "shortfall-cure",
        "makegood-accrual-policy",
        "preferred-position-guarantee",
        "competitive-separation",
        "category-exclusivity",
        "settlement-mechanics",
        "measurement-source",
        "precedence-clause",
        "force-majeure",
    ]:
        assert expected in terms, f"flagship lost its {expected} coverage"
    # The planted body-vs-appendix contradiction is recorded with resolution.
    conflicts = doc.stats["expected_conflicts"]
    assert len(conflicts) == 1
    assert set(conflicts[0]["instances"]) == {"gt-ladder-body", "gt-ladder-appendix"}
    assert conflicts[0]["winner"] == "gt-ladder-appendix"


def test_the_parametric_guarantee_is_honestly_incomplete(all_docs):
    doc = all_docs["heb-annual-framework-2026"]
    guarantee = next(i for i in doc.instances if i.instance_id == "gt-trp-guarantee")
    assert guarantee.incomplete
    assert "points" in guarantee.missing
