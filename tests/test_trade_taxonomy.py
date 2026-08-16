"""The trade taxonomy's three-way contract: registry, schemas, catalogue doc.

The taxonomy is the completeness contract for the whole trade engine — the
clause classifier can only emit labels that exist here, so a term missing from
the registry is a term that can never be extracted. These tests pin the
machine registry (kairos/trade/taxonomy.py), the parameter schemas
(taxonomy_schemas.py) and the human catalogue (docs/trade/term-taxonomy.md)
to one term list, so none can drift without failing loudly.
"""

import re
from pathlib import Path

import pytest

from kairos.trade import taxonomy, taxonomy_schemas

DOC = Path(__file__).resolve().parents[1] / "docs" / "trade" / "term-taxonomy.md"


def test_registry_is_well_formed():
    assert len(taxonomy.TERMS) >= 60, "the catalogue is exhaustive, not a sample"
    for term_id, spec in taxonomy.TERMS.items():
        assert term_id == spec.id
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", term_id), term_id
        assert spec.name_he.strip(), f"{term_id} has no Hebrew name"
        assert spec.name_en.strip(), f"{term_id} has no English name"
        assert spec.behaviours, f"{term_id} declares no behaviour"


def test_every_term_has_exactly_one_schema():
    assert set(taxonomy_schemas.SCHEMAS) == set(taxonomy.TERMS)
    for term_id, schema in taxonomy_schemas.SCHEMAS.items():
        assert schema.get("type") == "object", term_id
        assert "properties" in schema, term_id


def test_interactions_reference_real_terms():
    taxonomy.validate_interactions()


def test_classification_labels_cover_terms_and_irrelevant_classes():
    labels = taxonomy.classification_labels()
    assert set(labels) == set(taxonomy.TERMS) | set(taxonomy.IRRELEVANT_CLASSES)
    assert "unmapped" not in labels, (
        "'unmapped' is the pipeline's reserved refusal label and must never "
        "look like a positive classification"
    )


def test_catalogue_doc_and_registry_hold_the_same_terms():
    text = DOC.read_text(encoding="utf-8")
    doc_ids = set(re.findall(r"^### [A-Z]+\d* `([a-z0-9-]+)`", text, re.M))
    registry_ids = set(taxonomy.TERMS)
    # NA terms are catalogued in the doc's closing support-summary prose, not
    # as headed sections; every headed section must be a registry term and
    # every non-NA registry term must have a headed section.
    non_na = {t for t, s in taxonomy.TERMS.items() if s.family != "NA"}
    missing_from_doc = sorted(non_na - doc_ids)
    unknown_in_doc = sorted(doc_ids - registry_ids)
    assert not missing_from_doc, f"terms without a catalogue entry: {missing_from_doc}"
    assert not unknown_in_doc, f"catalogue entries without a registry term: {unknown_in_doc}"


def test_binding_claims_are_backed_by_behaviour():
    """A term whose status claims binding must declare a behaviour that can
    bind; process/meta-only terms may not claim BINDS."""
    for spec in taxonomy.TERMS.values():
        if spec.status == "BINDS":
            assert set(spec.behaviours) & {
                "prices",
                "constrains-hard",
                "constrains-soft",
                "obliges",
                "settles",
            }, f"{spec.id} claims BINDS with behaviours {spec.behaviours}"


def test_every_extractable_term_carries_hebrew_cues():
    for spec in taxonomy.TERMS.values():
        if spec.family == "NA":
            continue
        assert spec.cues, f"{spec.id} has no extraction cues"


@pytest.mark.parametrize("bad_id", ["", "no-such-term", "CPP"])
def test_get_refuses_unknown_ids(bad_id):
    with pytest.raises(KeyError):
        taxonomy.get(bad_id)
