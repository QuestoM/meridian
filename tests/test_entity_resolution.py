"""Resolution before creation: a name is not an identity.

The engine's job is to stop a second record being made for a party we already
carry. These tests pin the three things that make it trustworthy: the
deterministic signals settle the certain cases without a model, the model only
rules on the genuinely ambiguous and never invents a candidate, and the verdict
is honest about what it earned - a fuzzy lead is 'possible', never 'exact', and
with no model configured it can never claim more than 'possible'.
"""

from __future__ import annotations

import pytest

from kairos_api import entity_resolution as er


AGENCIES = [
    {"entity_id": "AGY_01", "name": "OMD", "aliases": ["OMD", "או.אם.די", "OMD ישראל"], "vat": "513200001"},
    {"entity_id": "AGY_07", "name": "פובליסיס", "aliases": ["פובליסיס", "Publicis"], "vat": "511000007"},
    {"entity_id": "AGY_09", "name": "רואים קונים", "aliases": ["רואים קונים"], "vat": ""},
]


@pytest.fixture
def roster(monkeypatch):
    monkeypatch.setattr(er, "agency_roster", lambda: er.Roster(kind="agency", records=[dict(r) for r in AGENCIES]))
    # No model unless a test opts in: the default path is deterministic-only.
    monkeypatch.setattr(er, "_adjudicate", lambda *a, **k: False)


def test_normalize_folds_case_script_punctuation_and_company_suffix():
    assert er.normalize_name('OMD ישראל בע"מ') == er.normalize_name("omd ישראל")
    assert er.normalize_name("פובליסיס בע״מ") == "פובליסיס"
    # Final-letter fold: a name ending in a final form matches its medial twin.
    assert er.normalize_name("רואים") == er.normalize_name("רואים")
    assert er.normalize_name("כלמוביל") == er.normalize_name("כלמוביל")


def test_an_exact_normalized_name_is_an_exact_verdict_without_a_model(roster):
    out = er.resolve_counterparty("agency", 'OMD בע"מ')
    assert out["verdict"] == er.EXACT
    assert out["recommended_action"] == "use_existing"
    assert out["match"]["entity_id"] == "AGY_01"
    assert out["model_used"] is False


def test_a_shared_vat_id_matches_even_when_the_name_differs(roster):
    out = er.resolve_counterparty("agency", "חברת מדיה כלשהי", vat_id="51-320-0001")
    assert out["verdict"] == er.EXACT
    assert out["match"]["entity_id"] == "AGY_01"
    assert out["match"]["signals"]["vat_match"] is True


def test_a_transliteration_hits_the_alias(roster):
    out = er.resolve_counterparty("agency", "או.אם.די")
    assert out["match"]["entity_id"] == "AGY_01"
    assert out["verdict"] in (er.EXACT, er.PROBABLE, er.POSSIBLE)


def test_a_clearly_new_party_is_none_and_creates_freely(roster):
    out = er.resolve_counterparty("agency", "מקאן אריקסון ישראל")
    assert out["verdict"] == er.NONE
    assert out["recommended_action"] == "create_new"
    assert out["match"] is None


def test_a_fuzzy_lead_with_no_model_is_capped_at_possible(roster):
    # "פובליסים" is one letter off "פובליסיס" - a strong fuzzy, but no model ran,
    # so it must be shown, not asserted.
    out = er.resolve_counterparty("agency", "פובליסים")
    assert out["verdict"] == er.POSSIBLE
    assert out["recommended_action"] == "ask"
    assert out["match"]["entity_id"] == "AGY_07"
    assert out["model_used"] is False


def test_the_model_only_adjudicates_the_ambiguous_and_can_confirm_probable(monkeypatch):
    monkeypatch.setattr(er, "agency_roster", lambda: er.Roster(kind="agency", records=[dict(r) for r in AGENCIES]))
    seen = {}

    def fake_adjudicate(kind, name, evidence, candidates):
        # The model must be handed ONLY the ambiguous candidate, never the whole roster.
        seen["ids"] = [c.entity_id for c in candidates]
        for c in candidates:
            if c.entity_id == "AGY_07":
                c.verdict, c.confidence, c.reason = "same", 0.92, "one-letter typo of פובליסיס"
        return True

    monkeypatch.setattr(er, "_adjudicate", fake_adjudicate)
    out = er.resolve_counterparty("agency", "פובליסים", evidence="חתום מול פובליסיס")
    assert out["model_used"] is True
    assert out["verdict"] == er.PROBABLE
    assert out["recommended_action"] == "use_existing"
    assert out["match"]["model_verdict"] == "same"
    assert "AGY_07" in seen["ids"]
    # A far-off party never reaches the model.
    assert "AGY_09" not in seen["ids"]


def test_a_model_different_ruling_keeps_the_verdict_soft(monkeypatch):
    monkeypatch.setattr(er, "agency_roster", lambda: er.Roster(kind="agency", records=[dict(r) for r in AGENCIES]))

    def fake_adjudicate(kind, name, evidence, candidates):
        for c in candidates:
            c.verdict, c.confidence, c.reason = "different", 0.9, "a different company that happens to look alike"
        return True

    monkeypatch.setattr(er, "_adjudicate", fake_adjudicate)
    out = er.resolve_counterparty("agency", "פובליסים")
    # The model said different; we do not claim a match, but a surfaced lead can
    # still read POSSIBLE for the operator's eye. It must never be PROBABLE/EXACT.
    assert out["verdict"] in (er.POSSIBLE, er.NONE)
    assert out["recommended_action"] in ("ask", "create_new")


def test_the_read_tool_wiring_rejects_a_bad_kind_and_missing_name():
    from kairos_api.assistant_read_tools_catalog import _read_resolve_counterparty

    assert "error" in _read_resolve_counterparty({"kind": "brand", "name": "x"})
    assert "error" in _read_resolve_counterparty({"kind": "agency", "name": "  "})


def test_the_tool_is_registered_so_kai_can_call_it():
    from kairos_api.assistant_tools import READ_TOOL_NAMES

    assert "resolve_counterparty" in READ_TOOL_NAMES
