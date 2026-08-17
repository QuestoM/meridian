"""Plain-language effects: what a term will DO, honest about what it will not.

The property that matters at review: a term the compiler could not bind must
read as "will not act automatically, because…" and never as a confident
description of an effect that will never happen.
"""

import pytest

from kairos.trade import explain
from kairos.trade.compile import compile_termset


def _head(counterparty=None, level="advertiser"):
    return {"agreement_id": "agr-e", "level": level,
            "counterparty": counterparty or {"advertiser": "טכנו-קור"},
            "window": {"starts_on": "2026-01-01", "ends_on": "2026-12-31"}}


def _termset(instances):
    return {"version_id": "v-e", "agreement_id": "agr-e", "instances": instances}


def _inst(iid, term_id, params, scope=None):
    return {"instance_id": iid, "term_id": term_id, "params": params,
            "scope": scope or {}, "window": {}, "citations": []}


def test_a_hard_restriction_reads_as_a_block_and_binds():
    termset = _termset([_inst("i-1", "content-adjacency-exclusion", {
        "excluded_content": ["סיקור אסונות", "תכני חדשות קשים"],
        "radius": "same_break", "hard": True,
    })])
    result = explain.explain_termset(termset, _head())
    (term,) = result["terms"]
    assert term["mechanism"] == explain.BLOCKS
    assert term["mechanism_he"] == "חוסם שיבוץ"
    assert "לא ישובצו באותו מקבץ" in term["sentence_he"]
    assert "סיקור אסונות" in term["sentence_he"]
    assert term["bound_rule_ids"], "a blocking term must name the rule it became"
    assert term["will_not_act_reasons"] == []


def test_a_ladder_reads_as_settlement_and_states_its_mechanics():
    termset = _termset([_inst("i-2", "volume-discount-ladder", {
        "tiers": [{"threshold": 0, "discount_percent": 12},
                  {"threshold": 8_000_000, "discount_percent": 15}],
        "basis": "ratecard_gross", "mechanics": "retroactive", "period": "year",
    })])
    (term,) = explain.explain_termset(termset, _head())["terms"]
    assert term["mechanism"] == explain.SETTLES
    assert "רטרואקטיבי" in term["sentence_he"]
    assert "2 מדרגות" in term["sentence_he"]
    assert term["settlement_kinds"] == ["discount_ladder"]


def test_a_term_the_compiler_skipped_says_it_will_not_act():
    # A brand-scoped exclusivity with no advertiser mapping: the compiler
    # refuses to guess, so the sentence must not promise an effect.
    termset = _termset([_inst("i-3", "category-exclusivity", {
        "category": "מזון", "exclusivity_scope": "מאסטר קלאס",
        "premium_percent": 12,
    }, scope={"brands": ["שדות"]})])
    (term,) = explain.explain_termset(termset, _head())["terms"]
    assert term["mechanism"] == explain.INERT
    assert term["mechanism_he"] == "לא יפעל אוטומטית"
    assert term["will_not_act_reasons"], "the compiler's reason must travel"
    assert any("מותג" in r for r in term["will_not_act_reasons"])
    assert term["bound_rule_ids"] == []


def test_a_guarantee_reads_as_continuously_measured():
    termset = _termset([_inst("i-4", "trp-delivery-guarantee", {
        "points": 320, "audience": "גברים 18-44", "window": "campaign",
        "tolerance_percent": 8,
    })])
    (term,) = explain.explain_termset(termset, _head())["terms"]
    assert term["mechanism"] == explain.MEASURES
    assert "320 נקודות" in term["sentence_he"]
    assert "גברים 18-44" in term["sentence_he"]
    assert "8%" in term["sentence_he"]


def test_missing_document_values_are_named_in_the_sentence():
    instance = _inst("i-5", "trp-delivery-guarantee",
                     {"audience": "כלל הצופים", "window": "campaign"})
    instance["missing"] = ["points"]
    (term,) = explain.explain_termset(_termset([instance]), _head())["terms"]
    assert term["incomplete"] is True
    assert "חסרים במסמך: points" in term["sentence_he"]
    assert "כמות שנקבעת בכל הזמנה" in term["sentence_he"]


def test_the_scope_phrase_reads_in_hebrew_labels():
    termset = _termset([_inst("i-6", "frequency-caps", {"unit": "day", "cap": 4},
                              scope={"programmes": ["מאסטר קלאס", "המרוץ הגדול"],
                                     "advertisers": ["Delta Motors"]})])
    (term,) = explain.explain_termset(termset, _head())["terms"]
    assert "תוכניות: מאסטר קלאס, המרוץ הגדול" in term["scope_he"]
    assert "מפרסמים: Delta Motors" in term["scope_he"]
    assert "לכל היותר 4 תשדירים ביממת שידור" in term["sentence_he"]


def test_every_taxonomy_term_produces_a_sentence():
    from kairos.trade import taxonomy

    for term_id, spec in taxonomy.TERMS.items():
        result = explain.explain_instance(
            {"instance_id": "x", "term_id": term_id, "params": {}, "scope": {}}
        )
        assert result["sentence_he"].strip(), f"{term_id} produced no sentence"
        assert result["mechanism"] in explain.MECHANISM_HE
        assert result["term_name_he"] == spec.name_he


def test_by_mechanism_summary_counts_the_families():
    termset = _termset([
        _inst("a", "frequency-caps", {"unit": "day", "cap": 4}),
        _inst("b", "agency-commission",
              {"percent": 15, "base": "gross", "form": "invoice_deduction"}),
        _inst("c", "budget-commitment",
              {"amount": {"amount": 5_000_000, "basis": "ratecard"}, "period": "year"}),
    ])
    result = explain.explain_termset(termset, _head())
    assert result["by_mechanism"][explain.BLOCKS] == 1
    assert result["by_mechanism"][explain.SETTLES] == 1
    assert result["by_mechanism"][explain.MEASURES] == 1
