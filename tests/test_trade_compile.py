"""The compiler, run against the corpus flagship's own ground truth.

The property: placement terms land in the product's existing rule primitives
with full attribution, period-arithmetic money lands in the settlement spec,
and what cannot bind is SKIPPED WITH A REASON — never silently."""

import pytest

from kairos.trade import compile as trade_compile
from kairos.trade import corpus


@pytest.fixture(scope="module")
def flagship_artifacts():
    doc = corpus.load_all()["heb-annual-framework-2026"]
    termset = {
        "version_id": "v-test",
        "agreement_id": "agr-flagship",
        "instances": [
            {
                "instance_id": i.instance_id,
                "term_id": i.term_id,
                "params": i.params,
                "scope": i.scope,
                "window": i.window,
                "citations": [
                    {"document_id": c.document_id, "page": c.page,
                     "clause_id": c.clause_id, "quote": c.quote}
                    for c in i.citations
                ],
            }
            for i in doc.instances
        ],
    }
    head = {
        "agreement_id": "agr-flagship",
        "level": "agency_framework",
        "counterparty": {"agency": "אופק מדיה"},
        "window": {"starts_on": "2026-01-01", "ends_on": "2026-12-31"},
    }
    return trade_compile.compile_termset(termset, head)


def test_every_compiled_rule_id_carries_its_source(flagship_artifacts):
    for row in flagship_artifacts.conditions + flagship_artifacts.frequency_rules:
        parsed = trade_compile.parse_rule_id(row["rule_id"])
        assert parsed is not None, row["rule_id"]
        assert parsed["agreement_id"] == "agr-flagship"
        assert parsed["version_id"] == "v-test"
        assert row["notes"], "every compiled row explains itself"


def test_frequency_caps_bind_per_day_and_refuse_the_unsupported_hour_unit(flagship_artifacts):
    rows = [r for r in flagship_artifacts.frequency_rules
            if "gt-freq" in r["rule_id"]]
    # The DAY cap binds for all 5 represented advertisers from נספח ד'; the
    # HOUR cap has no enforcing primitive in the frequency engine and must be
    # skipped BY NAME, never silently mapped to a different unit.
    assert len(rows) == 5
    assert {r["limit_type"] for r in rows} == {"max_per_day"}
    assert {r["value"] for r in rows} == {4}
    advertisers = {r["advertiser_id"] for r in rows}
    assert "Delta Motors" in advertisers
    assert "רשת מלונות גלים" in advertisers
    hour_skips = [s for s in flagship_artifacts.skipped
                  if s["instance_id"] == "gt-freq-hour"]
    assert hour_skips and "hour" in hour_skips[0]["reason_he"]


def test_the_delta_separation_without_a_rival_list_is_skipped_loudly(flagship_artifacts):
    skipped = {s["instance_id"]: s for s in flagship_artifacts.skipped}
    assert "gt-separation-delta" in skipped
    assert "מתחרים" in skipped["gt-separation-delta"]["reason_he"]
    assert not any(
        "gt-separation-delta" in r["rule_id"]
        for r in flagship_artifacts.frequency_rules
    )


def test_brand_scoped_exclusivity_refuses_to_guess_the_advertiser(flagship_artifacts):
    # The flagship grants exclusivity to the BRAND שדות with no advertiser in
    # scope. The compiler must not fan the premium out to all represented
    # advertisers and must not fuzzy-match the brand to a client name — the
    # binding is a review-screen act. Until then: zero rows, loud skip.
    premium_rows = [r for r in flagship_artifacts.conditions
                    if "gt-exclusivity-sadot" in r["rule_id"]]
    assert premium_rows == []
    reasons = [s["reason_he"] for s in flagship_artifacts.skipped
               if s["instance_id"] == "gt-exclusivity-sadot"]
    assert any("מותג" in r for r in reasons), reasons


def test_exclusivity_binds_once_the_reviewer_maps_the_brand():
    termset = {
        "version_id": "v-m", "agreement_id": "agr-m",
        "instances": [{
            "instance_id": "i-x", "term_id": "category-exclusivity",
            "params": {"category": "מזון", "exclusivity_scope": "מאסטר קלאס",
                       "premium_percent": 12},
            "scope": {"brands": ["שדות"],
                      "advertisers": ["שדות תעשיות מזון בע\"מ"],
                      "programmes": ["מאסטר קלאס"]},
            "window": {}, "citations": [],
        }],
    }
    head = {"agreement_id": "agr-m", "level": "agency_framework",
            "counterparty": {"agency": "אופק מדיה"}, "window": {}}
    artifacts = trade_compile.compile_termset(termset, head)
    (row,) = [r for r in artifacts.conditions if "i-x" in r["rule_id"]]
    assert row["advertiser_id"] == "שדות תעשיות מזון בע\"מ"
    assert row["effect"] == "premium"
    assert row["value"] == pytest.approx(1.12)
    assert "מאסטר קלאס" in row["scope_programmes"]
    # The blocking half still needs a rival-member mapping, and says so.
    assert any(s["instance_id"] == "i-x" for s in artifacts.skipped)


def test_adjacency_exclusion_becomes_a_forbid_condition(flagship_artifacts):
    rows = [r for r in flagship_artifacts.conditions
            if "gt-adjacency-exclusion" in r["rule_id"]]
    assert rows, "the adjacency exclusion must compile"
    assert all(r["effect"] == "forbid" for r in rows)
    assert all(r["scope_genres"] != "ANY" for r in rows)
    # Framework-level: one row per represented advertiser.
    assert len(rows) == 5


def test_period_arithmetic_money_lands_in_settlement_not_per_spot(flagship_artifacts):
    kinds = {t["kind"]: t for t in flagship_artifacts.settlement["terms"]}
    assert "discount_ladder" in kinds
    assert kinds["discount_ladder"]["mechanics"] == "retroactive"
    assert "agency_commission" in kinds
    assert kinds["agency_commission"]["percent"] == 15
    assert "cpp_table" in kinds
    assert "length_factors" in kinds
    assert "gold_rates" in kinds
    # And none of these produced a per-spot condition row.
    money_ids = ("gt-ladder-body", "gt-commission", "gt-cpp-table",
                 "gt-length-factors", "gt-gold-rates")
    for row in flagship_artifacts.conditions:
        assert not any(mid in row["rule_id"] for mid in money_ids)


def test_top_and_tail_right_steers_and_records_its_pair_note(flagship_artifacts):
    rows = [r for r in flagship_artifacts.conditions
            if "gt-top-and-tail" in r["rule_id"]]
    assert rows and all(r["effect"] == "pressure" for r in rows)
    kinds = {t["kind"] for t in flagship_artifacts.settlement["terms"]}
    assert "top_and_tail_right" in kinds


def test_summary_counts_add_up(flagship_artifacts):
    summary = flagship_artifacts.summary()
    assert summary["conditions"] == len(flagship_artifacts.conditions)
    assert summary["skipped"] == len(flagship_artifacts.skipped)
    assert summary["settlement_terms"] > 5
