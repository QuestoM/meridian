"""Simulation: real arithmetic on real activity, and loud about its own gaps.

The properties: a retroactive ladder and a marginal ladder give DIFFERENT
answers on the same spend (the distinction a shallow model erases); commission
follows its stated base; a ladder whose mechanics the document never stated is
refused rather than guessed; constraints are counted and never priced; and
every term the engine could not simulate appears by name.
"""

from datetime import date

import pandas as pd
import pytest

from kairos.trade import simulate
from kairos.trade.simulate import SimulationInputs


def _inputs(spend_rows, today=date(2026, 6, 30), window=None):
    delivery = pd.DataFrame(spend_rows, columns=[
        "campaign_id", "broadcast_date", "air_state", "channel", "spots",
        "seconds", "rating_points_planned", "spend_ils", "counted_as_of",
    ])
    return SimulationInputs(
        delivery=delivery,
        campaigns=pd.DataFrame([("C1", "טכנו-קור"), ("C2", "טכנו-קור")],
                               columns=["campaign_id", "advertiser"]),
        agency_links=pd.DataFrame(columns=["agency_id", "agency_name", "advertiser"]),
        today=today,
        window=window,
    )


def _head():
    return {"agreement_id": "agr-sim", "level": "advertiser",
            "counterparty": {"advertiser": "טכנו-קור"},
            "window": {"starts_on": "2026-01-01", "ends_on": "2026-12-31"}}


def _termset(instances):
    return {"version_id": "v-sim", "agreement_id": "agr-sim", "instances": instances}


def _ladder(mechanics):
    return {
        "instance_id": "i-ladder", "term_id": "volume-discount-ladder",
        "params": {
            "tiers": [
                {"threshold": 0, "discount_percent": 10},
                {"threshold": 1_000_000, "discount_percent": 15},
            ],
            "basis": "ratecard_gross", "mechanics": mechanics, "period": "year",
        },
        "scope": {}, "window": {}, "citations": [],
    }


ROWS = [
    ("C1", "2026-03-01", "aired", "רשת 13", 20, 600, 40.0, 900_000, "t"),
    ("C2", "2026-04-01", "aired", "רשת 13", 10, 300, 20.0, 300_000, "t"),
    ("C1", "2026-09-01", "scheduled", "רשת 13", 5, 150, 10.0, 200_000, "t"),
]


def test_retroactive_and_marginal_ladders_differ_on_the_same_spend():
    retro = simulate.simulate(_termset([_ladder("retroactive")]), _head(), _inputs(ROWS))
    marginal = simulate.simulate(_termset([_ladder("marginal")]), _head(), _inputs(ROWS))
    # Gross aired = 1,200,000. Retroactive: the 15% tier is reached, so the
    # whole period re-rates → 180,000. Marginal: 10% on the first million and
    # 15% on the 200k above it → 100,000 + 30,000 = 130,000.
    assert retro["money"]["gross_aired"] == 1_200_000.0
    assert retro["money"]["discount_ladder"]["discount_value"] == 180_000.0
    assert marginal["money"]["discount_ladder"]["discount_value"] == 130_000.0
    assert retro["money"]["discount_ladder"]["mechanics"] == "retroactive"


def test_an_amended_ladder_keeps_the_tiers_the_amendment_never_mentioned():
    """The measured bug this test exists for: the corpus flagship carries the
    body's 12/15/18% ladder AND an appendix restating ONLY the top tier as 17%.
    Keying settlement terms by kind let the appendix overwrite the body, so a
    ₪3.84M spend reported NO discount at all — the worst class of error this
    engine can make. Tiers merge by threshold instead."""
    body = _ladder("retroactive")
    body["params"]["tiers"] = [
        {"threshold": 0, "discount_percent": 12},
        {"threshold": 8_000_000, "discount_percent": 15},
        {"threshold": 12_000_000, "discount_percent": 18},
    ]
    appendix = {
        "instance_id": "i-ladder-appendix", "term_id": "volume-discount-ladder",
        "params": {
            "tiers": [{"threshold": 12_000_000, "discount_percent": 17}],
            "basis": "ratecard_gross", "mechanics": "retroactive", "period": "year",
        },
        "scope": {}, "window": {}, "citations": [],
    }
    result = simulate.simulate(_termset([body, appendix]), _head(), _inputs(ROWS))
    ladder = result["money"]["discount_ladder"]
    # 1.2M of aired spend sits in the first tier, which the appendix never
    # touched: 12% survives instead of vanishing.
    assert ladder["tier_reached_percent"] == 12
    assert ladder["discount_value"] == 144_000.0
    assert ladder["merged_from"] == ["i-ladder", "i-ladder-appendix"]
    # And the amendment's own tier is the one that binds at the top.
    assert ladder["next_tier"]["threshold"] == 8_000_000
    merged = simulate.merge_ladders([body["params"], appendix["params"]])
    top = [t for t in merged["tiers"] if t["threshold"] == 12_000_000]
    assert top == [{"threshold": 12_000_000, "discount_percent": 17}]
    assert len(merged["tiers"]) == 3, "no threshold may be lost in the merge"


def test_two_commission_percentages_refuse_to_merge():
    first = {
        "instance_id": "i-c1", "term_id": "agency-commission",
        "params": {"percent": 15, "base": "gross", "form": "invoice_deduction"},
        "scope": {}, "window": {}, "citations": [],
    }
    second = {**first, "instance_id": "i-c2",
              "params": {"percent": 12, "base": "gross", "form": "invoice_deduction"}}
    result = simulate.simulate(_termset([first, second]), _head(), _inputs(ROWS))
    assert "agency_commission" not in result["money"]
    reasons = [n["reason_he"] for n in result["not_simulated"]]
    assert any("הכרעה אנושית" in r for r in reasons)


def test_an_unstated_ladder_mechanic_is_refused_not_guessed():
    result = simulate.simulate(_termset([_ladder("unstated")]), _head(), _inputs(ROWS))
    assert result["money"]["discount_ladder"]["available"] is False
    reasons = [n["reason_he"] for n in result["not_simulated"]]
    assert any("רטרואקטיבי או שולי" in r for r in reasons)
    assert result["money"]["net_after_simulated_terms"] == 1_200_000.0


def test_commission_follows_its_stated_base():
    commission = {
        "instance_id": "i-com", "term_id": "agency-commission",
        "params": {"percent": 15, "base": "net_of_discount",
                   "form": "invoice_deduction"},
        "scope": {}, "window": {}, "citations": [],
    }
    result = simulate.simulate(
        _termset([_ladder("retroactive"), commission]), _head(), _inputs(ROWS))
    block = result["money"]["agency_commission"]
    assert block["base_value"] == 1_020_000.0        # 1.2M gross − 180k discount
    assert block["commission_value"] == 153_000.0    # 15% of the net base
    assert result["money"]["net_after_simulated_terms"] == 867_000.0


def test_constraints_are_counted_and_never_priced():
    restriction = {
        "instance_id": "i-r", "term_id": "programme-daypart-restrictions",
        "params": {"mode": "forbid", "hard": True},
        "scope": {"genres": ["חדשות"]}, "window": {}, "citations": [],
    }
    result = simulate.simulate(_termset([restriction]), _head(), _inputs(ROWS))
    assert result["placement"]["conditions"] == 1
    assert "המצאה" in result["placement"]["note_he"]
    assert result["money"]["net_after_simulated_terms"] == result["money"]["gross_aired"]


def test_a_guarantee_in_an_unmeasurable_audience_lands_in_not_simulated():
    guarantee = {
        "instance_id": "i-g", "term_id": "trp-delivery-guarantee",
        "params": {"points": 400, "audience": "נשים 25-54", "window": "year"},
        "scope": {}, "window": {}, "citations": [],
    }
    result = simulate.simulate(_termset([guarantee]), _head(), _inputs(ROWS))
    ids = {n.get("instance_id") for n in result["not_simulated"]}
    assert "i-g" in ids
    assert any("מטבעות" in n["reason_he"] for n in result["not_simulated"])


def test_exposure_lists_the_commitments_a_deal_would_endanger():
    budget = {
        "instance_id": "i-b", "term_id": "budget-commitment",
        "params": {"amount": {"amount": 6_000_000, "basis": "ratecard"},
                   "period": "year"},
        "scope": {}, "window": {}, "citations": [],
    }
    result = simulate.simulate(_termset([budget]), _head(), _inputs(ROWS))
    assert result["exposure"], "a 1.2M pace against a 6M commitment is exposure"
    assert result["exposure"][0]["term_id"] == "budget-commitment"
    assert "התחייבויות בסיכון" in result["headline_he"]


def test_the_window_narrows_what_is_counted():
    q1 = simulate.simulate(
        _termset([_ladder("retroactive")]), _head(),
        _inputs(ROWS, window={"from": "2026-01-01", "to": "2026-03-31"}),
    )
    assert q1["money"]["gross_aired"] == 900_000.0
    assert q1["money"]["discount_ladder"]["tier_reached_percent"] == 10
    assert q1["money"]["discount_ladder"]["distance_to_next"] == 100_000.0


def test_simulation_writes_nothing(tmp_path, monkeypatch):
    from kairos.optimize import _frequency_rules
    from kairos_api import advertiser_conditions

    adv = tmp_path / "advertiser_conditions.csv"
    adv.write_text("advertiser_id,rule_id,effect\n", encoding="utf-8-sig")
    freq = tmp_path / "frequency_rules.csv"
    freq.write_text("rule_id,limit_type\n", encoding="utf-8-sig")
    monkeypatch.setattr(advertiser_conditions, "CONDITIONS_PATH", adv)
    monkeypatch.setattr(_frequency_rules, "DEFAULT_FREQUENCY_PATH", freq)
    before = (adv.read_bytes(), freq.read_bytes())
    restriction = {
        "instance_id": "i-r", "term_id": "programme-daypart-restrictions",
        "params": {"mode": "forbid", "hard": True},
        "scope": {"genres": ["חדשות"]}, "window": {}, "citations": [],
    }
    simulate.simulate(_termset([restriction, _ladder("retroactive")]),
                      _head(), _inputs(ROWS))
    assert (adv.read_bytes(), freq.read_bytes()) == before
