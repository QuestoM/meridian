"""Agency layer: stores, links, conditions, and the daily-ledger money contract.

Covers the CRUD stores (validation, duplicate guards, deactivation), the link
derivation (observed from a daily frame, manual overrides), forbid-wins across
the agency and advertiser levels, the rebate math behind net_revenue, and the
gross-identity guarantee: with no agency conditions and rebate 0 (and with the
shipped seed, whose conditions file is header-only) gross revenue is exactly
what it was before the agency layer existed.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import kairos_api.agencies as ag
import kairos_api.agency_conditions as agc
from kairos.export.agency_layer import AgencyLayer, AgencyTerms
from kairos.export.spots import price_daily_spots
from kairos.optimize._frequency_rules import FrequencyRuleSet
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine, Condition
from kairos.optimize.pricing import PricingModel

ROOT = Path(__file__).resolve().parents[1]
DAILY_FIXTURE = ROOT / "data" / "daily_input" / "Wally_Prime_Reshet_Example_2025-04-27.csv"

SEED_AGENCY_NAMES = {
    "OMD", "יוניברסל", "יוניון", "ישירים", "לפמ",
    "מדיהקום", "פובליסיס", "פיתוח עסקי", "רואים קונים",
}


@pytest.fixture()
def temp_stores(tmp_path, monkeypatch):
    """Point every agency store at throwaway CSVs so tests never touch real data."""
    monkeypatch.setattr(ag, "AGENCIES_PATH", tmp_path / "agencies.csv")
    monkeypatch.setattr(ag, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agc, "LINKS_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(agc, "CONDITIONS_PATH", tmp_path / "agency_conditions.csv")
    monkeypatch.setattr(agc, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agc, "_latest_daily_pairs", lambda: ([], None))
    return tmp_path


def _create(agency_id="AGY_T1", name="סוכנות בדיקה", **kwargs):
    return ag.create_agency(ag.AgencyCreate(agency_id=agency_id, name=name, **kwargs))


# --- store CRUD and validation ------------------------------------------------

def test_create_get_update_roundtrip(temp_stores) -> None:
    record = _create(rebate_percent=4.0, payment_terms_days=90, agency_type="בוטיק")
    assert record["agency_id"] == "AGY_T1"
    assert record["rebate_percent"] == pytest.approx(4.0)
    assert record["payment_terms_days"] == 90
    assert record["status"] == "active"
    assert record["data_source"] == "manual"

    updated = ag.update_agency("AGY_T1", ag.AgencyUpdate(contact_name="דנה", rebate_percent=2.5))
    assert updated["contact_name"] == "דנה"
    assert updated["rebate_percent"] == pytest.approx(2.5)

    fetched = ag.get_agency("AGY_T1")
    assert fetched["contact_name"] == "דנה"
    assert fetched["boundary"]


def test_duplicate_id_and_name_are_rejected(temp_stores) -> None:
    _create()
    with pytest.raises(Exception) as excinfo:
        _create(name="שם אחר")
    assert getattr(excinfo.value, "status_code", None) == 409
    with pytest.raises(Exception) as excinfo:
        _create(agency_id="AGY_T2")
    assert getattr(excinfo.value, "status_code", None) == 409


def test_validation_rejects_bad_values(temp_stores) -> None:
    for kwargs in (
        {"rebate_percent": 120.0},
        {"commission_percent": -1.0},
        {"payment_terms_days": -5},
        {"status": "deleted"},
        {"agency_type": "לא קיים"},
        {"data_source": "guessed"},
    ):
        with pytest.raises(Exception) as excinfo:
            _create(**kwargs)
        assert getattr(excinfo.value, "status_code", None) == 400


def test_deactivate_suspends_instead_of_deleting(temp_stores) -> None:
    _create()
    record = ag.deactivate_agency("AGY_T1")
    assert record["status"] == "suspended"
    assert ag.get_agency("AGY_T1")["status"] == "suspended"
    back = ag.update_agency("AGY_T1", ag.AgencyUpdate(status="active"))
    assert back["status"] == "active"


def test_missing_agency_is_404(temp_stores) -> None:
    with pytest.raises(Exception) as excinfo:
        ag.get_agency("AGY_NOPE")
    assert getattr(excinfo.value, "status_code", None) == 404


# --- link derivation and manual override --------------------------------------

def test_observed_pairs_from_frame() -> None:
    daily = pd.DataFrame({
        "advertiser": ["מפרסם א", "מפרסם ב", "מפרסם א", ""],
        "agency": ["סוכנות בדיקה", "סוכנות אחרת", "סוכנות בדיקה", "סוכנות בדיקה"],
    })
    pairs = agc.observed_pairs_from_frame(daily)
    assert {(p["advertiser"], p["agency"]) for p in pairs} == {
        ("מפרסם א", "סוכנות בדיקה"), ("מפרסם ב", "סוכנות אחרת"),
    }


def test_links_merge_observed_with_manual_override(temp_stores, monkeypatch) -> None:
    _create()
    _create(agency_id="AGY_T2", name="סוכנות אחרת")
    daily_pairs = [
        {"advertiser": "מפרסם א", "agency": "סוכנות בדיקה"},
        {"advertiser": "מפרסם ב", "agency": "סוכנות בדיקה"},
    ]
    monkeypatch.setattr(agc, "_latest_daily_pairs", lambda: (daily_pairs, "fixture.csv"))

    links = agc.links_for("AGY_T1")
    assert links["observed"] == ["מפרסם א", "מפרסם ב"]
    assert links["effective"] == ["מפרסם א", "מפרסם ב"]

    # A manual link moves מפרסם ב to the other agency and wins over observed.
    agc.create_link("AGY_T2", agc.LinkCreate(advertiser="מפרסם ב"))
    assert agc.links_for("AGY_T1")["effective"] == ["מפרסם א"]
    assert agc.links_for("AGY_T2")["effective"] == ["מפרסם ב"]

    # One manual link per advertiser: linking it again elsewhere is a 409.
    with pytest.raises(Exception) as excinfo:
        agc.create_link("AGY_T1", agc.LinkCreate(advertiser="מפרסם ב"))
    assert getattr(excinfo.value, "status_code", None) == 409

    agc.delete_link("AGY_T2", "מפרסם ב")
    assert agc.links_for("AGY_T1")["effective"] == ["מפרסם א", "מפרסם ב"]


# --- conditions CRUD ----------------------------------------------------------

def test_condition_crud_roundtrip(temp_stores) -> None:
    _create()
    record = agc.create_condition("AGY_T1", agc.ConditionCreate(
        rule_id="r1", effect="premium", value=15.0, mode="percent", scope_dayparts="prime",
    ))
    assert record["mode"] == "percent"
    assert record["scope_dayparts"] == "prime"

    with pytest.raises(Exception) as excinfo:
        agc.create_condition("AGY_T1", agc.ConditionCreate(rule_id="r1", effect="forbid"))
    assert getattr(excinfo.value, "status_code", None) == 409

    updated = agc.update_condition("AGY_T1", "r1", agc.ConditionUpdate(value=1.1, mode="multiplier"))
    assert updated["value"] == pytest.approx(1.1)
    assert agc.list_conditions("AGY_T1")["conditions"][0]["mode"] == "multiplier"

    agc.delete_condition("AGY_T1", "r1")
    assert agc.conditions_for("AGY_T1") == []


def test_condition_rejects_unknown_effect(temp_stores) -> None:
    _create()
    with pytest.raises(Exception) as excinfo:
        agc.create_condition("AGY_T1", agc.ConditionCreate(rule_id="r1", effect="nonsense"))
    assert getattr(excinfo.value, "status_code", None) == 400


# --- pricing-path composition -------------------------------------------------

def _daily_frame(**overrides) -> pd.DataFrame:
    row = {
        "advertiser": "מפרסם א",
        "campaign": "קמפיין",
        "program": "תוכנית",
        "position_in_break": 2,
        "planned_tvr": 5.0,
        "duration_sec": 30.0,
        "pricing_type": "CPP",
        "price": None,
        "spot_time": "21:00:00",
        "agency": "סוכנות בדיקה",
        "date": "2025-04-27",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _layer(conditions=None, rebate=0.0, active=True) -> AgencyLayer:
    layer = AgencyLayer()
    layer.terms["AGY_T1"] = AgencyTerms(
        agency_id="AGY_T1", name="סוכנות בדיקה", rebate_percent=rebate, active=active,
    )
    layer.by_name["סוכנות בדיקה"] = "AGY_T1"
    layer.engine = AdvertiserRuleEngine(
        baselines={}, conditions={"AGY_T1": conditions or []},
    )
    return layer


def _neutral_engine() -> AdvertiserRuleEngine:
    return AdvertiserRuleEngine(baselines={}, conditions={})


PRICING = PricingModel(base_price_per_second_per_tvr_point=10.0)


def _price(daily, layer, engine=None):
    return price_daily_spots(
        daily, engine=engine or _neutral_engine(), pricing=PRICING,
        frequency=FrequencyRuleSet(), agency_layer=layer,
    )


def test_agency_forbid_wins_over_advertiser_allow() -> None:
    forbid = Condition(advertiser_id="AGY_T1", rule_id="f1", effect="forbid")
    result = _price(_daily_frame(), _layer(conditions=[forbid]))
    assert result.priced == []
    assert len(result.dropped) == 1
    drop = result.dropped[0]
    assert drop.agency == "סוכנות בדיקה"
    assert drop.reason.startswith("agency")
    assert "f1" in drop.reason


def test_agency_premium_composes_with_advertiser_premium() -> None:
    premium = Condition(advertiser_id="AGY_T1", rule_id="p1", effect="premium", value=1.2)
    adv = Condition(advertiser_id="מפרסם א", rule_id="a1", effect="premium", value=1.5)
    engine = AdvertiserRuleEngine(baselines={}, conditions={"מפרסם א": [adv]})
    result = _price(_daily_frame(), _layer(conditions=[premium]), engine=engine)
    spot = result.priced[0]
    assert spot.agency_premium == pytest.approx(1.2)
    assert spot.premium == pytest.approx(1.8)
    # 10 ILS/s/point x 5.0 TVR x 30 s x 1.8 = 2700
    assert spot.revenue == pytest.approx(2700.0)


def test_rebate_yields_net_beside_unchanged_gross() -> None:
    plain = _price(_daily_frame(), AgencyLayer())
    rebated = _price(_daily_frame(), _layer(rebate=10.0))
    assert rebated.priced[0].revenue == plain.priced[0].revenue  # gross untouched
    assert rebated.priced[0].net_revenue == pytest.approx(round(plain.priced[0].revenue * 0.9, 2))
    assert rebated.priced[0].rebate_percent == pytest.approx(10.0)
    assert rebated.total_net_revenue == pytest.approx(round(plain.total_revenue * 0.9, 2))


def test_suspended_agency_terms_are_inert() -> None:
    forbid = Condition(advertiser_id="AGY_T1", rule_id="f1", effect="forbid")
    result = _price(_daily_frame(), _layer(conditions=[forbid], rebate=10.0, active=False))
    assert len(result.priced) == 1  # forbid not applied
    spot = result.priced[0]
    assert spot.agency == "סוכנות בדיקה"  # still resolved for the record
    assert spot.rebate_percent == pytest.approx(0.0)
    assert spot.net_revenue == pytest.approx(spot.revenue)


def test_gross_identity_with_layer_inert() -> None:
    """No agency conditions and rebate 0: gross is exactly the no-layer figure."""
    without = _price(_daily_frame(), AgencyLayer())
    inert = _price(_daily_frame(), _layer(rebate=0.0))
    assert inert.priced[0].revenue == without.priced[0].revenue
    assert inert.priced[0].premium == without.priced[0].premium
    assert inert.priced[0].placement_value == without.priced[0].placement_value
    assert inert.priced[0].net_revenue == inert.priced[0].revenue


@pytest.mark.skipif(not DAILY_FIXTURE.exists(), reason="real daily file not present")
def test_shipped_seed_moves_no_gross_on_the_real_ledger() -> None:
    """The shipped seed (header-only conditions) prices the real daily file to the
    same gross, spot for spot, as a run with no agency layer at all."""
    from kairos.data.loaders import load_daily_input

    daily = load_daily_input(DAILY_FIXTURE)
    seeded = price_daily_spots(daily, pricing=PRICING, frequency=FrequencyRuleSet())
    bare = price_daily_spots(
        daily, pricing=PRICING, frequency=FrequencyRuleSet(), agency_layer=AgencyLayer(),
    )
    assert [s.revenue for s in seeded.priced] == [s.revenue for s in bare.priced]
    assert seeded.total_revenue == bare.total_revenue
    # Every spot resolves its agency from the file's own column, and the net
    # figure follows the seeded (synthetic, labeled) rebate exactly.
    assert all(s.agency for s in seeded.priced)
    for spot in seeded.priced:
        assert spot.net_revenue == pytest.approx(round(spot.revenue * (1 - spot.rebate_percent / 100.0), 2))
    assert seeded.total_net_revenue < seeded.total_revenue


# --- seed data honesty --------------------------------------------------------

def test_seeded_agencies_are_real_names_with_labeled_synthetic_details() -> None:
    frame = pd.read_csv(ROOT / "data" / "agencies.csv", encoding="utf-8-sig", dtype=str)
    assert set(frame["name"]) == SEED_AGENCY_NAMES
    assert len(frame) == 9
    assert (frame["data_source"] == "synthetic").all()
    assert frame["notes"].str.contains("סינתטיים").all()
    assert frame["contact_email"].str.contains(".example").all()


@pytest.mark.skipif(not DAILY_FIXTURE.exists(), reason="real daily file not present")
def test_seeded_links_match_the_measured_daily_mapping() -> None:
    from kairos.data.loaders import load_daily_input

    links = pd.read_csv(ROOT / "data" / "agency_advertisers.csv", encoding="utf-8-sig", dtype=str)
    assert (links["source"] == "observed").all()
    agencies = pd.read_csv(ROOT / "data" / "agencies.csv", encoding="utf-8-sig", dtype=str)
    id_to_name = dict(zip(agencies["agency_id"], agencies["name"]))
    seeded_pairs = {(row.advertiser, id_to_name[row.agency_id]) for row in links.itertuples(index=False)}
    daily = load_daily_input(DAILY_FIXTURE)
    measured_pairs = {(p["advertiser"], p["agency"]) for p in agc.observed_pairs_from_frame(daily)}
    assert seeded_pairs == measured_pairs


def test_seeded_conditions_file_is_header_only() -> None:
    frame = pd.read_csv(ROOT / "data" / "agency_conditions.csv", encoding="utf-8-sig")
    assert len(frame) == 0
    assert list(frame.columns) == agc.CONDITION_COLUMNS


# --- cross-level overlap detection --------------------------------------------

def test_cross_level_conflict_is_reported(temp_stores, monkeypatch) -> None:
    import kairos.optimize.advertiser_rules as rules_module

    _create()
    monkeypatch.setattr(agc, "_latest_daily_pairs",
                        lambda: ([{"advertiser": "מפרסם א", "agency": "סוכנות בדיקה"}], "f.csv"))
    agc.create_condition("AGY_T1", agc.ConditionCreate(
        rule_id="fx", effect="forbid", scope_dayparts="prime",
    ))
    adv_conditions = temp_stores / "advertiser_conditions.csv"
    pd.DataFrame([{
        "advertiser_id": "מפרסם א", "rule_id": "rq", "effect": "require",
        "value": "1.0", "scope_positions": "ANY", "scope_genres": "ANY",
        "scope_dayparts": "prime", "scope_programmes": "ANY", "mode": "multiplier", "notes": "",
    }]).to_csv(adv_conditions, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(rules_module, "DEFAULT_CONDITIONS_PATH", adv_conditions)

    findings = agc.cross_level_overlaps("AGY_T1")
    assert any(f["kind"] == "cross_level_conflict" for f in findings)
    conflict = next(f for f in findings if f["kind"] == "cross_level_conflict")
    assert conflict["agency_rule_id"] == "fx"
    assert conflict["advertiser_rule_id"] == "rq"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
