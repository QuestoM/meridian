"""Custom pricing: weekday scopes and premium-surcharge discounts, end to end.

Covers the frozen contract in docs/advertiser-custom-pricing.md: weekday
matching on the spot's real date (Saturday, שבת, is ISO 6; a dateless caller
never matches a weekday-scoped rule), the premium_discount mode (a percent
0..100 off the premium surcharge only, composed after every other mode,
stacking multiplicatively, floored at 1.0), tolerant legacy store reads,
spot-for-spot identity on the real daily file when no rule uses the new
fields, API validation at both the advertiser and the agency level,
weekday-aware overlap detection, cross-level composition, and the events
freshness group that exists only while pricing_activation.events is on.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import kairos_api.advertiser_conditions as ac
import kairos_api.agencies as ag
import kairos_api.agency_conditions as agc
from fastapi import HTTPException

from kairos.export.agency_layer import AgencyLayer, AgencyTerms
from kairos.export.spots import price_daily_spots
from kairos.optimize._frequency_rules import FrequencyRuleSet
from kairos.optimize._rule_helpers import apply_surcharge_discount, load_conditions
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine, Baseline, Condition
from kairos.optimize.pricing import PricingModel

ROOT = Path(__file__).resolve().parents[1]
DAILY_FIXTURE = ROOT / "data" / "daily_input" / "Wally_Prime_Reshet_Example_2025-04-27.csv"

SATURDAY = "2025-04-26"   # ISO weekday 6 (שבת)
SUNDAY = "2025-04-27"     # ISO weekday 7, a regular Israeli workday

PRICING = PricingModel(base_price_per_second_per_tvr_point=10.0)
BASE_VALUE = 5.0 * 30.0 * 10.0  # planned_tvr x duration_sec x base price = 1500 ILS


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
        "agency": "",
        "date": SUNDAY,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _engine(conditions, baseline_premium: float = 1.0) -> AdvertiserRuleEngine:
    baselines = {}
    if baseline_premium != 1.0:
        baselines["מפרסם א"] = Baseline(
            advertiser_id="מפרסם א", default_premium=baseline_premium
        )
    return AdvertiserRuleEngine(baselines=baselines, conditions={"מפרסם א": list(conditions)})


def _price(daily, engine, agency_layer=None):
    return price_daily_spots(
        daily, engine=engine, pricing=PRICING,
        frequency=FrequencyRuleSet(), agency_layer=agency_layer or AgencyLayer(),
    )


# --- weekday matching ---------------------------------------------------------

def test_saturday_scope_matches_saturday_never_sunday_or_dateless() -> None:
    rule = Condition(
        advertiser_id="מפרסם א", rule_id="shabbat", effect="premium",
        value=1.5, scope_weekdays=frozenset({"6"}),
    )
    assert rule.matches(weekday=6)
    assert not rule.matches(weekday=7)
    assert not rule.matches(weekday=None), "no date must never match a weekday scope"


def test_saturday_only_rule_prices_only_the_saturday_spot() -> None:
    engine = _engine([Condition(
        advertiser_id="מפרסם א", rule_id="shabbat", effect="premium",
        value=1.5, scope_weekdays=frozenset({"6"}),
    )])
    saturday = _price(_daily_frame(date=SATURDAY), engine)
    sunday = _price(_daily_frame(date=SUNDAY), engine)
    assert saturday.priced[0].revenue == pytest.approx(BASE_VALUE * 1.5)  # 2250 ILS
    assert sunday.priced[0].revenue == pytest.approx(BASE_VALUE)          # 1500 ILS
    assert sunday.priced[0].premium == pytest.approx(1.0)


def test_saturday_only_absolute_cpp_sets_the_saturday_price() -> None:
    engine = _engine([Condition(
        advertiser_id="מפרסם א", rule_id="shabbat-cpp", effect="premium",
        value=12.0, mode="cpp_absolute", scope_weekdays=frozenset({"6"}),
    )])
    saturday = _price(_daily_frame(date=SATURDAY), engine)
    sunday = _price(_daily_frame(date=SUNDAY), engine)
    assert saturday.priced[0].revenue == pytest.approx(5.0 * 30.0 * 12.0)  # 1800 ILS
    assert sunday.priced[0].revenue == pytest.approx(BASE_VALUE)


def test_missing_date_never_matches_a_weekday_scoped_rule() -> None:
    engine = _engine([Condition(
        advertiser_id="מפרסם א", rule_id="shabbat", effect="premium",
        value=2.0, scope_weekdays=frozenset({"6"}),
    )])
    dateless = _price(_daily_frame(date=None), engine)
    assert dateless.priced[0].premium == pytest.approx(1.0), "no guessing on a dateless spot"


# --- premium_discount math ----------------------------------------------------

def test_premium_discount_math_to_the_shekel() -> None:
    # Baseline 1.3 x programme rule 1.2 = 1.56; a 50 discount keeps half the
    # surcharge: 1 + 0.56 x 0.5 = 1.28 -> 1500 x 1.28 = 1920.00 ILS.
    engine = _engine([
        Condition(advertiser_id="מפרסם א", rule_id="uplift", effect="premium", value=1.2,
                  scope_programmes=frozenset({"תוכנית"})),
        Condition(advertiser_id="מפרסם א", rule_id="deal", effect="premium", value=50.0,
                  mode="premium_discount", scope_programmes=frozenset({"תוכנית"})),
    ], baseline_premium=1.3)
    spot = _price(_daily_frame(), engine).priced[0]
    assert spot.premium == pytest.approx(1.28)
    assert spot.revenue == 1920.00


def test_discounts_stack_multiplicatively_on_the_surcharge() -> None:
    premium = apply_surcharge_discount(apply_surcharge_discount(1.4, 50.0), 50.0)
    assert premium == pytest.approx(1.1), "two 50s leave a quarter of the surcharge"


def test_discount_composes_last_regardless_of_row_order() -> None:
    uplift = Condition(advertiser_id="מפרסם א", rule_id="u", effect="premium", value=1.4)
    deal = Condition(advertiser_id="מפרסם א", rule_id="d", effect="premium",
                     value=50.0, mode="premium_discount")
    first = _engine([deal, uplift]).effective_premium("מפרסם א")
    last = _engine([uplift, deal]).effective_premium("מפרסם א")
    assert first == pytest.approx(1.2)
    assert first == last


def test_discount_floors_at_one_and_never_inverts() -> None:
    assert apply_surcharge_discount(1.4, 100.0) == pytest.approx(1.0)
    assert apply_surcharge_discount(1.4, 150.0) == pytest.approx(1.0), "clamped, never below 1.0"
    assert apply_surcharge_discount(1.4, -10.0) == pytest.approx(1.4), "clamped, never an uplift"
    assert apply_surcharge_discount(0.9, 50.0) == pytest.approx(0.9), "no surcharge, no-op"
    engine = _engine(
        [Condition(advertiser_id="מפרסם א", rule_id="d", effect="premium",
                   value=100.0, mode="premium_discount")],
        baseline_premium=1.3,
    )
    assert engine.effective_premium("מפרסם א") == pytest.approx(1.0)


def test_discount_is_not_demand_for_the_placement_steer() -> None:
    engine = _engine([Condition(
        advertiser_id="מפרסם א", rule_id="d", effect="premium",
        value=50.0, mode="premium_discount",
    )], baseline_premium=1.3)
    assert engine.segment_demand(genre=None, daypart=None, programme=None) == pytest.approx(1.0)


# --- tolerant legacy reads and identity ---------------------------------------

LEGACY_HEADER = "advertiser_id,rule_id,scope_positions,scope_genres,scope_dayparts,effect,value,notes\n"
LEGACY_ROW = "מפרסם א,old1,ANY,ANY,ANY,premium,1.2,legacy\n"


def test_legacy_store_without_weekday_column_reads_as_any(tmp_path) -> None:
    path = tmp_path / "advertiser_conditions.csv"
    path.write_text(LEGACY_HEADER + LEGACY_ROW, encoding="utf-8")
    conditions = load_conditions(path)["מפרסם א"]
    assert conditions[0].scope_weekdays == frozenset(), "missing column reads as ANY"
    assert conditions[0].matches(weekday=6) and conditions[0].matches(weekday=None)


def test_legacy_agency_store_reads_as_any(tmp_path) -> None:
    path = tmp_path / "agency_conditions.csv"
    path.write_text(
        "agency_id,rule_id,scope_positions,scope_genres,scope_dayparts,scope_programmes,effect,value,mode,notes\n"
        "AGY_T1,a1,ANY,ANY,ANY,ANY,premium,1.1,multiplier,legacy\n",
        encoding="utf-8",
    )
    layer = AgencyLayer.from_files(
        agencies_path=tmp_path / "none.csv", links_path=tmp_path / "none2.csv",
        conditions_path=path,
    )
    condition = layer.engine.conditions["AGY_T1"][0]
    assert condition.scope_weekdays == frozenset()
    assert condition.matches(weekday=6) and condition.matches(weekday=None)


@pytest.mark.skipif(not DAILY_FIXTURE.exists(), reason="real daily file not present")
def test_identity_spot_for_spot_on_the_real_daily_file(tmp_path) -> None:
    """A legacy store and the same store upgraded with scope_weekdays=ANY price
    the real daily file identically, spot for spot, to the shekel."""
    from kairos.data.loaders import load_daily_input

    legacy = tmp_path / "legacy.csv"
    legacy.write_text(LEGACY_HEADER + LEGACY_ROW, encoding="utf-8")
    upgraded = tmp_path / "upgraded.csv"
    upgraded.write_text(
        "advertiser_id,rule_id,scope_positions,scope_genres,scope_dayparts,scope_weekdays,effect,value,mode,notes\n"
        "מפרסם א,old1,ANY,ANY,ANY,ANY,premium,1.2,multiplier,legacy\n",
        encoding="utf-8",
    )
    daily = load_daily_input(DAILY_FIXTURE)
    results = [
        price_daily_spots(
            daily, pricing=PRICING, frequency=FrequencyRuleSet(),
            engine=AdvertiserRuleEngine.from_files(
                rules_path=tmp_path / "none.csv", conditions_path=path
            ),
        )
        for path in (legacy, upgraded)
    ]
    assert [s.revenue for s in results[0].priced] == [s.revenue for s in results[1].priced]
    assert [s.premium for s in results[0].priced] == [s.premium for s in results[1].priced]
    assert results[0].total_revenue == results[1].total_revenue


# --- API validation, both levels ----------------------------------------------

@pytest.fixture()
def adv_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ac, "CONDITIONS_PATH", tmp_path / "advertiser_conditions.csv")
    monkeypatch.setattr(ac, "BACKUP_DIR", tmp_path / "_backups")
    return tmp_path


@pytest.fixture()
def agency_store(tmp_path, monkeypatch):
    monkeypatch.setattr(ag, "AGENCIES_PATH", tmp_path / "agencies.csv")
    monkeypatch.setattr(ag, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agc, "LINKS_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(agc, "CONDITIONS_PATH", tmp_path / "agency_conditions.csv")
    monkeypatch.setattr(agc, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agc, "_latest_daily_pairs", lambda: ([], None))
    ag.create_agency(ag.AgencyCreate(agency_id="AGY_T1", name="סוכנות בדיקה"))
    return tmp_path


def test_advertiser_api_round_trips_weekdays_and_discount(adv_store) -> None:
    record = ac.create_condition("ACME", ac.ConditionCreate(
        rule_id="r1", effect="premium", value=25.0, mode="premium_discount",
        scope_weekdays="6,5",
    ))
    assert record["mode"] == "premium_discount"
    assert record["scope_weekdays"] == "5,6"
    stored = ac.conditions_for("ACME")[0]
    assert stored["scope_weekdays"] == "5,6"


def test_advertiser_api_rejects_bad_weekday_tokens(adv_store) -> None:
    for bad in ("0", "8", "6,9", "sat"):
        with pytest.raises(HTTPException) as excinfo:
            ac.create_condition("ACME", ac.ConditionCreate(
                rule_id="bad", effect="premium", value=1.1, scope_weekdays=bad,
            ))
        assert excinfo.value.status_code == 400


def test_advertiser_api_rejects_discount_outside_0_100(adv_store) -> None:
    for bad in (150.0, -5.0):
        with pytest.raises(HTTPException) as excinfo:
            ac.create_condition("ACME", ac.ConditionCreate(
                rule_id="bad", effect="premium", value=bad, mode="premium_discount",
            ))
        assert excinfo.value.status_code == 400


def test_advertiser_update_validates_the_effective_pair(adv_store) -> None:
    ac.create_condition("ACME", ac.ConditionCreate(rule_id="r1", effect="premium", value=150.0))
    with pytest.raises(HTTPException) as excinfo:
        ac.update_condition("ACME", "r1", ac.ConditionUpdate(mode="premium_discount"))
    assert excinfo.value.status_code == 400
    updated = ac.update_condition("ACME", "r1", ac.ConditionUpdate(
        mode="premium_discount", value=40.0, scope_weekdays="6",
    ))
    assert updated["mode"] == "premium_discount"
    assert updated["scope_weekdays"] == "6"


def test_agency_api_round_trips_and_validates(agency_store) -> None:
    record = agc.create_condition("AGY_T1", agc.ConditionCreate(
        rule_id="a1", effect="premium", value=30.0, mode="premium_discount",
        scope_weekdays="6",
    ))
    assert record["mode"] == "premium_discount"
    assert record["scope_weekdays"] == "6"
    with pytest.raises(HTTPException) as excinfo:
        agc.create_condition("AGY_T1", agc.ConditionCreate(
            rule_id="a2", effect="premium", value=120.0, mode="premium_discount",
        ))
    assert excinfo.value.status_code == 400
    with pytest.raises(HTTPException) as excinfo:
        agc.create_condition("AGY_T1", agc.ConditionCreate(
            rule_id="a3", effect="premium", value=1.1, scope_weekdays="8",
        ))
    assert excinfo.value.status_code == 400
    with pytest.raises(HTTPException) as excinfo:
        agc.update_condition("AGY_T1", "a1", agc.ConditionUpdate(value=101.0))
    assert excinfo.value.status_code == 400


def test_weekday_vocabulary_is_iso_keyed_in_israeli_order() -> None:
    options = ac.scope_options()["weekdays"]
    assert [o["key"] for o in options] == ["7", "1", "2", "3", "4", "5", "6"]
    assert options[0]["he"] == "יום ראשון"
    assert options[-1]["he"] == "שבת"
    assert "premium_discount" in ac.scope_options()["modes"]


# --- overlap detection with weekday scopes ------------------------------------

def test_disjoint_weekday_scopes_do_not_overlap() -> None:
    saturday = Condition(advertiser_id="A", rule_id="s6", effect="premium",
                         value=1.2, scope_weekdays=frozenset({"6"}))
    sunday = Condition(advertiser_id="A", rule_id="s7", effect="premium",
                       value=1.3, scope_weekdays=frozenset({"7"}))
    engine = AdvertiserRuleEngine(baselines={}, conditions={"A": [saturday, sunday]})
    assert engine.overlaps("A") == []
    both = AdvertiserRuleEngine(baselines={}, conditions={"A": [
        saturday,
        Condition(advertiser_id="A", rule_id="s6b", effect="premium",
                  value=1.1, scope_weekdays=frozenset({"6", "7"})),
    ]})
    findings = both.overlaps("A")
    assert len(findings) == 1 and findings[0].kind == "stacked_premium"


# --- cross-level composition --------------------------------------------------

def _agency_layer(conditions) -> AgencyLayer:
    layer = AgencyLayer()
    layer.terms["AGY_T1"] = AgencyTerms(agency_id="AGY_T1", name="סוכנות בדיקה")
    layer.by_name["סוכנות בדיקה"] = "AGY_T1"
    layer.engine = AdvertiserRuleEngine(baselines={}, conditions={"AGY_T1": list(conditions)})
    return layer


def test_cross_level_composition_each_discount_bites_its_own_stack() -> None:
    # Advertiser: 1.5 uplift then a 50 discount -> 1.25. Agency: 1.2 uplift then
    # a 50 discount on the AGENCY surcharge only -> 1.1. Spot premium 1.375.
    engine = _engine([
        Condition(advertiser_id="מפרסם א", rule_id="u", effect="premium", value=1.5),
        Condition(advertiser_id="מפרסם א", rule_id="d", effect="premium",
                  value=50.0, mode="premium_discount"),
    ])
    layer = _agency_layer([
        Condition(advertiser_id="AGY_T1", rule_id="au", effect="premium", value=1.2),
        Condition(advertiser_id="AGY_T1", rule_id="ad", effect="premium",
                  value=50.0, mode="premium_discount"),
    ])
    spot = _price(_daily_frame(agency="סוכנות בדיקה"), engine, agency_layer=layer).priced[0]
    assert spot.agency_premium == pytest.approx(1.1)
    assert spot.premium == pytest.approx(1.25 * 1.1)
    assert spot.revenue == pytest.approx(round(BASE_VALUE * 1.375, 2))  # 2062.50 ILS


def test_agency_saturday_forbid_drops_only_saturday_spots() -> None:
    layer = _agency_layer([Condition(
        advertiser_id="AGY_T1", rule_id="no-shabbat", effect="forbid",
        scope_weekdays=frozenset({"6"}),
    )])
    engine = _engine([])
    saturday = _price(_daily_frame(agency="סוכנות בדיקה", date=SATURDAY), engine, agency_layer=layer)
    sunday = _price(_daily_frame(agency="סוכנות בדיקה", date=SUNDAY), engine, agency_layer=layer)
    assert saturday.priced == [] and len(saturday.dropped) == 1
    assert saturday.dropped[0].reason.startswith("agency")
    assert len(sunday.priced) == 1 and sunday.dropped == []


# --- events freshness group ---------------------------------------------------

def _write_settings(root: Path, events_on: bool) -> None:
    payload = {"pricing_overrides": {"pricing_activation": {"events": events_on}}} if events_on else {}
    path = root / "data" / "kairos_settings.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_events_group_is_omitted_while_the_layer_is_off(tmp_path) -> None:
    from kairos.export.schedule_freshness import schedule_input_fingerprints

    _write_settings(tmp_path, events_on=False)
    prints = schedule_input_fingerprints(tmp_path)
    assert "events" not in prints, "off state stays fingerprint-identical to the pre-events stamp"


def test_events_group_tracks_the_store_while_on(tmp_path, monkeypatch) -> None:
    import kairos.optimize.event_pricing as event_pricing
    from kairos.export.schedule_freshness import (
        schedule_freshness,
        schedule_input_fingerprints,
        write_schedule_meta,
    )

    events_path = tmp_path / "calendar_events.csv"
    events_path.write_text(
        "event_id,name,type,start_date,end_date,intensity,notes,active,price_multiplier\n"
        "e1,מונדיאל,sports,2026-06-11,2026-07-19,3,,True,1.25\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(event_pricing, "DEFAULT_EVENTS_PATH", events_path)
    _write_settings(tmp_path, events_on=True)

    prints = schedule_input_fingerprints(tmp_path)
    assert "events" in prints and prints["events"] != "absent"

    csv_path = tmp_path / "output" / "weekly_break_schedule.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("stub\n", encoding="utf-8")
    write_schedule_meta(csv_path, tmp_path)
    assert schedule_freshness(tmp_path, csv_path)["status"] == "fresh"

    with open(events_path, "a", encoding="utf-8") as handle:
        handle.write("e2,אירוויזיון,culture,2026-05-12,2026-05-16,2,,True,1.1\n")
    verdict = schedule_freshness(tmp_path, csv_path)
    assert verdict["status"] == "stale"
    assert "special events" in verdict["changed"]
