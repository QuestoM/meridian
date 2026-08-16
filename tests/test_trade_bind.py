"""Binding writes: snapshot-first, idempotent per agreement, refusal over
widening, and byte identity when no agreement holds rules."""

import hashlib
from pathlib import Path

import pytest

from kairos.trade.compile import CompiledArtifacts
from kairos_api import trade_bind


@pytest.fixture()
def stores(tmp_path, monkeypatch):
    from kairos.optimize import _frequency_rules
    from kairos_api import advertiser_conditions, agency_conditions

    adv = tmp_path / "advertiser_conditions.csv"
    agc = tmp_path / "agency_conditions.csv"
    freq = tmp_path / "frequency_rules.csv"
    adv.write_text(
        "advertiser_id,rule_id,scope_positions,scope_genres,scope_dayparts,"
        "scope_programmes,scope_weekdays,effect,value,mode,notes\n"
        "מפרסם קיים,R_EXISTING,ANY,ANY,ANY,ANY,ANY,premium,1.1,multiplier,קיים\n",
        encoding="utf-8-sig",
    )
    agc.write_text(
        "agency_id,rule_id,scope_positions,scope_genres,scope_dayparts,"
        "scope_programmes,effect,value,mode,notes\n",
        encoding="utf-8-sig",
    )
    freq.write_text(
        "rule_id,limit_type,scope,advertiser_id,campaign,ad,pair_lead,pair_closer,"
        "competing_group,members,value,value_max,unit,enabled,notes\n"
        "DEFAULT_ONE,max_per_break,default,,,,,,,,1,,,True,ברירת מחדל\n",
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(advertiser_conditions, "CONDITIONS_PATH", adv)
    monkeypatch.setattr(agency_conditions, "CONDITIONS_PATH", agc)
    monkeypatch.setattr(_frequency_rules, "DEFAULT_FREQUENCY_PATH", freq)
    monkeypatch.setenv("KAIROS_VERSIONS_DIR", str(tmp_path / "versions"))
    return {"adv": adv, "agc": agc, "freq": freq, "versions": tmp_path / "versions"}


def _artifacts(**overrides):
    art = CompiledArtifacts(agreement_id="agr-1", version_id="v-1")
    art.conditions = overrides.get("conditions", [
        {
            "_store": "advertiser_conditions",
            "advertiser_id": "Delta Motors",
            "rule_id": "TRD:agr-1:v-1:i-a",
            "scope_positions": "ANY", "scope_genres": "חדשות",
            "scope_dayparts": "ANY", "scope_programmes": "ANY",
            "scope_campaigns": "ANY", "scope_weekdays": "ANY",
            "effect": "forbid", "value": 1.0, "mode": "multiplier",
            "notes": "הרחקת תוכן [doc/5.4]",
        },
    ])
    art.frequency_rules = overrides.get("frequency_rules", [
        {
            "rule_id": "TRD:agr-1:v-1:i-f",
            "limit_type": "max_per_day", "scope": "advertiser",
            "advertiser_id": "Delta Motors", "campaign": "", "ad": "",
            "pair_lead": "", "pair_closer": "", "competing_group": "",
            "members": "", "value": 4, "value_max": "", "unit": "",
            "enabled": True, "notes": "תקרה יומית",
        },
    ])
    return art


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_bind_writes_rows_and_snapshots_first(stores):
    result = trade_bind.bind(_artifacts(), actor="dana")
    assert result["written"] == {"advertiser_conditions": 1, "frequency_rules": 1}
    assert result["refused"] == []
    assert result["snapshot_version"], "the touched stores must be versioned first"
    adv_text = stores["adv"].read_text(encoding="utf-8-sig")
    assert "TRD:agr-1:v-1:i-a" in adv_text
    assert "R_EXISTING" in adv_text, "existing rows survive untouched"
    manifests = list((stores["versions"]).glob("*/manifest.json"))
    assert manifests, "a version manifest exists for the bind snapshot"


def test_rebind_replaces_the_same_agreements_rows_only(stores):
    trade_bind.bind(_artifacts(), actor="dana")
    v2 = _artifacts()
    v2.version_id = "v-2"
    v2.conditions[0]["rule_id"] = "TRD:agr-1:v-2:i-a"
    v2.frequency_rules = []
    result = trade_bind.bind(v2, actor="dana")
    assert result["written"] == {"advertiser_conditions": 1}
    assert result["replaced"] == {"advertiser_conditions": 1, "frequency_rules": 1}
    adv_text = stores["adv"].read_text(encoding="utf-8-sig")
    assert "TRD:agr-1:v-2:i-a" in adv_text
    assert "TRD:agr-1:v-1:i-a" not in adv_text
    freq_text = stores["freq"].read_text(encoding="utf-8-sig")
    assert "TRD:agr-1:v-1:i-f" not in freq_text, "old frequency rows removed"
    assert "DEFAULT_ONE" in freq_text


def test_unbind_restores_byte_identity(stores):
    before = {name: _sha(stores[name]) for name in ("adv", "agc", "freq")}
    trade_bind.bind(_artifacts(), actor="dana")
    assert _sha(stores["adv"]) != before["adv"]
    result = trade_bind.unbind("agr-1", actor="dana")
    assert result["removed"] == {"advertiser_conditions": 1, "frequency_rules": 1}
    after = {name: _sha(stores[name]) for name in ("adv", "agc", "freq")}
    assert after == before, "a full unbind leaves every store byte-identical"


def test_a_scoped_dimension_the_store_lacks_is_refused_not_widened(stores):
    # scope_campaigns exists on neither conditions store; the compiler guards
    # it upstream, and bind refuses it independently (defense in depth).
    art = _artifacts(conditions=[{
        "_store": "agency_conditions",
        "agency_id": "אופק מדיה",
        "rule_id": "TRD:agr-1:v-1:i-w",
        "scope_positions": "ANY", "scope_genres": "ANY",
        "scope_dayparts": "ANY", "scope_programmes": "ANY",
        "scope_campaigns": "CMP_9",
        "scope_weekdays": "ANY",
        "effect": "premium", "value": 1.1, "mode": "multiplier",
        "notes": "קמפיין",
    }], frequency_rules=[])
    result = trade_bind.bind(art, actor="dana")
    assert result["written"] == {}
    assert len(result["refused"]) == 1
    assert "scope_campaigns" in result["refused"][0]["reason_he"]
    agc_text = stores["agc"].read_text(encoding="utf-8-sig")
    assert "TRD:agr-1:v-1:i-w" not in agc_text


def test_weekday_scope_is_representable_on_both_condition_stores(stores):
    # The custom-pricing wave added scope_weekdays to BOTH stores; a weekday-
    # scoped agency row therefore binds rather than being refused.
    art = _artifacts(conditions=[{
        "_store": "agency_conditions",
        "agency_id": "אופק מדיה",
        "rule_id": "TRD:agr-1:v-1:i-wd",
        "scope_positions": "ANY", "scope_genres": "ANY",
        "scope_dayparts": "ANY", "scope_programmes": "ANY",
        "scope_weekdays": "5,6",
        "effect": "premium", "value": 1.1, "mode": "multiplier",
        "notes": "סופ\"ש",
    }], frequency_rules=[])
    result = trade_bind.bind(art, actor="dana")
    assert result["written"] == {"agency_conditions": 1}
    assert result["refused"] == []


def test_bound_rules_reports_the_live_rows(stores):
    trade_bind.bind(_artifacts(), actor="dana")
    held = trade_bind.bound_rules("agr-1")
    assert set(held) == {"advertiser_conditions", "frequency_rules"}
    assert held["advertiser_conditions"][0]["advertiser_id"] == "Delta Motors"
    assert trade_bind.bound_rules("agr-none") == {}
