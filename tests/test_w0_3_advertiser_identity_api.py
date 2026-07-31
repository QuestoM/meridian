"""The advertiser identity API: the binding CRUD and the identity read.

The rules store now carries ``name`` and ``aliases`` beside ``display_name``,
the same three columns ``data/agencies.csv`` has. These tests cover what the API
does with them: the round trip, the refusal of a name another row already holds,
the tolerant read of a store written before the columns existed, and the shape
the identity read returns. Every case runs against a throwaway store, so the
shipped one is never written.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import kairos_api.advertisers as adv

LEGACY_HEADER = [
    "advertiser_id", "default_premium", "allow_positions", "allow_genres",
    "prime_time_only", "urgency_k", "ahead_k", "notes",
]
FULL_HEADER = LEGACY_HEADER + ["name", "display_name", "aliases"]


def _write(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(advertiser_id: str, premium: str = "1.0", **extra) -> dict:
    base = {
        "advertiser_id": advertiser_id, "default_premium": premium,
        "allow_positions": "ANY", "allow_genres": "ANY", "prime_time_only": "False",
        "urgency_k": "", "ahead_k": "", "notes": "",
        "name": "", "display_name": "", "aliases": "",
    }
    base.update(extra)
    return base


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Point the rules store, its side files and the engine at throwaway csvs."""
    import kairos.optimize.advertiser_rules as engine_module
    import kairos_api.advertiser_conditions as conditions_module

    rules = tmp_path / "advertiser_rules.csv"
    conditions = tmp_path / "advertiser_conditions.csv"
    _write(conditions, ["advertiser_id", "rule_id", "scope_positions", "scope_genres",
                        "scope_dayparts", "effect", "value", "notes"], [])
    monkeypatch.setattr(adv, "RULES_PATH", rules)
    monkeypatch.setattr(adv, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(adv, "OBSERVED_NAMES_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(conditions_module, "CONDITIONS_PATH", conditions)
    monkeypatch.setattr(conditions_module, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(engine_module, "DEFAULT_RULES_PATH", rules)
    monkeypatch.setattr(engine_module, "DEFAULT_CONDITIONS_PATH", conditions)
    return tmp_path


# --- the binding round trip ----------------------------------------------------


def test_put_stores_the_name_and_the_aliases(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER, [_row("ADV_01")])
    updated = adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(
        name="  בנק הפועלים  ", aliases="Bank Hapoalim| פועלים |",
    ))
    assert updated["name"] == "בנק הפועלים"
    assert updated["aliases"] == "Bank Hapoalim|פועלים"

    persisted = adv.list_advertisers()["advertisers"][0]
    assert persisted["name"] == "בנק הפועלים"
    assert persisted["aliases"] == "Bank Hapoalim|פועלים"


def test_a_bound_row_is_no_longer_unnamed(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER, [_row("ADV_01")])
    assert adv.list_advertisers()["advertisers"][0]["name_source"] == "unnamed"
    updated = adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(name="אסם"))
    assert updated["name_source"] == "operator"


def test_clearing_the_name_unbinds_the_row(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER, [_row("ADV_01", name="אסם")])
    cleared = adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(name="", aliases=""))
    assert cleared["name"] == ""
    assert cleared["aliases"] == ""
    assert cleared["name_source"] == "unnamed"


def test_a_name_edit_touches_no_other_field(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER, [_row("ADV_02", "1.27")])
    before = adv.list_advertisers()["advertisers"][0]
    adv.update_advertiser("ADV_02", adv.AdvertiserUpdate(name="שטראוס"))
    after = adv.list_advertisers()["advertisers"][0]
    for key in ("default_premium", "allow_positions", "allow_genres",
                "prime_time_only", "urgency_k", "ahead_k", "notes"):
        assert after[key] == before[key]


def test_create_carries_the_name_and_the_aliases(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER, [])
    created = adv.create_advertiser(adv.AdvertiserCreate(
        advertiser_id="ADV_90", name="טבע", aliases="Teva",
    ))
    assert created["name"] == "טבע"
    assert created["aliases"] == "Teva"
    bare = adv.create_advertiser(adv.AdvertiserCreate(advertiser_id="ADV_91"))
    assert bare["name"] == "" and bare["aliases"] == ""


# --- one advertiser, one row ---------------------------------------------------


def test_a_name_another_row_holds_is_refused(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER,
           [_row("ADV_01", name="אסם"), _row("ADV_02")])
    with pytest.raises(Exception) as error:
        adv.update_advertiser("ADV_02", adv.AdvertiserUpdate(name="אסם"))
    assert getattr(error.value, "status_code", None) == 409


def test_an_alias_another_row_holds_is_refused(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER,
           [_row("ADV_01", aliases="Osem"), _row("ADV_02")])
    with pytest.raises(Exception) as error:
        adv.update_advertiser("ADV_02", adv.AdvertiserUpdate(aliases="Osem|Other"))
    assert getattr(error.value, "status_code", None) == 409


def test_a_row_may_keep_its_own_name(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER, [_row("ADV_01", name="אסם")])
    updated = adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(name="אסם", notes="שוב"))
    assert updated["name"] == "אסם"
    assert updated["notes"] == "שוב"


def test_a_name_that_is_another_rows_id_is_refused(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER,
           [_row("ADV_01"), _row("כלמוביל")])
    with pytest.raises(Exception) as error:
        adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(name="כלמוביל"))
    assert getattr(error.value, "status_code", None) == 409


# --- the tolerant read ---------------------------------------------------------


def test_a_store_written_before_the_columns_reads_as_unbound(store) -> None:
    _write(store / "advertiser_rules.csv", LEGACY_HEADER,
           [_row("ADV_01"), _row("ADV_02", "1.27")])
    records = adv.list_advertisers()["advertisers"]
    assert [r["advertiser_id"] for r in records] == ["ADV_01", "ADV_02"]
    assert all(r["name"] == "" and r["aliases"] == "" for r in records)
    assert all(r["name_source"] == "unnamed" for r in records)
    assert records[1]["default_premium"] == pytest.approx(1.27)


def test_a_write_keeps_the_identity_columns_on_disk(store) -> None:
    _write(store / "advertiser_rules.csv", FULL_HEADER, [_row("ADV_01", name="אסם")])
    adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(notes="לא נוגע בשם"))
    with open(store / "advertiser_rules.csv", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert "name" in rows[0] and "aliases" in rows[0]
    assert rows[0]["name"] == "אסם"


# --- the identity read ---------------------------------------------------------


def test_identity_report_names_every_observed_advertiser(store, monkeypatch) -> None:
    import kairos_api.advertisers_identity as identity
    import kairos_api.spot_ledger as ledger_module

    names = store / "advertiser_names.csv"
    _write(names, ["name", "display_name", "aliases", "source", "first_seen", "notes"], [
        {"name": "אסם", "display_name": "", "aliases": "Osem",
         "source": "observed", "first_seen": "2025-04-27", "notes": ""},
        {"name": "שטראוס", "display_name": "", "aliases": "",
         "source": "observed", "first_seen": "2025-04-27", "notes": ""},
    ])
    _write(store / "advertiser_rules.csv", FULL_HEADER, [_row("ADV_01", "1.27", name="אסם")])
    monkeypatch.setattr(identity, "_names_path", lambda: names)
    monkeypatch.setattr(ledger_module, "read_ledger", lambda: ledger_module.LedgerRead(
        reason="no daily file in this test",
    ))

    report = identity.identity_report()
    assert report["count"] == 2
    assert report["resolved"] == 2
    assert report["bound_to_a_rules_row"] == 1
    by_name = {record["advertiser"]: record for record in report["advertisers"]}
    assert by_name["אסם"]["rules"]["bound"] is True
    assert by_name["אסם"]["rules"]["advertiser_id"] == "ADV_01"
    assert by_name["אסם"]["rules"]["effective_premium"] == pytest.approx(1.27)
    assert by_name["שטראוס"]["rules"]["bound"] is False
    assert by_name["שטראוס"]["rules"]["effective_premium"] == pytest.approx(1.0)
    assert by_name["שטראוס"]["rules"]["reason"] == identity.UNBOUND_REASON
    assert by_name["אסם"]["money"] is None


def test_identity_report_lists_a_daily_advertiser_the_name_space_lacks(store, monkeypatch) -> None:
    import kairos_api.advertisers_identity as identity
    import kairos_api.spot_ledger as ledger_module

    names = store / "advertiser_names.csv"
    _write(names, ["name", "display_name", "aliases", "source", "first_seen", "notes"], [])
    _write(store / "advertiser_rules.csv", FULL_HEADER, [])
    monkeypatch.setattr(identity, "_names_path", lambda: names)
    monkeypatch.setattr(ledger_module, "read_ledger", lambda: ledger_module.LedgerRead(
        available=True, basis="Wally_test.csv", gross=100.0, net=100.0, spots=1,
        dropped_by_rule=0, dropped_by_frequency=0,
        by_advertiser={"מפרסם חדש": ledger_module.AdvertiserMoney(
            advertiser="מפרסם חדש", gross=100.0, net=100.0, spots=1,
        )},
    ))

    report = identity.identity_report()
    assert report["unresolved"] == ["מפרסם חדש"]
    record = report["advertisers"][0]
    assert record["advertiser"] == "מפרסם חדש"
    assert record["resolved"] is False
    assert record["rules"]["reason"] == identity.UNRESOLVED_REASON
    assert record["money"]["gross"] == pytest.approx(100.0)


def test_resolve_one_answers_for_a_single_name(store, monkeypatch) -> None:
    import kairos_api.advertisers_identity as identity
    import kairos_api.spot_ledger as ledger_module

    names = store / "advertiser_names.csv"
    _write(names, ["name", "display_name", "aliases", "source", "first_seen", "notes"], [
        {"name": "אסם", "display_name": "אסם ישראל", "aliases": "Osem",
         "source": "observed", "first_seen": "2025-04-27", "notes": ""},
    ])
    _write(store / "advertiser_rules.csv", FULL_HEADER, [])
    monkeypatch.setattr(identity, "_names_path", lambda: names)
    monkeypatch.setattr(ledger_module, "read_ledger", lambda: ledger_module.LedgerRead(
        reason="no daily file in this test",
    ))

    found = identity.resolve_one("Osem")
    assert found["resolved"] is True
    assert found["advertiser"] == "אסם"
    assert found["shown_name"] == "אסם ישראל"
    assert found["matched_on"] == "alias"
    assert identity.resolve_one("מישהו אחר")["resolved"] is False
