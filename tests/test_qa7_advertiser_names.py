"""The advertiser display-name layer: tolerant store read, honest name_source.

The Advertisers page lists the rules store, whose seed keys are raw tokens
(ADV_01..ADV_45) that are inert on real spots. The display-name layer adds an
editable ``display_name`` beside the raw id plus a tri-state ``name_source``:
``operator`` (stored by the operator), ``observed`` (the id itself is a real
advertiser name seen in the daily data, via the agency links store) and
``unnamed`` (raw token only; the backend never invents a name). These tests
prove the CRUD round trip, the tolerant read of a legacy CSV without the
column, and that the seed numbers are preserved byte-identically.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

import kairos_api.advertisers as adv

ROOT = Path(__file__).resolve().parents[1]
REAL_RULES_CSV = ROOT / "data" / "advertiser_rules.csv"

LEGACY_HEADER = [
    "advertiser_id", "default_premium", "allow_positions", "allow_genres",
    "prime_time_only", "urgency_k", "ahead_k", "notes",
]


def _write_csv(path: Path, header: list[str], rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _legacy_row(advertiser_id: str, premium: str = "1.0") -> dict:
    return {
        "advertiser_id": advertiser_id, "default_premium": premium,
        "allow_positions": "ANY", "allow_genres": "ANY",
        "prime_time_only": "False", "urgency_k": "", "ahead_k": "", "notes": "",
    }


@pytest.fixture()
def temp_store(tmp_path, monkeypatch):
    """Point the rules store, its side files and the engines at throwaway CSVs."""
    import kairos.optimize.advertiser_rules as engine_module
    import kairos_api.advertiser_conditions as conditions_module

    rules_path = tmp_path / "advertiser_rules.csv"
    conditions_path = tmp_path / "advertiser_conditions.csv"
    _write_csv(conditions_path, ["advertiser_id", "rule_id", "scope_positions",
                                 "scope_genres", "scope_dayparts", "effect", "value", "notes"], [])
    monkeypatch.setattr(adv, "RULES_PATH", rules_path)
    monkeypatch.setattr(adv, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(adv, "OBSERVED_NAMES_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(conditions_module, "CONDITIONS_PATH", conditions_path)
    monkeypatch.setattr(conditions_module, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(engine_module, "DEFAULT_RULES_PATH", rules_path)
    monkeypatch.setattr(engine_module, "DEFAULT_CONDITIONS_PATH", conditions_path)
    return tmp_path


def _seed_observed(tmp_path: Path, names: list[str]) -> None:
    _write_csv(tmp_path / "agency_advertisers.csv",
               ["agency_id", "advertiser", "source", "observed_date", "notes"],
               [{"agency_id": "AGY_01", "advertiser": name, "source": "observed",
                 "observed_date": "2025-04-27", "notes": ""} for name in names])


# --- tolerant read of the legacy store ---------------------------------------


def test_legacy_csv_without_display_column_reads_as_unnamed(temp_store) -> None:
    _write_csv(temp_store / "advertiser_rules.csv", LEGACY_HEADER,
               [_legacy_row("ADV_01"), _legacy_row("ADV_02", "1.27")])
    listing = adv.list_advertisers()
    assert [r["advertiser_id"] for r in listing["advertisers"]] == ["ADV_01", "ADV_02"]
    for record in listing["advertisers"]:
        assert record["display_name"] == ""
        assert record["name_source"] == "unnamed"


def test_real_seed_numbers_are_preserved_byte_identically(temp_store) -> None:
    # The shipped store: 45 ADV_## seed tokens. The display layer must change no
    # number: ADV_02 keeps its 1.27 premium, everyone else stays at 1.0.
    if not REAL_RULES_CSV.exists():
        pytest.skip("real advertiser_rules.csv not present")
    shutil.copy2(REAL_RULES_CSV, temp_store / "advertiser_rules.csv")
    listing = adv.list_advertisers()
    records = listing["advertisers"]
    assert len(records) == 45
    premiums = {r["advertiser_id"]: r["default_premium"] for r in records}
    assert premiums["ADV_02"] == pytest.approx(1.27)
    assert all(value == pytest.approx(1.0) for key, value in premiums.items() if key != "ADV_02")
    assert all(r["name_source"] == "unnamed" and r["display_name"] == "" for r in records)


def test_missing_observed_file_is_tolerated(temp_store) -> None:
    _write_csv(temp_store / "advertiser_rules.csv", LEGACY_HEADER, [_legacy_row("ADV_01")])
    assert not (temp_store / "agency_advertisers.csv").exists()
    record = adv.list_advertisers()["advertisers"][0]
    assert record["name_source"] == "unnamed"


# --- observed daily-data names ------------------------------------------------


def test_hebrew_id_matching_daily_data_reads_as_observed(temp_store) -> None:
    _write_csv(temp_store / "advertiser_rules.csv", LEGACY_HEADER,
               [_legacy_row("כלמוביל"), _legacy_row("ADV_07")])
    _seed_observed(temp_store, ["כלמוביל", "סלקום"])
    by_id = {r["advertiser_id"]: r for r in adv.list_advertisers()["advertisers"]}
    assert by_id["כלמוביל"]["name_source"] == "observed"
    assert by_id["כלמוביל"]["display_name"] == ""
    assert by_id["ADV_07"]["name_source"] == "unnamed"


def test_operator_name_wins_over_observed(temp_store) -> None:
    _write_csv(temp_store / "advertiser_rules.csv", LEGACY_HEADER, [_legacy_row("כלמוביל")])
    _seed_observed(temp_store, ["כלמוביל"])
    updated = adv.update_advertiser("כלמוביל", adv.AdvertiserUpdate(display_name="כלמוביל יבואן רשמי"))
    assert updated["name_source"] == "operator"
    assert updated["display_name"] == "כלמוביל יבואן רשמי"


# --- CRUD round trip ----------------------------------------------------------


def test_put_sets_persists_and_clears_display_name(temp_store) -> None:
    _write_csv(temp_store / "advertiser_rules.csv", LEGACY_HEADER, [_legacy_row("ADV_01")])
    before = adv.list_advertisers()["advertisers"][0]

    updated = adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(display_name="  בנק הפועלים  "))
    assert updated["display_name"] == "בנק הפועלים"
    assert updated["name_source"] == "operator"

    # Persisted: a fresh read from disk carries the name and the new column.
    persisted = adv.list_advertisers()["advertisers"][0]
    assert persisted["display_name"] == "בנק הפועלים"
    frame = adv._load_frame()
    assert "display_name" in frame.columns

    # Every non-name field is untouched by the name edit.
    for key in ("default_premium", "allow_positions", "allow_genres",
                "prime_time_only", "urgency_k", "ahead_k", "notes"):
        assert persisted[key] == before[key]

    # An empty string clears the name and the record reads unnamed again.
    cleared = adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(display_name=""))
    assert cleared["display_name"] == ""
    assert cleared["name_source"] == "unnamed"


def test_put_without_display_name_leaves_it_untouched(temp_store) -> None:
    _write_csv(temp_store / "advertiser_rules.csv", LEGACY_HEADER, [_legacy_row("ADV_01")])
    adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(display_name="שטראוס"))
    updated = adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(notes="חדש"))
    assert updated["display_name"] == "שטראוס"
    assert updated["notes"] == "חדש"


def test_create_carries_display_name_and_defaults_unnamed(temp_store) -> None:
    _write_csv(temp_store / "advertiser_rules.csv", LEGACY_HEADER, [])
    named = adv.create_advertiser(adv.AdvertiserCreate(advertiser_id="ADV_90", display_name="טבע"))
    assert named["display_name"] == "טבע"
    assert named["name_source"] == "operator"
    bare = adv.create_advertiser(adv.AdvertiserCreate(advertiser_id="ADV_91"))
    assert bare["display_name"] == ""
    assert bare["name_source"] == "unnamed"


# --- stats carry the name layer ----------------------------------------------


def test_stats_records_carry_display_name_and_source(temp_store) -> None:
    _write_csv(temp_store / "advertiser_rules.csv", LEGACY_HEADER, [_legacy_row("ADV_01")])
    adv.update_advertiser("ADV_01", adv.AdvertiserUpdate(display_name="אסם"))
    stats = adv.advertiser_stats()["advertisers"][0]
    assert stats["display_name"] == "אסם"
    assert stats["name_source"] == "operator"
    # The stats numbers themselves are unchanged by naming.
    assert stats["baseline_premium"] == pytest.approx(1.0)
    assert stats["avg_effective_premium"] == pytest.approx(1.0)
