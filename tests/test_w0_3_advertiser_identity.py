"""Advertiser identity: the name space, the resolver, and the money it unlocks.

The rules store was keyed on ADV_01..ADV_45 while every real spot names its
advertiser in Hebrew, so the two key spaces did not intersect at all: all 45
advertisers read back with no name, no revenue and no rule, and not one of them
had ever priced a spot. These tests hold the bar that replaced that state.

  * Every advertiser in the real daily file resolves to a named record with its
    rules and its money: 41 of 41, proven on the shipped file.
  * Nothing shipped moved. The daily ledger still totals gross 699,450, net
    669,978 over 119 priced spots, the agency layer still resolves 9 of 9, and
    all 45 rules rows keep their premiums and stay bound to nothing, because no
    row carries a name until the operator writes one.
  * The mechanism is real rather than decorative: binding a name in a throwaway
    store makes that row's premium price that advertiser's spots, and clearing
    it puts the money back.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path

import pytest

from kairos.export.agency_layer import AgencyLayer
from kairos.export.spots import price_daily_file
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.advertiser_rules_identity import (
    IDENTITY_COLUMNS,
    build_name_index,
    load_advertiser_names,
    normalize_name,
    resolve_advertiser,
    split_aliases,
)
from kairos_api import spot_ledger

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RULES_CSV = DATA / "advertiser_rules.csv"
NAMES_CSV = DATA / "advertiser_names.csv"
LINKS_CSV = DATA / "agency_advertisers.csv"
DAILY_CSV = DATA / "daily_input" / "Wally_Prime_Reshet_Example_2025-04-27.csv"

# The measured baseline of the shipped daily file, from docs/ux-gauntlet.
BASELINE_GROSS = 699450.0
BASELINE_NET = 669978.0
BASELINE_SPOTS = 119
BASELINE_FREQUENCY_DROPPED = 56
OBSERVED_ADVERTISERS = 41
RULES_ROWS = 45

DAILY_ADVERTISER_COLUMN = "מפרסם"


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _daily_advertisers() -> list[str]:
    return sorted({
        str(row.get(DAILY_ADVERTISER_COLUMN, "")).strip()
        for row in _rows(DAILY_CSV)
        if str(row.get(DAILY_ADVERTISER_COLUMN, "")).strip()
    })


requires_real_data = pytest.mark.skipif(
    not (RULES_CSV.exists() and NAMES_CSV.exists() and DAILY_CSV.exists()),
    reason="the shipped advertiser stores and daily file are required",
)


# --- the bar -----------------------------------------------------------------


@requires_real_data
def test_every_daily_advertiser_resolves_to_a_named_record() -> None:
    names = load_advertiser_names(NAMES_CSV)
    engine = AdvertiserRuleEngine.from_files()
    observed = _daily_advertisers()
    assert len(observed) == OBSERVED_ADVERTISERS

    resolved = {
        name: resolve_advertiser(name, names=names, rules_index=engine.names)
        for name in observed
    }
    unresolved = [name for name, record in resolved.items() if record is None]
    assert unresolved == []
    assert all(record.name == name for name, record in resolved.items())
    assert all(record.shown_name for record in resolved.values())


@requires_real_data
def test_every_daily_advertiser_has_its_rules_and_its_money() -> None:
    from kairos_api.advertisers_identity import identity_report

    report = identity_report()
    assert report["resolved"] == OBSERVED_ADVERTISERS
    assert report["unresolved"] == []
    assert report["in_ledger"] == OBSERVED_ADVERTISERS
    assert report["rules_rows"] == RULES_ROWS

    for record in report["advertisers"]:
        assert record["rules"]["effective_premium"] == pytest.approx(1.0)
        assert record["rules"]["reason"]
        assert record["money"] is not None
        assert record["money"]["basis"] == DAILY_CSV.name

    assert sum(r["money"]["gross"] for r in report["advertisers"]) == pytest.approx(BASELINE_GROSS)
    assert sum(r["money"]["spots"] for r in report["advertisers"]) == BASELINE_SPOTS


@requires_real_data
def test_the_name_space_holds_exactly_what_was_observed() -> None:
    names = load_advertiser_names(NAMES_CSV)
    linked = {
        normalize_name(row.get("advertiser", ""))
        for row in _rows(LINKS_CSV)
        if str(row.get("advertiser", "")).strip()
    }
    observed = {normalize_name(name) for name in _daily_advertisers()}
    assert set(names) == observed
    assert set(names) == linked
    assert all(record.source == "observed" for record in names.values())
    assert all(record.first_seen for record in names.values())


# --- bar 3: nothing shipped moved ---------------------------------------------


@requires_real_data
def test_daily_ledger_totals_are_unchanged() -> None:
    result = price_daily_file(DAILY_CSV)
    assert result.total_revenue == pytest.approx(BASELINE_GROSS)
    assert result.total_net_revenue == pytest.approx(BASELINE_NET)
    assert len(result.priced) == BASELINE_SPOTS
    assert len(result.dropped) == 0
    assert len(result.frequency_dropped) == BASELINE_FREQUENCY_DROPPED


@requires_real_data
def test_ledger_grouping_sums_to_the_ledger_totals() -> None:
    ledger = spot_ledger.read_ledger()
    assert ledger.available
    assert ledger.gross == pytest.approx(BASELINE_GROSS)
    assert ledger.net == pytest.approx(BASELINE_NET)
    assert ledger.spots == BASELINE_SPOTS
    assert len(ledger.by_advertiser) == OBSERVED_ADVERTISERS
    assert sum(m.gross for m in ledger.by_advertiser.values()) == pytest.approx(BASELINE_GROSS)
    assert sum(m.net for m in ledger.by_advertiser.values()) == pytest.approx(BASELINE_NET)
    assert sum(m.spots for m in ledger.by_advertiser.values()) == BASELINE_SPOTS
    assert sum(m.dropped_by_frequency for m in ledger.by_advertiser.values()) == BASELINE_FREQUENCY_DROPPED


@requires_real_data
def test_agencies_still_resolve_nine_of_nine() -> None:
    layer = AgencyLayer.from_files()
    daily_agencies = sorted({
        str(row.get("משרד / MB", "")).strip()
        for row in _rows(DAILY_CSV)
        if str(row.get("משרד / MB", "")).strip()
    })
    assert len(daily_agencies) == 9
    assert [name for name in daily_agencies if layer.by_name.get(name) is None] == []


@requires_real_data
def test_the_forty_five_rules_rows_are_untouched_and_bound_to_nothing() -> None:
    rows = _rows(RULES_CSV)
    assert len(rows) == RULES_ROWS
    assert [column in rows[0] for column in IDENTITY_COLUMNS] == [True, True, True]
    assert all(row["name"] == "" and row["aliases"] == "" for row in rows)
    assert all(row["display_name"] == "" for row in rows)
    premiums = {row["advertiser_id"]: float(row["default_premium"]) for row in rows}
    assert premiums["ADV_02"] == pytest.approx(1.27)
    assert all(value == pytest.approx(1.0) for key, value in premiums.items() if key != "ADV_02")


@requires_real_data
def test_no_observed_advertiser_is_bound_to_a_rules_row_today() -> None:
    engine = AdvertiserRuleEngine.from_files()
    for name in _daily_advertisers():
        assert engine.key_for(name) == name
        assert engine.effective_premium(name) == pytest.approx(1.0)
        assert engine.is_allowed(name)
    assert engine.names.collisions == []


@requires_real_data
def test_stored_ids_still_address_their_own_row() -> None:
    engine = AdvertiserRuleEngine.from_files()
    assert engine.key_for("ADV_02") == "ADV_02"
    assert engine.effective_premium("ADV_02") == pytest.approx(1.27)
    assert engine.effective_premium("ADV_01") == pytest.approx(1.0)


# --- the mechanism, on a throwaway store --------------------------------------


HEADER = [
    "advertiser_id", "default_premium", "allow_positions", "allow_genres",
    "prime_time_only", "urgency_k", "ahead_k", "notes",
    "name", "display_name", "aliases",
]


def _row(advertiser_id: str, premium: str = "1.0", name: str = "",
         display_name: str = "", aliases: str = "") -> dict[str, str]:
    return {
        "advertiser_id": advertiser_id, "default_premium": premium,
        "allow_positions": "ANY", "allow_genres": "ANY", "prime_time_only": "False",
        "urgency_k": "", "ahead_k": "", "notes": "",
        "name": name, "display_name": display_name, "aliases": aliases,
    }


def _write(path: Path, rows: list[dict[str, str]], header: list[str] = HEADER) -> None:
    with open(path, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n",
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture()
def store(tmp_path):
    """A throwaway rules store with an empty conditions file beside it."""
    conditions = tmp_path / "advertiser_conditions.csv"
    _write(conditions, [], ["advertiser_id", "rule_id", "scope_positions",
                            "scope_genres", "scope_dayparts", "effect", "value", "notes"])
    return tmp_path


def _engine(store: Path) -> AdvertiserRuleEngine:
    return AdvertiserRuleEngine.from_files(
        rules_path=store / "advertiser_rules.csv",
        conditions_path=store / "advertiser_conditions.csv",
    )


def test_a_store_re_keyed_on_the_observed_names_resolves(store) -> None:
    """Owner option A: the rows themselves are keyed on the real names."""
    _write(store / "advertiser_rules.csv", [_row("בנק הפועלים", "1.27"), _row("סלקום")])
    engine = _engine(store)
    assert engine.key_for("בנק הפועלים") == "בנק הפועלים"
    assert engine.effective_premium("בנק הפועלים") == pytest.approx(1.27)
    assert engine.effective_premium("סלקום") == pytest.approx(1.0)


def test_a_bound_name_prices_that_advertisers_spots(store) -> None:
    """Owner option B: the ADV rows keep their premiums and gain a name."""
    _write(store / "advertiser_rules.csv", [_row("ADV_02", "1.27", name="בנק הפועלים")])
    engine = _engine(store)
    assert engine.key_for("בנק הפועלים") == "ADV_02"
    assert engine.effective_premium("בנק הפועלים") == pytest.approx(1.27)
    assert engine.effective_premium("סלקום") == pytest.approx(1.0)


def test_clearing_the_name_puts_the_price_back(store) -> None:
    _write(store / "advertiser_rules.csv", [_row("ADV_02", "1.27", name="בנק הפועלים")])
    assert _engine(store).effective_premium("בנק הפועלים") == pytest.approx(1.27)
    _write(store / "advertiser_rules.csv", [_row("ADV_02", "1.27")])
    assert _engine(store).effective_premium("בנק הפועלים") == pytest.approx(1.0)


def test_aliases_and_display_name_resolve_like_the_agency_store(store) -> None:
    _write(store / "advertiser_rules.csv", [
        _row("ADV_02", "1.27", name="בנק הפועלים", display_name="הפועלים",
             aliases="Bank Hapoalim|פועלים"),
    ])
    engine = _engine(store)
    for token in ("בנק הפועלים", "הפועלים", "Bank Hapoalim", "פועלים"):
        assert engine.effective_premium(token) == pytest.approx(1.27)
    assert engine.effective_premium("בנק אחר") == pytest.approx(1.0)


def test_resolution_folds_whitespace_case_and_hebrew_punctuation(store) -> None:
    _write(store / "advertiser_rules.csv", [
        _row("ADV_02", "1.27", name='סיימן בע"מ', aliases="Bank Hapoalim"),
    ])
    engine = _engine(store)
    assert engine.effective_premium("  סיימן   בע״מ ") == pytest.approx(1.27)
    assert engine.effective_premium("bank hapoalim") == pytest.approx(1.27)


def test_a_name_two_rows_claim_is_recorded_not_guessed(store) -> None:
    _write(store / "advertiser_rules.csv", [
        _row("ADV_01", "1.10", name="שטראוס"),
        _row("ADV_02", "1.27", name="שטראוס"),
    ])
    engine = _engine(store)
    assert engine.names.collisions == [(normalize_name("שטראוס"), "ADV_01", "ADV_02")]
    assert engine.effective_premium("שטראוס") == pytest.approx(1.10)


def test_an_unknown_advertiser_is_still_allowed_at_premium_one(store) -> None:
    _write(store / "advertiser_rules.csv", [_row("ADV_01", "1.10", name="שטראוס")])
    engine = _engine(store)
    assert engine.effective_premium("מישהו אחר") == pytest.approx(1.0)
    assert engine.is_allowed("מישהו אחר")


@requires_real_data
def test_binding_one_name_moves_only_that_advertisers_money(store) -> None:
    """The proof that the ledger's gross is unchanged because nothing is bound."""
    _write(store / "advertiser_rules.csv", [_row("ADV_02", "2.0", name="בנק הפועלים")])
    bound = price_daily_file(DAILY_CSV, engine=_engine(store))
    plain = price_daily_file(DAILY_CSV, engine=AdvertiserRuleEngine.from_files())

    assert plain.total_revenue == pytest.approx(BASELINE_GROSS)
    assert bound.total_revenue > plain.total_revenue
    moved = {
        spot.advertiser
        for spot, before in zip(bound.priced, plain.priced)
        if spot.revenue != before.revenue
    }
    assert moved == {"בנק הפועלים"}


# --- the stores themselves ----------------------------------------------------


def test_the_index_never_claims_a_blank_token() -> None:
    index = build_name_index([
        {"advertiser_id": "ADV_01", "name": "", "display_name": "  ", "aliases": "||"},
        {"advertiser_id": "", "name": "אסם", "display_name": "", "aliases": ""},
    ])
    assert index.by_token == {normalize_name("ADV_01"): "ADV_01"}


def test_alias_splitting_drops_empty_parts() -> None:
    assert split_aliases("א|| ב |") == ("א", "ב")
    assert split_aliases("") == ()
    assert split_aliases(None) == ()


def test_a_missing_name_space_is_empty_rather_than_an_error(tmp_path) -> None:
    assert load_advertiser_names(tmp_path / "nothing.csv") == {}
    assert resolve_advertiser("אסם", names={}, rules_index=None) is None


def test_an_unavailable_ledger_reports_no_figure(monkeypatch) -> None:
    monkeypatch.setattr(spot_ledger, "_basis_name", lambda: None)
    import kairos_api.exporters as exporters

    monkeypatch.setattr(exporters, "_load_daily_pricing", lambda: None)
    ledger = spot_ledger.read_ledger()
    assert not ledger.available
    assert ledger.gross is None and ledger.net is None and ledger.spots is None
    assert ledger.reason
    assert ledger.by_advertiser == {}


# --- the migration ------------------------------------------------------------


@requires_real_data
def test_the_migration_is_idempotent_and_invents_nothing(tmp_path, monkeypatch) -> None:
    import scripts.migrate_advertiser_identity as migrate

    rules = tmp_path / "advertiser_rules.csv"
    names = tmp_path / "advertiser_names.csv"
    daily_dir = tmp_path / "daily_input"
    daily_dir.mkdir()
    # The pre-migration store: the shipped rows without the identity columns.
    legacy = [column for column in _rows(RULES_CSV)[0] if column not in IDENTITY_COLUMNS]
    _write(rules, _rows(RULES_CSV), legacy)
    shutil.copy2(DAILY_CSV, daily_dir / DAILY_CSV.name)
    monkeypatch.setattr(migrate, "RULES_PATH", rules)
    monkeypatch.setattr(migrate, "NAMES_PATH", names)
    monkeypatch.setattr(migrate, "DAILY_DIR", daily_dir)
    monkeypatch.setattr(migrate, "BACKUP_DIR", tmp_path / "_backups")

    first = migrate.add_identity_columns()
    migrate.build_names()
    once = names.read_bytes()
    rules_once = rules.read_bytes()

    assert first["changed"] is True
    assert migrate.add_identity_columns()["changed"] is False
    migrate.build_names()
    assert names.read_bytes() == once
    assert rules.read_bytes() == rules_once

    written = {row["name"] for row in _rows(names)}
    assert written == set(_daily_advertisers())
    assert all(row["source"] == "observed" and row["first_seen"] for row in _rows(names))

    # The migration adds columns and changes no cell that was already there.
    migrated = _rows(rules)
    for before, after in zip(_rows(RULES_CSV), migrated):
        assert all(before[column] == after[column] for column in legacy)
    assert all(after[column] == "" for after in migrated for column in IDENTITY_COLUMNS)
