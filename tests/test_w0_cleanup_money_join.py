"""The money join folds exactly as identity resolution folds.

W0-3 built the advertiser name space so that the string a daily file carries
resolves to a stored record after a documented fold: surrounding and repeated
whitespace, letter case, and the Hebrew geresh and gershayim against their ASCII
equivalents. The money half of the same record then joined on the raw string,
``ledger.for_advertiser(record.name)``, an exact dict lookup. So an advertiser
could resolve perfectly and still report zero money with "This advertiser has no
priced spot in the daily file being read", which is the exact failure the
identity work existed to end.

Measured on the shipped data before the fix: all 41 observed advertisers hit the
ledger by exact string, so the join reported the right money and the defect was
latent, not live. It becomes live the moment one daily file spells one
advertiser with a gershayim and the name store spells it with a quote, which is
the difference the fold exists to absorb.

These tests use a hand-built ledger rather than the daily file, because the
defect is about a spelling the shipped file does not happen to contain.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import kairos_api.advertisers_identity as identity
import kairos_api.spot_ledger as ledger_module

NAMES_HEADER = ["name", "display_name", "aliases", "source", "first_seen", "notes"]
RULES_HEADER = [
    "advertiser_id", "name", "display_name", "aliases", "default_premium",
    "notes", "active",
]

# One name in three spellings. The store holds the gershayim form, the daily
# file carries the ASCII-quote form with a doubled space, and they are the same
# advertiser by the fold the resolver documents.
STORED = 'שטראוס עלית בע״מ'
IN_THE_DAILY_FILE = 'שטראוס  עלית בע"מ'
ANOTHER_SPELLING = 'שטראוס עלית בע"מ'


def _write(path: Path, header: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _redirect_stores(tmp_path, monkeypatch) -> tuple[Path, Path]:
    """Point the name space, the rules store and its conditions at throwaway csvs.

    Every setattr here raises if the name goes away, deliberately: a redirect
    that silently missed would leave these tests reading the shipped stores and
    asserting nothing about the fixture they think they built.
    """
    from kairos.optimize import advertiser_rules

    names = tmp_path / "advertiser_names.csv"
    rules = tmp_path / "advertiser_rules.csv"
    conditions = tmp_path / "advertiser_conditions.csv"
    _write(conditions, [
        "advertiser_id", "rule_id", "scope_positions", "scope_genres",
        "scope_dayparts", "effect", "value", "notes",
    ], [])
    monkeypatch.setattr(identity, "_names_path", lambda: names)
    monkeypatch.setattr(advertiser_rules, "DEFAULT_RULES_PATH", rules)
    monkeypatch.setattr(advertiser_rules, "DEFAULT_CONDITIONS_PATH", conditions)
    return names, rules


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """One advertiser in the name space, no rules row, and no daily file."""
    names, rules = _redirect_stores(tmp_path, monkeypatch)
    _write(names, NAMES_HEADER, [{
        "name": STORED, "display_name": "", "aliases": "", "source": "observed",
        "first_seen": "2024-11-01", "notes": "",
    }])
    _write(rules, RULES_HEADER, [])
    return tmp_path


def _ledger(**by_advertiser: float) -> ledger_module.LedgerRead:
    """A ledger keyed on whatever spellings the caller names."""
    grouped = {
        name: ledger_module.AdvertiserMoney(
            advertiser=name, gross=gross, net=gross, spots=1,
        )
        for name, gross in by_advertiser.items()
    }
    return ledger_module.LedgerRead(
        available=True, basis="Wally_test.csv",
        gross=sum(by_advertiser.values()), net=sum(by_advertiser.values()),
        spots=len(grouped), dropped_by_rule=0, dropped_by_frequency=0,
        by_advertiser=grouped,
    )


def _report(monkeypatch, ledger: ledger_module.LedgerRead) -> dict:
    monkeypatch.setattr(ledger_module, "read_ledger", lambda: ledger)
    return identity.identity_report()


def test_a_name_that_differs_only_by_the_folded_characters_still_carries_its_money(
    store, monkeypatch,
) -> None:
    """The whole gap, in one assertion.

    The stored name and the ledger key differ by a gershayim against an ASCII
    quote and by one repeated space. Resolution folds both away. Before the fix
    the money join did not, so this record reported 0.0 and the reason
    "no priced spot", while the ledger held 12,500 for it.
    """
    report = _report(monkeypatch, _ledger(**{IN_THE_DAILY_FILE: 12500.0}))

    assert report["unresolved"] == []
    record = report["advertisers"][0]
    assert record["advertiser"] == STORED
    assert record["resolved"] is True
    assert record["money"]["gross"] == pytest.approx(12500.0)
    assert record["money"]["spots"] == 1
    assert record["money"]["reason"] == ""
    assert record["money"]["ledger_keys"] == [IN_THE_DAILY_FILE]
    # And the report's own coverage figure counts it, which it did not before.
    assert report["in_ledger"] == 1


def test_two_spellings_of_one_advertiser_add_up_and_both_are_named(
    store, monkeypatch,
) -> None:
    """A merge is money moving, so it may never be silent.

    ``ledger_keys`` names every spelling that fed the figure, so a reader can
    see that two rows of the daily file were treated as one advertiser and can
    check that judgement rather than take it.
    """
    report = _report(monkeypatch, _ledger(**{
        IN_THE_DAILY_FILE: 12500.0, ANOTHER_SPELLING: 500.0,
    }))

    record = report["advertisers"][0]
    assert record["money"]["gross"] == pytest.approx(13000.0)
    assert record["money"]["spots"] == 2
    assert sorted(record["money"]["ledger_keys"]) == sorted(
        [IN_THE_DAILY_FILE, ANOTHER_SPELLING]
    )
    assert report["unresolved"] == []


def test_an_advertiser_with_no_spot_still_reports_the_honest_zero(
    store, monkeypatch,
) -> None:
    """The fold must not turn "no money" into "some money"."""
    report = _report(monkeypatch, _ledger(**{"מפרסם אחר": 900.0}))

    by_name = {record["advertiser"]: record for record in report["advertisers"]}
    ours = by_name[STORED]
    assert ours["money"]["gross"] == 0.0
    assert ours["money"]["spots"] == 0
    assert ours["money"]["ledger_keys"] == []
    assert ours["money"]["reason"] == identity.NO_LEDGER_REASON
    # The other name is nobody's, so it is listed as itself with its own money
    # rather than folded into the record it does not belong to.
    assert report["unresolved"] == ["מפרסם אחר"]
    assert by_name["מפרסם אחר"]["money"]["gross"] == pytest.approx(900.0)


def test_resolve_one_joins_the_same_way_the_report_does(store, monkeypatch) -> None:
    """The single-name read is the same join, or a surface disagrees with a list."""
    monkeypatch.setattr(
        ledger_module, "read_ledger", lambda: _ledger(**{IN_THE_DAILY_FILE: 12500.0}),
    )
    answer = identity.resolve_one(ANOTHER_SPELLING)

    assert answer["resolved"] is True
    assert answer["advertiser"] == STORED
    assert answer["money"]["gross"] == pytest.approx(12500.0)
    assert answer["money"]["ledger_keys"] == [IN_THE_DAILY_FILE]


def test_the_attribution_groups_every_ledger_key_exactly_once(store, monkeypatch) -> None:
    """The mechanism: no key is dropped and no key is counted twice.

    A ledger key that resolves to nothing groups under itself, which is the key
    the unresolved record reports it under, so every shekel in the ledger is
    reachable from exactly one record.
    """
    from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
    from kairos.optimize.advertiser_rules_identity import (
        _names_token_index,
        load_advertiser_names,
    )

    ledger = _ledger(**{IN_THE_DAILY_FILE: 12500.0, ANOTHER_SPELLING: 500.0, "מפרסם אחר": 900.0})
    names = load_advertiser_names(identity._names_path())
    engine = AdvertiserRuleEngine.from_files()
    attribution = identity.ledger_attribution(
        ledger, names=names, engine=engine, tokens=_names_token_index(names),
    )

    grouped = [key for keys in attribution.values() for key in keys]
    assert sorted(grouped) == sorted(ledger.by_advertiser)
    assert len(grouped) == len(set(grouped))
    assert sorted(attribution[STORED]) == sorted([IN_THE_DAILY_FILE, ANOTHER_SPELLING])
    assert attribution["מפרסם אחר"] == ["מפרסם אחר"]


def test_an_advertiser_only_a_rules_row_names_is_listed_with_its_money(
    tmp_path, monkeypatch,
) -> None:
    """Nobody named this one, and it is the same failure one level up.

    The report built its list from the name space and then added the ledger keys
    that resolved to nothing. An advertiser the operator bound on a rules row
    before the observed name space caught up resolves, so it was in neither list.
    Measured before the fix, on an empty name space with one bound rules row: 0
    advertisers listed, ``unresolved`` empty, and 5,000 of ledger gross reachable
    from no record at all.
    """
    names, rules = _redirect_stores(tmp_path, monkeypatch)
    _write(names, NAMES_HEADER, [])
    _write(rules, RULES_HEADER, [{
        "advertiser_id": "ADV_01", "name": "אסם", "display_name": "", "aliases": "",
        "default_premium": "1.27", "notes": "", "active": "true",
    }])

    report = _report(monkeypatch, _ledger(**{"אסם": 5000.0}))

    assert report["count"] == 1
    assert report["unresolved"] == []
    record = report["advertisers"][0]
    assert record["advertiser"] == "אסם"
    assert record["resolved"] is True
    assert record["source"] == "rules"
    assert record["rules"]["bound"] is True
    assert record["rules"]["advertiser_id"] == "ADV_01"
    assert record["money"]["gross"] == pytest.approx(5000.0)
    # The accounting statement: the ledger total is reachable from the records.
    assert sum(r["money"]["gross"] for r in report["advertisers"]) == pytest.approx(
        report["ledger"]["gross"]
    )


def test_the_shipped_data_still_reports_the_same_money(monkeypatch) -> None:
    """Bar 3: the fix is latent on the real file, so no figure moves.

    Measured on the shipped daily file: 41 observed advertisers, every one of
    them hit by exact string before the fix and by the fold after it, totalling
    the same gross the agency summary and the spots export report.
    """
    ledger = ledger_module.read_ledger()
    if not ledger.available:  # pragma: no cover - environment without a daily file
        pytest.skip(f"no daily ledger to read: {ledger.reason}")

    report = identity.identity_report()
    total = sum(record["money"]["gross"] for record in report["advertisers"])
    assert total == pytest.approx(ledger.gross, abs=0.01)
    assert sum(record["money"]["spots"] for record in report["advertisers"]) == ledger.spots
    # Every record's money came from its own spelling and nothing was merged, so
    # the fix is provably a no-op on this data.
    assert all(
        record["money"]["ledger_keys"] in ([], [record["advertiser"]])
        for record in report["advertisers"]
    )
