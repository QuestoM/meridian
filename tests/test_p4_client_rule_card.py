"""P4: the client record's identity and price controls act, instead of leading away.

The measured defect. On פריסבי, rank 1 of 41 with a gross of 56,034 in the priced
daily ledger, the record showed two empty properties whose controls read "הוסיפו
כתיב בכרטיס הלקוח" and "קשרו כלל בכרטיס הלקוח". Both called back into the
workspace, which switched to the advertiser records tab and passed it nothing, so
the reader landed on 45 cards keyed ADV_01 to ADV_45, not one of which carries a
client name and not one of which could have been פריסבי's card. Measured on the
shipped store: 45 rows, 0 of them bound to any advertiser, against 41 observed
advertisers of whom 41 are named and 0 are bound.

What this file measures.

  * the state that made it a dead end, on the shipped store, so nothing below can
    pass because the store changed shape underneath it,
  * the write the record now performs, on a throwaway store: creating the
    client's pricing row binds it, prices it, and shows up on the client tree the
    record itself reads,
  * the second write, adding a spelling, and the store's refusal of a spelling
    another row holds,
  * the wiring: the records tab is now entered with a row id in hand, and the
    control that used to lead nowhere is gone from the record.

The rendered section itself, in all four of its states, is measured beside this
file in ``test_p4_client_rule_card_render.py``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

import kairos_api.advertisers as adv

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
CLIENTS = APP / "src" / "clients"
RECORD = CLIENTS / "ClientRecord.jsx"
WORKSPACE = CLIENTS / "ClientsWorkspace.jsx"
PANEL = CLIENTS / "AdvertiserRecordsPanel.jsx"

# The client the critic measured, and its figure in the shipped priced ledger.
CLIENT = "פריסבי"
CLIENT_GROSS = 56034.0

FULL_HEADER = [
    "advertiser_id", "default_premium", "allow_positions", "allow_genres",
    "prime_time_only", "urgency_k", "ahead_k", "notes",
    "name", "display_name", "aliases",
]

# The two sentences the two dead controls carried, which may not come back.
DEAD_SPELLING = "הוסיפו כתיב בכרטיס הלקוח"
DEAD_RULE = "קשרו כלל בכרטיס הלקוח"


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
    _write(rules, FULL_HEADER, [_row("ADV_01"), _row("ADV_02", "1.27")])
    monkeypatch.setattr(adv, "RULES_PATH", rules)
    monkeypatch.setattr(adv, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(adv, "OBSERVED_NAMES_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(conditions_module, "CONDITIONS_PATH", conditions)
    monkeypatch.setattr(conditions_module, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(engine_module, "DEFAULT_RULES_PATH", rules)
    monkeypatch.setattr(engine_module, "DEFAULT_CONDITIONS_PATH", conditions)
    return tmp_path


@pytest.fixture()
def named(store, monkeypatch):
    """A throwaway name space and a throwaway ledger carrying the one client."""
    import kairos_api.advertisers_identity as identity
    import kairos_api.spot_ledger as ledger_module

    names = store / "advertiser_names.csv"
    _write(names, ["name", "display_name", "aliases", "source", "first_seen", "notes"], [
        {"name": CLIENT, "display_name": "", "aliases": "", "source": "observed",
         "first_seen": "2025-04-27", "notes": ""},
    ])
    monkeypatch.setattr(identity, "_names_path", lambda: names)
    monkeypatch.setattr(ledger_module, "read_ledger", lambda: ledger_module.LedgerRead(
        available=True, basis="Wally_test.csv", gross=CLIENT_GROSS, net=CLIENT_GROSS,
        spots=6, dropped_by_rule=0, dropped_by_frequency=3,
        by_advertiser={CLIENT: ledger_module.AdvertiserMoney(
            advertiser=CLIENT, gross=CLIENT_GROSS, net=CLIENT_GROSS, spots=6,
        )},
    ))
    return store


# --- the state that made it a dead end -----------------------------------------


def test_the_premise_this_file_was_written_against_is_closed_on_the_shipped_store() -> None:
    """This asserted zero bound rows, and it failed because the product got better.

    What it measured when written, on 2026-08-08: 45 rules rows, **0 of them
    bound**, against 41 observed advertisers of whom 41 were named. Every card
    was keyed ADV_01 to ADV_45 and none of them could have been this client's.

    Measured again 2026-08-09, after the naming and binding work: same 45 rules
    rows, same 41 advertisers, and **41 of 41 bound**. The dead end is gone on the
    shipped store, so the assertion that it is still there is the wrong assertion
    to keep.

    It is kept as its inverse rather than deleted, because the premise closing is
    exactly the thing that could silently reopen: a rules row rewritten without
    its name, or a store restored from a pre-naming backup, drops the count and
    this fails again with the reason attached.

    The write path itself is measured below on a throwaway store, which is why
    those tests never depended on this one.
    """
    from kairos_api.advertisers_identity import identity_report

    report = identity_report()
    assert report["rules_rows"] == 45
    assert report["count"] == 41 and report["resolved"] == 41
    assert report["bound_to_a_rules_row"] == 41, (
        "advertisers came unbound from their rules rows; this file's whole premise "
        "was that state, and the client record leads nowhere again while it holds"
    )
    record = next(r for r in report["advertisers"] if r["advertiser"] == CLIENT)
    assert record["rules"]["bound"] is True
    # Unmoved by the binding, and that is the honest result: every bound row
    # carries premium 1.0, so the money is identical and only the layer is live.
    assert record["money"]["gross"] == pytest.approx(CLIENT_GROSS)


# --- the write the record now performs -----------------------------------------


def test_creating_the_client_rule_binds_it_and_prices_the_client(named) -> None:
    """What the create control does, end to end, on a throwaway store."""
    from kairos_api.advertisers_identity import identity_report

    created = adv.create_advertiser(adv.AdvertiserCreate(
        advertiser_id="ADV_03", name=CLIENT, default_premium=1.15,
    ))
    assert created["name"] == CLIENT
    assert created["name_source"] == "operator"

    report = identity_report()
    record = next(r for r in report["advertisers"] if r["advertiser"] == CLIENT)
    assert record["rules"]["bound"] is True
    assert record["rules"]["advertiser_id"] == "ADV_03"
    assert record["rules"]["effective_premium"] == pytest.approx(1.15)
    assert record["money"]["gross"] == pytest.approx(CLIENT_GROSS)
    assert report["bound_to_a_rules_row"] == 1


def test_the_client_tree_the_record_reads_shows_the_new_binding(named, monkeypatch) -> None:
    """The record's own read, so the figure on screen is the one that moved."""
    from kairos_api.campaigns_read_clients import _client_record, _identity_index

    before = _client_record(CLIENT, None, _identity_index(), {}, "")
    assert before["bound_to_rules_row"] is False
    assert before["effective_premium"] == pytest.approx(1.0)

    adv.create_advertiser(adv.AdvertiserCreate(
        advertiser_id="ADV_03", name=CLIENT, default_premium=1.15,
    ))
    after = _client_record(CLIENT, None, _identity_index(), {}, "")
    assert after["bound_to_rules_row"] is True
    assert after["effective_premium"] == pytest.approx(1.15)


def test_a_spelling_added_on_the_rule_is_priced_as_this_client(named) -> None:
    """The second write, and the reason the section states for it."""
    from kairos.optimize.advertiser_rules import AdvertiserRuleEngine

    adv.create_advertiser(adv.AdvertiserCreate(
        advertiser_id="ADV_03", name=CLIENT, default_premium=1.15,
    ))
    adv.update_advertiser("ADV_03", adv.AdvertiserUpdate(aliases='פריסבי בע"מ'))

    row = next(r for r in adv.list_advertisers()["advertisers"] if r["advertiser_id"] == "ADV_03")
    assert row["aliases"] == 'פריסבי בע"מ'
    engine = AdvertiserRuleEngine.from_files()
    assert engine.effective_premium('פריסבי בע"מ') == pytest.approx(1.15)
    assert engine.effective_premium("מישהו אחר") == pytest.approx(1.0)


def test_the_new_rule_really_moves_this_client_money_on_the_real_daily_file(store) -> None:
    """The claim the create form makes, measured on the real priced daily file.

    The section tells the operator that from the moment the rule exists it prices
    this client's spots on the daily pricing path. That is the one sentence on
    this surface that moves money, so it is priced twice against the real file:
    once with the throwaway store empty, once with the rule bound, and the
    difference is exactly the premium.
    """
    from kairos_api import spot_ledger

    before = spot_ledger.read_ledger()
    if not before.available or CLIENT not in before.by_advertiser:
        pytest.skip("no daily file carrying this client on disk, so there is nothing to price")
    base = before.by_advertiser[CLIENT].gross

    adv.create_advertiser(adv.AdvertiserCreate(
        advertiser_id="ADV_03", name=CLIENT, default_premium=1.15,
    ))
    after = spot_ledger.read_ledger()
    assert after.by_advertiser[CLIENT].gross == pytest.approx(base * 1.15, rel=1e-6)
    # And nobody else moves, because a rule prices the advertiser it names.
    other = next(key for key in before.by_advertiser if key != CLIENT)
    assert after.by_advertiser[other].gross == pytest.approx(before.by_advertiser[other].gross)


def test_the_store_refuses_a_spelling_another_row_already_holds(named) -> None:
    """The refusal the section shows before it writes, proven to be real."""
    from fastapi import HTTPException

    adv.create_advertiser(adv.AdvertiserCreate(advertiser_id="ADV_03", name=CLIENT))
    with pytest.raises(HTTPException) as raised:
        adv.update_advertiser("ADV_02", adv.AdvertiserUpdate(name=CLIENT))
    assert raised.value.status_code == 409


# --- the wiring ----------------------------------------------------------------


def test_the_record_no_longer_carries_a_control_that_leads_to_the_grid() -> None:
    """The dead end itself: both sentences and the callback are gone."""
    record = RECORD.read_text(encoding="utf-8")
    assert DEAD_SPELLING not in record
    assert DEAD_RULE not in record
    assert "onOpenRecords" not in record
    assert "ClientRuleCard" in record


def test_the_records_tab_is_only_ever_entered_with_a_row_id_in_hand() -> None:
    """The tab now receives the row, so it can open one card rather than all."""
    workspace = WORKSPACE.read_text(encoding="utf-8")
    assert "openAdvertiserId={openRuleId}" in workspace
    assert "function openRuleCard(advertiserId)" in workspace
    panel = PANEL.read_text(encoding="utf-8")
    assert "openAdvertiserId" in panel
    assert "setOpenId(openAdvertiserId)" in panel
