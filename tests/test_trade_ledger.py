"""The credit ledger's arithmetic: FIFO lots, overdraft refusal, unit walls.

The property under test is the domain's three-level accrual-and-utilisation
shape (docs/trade/domain.md section 8): credit enters as lots with expiry
dates, spending consumes the oldest live lots first and stamps which, nothing
takes a balance below zero, and shekels, seconds and rating points never meet
in one sum. The FIFO math is checked twice — once against the hand-computed
scenario the spec names, once against an independent simulator on a seeded
random walk — because an allocator that agrees only with itself proves nothing.
"""

from datetime import date, timedelta
from random import Random

import pytest

from kairos_api import trade_ledger as ledger


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "LEDGER_PATH", tmp_path / "trade_credit_ledger.csv")
    monkeypatch.setattr(ledger, "BACKUP_DIR", tmp_path / "_backups")
    return tmp_path


def _accrue(quantity, effective_on, expires_on="", party="agency-x", level=ledger.AGENCY,
            unit=ledger.ILS_MEDIA_VALUE, reason=ledger.POLICY_ACCRUAL):
    return ledger.append_entry(
        level=level, party_ref=party, direction=ledger.ACCRUE, quantity=quantity,
        unit=unit, reason_code=reason, actor="dana", effective_on=effective_on,
        expires_on=expires_on,
    )


def _utilise(quantity, effective_on, party="agency-x", level=ledger.AGENCY,
             unit=ledger.ILS_MEDIA_VALUE, reason=ledger.SHORTFALL_CURE, **extra):
    return ledger.append_entry(
        level=level, party_ref=party, direction=ledger.UTILISE, quantity=quantity,
        unit=unit, reason_code=reason, actor="dana", effective_on=effective_on, **extra,
    )


def _block(party="agency-x", level=ledger.AGENCY, unit=ledger.ILS_MEDIA_VALUE, as_of=None):
    found = ledger.balances(level=level, party_ref=party, unit=unit, as_of=as_of)
    assert len(found) <= 1
    return found[0] if found else None


def test_fifo_scenario_exact_at_each_step(store):
    """The spec's own numbers: 100 (dies March) + 50 (dies June), utilise 120,
    expire in July — first lot exhausted, 20 taken from the second, 30 expire."""
    first = _accrue(100, "2026-01-10", expires_on="2026-03-31")
    second = _accrue(50, "2026-02-10", expires_on="2026-06-30")
    assert [first["entry_id"], second["entry_id"]] == ["TC_0001", "TC_0002"]

    opening = _block(as_of="2026-02-15")
    assert (opening["accrued"], opening["available"]) == (150.0, 150.0)
    assert (opening["utilised"], opening["expired"], opening["lapsed"]) == (0.0, 0.0, 0.0)

    spend = _utilise(120, "2026-03-01", source_makegood_id="MG_0007")
    assert spend["consumes"] == [
        {"entry_id": "TC_0001", "quantity": 100.0},   # oldest lot exhausted first
        {"entry_id": "TC_0002", "quantity": 20.0},
    ]
    assert spend["source_makegood_id"] == "MG_0007"

    after_spend = _block(as_of="2026-03-02")
    assert (after_spend["accrued"], after_spend["utilised"]) == (150.0, 120.0)
    assert (after_spend["available"], after_spend["expired"]) == (30.0, 0.0)

    # July, before the sweep: the 30 left in the June lot is past expiry —
    # not spendable, not yet expired, honestly named lapsed.
    before_sweep = _block(as_of="2026-07-01")
    assert (before_sweep["available"], before_sweep["lapsed"]) == (0.0, 30.0)
    assert before_sweep["expired"] == 0.0

    written = ledger.expire_due("2026-07-15", "dana")
    assert len(written) == 1
    assert written[0]["quantity"] == 30.0
    assert written[0]["reason_code"] == ledger.EXPIRY
    assert written[0]["consumes"] == [{"entry_id": "TC_0002", "quantity": 30.0}]

    closed = _block(as_of="2026-07-15")
    assert (closed["accrued"], closed["utilised"], closed["expired"]) == (150.0, 120.0, 30.0)
    assert (closed["available"], closed["lapsed"]) == (0.0, 0.0)
    assert closed["accrued"] + closed["adjusted"] == (
        closed["utilised"] + closed["expired"] + closed["lapsed"] + closed["available"])

    # The sweep is idempotent: nothing is left to expire.
    assert ledger.expire_due("2026-08-01", "dana") == []


def test_overdraft_refused_by_name(store):
    _accrue(90, "2026-01-10", unit=ledger.SECONDS, party="acme", level=ledger.ADVERTISER)
    with pytest.raises(ValueError) as refusal:
        _utilise(120, "2026-02-01", unit=ledger.SECONDS, party="acme", level=ledger.ADVERTISER)
    message = str(refusal.value)
    assert "overdraft" in message
    assert "90" in message and "120" in message
    assert "acme" in message and ledger.SECONDS in message
    # The refusal wrote nothing: the balance still stands at 90.
    assert _block(party="acme", level=ledger.ADVERTISER, unit=ledger.SECONDS,
                  as_of="2026-02-01")["available"] == 90.0


def test_overdraft_refusal_names_lapsed_credit(store):
    """Credit past its date is not silently unavailable: the refusal says where it went."""
    _accrue(100, "2026-01-10", expires_on="2026-02-28")
    with pytest.raises(ValueError) as refusal:
        _utilise(50, "2026-03-15")
    assert "lapsed past expiry" in str(refusal.value)
    assert "expire_due" in str(refusal.value)


def test_units_never_mix(store):
    _accrue(100, "2026-01-10", unit=ledger.SECONDS)
    _accrue(1000, "2026-01-10", unit=ledger.ILS_MEDIA_VALUE)
    # 150 seconds is an overdraft even though 1000 shekels sit beside it.
    with pytest.raises(ValueError, match="overdraft"):
        _utilise(150, "2026-02-01", unit=ledger.SECONDS)
    _utilise(80, "2026-02-01", unit=ledger.SECONDS)
    blocks = ledger.balances(level=ledger.AGENCY, party_ref="agency-x", as_of="2026-02-02")
    by_unit = {block["unit"]: block for block in blocks}
    assert set(by_unit) == {ledger.SECONDS, ledger.ILS_MEDIA_VALUE}
    assert by_unit[ledger.SECONDS]["available"] == 20.0
    assert by_unit[ledger.ILS_MEDIA_VALUE]["available"] == 1000.0


def test_three_levels_hold_independent_balances(store):
    """The domain's correction of the foreign model: campaign, advertiser and
    agency balances are three pots, never one pot read three ways."""
    _accrue(500, "2026-01-10", party="C_0001", level=ledger.CAMPAIGN)
    _accrue(300, "2026-01-10", party="acme", level=ledger.ADVERTISER)
    _accrue(800, "2026-01-10", party="agency-x", level=ledger.AGENCY)
    # An advertiser cannot draw on the agency's pot, nor a campaign on either.
    with pytest.raises(ValueError, match="overdraft"):
        _utilise(400, "2026-02-01", party="acme", level=ledger.ADVERTISER)
    _utilise(600, "2026-02-01", party="agency-x", level=ledger.AGENCY)
    assert _block("C_0001", ledger.CAMPAIGN, as_of="2026-02-02")["available"] == 500.0
    assert _block("acme", ledger.ADVERTISER, as_of="2026-02-02")["available"] == 300.0
    assert _block("agency-x", ledger.AGENCY, as_of="2026-02-02")["available"] == 200.0
    # The agency layer is the point of the shape: framework credit spent on a
    # DIFFERENT campaign is one agency-level utilisation, linked by source ids.
    spend = _utilise(150, "2026-02-03", party="agency-x", level=ledger.AGENCY,
                     source_agreement_id="agr-frame1", note="spent on campaign C_0002")
    assert spend["source_agreement_id"] == "agr-frame1"


def test_statement_running_balance(store):
    _accrue(100, "2026-01-10", expires_on="2026-03-31")
    _accrue(50, "2026-02-10", expires_on="2026-06-30")
    _utilise(120, "2026-03-01")
    _accrue(40, "2026-03-20", unit=ledger.SECONDS)
    rows = ledger.statement("agency-x", ledger.AGENCY)
    assert [row["entry_id"] for row in rows] == ["TC_0001", "TC_0002", "TC_0003", "TC_0004"]
    assert [row["running_balance"] for row in rows] == [100.0, 150.0, 30.0, 40.0]
    # The fourth row runs in its own unit: seconds open at 40, shekels stay 30.
    assert rows[3]["unit"] == ledger.SECONDS


def test_expiring_soon_window(store):
    as_of = date(2026, 5, 1)
    _accrue(100, "2026-04-01", expires_on=(as_of + timedelta(days=40)).isoformat())
    _accrue(200, "2026-04-01", expires_on=(as_of + timedelta(days=90)).isoformat())
    _accrue(300, "2026-04-01")  # no expiry, never "soon"
    soon = _block(as_of=as_of)["expiring_soon"]
    assert [entry["entry_id"] for entry in soon] == ["TC_0001"]
    assert soon[0]["remaining"] == 100.0
    assert soon[0]["unit"] == ledger.ILS_MEDIA_VALUE


def test_append_validation_refuses_by_name(store):
    good = dict(level=ledger.AGENCY, party_ref="agency-x", direction=ledger.ACCRUE,
                quantity=10, unit=ledger.ILS_MEDIA_VALUE,
                reason_code=ledger.POLICY_ACCRUAL, actor="dana", effective_on="2026-01-10")

    def refused(match, **overrides):
        with pytest.raises(ValueError, match=match):
            ledger.append_entry(**{**good, **overrides})

    refused("level must be one of", level="network")
    refused("direction must be one of", direction="spend")
    refused("never assumed", unit="euros")
    refused("reason_code must be one of", reason_code="because")
    # A reason that exists but belongs elsewhere: expiry is the sweep's, not a hand's.
    refused("does not belong to direction", reason_code=ledger.EXPIRY)
    refused("must be positive", quantity=0)
    refused("must be positive", quantity=-5)
    refused("must be a number", quantity="lots")
    refused("needs a party_ref", party_ref="  ")
    refused("actor is required", actor="")
    refused("must be an ISO date", effective_on="next Tuesday")
    refused("is before effective_on", expires_on="2026-01-01")
    # An expiry date on an entry that opens no lot means nothing, so it is refused.
    _accrue(50, "2026-01-05")
    refused("belongs to credit entering the ledger",
            direction=ledger.UTILISE, reason_code=ledger.SHORTFALL_CURE,
            expires_on="2026-06-30")
    # A hand that moves credit must say why.
    refused("requires a note", direction=ledger.ADJUST, reason_code=ledger.MANUAL_ADJUST)
    adjusted = ledger.append_entry(**{**good, "direction": ledger.ADJUST,
                                      "reason_code": ledger.MANUAL_ADJUST,
                                      "note": "counted twice in the seed"})
    assert adjusted["direction"] == ledger.ADJUST


def test_adjust_adds_and_correction_downwards_cannot_overdraw(store):
    """An adjustment only adds; the downward correction is a utilisation, so the
    overdraft refusal covers every movement that reduces a balance."""
    _accrue(100, "2026-01-10")
    ledger.append_entry(level=ledger.AGENCY, party_ref="agency-x", direction=ledger.ADJUST,
                        quantity=20, unit=ledger.ILS_MEDIA_VALUE,
                        reason_code=ledger.MANUAL_ADJUST, actor="dana",
                        effective_on="2026-01-15", note="seed undercounted")
    block = _block(as_of="2026-01-16")
    assert (block["accrued"], block["adjusted"], block["available"]) == (100.0, 20.0, 120.0)
    with pytest.raises(ValueError, match="overdraft"):
        _utilise(130, "2026-01-20", reason=ledger.MANUAL_ADJUST, note="undo the whole pot and more")
    down = _utilise(120, "2026-01-20", reason=ledger.MANUAL_ADJUST, note="grant cancelled")
    assert [take["entry_id"] for take in down["consumes"]] == ["TC_0001", "TC_0002"]
    assert _block(as_of="2026-01-21")["available"] == 0.0


def test_fifo_against_independent_simulator(store):
    """A seeded random walk, checked against a naive lot simulator at every step:
    same allocation, same balance, and never a negative pot."""
    rng = Random(20260817)
    day = date(2026, 1, 5)
    lots: list[dict] = []  # the simulator: {"id", "remaining", "dies"}

    def sim_live(on):
        return [lot for lot in lots if lot["remaining"] > 0
                and (lot["dies"] is None or lot["dies"] >= on)]

    for step in range(120):
        day += timedelta(days=rng.randrange(0, 5))
        roll = rng.random()
        if roll < 0.45:
            quantity = rng.randrange(1, 300)
            dies = None if rng.random() < 0.25 else day + timedelta(days=rng.randrange(10, 120))
            entry = _accrue(quantity, day, expires_on=dies.isoformat() if dies else "")
            lots.append({"id": entry["entry_id"], "remaining": float(quantity), "dies": dies})
        elif roll < 0.85:
            live = sim_live(day)
            pool = round(sum(lot["remaining"] for lot in live), 4)
            if pool < 1 or rng.random() < 0.2:
                # Ask for more than the pot holds and demand the refusal.
                with pytest.raises(ValueError, match="overdraft"):
                    _utilise(pool + rng.randrange(1, 50), day)
                continue
            quantity = rng.randrange(1, int(pool) + 1)
            entry = _utilise(quantity, day)
            expected, need = [], float(quantity)
            for lot in live:
                if need <= 0:
                    break
                take = round(min(lot["remaining"], need), 4)
                lot["remaining"] = round(lot["remaining"] - take, 4)
                need = round(need - take, 4)
                expected.append({"entry_id": lot["id"], "quantity": take})
            assert entry["consumes"] == expected, f"allocation diverged at step {step}"
        else:
            for record in ledger.expire_due(day, "dana"):
                total = 0.0
                for lot in lots:
                    if lot["dies"] is not None and lot["dies"] < day and lot["remaining"] > 0:
                        total = round(total + lot["remaining"], 4)
                        lot["remaining"] = 0.0
                assert record["quantity"] == total
        block = _block(as_of=day)
        sim_available = round(sum(lot["remaining"] for lot in sim_live(day)), 4)
        if block is None:
            assert sim_available == 0.0
        else:
            assert block["available"] == sim_available, f"balance diverged at step {step}"
            assert block["available"] >= 0.0
            assert round(block["accrued"] + block["adjusted"], 4) == round(
                block["utilised"] + block["expired"] + block["lapsed"] + block["available"], 4)


def test_nothing_is_deleted_and_file_round_trips(store):
    _accrue(100, "2026-01-10", expires_on="2026-02-28")
    _utilise(40, "2026-02-01")
    ledger.expire_due("2026-03-15", "dana")
    frame = ledger.load_frame()
    assert list(frame["entry_id"]) == ["TC_0001", "TC_0002", "TC_0003"]
    # The consumes stamps survive the CSV round trip and answer "where did it go".
    reread = [ledger.record(row) for _, row in frame.iterrows()]
    assert reread[1]["consumes"] == [{"entry_id": "TC_0001", "quantity": 40.0}]
    assert reread[2]["consumes"] == [{"entry_id": "TC_0001", "quantity": 60.0}]


def test_vocabularies_publish_both_languages(store):
    words = ledger.vocabularies()
    assert {entry["value"] for entry in words["levels"]} == set(ledger.LEVELS)
    assert {entry["value"] for entry in words["directions"]} == set(ledger.DIRECTIONS)
    assert {entry["value"] for entry in words["units"]} == set(ledger.UNITS)
    assert {entry["value"] for entry in words["reasons"]} == set(ledger.REASONS)
    for family in ("levels", "directions", "units", "reasons"):
        for entry in words[family]:
            assert entry["label_he"] and entry["meaning_he"], entry["value"]
            assert entry["label_en"] and entry["meaning_en"], entry["value"]
