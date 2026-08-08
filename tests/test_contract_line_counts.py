"""A contract's published line counts must be true, and nothing re-derived them.

Each contract under docs/ux-gauntlet/contracts/ publishes the files its piece
owns with a line count beside each one. A builder reads that table to learn what
exists and what is near the 450 cap before it opens anything, so a stale count
is not a cosmetic problem: it sends somebody to add to a file that has no room.

Measured on 2026-08-09, before this test existed:

    P12.md   39 rows   27 stale
    P7.md    20 rows   13 stale
    P8.md     6 rows    1 stale
    W0-2.md  14 rows   14 stale

P12's round-10 critic had already found and closed this once. It came back,
because closing it meant editing numbers and nothing re-counted them afterwards.
That is the same shape as the native-control budget that went unenforced for a
whole wave: a number nobody re-derives is a number nobody checks.

The fix is one command, and it is safe to run in bulk because a count is a
measurement rather than a decision:

    python3 scripts/gauntlet/contract_line_counts.py --fix

A row naming a file that does not exist is deliberately NOT auto-fixable. Either
the path moved and the contract has to say where, or the row is dead, and both
are decisions a person makes.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "gauntlet" / "contract_line_counts.py"

spec = importlib.util.spec_from_file_location("contract_line_counts", MODULE)
counts = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(counts)


def test_no_contract_publishes_a_line_count_that_is_no_longer_true():
    drifted, _ = counts.audit(fix=False)
    assert drifted == [], (
        f"{len(drifted)} contract row(s) publish a stale line count:\n  "
        + "\n  ".join(drifted)
        + "\nRun: python3 scripts/gauntlet/contract_line_counts.py --fix"
    )


def test_every_file_a_contract_names_still_exists():
    _, missing = counts.audit(fix=False)
    assert missing == [], (
        f"{len(missing)} contract row(s) name a file that is not on disk:\n  "
        + "\n  ".join(missing)
        + "\nThis one is not auto-fixable. Either the path moved and the contract must say "
        "where, or the row is dead and should be removed by a person."
    )


def test_the_check_bites_when_a_count_drifts(tmp_path, monkeypatch):
    """A guard that has never failed has never been shown to work."""
    folder = tmp_path / "contracts"
    folder.mkdir()
    (folder / "FAKE.md").write_text(
        "| file | lines | note |\n|---|---|---|\n"
        "| `scripts/gauntlet/contract_line_counts.py` | 1 | deliberately wrong |\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(counts, "CONTRACTS", folder)
    drifted, missing = counts.audit(fix=False)
    assert missing == []
    assert len(drifted) == 1 and "says 1" in drifted[0]


def test_a_row_naming_a_deleted_file_is_reported_and_never_auto_fixed(tmp_path, monkeypatch):
    folder = tmp_path / "contracts"
    folder.mkdir()
    contract = folder / "FAKE.md"
    body = "| file | lines | note |\n|---|---|---|\n| `kairos/this_was_deleted.py` | 12 | gone |\n"
    contract.write_text(body, encoding="utf-8")
    monkeypatch.setattr(counts, "CONTRACTS", folder)
    drifted, missing = counts.audit(fix=True)
    assert drifted == []
    assert len(missing) == 1
    assert contract.read_text(encoding="utf-8") == body, "--fix must not touch a row it cannot verify"
