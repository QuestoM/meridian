"""Proof that the wave gate bites.

A guard that has never failed has never been shown to work. This campaign has
paid for that sentence more than once, so every case below constructs the exact
shape of a dossier that would have wasted a builder round, and asserts the gate
refuses it.

The last test is the one that matters most: a dossier whose line counts have
drifted from the repository. That is not a hypothetical. Every dossier written
today will be wrong within two waves, and a dossier that is wrong but believed is
worse than one that is absent.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts" / "gauntlet" / "wave_preflight.py"

spec = importlib.util.spec_from_file_location("wave_preflight", MODULE)
preflight = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(preflight)


COMPLETE = """# P99 dossier

## Job stories and their done conditions
A planner sees the pace. Done when the number matches the ledger.

## Baseline numbers
42 rows, 3 refusals.

## File inventory

| path | lines | note |
|---|---|---|
| `{path}` | {lines} lines | the only file |

## The API surface this piece owns
GET /api/nothing

## Reference product, and what to compare
The reference board, its column order.

## Trade facts that bind this piece
A make good is owed, not gifted.

## What is already built
The store and its two refusals.

## Exact commands
~/.venvs/meridian/bin/python -m pytest tests/test_nothing.py
"""


def _dossier(tmp_path: Path, body: str, piece: str = "P99") -> None:
    """Point the module at a scratch dossier directory holding one file."""
    folder = tmp_path / "dossiers"
    folder.mkdir(exist_ok=True)
    (folder / f"{piece}.md").write_text(body, encoding="utf-8")
    preflight.DOSSIERS = folder


def _real_file() -> tuple[str, int]:
    """A file that exists in this repository, and its true line count."""
    rel = "scripts/gauntlet/wave_preflight.py"
    lines = len((ROOT / rel).read_text(encoding="utf-8").splitlines())
    return rel, lines


def test_a_missing_dossier_stops_the_wave(tmp_path, monkeypatch):
    monkeypatch.setattr(preflight, "DOSSIERS", tmp_path / "dossiers")
    problems = preflight.check_dossier("P99")
    assert problems, "a piece with no dossier must not be launchable"
    assert "no dossier" in problems[0]


def test_a_complete_dossier_passes(tmp_path, monkeypatch):
    rel, lines = _real_file()
    monkeypatch.setattr(preflight, "ROOT", ROOT)
    _dossier(tmp_path, COMPLETE.format(path=rel, lines=lines))
    assert preflight.check_dossier("P99") == []


def test_a_missing_section_stops_the_wave(tmp_path, monkeypatch):
    rel, lines = _real_file()
    body = COMPLETE.format(path=rel, lines=lines).replace(
        "## Trade facts that bind this piece", "## Some other heading"
    )
    _dossier(tmp_path, body)
    problems = preflight.check_dossier("P99")
    assert any("Trade facts" in p for p in problems)


def test_an_unfinished_dossier_stops_the_wave(tmp_path):
    rel, lines = _real_file()
    body = COMPLETE.format(path=rel, lines=lines).replace("GET /api/nothing", "TODO")
    _dossier(tmp_path, body)
    problems = preflight.check_dossier("P99")
    assert any("started and not finished" in p for p in problems)


def test_an_inventory_with_no_rows_stops_the_wave(tmp_path):
    rel, lines = _real_file()
    body = COMPLETE.format(path=rel, lines=lines).replace(f"| `{rel}` |", "| plain text |")
    _dossier(tmp_path, body)
    problems = preflight.check_dossier("P99")
    assert any("no rows" in p for p in problems)


def test_a_path_that_no_longer_exists_stops_the_wave(tmp_path):
    body = COMPLETE.format(path="kairos/this_was_deleted.py", lines=100)
    _dossier(tmp_path, body)
    problems = preflight.check_dossier("P99")
    assert any("does not exist" in p for p in problems)


def test_a_rotted_line_count_stops_the_wave(tmp_path):
    """The case this gate is really for.

    The dossier was true when it was written. The file grew. Nothing announced
    it. A builder reads a count that says there is room and opens a file that is
    over the cap.
    """
    rel, lines = _real_file()
    body = COMPLETE.format(path=rel, lines=lines + 37)
    _dossier(tmp_path, body)
    problems = preflight.check_dossier("P99")
    assert any("has rotted" in p for p in problems)


def test_the_settings_check_catches_a_locale_left_behind(tmp_path, monkeypatch):
    """The pollution that shipped twice, caught before a wave rather than after."""
    store = tmp_path / "kairos_settings.json"
    store.write_text(
        '{"locale": "en", "direction": "ltr", "operator_channel": "x"}', encoding="utf-8"
    )
    monkeypatch.setattr(preflight, "SETTINGS", store)
    problems = preflight.check_settings()
    assert len(problems) == 2
    assert any("locale" in p for p in problems)
    assert any("direction" in p for p in problems)


def test_a_cleared_operator_channel_stops_the_wave(tmp_path, monkeypatch):
    store = tmp_path / "kairos_settings.json"
    store.write_text(
        '{"locale": "he", "direction": "rtl", "operator_channel": ""}', encoding="utf-8"
    )
    monkeypatch.setattr(preflight, "SETTINGS", store)
    problems = preflight.check_settings()
    assert any("operator_channel" in p for p in problems)


def test_the_live_settings_are_clean_right_now():
    """Not a unit test. A standing assertion about the tree you are about to work in."""
    assert preflight.check_settings() == []


@pytest.mark.parametrize("argv", [["--pieces", ""], ["--pieces", "  "]])
def test_naming_no_pieces_is_refused(argv):
    assert preflight.main(argv) == 2
