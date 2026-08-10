"""The comparison harness must preserve failures from each golden script."""

from __future__ import annotations

import sys
from pathlib import Path

GAUNTLET = Path(__file__).resolve().parents[1] / "scripts" / "gauntlet"
if str(GAUNTLET) not in sys.path:
    sys.path.insert(0, str(GAUNTLET))

import checks_engine  # noqa: E402


HASH = "a" * 64


def _answer(returncode: int = 0, matches: list[bool] | None = None) -> dict:
    return {
        "csv_sha256": HASH,
        "aggregate_sha256": HASH,
        "matches_own_golden": [True] if matches is None else matches,
        "returncode": returncode,
    }


def test_equal_hashes_cannot_hide_a_failed_working_golden(monkeypatch, tmp_path) -> None:
    answers = iter([(_answer(), ""), (_answer(returncode=1), "")])
    monkeypatch.setattr(checks_engine, "_golden", lambda *_args: next(answers))

    result = checks_engine.check_engine_golden(
        sys.executable, tmp_path / "ref", tmp_path / "work", tmp_path / "scratch", 10,
    )

    assert result.status == "fail"
    assert "exited 1" in result.summary


def test_false_own_golden_match_cannot_be_only_a_note(monkeypatch, tmp_path) -> None:
    answers = iter([(_answer(), ""), (_answer(matches=[True, False]), "")])
    monkeypatch.setattr(checks_engine, "_golden", lambda *_args: next(answers))

    result = checks_engine.check_engine_golden(
        sys.executable, tmp_path / "ref", tmp_path / "work", tmp_path / "scratch", 10,
    )

    assert result.status == "fail"
    assert "does not reproduce" in result.summary
