"""The shipped plan can be made read-only, and an allowed write says who made it.

``write_weekly_schedule`` refuses a caller that names no path unless it passes
``replace_shipped_plan=True``. That closes the writer that never knows it wrote.
It does not close the writer that actually replaced the plan four times on
2026-08-09: ``POST /api/recompute-schedule``, reached from Kai's apply path,
whose entire job is to replace the plan and which therefore passes the flag.

These tests hold the two controls that fit that shape. ``KAIROS_PLAN_READONLY``
refuses every write to the artifact, deliberate ones included, so a tree where
agents drive the product cannot restate the operator's money. And an allowed
write leaves provenance, because attributing one of those four took a day of
bisection against a plan that recorded nothing about who wrote it.

Nothing here writes the real artifact. The refusal tests assert its bytes are
untouched; the allowed-write test relocates the shipped path to a tmp file first.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.export import schedule as schedule_module  # noqa: E402
from kairos.export.plan_guard import (  # noqa: E402
    PROVENANCE_SUFFIX,
    READONLY_ENV,
    PlanArtifactProtected,
)
from kairos.export.schedule import COLUMNS, DEFAULT_OUTPUT_PATH, write_weekly_schedule  # noqa: E402


def _one_row_frame() -> pd.DataFrame:
    """A frame shaped like the export, small enough to make a write obvious."""
    return pd.DataFrame([{column: "" for column in COLUMNS}], columns=COLUMNS)


def test_readonly_mode_refuses_a_deliberate_recompute(monkeypatch) -> None:
    """The recompute endpoint's own call is refused when the tree is read-only.

    This is the write that really happened. It passed ``replace_shipped_plan``
    because replacing the plan is what a recompute is for, so the deliberate-write
    flag let it through, correctly, four times.
    """
    monkeypatch.setenv(READONLY_ENV, "1")
    before = DEFAULT_OUTPUT_PATH.read_bytes()

    with pytest.raises(PlanArtifactProtected) as raised:
        write_weekly_schedule(frame=_one_row_frame(), replace_shipped_plan=True)

    assert READONLY_ENV in str(raised.value), "the refusal must name the switch that caused it"
    assert DEFAULT_OUTPUT_PATH.read_bytes() == before, "a refused write must not touch the plan"


def test_readonly_mode_refuses_a_caller_that_names_the_shipped_path(monkeypatch) -> None:
    """Naming the artifact explicitly is the gap the deliberate-write flag cannot see.

    The flag only inspects callers that pass no path at all, so passing
    ``DEFAULT_OUTPUT_PATH`` by hand walks straight past it.
    """
    monkeypatch.setenv(READONLY_ENV, "1")
    before = DEFAULT_OUTPUT_PATH.read_bytes()

    with pytest.raises(PlanArtifactProtected):
        write_weekly_schedule(DEFAULT_OUTPUT_PATH, frame=_one_row_frame())

    assert DEFAULT_OUTPUT_PATH.read_bytes() == before, "a refused write must not touch the plan"


def test_read_only_mode_never_blocks_a_write_to_any_other_path(monkeypatch, tmp_path) -> None:
    """The guard protects one file, not the exporter. Scenarios and tests still write."""
    monkeypatch.setenv(READONLY_ENV, "1")
    target = tmp_path / "somewhere_else.csv"

    assert write_weekly_schedule(target, frame=_one_row_frame()) == target
    assert target.exists()


def test_an_allowed_write_records_who_made_it(monkeypatch, tmp_path) -> None:
    """With the tree writable, the plan carries a record of the process that wrote it."""
    monkeypatch.delenv(READONLY_ENV, raising=False)
    relocated = tmp_path / "weekly_break_schedule.csv"
    monkeypatch.setattr(schedule_module, "DEFAULT_OUTPUT_PATH", relocated)

    write_weekly_schedule(frame=_one_row_frame(), replace_shipped_plan=True)

    record = json.loads(Path(str(relocated) + PROVENANCE_SUFFIX).read_text(encoding="utf-8"))
    assert len(record["writes"]) == 1
    entry = record["writes"][0]
    assert entry["written_at"] and entry["pid"]
    assert Path(__file__).name in entry["caller"], (
        "the record must name the calling frame, or it cannot attribute the next one"
    )
