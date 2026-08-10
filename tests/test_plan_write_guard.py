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
import os
import sys
from pathlib import Path
from types import SimpleNamespace

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


def _frame_with_current_snapshots() -> pd.DataFrame:
    from kairos.export.schedule_fingerprint import (
        active_override_digest,
        pricing_config_digest,
        settings_slice,
    )
    from kairos.export.schedule_freshness import schedule_input_fingerprints
    from kairos_api.server import _load_settings

    frame = _one_row_frame()
    frame.attrs["fingerprint_snapshot"] = {
        "settings": settings_slice(_load_settings()),
        "pricing_config_sha256": pricing_config_digest(ROOT),
        "active_overrides": active_override_digest(ROOT),
        "run_context": None,
    }
    frame.attrs["input_fingerprints_snapshot"] = schedule_input_fingerprints(ROOT)
    frame.attrs["revenue_provenance"] = {"basis": "engine_segment_tvr"}
    frame.attrs["shipped_input_eligible"] = True
    frame.attrs["shipped_input_refusal_reasons"] = ()
    return frame


def test_pytest_defaults_the_shipped_plan_to_read_only() -> None:
    """The shared test bootstrap protects the plan before product code imports."""
    assert os.environ.get(READONLY_ENV) == "1"


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

    write_weekly_schedule(frame=_frame_with_current_snapshots(), replace_shipped_plan=True)

    record = json.loads(Path(str(relocated) + PROVENANCE_SUFFIX).read_text(encoding="utf-8"))
    assert len(record["writes"]) == 1
    entry = record["writes"][0]
    assert entry["written_at"] and entry["pid"]
    assert Path(__file__).name in entry["caller"], (
        "the record must name the calling frame, or it cannot attribute the next one"
    )


def test_fingerprint_publish_failure_rolls_back_the_csv_pair(monkeypatch, tmp_path) -> None:
    target = tmp_path / "weekly_break_schedule.csv"
    first = _one_row_frame()
    first.loc[0, "channel"] = "before"
    write_weekly_schedule(target, frame=first)
    fingerprint = Path(str(target) + ".fingerprint.json")
    before_csv = target.read_bytes()
    before_fingerprint = fingerprint.read_bytes()

    second = _one_row_frame()
    second.loc[0, "channel"] = "after"
    real_replace = schedule_module.os.replace
    calls = 0

    def fail_fingerprint_replace(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected fingerprint publish failure")
        return real_replace(source, destination)

    monkeypatch.setattr(schedule_module.os, "replace", fail_fingerprint_replace)
    with pytest.raises(OSError, match="injected fingerprint"):
        write_weekly_schedule(target, frame=second)

    assert target.read_bytes() == before_csv
    assert fingerprint.read_bytes() == before_fingerprint


def test_successful_recompute_test_is_redirected_and_never_dirties_output(
    monkeypatch, tmp_path,
) -> None:
    """Exercise the real recompute body while its writer targets ``tmp_path``.

    The global read-only default proves an accidental shipped-path write would
    fail. This test also proves the intended testing pattern remains useful: a
    successful recompute can run against a temporary artifact without changing
    either committed output file.
    """
    from kairos_api import recompute_api

    fingerprint = Path(str(DEFAULT_OUTPUT_PATH) + ".fingerprint.json")
    before_plan = DEFAULT_OUTPUT_PATH.read_bytes()
    before_fingerprint = fingerprint.read_bytes()
    target = tmp_path / "weekly_break_schedule.csv"
    frame = _one_row_frame()
    frame.loc[0, "channel"] = "רשת 13"
    frame.loc[0, "date"] = "2024-11-01"
    frame.loc[0, "num_breaks"] = 1
    frame.loc[0, "predicted_revenue"] = 10.0
    saved = SimpleNamespace(
        revenue_weight=60,
        risk_lambda=0.0,
        operator_channel="רשת 13",
        objective_mode="blend",
    )

    monkeypatch.setattr(recompute_api, "_load_settings", lambda: saved)
    monkeypatch.setattr(recompute_api, "_model_dump", lambda _saved: {})
    monkeypatch.setattr(recompute_api, "_reference_today", lambda _saved: None)
    monkeypatch.setattr(recompute_api, "build_weekly_schedule", lambda **_kwargs: frame)

    def write_to_tmp(*, frame, replace_shipped_plan):
        assert replace_shipped_plan is True
        return write_weekly_schedule(target, frame=frame)

    monkeypatch.setattr(recompute_api, "write_weekly_schedule", write_to_tmp)

    result = recompute_api._run_recompute()

    assert result["ok"] is True
    assert result["path"] == str(target)
    assert target.exists()
    assert DEFAULT_OUTPUT_PATH.read_bytes() == before_plan
    assert fingerprint.read_bytes() == before_fingerprint
