"""The regular golden entry point must inspect the artifacts we actually ship."""

from __future__ import annotations

import hashlib
import json

import pandas as pd

from tests import golden_weekly_schedule as golden


def _tiny_frame() -> pd.DataFrame:
    return pd.DataFrame([{
        "channel": "owned",
        "date": "2024-11-01",
        "predicted_revenue": 12.5,
        "predicted_retention": 0.25,
        "num_breaks": 1,
    }])


def _patch_embedded_baseline(monkeypatch, frame: pd.DataFrame) -> None:
    records = golden.aggregate_records(frame)
    monkeypatch.setattr(golden, "build_reference_frame", lambda: frame)
    monkeypatch.setattr(golden, "GOLDEN_ROWS", len(frame))
    monkeypatch.setattr(golden, "GOLDEN_CSV_SHA256", golden.csv_hash(frame))
    monkeypatch.setattr(golden, "GOLDEN_AGG", records)
    monkeypatch.setattr(golden, "GOLDEN_AGG_SHA256", golden.agg_hash(records))


def test_evaluate_fails_when_shipped_artifacts_are_missing(monkeypatch, tmp_path) -> None:
    frame = _tiny_frame()
    _patch_embedded_baseline(monkeypatch, frame)
    monkeypatch.setattr(golden, "SHIPPED_PLAN_PATH", tmp_path / "missing.csv")
    monkeypatch.setattr(golden, "SHIPPED_FINGERPRINT_PATH", tmp_path / "missing.json")

    problems = golden.evaluate()[-1]
    assert any("unreadable" in problem for problem in problems)


def test_evaluate_fails_when_shipped_bytes_are_not_the_rebuild(monkeypatch, tmp_path) -> None:
    frame = _tiny_frame()
    _patch_embedded_baseline(monkeypatch, frame)
    shipped = tmp_path / "weekly_break_schedule.csv"
    shipped.write_text(frame.assign(predicted_revenue=99).to_csv(index=False), encoding="utf-8")
    fingerprint = tmp_path / "weekly_break_schedule.csv.fingerprint.json"
    fingerprint.write_text(json.dumps({
        "sha256": hashlib.sha256(shipped.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    monkeypatch.setattr(golden, "SHIPPED_PLAN_PATH", shipped)
    monkeypatch.setattr(golden, "SHIPPED_FINGERPRINT_PATH", fingerprint)

    problems = golden.evaluate()[-1]
    assert "the rebuilt golden bytes differ from the shipped plan" in problems
