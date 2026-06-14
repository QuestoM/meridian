"""Tests for the run log observability layer.

These prove the audit trail is honest and reproducible: a record built from a
fixed run id, a fixed timestamp and a real temp file round-trips through the
JSONL log unchanged, and the recorded checksum is exactly the sha256 of the
file's bytes (and stable across calls). Everything uses tmp_path, so no real
xlsx or output directory is touched.
"""

from __future__ import annotations

import hashlib
import json

from kairos.observability.run_log import (
    KAIROS_ENGINE_VERSION,
    RunRecord,
    build_run_record,
    checksum_file,
    read_run_log,
    record_to_dict,
    write_run_log,
)

FIXED_RUN_ID = "run-2026-06-14-0001"
FIXED_CREATED_AT = "2026-06-14T12:00:00+00:00"

SUMMARY = {
    "total_breaks": 12,
    "projected_revenue": 48250.5,
    "average_retention": 87.3,
    "compliant": True,
}
GUARDRAILS = {"max_breaks_per_hour": 4, "min_retention_floor": 0.8}
ASSUMPTIONS = {"revenue_weight": 0.5}


def _source_file(tmp_path) -> "object":
    source = tmp_path / "programmes.csv"
    source.write_bytes(b"Title,Channel\nshow,channel\n")
    return source


def _build(tmp_path) -> RunRecord:
    return build_run_record(
        run_id=FIXED_RUN_ID,
        created_at=FIXED_CREATED_AT,
        channel="קשת 12",
        day="2026-06-14",
        source_paths={"programmes": _source_file(tmp_path)},
        guardrails=GUARDRAILS,
        assumptions=ASSUMPTIONS,
        summary=SUMMARY,
        segment_count=3,
    )


def test_checksum_matches_hashlib_and_is_stable(tmp_path) -> None:
    source = _source_file(tmp_path)
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    first = checksum_file(source)
    second = checksum_file(source)
    assert first == expected
    # Provenance must be deterministic: the same bytes hash the same every time.
    assert first == second


def test_checksum_of_missing_file_is_none(tmp_path) -> None:
    assert checksum_file(tmp_path / "not_here.csv") is None


def test_build_run_record_stamps_engine_version_and_checksum(tmp_path) -> None:
    source = _source_file(tmp_path)
    record = _build(tmp_path)
    assert record.engine_version == KAIROS_ENGINE_VERSION
    assert record.run_id == FIXED_RUN_ID
    assert record.created_at == FIXED_CREATED_AT
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    assert record.input_checksums["programmes"] == expected


def test_record_only_holds_what_the_caller_passes(tmp_path) -> None:
    record = _build(tmp_path)
    # No fabrication: the headline metrics are exactly the caller's summary.
    assert record.summary == SUMMARY
    assert record.guardrails == GUARDRAILS
    assert record.assumptions == ASSUMPTIONS
    assert record.segment_count == 3


def test_write_then_read_round_trips(tmp_path) -> None:
    record = _build(tmp_path)
    log_path = tmp_path / "output" / "run_log.jsonl"
    written = write_run_log(record, log_path)
    assert written == log_path

    rows = read_run_log(log_path)
    assert len(rows) == 1
    assert rows[0] == record_to_dict(record)
    # The checksum survives the JSONL round-trip intact.
    expected = hashlib.sha256((tmp_path / "programmes.csv").read_bytes()).hexdigest()
    assert rows[0]["input_checksums"]["programmes"] == expected


def test_write_appends_one_line_per_call(tmp_path) -> None:
    record = _build(tmp_path)
    log_path = tmp_path / "output" / "run_log.jsonl"
    write_run_log(record, log_path)
    write_run_log(record, log_path)

    # Two appends, two lines, two records read back.
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 2
    assert len(read_run_log(log_path)) == 2


def test_read_missing_log_is_empty(tmp_path) -> None:
    assert read_run_log(tmp_path / "nope.jsonl") == []


def test_record_to_dict_is_json_serialisable(tmp_path) -> None:
    record = _build(tmp_path)
    payload = record_to_dict(record)
    # If this does not raise, the dict is genuinely json-serialisable.
    reloaded = json.loads(json.dumps(payload))
    assert reloaded["run_id"] == FIXED_RUN_ID
