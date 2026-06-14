"""Run log: make every optimization run auditable and reproducible.

A single optimization run decides where commercial breaks go and for how long.
To trust those decisions later we need to know exactly what produced them: which
input files were read (by content, not just by name), which guardrails and
assumptions were in force, and the headline metrics that came out. This module
records that provenance as one immutable :class:`RunRecord` per run and appends
it as a JSON line to ``output/run_log.jsonl``.

The record stores only what the caller passes. It never invents a metric, never
reads the clock, and never generates a run id: timestamps and identifiers come
from the caller so the same inputs always produce the same record. The headline
metric names mirror :func:`kairos.service.result_to_dict` exactly
(``total_breaks``, ``projected_revenue``, ``average_retention``, ``compliant``)
so the audit trail speaks the same language as the live plan.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

# A single source of truth for the engine version stamped onto every record.
# Bump this when a change to the engine would change a run's decisions, so the
# log distinguishes plans produced by different engine behaviour.
KAIROS_ENGINE_VERSION = "1.0.0"

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_LOG_PATH = ROOT / "output" / "run_log.jsonl"

_CHECKSUM_CHUNK_BYTES = 1024 * 1024


def checksum_file(path: str | Path) -> Optional[str]:
    """Return the sha256 hex digest of a file's bytes, or ``None`` if absent.

    Provenance is honest: a file that is not there has no checksum, so this
    returns ``None`` rather than hashing emptiness or pretending the file
    existed. A path that exists but is not a regular file (a directory, say) is
    a caller error and raises, because that is a mistake, not a missing input.
    """
    path = Path(path)
    if not path.exists():
        return None
    if not path.is_file():
        raise ValueError(f"checksum_file expects a file, got: {path}")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHECKSUM_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunRecord:
    """An immutable, audit-ready snapshot of one optimization run.

    Every field is supplied by the caller except ``engine_version``, which is
    stamped from :data:`KAIROS_ENGINE_VERSION`. The record holds no logic and
    reads no external state; it is a faithful container for what the run layer
    chose to record.
    """

    run_id: str
    created_at: str
    channel: Optional[str]
    day: Optional[str]
    input_checksums: dict[str, Optional[str]]
    guardrails: dict[str, Any]
    assumptions: dict[str, Any]
    summary: dict[str, Any]
    segment_count: int
    engine_version: str = KAIROS_ENGINE_VERSION


def build_run_record(
    *,
    run_id: str,
    created_at: str,
    channel: Optional[str],
    day: Optional[str],
    source_paths: Mapping[str, str | Path],
    guardrails: Mapping[str, Any],
    assumptions: Mapping[str, Any],
    summary: Mapping[str, Any],
    segment_count: int,
) -> RunRecord:
    """Assemble a :class:`RunRecord` from a run's inputs and outputs.

    ``source_paths`` maps a logical source name (for example ``"programmes"``)
    onto the file that fed the run; each is hashed for provenance, with a
    missing file recorded as ``None`` rather than dropped. ``run_id`` and
    ``created_at`` are passed in by the caller so the record is reproducible;
    this function never calls the clock or invents randomness, and it copies the
    mappings so later mutation by the caller cannot rewrite the record.
    """
    input_checksums = {
        name: checksum_file(path) for name, path in source_paths.items()
    }
    return RunRecord(
        run_id=run_id,
        created_at=created_at,
        channel=channel,
        day=day,
        input_checksums=input_checksums,
        guardrails=dict(guardrails),
        assumptions=dict(assumptions),
        summary=dict(summary),
        segment_count=int(segment_count),
    )


def record_to_dict(record: RunRecord) -> dict[str, Any]:
    """Serialise a :class:`RunRecord` into a json-serialisable dict."""
    return asdict(record)


def write_run_log(record: RunRecord, path: Optional[str | Path] = None) -> Path:
    """Append one record as a single JSON line to the run log.

    Defaults to ``output/run_log.jsonl`` under the repo root, creating the
    parent directory if needed. Returns the path written so the caller can
    surface or assert on it. Each call appends exactly one line, so the log is a
    growing, append-only audit trail rather than a file that gets overwritten.
    """
    target = Path(path) if path is not None else DEFAULT_RUN_LOG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record_to_dict(record), ensure_ascii=False, sort_keys=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    return target


def read_run_log(path: Optional[str | Path] = None) -> list[dict[str, Any]]:
    """Read the run log back as a list of dicts, oldest first.

    Returns an empty list when the log does not exist yet, so a first read on a
    fresh checkout is not an error. Blank lines are skipped so a trailing
    newline never produces a phantom record.
    """
    target = Path(path) if path is not None else DEFAULT_RUN_LOG_PATH
    if not target.exists():
        return []
    records: list[dict[str, Any]] = []
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
