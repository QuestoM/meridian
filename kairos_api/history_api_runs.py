"""One run, what it produced, and how it compares to the run before it.

``output/run_log.jsonl`` has recorded every optimization run since the engine
was built: the run id, the engine version, the checksum of every input file,
the guardrails and assumptions in force, and the headline outcome. Measured on
this deployment: 523 records between 2026-07-01 and 2026-07-31. Nothing in the
product has ever read it, so the question "what did the last run produce, and
how does it compare to the one before" has had an exact answer on disk and no
way to ask it.

Two rules hold here.

**Nothing is computed that the engine did not record.** A delta is a
subtraction of two recorded figures and it is present only when both sides
recorded the field. A missing field yields no delta rather than a zero, because
a zero would read as "nothing changed" when the truth is "not recorded".

**Comparable means the same scope.** A run over one day and a run over a whole
week are not two measurements of the same thing, so the previous run is the
previous run of the same channel and the same day, and when there is none the
answer says so instead of comparing across scopes.
"""

from __future__ import annotations

from typing import Any, Optional

# The recorded outcome fields a delta is taken on. ``compliant`` is a verdict
# rather than a quantity, so it is reported as a pair and never subtracted.
NUMERIC_FIELDS = (
    "projected_revenue",
    "total_breaks",
    "total_ad_seconds",
    "average_retention",
    "objective",
    "segment_count",
)

VERDICT_FIELDS = ("compliant",)


def _value(record: dict[str, Any], field: str) -> Any:
    if field in record:
        return record.get(field)
    summary = record.get("summary") or {}
    return summary.get(field)


def find_run(records: list[dict[str, Any]], run_id: str) -> Optional[dict[str, Any]]:
    """The record with this run id, or None. Callers pass an already scoped list,
    so a run on a channel the operator does not own is simply not found."""
    wanted = str(run_id or "")
    for record in records:
        if str(record.get("run_id") or "") == wanted:
            return record
    return None


def previous_run(records: list[dict[str, Any]], record: dict[str, Any]) -> Optional[dict[str, Any]]:
    """The run before this one over the same channel and the same day.

    The log is append-only and in file order, so "before" is position in the
    file, which is the order the runs actually happened in.
    """
    run_id = str(record.get("run_id") or "")
    channel = record.get("channel")
    day = record.get("day")
    seen: Optional[dict[str, Any]] = None
    for candidate in records:
        if str(candidate.get("run_id") or "") == run_id:
            return seen
        if candidate.get("channel") == channel and candidate.get("day") == day:
            seen = candidate
    return None


def delta(record: dict[str, Any], earlier: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The change from the earlier run to this one, field by field.

    ``state`` is tri-state on purpose: ``measured`` when both runs recorded the
    field, ``unavailable`` when one of them did not, and ``no_earlier_run`` for
    the whole block when there is nothing comparable to subtract from.
    """
    if earlier is None:
        return {"state": "no_earlier_run", "compared_run_id": None, "fields": {}}
    fields: dict[str, Any] = {}
    for field in NUMERIC_FIELDS:
        now = _value(record, field)
        before = _value(earlier, field)
        if isinstance(now, (int, float)) and isinstance(before, (int, float)) \
                and not isinstance(now, bool) and not isinstance(before, bool):
            fields[field] = {
                "state": "measured",
                "from": before,
                "to": now,
                "delta": round(float(now) - float(before), 6),
            }
        else:
            fields[field] = {"state": "unavailable", "from": before, "to": now, "delta": None}
    for field in VERDICT_FIELDS:
        now = _value(record, field)
        before = _value(earlier, field)
        state = "measured" if isinstance(now, bool) and isinstance(before, bool) else "unavailable"
        fields[field] = {"state": state, "from": before, "to": now, "delta": None}
    return {
        "state": "measured",
        "compared_run_id": earlier.get("run_id"),
        "compared_created_at": earlier.get("created_at"),
        "fields": fields,
    }


def detail(records: list[dict[str, Any]], record: dict[str, Any]) -> dict[str, Any]:
    """The full run record as a surface reads it, with its comparison.

    ``inputs`` carries the checksum of every file the run read. A checksum that
    is None means the file was absent at run time, which the run log records
    honestly rather than hashing emptiness.
    """
    earlier = previous_run(records, record)
    summary = record.get("summary") or {}
    return {
        "run_id": record.get("run_id"),
        "created_at": record.get("created_at"),
        "channel": record.get("channel"),
        "day": record.get("day"),
        "engine_version": record.get("engine_version"),
        "segment_count": record.get("segment_count"),
        "summary": dict(summary),
        "guardrails": dict(record.get("guardrails") or {}),
        "assumptions": dict(record.get("assumptions") or {}),
        "inputs": dict(record.get("input_checksums") or {}),
        "comparison": delta(record, earlier),
    }
