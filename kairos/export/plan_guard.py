"""A read-only mode for the shipped plan, and a record of who replaced it.

:func:`kairos.export.schedule.write_weekly_schedule` already refuses to write the
committed artifact unless the caller passes ``replace_shipped_plan=True``. That
guard closes the class of writer that names no path and never knows it wrote. It
does not close the one that actually rewrote the plan four times on 2026-08-09.

Every one of those writes came through ``POST /api/recompute-schedule``, reached
from Kai's apply path (``kairos_api.assistant_apply_kinds._apply_recompute``),
which starts a background recompute job. Replacing the plan is that endpoint's
entire job, so it passes the flag and the refusal lets it through, correctly. The
writes were legitimate product actions. What was wrong was the tree they landed
on: a development checkout where agents drive the product and the plan of record
is version-controlled.

So this module adds the control that fits that shape.

- **``KAIROS_PLAN_READONLY`` refuses every write to the shipped artifact**,
  deliberate ones included. Set it wherever an agent may drive the product, and a
  recompute fails loudly instead of quietly restating the operator's money.
- **An allowed write leaves provenance.** Attributing one of those four writes
  took a day of bisection: the plan carries no record of who wrote it, and the
  freshness sidecar's timestamp was the only thread to pull. A line naming the
  process, the argv and the calling frame turns the next one into a grep.

Provenance is best-effort and never blocks a write. The plan is the critical
path; a missing record is a worse day later, a failed write is a worse day now.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

READONLY_ENV = "KAIROS_PLAN_READONLY"
PROVENANCE_SUFFIX = ".writes.meta.json"
MAX_RECORDED_WRITES = 20


class PlanArtifactProtected(RuntimeError):
    """Raised when the shipped plan is read-only on this tree and something wrote."""


def _truthy(value: Any) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def is_shipped_plan(target: Any, shipped: Any) -> bool:
    """Whether these name the same file, resolved, without needing it to exist."""
    try:
        return Path(str(target)).resolve() == Path(str(shipped)).resolve()
    except OSError:  # pragma: no cover - an unresolvable path is not the artifact
        return False


def _caller() -> str:
    """The innermost frame outside this package: who asked for the write."""
    package = str(Path(__file__).resolve().parent)
    for frame in reversed(traceback.extract_stack()[:-2]):
        if not str(Path(frame.filename).resolve().parent).startswith(package):
            return f"{frame.filename}:{frame.lineno} in {frame.name}()"
    return "unknown"


def authorize_shipped_plan_write(target: Any, shipped: Any) -> None:
    """Refuse this write if the plan is read-only here; otherwise record it.

    Runs before a single byte is written, so a refusal leaves the plan, its
    freshness sidecar and its committed fingerprint exactly as they were. Any
    path other than the shipped artifact passes straight through.
    """
    if not is_shipped_plan(target, shipped):
        return
    if _truthy(os.environ.get(READONLY_ENV)):
        raise PlanArtifactProtected(
            f"{READONLY_ENV} is set, so the shipped plan {shipped} is read-only on this tree "
            f"and nothing was written. Asked for by {_caller()}. A recompute is a real product "
            "action and the deliberate-write flag cannot tell it apart from an accident, so a "
            "tree where agents drive the product refuses all of them. Unset the variable where "
            "replacing the plan of record is the point."
        )
    _record_write(target)


def _record_write(target: Any) -> None:
    """Append this write to the plan's provenance record. Never raises."""
    try:
        path = Path(str(target) + PROVENANCE_SUFFIX)
        writes: list[dict[str, Any]] = []
        if path.exists():
            parsed = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict) and isinstance(parsed.get("writes"), list):
                writes = parsed["writes"]
        writes.append({
            "written_at": datetime.now(timezone.utc).isoformat(),
            "pid": os.getpid(),
            "argv": [str(arg) for arg in sys.argv[:8]],
            "caller": _caller(),
        })
        payload = {"artifact": Path(str(target)).name, "writes": writes[-MAX_RECORDED_WRITES:]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:  # pragma: no cover - provenance never blocks the plan write
        pass
