"""Training runs started from the console, into the console's own store.

Training is the one activity the operator side may never reach and the one the
model steward's day is made of. Until now it was a shell command that left no
trace in any of the product's three audit systems, so nobody could answer who
ran the model, with which gate overrides, or what the run would have changed.

This module makes a run an object:

- **It never writes the shipped artifact.** The output path is computed here and
  is always inside ``models/releases/training_runs/<run_id>/``. The caller
  cannot choose it. Overwriting ``models/tv_break_coefficients.json`` would move
  revenue in every plan run afterwards, which is an owner-approved act, so this
  module cannot do it even by mistake, and a test asserts that.
- **It records the flags.** The five gate-override flags are the provenance hole
  the console itself reports: none of them is written into the artifact, so a
  forced gate and a self-activated one are indistinguishable afterwards. A run
  started here records exactly which were used, so for every such run the hole
  is closed.
- **It records who and how long.** Measured, from the process itself.
- **It reports what the run would change**, by comparing the produced artifact's
  gates and coefficients against the shipped ones.

Measured cost on the reference data: the retention training takes about twenty
seconds and the audience training about fourteen, so a run is a foreground act
with a spinner rather than an overnight job. It still runs in a thread, because
a request that blocks for twenty seconds is a request that times out somewhere.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kairos_api import model_console_artifacts as artifacts
from kairos_api import model_console_candidates as candidates
from kairos_api import model_version_store as store

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
RUNS_FILE = "training_runs.json"
RUNS_SUBDIR = "training_runs"

# The two training acts, and nothing else is one. Each names the script that
# performs it, the artifact it produces and the flags it accepts.
TRAINERS: dict[str, dict[str, Any]] = {
    "retention": {
        "script": "scripts/compute_measured_coefficients.py",
        "output_name": "tv_break_coefficients.json",
        "label_en": "Retention coefficients",
        "label_he": "מקדמי שימור",
        "measured_seconds": None,
    },
    "audience": {
        "script": "scripts/compute_audience_model.py",
        "output_name": "audience_model.json",
        "label_en": "Audience model",
        "label_he": "מודל הקהל",
        "measured_seconds": None,
    },
}

_RUNNING: dict[str, str] = {}
_LOCK = threading.RLock()


class TrainingError(ValueError):
    """Raised when a requested run is not one this module will perform."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def runs_path() -> Path:
    return store.store_dir() / RUNS_FILE


def output_dir(run_id: str) -> Path:
    """Always inside the releases store. The caller never chooses this."""
    return store.store_dir() / RUNS_SUBDIR / run_id


def runs() -> list[dict[str, Any]]:
    """Every recorded run, newest first."""
    path = runs_path()
    if not path.is_file():
        return []
    try:
        records = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("training run log unreadable (%s)", exc)
        return []
    if not isinstance(records, dict):
        return []
    out = [record for record in records.values() if isinstance(record, dict)]
    return sorted(out, key=lambda record: str(record.get("started_at") or ""), reverse=True)


def _save(record: dict[str, Any]) -> dict[str, Any]:
    with _LOCK:
        path = runs_path()
        try:
            existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        except (OSError, ValueError):
            existing = {}
        if not isinstance(existing, dict):
            existing = {}
        existing[record["run_id"]] = record
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(existing, handle, ensure_ascii=False, indent=1)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    return record


def available_flags(artifact: str) -> list[dict[str, Any]]:
    """The gate overrides this trainer accepts, read from its own script."""
    if artifact != "retention":
        return []
    return artifacts.gate_override_flags()


def build_command(artifact: str, run_id: str, flags: Optional[dict[str, str]] = None) -> list[str]:
    """The exact command, with the output forced inside the releases store."""
    if artifact not in TRAINERS:
        raise TrainingError(f"artifact must be one of {sorted(TRAINERS)}, got {artifact!r}")
    spec = TRAINERS[artifact]
    accepted = {row["flag"] for row in available_flags(artifact)}
    command = [sys.executable, str(ROOT / spec["script"])]
    for flag, value in sorted((flags or {}).items()):
        if flag not in accepted:
            raise TrainingError(f"{flag} is not a gate override this trainer accepts")
        if value not in ("force-on", "force-off"):
            raise TrainingError(f"{flag} must be force-on or force-off, got {value!r}")
        command += [flag, value]
    command += ["--output", str(output_dir(run_id) / spec["output_name"])]
    return command


def start(artifact: str, *, actor: str = "", flags: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """Start one training run in the background and record it immediately."""
    if artifact not in TRAINERS:
        raise TrainingError(f"artifact must be one of {sorted(TRAINERS)}, got {artifact!r}")
    with _LOCK:
        if artifact in _RUNNING:
            raise TrainingError(f"a {artifact} training run is already in flight")
        run_id = f"tr-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:6]}"
        _RUNNING[artifact] = run_id
    command = build_command(artifact, run_id, flags)
    record = {
        "run_id": run_id,
        "artifact": artifact,
        "label_en": TRAINERS[artifact]["label_en"],
        "label_he": TRAINERS[artifact]["label_he"],
        "state": "running",
        "started_at": _now(),
        "actor": str(actor or "").strip() or "unknown (login is not set up)",
        "flags": dict(sorted((flags or {}).items())),
        "command": _printable(command),
        "output_path": str(output_dir(run_id) / TRAINERS[artifact]["output_name"]),
        "writes_shipped_artifact": False,
        "shipped_before": artifacts.model_version().get("id"),
    }
    _save(record)
    threading.Thread(target=_run, args=(record, command), name=f"kairos-train-{artifact}",
                     daemon=True).start()
    return record


def _printable(command: list[str]) -> str:
    return " ".join(part if part.startswith("-") or "/" not in part
                    else Path(part).relative_to(ROOT).as_posix()
                    if str(part).startswith(str(ROOT)) else part
                    for part in command)


def _run(record: dict[str, Any], command: list[str]) -> None:
    started = time.monotonic()
    output = Path(record["output_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONUTF8": "1"}
    try:
        completed = subprocess.run(command, cwd=str(ROOT), env=environment,
                                   capture_output=True, text=True, timeout=1800)
        record["exit_code"] = completed.returncode
        record["stdout_tail"] = "\n".join(completed.stdout.strip().splitlines()[-24:])
        record["stderr_tail"] = "\n".join(completed.stderr.strip().splitlines()[-8:])
        record["state"] = "done" if completed.returncode == 0 and output.is_file() else "failed"
    except Exception as exc:  # noqa: BLE001 - a failed run is recorded, never hidden
        logger.exception("training run %s failed", record["run_id"])
        record["state"] = "failed"
        record["error"] = repr(exc)
    finally:
        record["duration_seconds"] = round(time.monotonic() - started, 1)
        record["finished_at"] = _now()
        if record["state"] == "done":
            record["produced"] = _produced(output)
            record["would_change"] = _would_change(record["artifact"], output)
        _save(record)
        with _LOCK:
            _RUNNING.pop(record["artifact"], None)


def _produced(output: Path) -> dict[str, Any]:
    import hashlib

    payload = {}
    try:
        payload = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return {
        "path": output.relative_to(ROOT).as_posix(),
        "bytes": output.stat().st_size if output.is_file() else None,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest() if output.is_file() else None,
        "computed_at": metadata.get("computed_at") or payload.get("computed_at"),
    }


def _would_change(artifact: str, output: Path) -> dict[str, Any]:
    """What this run would change if it were adopted, gates first, then cells."""
    if artifact != "retention":
        return _audience_change(output)
    try:
        produced = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False, "reason": "the produced artifact could not be read"}
    shipped = artifacts.read_artifact(artifacts.RETENTION_ARTIFACT) or {}
    shipped_meta = shipped.get("metadata") if isinstance(shipped.get("metadata"), dict) else {}
    produced_meta = produced.get("metadata") if isinstance(produced.get("metadata"), dict) else {}
    return {
        "available": True,
        "gate_deltas": candidates.gate_deltas(shipped_meta, produced_meta),
        "coefficient_deltas": candidates.coefficient_deltas(shipped, produced),
        "money_state": "not_measured",
        "money_note_en": "Measuring the money needs the whole weekly plan computed twice, which this run does not do.",
        "money_note_he": "מדידת הכסף דורשת חישוב של כל התוכנית השבועית פעמיים, וההרצה הזו אינה עושה זאת.",
    }


def _audience_change(output: Path) -> dict[str, Any]:
    try:
        produced = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"available": False, "reason": "the produced artifact could not be read"}
    shipped_gates = artifacts.audience_gates()
    produced_gates = produced.get("gates") if isinstance(produced.get("gates"), dict) else {}
    rows = []
    for family in sorted(set(shipped_gates) | set(produced_gates)):
        before = (shipped_gates.get(family) or {}).get("verdict")
        after = (produced_gates.get(family) or {}).get("verdict")
        if before != after:
            rows.append({"key": family, "shipped": before, "candidate": after})
    return {"available": True, "gate_deltas": rows, "coefficient_deltas": None,
            "money_state": "not_applicable",
            "money_note_en": "The audience model changes forward-dated ratings only while the activation switch is on.",
            "money_note_he": "מודל הקהל משנה רייטינג לתאריכים עתידיים רק כשמתג ההפעלה דלוק."}


def in_flight() -> dict[str, str]:
    with _LOCK:
        return dict(_RUNNING)


def payload() -> dict[str, Any]:
    """The training section: what can be run, what is running, what has run."""
    recorded = runs()
    return {
        "trainers": [
            {
                "artifact": artifact,
                "label_en": spec["label_en"],
                "label_he": spec["label_he"],
                "script": spec["script"],
                "flags": available_flags(artifact),
                "measured_seconds": _median_duration(recorded, artifact),
                "writes": f"models/releases/{RUNS_SUBDIR}/<run id>/{spec['output_name']}",
            }
            for artifact, spec in TRAINERS.items()
        ],
        "in_flight": in_flight(),
        "runs": recorded,
        "safety_en": "A run started here writes into the releases store and never over the shipped artifact, so no plan figure moves until somebody adopts it deliberately.",
        "safety_he": "הרצה שמתחילה כאן כותבת למאגר הגרסאות ולעולם לא על הקובץ המשודר, ולכן שום מספר בתוכנית אינו זז עד שמישהו מטמיע אותה במפורש.",
    }


def _median_duration(recorded: list[dict[str, Any]], artifact: str) -> Optional[float]:
    values = sorted(float(record["duration_seconds"]) for record in recorded
                    if record.get("artifact") == artifact
                    and isinstance(record.get("duration_seconds"), (int, float)))
    if not values:
        return None
    return round(values[len(values) // 2], 1)
