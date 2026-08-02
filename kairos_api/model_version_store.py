"""Model version identity on disk, and the ship or no-ship decisions against it.

Before this store, a trained model had no identity and a decision about one had
nowhere to live. Training overwrote an artifact in place, so the only handle
anybody had was a timestamp that the next training destroyed, and the question
"did we ship this, and why" had no answer a later reader could find.

Three records, three files, all under ``models/releases/``:

- **Observed versions.** Every model version the console has seen, keyed by the
  content digest of the artifacts themselves. The store never mints a version;
  it records the one the tree already has, the first time it sees it.
- **Decisions.** An append-only log of ship and no-ship verdicts, each against a
  named version, each carrying the evidence that existed when it was taken. A
  decision is never edited and never deleted.
- **Candidate measurements.** The money a candidate would move, measured by
  running the plan twice, stored against a fingerprint of everything that went
  into it so a stale figure can never read as current.

**Recording a ship verdict does not ship anything.** Copying a candidate over
the shipped artifact would move revenue in every future plan, which is an
owner-approved act, so the store records the verdict, names the measured
movement, and marks the adoption itself as escalated and not performed. That is
the whole safety property of this module and a test asserts it.

The release note is the one piece of training-side text that crosses to the
operator side, so its rule is enforced here rather than trusted: it may not
contain a gate verdict, a p-value or a coefficient, and a note that does is
refused with the word that refused it.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "models" / "releases"
DIR_ENV = "KAIROS_MODEL_RELEASES_DIR"

VERSIONS_FILE = "observed_versions.json"
DECISIONS_FILE = "decisions.jsonl"
MEASUREMENTS_FILE = "candidate_measurements.json"

DECISIONS = ("shipped", "not_shipped")
SUBJECTS = ("current", "candidate")

# Section 4.2's training lexicon, plus the Hebrew words that carry the same
# meaning. A release note is read by an operator who is not allowed a gate
# verdict, so a note carrying one is a leak and is refused at the door.
NOTE_FORBIDDEN = (
    "gate", "held_out", "held-out", "tau", "coefficient", "pooling", "p_value",
    "p-value", "p=", "rmse", "training_window", "wartime", "holdout",
    "שער", "מקדם", "מובהקות", "מבחן מוחזק", "איחוד", "חלון האימון",
)

NOTE_REFUSAL = {
    "en": "A release note may not carry a gate verdict, a p-value or a coefficient. Remove: ",
    "he": "הערת גרסה אינה יכולה לשאת הכרעת שער, ערך מובהקות או מקדם. יש להסיר: ",
}

MONEY_DIRECTIONS = ("up", "down", "none", "unknown")

_LOCK = threading.RLock()


class ModelVersionError(ValueError):
    """Raised when a proposed decision or release note is not usable."""


def store_dir() -> Path:
    value = os.getenv(DIR_ENV, "").strip()
    if not value:
        return DEFAULT_DIR
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _read_json(path: Path, fallback: Any) -> Any:
    if not path.is_file():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("model version store file %s unreadable (%s)", path, exc)
        return fallback


# ---------------------------------------------------------------------------
# Observed versions
# ---------------------------------------------------------------------------


def versions() -> list[dict[str, Any]]:
    """Every version the console has seen, oldest first."""
    records = _read_json(store_dir() / VERSIONS_FILE, [])
    return [record for record in records if isinstance(record, dict)] if isinstance(records, list) else []


def observe(version: dict[str, Any]) -> dict[str, Any]:
    """Record a version the tree currently holds, if it is not recorded already.

    Returns the stored record, which carries ``first_seen_at``. A version with
    no id (no artifact on disk) is not recorded: there is nothing to name.
    """
    version_id = str(version.get("id") or "")
    if not version_id:
        return {}
    with _LOCK:
        records = versions()
        for record in records:
            if record.get("id") == version_id:
                return record
        record = {
            "id": version_id,
            "name": version.get("name"),
            "short": version.get("short"),
            "first_seen_at": _now(),
            "trained_at": version.get("trained_at"),
            "artifacts": {
                kind: {key: block.get(key) for key in ("path", "sha256", "computed_at", "bytes")}
                for kind, block in (version.get("artifacts") or {}).items()
                if isinstance(block, dict) and block.get("present")
            },
        }
        records.append(record)
        _write_atomic(store_dir() / VERSIONS_FILE,
                      json.dumps(records, ensure_ascii=False, indent=1) + "\n")
        return record


# ---------------------------------------------------------------------------
# Decisions
# ---------------------------------------------------------------------------


def decisions() -> list[dict[str, Any]]:
    """The whole append-only decision log, newest first."""
    path = store_dir() / DECISIONS_FILE
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                logger.warning("skipping an unreadable decision line in %s", path)
                continue
            if isinstance(record, dict):
                out.append(record)
    except OSError as exc:  # pragma: no cover - defensive
        logger.warning("decision log unreadable (%s)", exc)
    return list(reversed(out))


def latest_decision(model_version_id: str, subject: str = "current",
                    candidate_id: Optional[str] = None) -> "dict[str, Any] | None":
    for record in decisions():
        if record.get("model_version_id") != model_version_id:
            continue
        if record.get("subject") != subject:
            continue
        if subject == "candidate" and record.get("candidate_id") != candidate_id:
            continue
        return record
    return None


def check_release_note(text: str) -> None:
    """Refuse a note that carries a gate verdict, a p-value or a coefficient."""
    lowered = str(text or "").lower()
    hits = [word for word in NOTE_FORBIDDEN if word in lowered]
    if hits:
        raise ModelVersionError(NOTE_REFUSAL["he"] + ", ".join(sorted(set(hits))))
    if re.search(r"\bp\s*[=<>]\s*0?\.\d+", lowered):
        raise ModelVersionError(NOTE_REFUSAL["he"] + "p value")


def record_decision(*, model_version: dict[str, Any], decision: str, subject: str,
                    candidate_id: Optional[str], reason: str,
                    release_note_he: str = "", release_note_en: str = "",
                    money_direction: str = "unknown", actor: str = "",
                    evidence: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Append one ship or no-ship verdict, and never move an artifact.

    A ship verdict on a candidate is a decision, not an adoption: the record
    carries ``adoption`` saying the copy has not been performed and why, so a
    later reader cannot mistake an approved candidate for a shipped one.
    """
    if decision not in DECISIONS:
        raise ModelVersionError(f"decision must be one of {DECISIONS}, got {decision!r}")
    if subject not in SUBJECTS:
        raise ModelVersionError(f"subject must be one of {SUBJECTS}, got {subject!r}")
    if subject == "candidate" and not candidate_id:
        raise ModelVersionError("a candidate decision needs the candidate it is about")
    if money_direction not in MONEY_DIRECTIONS:
        raise ModelVersionError(f"money_direction must be one of {MONEY_DIRECTIONS}")
    if not str(reason or "").strip():
        raise ModelVersionError("a decision needs its reason; a verdict with no reason is not a record")
    if decision == "shipped" and not str(release_note_he or "").strip():
        raise ModelVersionError("a ship decision needs a release note in Hebrew for the operator side")
    check_release_note(release_note_he)
    check_release_note(release_note_en)
    version_id = str(model_version.get("id") or "")
    if not version_id:
        raise ModelVersionError("there is no model version on disk to record a decision against")
    observe(model_version)
    record = {
        "decision_id": f"md-{uuid.uuid4().hex[:12]}",
        "recorded_at": _now(),
        "actor": str(actor or "").strip() or "unknown (login is not set up)",
        "model_version_id": version_id,
        "model_version_name": model_version.get("name"),
        "subject": subject,
        "candidate_id": candidate_id if subject == "candidate" else None,
        "decision": decision,
        "reason": str(reason).strip(),
        "release_note_he": str(release_note_he or "").strip(),
        "release_note_en": str(release_note_en or "").strip(),
        "money_direction": money_direction,
        "evidence": evidence or {},
        "adoption": _adoption(decision, subject, evidence or {}),
    }
    with _LOCK:
        path = store_dir() / DECISIONS_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    return record


def _adoption(decision: str, subject: str, evidence: dict[str, Any]) -> dict[str, Any]:
    if subject != "candidate" or decision != "shipped":
        return {"state": "not_applicable"}
    movement = evidence.get("revenue_delta")
    moves = isinstance(movement, (int, float)) and abs(float(movement)) > 0
    return {
        "state": "escalated" if moves else "recorded",
        "performed": False,
        "reason_en": (
            "Copying this candidate over the shipped artifact would move revenue in every plan run afterwards, so the verdict is recorded and the adoption is escalated for owner approval rather than performed here."
            if moves else
            "The verdict is recorded. The measured movement is zero, so nothing would move, and the adoption is still a separate act performed outside this console."),
        "reason_he": (
            "העתקת המועמד הזה על הקובץ המשודר תזיז הכנסה בכל הרצה שאחריה, ולכן ההכרעה נרשמת וההטמעה מועברת לאישור בעלים במקום להתבצע כאן."
            if moves else
            "ההכרעה נרשמת. התנועה הנמדדת היא אפס, ולכן דבר לא יזוז, וההטמעה עדיין פעולה נפרדת המתבצעת מחוץ לקונסולה הזו."),
        "measured_revenue_delta": movement if isinstance(movement, (int, float)) else None,
        "scope": evidence.get("scope"),
    }


# ---------------------------------------------------------------------------
# Candidate money measurements
# ---------------------------------------------------------------------------


def measurements() -> dict[str, Any]:
    records = _read_json(store_dir() / MEASUREMENTS_FILE, {})
    return records if isinstance(records, dict) else {}


def measurement(candidate_id: str) -> "dict[str, Any] | None":
    record = measurements().get(str(candidate_id))
    return record if isinstance(record, dict) else None


def save_measurement(record: dict[str, Any]) -> dict[str, Any]:
    candidate = str(record.get("candidate_id") or "")
    if not candidate:
        raise ModelVersionError("a measurement must name the candidate it measured")
    with _LOCK:
        store = measurements()
        store[candidate] = record
        _write_atomic(store_dir() / MEASUREMENTS_FILE,
                      json.dumps(store, ensure_ascii=False, indent=1) + "\n")
    return record
