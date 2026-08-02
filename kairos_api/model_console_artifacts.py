"""Model version identity, read from the artifacts the runs actually consume.

Before this module the only identity a trained model had was a ``computed_at``
timestamp inside a file that every training overwrites in place, so two
different models could carry the same name, a verdict could not be recorded
against anything, and no reader could tell which model a plan was computed
with.

Identity here is the content, never a label anybody types. A **model version**
is the pair of artifacts in force together, because a plan consumes both: the
retention coefficients that price the cost of a break, and the audience model
that predicts a forward-dated rating. Its id is a digest of both files' bytes,
its name is the date of the later of the two, and both are reproducible from
the files alone. Two identical trees produce the same id and a single changed
byte produces a different one.

Everything is a tolerant read. An absent artifact is an honest absence with the
path that would hold it, never a fabricated version.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"

RETENTION_ARTIFACT = "tv_break_coefficients.json"
AUDIENCE_ARTIFACT = "audience_model.json"

# What each artifact is for, in the plain language the console prints. The
# distinction matters: they are two different models and a reader who fuses
# them will read a rating gate as a retention gate.
ARTIFACT_SUBJECTS = {
    "retention": {
        "en": "Predicted retention: what a break costs in audience.",
        "he": "שימור חזוי: כמה ברייק עולה בקהל.",
    },
    "audience": {
        "en": "Expected rating: what a segment is predicted to draw.",
        "he": "רייטינג צפוי: כמה מקטע צפוי למשוך.",
    },
}


def _sha256(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def read_artifact(name: str) -> "dict[str, Any] | None":
    """The parsed artifact, or None when absent or unreadable (logged)."""
    path = MODELS_DIR / name
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("model artifact %s is unreadable (%s); treating as absent", path, exc)
        return None
    return payload if isinstance(payload, dict) else None


def retention_metadata() -> dict[str, Any]:
    """The retention artifact's metadata block, or an empty dict."""
    payload = read_artifact(RETENTION_ARTIFACT) or {}
    metadata = payload.get("metadata")
    return metadata if isinstance(metadata, dict) else {}


def audience_gates() -> dict[str, Any]:
    """The audience artifact's gate block, or an empty dict."""
    payload = read_artifact(AUDIENCE_ARTIFACT) or {}
    gates = payload.get("gates")
    return gates if isinstance(gates, dict) else {}


def _artifact_identity(kind: str, name: str, computed_at_of) -> dict[str, Any]:
    path = MODELS_DIR / name
    if not path.is_file():
        return {
            "kind": kind,
            "present": False,
            "path": path.relative_to(ROOT).as_posix(),
            "reason_en": f"No trained artifact on disk at {path.relative_to(ROOT).as_posix()}.",
            "reason_he": f"אין קובץ מאומן בדיסק בנתיב {path.relative_to(ROOT).as_posix()}.",
            "subject_en": ARTIFACT_SUBJECTS[kind]["en"],
            "subject_he": ARTIFACT_SUBJECTS[kind]["he"],
        }
    payload = read_artifact(name) or {}
    digest = _sha256(path) or ""
    stat = path.stat()
    return {
        "kind": kind,
        "present": True,
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": digest,
        "short": digest[:8],
        "bytes": stat.st_size,
        "computed_at": computed_at_of(payload),
        "subject_en": ARTIFACT_SUBJECTS[kind]["en"],
        "subject_he": ARTIFACT_SUBJECTS[kind]["he"],
        "source_fingerprints": _source_fingerprints(payload),
    }


def _source_fingerprints(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    prints = payload.get("source_fingerprints") or metadata.get("source_fingerprints")
    return prints if isinstance(prints, dict) else {}


def _retention_computed_at(payload: dict[str, Any]) -> Optional[str]:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    value = metadata.get("computed_at")
    return str(value) if value else None


def _audience_computed_at(payload: dict[str, Any]) -> Optional[str]:
    value = payload.get("computed_at")
    return str(value) if value else None


def artifacts() -> dict[str, dict[str, Any]]:
    """Both artifacts' identity, present or honestly absent."""
    return {
        "retention": _artifact_identity("retention", RETENTION_ARTIFACT, _retention_computed_at),
        "audience": _artifact_identity("audience", AUDIENCE_ARTIFACT, _audience_computed_at),
    }


def _day_of(value: Optional[str]) -> str:
    text = str(value or "").strip()
    return text.split("T")[0].split(" ")[0] if text else ""


def model_version(records: Optional[dict[str, dict[str, Any]]] = None) -> dict[str, Any]:
    """The version identity of the pair of artifacts in force.

    ``id`` is a digest of both files' content, so it is reproducible from the
    tree and cannot be typed wrongly. ``name`` is the later of the two training
    dates, which is what a person says out loud. When neither artifact is
    present there is no version, and the payload says so rather than inventing
    one.
    """
    found = artifacts() if records is None else records
    present = [record for record in found.values() if record.get("present")]
    if not present:
        return {
            "available": False,
            "id": None,
            "name": None,
            "reason_en": "No trained model artifact is on disk, so there is no model version to name.",
            "reason_he": "אין בדיסק אף קובץ מודל מאומן, ולכן אין גרסת מודל לציין.",
            "artifacts": found,
        }
    digest = hashlib.sha256()
    for kind in ("retention", "audience"):
        record = found[kind]
        digest.update(kind.encode("utf-8"))
        digest.update(b"=")
        digest.update(str(record.get("sha256") or "absent").encode("utf-8"))
        digest.update(b";")
    days = sorted(day for day in (_day_of(r.get("computed_at")) for r in present) if day)
    name = days[-1] if days else "undated"
    short = digest.hexdigest()[:8]
    return {
        "available": True,
        "id": f"mv-{name}-{short}",
        "name": name,
        "short": short,
        "trained_at": {
            "retention": found["retention"].get("computed_at"),
            "audience": found["audience"].get("computed_at"),
        },
        "artifacts": found,
        "basis_en": "The identity is a digest of both artifacts' bytes, so it is reproducible from the tree.",
        "basis_he": "הזהות היא טביעת אצבע של הבייטים של שני הקבצים, ולכן היא ניתנת לשחזור מהעץ.",
    }


def gate_override_flags() -> list[dict[str, Any]]:
    """The flags that can force a gate, read from the training script itself.

    A forced gate and a self-activated one are indistinguishable in the
    artifact after the fact, which is the provenance hole section 4.4 names.
    The list is parsed from the script's own ``add_argument`` calls so it cannot
    drift from what the script accepts, and ``--output`` is excluded because it
    is an output path and not a gate override.
    """
    script = ROOT / "scripts" / "compute_measured_coefficients.py"
    if not script.is_file():
        return []
    import re

    text = script.read_text(encoding="utf-8")
    rows: list[dict[str, Any]] = []
    # Each flag's environment twin is named inside that flag's own help text, so
    # it is read from the same block rather than derived from the flag's
    # spelling: --series is KAIROS_SERIES_LAYER, which no derivation produces.
    for block in text.split("add_argument(")[1:]:
        match = re.match(r'\s*"(--[a-z-]+)"', block)
        if match is None or match.group(1) == "--output":
            continue
        twin = re.search(r"(KAIROS_[A-Z_]+)=force-on/force-off", block)
        rows.append({
            "flag": match.group(1),
            "env": twin.group(1) if twin else None,
            "recorded_in_artifact": False,
        })
    return rows
