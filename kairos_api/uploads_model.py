"""The model version the plan's numbers rest on, split out of ``uploads_status``.

Split under the file-size cap and named by the ``<parent stem>_<role>.py`` rule
the package already follows. The status module was one line under the cap, which
is exactly where a law stops holding by itself, and this block is a different
subject from the state of an input: it is the only place in this destination
that reads anything under ``models/``.

The operator needs exactly two model facts and this is where the second one is
decided: which version the numbers rest on, and whether the sources it was
measured on still match what is on disk. The verdict is the engine's own
tri-state ``fresh`` / ``stale`` / ``unknown`` from :mod:`kairos.model.freshness`,
which is read and never written, and which never invents a fresh. No gate
verdict, coefficient value, coverage figure or p-value crosses over: that is the
company side's and it lives in the model console.

Training is a company act, so the remedy names its owner and carries no verb the
operator cannot perform.

Both engine imports stay inside the functions that use them. Measured: the
import chain costs 7.564 s in a cold interpreter, so hoisting it would move that
bill to every server start, including the ones that never open this endpoint.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from kairos_api import read_cache

logger = logging.getLogger(__name__)

# Keyed on the artifact's signature AND on the signatures of the sources it was
# measured on, so a changed source is a changed key and a stale verdict cannot
# be served from the cache.
MODEL_NAMESPACE = "uploads.model_version"

# The remedy for a model version whose sources have moved on. Training is a
# company act, so the operator's control is a request with a named owner and
# never a verb they cannot perform.
STALE_NOTE = {
    "en": "The model needs training when new data lands. Training is the company team's act, not an operator action.",
    "he": "המודל דורש אימון כשנקלטים נתונים חדשים. האימון הוא פעולה של צוות החברה, לא פעולה של המפעיל.",
}
FRESH_NOTE = {
    "en": "Every source the model version was measured on still matches the file on disk.",
    "he": "כל מקור שגרסת המודל נמדדה עליו עדיין תואם לקובץ שעל הדיסק.",
}
UNKNOWN_NOTE = {
    "en": "The model version cannot be checked against its sources, so its state is unknown rather than assumed current.",
    "he": "לא ניתן לבדוק את גרסת המודל מול המקורות שלה, ולכן מצבה לא ידוע ולא מונח שהיא עדכנית.",
}


def version(models_dir: Path, root: Path) -> dict[str, Any]:
    """The model version the plan's numbers rest on, and its tri-state state.

    ``status`` is the engine's own ``fresh`` / ``stale`` / ``unknown`` verdict,
    which never invents a fresh. ``measured_on`` names the source files the
    version was measured against, so the operator can see whether the file they
    are about to replace is one of them.
    """
    artifact = Path(models_dir) / "tv_break_coefficients.json"
    held = read_cache.cached(
        MODEL_NAMESPACE,
        str(artifact),
        (read_cache.file_signature(artifact), _fingerprint_signature(artifact, root)),
        lambda: _build(artifact, root),
    )
    # The cache shares values rather than copying them, so a caller that puts
    # this block in a payload gets its own copy and cannot edit the cache.
    return dict(held)


def _fingerprint_signature(artifact: Path, root: Path) -> tuple[tuple[str, int, int], ...]:
    """Signatures of the sources the version was measured on, for the cache key."""
    try:
        return read_cache.file_signatures(Path(root) / name for name in measured_on(artifact))
    except Exception:  # pragma: no cover - a missing artifact is handled below
        return ()


def measured_on(artifact: Path) -> list[str]:
    """The source files this version was measured against, by name."""
    from kairos.model.measure import read_coefficients_metadata

    metadata = read_coefficients_metadata(artifact)
    sources = metadata.get("source_fingerprints")
    return sorted(str(name) for name in sources) if isinstance(sources, dict) else []


def _build(artifact: Path, root: Path) -> dict[str, Any]:
    from kairos.model.freshness import coefficient_freshness
    from kairos.model.measure import read_coefficients_metadata

    unavailable = {
        "available": False,
        "version": None,
        "trained_at": None,
        "status": "unknown",
        "changed_sources": [],
        "measured_on": [],
        "note_en": UNKNOWN_NOTE["en"],
        "note_he": UNKNOWN_NOTE["he"],
    }
    try:
        metadata = read_coefficients_metadata(artifact)
    except Exception:
        logger.exception("the model version block could not read its metadata")
        return unavailable
    if not metadata:
        return unavailable
    try:
        verdict = coefficient_freshness(metadata, root=Path(root))
    except Exception:
        logger.exception("the model version freshness check failed")
        return unavailable
    status = str(verdict.get("status") or "unknown")
    trained_at = verdict.get("computed_at")
    note = FRESH_NOTE if status == "fresh" else STALE_NOTE if status == "stale" else UNKNOWN_NOTE
    return {
        "available": True,
        "version": _version_name(trained_at),
        "trained_at": trained_at,
        "status": status,
        "changed_sources": [str(name) for name in verdict.get("changed_files") or []],
        "measured_on": measured_on(artifact),
        "note_en": note["en"],
        "note_he": note["he"],
    }


def _version_name(trained_at: Any) -> str | None:
    """The version's name, which is the calendar day it was trained on."""
    text = str(trained_at or "").strip()
    if not text:
        return None
    return text[:10]
