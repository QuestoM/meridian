"""The console's composed blocks, split out of the router under the size cap.

Nothing here is a route. These are the four blocks more than one route needs:
the version identity as the console reports it, the drift block, the read-only
mirror of the activation switch, and the evidence a decision freezes into
itself. Keeping them out of the router leaves that module a list of routes,
which is what a reader wants it to be.
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Optional

from kairos_api import model_console_artifacts as artifacts
from kairos_api import model_console_candidates as candidates
from kairos_api import model_console_gates as gates
from kairos_api import model_version_store as store

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_BOARD = ROOT / "tv-break-dashboard" / "src" / "model" / "candidates" / "candidate-board.json"


def current_version() -> dict[str, Any]:
    """The version on disk, plus whether the store has recorded it.

    **A read never writes.** Recording is an act with its own route, and a
    version the store has not seen renders as a control rather than as a fact,
    so no GET on this surface can produce a change in a tracked file.
    """
    version = artifacts.model_version()
    version_id = str(version.get("id") or "")
    recorded = (any(record.get("id") == version_id for record in store.versions())
                if version_id else False)
    return {**version, "recorded": recorded}


def drift_block() -> dict[str, Any]:
    """The level-drift measurement, or an honest unavailable with its reason."""
    metadata = artifacts.retention_metadata()
    block = metadata.get("level_drift")
    if isinstance(block, dict) and block:
        return block
    return {
        "status": "unavailable",
        "reason": "the coefficients artifact carries no level-drift measurement; training the model computes it",
    }


def activation_mirror() -> dict[str, Any]:
    """The activation switch as state, never as a control.

    Read from the store that owns it, so the console and the Rules surface
    cannot disagree, and stripped of ``can_edit`` on purpose: the console owns
    no control over it and must not render one. Throwing this switch changes a
    run, and a run-side act on the training side is the blur the whole
    training-versus-runs rule exists to prevent.
    """
    try:
        from kairos_api import model_activation

        payload = model_activation.payload(None)
    except Exception as exc:  # pragma: no cover - defensive, a read must not 500
        logger.warning("activation state unavailable (%s)", exc)
        return {"available": False,
                "reason_en": "The activation state could not be read in this build.",
                "reason_he": "לא ניתן היה לקרוא את מצב ההפעלה בגרסה הזו."}
    return {
        "available": True,
        "field": payload.get("field"),
        "active": payload.get("active"),
        "state": payload.get("state"),
        "computed_at": payload.get("computed_at"),
        "control_lives_on": "rules",
        "note_en": "Whether runs consume the audience model is a run-side change, so its control lives on Rules.",
        "note_he": "האם ההרצות צורכות את מודל הקהל הוא שינוי בצד ההרצה, ולכן הפקד שלו נמצא בכללים.",
    }


def past_durations() -> list[float]:
    """How long past money measurements actually took, in this store."""
    return sorted(
        round(float(record["duration_seconds"]), 1)
        for record in store.measurements().values()
        if isinstance(record, dict) and isinstance(record.get("duration_seconds"), (int, float))
    )


def shipped_training_commands() -> list[dict[str, Any]]:
    """The commands that overwrite the shipped artifacts, and what each moves.

    The console runs neither. Both write under ``models/`` and the retention
    artifact is read by every plan run, so running one here would move revenue
    with no approval. The console's own training runs write into the releases
    store instead, which is what makes them safe to start from a button.
    """
    return [
        {
            "artifact": "retention",
            "command": "PYTHONUTF8=1 python scripts/compute_measured_coefficients.py",
            "writes": "models/tv_break_coefficients.json",
            "consequence_en": "Every plan run afterwards prices retention with the new coefficients.",
            "consequence_he": "כל הרצת תוכנית שאחריה מתמחרת שימור עם המקדמים החדשים.",
        },
        {
            "artifact": "audience",
            "command": "PYTHONUTF8=1 python scripts/compute_audience_model.py",
            "writes": "models/audience_model.json",
            "consequence_en": "Forward-dated ratings change source, but only while the activation switch is on.",
            "consequence_he": "מקור הרייטינג לתאריכים עתידיים משתנה, אך רק כשמתג ההפעלה דלוק.",
        },
    ]


def decision_evidence(subject: str, candidate_id: Optional[str]) -> dict[str, Any]:
    """The measurement that existed when a verdict was taken, frozen into it.

    A verdict recorded without its evidence is an opinion. This is what makes a
    later reader able to see not just what was decided but what it was decided
    on, including the honest case where the money was never measured.
    """
    ledger = gates.ledger()
    evidence: dict[str, Any] = {"gate_counts": ledger["counts"], "gate_total": ledger["total"]}
    if subject != "candidate" or not candidate_id:
        return evidence
    record = store.measurement(candidate_id)
    path = candidates.candidate_path(candidate_id)
    if record is None or path is None:
        evidence["money_state"] = "not_measured"
        return evidence
    current = candidates.measurement_fingerprint(path)
    evidence["money_state"] = "measured" if record.get("fingerprint") == current else "stale"
    delta = record.get("operator_channel_delta") or {}
    evidence["revenue_delta"] = delta.get("revenue_delta")
    evidence["revenue_delta_pct"] = delta.get("revenue_delta_pct")
    evidence["scope"] = (record.get("scope") or {}).get("operator_channel")
    evidence["measured_at"] = record.get("measured_at")
    return evidence


def complete_decision_evidence(subject: str, candidate_id: Optional[str]) -> dict[str, Any]:
    """Freeze the published common-basis snapshot into a browser verdict.

    The candidate board is P12's read-only publication of the measurement. It
    carries the same re-score, gates, baselines, fit limit and coefficient
    movement as the terminal, but no route into the training act. Reading that
    artifact here keeps the console rich without making adoption reachable
    from the application package.
    """
    if subject != "candidate" or not candidate_id:
        return decision_evidence(subject, candidate_id)
    evidence = decision_evidence(subject, candidate_id)
    try:
        board = json.loads(CANDIDATE_BOARD.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        evidence["comparison_state"] = "unavailable"
        evidence["comparison_reason"] = f"published candidate measurement unavailable: {exc}"
        return evidence
    row = next((item for item in board.get("candidates") or []
                if item.get("id") == candidate_id), None)
    if row is None:
        evidence["comparison_state"] = "unavailable"
        evidence["comparison_reason"] = "candidate is absent from the published common-basis measurement"
        return evidence
    evidence.update({
        "basis_en": "This verdict uses the published common-basis comparison.",
        "basis_he": "ההכרעה הזו משתמשת בהשוואה המפורסמת על בסיס משותף.",
        "rescore": {
            "rmse": row.get("rmse"),
            "shipped_rmse": (board.get("shipped") or {}).get("rmse"),
            "rmse_delta": row.get("rmse_delta"),
            "paired_statistic": row.get("paired_statistic"),
            "paired_bar": row.get("paired_bar"),
            "fold_dispersion": row.get("fold_dispersion"),
            "breaks_improved": row.get("breaks_improved"),
            "breaks_worsened": row.get("breaks_worsened"),
            "state": row.get("verdict"),
            "en": row.get("verdict_en"),
            "he": row.get("verdict_he"),
            "rule_en": row.get("rule_en"),
            "rule_he": row.get("rule_he"),
            "duplicate_of": row.get("duplicate_of") or [],
            "breaks_fitted_on": row.get("breaks_fitted_on"),
            "self_reported": row.get("self_reported") or {},
            "cells": row.get("cells") or {},
            "sha256": row.get("sha256"),
            "file": row.get("file"),
            "measured_at": board.get("measured_at"),
            "fingerprint": board.get("fingerprint"),
        },
        "gates": row.get("gates") or {},
        "evaluation": board.get("evaluation"),
        "limit": board.get("limit"),
        "fit_basis": board.get("fit_basis"),
        "baselines": board.get("baselines") or [],
        "cell_structure": board.get("cell_structure"),
    })
    return evidence
