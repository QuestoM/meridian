"""The decision plane behind Today's priority decisions: the log and the shortcut.

Split from :mod:`kairos_api.overview_api` under the 450-line law, same owner. Both
routes are driven only by the recommendation list in the overview payload, so
they belong to Today rather than to the day board.

Moved verbatim from dashboard_api.py. Approve and reject still persist a REAL
override through the override store's own public entry point, never a
display-only log row, so the log and the plan cannot drift.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()


class BreakDecisionRequest(BaseModel):
    """Operator decision captured from the dashboard command surface."""

    action: Literal["approve", "reject", "apply_similar"]
    recommendation_id: str | None = Field(default=None)
    break_id: str | None = Field(default=None)
    program_type: str | None = Field(default=None)
    scenario: str | None = Field(default=None)
    note: str | None = Field(default=None, max_length=500)
    # Fields that let an approve/reject resolve into a REAL override. target_id is the
    # owned-channel segment_id (falls back to break_id); kind is the override kind
    # (pin/force/forbid/gold); the anchor trio is copied from the recommendation so a
    # later re-ingest cannot silently rebind the override to a different break.
    target_id: str | None = Field(default=None)
    kind: str | None = Field(default=None)
    value: str | None = Field(default=None)
    gold: bool = Field(default=False)
    anchor_date: str | None = Field(default=None)
    anchor_start: str | None = Field(default=None)
    anchor_title: str | None = Field(default=None)


def _decision_log() -> list[dict[str, Any]]:
    """The operator's decision log, read from the REAL override store.

    Retires the old data/kairos_decisions.json, which was written on every
    approve/reject but read only for display, so the log and the plan could drift.
    Now every row here is a persisted Override that actually resolves through the
    decision plane: an approved recommendation (source=recommendation) or a dismissed
    rejection (status=dismissed, recorded but never applied).
    """
    from kairos.optimize.overrides import (
        OverrideSet,
        SOURCE_RECOMMENDATION,
        STATUS_DISMISSED,
    )

    records: list[dict[str, Any]] = []
    for override in OverrideSet.from_csv().overrides:
        if override.source != SOURCE_RECOMMENDATION and override.status != STATUS_DISMISSED:
            continue
        records.append({
            "id": override.override_id,
            "action": "reject" if override.status == STATUS_DISMISSED else "approve",
            "recommendation_id": override.rec_id or None,
            "break_id": override.target_id,
            "kind": override.kind,
            "value": override.value,
            "status": override.status,
            "note": override.notes,
            "created_at": override.created_at,
            "source": override.source,
            "anchor": {
                "date": override.anchor_date,
                "start_clock": override.anchor_start,
                "program": override.anchor_title,
            },
        })
    records.sort(key=lambda record: str(record.get("created_at") or ""), reverse=True)
    return records


def _resolve_decision(request: BreakDecisionRequest) -> dict[str, Any]:
    """Turn an approve/reject decision into a REAL override (no dead log write).

    Approve creates an active segment override stamped source=recommendation with the
    rec_id and the semantic anchor, so the anchor guard protects it on re-ingest.
    Reject creates a dismissed record (forbid by default) that the plan never applies
    because only active overrides bend the schedule. The console can equivalently POST
    /api/overrides directly; this shortcut just routes through the same honest store.
    """
    from kairos.optimize.overrides import (
        FORBID,
        SOURCE_RECOMMENDATION,
        STATUS_ACTIVE,
        STATUS_DISMISSED,
    )
    from kairos_api.overrides import OverrideCreate, create_override

    target = str(request.target_id or request.break_id or "").strip()
    if not target:
        raise HTTPException(
            status_code=400,
            detail="a target segment_id (target_id or break_id) is required to resolve a decision into an override",
        )
    reject = request.action == "reject"
    kind = str(request.kind or "").strip().lower() or (FORBID if reject else "")
    if not kind:
        raise HTTPException(
            status_code=400,
            detail="kind is required to approve a decision (pin, force, forbid, or gold)",
        )
    payload = OverrideCreate(
        scope="segment",
        target_id=target,
        kind=kind,
        value=str(request.value or ""),
        gold=bool(request.gold),
        notes=str(request.note or ""),
        source=SOURCE_RECOMMENDATION,
        rec_id=str(request.recommendation_id or "").strip(),
        status=STATUS_DISMISSED if reject else STATUS_ACTIVE,
        anchor_date=str(request.anchor_date or "").strip(),
        anchor_start=str(request.anchor_start or "").strip(),
        anchor_title=str(request.anchor_title or "").strip(),
    )
    return create_override(payload)


@router.get("/api/break-decisions", tags=["dashboard"])
def break_decisions() -> dict[str, Any]:
    # Display is driven by the real override store, not a parallel decision-log file.
    return {"decisions": _decision_log()}


@router.post("/api/break-decisions", tags=["dashboard"])
def create_break_decision(request: BreakDecisionRequest) -> dict[str, Any]:
    # Approve/Reject shortcut: persists a real Override (source=recommendation, rec_id,
    # anchor) rather than a display-only log entry.
    return {"decision": _resolve_decision(request)}
