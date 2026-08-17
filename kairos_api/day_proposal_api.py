"""Routes for competing day proposals: propose, compare N of them, decide once.

The shape of the surface follows the shape of the argument it hosts. Several
people put a named version of one day on the table; anybody can lay two or more
of them side by side and read what each does to money, to inventory and to the
commitments the channel has signed; one person adopts one of them with a written
reason; the rest stay on the record as rejected alternatives that can still be
opened and read.

Two disciplines this module does not get to opt out of.

**A write that changes the live plan snapshots first.** Adoption replaces rows
that every export and every board downstream reads, so
:func:`kairos_api.day_proposal_rows.publish_day` freezes the plan as it stood
under a named version before a byte moves, and it goes through the shipped-plan
guard so a read-only tree refuses rather than quietly rewrites.

**A decision needs its reason in writing.** Adoption without an annotation is
refused, and so is rejection without one. The annotation is the part of the
record that is still useful in six months, when the numbers are stale and the
question is why.
"""

from __future__ import annotations

import logging
from datetime import date as date_type
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from kairos_api import day_compare
from kairos_api import day_proposal_rows as rows_api
from kairos_api import day_proposal_store as store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/plan/day-proposals", tags=["plan-day-proposals"])


class BreakMove(BaseModel):
    break_id: str
    offset_seconds: Optional[float] = None
    duration_seconds: Optional[float] = None
    is_gold: Optional[bool] = None


class CreateBody(BaseModel):
    day: str
    name: str
    note: str = ""
    moves: list[BreakMove] = []


class CompareBody(BaseModel):
    day: str
    proposal_ids: list[str] = []
    include_live: bool = False


class DecideBody(BaseModel):
    day: str
    verdict: str
    note: str = ""
    allow_stale: bool = False


class RebaseBody(BaseModel):
    day: str
    note: str = ""


def _actor(request: "Request | None") -> str:
    """Who is proposing or deciding, from the one implementation of that question."""
    from kairos_api.affiliation_wall import session_for

    session = session_for(request) or {}
    return str(session.get("username", "") or "")


def _refused(exc: store.ProposalRefused, status_code: int = 409) -> HTTPException:
    """A refusal as an HTTP detail that still carries the operator's own sentence."""
    return HTTPException(status_code=status_code, detail={
        "code": exc.code, "reason": exc.reason, "reason_he": exc.reason_he,
    })


def _baseline_or_404(day: str) -> dict[str, Any]:
    try:
        return rows_api.baseline_for_day(day)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - a data or environment fault is a state
        logger.exception("day proposal baseline failed for %s", day)
        raise HTTPException(status_code=503,
                            detail=f"Could not build this day: {exc}") from None


def trade_context() -> Optional[dict[str, Any]]:
    """The approved agreements and the ledgers their obligations are measured on.

    ``None`` when the trade stores cannot be read at all, which the comparison
    renders as an explicit unmeasured dimension rather than as no commitments.
    """
    try:
        import pandas as pd

        from kairos_api import (agency_conditions, campaigns_api_store,
                                campaigns_delivery, trade_store)

        approved: list[tuple[dict[str, Any], dict[str, Any]]] = []
        for head in trade_store.list_agreements():
            version_id = head.get("current_version_id")
            if not version_id:
                continue
            try:
                approved.append((head, trade_store.load_termset(
                    str(head["agreement_id"]), str(version_id))))
            except (KeyError, OSError):
                continue
        links_path = Path(agency_conditions.LINKS_PATH)
        links = (
            pd.read_csv(links_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
            if links_path.exists() else pd.DataFrame(columns=["agency_id", "advertiser"])
        )
        return {
            "approved": approved,
            "delivery": campaigns_delivery.load_frame(),
            "campaigns": campaigns_api_store.load_frame(),
            "links": links,
            "today": date_type.today(),
        }
    except Exception as exc:  # noqa: BLE001 - an unreadable trade store is a state
        logger.warning("trade context unavailable for a day comparison: %s", exc)
        return None


def _decorated(manifest: dict[str, Any], current_ref: dict[str, Any]) -> dict[str, Any]:
    item = dict(manifest)
    item["staleness"] = store.staleness(manifest, current_ref)
    return item


# ------------------------------------------------------------------- read side

@router.get("")
def list_proposals(day: str = Query("", description="ISO broadcast date")) -> dict[str, Any]:
    """Every competing version on the table for one day, with the day it is against.

    Each proposal carries its staleness against the day as it stands right now,
    so a version authored before the day moved cannot read as current merely
    because its own figures are internally consistent.
    """
    wanted = str(day or "").strip()
    if not wanted:
        from kairos_api import break_store

        available = break_store.plan_days()
        if not available:
            return {"available": False,
                    "reason": "no operator plan covers any day yet",
                    "reason_he": "אין תוכנית מפעיל שמכסה יום כלשהו",
                    "proposals": []}
        wanted = available[0]
    baseline = _baseline_or_404(wanted)
    manifests = store.list_for_day(baseline["channel"], baseline["day"])
    counts: dict[str, int] = {}
    for manifest in manifests:
        status = str(manifest.get("status") or store.PROPOSED)
        counts[status] = counts.get(status, 0) + 1
    return {
        "available": True,
        "channel": baseline["channel"],
        "day": baseline["day"],
        "proposals": [_decorated(item, baseline["ref"]) for item in manifests],
        "count": len(manifests),
        "status_counts": counts,
        "adopted": (store.adopted_for_day(baseline["channel"], baseline["day"]) or {}).get("proposal_id"),
        "baseline": {
            "ref": baseline["ref"],
            "engine": baseline["engine"],
            "caps": baseline["caps"],
        },
    }


@router.get("/history")
def history(day: str = Query(..., description="ISO broadcast date")) -> dict[str, Any]:
    """Every proposal this day ever carried, rejected ones included, with lineage.

    Nothing is deleted on a decision, so this is the record of the argument: who
    proposed what, what each version was worth, who decided, what they wrote, and
    which proposal superseded the ones that lost.
    """
    baseline = _baseline_or_404(day)
    manifests = store.list_for_day(baseline["channel"], baseline["day"])
    entries = []
    for manifest in manifests:
        summary = (manifest.get("summary") or {}).get("owned") or {}
        entries.append({
            "proposal_id": manifest.get("proposal_id"),
            "name": manifest.get("name"),
            "note": manifest.get("note"),
            "author": manifest.get("author"),
            "created_at": manifest.get("created_at"),
            "seq": manifest.get("seq"),
            "status": manifest.get("status"),
            "decision": manifest.get("decision"),
            "lineage": manifest.get("lineage"),
            "edit_count": manifest.get("edit_count"),
            "rows_source": manifest.get("rows_source"),
            "revenue": summary.get("revenue"),
            "breaks": summary.get("breaks"),
            "ad_seconds": summary.get("ad_seconds"),
            "scope": (manifest.get("summary") or {}).get("scope"),
            "staleness": store.staleness(manifest, baseline["ref"]),
        })
    return {
        "available": True,
        "channel": baseline["channel"],
        "day": baseline["day"],
        "entries": entries,
        "count": len(entries),
        "note_he": "הצעה שנדחתה נשמרת במלואה וניתנת לקריאה; דחייה אינה מחיקה",
    }


@router.get("/{proposal_id}")
def get_proposal(proposal_id: str,
                 day: str = Query(..., description="ISO broadcast date")) -> dict[str, Any]:
    baseline = _baseline_or_404(day)
    manifest = store.get(baseline["channel"], baseline["day"], proposal_id)
    if manifest is None:
        raise HTTPException(status_code=404, detail=f"no proposal {proposal_id} for this day")
    return {
        "available": True,
        "proposal": _decorated(manifest, baseline["ref"]),
        "edits": store.edits_for(baseline["channel"], baseline["day"], proposal_id),
        "baseline": {"ref": baseline["ref"], "caps": baseline["caps"]},
    }


# ------------------------------------------------------------------ write side

@router.post("", status_code=201)
def create_proposal(body: CreateBody, request: Request = None) -> dict[str, Any]:
    """Put a named version of this day on the table, priced by the engine itself.

    With no moves the proposal is the day as the engine plans it now, which is
    how a person opens an alternative from the shared starting point. With moves
    the day is planned again with those edits pinned and nothing written, so the
    money the proposal carries is the money the engine would really produce,
    including how it re-arranges everything around the edit.
    """
    moves = [move.model_dump() for move in body.moves]
    try:
        built = rows_api.proposal_rows(body.day, moves)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from None
    edit_map = {str(move["break_id"]): {
        key: value for key, value in move.items() if key != "break_id" and value is not None
    } for move in moves}
    try:
        manifest = store.create_proposal(
            channel=built["channel"], date=built["day"], name=body.name,
            author=_actor(request), rows=built["rows"], baseline_ref=built["baseline_ref"],
            edits=edit_map, note=body.note, rows_source=built["rows_source"],
            engine=built["engine"],
        )
    except store.ProposalRefused as exc:
        raise _refused(exc, status_code=422) from None
    return {"ok": True, "proposal": manifest,
            "engine_ms": (built["engine"] or {}).get("engine_ms")}


@router.post("/compare")
def compare(body: CompareBody) -> dict[str, Any]:
    """Two or more competing versions of one day, with the reasoning for each delta.

    The comparison is against the day as it stands right now, which is also the
    day the proposals recorded as their baseline. Money, the attributed
    explanation of every agora of it, the inventory consequence, the effect on
    signed commitments and the engine's guardrail verdict all arrive per side,
    each rolled into one line a decision-maker reads first.
    """
    baseline = _baseline_or_404(body.day)
    live_rows = rows_api.live_day_rows(baseline["channel"], baseline["day"])
    return day_compare.compare(
        baseline["channel"], baseline["day"], list(body.proposal_ids),
        include_live=body.include_live,
        baseline_rows=baseline["rows"], baseline_ref=baseline["ref"],
        live_rows=live_rows, live_basis=rows_api.COMMITTED_BASIS,
        caps=baseline["caps"], trade_context=trade_context(),
    )


@router.post("/{proposal_id}/decide")
def decide(proposal_id: str, body: DecideBody, request: Request = None) -> dict[str, Any]:
    """Settle the day: adopt this version with a reason, or reject it with one.

    Adoption publishes the proposal's frozen rows into the plan of record and
    closes every still-open rival as rejected, naming this proposal in their
    lineage. Both verdicts require the annotation, because a decision whose
    reason was never written down cannot be reviewed later.
    """
    verdict = str(body.verdict or "").strip().lower()
    if verdict not in {"adopt", "reject"}:
        raise HTTPException(status_code=422, detail={
            "code": "bad_verdict", "reason": "verdict must be adopt or reject",
            "reason_he": "ההחלטה יכולה להיות אימוץ או דחייה",
        })
    annotation = str(body.note or "").strip()
    if not annotation:
        raise HTTPException(status_code=422, detail={
            "code": "no_annotation",
            "reason": "a decision on a day needs its reason in writing",
            "reason_he": "החלטה על יום דורשת נימוק בכתב",
        })
    baseline = _baseline_or_404(body.day)
    channel, day = baseline["channel"], baseline["day"]
    actor = _actor(request)

    if verdict == "reject":
        try:
            manifest = store.update_status(channel, day, proposal_id, store.REJECTED,
                                           actor=actor, note=annotation)
        except store.ProposalRefused as exc:
            raise _refused(exc) from None
        return {"ok": True, "verdict": "rejected", "proposal": manifest}

    try:
        manifest = store.check_adoptable(channel, day, proposal_id,
                                        current_ref=baseline["ref"],
                                        allow_stale=bool(body.allow_stale))
    except store.ProposalRefused as exc:
        raise _refused(exc) from None
    rows = store.rows_for(channel, day, proposal_id)
    if rows is None or not len(rows):
        raise HTTPException(status_code=409, detail={
            "code": "no_frozen_file",
            "reason": "this proposal has no frozen day file, so there is nothing to publish",
            "reason_he": "להצעה הזו אין קובץ יום שמור, ולכן אין מה לפרסם",
        })
    try:
        published = rows_api.publish_day(channel, day, rows, actor=actor,
                                         proposal_name=str(manifest.get("name") or proposal_id))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail={
            "code": "no_live_plan",
            "reason": f"there is no saved plan at {exc} to publish into",
            "reason_he": "אין תוכנית שמורה שאליה ניתן לפרסם את היום",
        }) from None
    except Exception as exc:  # noqa: BLE001 - the read-only wall is one of these
        logger.exception("publishing an adopted day failed for %s %s", channel, day)
        raise HTTPException(status_code=409, detail={
            "code": "publish_refused", "reason": str(exc),
            "reason_he": "פרסום היום לתוכנית נדחה; דבר לא נכתב",
        }) from None
    adopted = store.update_status(channel, day, proposal_id, store.ADOPTED, actor=actor,
                                 note=annotation, current_ref=baseline["ref"],
                                 allow_stale=bool(body.allow_stale))
    rejected = store.reject_rivals(
        channel, day, proposal_id, actor=actor,
        note=f"superseded by the adopted proposal {adopted.get('name')!r}",
    )
    return {
        "ok": True,
        "verdict": "adopted",
        "proposal": adopted,
        "published": published,
        "rejected": [{"proposal_id": item.get("proposal_id"), "name": item.get("name"),
                      "lineage": item.get("lineage")} for item in rejected],
        "note_he": "הגרסאות שנדחו נשמרות במלואן וניתנות לקריאה בהיסטוריית היום",
    }


@router.post("/{proposal_id}/rebase")
def rebase(proposal_id: str, body: RebaseBody, request: Request = None) -> dict[str, Any]:
    """Say on the record that this version still stands against the day as it is now."""
    baseline = _baseline_or_404(body.day)
    try:
        manifest = store.rebase(baseline["channel"], baseline["day"], proposal_id,
                                actor=_actor(request), new_ref=baseline["ref"],
                                note=body.note)
    except store.ProposalRefused as exc:
        raise _refused(exc) from None
    return {"ok": True, "proposal": _decorated(manifest, baseline["ref"])}


@router.post("/{proposal_id}/withdraw")
def withdraw(proposal_id: str, body: RebaseBody, request: Request = None) -> dict[str, Any]:
    """The author taking their own version off the table. It stays readable."""
    baseline = _baseline_or_404(body.day)
    try:
        manifest = store.withdraw(baseline["channel"], baseline["day"], proposal_id,
                                  actor=_actor(request), note=body.note)
    except store.ProposalRefused as exc:
        raise _refused(exc) from None
    return {"ok": True, "proposal": manifest}
