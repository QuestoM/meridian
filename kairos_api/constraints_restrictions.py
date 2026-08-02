"""Restrictions: the programming side's own door onto the constraint store.

Everything here is a thin, honest layer over machinery that already exists. The
predicate is the frozen contract, the matcher is the engine's own, the money is
the commit path's own optimizer or the frozen scoring seam, and the rows written
are ordinary constraint rows that the resolver reads exactly as it always did. A
restriction is not a second engine; it is a sentence that compiles to the first
one, plus the four things the store never carried: who wrote it, why, when it
starts and when it stops.

The preview is the point of the surface. A restriction is a decision about
somebody else's revenue, so the cost is on screen before the save, never after,
and it is reported on two bases that are named rather than blended:

- **Scored.** The exact revenue and retention of the breaks this restriction
  removes, at the counts it sets, through :mod:`kairos.optimize.evaluate`. It is
  microseconds, it covers every affected day, and it does not let the optimizer
  move a break somewhere else.
- **Re-allocated.** The commit path's own optimizer run twice on the affected
  days, which is the plan the save would produce. It costs about a second a day,
  so it runs when the restriction touches a small number of days and says so
  plainly when it does not.

Neither is presented as the other and neither is summed into the other.
"""

from __future__ import annotations

import logging
import uuid
from datetime import date, datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos_api import constraints_airings as airings_lib
from kairos_api import constraints_cost
from kairos_api import constraints_sentence as sentence_lib
from kairos_api.constraints_language import (
    AUTHORING_COLUMNS,
    KINDS,
    PER_AIRING_KINDS,
    CompiledRow,
    RestrictionError,
    compile_restriction,
    params_cell,
    parse_params,
)

logger = logging.getLogger(__name__)

router = APIRouter()


class RestrictionDraft(BaseModel):
    """A restriction as its author states it, before anything is compiled."""

    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    where: Optional[dict[str, Any]] = None
    starts_on: str = ""
    expires_on: str = ""
    author: str = ""
    reason: str = ""


def _iso(value: str, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"{label} must be an ISO date, got {text!r}.") from exc


def _validated(draft: RestrictionDraft) -> tuple[Optional[dict[str, Any]], str, str]:
    from kairos_api._constraint_options import validate_where

    if draft.kind not in KINDS:
        raise HTTPException(status_code=400, detail=f"kind must be one of {sorted(KINDS)}")
    where = validate_where(draft.where)
    starts_on = _iso(draft.starts_on, "starts_on")
    expires_on = _iso(draft.expires_on, "expires_on")
    if starts_on and expires_on and expires_on <= starts_on:
        raise HTTPException(status_code=400, detail="The end date has to fall after the start date.")
    return where, starts_on, expires_on


def _compile(draft: RestrictionDraft) -> tuple[list[CompiledRow], list[Any], Optional[dict[str, Any]], str, str]:
    # Every kind is matched against the plan of record, not only the two that
    # need the match in order to compile. A scope-level rule compiles to one row
    # whatever the plan holds, but it still lands on a knowable set of airings,
    # and a composer that reports zero of them is telling its author that a rule
    # binding forty-three nights binds nothing.
    where, starts_on, expires_on = _validated(draft)
    matched = airings_lib.matching(where)
    try:
        rows = compile_restriction(
            draft.kind, draft.params, where, matched,
            starts_on=starts_on, expires_on=expires_on,
        )
    except RestrictionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return rows, matched, where, starts_on, expires_on


@router.post("/restrictions/preview")
def preview_restriction(draft: RestrictionDraft) -> dict[str, Any]:
    """What this restriction would do, before it is saved.

    Returns the sentence it reads as, every airing it touches, the breaks it
    moves, and the money on both named bases. Nothing is written.

    Three counts, because they are three different facts and collapsing them is
    what made the first round of this surface lie. ``matched_airings`` is what
    the author's scope selects, ``bound_airings`` is what the engine's own
    resolver actually holds, and ``changes`` is where a break count moves. A
    scope-level rule matches forty-three, binds forty-three and moves forty-one,
    and a gold rule with no pinned break to gild matches forty-three and binds
    none.

    A fourth fact rides on the change list, because three were not enough. Every
    change carries ``asked_for``: true when the sentence itself named that airing,
    false when a compiled row reached it anyway. ``collateral`` is the false half
    counted and priced on its own, so a rule that removes more than it says is
    read as that on screen instead of arriving as one undifferentiated total.
    """
    rows, matched, where, starts_on, expires_on = _compile(draft)
    words = sentence_lib.render(draft.kind, draft.params, where)
    effect = airings_lib.resolved_changes(rows, matched)
    changes = effect["changes"]
    # A per-airing kind derives one row per airing that breaches the sentence, so
    # those airings are exactly what it asked for. A scope-level kind names its
    # whole scope, so every airing it binds was asked for and the distinction does
    # not exist.
    per_airing = draft.kind in PER_AIRING_KINDS
    derived_ids = {row.airing.segment_id for row in rows if row.airing is not None}
    for change in changes:
        change["asked_for"] = (not per_airing) or change["segment_id"] in derived_ids
    scored_side = constraints_cost.scored(changes, bound=effect["bound"], matched=len(matched))
    exact_side = constraints_cost.reallocated(
        rows,
        days=effect["bound_days"] or None,
        scored_days=int(scored_side.get("days") or 0),
    )
    body: dict[str, Any] = {
        **words,
        "kind": draft.kind,
        "params": draft.params,
        "starts_on": starts_on,
        "expires_on": expires_on,
        "channel": airings_lib.operator_channel(),
        "matched_airings": len(matched),
        "compiled_rows": len(rows),
        "per_airing": draft.kind in PER_AIRING_KINDS,
        "bound_airings": effect["bound"],
        "bound_days": len(effect["bound_days"]),
        "changes": changes,
        "asked_for_airings": len(derived_ids) if per_airing else effect["bound"],
        "collateral": constraints_cost.collateral(
            changes, derived_ids, effect["bound_ids"], applies=per_airing,
        ),
        "unchanged_airings": max(effect["bound"] - len(changes) - effect["unknown"], 0),
        "airings_without_a_plan": effect["unknown"],
        "engine_skipped": constraints_cost.refusals(effect["skipped"][:8]),
        "scored": scored_side,
        "exact": exact_side,
        "starting_points": constraints_cost.starting_points(scored_side, exact_side),
        "already_in_force": constraints_cost.already_in_force(rows),
    }
    return body


@router.get("/restrictions/titles")
def restriction_titles(q: str = "") -> dict[str, Any]:
    """Programme titles on the operator's own channel, with their airing counts."""
    return airings_lib.titles(q)


@router.get("/restrictions/airings")
def restriction_airings(title: str = "") -> dict[str, Any]:
    """Every airing of one programme, so a restriction can name a single night.

    With no title this is not an error, it is an empty list with the input it
    is waiting for named in the payload. A missing input is a state the caller
    can render, and a refusal is not.
    """
    wanted = str(title or "").strip()
    matched = []
    if wanted:
        where = {"combinator": "and", "conditions": [{"field": "programme", "operator": "is", "value": wanted}]}
        matched = airings_lib.matching(where)
    return {
        "title": wanted,
        "channel": airings_lib.operator_channel(),
        "count": len(matched),
        "airings": airings_lib.airing_records(matched) if wanted else [],
        "reason": "" if wanted else "Name a programme to list its airings.",
    }


def _restriction_record(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    kind = str(first.get("rule_kind") or "")
    params = parse_params(first.get("rule_params_json"))
    # The sentence renders from the scope its author wrote, not from the scope of
    # one compiled row. A window restriction compiles to one row per airing, so
    # rendering off a row would read the whole rule back as the single night that
    # happened to sort first.
    where = parse_params(first.get("rule_where_json")) or first.get("where")
    words = sentence_lib.render(kind, params, where)
    expires_on = str(first.get("expires_on") or "")
    today = date.today().isoformat()
    return {
        **words,
        "restriction_id": str(first.get("restriction_id") or ""),
        "kind": kind,
        "params": params,
        "author": str(first.get("author") or ""),
        "reason": str(first.get("reason") or ""),
        "starts_on": str(first.get("starts_on") or ""),
        "expires_on": expires_on,
        "created_at": str(first.get("created_at") or ""),
        "status": "expired" if expires_on and expires_on <= today else "active",
        "row_count": len(rows),
        "constraint_ids": [str(row.get("constraint_id") or "") for row in rows],
        "effects": sorted({str(row.get("effect") or "") for row in rows}),
    }


@router.get("/restrictions")
def list_restrictions() -> dict[str, Any]:
    """Every authored restriction, newest first, plus any pre-authoring rows.

    A row written before restrictions existed carries no author and no sentence.
    It is reported as its own group rather than hidden, because a rule that binds
    the plan and cannot be read is exactly what this surface exists to end.
    """
    from kairos_api.constraints import _load_frame, _record

    frame = _load_frame()
    grouped: dict[str, list[dict[str, Any]]] = {}
    legacy: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        record = _record(row)
        key = str(record.get("restriction_id") or "").strip()
        if key:
            grouped.setdefault(key, []).append(record)
        else:
            legacy.append(record)
    records = [_restriction_record(rows) for rows in grouped.values()]
    records.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {
        "restrictions": records,
        "unauthored_rows": legacy,
        "channel": airings_lib.operator_channel(),
    }


@router.post("/restrictions", status_code=201)
def create_restriction(draft: RestrictionDraft, request: Request = None) -> dict[str, Any]:
    """Save a restriction: the sentence, its attribution, and the rows that bind."""
    import pandas as pd

    from kairos_api.constraints import (
        _STORE_LOCK,
        _load_frame,
        _record,
        _snapshot_before_write,
        _write_frame,
    )
    from kairos_api._constraint_options import where_json_cell

    rows, matched, where, starts_on, expires_on = _compile(draft)
    if not rows:
        raise HTTPException(
            status_code=400,
            detail="This restriction changes nothing in the current plan window, so there is nothing to save.",
        )
    words = sentence_lib.render(draft.kind, draft.params, where)
    restriction_id = uuid.uuid4().hex[:12]
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new_rows = [
        {
            "constraint_id": uuid.uuid4().hex[:12],
            "scope_type": "always",
            "scope_value": "",
            "channel": "",
            "effect": row.effect,
            "offset_seconds": "" if row.offset_seconds is None else str(row.offset_seconds),
            "offset_min_seconds": "",
            "offset_max_seconds": "",
            "count": "" if row.count is None else str(row.count),
            "duration_seconds": "",
            "duration_min_seconds": "",
            "duration_max_seconds": "",
            "order_index": "",
            "notes": words["sentence_en"],
            "where_json": where_json_cell(row.where),
            "restriction_id": restriction_id,
            "rule_kind": draft.kind,
            "rule_params_json": params_cell(draft.params),
            "rule_where_json": params_cell(where) if where else "",
            "author": str(draft.author or "").strip(),
            "reason": str(draft.reason or "").strip(),
            "starts_on": starts_on,
            "expires_on": expires_on,
            "created_at": created_at,
        }
        for row in rows
    ]
    with _STORE_LOCK:
        frame = _load_frame()
        frame = pd.concat([frame, pd.DataFrame(new_rows)], ignore_index=True)
        _snapshot_before_write(request)
        _write_frame(frame)
        saved = [_record(frame.iloc[index]) for index in range(len(frame) - len(new_rows), len(frame))]
    return _restriction_record(saved)


@router.delete("/restrictions/{restriction_id}")
def delete_restriction(restriction_id: str, request: Request = None) -> dict[str, Any]:
    """Remove every row one restriction wrote. Nothing else in the store moves."""
    from kairos_api.constraints import (
        _STORE_LOCK,
        _load_frame,
        _snapshot_before_write,
        _write_frame,
    )

    with _STORE_LOCK:
        frame = _load_frame()
        if "restriction_id" not in frame.columns:
            raise HTTPException(status_code=404, detail=f"restriction '{restriction_id}' not found")
        mask = frame["restriction_id"].astype(str) == restriction_id
        removed = int(mask.sum())
        if not removed:
            raise HTTPException(status_code=404, detail=f"restriction '{restriction_id}' not found")
        frame = frame[~mask].reset_index(drop=True)
        _snapshot_before_write(request)
        _write_frame(frame)
    return {"deleted": restriction_id, "rows_removed": removed}


__all__ = ["AUTHORING_COLUMNS", "router"]
