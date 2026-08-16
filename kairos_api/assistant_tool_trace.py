"""Small, safe result snapshots for durable assistant traces.

Most tool results belong only in the model turn: persisting every payload would
turn a conversation into a second copy of the product database.  A named
advertiser-airings answer is different because its coverage is part of the
claim.  This module keeps the compact evidence a reader needs after a reload,
with hard caps below the read tool's own pagination cap.
"""

from __future__ import annotations

from typing import Any

ADVERTISER_TOOL = "get_advertiser_airings"
MAX_TRACE_AIRINGS = 10
MAX_TRACE_GROUPS = 5


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _rows(value: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [dict(row) for row in value[:limit] if isinstance(row, dict)]


def compact_result(name: str, payload: Any) -> dict[str, Any] | None:
    """The bounded evidence a finished trace may retain, or ``None``."""
    if name != ADVERTISER_TOOL or not isinstance(payload, dict) or "error" in payload:
        return None
    airings = _rows(payload.get("airings"), MAX_TRACE_AIRINGS)
    pagination = _mapping(payload.get("pagination"))
    total = int(pagination.get("total") or len(airings))
    return {
        "kind": "advertiser_airings",
        "status": payload.get("status"),
        "identity": _mapping(payload.get("identity")),
        "coverage": _mapping(payload.get("coverage")),
        "summary": _mapping(payload.get("summary")),
        "campaigns": _rows(payload.get("campaigns"), MAX_TRACE_GROUPS),
        "creatives": _rows(payload.get("creatives"), MAX_TRACE_GROUPS),
        "airings": airings,
        "pagination": pagination,
        "trace_airings_omitted": max(total - len(airings), 0),
        "basis": payload.get("basis"),
    }


def trace_step(name: str, ok: bool, source: str | None, payload: Any) -> dict[str, Any]:
    """One public trace row, adding a bounded result only where supported."""
    step: dict[str, Any] = {"tool": name, "ok": ok}
    if source:
        step["source"] = source
    result = compact_result(name, payload)
    if result is not None:
        step["result"] = result
    return step

