"""Metadata retained with a successful assistant conversation entry.

The answer alone is not enough to audit a data-backed claim after a reload.
This selects fields the ask already produced: its sources, bounded public tool
trace, advertiser coverage, disclosure and elapsed time.  It performs no new
read and changes no response contract.
"""

from __future__ import annotations

from typing import Any


def _trace(body: dict[str, Any]) -> list[dict[str, Any]]:
    raw = body.get("tool_trace")
    return [dict(step) for step in raw if isinstance(step, dict)] if isinstance(raw, list) else []


def _advertiser_coverage(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in trace:
        result = step.get("result")
        if isinstance(result, dict) and result.get("kind") == "advertiser_airings":
            coverage = result.get("coverage")
            return dict(coverage) if isinstance(coverage, dict) else None
    return None


def from_ask(body: dict[str, Any], elapsed_seconds: float | None) -> dict[str, Any]:
    """Return backward-compatible optional fields for one stored entry."""
    grounding = body.get("grounding") if isinstance(body.get("grounding"), dict) else {}
    sources = grounding.get("sources") if isinstance(grounding.get("sources"), list) else []
    trace = _trace(body)
    try:
        elapsed = round(max(float(elapsed_seconds or 0.0), 0.0), 3)
    except (TypeError, ValueError):
        elapsed = None
    disclosure = body.get("context_disclosure")
    if not isinstance(disclosure, (str, dict)):
        disclosure = None
    return {
        "sources": [str(source) for source in sources],
        "tool_trace": trace,
        "coverage": _advertiser_coverage(trace),
        "elapsed_seconds": elapsed,
        "context_disclosure": disclosure,
    }

