"""Option-list helpers for the constraints API.

Separated to keep kairos_api/constraints.py under the 450-line limit.
These functions build the serialisable payload for /api/constraints/options.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException

from kairos.optimize.predicate import ALLOWED_FIELDS, ALLOWED_OPERATORS

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Predicate validation (used by create / update handlers)
# ---------------------------------------------------------------------------

def validate_where(where: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Validate a predicate Group tree against the frozen vocab.

    Rejects (HTTP 400) any Condition node whose field or operator is not in the
    frozen contract. Unknown combinators are also rejected. Returns the unchanged
    dict when valid, or None when where is None.
    """
    if where is None:
        return None
    if not isinstance(where, dict):
        raise HTTPException(status_code=400, detail="where must be a JSON object (Group)")
    _validate_node(where)
    return where


def _validate_node(node: dict[str, Any]) -> None:
    """Recursively validate a Node (Group or Condition), raising HTTP 400 on errors."""
    if not isinstance(node, dict):
        raise HTTPException(
            status_code=400,
            detail=f"each node in the predicate must be a JSON object, got {type(node).__name__}",
        )
    if "combinator" in node:
        combinator = str(node.get("combinator", "") or "")
        if combinator not in ("and", "or"):
            raise HTTPException(
                status_code=400,
                detail=f"Group combinator must be 'and' or 'or', got {combinator!r}",
            )
        conditions = node.get("conditions")
        if not isinstance(conditions, list):
            raise HTTPException(status_code=400, detail="Group must have a 'conditions' array")
        for child in conditions:
            _validate_node(child)
    else:
        field = str(node.get("field", "") or "")
        operator = str(node.get("operator", "") or "")
        if field not in ALLOWED_FIELDS:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown predicate field {field!r}. Allowed: {sorted(ALLOWED_FIELDS)}",
            )
        allowed_ops = ALLOWED_OPERATORS.get(field, frozenset())
        if operator not in allowed_ops:
            raise HTTPException(
                status_code=400,
                detail=f"Operator {operator!r} not allowed for field {field!r}. Allowed: {sorted(allowed_ops)}",
            )


def where_json_cell(where: Optional[dict[str, Any]]) -> str:
    """Serialize the where predicate to a compact JSON string for the CSV cell."""
    if where is None:
        return ""
    return json.dumps(where, ensure_ascii=False, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Option lists
# ---------------------------------------------------------------------------

def channel_options() -> list[str]:
    """Real channel names from the reference EPG, sorted and de-duplicated."""
    try:
        from kairos.data.loaders import load_programmes

        frame = load_programmes()
        if "Channel" not in frame.columns:
            return []
        names = {str(c).strip() for c in frame["Channel"].dropna() if str(c).strip()}
        return sorted(names)
    except Exception:
        return []


def genre_options() -> list[str]:
    """Distinct program_type values from the reference EPG, sorted."""
    try:
        from kairos.data.loaders import load_programmes

        frame = load_programmes()
        col = next((c for c in ("program_type", "programme_type") if c in frame.columns), None)
        if col is None:
            return []
        names = {str(v).strip() for v in frame[col].dropna() if str(v).strip()}
        return sorted(names)
    except Exception:
        return []


def weekday_options() -> list[dict[str, Any]]:
    """ISO weekday tokens 1..7 with bilingual labels (Mon..Sun)."""
    names_en = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    names_he = ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
    return [{"key": str(i + 1), "en": names_en[i], "he": names_he[i]} for i in range(7)]


def daypart_options_list() -> list[dict[str, Any]]:
    """Active daypart vocab with keys and labels, for the predicate builder."""
    from kairos.data.dayparts import daypart_options
    return daypart_options()


def predicate_field_schema() -> list[dict[str, Any]]:
    """Machine-readable field schema: field, allowed operators, value type hints."""
    return [
        {
            "field": "programme",
            "label_en": "Programme title",
            "label_he": "שם תוכנית",
            "operators": sorted(ALLOWED_OPERATORS["programme"]),
            "value_type": "string (or array for 'in')",
        },
        {
            "field": "genre",
            "label_en": "Genre / programme type",
            "label_he": "ז'אנר",
            "operators": sorted(ALLOWED_OPERATORS["genre"]),
            "value_type": "string (or array for 'in')",
        },
        {
            "field": "daypart",
            "label_en": "Daypart",
            "label_he": "פס שידור",
            "operators": sorted(ALLOWED_OPERATORS["daypart"]),
            "value_type": "string key (or array for 'in')",
        },
        {
            "field": "weekday",
            "label_en": "Weekday",
            "label_he": "יום שבוע",
            "operators": sorted(ALLOWED_OPERATORS["weekday"]),
            "value_type": "Mon|Tue|Wed|Thu|Fri|Sat|Sun (or array for 'in')",
        },
        {
            "field": "date",
            "label_en": "Date",
            "label_he": "תאריך",
            "operators": sorted(ALLOWED_OPERATORS["date"]),
            "value_type": "ISO yyyy-mm-dd string; 'between' uses {min, max}; 'in' uses array",
        },
        {
            "field": "hour",
            "label_en": "Hour of day",
            "label_he": "שעה ביום",
            "operators": sorted(ALLOWED_OPERATORS["hour"]),
            "value_type": "integer 0..23; 'between' uses {min, max}",
        },
    ]


def load_operator_channel() -> str:
    """Read operator_channel from kairos_settings.json, defaulting to empty string."""
    settings_path = ROOT / "data" / "kairos_settings.json"
    if not settings_path.exists():
        return ""
    try:
        with settings_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return str(data.get("operator_channel", "") or "")
    except Exception:
        return ""
