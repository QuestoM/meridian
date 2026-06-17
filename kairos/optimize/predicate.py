"""Rich predicate evaluator for placement-constraint WHERE clauses.

The frozen predicate contract (also in docs/constraint-predicate-contract.md):

  Group     = {"combinator": "and" | "or", "conditions": [Node, ...]}
  Node      = Group | Condition
  Condition = {"field": <field>, "operator": <op>, "value": <value>}

Fields, their operators, and value types:
  "programme"  - the show Title (string): is, is_not, contains, not_contains,
                 starts_with, ends_with, regex, in
  "genre"      - program_type (string): same operators as programme
  "daypart"    - one of morning|noon|evening|prime|night:
                 operators is, is_not, in; value string or array
  "weekday"    - Mon|Tue|Wed|Thu|Fri|Sat|Sun:
                 operators is, is_not, in; value string or array
  "date"       - ISO yyyy-mm-dd: operators is, before, after, between, in;
                 value ISO date string; for 'between' {"min": iso, "max": iso};
                 for 'in' an array
  "hour"       - integer 0..23 (from segment broadcast time):
                 operators eq, lt, lte, gt, gte, between;
                 value integer; for 'between' {"min": int, "max": int}

The channel is NOT a predicate field. A constraint only matches segments on the
operator's own channel, enforced by the operator_channel parameter.

Defensive contract:
  - An unknown field, unknown operator, or malformed regex yields False for that
    single Condition (never raises, never matches everything by accident).
  - A Group with an empty conditions list yields False for "and" and False for
    "or" (safest empty-collection behavior: an empty filter never matches).
  - operator_channel is None or "" -> skip the channel filter (matches any
    channel; honest no-op before the operator picks one).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from kairos.data.dayparts import daypart_for_hour

logger = logging.getLogger(__name__)

# Canonical field names and their allowed operators.
_FIELD_OPS: dict[str, frozenset[str]] = {
    "programme":  frozenset({"is", "is_not", "contains", "not_contains",
                              "starts_with", "ends_with", "regex", "in"}),
    "genre":      frozenset({"is", "is_not", "contains", "not_contains",
                              "starts_with", "ends_with", "regex", "in"}),
    "daypart":    frozenset({"is", "is_not", "in"}),
    "weekday":    frozenset({"is", "is_not", "in"}),
    "date":       frozenset({"is", "before", "after", "between", "in"}),
    "hour":       frozenset({"eq", "lt", "lte", "gt", "gte", "between"}),
}

# Weekday abbreviation -> isoweekday int (1=Mon..7=Sun)
_WEEKDAY_ISO: dict[str, int] = {
    "mon": 1, "tue": 2, "wed": 3, "thu": 4,
    "fri": 5, "sat": 6, "sun": 7,
}
# isoweekday int -> abbreviated key
_ISO_WEEKDAY: dict[int, str] = {v: k.title() for k, v in _WEEKDAY_ISO.items()}

ALLOWED_FIELDS = frozenset(_FIELD_OPS)
# Flat union of all operators across all fields (for API validation).
ALLOWED_OPERATORS: dict[str, frozenset[str]] = _FIELD_OPS


def _norm(text: object) -> str:
    """Lowercase-strip for case-insensitive string comparison."""
    return str(text if text is not None else "").strip().lower()


def _segment_weekday(day: str) -> str:
    """Return the Mon..Sun abbreviation for a YYYY-MM-DD date string."""
    try:
        import pandas as pd
        return pd.Timestamp(day).strftime("%a")
    except Exception:
        return ""


def _extract_field(field: str, segment: "Any") -> "Any":
    """Derive the value of a predicate field from a ProgramSegment.

    Returns None when the value cannot be determined (segment attribute missing
    or un-parseable); a Condition using that field will then return False.
    """
    if field == "programme":
        return str(getattr(segment, "program_title", "") or "")
    if field == "genre":
        return str(getattr(segment, "program_type", "") or "")
    if field == "daypart":
        start = getattr(segment, "start_seconds", None)
        if start is None:
            return None
        hour = int(start // 3600) % 24
        return daypart_for_hour(hour)
    if field == "weekday":
        day = getattr(segment, "day", None)
        if not day:
            return None
        return _segment_weekday(str(day))
    if field == "date":
        return str(getattr(segment, "day", "") or "")
    if field == "hour":
        start = getattr(segment, "start_seconds", None)
        if start is None:
            return None
        return int(start // 3600) % 24
    return None


def _eval_string_op(op: str, field_value: str, cond_value: object) -> bool:
    """Evaluate string field operators (programme, genre)."""
    fv = _norm(field_value)
    if op == "is":
        return fv == _norm(cond_value)
    if op == "is_not":
        return fv != _norm(cond_value)
    if op == "contains":
        return _norm(cond_value) in fv
    if op == "not_contains":
        return _norm(cond_value) not in fv
    if op == "starts_with":
        return fv.startswith(_norm(cond_value))
    if op == "ends_with":
        return fv.endswith(_norm(cond_value))
    if op == "regex":
        try:
            return bool(re.search(str(cond_value or ""), field_value, re.IGNORECASE))
        except re.error:
            logger.debug("predicate: malformed regex %r -> False", cond_value)
            return False
    if op == "in":
        if not isinstance(cond_value, list):
            return False
        return fv in {_norm(v) for v in cond_value}
    return False


def _eval_category_op(op: str, field_value: str, cond_value: object) -> bool:
    """Evaluate category-enum operators (daypart, weekday)."""
    fv = _norm(field_value)
    if op == "is":
        return fv == _norm(cond_value)
    if op == "is_not":
        return fv != _norm(cond_value)
    if op == "in":
        if not isinstance(cond_value, list):
            return False
        return fv in {_norm(v) for v in cond_value}
    return False


def _eval_date_op(op: str, field_value: str, cond_value: object) -> bool:
    """Evaluate date operators."""
    fv = str(field_value or "").strip()
    if not fv:
        return False
    if op == "is":
        return fv == str(cond_value or "").strip()
    if op == "before":
        return fv < str(cond_value or "").strip()
    if op == "after":
        return fv > str(cond_value or "").strip()
    if op == "between":
        if not isinstance(cond_value, dict):
            return False
        lo = str(cond_value.get("min", "") or "").strip()
        hi = str(cond_value.get("max", "") or "").strip()
        if not lo or not hi:
            return False
        return lo <= fv <= hi
    if op == "in":
        if not isinstance(cond_value, list):
            return False
        return fv in {str(v or "").strip() for v in cond_value}
    return False


def _eval_hour_op(op: str, field_value: int, cond_value: object) -> bool:
    """Evaluate hour operators."""
    try:
        fv = int(field_value)
    except (TypeError, ValueError):
        return False
    if op == "eq":
        try:
            return fv == int(cond_value)
        except (TypeError, ValueError):
            return False
    if op == "lt":
        try:
            return fv < int(cond_value)
        except (TypeError, ValueError):
            return False
    if op == "lte":
        try:
            return fv <= int(cond_value)
        except (TypeError, ValueError):
            return False
    if op == "gt":
        try:
            return fv > int(cond_value)
        except (TypeError, ValueError):
            return False
    if op == "gte":
        try:
            return fv >= int(cond_value)
        except (TypeError, ValueError):
            return False
    if op == "between":
        if not isinstance(cond_value, dict):
            return False
        try:
            lo = int(cond_value.get("min", 0))
            hi = int(cond_value.get("max", 23))
        except (TypeError, ValueError):
            return False
        return lo <= fv <= hi
    return False


def _eval_condition(condition: dict[str, Any], segment: "Any") -> bool:
    """Evaluate a single Condition node against a segment.

    Returns False on any unexpected field, operator, or value type, never raises.
    """
    field = str(condition.get("field", "") or "")
    op = str(condition.get("operator", "") or "")
    value = condition.get("value")

    if field not in _FIELD_OPS:
        logger.debug("predicate: unknown field %r -> False", field)
        return False
    if op not in _FIELD_OPS[field]:
        logger.debug("predicate: unknown operator %r for field %r -> False", op, field)
        return False

    raw = _extract_field(field, segment)
    if raw is None:
        return False

    if field in ("programme", "genre"):
        return _eval_string_op(op, str(raw), value)
    if field in ("daypart", "weekday"):
        return _eval_category_op(op, str(raw), value)
    if field == "date":
        return _eval_date_op(op, str(raw), value)
    if field == "hour":
        return _eval_hour_op(op, raw, value)
    return False


def _is_group(node: dict[str, Any]) -> bool:
    """True when node has a 'combinator' key (it is a Group, not a Condition)."""
    return "combinator" in node


def _eval_node(node: dict[str, Any], segment: "Any") -> bool:
    """Evaluate a Node (Group or Condition) recursively."""
    if _is_group(node):
        return _eval_group(node, segment)
    return _eval_condition(node, segment)


def _eval_group(group: dict[str, Any], segment: "Any") -> bool:
    """Evaluate a Group (combinator + conditions list) against a segment.

    An empty conditions list always yields False (safest: an empty filter
    that accidentally matches all segments would be a silent source of wrong
    constraint application).
    """
    combinator = str(group.get("combinator", "") or "").strip().lower()
    conditions = group.get("conditions") or []
    if not conditions:
        return False
    if combinator == "and":
        return all(_eval_node(node, segment) for node in conditions)
    if combinator == "or":
        return any(_eval_node(node, segment) for node in conditions)
    # Unknown combinator: defensive False.
    logger.debug("predicate: unknown combinator %r -> False", combinator)
    return False


def evaluate_predicate(
    group: dict[str, Any],
    segment: "Any",
    *,
    operator_channel: str | None = None,
) -> bool:
    """Top-level evaluator: does this predicate match this segment?

    Channel filter: if ``operator_channel`` is a non-empty string, the segment's
    channel must equal it exactly. If ``operator_channel`` is None or ``""``,
    the channel filter is skipped (matches any channel), which is the honest
    no-op before the operator has picked a channel in the dashboard settings.

    The predicate tree is then evaluated: a Group with 'and' requires all
    Conditions/sub-groups to be True; 'or' requires at least one. Nesting is
    fully recursive. Any evaluation error in a node yields False for that
    node only and does not propagate.
    """
    if operator_channel:
        seg_channel = str(getattr(segment, "channel", "") or "")
        if seg_channel != operator_channel:
            return False
    try:
        return _eval_group(group, segment)
    except Exception:
        logger.debug("predicate: unexpected error evaluating group -> False", exc_info=True)
        return False
