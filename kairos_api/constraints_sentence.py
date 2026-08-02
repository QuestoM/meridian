"""A restriction, read back as the sentence its author would have said.

The store speaks in effects and offsets. A programming representative speaks in
sentences, and the two have to be the same statement or the list they read is
not the list the engine runs. So one renderer, on the server, used by the list,
by the composer's own echo of the draft and by anything else that quotes a
restriction: there is exactly one wording of a given rule in the product.

Both languages are produced together for the same reason. A rule that reads
correctly in Hebrew and drifts in English is a rule two people disagree about.

The scope is phrased per field rather than field-operator-value, because
"programme is X and date is Y" is a query and "X on 2024-11-01" is a sentence.
Where a field and operator have no natural phrase the plain form is used, which
is honest about being a filter rather than pretending to read as prose.
"""

from __future__ import annotations

from typing import Any, Optional

from kairos_api.constraints_language import (
    CLEAN_OPEN,
    CLEAN_TAIL,
    EXACT_BREAKS,
    FIXED_SLOT,
    GOLD,
    NO_BREAKS,
)

EN, HE = 0, 1

_EVERYTHING = ("every programme on your channel", "כל התוכניות בערוץ שלכם")

_WEEKDAYS = {
    "Sun": ("Sundays", "ימי ראשון"),
    "Mon": ("Mondays", "ימי שני"),
    "Tue": ("Tuesdays", "ימי שלישי"),
    "Wed": ("Wednesdays", "ימי רביעי"),
    "Thu": ("Thursdays", "ימי חמישי"),
    "Fri": ("Fridays", "ימי שישי"),
    "Sat": ("Saturdays", "שבתות"),
}

_DAYPARTS = {
    "morning": ("the morning", "רצועת הבוקר"),
    "noon": ("the afternoon", "רצועת הצהריים"),
    "evening": ("the evening", "רצועת הערב"),
    "prime": ("primetime", "רצועת הפריים"),
    "night": ("the night", "רצועת הלילה"),
}


def _values(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        return [str(value.get("min", "")), str(value.get("max", ""))]
    return [str(value)]


def _join(parts: list[str], locale_index: int) -> str:
    words = [part for part in parts if part]
    if not words:
        return ""
    if len(words) == 1:
        return words[0]
    tail = " and " if locale_index == EN else " ו"
    return ", ".join(words[:-1]) + tail + words[-1]


def _phrase(node: dict[str, Any], locale_index: int) -> str:
    field = str(node.get("field", ""))
    operator = str(node.get("operator", ""))
    values = _values(node.get("value"))
    listed = _join(values, locale_index)
    if field == "programme":
        if operator == "is":
            return listed
        if operator == "in":
            return listed
        if operator == "contains":
            return f"programmes containing {listed}" if locale_index == EN else f"תוכניות שמכילות {listed}"
        if operator == "is_not":
            return f"every programme except {listed}" if locale_index == EN else f"כל התוכניות חוץ מ{listed}"
        return f"programmes matching {listed}" if locale_index == EN else f"תוכניות שתואמות {listed}"
    if field == "genre":
        if operator in ("is", "in"):
            return f"{listed} programmes" if locale_index == EN else f"תוכניות מסוג {listed}"
        return f"programmes not of type {listed}" if locale_index == EN else f"תוכניות שאינן מסוג {listed}"
    if field == "date":
        if operator == "is":
            return f"on {listed}" if locale_index == EN else f"בתאריך {listed}"
        if operator == "before":
            return f"before {listed}" if locale_index == EN else f"לפני {listed}"
        if operator == "after":
            return f"from {listed}" if locale_index == EN else f"החל מ-{listed}"
        if operator == "between":
            return (f"between {values[0]} and {values[1]}" if locale_index == EN
                    else f"בין {values[0]} ל-{values[1]}")
        return f"on {listed}" if locale_index == EN else f"בתאריכים {listed}"
    if field == "weekday":
        words = _join([_WEEKDAYS.get(value, (value, value))[locale_index] for value in values], locale_index)
        if operator == "is_not":
            return f"except on {words}" if locale_index == EN else f"חוץ מ{words}"
        return f"on {words}" if locale_index == EN else f"ב{words}"
    if field == "daypart":
        words = _join([_DAYPARTS.get(value, (value, value))[locale_index] for value in values], locale_index)
        if operator == "is_not":
            return f"outside {words}" if locale_index == EN else f"מחוץ ל{words}"
        return f"in {words}" if locale_index == EN else f"ב{words}"
    if field == "hour":
        if operator == "between":
            return (f"between {values[0]}:00 and {values[1]}:00" if locale_index == EN
                    else f"בין {values[0]}:00 ל-{values[1]}:00")
        if operator in ("gte", "gt"):
            return f"from {listed}:00" if locale_index == EN else f"מהשעה {listed}:00"
        if operator in ("lte", "lt"):
            return f"until {listed}:00" if locale_index == EN else f"עד השעה {listed}:00"
        return f"at {listed}:00" if locale_index == EN else f"בשעה {listed}:00"
    return f"{field} {operator} {listed}".strip()


def _scope_text(where: Optional[dict[str, Any]], locale_index: int) -> str:
    if not where:
        return _EVERYTHING[locale_index]
    if "combinator" in where:
        children = [_scope_text(child, locale_index) for child in (where.get("conditions") or [])]
        children = [child for child in children if child and child != _EVERYTHING[locale_index]]
        if not children:
            return _EVERYTHING[locale_index]
        if where.get("combinator") == "or":
            joiner = " or " if locale_index == EN else " או "
            return joiner.join(children)
        return " ".join(children)
    return _phrase(where, locale_index)


def _mmss(offset_seconds: Any) -> str:
    try:
        total = int(float(offset_seconds))
    except (TypeError, ValueError):
        return "00:00"
    return f"{total // 60:02d}:{total % 60:02d}"


def _english(kind: str, params: dict[str, Any], scope: str) -> str:
    minutes = params.get("protected_minutes")
    if kind == CLEAN_TAIL:
        return f"No breaks in the last {minutes} minutes of {scope}"
    if kind == CLEAN_OPEN:
        return f"No breaks in the first {minutes} minutes of {scope}"
    if kind == NO_BREAKS:
        return f"No breaks at all in {scope}"
    if kind == EXACT_BREAKS:
        return f"Exactly {params.get('count')} breaks in {scope}"
    if kind == FIXED_SLOT:
        return f"A break {_mmss(params.get('offset_seconds'))} into {scope}"
    if kind == GOLD:
        return f"Gold breaks in {scope}"
    return scope


def _hebrew(kind: str, params: dict[str, Any], scope: str) -> str:
    minutes = params.get("protected_minutes")
    if kind == CLEAN_TAIL:
        return f"אין ברייקים ב-{minutes} הדקות האחרונות של {scope}"
    if kind == CLEAN_OPEN:
        return f"אין ברייקים ב-{minutes} הדקות הראשונות של {scope}"
    if kind == NO_BREAKS:
        return f"אין ברייקים כלל ב{scope}"
    if kind == EXACT_BREAKS:
        return f"בדיוק {params.get('count')} ברייקים ב{scope}"
    if kind == FIXED_SLOT:
        return f"ברייק בדקה {_mmss(params.get('offset_seconds'))} של {scope}"
    if kind == GOLD:
        return f"ברייקי זהב ב{scope}"
    return scope


def render(kind: str, params: dict[str, Any], where: Optional[dict[str, Any]]) -> dict[str, str]:
    """The restriction as one sentence in each language, plus its scope phrase."""
    scope_en = _scope_text(where, EN)
    scope_he = _scope_text(where, HE)
    return {
        "sentence_en": _english(kind, params or {}, scope_en),
        "sentence_he": _hebrew(kind, params or {}, scope_he),
        "scope_en": scope_en,
        "scope_he": scope_he,
    }
