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


# Every refusal these routes write themselves, one entry each, both languages in
# it. The rule this module opens with covers a rule READ BACK; a rule REFUSED is
# the same sentence with the same reader, and it was authored in English alone,
# so a Hebrew composer got "This restriction changes nothing in the current plan
# window" and, where the sentence named a field, the store's own Latin column
# name inside it. :func:`refuse` is the only way one of these leaves the
# process, and it carries both halves or it carries neither.
_FIELD_NAMES = {
    "starts_on": ("The start date", "תאריך ההתחלה"),
    "expires_on": ("The end date", "תאריך הסיום"),
}

REFUSALS: dict[str, dict[str, str]] = {
    "bad_iso_date": {"en": "{field} has to be a calendar date, and {value} is not one.",
                     "he": "{field} חייב להיות תאריך בלוח השנה, ו-{value} אינו כזה."},
    "end_before_start": {"en": "The end date has to fall after the start date.",
                         "he": "תאריך הסיום חייב לחול אחרי תאריך ההתחלה."},
    "unknown_kind": {"en": "There is no restriction of that sort to write.",
                     "he": "אין סוג הגבלה כזה לכתוב."},
    "will_not_compile": {"en": "This restriction cannot be compiled as written: {problem}",
                         "he": "לא ניתן להרכיב את ההגבלה הזאת כפי שנכתבה: {problem}"},
    "nothing_to_save": {"en": "This restriction changes nothing in the current plan window, so there is nothing to save.",
                        "he": "ההגבלה הזאת אינה משנה דבר בחלון התוכנית הנוכחי, ולכן אין מה לשמור."},
    "restriction_gone": {"en": "That restriction is not in the store, so there is nothing to remove.",
                         "he": "ההגבלה הזאת אינה במאגר, ולכן אין מה להסיר."},
    "constraint_gone": {"en": "That condition is not in the store, so there is nothing to remove.",
                        "he": "התנאי הזה אינו במאגר, ולכן אין מה להסיר."},
    "bad_scope_type": {"en": "That is not a scope a condition can be written against.",
                       "he": "זה אינו היקף שאפשר לכתוב מולו תנאי."},
    "bad_effect": {"en": "That is not an effect a condition can hold.",
                   "he": "זו אינה השפעה שתנאי יכול לשאת."},
    "no_segments": {"en": "The plan of record holds no broadcast day for that channel and date.",
                    "he": "בתוכנית הרשומה אין יום שידור לערוץ ולתאריך האלה."},
    "segments_failed": {"en": "The plan segments could not be built for that day, so there is nothing to preview.",
                        "he": "לא ניתן היה לבנות את מקטעי התוכנית ליום הזה, ולכן אין מה להציג."},
    "name_a_programme": {"en": "Name a programme to list its airings.",
                         "he": "נקבו בשם תוכנית כדי לראות את השידורים שלה."},
}


def say(code: str, **fields: object) -> tuple[str, str]:
    """This refusal in English and in Hebrew, together or not at all.

    A field whose value is a two-item tuple carries its own two languages, which
    is how the name of a date field stays in the language of the sentence.
    """
    words = REFUSALS.get(code)
    if words is None:
        return "", ""
    rendered = [
        {name: str(value[index] if isinstance(value, tuple) and len(value) == 2 else value)
         for name, value in fields.items()}
        for index in (EN, HE)
    ]
    try:
        return words["en"].format(**rendered[EN]), words["he"].format(**rendered[HE])
    except (KeyError, IndexError):
        return "", ""


def refuse(code: str, status: int = 400, **fields: object) -> Exception:
    """The refusal as the exception a route raises, both languages on the wire.

    ``detail`` is an object rather than a string because a refusal has one reader
    and two possible languages and the server does not know which. The surface
    reads its own half off it. :func:`kairos_api.uploads_messages.reject` puts
    the same pair on the same wire under ``detail`` and ``detail_he``; this is
    that contract for a route that has to raise rather than return.
    """
    from fastapi import HTTPException

    english, hebrew = say(code, **fields)
    return HTTPException(status_code=status, detail={"en": english, "he": hebrew, "code": code})


def field_name(label: str) -> tuple[str, str]:
    """One authoring field as a person says it, in both languages."""
    return _FIELD_NAMES.get(label, (label, label))
