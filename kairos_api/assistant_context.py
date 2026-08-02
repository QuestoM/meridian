"""Day-level grounding for the in-product assistant.

Builds the drill-down sections of the assistant context: ``per_day_plan``, a
per-day table of the operator's OWN channel that is always included
(competitor channels stay out of the context entirely, surfacing only as an
aggregate count), and question-aware ``day_detail`` sections listing one plan
day's segments whenever the question names a date the saved plan actually
contains (ISO yyyy-mm-dd, dd/mm, dd.mm, or a weekday name). A clock (HH:MM) or
a programme-type word in the question upgrades the matching segments to their
complete saved rows. Parsing is conservative: no match adds nothing and no
date is ever guessed. ``enforce_budget`` keeps the whole serialized context
under a character budget (KAIROS_ASSISTANT_CONTEXT_BUDGET, default 60000) by
dropping day-detail rows lowest-revenue-first, and every cut is flagged in
``day_detail_truncated`` so the model can disclose it. Nothing is fabricated:
a section that cannot be built is reported absent by the caller, never
substituted.
"""

from __future__ import annotations

import math
import re
from datetime import date as _date
from typing import Any

# Re-exported so every caller and every test reaches the budget through this
# module exactly as before the split. The definitions live in
# kairos_api.assistant_context_budget.
from kairos_api.assistant_context_budget import (  # noqa: F401
    BUDGET_ENV,
    DAY_DETAIL_PREFIX,
    DEFAULT_CONTEXT_BUDGET,
    _context_budget,
    _serialized_size,
    _trim_recommendations,
    enforce_budget,
)

PER_DAY_SECTION = "per_day_plan"
DAY_DETAIL_ROW_CAP = 90
FULL_ROW_CAP = 30

_ISO_DATE_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_DAY_MONTH_RE = re.compile(r"\b(\d{1,2})[./](\d{1,2})(?:[./](\d{4}))?\b")
_CLOCK_RE = re.compile(r"\b([01]?\d|2[0-3]):([0-5]\d)\b")

# Python date.weekday(): Monday is 0 and Sunday is 6.
_HEBREW_WEEKDAYS = {"ראשון": 6, "שני": 0, "שלישי": 1, "רביעי": 2, "חמישי": 3, "שישי": 4, "שבת": 5}
_ENGLISH_WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6,
}

# Hebrew aliases for the English program_type vocabulary the weekly CSV carries.
# An alias fires only when its target type exists in the loaded plan, so the
# effective vocabulary always comes from the real saved data.
_HEBREW_TYPE_ALIASES = {
    "חדשות": "News", "דרמה": "Drama", "קומדיה": "Comedy", "ילדים": "Children",
    "מוזיקה": "Music", "מוסיקה": "Music", "דוקומנטרי": "Documentary", "תעודה": "Documentary",
    "ריאליטי": "Reality", "פרומו": "Promo", "לייפסטייל": "Lifestyle", "דיגיטל": "Digital",
    "תוכנית בוקר": "Morning Program", "תוכניות בוקר": "Morning Program",
    "תוכנית אירוח": "Talk Show", "אירוע מיוחד": "Special Event",
}

# Vocabulary words too common in ordinary questions to treat as a type filter.
_AMBIGUOUS_TYPE_WORDS = {"other"}

_HELPER_COLUMNS = ("date_text", "start_norm")


def _server() -> Any:
    """Lazy handle to server.py helpers (avoids an import cycle at module load)."""
    from kairos_api import server

    return server


def _num(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _retention_pct(value: Any) -> float | None:
    """A retention value as a percent with one decimal, or None when unknown."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    if numeric <= 1.5:
        numeric *= 100
    return round(numeric, 1)


def _normalize_clock(value: Any) -> str:
    text = str(value or "").strip()
    match = re.fullmatch(r"(\d{1,2}):(\d{2})(?::\d{2})?", text)
    if not match:
        return ""
    hour, minute = int(match.group(1)), int(match.group(2))
    if hour > 23 or minute > 59:
        return ""
    return f"{hour:02d}:{minute:02d}"


# --- owned-channel plan access. The competitor boundary lives here: everything
# below only ever sees rows of settings.operator_channel, and other channels
# survive solely as a count.
def _owned_frame() -> tuple[Any, str, int, str | None]:
    """The saved weekly plan scoped to the operator's own channel.

    Returns (frame, owned_channel, competitor_channel_count, reason). The frame
    is None whenever no owned rows can be produced, with reason naming exactly
    what is missing: honest absence over substitution.
    """
    import pandas as pd

    server = _server()
    owned = str(server._load_settings().operator_channel or "").strip()
    if not owned:
        return None, "", 0, "operator channel is not configured in settings"
    schedule = server._load_break_schedule()
    if schedule.empty:
        return None, owned, 0, "no saved weekly plan"
    if "channel" not in schedule.columns or "date" not in schedule.columns:
        return None, owned, 0, "saved plan lacks channel or date columns"
    frame = server._augment_segment_ids(schedule)
    channels = frame["channel"].astype(str).str.strip()
    competitors = int(channels[(channels != owned) & (channels != "")].nunique())
    frame = frame[channels == owned].copy()
    if frame.empty:
        return None, owned, competitors, "saved plan has no rows for the operator channel"
    frame["date_text"] = frame["date"].astype(str).str.strip()
    start = frame["start_time"] if "start_time" in frame.columns else pd.Series("", index=frame.index)
    frame["start_norm"] = start.map(_normalize_clock)
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue"), errors="coerce").fillna(0.0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention"), errors="coerce")
    frame["num_breaks"] = pd.to_numeric(frame.get("num_breaks"), errors="coerce").fillna(0).astype(int)
    return frame, owned, competitors, None


def _weekday_label(date_text: str, group: Any) -> str | None:
    if group is not None and "day" in getattr(group, "columns", []):
        for value in group["day"]:
            text = str(value or "").strip()
            if text and text.lower() != "nan":
                return text
    try:
        return _date.fromisoformat(date_text[:10]).strftime("%a")
    except ValueError:
        return None


def _per_day_plan_section(frame: Any, owned: str, competitors: int, reason: str | None) -> dict[str, Any]:
    """One compact row per saved plan day of the operator's own channel."""
    if frame is None:
        return {"channel": owned or None, "days": [], "reason": reason}
    days: list[dict[str, Any]] = []
    for date_text, group in frame.groupby("date_text", sort=True):
        days.append(
            {
                "date": str(date_text),
                "weekday": _weekday_label(str(date_text), group),
                "breaks": int(group["num_breaks"].sum()),
                "revenue_ils": int(round(float(group["predicted_revenue"].sum()))),
                "avg_retention_pct": _retention_pct(group["predicted_retention"].mean()),
            }
        )
    return {
        "channel": owned,
        "competitor_channels_excluded": competitors,
        "days": days,
        "totals": {
            "breaks": sum(day["breaks"] for day in days),
            "revenue_ils": sum(day["revenue_ils"] for day in days),
        },
    }


# --- conservative question parsing. Only expressions that resolve to data the
# saved plan actually contains ever match; anything else adds nothing.
def _strip_hebrew_prefixes(word: str) -> str:
    while len(word) > 3 and word[0] in "ובלמהשכ":
        word = word[1:]
    return word


def _question_weekdays(question: str) -> set[int]:
    """Weekday numbers the question names. Hebrew weekday words other than
    Shabbat require the word yom before them because bare words like shni or
    shlishi also mean ordinary numerals; English weekdays match full names only."""
    weekdays: set[int] = set()
    words = re.findall(r"[א-ת]+", question)
    for index, word in enumerate(words):
        if _strip_hebrew_prefixes(word) == "שבת":
            weekdays.add(_HEBREW_WEEKDAYS["שבת"])
            continue
        if _strip_hebrew_prefixes(word) != "יום" or index + 1 >= len(words):
            continue
        candidate = words[index + 1]
        if candidate.startswith("ה") and candidate[1:] in _HEBREW_WEEKDAYS:
            candidate = candidate[1:]
        if candidate in _HEBREW_WEEKDAYS:
            weekdays.add(_HEBREW_WEEKDAYS[candidate])
    lowered = question.lower()
    for name, number in _ENGLISH_WEEKDAYS.items():
        if re.search(rf"\b{name}\b", lowered):
            weekdays.add(number)
    return weekdays


def _question_dates(question: str, plan_dates: list[str]) -> list[str]:
    """The plan dates the question explicitly names, sorted ascending.

    A date expression that does not resolve to a saved plan day matches
    nothing, and a weekday name matches every plan day of that weekday rather
    than guessing one of them.
    """
    plan: list[tuple[str, _date]] = []
    for iso in plan_dates:
        try:
            plan.append((str(iso), _date.fromisoformat(str(iso)[:10])))
        except ValueError:
            continue
    found: set[str] = set()
    for match in _ISO_DATE_RE.finditer(question):
        year, month, day = (int(part) for part in match.groups())
        found.update(
            iso for iso, parsed in plan if (parsed.year, parsed.month, parsed.day) == (year, month, day)
        )
    for match in _DAY_MONTH_RE.finditer(question):
        day, month = int(match.group(1)), int(match.group(2))
        year = int(match.group(3)) if match.group(3) else None
        if not (1 <= day <= 31 and 1 <= month <= 12):
            continue
        found.update(
            iso
            for iso, parsed in plan
            if parsed.day == day and parsed.month == month and (year is None or parsed.year == year)
        )
    weekdays = _question_weekdays(question)
    if weekdays:
        found.update(iso for iso, parsed in plan if parsed.weekday() in weekdays)
    return sorted(found)


def _question_clocks(question: str) -> list[str]:
    return sorted({f"{int(hour):02d}:{minute}" for hour, minute in _CLOCK_RE.findall(question)})


def _type_vocabulary(frame: Any) -> list[str]:
    if "program_type" not in frame.columns:
        return []
    values = {str(value).strip() for value in frame["program_type"].dropna()}
    return sorted(value for value in values if value)


def _question_program_types(question: str, vocabulary: list[str]) -> list[str]:
    lowered = question.lower()
    found: set[str] = set()
    for vocab_type in vocabulary:
        term = vocab_type.lower()
        if term in _AMBIGUOUS_TYPE_WORDS:
            continue
        if re.search(rf"\b{re.escape(term)}\b", lowered):
            found.add(vocab_type)
    by_lower = {vocab_type.lower(): vocab_type for vocab_type in vocabulary}
    for alias, target in _HEBREW_TYPE_ALIASES.items():
        actual = by_lower.get(target.lower())
        if actual and actual.lower() not in _AMBIGUOUS_TYPE_WORDS and alias in question:
            found.add(actual)
    return sorted(found)


# --- day detail sections and the context budget.
def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric) or math.isinf(numeric):
        return None
    if numeric.is_integer() and abs(numeric) < 1e15:
        return int(numeric)
    return round(numeric, 4)


def _compact_row(row: Any) -> dict[str, Any]:
    entry = {
        "segment_id": str(row.get("segment_id", "")).strip(),
        "start": str(row.get("start_time", "")).strip(),
        "program_type": str(row.get("program_type", "")).strip(),
        "breaks": int(_num(row.get("num_breaks"), 0.0)),
        "revenue_ils": int(round(_num(row.get("predicted_revenue"), 0.0))),
        "retention_pct": _retention_pct(row.get("predicted_retention")),
    }
    if bool(row.get("is_gold", False)):
        entry["is_gold"] = True
    return entry


def _full_row(row: Any) -> dict[str, Any]:
    return {str(key): _json_value(value) for key, value in row.items() if str(key) not in _HELPER_COLUMNS}


def _day_detail_section(frame: Any, date_text: str, clocks: list[str], types: list[str]) -> dict[str, Any]:
    """Every owned-channel segment of one plan day, ordered by revenue
    descending so budget truncation always keeps the highest-revenue rows."""
    import pandas as pd

    day = frame[frame["date_text"] == date_text]
    day = day.sort_values(["predicted_revenue", "segment_id"], ascending=[False, True], kind="mergesort")
    rows_total = int(len(day))
    kept = day.head(DAY_DETAIL_ROW_CAP)
    section: dict[str, Any] = {
        "date": date_text,
        "weekday": _weekday_label(date_text, day),
        "channel": str(day["channel"].iloc[0]).strip() if rows_total else None,
        "rows_total": rows_total,
        "segments": [_compact_row(row) for _, row in kept.iterrows()],
    }
    omitted = rows_total - int(len(kept))
    if omitted > 0:
        section["truncated"] = True
        section["rows_omitted"] = omitted
    if clocks or types:
        mask = pd.Series(False, index=day.index)
        if clocks:
            mask |= day["start_norm"].isin(clocks)
        if types and "program_type" in day.columns:
            mask |= day["program_type"].astype(str).str.strip().isin(types)
        matched = day[mask]
        section["match"] = {}
        if clocks:
            section["match"]["clocks"] = clocks
        if types:
            section["match"]["program_types"] = types
        section["matched_full_rows"] = [_full_row(row) for _, row in matched.head(FULL_ROW_CAP).iterrows()]
        over = int(len(matched)) - FULL_ROW_CAP
        if over > 0:
            section["truncated"] = True
            section["rows_omitted"] = int(section.get("rows_omitted", 0)) + over
    return section


def extend_with_day_grounding(context: dict[str, Any], sources: list[str], question: str) -> None:
    """Append per_day_plan and any question-matched day_detail sections.

    Mutates context and sources in place under the caller's contract: a section
    that cannot be built is listed with an absent marker and omitted from the
    context, never substituted. Day-detail sections appear only when the
    question names a date the saved plan contains.
    """
    frame = None
    try:
        frame, owned, competitors, reason = _owned_frame()
        context[PER_DAY_SECTION] = _per_day_plan_section(frame, owned, competitors, reason)
        sources.append(PER_DAY_SECTION)
    except Exception:
        frame = None
        sources.append(f"{PER_DAY_SECTION} (absent)")
    if frame is None:
        return
    try:
        matched_dates = _question_dates(question, sorted(set(frame["date_text"])))
        if not matched_dates:
            return
        clocks = _question_clocks(question)
        types = _question_program_types(question, _type_vocabulary(frame))
        for date_text in matched_dates:
            name = f"{DAY_DETAIL_PREFIX} {date_text}"
            context[name] = _day_detail_section(frame, date_text, clocks, types)
            sources.append(name)
    except Exception:
        sources.append(f"{DAY_DETAIL_PREFIX} (absent)")
