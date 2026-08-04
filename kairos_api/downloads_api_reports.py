"""The report shelf: what each report carries, and the basis it was built on.

Split out of ``downloads_api.py`` under the file-size cap and named by the
``<parent stem>_<role>.py`` rule. Nothing here reads a request or writes a
file: it is handed the plan, the settings and the counts, and it returns
records.

**A row count is a promise about a file.** The shelf prints a row count beside
every report and the surface tells the reader it is the exact number of rows
that download will carry, so this module computes each count from the same
source the download reads and never from a nearby number that happens to be
handy. Measured before this change: the revenue card printed 2,391, which is
the operator's break count on the saved plan, while the file it downloads
carries one row per calendar day, which is 30. Four of the five were right and
the fifth was off by a factor of eighty.

**Every report declares its basis.** Four facts, attached to the report rather
than to a tooltip: what one row is, the period the rows cover, the scope they
are summed over, and the file they are built from with the moment that source
last changed. A fact that cannot be computed is left out rather than filled
with a placeholder.

**And the weekly plan declares what is inside the file it hands over.** That
download is the one report built from a file carrying every channel the
optimizer schedules, and Bar 3 freezes its row count at the whole file. Measured
on the shipped plan with the operator channel configured: 8,704 rows, of which
2,540 are the operator's own and 6,164 are on 3 other channels. The screen was
scoped and the file was not, and the only place that said so was a preview the
person who clicks download never opens. The split now travels on the report
itself, as counts and never as names, so nobody takes that file away without
being told what is in it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from kairos_api import channel_scope

REPORT_IDS = ("weekly-plan", "compliance", "revenue", "daily-spots", "data-quality")

PLAN_FILE = "output/weekly_break_schedule.csv"

# What one row of the download is. A count without this is a number nobody can
# check, because two readers will count two different things.
UNITS: dict[str, dict[str, str]] = {
    "weekly-plan": {
        "en": "One row is one programme segment on the saved plan, with the breaks placed in it.",
        "he": "כל שורה היא רצועת שידור אחת בתוכנית השמורה, עם הברייקים שנקבעו בה.",
    },
    "compliance": {
        "en": "One row is one regulatory check, with the observed value against its limit.",
        "he": "כל שורה היא בדיקת רגולציה אחת, עם הערך שנמדד מול המגבלה שלה.",
    },
    "revenue": {
        "en": "One row is one calendar day of the saved plan, on your own channel.",
        "he": "כל שורה היא יום קלנדרי אחד בתוכנית השמורה, בערוץ שלכם.",
    },
    "daily-spots": {
        "en": "One row is one ad in the daily log, priced or dropped with the reason.",
        "he": "כל שורה היא פרסומת אחת ביומן היומי, מתומחרת או שנשמטה עם הסיבה.",
    },
    "data-quality": {
        "en": "One row is one audited file, present or missing.",
        "he": "כל שורה היא קובץ אחד בבקרה, קיים או חסר.",
    },
}

BASIS_LABELS: dict[str, dict[str, str]] = {
    "period": {"en": "Period", "he": "תקופה"},
    "scope": {"en": "Scope", "he": "היקף"},
    "built_from": {"en": "Built from", "he": "נבנה מתוך"},
    "updated": {"en": "Source updated", "he": "המקור עודכן"},
}

# Scope sentences that are not a channel name. The plan file carries every
# channel the optimizer schedules, so it is described as an unnamed whole
# rather than as a list, which is the only shape a competitor may take.
SCOPES: dict[str, dict[str, str]] = {
    "whole_plan": {
        "en": "Every channel-day on the saved plan file",
        "he": "כל שילוב ערוץ־יום בקובץ התוכנית השמורה",
    },
    "audited_files": {
        "en": "The audited source file set",
        "he": "מערך קבצי המקור שבבקרה",
    },
    "daily_log": {
        "en": "One broadcast day of booked ads",
        "he": "יום שידור אחד של פרסומות שהוזמנו",
    },
    "plan_guardrails": {
        "en": "The saved plan, against the licence limits in force",
        "he": "התוכנית השמורה, מול מגבלות הרישיון שבתוקף",
    },
    "no_channel": {
        "en": "Not scoped to a channel: no operator channel is set, so set it on the settings screen before reading this as yours",
        "he": "לא מוגבל לערוץ: לא מוגדר ערוץ מפעיל, אז הגדירו אותו במסך ההגדרות לפני שקוראים את זה כשלכם",
    },
}


# What is inside the file this card hands over, in both languages. A count of
# other channels is the unnamed aggregate the boundary allows; a name is not,
# and none of these sentences carries one. The two that cannot be computed name
# the missing input and the screen that supplies it instead of printing a zero.
#
# The rows are said as "yours" and "not yours" rather than as two channel
# groups, because the second figure is every row the boundary would drop and a
# plan row with no channel in it is one of those. The count of other channels
# beside them counts named channels only, so both halves are true of a file
# nobody has looked at yet and not only of the one measured today.
DOWNLOAD_SCOPE: dict[str, dict[str, str]] = {
    "real": {
        "en": "This download carries the whole plan file: {owned} rows are on your own channel and {other} rows are not. The file names {channels} other channel(s).",
        "he": "ההורדה הזו נושאת את כל קובץ התוכנית: {owned} שורות בערוץ שלכם ו־{other} שורות שאינן שלכם, ובקובץ מופיעים ערוצים נוספים, {channels} במספר.",
    },
    "owned_only": {
        "en": "Every one of the {owned} rows in this download is on your own channel.",
        "he": "כל {owned} השורות בהורדה הזו הן בערוץ שלכם.",
    },
    "unknown": {
        "en": "No operator channel is set, so there is no way to say which of the {total} rows in this download are yours. Set the operator channel on the settings screen.",
        "he": "לא מוגדר ערוץ מפעיל, ולכן אין דרך לומר אילו מ־{total} השורות בהורדה הזו הן שלכם. הגדירו את ערוץ המפעיל במסך ההגדרות.",
    },
    "unavailable": {
        "en": "The plan file carries no channel column, so the split of this download between your own channel and the others cannot be read.",
        "he": "בקובץ התוכנית אין עמודת ערוץ, ולכן לא ניתן לקרוא את הפילוח של ההורדה הזו בין הערוץ שלכם לבין האחרים.",
    },
}

# The three states a count can be in, and the wording each one is said with.
SCOPE_STATE = {"real": "real", "owned_only": "real", "unknown": "unknown", "unavailable": "unavailable"}

# First-strong isolate, around a figure inside a right-to-left sentence, the way
# every other bilingual sentence in this destination wraps a left-to-right run.
ISOLATE_START = "⁨"
ISOLATE_END = "⁩"


def _figures(figures: dict[str, int]) -> tuple[dict[str, str], dict[str, str]]:
    """One set of counts, written for each language: plain, and isolated."""
    plain = {name: f"{int(value):,}" for name, value in figures.items()}
    return plain, {name: f"{ISOLATE_START}{text}{ISOLATE_END}" for name, text in plain.items()}


def _scope_record(code: str, figures: dict[str, int]) -> dict[str, Any]:
    plain, isolated = _figures(figures)
    words = DOWNLOAD_SCOPE[code]
    return {
        "state": SCOPE_STATE[code],
        "rows_total": int(figures["total"]),
        "en": words["en"].format(**plain),
        "he": words["he"].format(**isolated),
    }


def download_scope(schedule: pd.DataFrame, channel: str) -> dict[str, Any] | None:
    """What the weekly plan download really carries, as counts and never names.

    The same boundary call the preview of this report already makes, so the
    figure on the card and the figure behind the row count are one measurement
    rather than two. ``None`` when there is no file to hand over at all, because
    a card with nothing to download has nothing to disclose.

    A count that cannot be computed is stated as such: with no operator channel
    configured nothing here knows which rows are the operator's, so the record
    says that and names the screen that fixes it, and the two row figures are
    absent rather than zero.
    """
    total = 0 if schedule is None else int(len(schedule))
    if not total:
        return None
    _, note = channel_scope.scope_frame(schedule, column="channel", channel=channel)
    if not note.get("scoped"):
        unset = note.get("reason") == channel_scope.NO_OPERATOR_CHANNEL_REASON
        return _scope_record("unknown" if unset else "unavailable", {"total": total})
    owned = int(note.get("rows_out") or 0)
    other = int(note.get("competitor_rows_excluded") or 0)
    channels = int(note.get("competitor_channels_excluded") or 0)
    figures = {"total": total, "owned": owned, "other": other, "channels": channels}
    record = _scope_record("real" if other else "owned_only", figures)
    record.update({"rows_owned": owned, "rows_other": other, "channels_other": channels})
    return record


def _fact(code: str, value: Any, value_he: Any = None) -> dict[str, str] | None:
    """One declared basis fact, or nothing at all when it cannot be computed.

    A date, a path and a channel name read the same in both languages, so the
    two value fields hold the same string unless a sentence is given.
    """
    text = "" if value is None else str(value).strip()
    if not text:
        return None
    words = BASIS_LABELS[code]
    return {
        "code": code,
        "label_en": words["en"],
        "label_he": words["he"],
        "value_en": text,
        "value_he": str(value_he).strip() if value_he else text,
    }


def _scope(code: str) -> dict[str, str] | None:
    return _fact("scope", SCOPES[code]["en"], SCOPES[code]["he"])


def _channel_scope(channel: str) -> dict[str, str] | None:
    """The channel this report is summed over, or the honest unknown.

    A money figure with no scope does not render, and a blank operator channel
    is not a scope: it means nobody has told the product which channel is the
    operator's, so the count stands and the fact says exactly that instead of
    disappearing and leaving a figure that reads as the operator's own.
    """
    return _fact("scope", channel) if str(channel or "").strip() else _scope("no_channel")


def _modified(path: Path | None) -> str | None:
    """When the file the rows come from last changed, local time with its offset."""
    if path is None or not Path(path).exists():
        return None
    return datetime.fromtimestamp(Path(path).stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _period(date_from: Any, date_to: Any) -> str | None:
    start = str(date_from or "").strip()[:10]
    end = str(date_to or "").strip()[:10]
    if not start or not end:
        return None
    return start if start == end else f"{start} - {end}"


def _basis(*facts: dict[str, str] | None) -> list[dict[str, str]]:
    return [fact for fact in facts if fact]


def owned_day_rows(schedule: pd.DataFrame, channel: str) -> pd.DataFrame:
    """The operator's own rows of the saved plan.

    The revenue forecast is grouped from exactly these rows, so this is the
    frame its row count has to be taken over.
    """
    if schedule is None or schedule.empty or "channel" not in schedule.columns:
        return schedule if schedule is not None else pd.DataFrame()
    owned = str(channel or "").strip()
    if not owned:
        return schedule
    return schedule[schedule["channel"].astype(str).str.strip() == owned]


def revenue_day_count(schedule: pd.DataFrame, channel: str) -> int:
    """The number of rows the revenue forecast download carries.

    One per distinct calendar date of the operator's own plan rows, which is the
    grouping the forecast endpoint publishes and the CSV writes.
    """
    frame = owned_day_rows(schedule, channel)
    if frame is None or frame.empty:
        return 0
    column = "date" if "date" in frame.columns else "day"
    if column not in frame.columns:
        return 0
    return int(frame[column].astype(str).nunique())


def build(
    *,
    schedule: pd.DataFrame,
    summary: dict[str, Any],
    compliance: dict[str, Any],
    channel: str,
    plan_path: Path,
    ledger_rows: int,
    ledger_status: str,
    daily_path: Path | None,
    daily_date: str | None,
    audited_files: int,
    present_files: int,
    audit_updated: str | None,
) -> list[dict[str, Any]]:
    """The five reports, each with the exact number of rows its download carries."""
    plan_rows = int(len(schedule)) if schedule is not None else 0
    plan_scope = download_scope(schedule, channel)
    period = _period(summary.get("date_from"), summary.get("date_to"))
    plan_updated = _modified(plan_path)
    checks = list(compliance.get("checks") or [])
    day_rows = revenue_day_count(schedule, channel)
    daily_name = str(daily_path).replace("\\", "/") if daily_path else None
    return [
        {
            "id": "weekly-plan",
            "title": "Weekly traffic plan",
            "status": "ready" if plan_rows else "empty",
            "rows": plan_rows,
            "owner": "Traffic",
            "unit": {"code": "weekly-plan", **UNITS["weekly-plan"]},
            # The one report whose file carries more than the operator's own
            # channel, so the one report that says so where the click is.
            **({"download_scope": plan_scope} if plan_scope else {}),
            "basis": _basis(
                _fact("period", period),
                _scope("whole_plan"),
                _fact("built_from", PLAN_FILE),
                _fact("updated", plan_updated),
            ),
        },
        {
            "id": "compliance",
            "title": "Compliance and guardrails",
            "status": compliance["status"],
            "rows": len(checks),
            "owner": "Legal / Ops",
            "unit": {"code": "compliance", **UNITS["compliance"]},
            "basis": _basis(
                _fact("period", compliance.get("effective_date")),
                _scope("plan_guardrails"),
                _fact("built_from", PLAN_FILE),
                _fact("updated", plan_updated),
            ),
        },
        {
            "id": "revenue",
            "title": "Revenue forecast",
            "status": "ready" if day_rows else "empty",
            "rows": day_rows,
            "owner": "Revenue",
            "unit": {"code": "revenue", **UNITS["revenue"]},
            "basis": _basis(
                _fact("period", period),
                _channel_scope(channel),
                _fact("built_from", PLAN_FILE),
                _fact("updated", plan_updated),
            ),
        },
        {
            "id": "daily-spots",
            "title": "Daily spot ledger",
            "status": ledger_status,
            "rows": int(ledger_rows),
            "owner": "Revenue",
            "unit": {"code": "daily-spots", **UNITS["daily-spots"]},
            "basis": _basis(
                _fact("period", daily_date),
                _scope("daily_log"),
                _fact("built_from", daily_name),
                _fact("updated", _modified(daily_path)),
            ),
        },
        {
            "id": "data-quality",
            "title": "Source file audit",
            "status": "ready" if present_files == audited_files else "attention",
            "rows": int(audited_files),
            "owner": "Data",
            "unit": {"code": "data-quality", **UNITS["data-quality"]},
            "basis": _basis(
                _scope("audited_files"),
                _fact("built_from", "/api/files"),
                _fact("updated", audit_updated),
            ),
        },
    ]
