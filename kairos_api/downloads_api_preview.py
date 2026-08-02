"""The rows behind a report's row count, for the two reports the server streams.

Split out of ``downloads_api.py`` under the file-size cap and named by the
``<parent stem>_<role>.py`` rule.

A report card prints a row count and tells the reader it is the exact number of
rows the download will carry. A count that cannot be opened is a claim nobody
can check, so this module answers "show me those rows" from the same source the
download reads: the saved plan frame for the weekly plan, and the daily pricing
ledger for the spot ledger. The other three reports are built in the browser
from a live endpoint, and their preview reads that same endpoint, so every
preview in the product comes from the same place as the file it describes.

**The boundary applies to the preview and not to the download.** The weekly plan
file has carried every channel since long before this rebuild and Bar 3 freezes
it at 8,704 rows. A preview is a screen, so it is scoped to the one owned
channel and discloses the count it withheld without naming any of it.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from kairos_api import channel_scope

logger = logging.getLogger(__name__)

# A preview is a look, not an export. The whole file is one click away on the
# same card.
MAX_PREVIEW_ROWS = 100
DEFAULT_PREVIEW_ROWS = 20

# The two reports whose rows this module can serve. The other three name the
# endpoint their rows come from instead, so nothing here pretends to know a
# payload it does not build.
SERVED = ("weekly-plan", "daily-spots")

CLIENT_SOURCES: dict[str, str] = {
    "compliance": "/api/compliance",
    "revenue": "/api/forecasts",
    "data-quality": "/api/files",
}

NOTES: dict[str, dict[str, str]] = {
    "scoped": {
        "en": "Rows on other channels are not shown. The download carries the whole plan file.",
        "he": "שורות של ערוצים אחרים אינן מוצגות. ההורדה נושאת את כל קובץ התוכנית.",
    },
    "no_rows": {
        "en": "This report has no rows yet, so there is nothing to show.",
        "he": "לדוח הזה אין עדיין שורות, ולכן אין מה להציג.",
    },
    "unavailable": {
        "en": "The rows behind this report could not be read.",
        "he": "לא ניתן היה לקרוא את השורות שמאחורי הדוח הזה.",
    },
    "no_channel": {
        "en": "Every row of the plan names a channel and no operator channel is set, so there is no way to tell which rows are yours. Set the operator channel on the settings screen and these rows open.",
        "he": "כל שורה בתוכנית נושאת שם ערוץ ולא מוגדר ערוץ מפעיל, ולכן אין דרך לדעת אילו שורות הן שלכם. הגדירו את ערוץ המפעיל במסך ההגדרות והשורות האלה ייפתחו.",
    },
}


def note(code: str) -> dict[str, str]:
    words = NOTES.get(code) or {"en": "", "he": ""}
    return {"code": code, "en": words["en"], "he": words["he"]}


def _cell(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _empty(report_id: str, code: str) -> dict[str, Any]:
    return {
        "report_id": report_id,
        "available": False,
        "columns": [],
        "rows": [],
        "total_rows": 0,
        "scoped_rows": 0,
        "shown_rows": 0,
        "scope": None,
        "notes": [note(code)],
    }


def _frame_preview(report_id: str, frame: pd.DataFrame, limit: int, scoped: bool) -> dict[str, Any]:
    total = int(len(frame))
    scope: dict[str, Any] | None = None
    notes: list[dict[str, str]] = []
    if scoped and "channel" in frame.columns:
        frame, scope = channel_scope.scope_frame(frame, column="channel")
        if scope and int(scope.get("competitor_rows_excluded") or 0) > 0:
            notes.append(note("scoped"))
    shown = frame.head(limit)
    return {
        "report_id": report_id,
        "available": True,
        "columns": [str(column) for column in shown.columns],
        "rows": [[_cell(value) for value in record] for record in shown.itertuples(index=False, name=None)],
        "total_rows": total,
        "scoped_rows": int(len(frame)),
        "shown_rows": int(len(shown)),
        "scope": scope,
        "notes": notes,
    }


def weekly_plan(schedule: pd.DataFrame, limit: int) -> dict[str, Any]:
    """The saved plan's own rows, scoped to the owned channel for the screen."""
    if schedule is None or schedule.empty:
        return _empty("weekly-plan", "no_rows")
    if "channel" in schedule.columns and not channel_scope.operator_channel():
        return _empty("weekly-plan", "no_channel")
    return _frame_preview("weekly-plan", schedule, limit, scoped=True)


def daily_spots(limit: int) -> dict[str, Any]:
    """The daily pricing ledger's own rows, exactly as the CSV writes them."""
    try:
        from kairos_api.exporters import SPOTS_COLUMNS, _load_daily_pricing, _spot_records

        result = _load_daily_pricing()
    except Exception:
        logger.exception("the daily spot ledger preview could not price the daily file")
        return _empty("daily-spots", "unavailable")
    if result is None:
        return _empty("daily-spots", "no_rows")
    records = _spot_records(result)
    if not records:
        return _empty("daily-spots", "no_rows")
    frame = pd.DataFrame(records, columns=SPOTS_COLUMNS)
    return _frame_preview("daily-spots", frame, limit, scoped=False)


def bounded(limit: Any) -> int:
    try:
        requested = int(limit)
    except (TypeError, ValueError):
        requested = DEFAULT_PREVIEW_ROWS
    return max(1, min(requested or DEFAULT_PREVIEW_ROWS, MAX_PREVIEW_ROWS))
