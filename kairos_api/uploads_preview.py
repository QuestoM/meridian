"""The rows behind a row count, split out of ``uploads.py``.

A source card prints a row count, and a count that cannot be opened is a dead
end. This module answers "show me what is actually in that file" from the file
itself: the real header, the first rows, the real total, and how many of those
rows the reader is not being shown.

**The competitor boundary applies here and it is not theoretical.** Three of
the seven inputs carry rival channels in the file the operator uploaded: the
programme lineup and the historical spots carry a ``Channel`` column, and the
rate card carries a ``channel`` column whose shipped rows include channels this
operator does not own. The dayparts file carries one column per channel. So a
preview is scoped through :mod:`kairos_api.channel_scope`, which takes the one
owned channel from settings, and the disclosure that travels with it counts
what was excluded without naming any of it.

Nothing is fabricated: a missing value renders as an empty cell, the total is
the file's own row count, and a file that cannot be read says so rather than
returning an empty table that reads as an empty file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from kairos_api import channel_scope, read_cache
from kairos.data.loaders import CHANNELS

logger = logging.getLogger(__name__)

PREVIEW_NAMESPACE = "uploads.preview"
read_cache.configure(PREVIEW_NAMESPACE, capacity=16)

# The most rows a preview will ever return. A drill is a look, not an export;
# the whole file is downloadable from the report shelf.
MAX_PREVIEW_ROWS = 100
DEFAULT_PREVIEW_ROWS = 20

# The column each kind carries its channel in, when it carries one as a column.
ROW_CHANNEL_COLUMN: dict[str, str] = {
    "programmes": "Channel",
    "spots": "Channel",
    "rate_card": "channel",
}

# Kinds whose channels are columns rather than rows: one column per channel.
COLUMN_CHANNEL_KINDS = frozenset({"dayparts"})

# Every kind whose file names a channel somewhere. Without a configured
# operator channel there is no way to tell which of those names is the
# operator's own, so none of them can be shown: the boundary is not "hide the
# three rivals", it is "never render a channel this account may not own".
CHANNEL_KINDS = frozenset(ROW_CHANNEL_COLUMN) | COLUMN_CHANNEL_KINDS

# Three columns in the shipped historical spots file are inverted against this
# operator: measured, ``is_target_channel`` marks קשת 12 on all 23,707 of its
# rows and ``competitor_flag`` marks רשת 13, which settings names as the
# operator's own channel. Nothing in the product reads them. Rendering them on
# the operator's own scoped rows would show their channel flagged as a
# competitor, so they are withheld and the count is disclosed.
INVERTED_COLUMNS = ("is_target_channel", "competitor_flag", "include_as_media")

# Every note this module can attach, in both languages. A surface renders the
# words it is sent, so an English-only sentence never lands inside a Hebrew
# screen.
NOTES: dict[str, dict[str, str]] = {
    "inverted_columns": {
        "en": "Three columns in this file flag the wrong channel as the operator's and nothing reads them, so they are not shown.",
        "he": "שלוש עמודות בקובץ הזה מסמנות ערוץ שגוי כערוץ של המפעיל ושום דבר לא קורא אותן, ולכן הן אינן מוצגות.",
    },
    "unreadable": {
        "en": "The file could not be read as a CSV table, so its rows cannot be shown.",
        "he": "לא ניתן היה לקרוא את הקובץ כטבלת CSV, ולכן לא ניתן להציג את השורות שלו.",
    },
    "no_file": {
        "en": "No file has been uploaded for this input yet, so there are no rows to show.",
        "he": "עדיין לא הועלה קובץ עבור הקלט הזה, ולכן אין שורות להצגה.",
    },
    "no_channel": {
        "en": "This file names a channel on every row and no operator channel is set, so there is no way to tell which rows are yours. Set the operator channel on the settings screen and these rows open.",
        "he": "הקובץ הזה נושא שם ערוץ בכל שורה ולא מוגדר ערוץ מפעיל, ולכן אין דרך לדעת אילו שורות הן שלכם. הגדירו את ערוץ המפעיל במסך ההגדרות והשורות האלה ייפתחו.",
    },
}


def note(code: str) -> dict[str, str]:
    """One note as a bilingual record the surface renders directly."""
    words = NOTES.get(code) or {"en": "", "he": ""}
    return {"code": code, "en": words["en"], "he": words["he"]}


def preview(path: Path, kind: str, limit: int = DEFAULT_PREVIEW_ROWS) -> dict[str, Any]:
    """The first ``limit`` rows of this input, scoped to the owned channel."""
    rows = max(1, min(int(limit or DEFAULT_PREVIEW_ROWS), MAX_PREVIEW_ROWS))
    owned = channel_scope.operator_channel()
    fingerprint = (read_cache.file_signature(path), owned, rows)
    held = read_cache.cached(
        PREVIEW_NAMESPACE,
        f"{kind}:{path}:{rows}",
        fingerprint,
        lambda: _build(path, kind, rows, owned),
    )
    # The cache shares values, so the caller gets its own copy of the record.
    return dict(held)


def no_file(kind: str) -> dict[str, Any]:
    """The honest answer for an input that has no file on disk to open yet."""
    return _unavailable(kind, "no_file")


def _unavailable(kind: str, code: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "available": False,
        "notes": [note(code)],
        "columns": [],
        "rows": [],
        "total_rows": 0,
        "shown_rows": 0,
        "scope": None,
        "columns_hidden": 0,
    }


def _build(path: Path, kind: str, limit: int, owned: str) -> dict[str, Any]:
    if kind in CHANNEL_KINDS and not owned:
        return _unavailable(kind, "no_channel")
    try:
        frame = pd.read_csv(path, encoding="utf-8-sig")
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        logger.warning("preview could not read %s: %s", path, exc)
        return _unavailable(kind, "unreadable")
    total_rows = int(len(frame))
    scope: dict[str, Any] | None = None
    columns_hidden = 0
    row_column = ROW_CHANNEL_COLUMN.get(kind)
    if row_column and row_column in frame.columns:
        frame, scope = channel_scope.scope_frame(frame, column=row_column, channel=owned or None)
    elif kind in COLUMN_CHANNEL_KINDS:
        frame, columns_hidden = _hide_rival_columns(frame, owned)
        scope = channel_scope.scope_note(owned, total_rows, total_rows, len(CHANNELS), scoped=bool(owned))
    scoped_rows = int(len(frame))
    inverted = [column for column in frame.columns if str(column) in INVERTED_COLUMNS]
    if inverted:
        frame = frame.drop(columns=inverted)
    head = frame.head(limit)
    return {
        "kind": kind,
        "available": True,
        "notes": [note("inverted_columns")] if inverted else [],
        "columns_withheld": len(inverted),
        "columns": [str(column) for column in head.columns],
        "rows": [[_cell(value) for value in record] for record in head.itertuples(index=False, name=None)],
        "total_rows": total_rows,
        "scoped_rows": scoped_rows,
        "shown_rows": int(len(head)),
        "scope": scope,
        "columns_hidden": columns_hidden,
    }


def _hide_rival_columns(frame: pd.DataFrame, owned: str) -> tuple[pd.DataFrame, int]:
    """Drop one column per channel the operator does not own, and count them.

    The count is the disclosure; the names are exactly what the boundary
    exists to keep off an operator surface.
    """
    if not owned:
        return frame, 0
    rivals = [column for column in frame.columns if str(column) in CHANNELS and str(column) != owned]
    if not rivals:
        return frame, 0
    return frame.drop(columns=rivals), len(rivals)


def _cell(value: Any) -> str:
    """One cell as text, with a missing value left genuinely empty."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)
