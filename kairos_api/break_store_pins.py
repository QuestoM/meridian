"""The register of break placements an operator saved, on ``data/breaks.csv``.

A saved move has to reach the engine or it is theatre, and the only store the
weekly commit path reads for a placement is the scoped constraint store. So a
save writes a constraint, and this file is the break's own record of it: which
break it was, where it was put, who put it there, when, and which constraint now
carries it.

That record buys three things nothing else in the product has.

* **An exact undo.** Undo deletes the constraint this row names, so reversing a
  save never has to guess which of several constraints was the one just written.
* **Provenance on the board.** A break can say whether its position is the
  optimizer's own or the operator's, which is the difference between a
  suggestion and a decision.
* **An audit line per break.** The override store records segment-level acts and
  the constraint store records rules; neither records that a person moved one
  break at 21:42 into one programme on one date.

The register is written with the same discipline as every other operator store
in this package: one lock over load, mutate and write, a backup before the
write, and a temp file plus ``os.replace`` so a reader never sees a torn CSV.
Nothing here prices anything or moves any figure.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
BREAKS_PATH = DATA_DIR / "breaks.csv"

COLUMNS = (
    "break_id",
    "segment_id",
    "ordinal",
    "channel",
    "day",
    "programme",
    "offset_seconds",
    "duration_seconds",
    "is_gold",
    "constraint_id",
    "actor",
    "saved_at",
    "note",
)

_STORE_LOCK = threading.Lock()


def _load_frame() -> pd.DataFrame:
    if not BREAKS_PATH.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    try:
        frame = pd.read_csv(BREAKS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception:
        logger.exception("break register read failed")
        return pd.DataFrame(columns=list(COLUMNS))
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _write_frame(frame: pd.DataFrame) -> None:
    if BREAKS_PATH.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        shutil.copy2(BREAKS_PATH, BACKUP_DIR / f"breaks_{stamp}.csv")
    BREAKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = BREAKS_PATH.with_name(BREAKS_PATH.name + ".tmp")
    frame[list(COLUMNS)].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, BREAKS_PATH)


def _record(row: "pd.Series[Any]") -> dict[str, Any]:
    return {column: str(row.get(column, "")) for column in COLUMNS}


def records() -> list[dict[str, Any]]:
    """Every saved placement, oldest first, exactly as stored."""
    frame = _load_frame()
    return [_record(row) for _, row in frame.iterrows()]


def for_day(day: str) -> dict[str, dict[str, Any]]:
    """The saved placements on one broadcast day, keyed by break id."""
    wanted = str(day or "").strip()
    return {
        record["break_id"]: record
        for record in records()
        if record.get("day", "").strip() == wanted and record.get("break_id")
    }


def save(record: dict[str, Any]) -> dict[str, Any]:
    """Record one saved placement, replacing any earlier record of that break.

    One break carries at most one saved placement, because a second save of the
    same break is the same decision restated, not a second decision.
    """
    row = {column: str(record.get(column, "") or "") for column in COLUMNS}
    row["saved_at"] = row["saved_at"] or datetime.now(timezone.utc).isoformat()
    with _STORE_LOCK:
        frame = _load_frame()
        if not frame.empty:
            frame = frame[frame["break_id"].astype(str) != row["break_id"]].reset_index(drop=True)
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        _write_frame(frame)
    return row


def forget(break_id: str) -> Optional[dict[str, Any]]:
    """Drop one break's saved placement and return the record that was dropped."""
    wanted = str(break_id or "").strip()
    with _STORE_LOCK:
        frame = _load_frame()
        if frame.empty:
            return None
        mask = frame["break_id"].astype(str) == wanted
        if not mask.any():
            return None
        dropped = _record(frame[mask].iloc[0])
        frame = frame[~mask].reset_index(drop=True)
        _write_frame(frame)
    return dropped
