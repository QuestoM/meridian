"""The technical facts about a commercial's actual file, and the store holding them.

WHY THIS SHIPS EMPTY, AND WHY THAT IS THE HONEST SHAPE. The verdict this piece
prints is about the FILE that will air: its measured duration, its format, its
frame shape and whether it carries audio. Nothing in this repository observes a
media file. There is no media feed, no transcode report and no ingest log, so
today every one of those four facts is genuinely UNAVAILABLE, and the one thing
this module must never do is invent one. A fabricated "verified" here is worse
than a blank: it would clear a corrupt file to air.

So `data/media_assets.csv` is a header and no rows, exactly as
`campaign_flights.csv` is, and the surface says which feed is missing and how to
supply it. The moment a real row lands the verdict computes for real, per asset,
with no code change. The test proves both directions rather than only the empty
one, because an empty store makes every assertion pass for free.

WHAT THIS DELIBERATELY DOES NOT DO. It does not re-check the copy length. That
check already exists, is tri-state, and is measured on the pod board:
``break_api_pod_spots.copy_length_check`` compares the copy version's own
declared length against the booked duration. This module answers a DIFFERENT
question, the file against the booking, and when a media row is absent it says so
rather than silently borrowing the other check's answer and presenting a
copy-versus-booking agreement as if the file had been inspected.
"""

from __future__ import annotations

import csv
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Iterable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ASSETS_PATH = DATA_DIR / "media_assets.csv"

# The order the file is written in. Anything else a row carries is kept and
# appended, per the store-column rule: the writer is not the authority on which
# columns exist, because migrations and seeds add columns and never edit writers.
COLUMNS = (
    "creative_id",
    "duration_seconds",
    "container_format",
    "aspect_ratio",
    "has_audio",
    "measured_at",
    "source",
)

# The four technical facts a verdict is made of, each answerable only from a real
# media row. Named once here so the API cannot drift from the store.
FACTS = ("duration", "format", "aspect_ratio", "audio")

# Tri-state, and never a fourth. `unavailable` is not a failure: it means nobody
# has measured this file, which is a different thing from measuring it and
# finding it wrong, and clearing it to air on that basis would be the defect.
VERIFIED = "verified"
FAILED = "failed"
UNAVAILABLE = "unavailable"

NO_FEED = (
    "No media file has been inspected for this commercial. "
    "data/media_assets.csv is header-only: no ingest or transcode report is connected yet."
)
NO_FEED_HE = (
    "לא נבדק שום קובץ מדיה עבור התשדיר הזה. "
    "הקובץ data/media_assets.csv מכיל כותרת בלבד: עדיין לא חובר דוח קליטה או המרה."
)

_LOCK = threading.RLock()


def _text(row: dict[str, Any], key: str) -> str:
    value = row.get(key)
    return "" if value is None else str(value).strip()


def _number(row: dict[str, Any], key: str) -> float | None:
    raw = _text(row, key)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        # A malformed figure is UNKNOWN, never zero. A zero-second commercial and
        # an unparseable one must not read the same, because one of them is a
        # real defect in the file and the other is a defect in the row.
        return None


def _flag(row: dict[str, Any], key: str) -> bool | None:
    raw = _text(row, key).lower()
    if raw in ("1", "true", "yes", "y"):
        return True
    if raw in ("0", "false", "no", "n"):
        return False
    return None


def read_assets(path: Path | None = None) -> list[dict[str, Any]]:
    """Every media row on disk, normalised. An absent file reads as no rows."""
    target = Path(path) if path is not None else ASSETS_PATH
    if not target.exists():
        return []
    with _LOCK, target.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assets = []
    for row in rows:
        creative = _text(row, "creative_id")
        if not creative:
            continue
        asset = dict(row)
        asset["creative_id"] = creative
        asset["duration_seconds"] = _number(row, "duration_seconds")
        asset["has_audio"] = _flag(row, "has_audio")
        asset["container_format"] = _text(row, "container_format") or None
        asset["aspect_ratio"] = _text(row, "aspect_ratio") or None
        assets.append(asset)
    return assets


def assets_by_creative(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The rows keyed for lookup. A later row for the same creative wins, so a
    re-ingest supersedes rather than duplicating."""
    return {asset["creative_id"]: asset for asset in read_assets(path)}


def write_assets(rows: Iterable[dict[str, Any]], path: Path | None = None) -> Path:
    """Atomic write: a sibling temp file, fsync, then replace, so a reader can
    never see a half-written store."""
    target = Path(path) if path is not None else ASSETS_PATH
    rows = list(rows)
    extra: list[str] = []
    for row in rows:
        for key in row:
            if key not in COLUMNS and key not in extra:
                extra.append(key)
    header = list(COLUMNS) + extra
    with _LOCK:
        target.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=str(target.parent), delete=False
        )
        try:
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: row.get(key, "") for key in header})
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, target)
    return target
