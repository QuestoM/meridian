"""The technical facts about a commercial's actual file, and the store holding them.

WHY THIS SHIPS EMPTY, AND WHY THAT IS THE HONEST SHAPE. The verdict this piece
prints is about the FILE that will air: its measured duration and frame count,
format and codec, frame shape, audio, loudness and approval. Nothing here observes a
media file. There is no media feed, no transcode report and no ingest log, so
today every one of those measurement families is genuinely UNAVAILABLE, and the one thing
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
    "house_number",
    "duration_seconds",
    "duration_frames",
    "frame_rate",
    "container_format",
    "video_codec",
    "pixel_width",
    "pixel_height",
    "display_aspect_ratio",
    "audio_present",
    "audio_channel_layout",
    "loudness_lufs",
    "loudness_standard",
    "approval_state",
    "approval_authority",
    "approved_at",
    "measured_at",
    "source",
)

# The technical facts a verdict is made of, each answerable only from a real
# media row. Named once here so the API cannot drift from the store.
FACTS = ("duration", "container", "codec", "frame_rate", "frame_shape", "audio", "loudness", "approval")

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


def _rate(row: dict[str, Any], key: str) -> float | None:
    """A decimal or the rational form ffprobe emits, such as ``25/1``."""
    raw = _text(row, key)
    if "/" not in raw:
        return _number(row, key)
    left, right = raw.split("/", 1)
    try:
        denominator = float(right)
        return None if denominator == 0 else float(left) / denominator
    except ValueError:
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
        # ``creative_id`` is accepted only as a migration alias for the first
        # header-only implementation.  Every returned and newly written row is
        # keyed by the traffic industry's own identifier: House Number.
        house = _text(row, "house_number") or _text(row, "creative_id")
        if not house:
            continue
        asset = dict(row)
        asset["house_number"] = house
        asset.pop("creative_id", None)
        asset["duration_seconds"] = _number(row, "duration_seconds")
        asset["duration_frames"] = _number(row, "duration_frames")
        asset["frame_rate"] = _rate(row, "frame_rate")
        asset["pixel_width"] = _number(row, "pixel_width")
        asset["pixel_height"] = _number(row, "pixel_height")
        asset["loudness_lufs"] = _number(row, "loudness_lufs")
        asset["audio_present"] = _flag(row, "audio_present")
        asset["container_format"] = _text(row, "container_format") or None
        asset["video_codec"] = _text(row, "video_codec") or None
        asset["display_aspect_ratio"] = _text(row, "display_aspect_ratio") or None
        asset["audio_channel_layout"] = _text(row, "audio_channel_layout") or None
        asset["loudness_standard"] = _text(row, "loudness_standard") or None
        asset["approval_state"] = _text(row, "approval_state") or None
        assets.append(asset)
    return assets


def assets_by_house_number(path: Path | None = None) -> dict[str, dict[str, Any]]:
    """The rows keyed for lookup. A later row for the same asset wins, so a
    re-ingest supersedes rather than duplicating."""
    return {asset["house_number"]: asset for asset in read_assets(path)}


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
