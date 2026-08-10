"""Validated, atomic import seam for a real ingest/transcode/QC report."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from kairos_api.media_store import ASSETS_PATH, COLUMNS, read_assets, write_assets

ALIASES = {
    "creative_id": "house_number",
    "aspect_ratio": "display_aspect_ratio",
    "has_audio": "audio_present",
    "codec": "video_codec",
    "audio_channels": "audio_channel_layout",
    "qc_state": "approval_state",
}
MEASURED_FIELDS = tuple(key for key in COLUMNS if key not in {
    "house_number", "measured_at", "source", "approval_authority", "approved_at",
})


class MediaReportError(ValueError):
    """The report cannot be safely joined or contains no measurements."""


def _normalise(row: dict[str, Any], source: str, row_number: int) -> dict[str, Any]:
    out = {ALIASES.get(str(key).strip(), str(key).strip()): value for key, value in row.items() if key}
    house = str(out.get("house_number") or "").strip()
    if not house:
        raise MediaReportError(f"row {row_number} has no House Number")
    out["house_number"] = house
    if source:
        out["source"] = str(out.get("source") or source).strip()
    if not any(str(out.get(key) or "").strip() for key in MEASURED_FIELDS):
        raise MediaReportError(f"row {row_number} for {house} contains no measured media fact")
    return out


def import_report(report: Path, store: Path | None = None, source: str = "") -> dict[str, Any]:
    """Upsert a CSV report by House Number; duplicate rows in the report fail."""
    report = Path(report)
    target = Path(store) if store is not None else ASSETS_PATH
    with report.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        incoming = [_normalise(row, source or report.name, index) for index, row in enumerate(reader, start=2)]
    seen: set[str] = set()
    for row in incoming:
        house = row["house_number"]
        if house in seen:
            raise MediaReportError(f"report contains House Number {house} more than once")
        seen.add(house)
    existing = {row["house_number"]: row for row in read_assets(target)}
    inserted = sum(1 for house in seen if house not in existing)
    existing.update({row["house_number"]: row for row in incoming})
    write_assets(existing.values(), target)
    return {
        "source": str(report),
        "store": str(target),
        "received": len(incoming),
        "inserted": inserted,
        "updated": len(incoming) - inserted,
        "total": len(existing),
    }
