"""The read routes for the technical verdict on a commercial.

Read-only on purpose. A media row is a MEASUREMENT of a file, and this product
does not measure files: it reads what an ingest or transcode report says. Until
such a feed exists there is nothing for an operator to type here, and offering a
form would invite exactly the fabrication this piece exists to prevent, someone
marking a file verified by hand.

Every payload states how many assets are on file, so a reader can tell "verified
nothing because there is nothing to verify" from "verified and found clean".
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, Request

from kairos_api.media_store import ASSETS_PATH, FACTS, NO_FEED, NO_FEED_HE, read_assets
from kairos_api.media_standards import load_standards
from kairos_api.media_verdict import verdict_for

router = APIRouter(prefix="/api/media", tags=["media"])


def _basis() -> dict[str, Any]:
    """What a verdict was measured against, printed beside it rather than
    implied, so nobody has to read this module to know what verified means."""
    standards = load_standards()
    return {
        "standards": standards,
        "configured": standards["configured"],
        "source_file": str(ASSETS_PATH.relative_to(ASSETS_PATH.parent.parent)),
        "note": "Owner-supplied playout standards. An empty field is unavailable, never an implicit pass.",
        "note_he": "תקני שידור שסופקו על ידי הבעלים. שדה ריק אינו זמין ולעולם אינו אישור משתמע.",
    }


@router.get("/assets")
def assets(request: Request = None) -> dict[str, Any]:
    """Every inspected media file. Header-only today, and it says so."""
    rows = read_assets()
    body: dict[str, Any] = {
        "assets": rows,
        "count": len(rows),
        "facts": list(FACTS),
        "basis": _basis(),
        "available": bool(rows),
    }
    if not rows:
        body["reason"], body["reason_he"] = NO_FEED, NO_FEED_HE
    return body


@router.get("/verdict/{house_number}")
def verdict(house_number: str, booked_seconds: Optional[float] = None,
            request: Request = None) -> dict[str, Any]:
    """The technical facts and the verdict over them for one commercial.

    ``booked_seconds`` is optional because the caller may not have it; when it is
    absent the duration fact reads unavailable rather than inventing a comparison.
    """
    body = verdict_for(house_number, booked_seconds)
    body["basis"] = _basis()
    return body
