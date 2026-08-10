"""The technical verdict on one commercial, and whether it may go to air.

Four facts, each tri-state, and a verdict that is the worst of them. The rule
that shapes every branch here: **UNAVAILABLE IS NOT A PASS AND IT IS NOT A
FAILURE.** A file nobody has inspected has not been cleared, so the verdict
cannot read `verified`; but it has not been found wrong either, so it must not
read `failed` and must not be reported as a corrupt file. It is its own state and
it names the feed that would settle it.

The blocking rule follows from that and is deliberately narrow: **only a MEASURED
failure blocks the lock.** If absence blocked, nothing could ever be locked
today, because the media store ships header-only; the bar would be met by a
product that simply refuses everything, which is not the job JS-8 describes.
"""

from __future__ import annotations

from typing import Any

from kairos_api.media_store import (
    FACTS,
    FAILED,
    NO_FEED,
    NO_FEED_HE,
    UNAVAILABLE,
    VERIFIED,
    assets_by_creative,
)

# What a commercial is expected to be. These are the operator's own house
# standards rather than a regulator's figures, and they are stated here so a
# reader can see exactly what "verified" was measured against.
EXPECTED_FORMATS = ("mxf", "mov", "mp4")
EXPECTED_ASPECT = "16:9"

REASONS = {
    "format": ("The container format is not one the playout chain accepts.",
               "פורמט המכל אינו אחד מאלה ששרשרת השידור מקבלת."),
    "aspect_ratio": ("The frame shape is not the one this channel airs.",
                     "צורת הפריים אינה זו שהערוץ הזה משדר."),
    "audio": ("The file carries no audio track.",
              "הקובץ אינו נושא ערוץ שמע."),
    "duration": ("The file's measured duration disagrees with the booked duration.",
                 "משך הקובץ שנמדד אינו תואם את המשך שהוזמן."),
}


def _fact(state: str, detail: dict[str, Any] | None = None, key: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"state": state}
    if detail:
        out.update(detail)
    if state == UNAVAILABLE:
        out["reason"], out["reason_he"] = NO_FEED, NO_FEED_HE
    elif state == FAILED and key in REASONS:
        out["reason"], out["reason_he"] = REASONS[key]
    return out


def _duration_fact(asset: dict[str, Any], booked_seconds: float | None) -> dict[str, Any]:
    measured = asset.get("duration_seconds")
    if measured is None:
        return _fact(UNAVAILABLE)
    if booked_seconds is None:
        # The file was measured and there is nothing to measure it against. That
        # is not a failure of the file, so it does not block.
        return _fact(UNAVAILABLE, {"measured_seconds": measured})
    detail = {"measured_seconds": round(float(measured), 1),
              "booked_seconds": round(float(booked_seconds), 1)}
    if round(float(measured), 1) == round(float(booked_seconds), 1):
        return _fact(VERIFIED, detail)
    detail["difference_seconds"] = round(abs(float(measured) - float(booked_seconds)), 1)
    return _fact(FAILED, detail, "duration")


def verdict_for(creative_id: str, booked_seconds: float | None = None,
                assets: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    """The four facts and the verdict over them, for one commercial."""
    table = assets_by_creative() if assets is None else assets
    asset = table.get(str(creative_id or "").strip())
    if asset is None:
        facts = {name: _fact(UNAVAILABLE) for name in FACTS}
        return {"creative_id": creative_id, "state": UNAVAILABLE, "blocks_lock": False,
                "facts": facts, "reason": NO_FEED, "reason_he": NO_FEED_HE}

    fmt = (asset.get("container_format") or "").lower()
    aspect = asset.get("aspect_ratio") or ""
    audio = asset.get("has_audio")

    facts = {
        "duration": _duration_fact(asset, booked_seconds),
        "format": _fact(UNAVAILABLE) if not fmt else _fact(
            VERIFIED if fmt in EXPECTED_FORMATS else FAILED, {"container_format": fmt}, "format"),
        "aspect_ratio": _fact(UNAVAILABLE) if not aspect else _fact(
            VERIFIED if aspect == EXPECTED_ASPECT else FAILED, {"aspect_ratio": aspect}, "aspect_ratio"),
        "audio": _fact(UNAVAILABLE) if audio is None else _fact(
            VERIFIED if audio else FAILED, {"has_audio": audio}, "audio"),
    }

    states = [fact["state"] for fact in facts.values()]
    if FAILED in states:
        state = FAILED
    elif all(value == VERIFIED for value in states):
        state = VERIFIED
    else:
        state = UNAVAILABLE

    out: dict[str, Any] = {
        "creative_id": creative_id,
        "state": state,
        # ONLY A MEASURED FAILURE BLOCKS. Absence must not, or nothing could ever
        # be locked while the store is header-only, and a product that refuses
        # everything has not done the job.
        "blocks_lock": state == FAILED,
        "facts": facts,
        "measured_at": asset.get("measured_at") or None,
        "source": asset.get("source") or None,
    }
    if state == UNAVAILABLE:
        out["reason"], out["reason_he"] = NO_FEED, NO_FEED_HE
    return out


def verdicts_for(spots: list[dict[str, Any]]) -> dict[str, Any]:
    """Every spot in a pod, plus what the pod's lock should do about them.

    ``spots`` are the pod board's own spot records, so this reads the creative id
    and booked duration already computed there rather than re-deriving either.
    """
    table = assets_by_creative()
    rows = []
    for item in spots:
        creative = item.get("creative") or {}
        duration = item.get("duration") or {}
        rows.append(verdict_for(
            str(creative.get("value") or "").strip(),
            duration.get("seconds") if isinstance(duration, dict) else None,
            table,
        ))
    blocking = [row for row in rows if row["blocks_lock"]]
    counts = {state: sum(1 for row in rows if row["state"] == state)
              for state in (VERIFIED, FAILED, UNAVAILABLE)}
    return {
        "spots": rows,
        "counts": counts,
        "blocks_lock": bool(blocking),
        "blocking_creatives": [row["creative_id"] for row in blocking],
        "assets_on_file": len(table),
    }
