"""Technical verdict for a measured broadcast asset, joined by House Number.

Measurements and standards are separate inputs. Missing measurement or an
unconfigured house rule is ``unavailable``; a measured mismatch is ``failed``;
only a complete match is ``verified``. Only ``failed`` blocks finalisation.
"""

from __future__ import annotations

from typing import Any

from kairos_api.media_standards import load_standards
from kairos_api.media_store import (
    FACTS, FAILED, NO_FEED, NO_FEED_HE, UNAVAILABLE, VERIFIED,
    assets_by_house_number,
)

REASONS = {
    "duration": ("The measured file duration does not match the booking.", "משך הקובץ שנמדד אינו תואם את ההזמנה."),
    "container": ("The container is not accepted by the configured playout standard.", "המכל אינו מתקבל לפי תקן השידור שהוגדר."),
    "codec": ("The video codec is not accepted by the configured playout standard.", "קודק הווידאו אינו מתקבל לפי תקן השידור שהוגדר."),
    "frame_rate": ("The frame rate is not accepted by the configured playout standard.", "קצב הפריימים אינו מתקבל לפי תקן השידור שהוגדר."),
    "frame_shape": ("The frame shape does not match the configured playout standard.", "צורת הפריים אינה תואמת את תקן השידור שהוגדר."),
    "audio": ("The audio or channel layout does not match the configured playout standard.", "השמע או פריסת הערוצים אינם תואמים את תקן השידור שהוגדר."),
    "loudness": ("The measured loudness is outside the configured tolerance.", "עוצמת השמע שנמדדה נמצאת מחוץ לטווח שהוגדר."),
    "approval": ("The media workflow rejected this asset.", "תהליך אישור המדיה דחה את הנכס הזה."),
}
INCOMPLETE = (
    "The file has not been fully checked: a measurement or owner-supplied playout rule is missing.",
    "בדיקת הקובץ אינה מלאה: חסרה מדידה או שחסר כלל שידור שסופק על ידי הבעלים.",
)


def _fact(state: str, key: str, detail: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"state": state}
    if detail:
        out.update(detail)
    if state == FAILED:
        out["reason"], out["reason_he"] = REASONS[key]
    elif state == UNAVAILABLE:
        out["reason"], out["reason_he"] = INCOMPLETE
    return out


def _normal(value: Any) -> str:
    return str(value or "").strip().lower()


def _listed(key: str, value: Any, accepted: list[str], detail: dict[str, Any]) -> dict[str, Any]:
    measured, allowed = _normal(value), {_normal(item) for item in accepted}
    if not measured or not allowed:
        return _fact(UNAVAILABLE, key, detail)
    return _fact(VERIFIED if measured in allowed else FAILED, key, detail)


def _duration(asset: dict[str, Any], booked: float | None, rules: dict[str, Any]) -> dict[str, Any]:
    seconds, frames, rate = (asset.get(key) for key in ("duration_seconds", "duration_frames", "frame_rate"))
    tolerance = rules.get("duration_tolerance_seconds")
    detail = {"measured_seconds": seconds, "duration_frames": frames, "frame_rate": rate,
              "booked_seconds": booked, "tolerance_seconds": tolerance}
    if None in (seconds, frames, rate, booked, tolerance):
        return _fact(UNAVAILABLE, "duration", detail)
    booking_difference = abs(float(seconds) - float(booked))
    frame_seconds = float(frames) / float(rate) if float(rate) else None
    frame_difference = None if frame_seconds is None else abs(frame_seconds - float(seconds))
    detail.update({"frame_seconds": frame_seconds, "difference_seconds": round(booking_difference, 4),
                   "frame_consistency_difference_seconds": None if frame_difference is None else round(frame_difference, 4)})
    ok = frame_difference is not None and max(booking_difference, frame_difference) <= float(tolerance)
    return _fact(VERIFIED if ok else FAILED, "duration", detail)


def _frame_shape(asset: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    width, height, aspect = asset.get("pixel_width"), asset.get("pixel_height"), asset.get("display_aspect_ratio")
    dimensions = f"{int(width)}x{int(height)}" if width is not None and height is not None else ""
    detail = {"pixel_dimensions": dimensions or None, "display_aspect_ratio": aspect}
    dimensions_ok = {_normal(item) for item in rules["accepted_pixel_dimensions"]}
    aspects_ok = {_normal(item) for item in rules["accepted_display_aspect_ratios"]}
    if not dimensions or not aspect or not dimensions_ok or not aspects_ok:
        return _fact(UNAVAILABLE, "frame_shape", detail)
    ok = _normal(dimensions) in dimensions_ok and _normal(aspect) in aspects_ok
    return _fact(VERIFIED if ok else FAILED, "frame_shape", detail)


def _audio(asset: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    present, layout, required = asset.get("audio_present"), asset.get("audio_channel_layout"), rules.get("required_audio")
    allowed = {_normal(item) for item in rules["accepted_audio_channel_layouts"]}
    detail = {"audio_present": present, "audio_channel_layout": layout, "required_audio": required}
    if present is None or required is None:
        return _fact(UNAVAILABLE, "audio", detail)
    if required and not present:
        return _fact(FAILED, "audio", detail)
    if not present and not required:
        return _fact(VERIFIED, "audio", detail)
    if not layout or not allowed:
        return _fact(UNAVAILABLE, "audio", detail)
    return _fact(VERIFIED if _normal(layout) in allowed else FAILED, "audio", detail)


def _loudness(asset: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    measured, standard = asset.get("loudness_lufs"), asset.get("loudness_standard")
    target, tolerance = rules.get("loudness_target_lufs"), rules.get("loudness_tolerance_lu")
    allowed = {_normal(item) for item in rules["accepted_loudness_standards"]}
    detail = {"measured_lufs": measured, "loudness_standard": standard,
              "target_lufs": target, "tolerance_lu": tolerance}
    if None in (measured, target, tolerance) or not standard or not allowed:
        return _fact(UNAVAILABLE, "loudness", detail)
    ok = _normal(standard) in allowed and abs(float(measured) - float(target)) <= float(tolerance)
    return _fact(VERIFIED if ok else FAILED, "loudness", detail)


def _approval(asset: dict[str, Any], rules: dict[str, Any]) -> dict[str, Any]:
    state = _normal(asset.get("approval_state"))
    approved = {_normal(item) for item in rules["approved_states"]}
    rejected = {_normal(item) for item in rules["rejected_states"]}
    detail = {"approval_state": asset.get("approval_state"),
              "approval_authority": asset.get("approval_authority"), "approved_at": asset.get("approved_at")}
    if not state:
        return _fact(UNAVAILABLE, "approval", detail)
    if rejected and state in rejected:
        return _fact(FAILED, "approval", detail)
    return _fact(VERIFIED if approved and state in approved else UNAVAILABLE, "approval", detail)


def verdict_for(house_number: str, booked_seconds: float | None = None,
                assets: dict[str, dict[str, Any]] | None = None,
                standards: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return all independent facts and their worst state for one House Number."""
    house = str(house_number or "").strip()
    table = assets_by_house_number() if assets is None else assets
    rules = load_standards() if standards is None else standards
    asset = table.get(house)
    if asset is None:
        facts = {name: {"state": UNAVAILABLE, "reason": NO_FEED, "reason_he": NO_FEED_HE} for name in FACTS}
        return {"house_number": house, "state": UNAVAILABLE, "blocks_lock": False,
                "facts": facts, "reason": NO_FEED, "reason_he": NO_FEED_HE}
    facts = {
        "duration": _duration(asset, booked_seconds, rules),
        "container": _listed("container", asset.get("container_format"), rules["accepted_container_formats"], {"container_format": asset.get("container_format")}),
        "codec": _listed("codec", asset.get("video_codec"), rules["accepted_video_codecs"], {"video_codec": asset.get("video_codec")}),
        "frame_rate": _listed("frame_rate", asset.get("frame_rate"), rules["accepted_frame_rates"], {"frame_rate": asset.get("frame_rate")}),
        "frame_shape": _frame_shape(asset, rules),
        "audio": _audio(asset, rules),
        "loudness": _loudness(asset, rules),
        "approval": _approval(asset, rules),
    }
    states = [fact["state"] for fact in facts.values()]
    state = FAILED if FAILED in states else VERIFIED if all(item == VERIFIED for item in states) else UNAVAILABLE
    out = {"house_number": house, "state": state, "blocks_lock": state == FAILED,
           "facts": facts, "measured_at": asset.get("measured_at") or None, "source": asset.get("source") or None}
    if state == UNAVAILABLE:
        out["reason"], out["reason_he"] = INCOMPLETE
    return out


def verdicts_for(spots: list[dict[str, Any]]) -> dict[str, Any]:
    """Join the traffic row to the measured file on House Number, never version name."""
    table, rules, rows = assets_by_house_number(), load_standards(), []
    for item in spots:
        house, duration = item.get("house_number") or {}, item.get("duration") or {}
        rows.append(verdict_for(house.get("value"), duration.get("seconds"), table, rules))
    blocking = [row for row in rows if row["blocks_lock"]]
    return {
        "spots": rows,
        "counts": {state: sum(row["state"] == state for row in rows) for state in (VERIFIED, FAILED, UNAVAILABLE)},
        "blocks_lock": bool(blocking),
        "blocking_house_numbers": [row["house_number"] for row in blocking],
        "assets_on_file": len(table),
        "standards_configured": bool(rules.get("configured")),
    }


def lock_refusal(media: dict[str, Any] | None) -> str | None:
    """Server-side finalisation gate; UI state is never the authority."""
    block = media or {}
    if not block.get("blocks_lock"):
        return None
    houses = ", ".join(str(item) for item in block.get("blocking_house_numbers") or [])
    return f"This pod cannot be locked because measured media verification failed for House Number: {houses or 'unknown'}."
