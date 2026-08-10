"""Owner-supplied playout standards used to judge measured media facts.

The repository can implement the comparison, but it cannot invent the channel's
accepted codecs, frame rates, loudness target or approval vocabulary.  The
shipped JSON therefore contains the contract and no values.  A missing value
makes only that fact unavailable; it never quietly becomes a pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STANDARDS_PATH = ROOT / "config" / "media_standards.json"

LIST_FIELDS = (
    "accepted_container_formats",
    "accepted_video_codecs",
    "accepted_frame_rates",
    "accepted_display_aspect_ratios",
    "accepted_pixel_dimensions",
    "accepted_audio_channel_layouts",
    "accepted_loudness_standards",
    "approved_states",
    "rejected_states",
)


def _texts(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def load_standards(path: Path | None = None) -> dict[str, Any]:
    """Return a normalised standards contract, or an honest empty contract."""
    target = Path(path) if path is not None else STANDARDS_PATH
    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = {}
    if not isinstance(raw, dict):
        raw = {}
    out = {key: _texts(raw.get(key)) for key in LIST_FIELDS}
    for key in ("duration_tolerance_seconds", "loudness_target_lufs", "loudness_tolerance_lu"):
        try:
            out[key] = float(raw[key]) if raw.get(key) is not None else None
        except (TypeError, ValueError):
            out[key] = None
    required_audio = raw.get("required_audio")
    out["required_audio"] = required_audio if isinstance(required_audio, bool) else None
    out["source"] = str(raw.get("source") or "").strip() or None
    out["effective_from"] = str(raw.get("effective_from") or "").strip() or None
    numeric = any(out[key] is not None for key in (
        "duration_tolerance_seconds", "loudness_target_lufs", "loudness_tolerance_lu",
    ))
    out["configured"] = bool(out["source"] and (
        any(out[key] for key in LIST_FIELDS) or numeric or out["required_audio"] is not None
    ))
    return out
