"""The creative assets of a campaign: the tapes, and what is known about them.

A television creative is a spot: a video with a declared length, a house number
the broadcast house files it under, and a version name the agency calls it. All
three of those are real here, because the traffic log on disk carries all three
for every airing it records. ``data/campaign_assets.csv`` is that ledger, one
row per house number per campaign.

What this product cannot do is open the video. It has no media file, no playout
asset registry and no clearance system, so the four properties that only a video
can answer are reported as **unknown** with the path to supply each one, and
never as a plausible default:

  * the file itself, as a link or an upload,
  * the video format and the aspect ratio,
  * the loudness, and the standard it is measured against,
  * the clearance verdict, which is whether the spot is fit to air.

An unknown is not a zero and it is not a blank. It crosses the wire as a record
carrying its own state, its reason and the action that would resolve it, so a
surface can render "not known yet, here is how to know" instead of an empty cell
that reads as "fine".

A creative also carries a **validity window**: until when it may be scheduled.
The trade calls that a constraint, and it is a property of one tape rather than a
relation between two, so it is a column here (``valid_from`` and ``valid_until``)
and not a rule elsewhere. The traffic log declares no window, so every row read
from it reports the window unknown with the path to supply it, exactly like the
four properties only a video can answer. The window's own arithmetic, and the
paired-creative constraint that is NOT a property of one tape, are in
:mod:`kairos_api.campaigns_assets_constraints`.

Every row the demo seed writes carries ``is_demo`` true, because the campaign it
hangs on is a seeded booking rather than a signed one. The identity half of the
row is still real and says so in ``identity_source``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import pandas as pd

from kairos_api import campaigns_assets_constraints as constraints
from kairos_api import campaigns_commitment as commitment

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ASSETS_PATH = DATA_DIR / "campaign_assets.csv"

COLUMNS = [
    "asset_id",
    "campaign_id",
    "advertiser",
    "channel",
    "house_number",
    "version_name",
    "spot_type",
    "length_class",
    "duration_seconds",
    "media_url",
    "media_state",
    "video_format",
    "aspect_ratio",
    "loudness_lufs",
    "loudness_standard",
    "clearance_verdict",
    "clearance_authority",
    "clearance_checked_at",
    "valid_from",
    "valid_until",
    "validity_source",
    "first_observed_on",
    "last_observed_on",
    "airings_observed",
    "identity_source",
    "source_file",
    "is_demo",
    "notes",
]

# The verdicts a clearance check can return. ``unknown`` is a first-class
# member, because "we have not checked" is the true state of every asset on
# this product today and it is not the same answer as "not fit to air".
CLEARANCE_VERDICTS = (
    {
        "value": "fit_to_air",
        "label_en": "Fit to air",
        "label_he": "כשיר לשידור",
    },
    {
        "value": "not_fit_to_air",
        "label_en": "Not fit to air",
        "label_he": "אינו כשיר לשידור",
    },
    {
        "value": "awaiting_check",
        "label_en": "Awaiting check",
        "label_he": "ממתין לבדיקה",
    },
    {
        "value": "unknown",
        "label_en": "Unknown",
        "label_he": "לא ידוע",
    },
)

# One entry per property this product cannot read, with the action that supplies
# it. The action is the whole point: a surface that says "unknown" and stops has
# told the operator nothing they can act on.
UNKNOWN_PATHS = {
    "media": (
        "Attach the broadcast master to this house number, or connect the playout asset registry, "
        "to read the file.",
        "צרפו את חומר השידור למספר הבית הזה, או חברו את מרשם החומרים של השידור, כדי לקרוא את הקובץ.",
    ),
    "video_format": (
        "The video format is read from the file. Attach the master to read it.",
        "פורמט הווידאו נקרא מהקובץ. צרפו את חומר השידור כדי לקרוא אותו.",
    ),
    "aspect_ratio": (
        "The aspect ratio is read from the file. Attach the master to read it.",
        "יחס הממדים נקרא מהקובץ. צרפו את חומר השידור כדי לקרוא אותו.",
    ),
    "loudness": (
        "Loudness is measured from the audio. Attach the master and declare the loudness standard "
        "the playout enforces.",
        "העוצמה נמדדת מהשמע. צרפו את חומר השידור והצהירו על תקן העוצמה שהשידור אוכף.",
    ),
    "clearance": (
        "Record the clearance verdict for this house number, or connect the clearance system, to "
        "say whether the spot is fit to air.",
        "רשמו את החלטת הכשירות עבור מספר הבית הזה, או חברו את מערכת הכשירות, כדי לומר אם התשדיר "
        "כשיר לשידור.",
    ),
    "validity": (
        "Record the last day this creative may be scheduled. The traffic log records what aired, "
        "not until when a tape may air, so no window can be read from it.",
        "רשמו את היום האחרון שבו מותר לתזמן את התשדיר הזה. קובץ הטראפיק רושם מה שודר, ולא עד מתי "
        "מותר לשדר, ולכן לא ניתן לקרוא ממנו חלון תוקף.",
    ),
}

NO_ASSETS_EN = "No creative asset is recorded for this campaign."
NO_ASSETS_HE = "לא רשום חומר שידור לקמפיין הזה."


def _text(row: Any, column: str) -> str:
    return str(row.get(column, "") or "").strip()


def _number(raw: Any) -> Optional[float]:
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        return round(float(text), 3)
    except (TypeError, ValueError):
        return None


def load_frame() -> pd.DataFrame:
    """Every asset row, or an empty frame when the ledger has never been written."""
    if not ASSETS_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(ASSETS_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def unknown_property(kind: str, value: Any = None) -> dict[str, Any]:
    """One property, in the tri-state: a real value, or unknown with the way out.

    ``state`` is ``real`` when a value is present and ``unknown`` when it is not.
    There is no third silent branch: a property this product has not read reports
    the reason it has not and the action that would let it.
    """
    path_en, path_he = UNKNOWN_PATHS.get(kind, ("", ""))
    text = str(value if value is not None else "").strip()
    if text and text.lower() != "unknown":
        return {"state": "real", "value": text, "path_en": "", "path_he": ""}
    return {"state": "unknown", "value": None, "path_en": path_en, "path_he": path_he}


def asset_record(row: Any) -> dict[str, Any]:
    """One creative, with what is known real and what is not known named as unknown."""
    demo = commitment.is_demo(row.get("is_demo"))
    verdict = _text(row, "clearance_verdict") or "unknown"
    return {
        "asset_id": _text(row, "asset_id"),
        "campaign_id": _text(row, "campaign_id"),
        "advertiser": _text(row, "advertiser"),
        "channel": _text(row, "channel"),
        # Real, from the traffic log: the broadcast house filing, the agency's
        # version name, what kind of spot it is and how long it runs.
        "house_number": _text(row, "house_number"),
        "version_name": _text(row, "version_name"),
        "spot_type": _text(row, "spot_type"),
        "length_class": _text(row, "length_class"),
        "duration_seconds": _number(row.get("duration_seconds")),
        "first_observed_on": _text(row, "first_observed_on"),
        "last_observed_on": _text(row, "last_observed_on"),
        "airings_observed": int(_number(row.get("airings_observed")) or 0),
        "identity_source": _text(row, "identity_source"),
        "source_file": _text(row, "source_file"),
        # Unknown until a video is supplied. Each carries its own way out.
        "media": unknown_property("media", _text(row, "media_url")),
        "media_state": _text(row, "media_state") or "unknown",
        "video_format": unknown_property("video_format", _text(row, "video_format")),
        "aspect_ratio": unknown_property("aspect_ratio", _text(row, "aspect_ratio")),
        "loudness": unknown_property("loudness", _text(row, "loudness_lufs")),
        "loudness_standard": unknown_property("loudness", _text(row, "loudness_standard")),
        "clearance": {
            "verdict": verdict,
            "fit_to_air": True if verdict == "fit_to_air" else (False if verdict == "not_fit_to_air" else None),
            "authority": _text(row, "clearance_authority"),
            "checked_at": _text(row, "clearance_checked_at"),
            "path_en": UNKNOWN_PATHS["clearance"][0] if verdict == "unknown" else "",
            "path_he": UNKNOWN_PATHS["clearance"][1] if verdict == "unknown" else "",
        },
        # Authored, not observed: until when this tape may be scheduled.
        "validity": constraints.validity_window(row),
        "is_demo": demo,
        "demo": commitment.demo_block(demo),
        "notes": _text(row, "notes"),
    }


def assets_by_campaign() -> dict[str, list[dict[str, Any]]]:
    """Every creative, grouped by the campaign it belongs to, longest spot first."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for _, row in load_frame().iterrows():
        record = asset_record(row)
        grouped.setdefault(record["campaign_id"], []).append(record)
    for records in grouped.values():
        records.sort(key=lambda item: (-(item["duration_seconds"] or 0.0), item["house_number"]))
    return grouped


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The badge a list row needs: how many, how long, and how many unresolved.

    ``fit_to_air_unknown`` is counted rather than assumed away, because a
    campaign whose every tape is unchecked is the normal state of this product
    and the board has to be able to say so. ``validity_unknown`` is counted for
    the same reason and today equals ``count`` on every campaign, because no
    window has ever been recorded.
    """
    if not records:
        return {
            "count": 0,
            "reason_en": NO_ASSETS_EN,
            "reason_he": NO_ASSETS_HE,
            "seconds_total": None,
            "fit_to_air": 0,
            "fit_to_air_unknown": 0,
            "media_unknown": 0,
            "validity_unknown": 0,
            "paired": 0,
        }
    campaigns = {item["campaign_id"] for item in records}
    paired = sum(len(constraints.pairs_for_campaign(campaign)) for campaign in campaigns)
    return {
        "count": len(records),
        "reason_en": "",
        "reason_he": "",
        "seconds_total": round(sum(item["duration_seconds"] or 0.0 for item in records), 2),
        "fit_to_air": sum(1 for item in records if item["clearance"]["fit_to_air"] is True),
        "fit_to_air_unknown": sum(1 for item in records if item["clearance"]["fit_to_air"] is None),
        "media_unknown": sum(1 for item in records if item["media"]["state"] == "unknown"),
        "validity_unknown": sum(1 for item in records if item["validity"]["state"] == "unknown"),
        # How many pairs an operator has authored over this campaign's creatives.
        # Zero everywhere today, and zero is the true count of authored agreements
        # rather than a stand-in for one nobody has entered.
        "paired": paired,
    }


def vocabularies() -> dict[str, Any]:
    return {
        "clearance_verdicts": [dict(entry) for entry in CLEARANCE_VERDICTS],
        "unknown_paths": {
            key: {"path_en": value[0], "path_he": value[1]}
            for key, value in UNKNOWN_PATHS.items()
        },
    }
