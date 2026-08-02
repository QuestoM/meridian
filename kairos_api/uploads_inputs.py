"""What the seven inputs are, and what a valid one looks like.

Split out of ``uploads.py`` under the file-size cap, named by the
``<parent stem>_<role>.py`` rule. Everything here is a declaration or a pure
function over a header: no path this module holds is ever written, and every
writable location stays in ``uploads.py``.

The three tables are the product's own truth about where an input goes and who
reads it, and each carries the reason it exists, because a table like this is
where an honest ``in_use`` verdict either comes from or is quietly invented.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pandas as pd

from kairos.data.loaders import DAILY_COLUMN_MAP

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = ROOT / "data" / "reference"

# The three channel-source kinds land as flat CSVs under data/. The engine
# loaders (kairos.data.loaders) read data/reference/*.xlsx FIRST and fall back to
# the uploaded CSV only when that xlsx is absent. So while the reference xlsx
# exists, an uploaded CSV is stored and backed up but SHADOWED: the optimizer
# reads the xlsx, not the upload. We map each shadowed kind to the reference file
# that takes precedence so the status can say so honestly instead of reporting a
# bare green "valid" that implies ingestion. Remove the reference xlsx and the
# upload becomes the live input (the loader adopts the CSV fallback).
SHADOWING_REFERENCE: dict[str, Path] = {
    "programmes": REFERENCE_DIR / "Programmes.xlsx",
    "spots": REFERENCE_DIR / "Spots.xlsx",
    "dayparts": REFERENCE_DIR / "Dayparts.xlsx",
}

# Kinds that are stored on disk but which NO engine code reads. The rate card
# uploads to data/rate_card_premiums.csv, yet the pricing engine
# (kairos.optimize.pricing.PricingModel) reads its rate card from
# config/optimization_weights.yaml, deep-merged with the dashboard's
# pricing_overrides; nothing in the optimizer, forecast, or export path opens
# data/rate_card_premiums.csv (the only other reference to it is a file-existence
# count in the data-quality report). Reporting such a kind as in_use would imply
# an ingestion that never happens, so it is reported in_use False with the real
# reason. The mapped string names the file the engine actually consumes instead.
STORED_UNREAD: dict[str, str] = {
    "rate_card": "config/optimization_weights.yaml",
}

# Required columns per kind. These are the canonical headers the loaders and
# the optimizer read; extra columns are tolerated (reported as warnings).
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "programmes": ["Title", "Channel", "Date", "Start time", "End time", "Duration"],
    "spots": ["Campaign", "Channel", "Date", "Start time", "Duration"],
    "dayparts": ["Dates", "Timebands"],
    "advertiser_rules": [
        "advertiser_id",
        "default_premium",
        "allow_positions",
        "allow_genres",
        "prime_time_only",
        "notes",
    ],
    "rate_card": ["channel", "hour_of_day", "base_rate_ils_per_sec"],
    # The pacing flight file: the exact header of the shipped seed
    # data/campaign_flights.csv, which kairos.optimize.pacing.load_campaigns reads.
    "campaign_flights": [
        "campaign_id",
        "flight_start",
        "flight_end",
        "target_impressions",
        "target_grp",
        "delivered_to_date",
        "scope_channels",
        "scope_genres",
        "scope_dayparts",
        "scope_programmes",
        "notes",
    ],
    # The daily Wally file ships with Hebrew headers; the loader maps them.
    "daily": list(DAILY_COLUMN_MAP.keys()),
}

# Per-kind presentation metadata for the dashboard.
#
# The channel provides THREE source data files (programmes, spots, dayparts);
# the optimizer also takes ONE daily operational file (the Wally ad log). The
# advertiser rules, the rate card and the campaign flights are CONFIGURATION,
# not periodic data the channel uploads, so they are grouped separately.
# Advertiser rules are also editable directly in the Clients screen.
INPUTS: list[dict[str, str]] = [
    {"kind": "programmes", "label_en": "Programme lineup", "label_he": "לוח תוכניות", "cadence": "weekly"},
    {"kind": "daily", "label_en": "Daily ad log (Wally)", "label_he": "קובץ פרסומות יומי", "cadence": "daily"},
    {"kind": "spots", "label_en": "Historical spots", "label_he": "תשדירים היסטוריים", "cadence": "reference"},
    {"kind": "dayparts", "label_en": "Dayparts (ratings by time)", "label_he": "חלקי יום (רייטינג לפי שעה)", "cadence": "reference"},
    {"kind": "advertiser_rules", "label_en": "Advertiser rules", "label_he": "כללי מפרסמים", "cadence": "config"},
    {"kind": "rate_card", "label_en": "Rate card", "label_he": "כרטיס תעריפים", "cadence": "config"},
    {"kind": "campaign_flights", "label_en": "Campaign flights (delivery pacing)", "label_he": "קמפיינים ויעדי אספקה (קצב)", "cadence": "config"},
]

FILENAME_DATE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def airing_date_from_name(path: Path) -> date | None:
    """The latest real ISO date named in the filename, or None when there is none."""
    found: list[date] = []
    for text in FILENAME_DATE.findall(path.name):
        try:
            found.append(date.fromisoformat(text))
        except ValueError:
            continue
    return max(found) if found else None


def missing_columns(kind: str, columns: list[str]) -> list[str]:
    """Return the required columns that are missing from the header.

    Extra columns are always accepted: the loaders read the columns they need
    and pass the rest through. The channel's enriched exports legitimately
    carry many additional columns (TVR, computed premiums, per-channel ratings,
    and so on), so they are never "ignored" and we never warn about them.
    Only a genuinely MISSING required column is worth flagging, because that
    would actually break the optimizer.
    """
    required = REQUIRED_COLUMNS.get(kind, [])
    present = set(columns)
    return [column for column in required if column not in present]


def read_header_and_rows(path: Path) -> tuple[list[str], int, list[str]]:
    """Cheaply read the CSV header and count data rows without loading values."""
    warnings: list[str] = []
    try:
        header_frame = pd.read_csv(path, encoding="utf-8-sig", nrows=0)
        columns = [str(column) for column in header_frame.columns]
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        return [], 0, [f"Could not parse CSV header: {exc}"]
    try:
        # Count rows by reading a single column; falls back to full read.
        usecol = [header_frame.columns[0]] if len(header_frame.columns) else None
        counted = pd.read_csv(path, encoding="utf-8-sig", usecols=usecol)
        rows = int(len(counted))
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        warnings.append(f"Could not count rows precisely: {exc}")
        rows = 0
    return columns, rows, warnings
