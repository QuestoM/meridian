"""The canonical Israeli TV daypart taxonomy, the single source of truth.

A spot, a slot and a training row must all agree on what daypart a given clock
time belongs to, otherwise a rule scoped to "prime" would mean different minutes
in the weekly plan, the daily spot file and the training data. This module is that
one agreement: a fixed, bilingual, full-24h taxonomy that every path imports,
instead of each path inventing its own buckets or letting an operator type a free
token.

The default taxonomy follows how Israeli television is sold, aligned to the
broadcast day that starts at 02:00:

  morning  בוקר        06:00-12:00
  noon     צהריים       12:00-17:00
  evening  ערב          17:00-20:00   (early evening, pre-prime)
  prime    פריים טיים   20:00-23:00   (matches the engine prime window)
  night    לילה         23:00-06:00   (wraps midnight)

Coverage is complete: every clock hour maps to exactly one daypart. The taxonomy
is overridable from ``config/dayparts.yaml`` (a list of {key, he, en, start, end});
when that file is absent or unreadable the built-in default is used, so the engine
never depends on an external file being present. ``prime`` is the daypart the
prime-time pricing and the ``prime_time_only`` baseline match.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = ROOT / "config" / "dayparts.yaml"

# The daypart the prime-time pricing window and prime_time_only baselines key on.
PRIME_KEY = "prime"


@dataclass(frozen=True)
class Daypart:
    """One named part of the broadcast day with its clock window.

    ``start_hour`` is inclusive and ``end_hour`` is exclusive, both on the 0..24
    clock. A daypart that wraps midnight (``start_hour > end_hour``, e.g. night
    23->06) covers ``hour >= start_hour OR hour < end_hour``.
    """

    key: str
    label_he: str
    label_en: str
    start_hour: int
    end_hour: int

    @property
    def wraps_midnight(self) -> bool:
        return self.start_hour > self.end_hour

    def contains_hour(self, hour: int) -> bool:
        """True when a 0..23 clock hour falls inside this daypart's window."""
        if self.wraps_midnight:
            return hour >= self.start_hour or hour < self.end_hour
        return self.start_hour <= hour < self.end_hour


# The built-in default taxonomy (used when config/dayparts.yaml is absent).
_DEFAULT_DAYPARTS: tuple[Daypart, ...] = (
    Daypart("morning", "בוקר", "Morning", 6, 12),
    Daypart("noon", "צהריים", "Noon", 12, 17),
    Daypart("evening", "ערב", "Evening", 17, 20),
    Daypart("prime", "פריים טיים", "Prime time", 20, 23),
    Daypart("night", "לילה", "Night", 23, 6),
)


def _load_from_config(path: Path) -> Optional[tuple[Daypart, ...]]:
    """Load a daypart taxonomy from a YAML override, or None when unavailable.

    The YAML is a list under ``dayparts`` of {key, he, en, start, end}. Any parse
    or schema problem returns None so the caller falls back to the default; this
    module never raises because a config file is malformed.
    """
    if not path.exists():
        return None
    try:
        import yaml  # local import: the engine must import without PyYAML present

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rows = payload.get("dayparts") or []
        parts = tuple(
            Daypart(
                key=str(row["key"]).strip(),
                label_he=str(row.get("he", row["key"])),
                label_en=str(row.get("en", row["key"])),
                start_hour=int(row["start"]),
                end_hour=int(row["end"]),
            )
            for row in rows
        )
        return parts or None
    except Exception:  # noqa: BLE001 - any config problem -> use the default
        logger.warning("Could not read daypart config at %s; using the default taxonomy.", path)
        return None


_DAYPARTS: tuple[Daypart, ...] = _load_from_config(_CONFIG_PATH) or _DEFAULT_DAYPARTS


def dayparts() -> tuple[Daypart, ...]:
    """The active daypart taxonomy (config override or built-in default)."""
    return _DAYPARTS


def daypart_keys() -> tuple[str, ...]:
    """The active daypart keys in order (for validation and dropdowns)."""
    return tuple(part.key for part in _DAYPARTS)


def is_daypart_key(value: object) -> bool:
    """True when ``value`` is one of the active daypart keys."""
    return str(value or "").strip() in daypart_keys()


def daypart_for_hour(hour: Optional[int]) -> Optional[str]:
    """Map a 0..23 clock hour to its daypart key, or None when unknown.

    Returns None (never a guessed bucket) for a missing or out-of-range hour, so a
    spot with no usable clock time stays honestly unclassified.
    """
    if hour is None:
        return None
    try:
        hour = int(hour)
    except (TypeError, ValueError):
        return None
    if not 0 <= hour <= 23:
        return None
    for part in _DAYPARTS:
        if part.contains_hour(hour):
            return part.key
    return None


def daypart_for_timestamp(timestamp: object) -> Optional[str]:
    """Map a timestamp to its daypart key, or None when it is missing/invalid."""
    stamp = pd.Timestamp(timestamp) if not isinstance(timestamp, pd.Timestamp) else timestamp
    if pd.isna(stamp):
        return None
    return daypart_for_hour(int(stamp.hour))


def daypart_options() -> list[dict[str, object]]:
    """The taxonomy as serialisable option dicts for the dashboard dropdown.

    Each option carries the key, both labels and the clock window, so the UI can
    render a fixed, bilingual multi-select instead of a free-text field.
    """
    return [
        {
            "key": part.key,
            "he": part.label_he,
            "en": part.label_en,
            "start_hour": part.start_hour,
            "end_hour": part.end_hour,
        }
        for part in _DAYPARTS
    ]
