"""Manual override layer so a channel operator can hand-fix the optimizer.

Background. The greedy optimizer (:mod:`kairos.optimize.optimizer`) decides how
many breaks each programme segment carries, and the daily per-spot pricing path
(:mod:`kairos.export.spots`) prices each individual spot. Both are automatic. An
operator sometimes needs to override that automation: pin a segment's break count
so optimization cannot move it, force at least N breaks, forbid breaks entirely,
mark a segment's breaks as gold (premium), or, at the daily level, lock or move a
single spot. This module is the pure, honest model and store for those overrides.

Where each override bites (no pretending):

  * scope="segment" overrides feed the weekly break-count optimizer through
    :meth:`OverrideSet.segment_constraints`. The optimizer genuinely supports
    pinning a segment's count, flooring it, zeroing it, and tagging its breaks
    gold, because it allocates COUNTS per segment.
  * scope="spot" overrides feed the daily pricing path through
    :meth:`OverrideSet.spot_overrides`. A ``lock`` keeps a spot exactly as-is
    (never dropped by an advertiser rule). A ``move`` re-tags the spot's position
    or daypart before pricing; the daily path cannot re-place a spot at a clock
    time it never owned, so a move records intent and re-tags what it honestly can.

Honesty rules:

  * An unknown override kind is skipped, so a malformed row never silently bends
    the plan. An empty file (header only, the seeded state) yields no constraints.
  * ``value`` is read as an int for segment counts and as a small token string for
    a spot move; both fall back cleanly when the cell is blank or malformed.
  * Nothing is invented to fill an empty store: with no overrides every segment
    and spot keeps its automatic behaviour.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERRIDES_PATH = ROOT / "data" / "manual_overrides.csv"

# Scopes.
SEGMENT = "segment"
SPOT = "spot"
_SCOPES = (SEGMENT, SPOT)

# Segment-scope kinds (weekly / programme level).
PIN = "pin"        # lock the break count at `value`
FORCE = "force"    # require AT LEAST `value` breaks
FORBID = "forbid"  # this segment carries 0 breaks
GOLD = "gold"      # mark this segment's breaks as gold (is_gold=True)
_SEGMENT_KINDS = (PIN, FORCE, FORBID, GOLD)

# Spot-scope kinds (daily level).
LOCK = "lock"      # pin this spot; keep it exactly as-is, never drop/reprice-away
MOVE = "move"      # relocate the spot to the position/daypart given in `value`
_SPOT_KINDS = (LOCK, MOVE)

# CSV columns, in the order they are written.
#
# The first block is the original override record. The trailing block is the
# additive decision-plane extension: ``source`` records where an override came
# from (a lever tweak, an approved recommendation, or a hand edit), ``rec_id``
# links it back to the recommendation that produced it, ``status`` tracks its
# lifecycle, and the ``anchor_*`` trio is a semantic anchor recorded alongside
# the build-order ``target_id`` so a later resolve can confirm the override
# still points at the same real break after a re-ingest reorders segment ids.
COLUMNS = (
    "override_id",
    "scope",
    "target_id",
    "kind",
    "value",
    "gold",
    "notes",
    "created_at",
    "source",
    "rec_id",
    "status",
    "anchor_date",
    "anchor_start",
    "anchor_title",
)

# Provenance sources.
SOURCE_LEVER = "lever"
SOURCE_RECOMMENDATION = "recommendation"
SOURCE_MANUAL = "manual"
_SOURCES = (SOURCE_LEVER, SOURCE_RECOMMENDATION, SOURCE_MANUAL)
DEFAULT_SOURCE = SOURCE_MANUAL

# Lifecycle statuses.
STATUS_ACTIVE = "active"
STATUS_STALE = "stale"
STATUS_DISMISSED = "dismissed"
_STATUSES = (STATUS_ACTIVE, STATUS_STALE, STATUS_DISMISSED)
DEFAULT_STATUS = STATUS_ACTIVE


def _to_int(raw: object) -> Optional[int]:
    """Parse a break-count value to a non-negative int, or None when unusable."""
    text = str(raw if raw is not None else "").strip()
    if not text:
        return None
    try:
        value = int(float(text))
    except (TypeError, ValueError):
        return None
    return value if value >= 0 else None


def _to_bool(raw: object) -> bool:
    return str(raw if raw is not None else "").strip().lower() in {"true", "1", "yes", "y"}


@dataclass(frozen=True)
class Override:
    """One operator override, as stored in manual_overrides.csv.

    ``scope`` is :data:`SEGMENT` or :data:`SPOT`. ``target_id`` is a programme
    segment id (channel|day|programme or segment_id) for a segment override, or a
    daily spot identifier (advertiser|campaign|date|position) for a spot override.
    ``kind`` is one of the kinds valid for the scope. ``value`` carries the break
    count (segment pin/force) or the move token (spot move); it is ignored for the
    others. ``gold`` is a convenience flag that also marks the breaks gold.
    """

    override_id: str
    scope: str
    target_id: str
    kind: str
    value: str = ""
    gold: bool = False
    notes: str = ""
    created_at: str = ""
    # Decision-plane extension (additive; does not affect fold/apply here).
    source: str = DEFAULT_SOURCE
    rec_id: str = ""
    status: str = DEFAULT_STATUS
    anchor_date: str = ""
    anchor_start: str = ""
    anchor_title: str = ""

    def is_valid(self) -> bool:
        """True when scope and kind are recognised and the target is non-empty."""
        if not self.override_id or not self.target_id:
            return False
        if self.scope == SEGMENT:
            return self.kind in _SEGMENT_KINDS
        if self.scope == SPOT:
            return self.kind in _SPOT_KINDS
        return False


def _parse_move_token(value: str) -> dict[str, str]:
    """Parse a move token like ``position=1`` or ``position=1;daypart=prime``.

    Returns a dict of the recognised keys (``position`` and/or ``daypart``). An
    empty or unrecognised token yields an empty dict, so a move with no parseable
    target re-tags nothing rather than guessing.
    """
    out: dict[str, str] = {}
    for part in str(value or "").replace(",", ";").split(";"):
        if "=" not in part:
            continue
        key, _, raw = part.partition("=")
        key = key.strip().lower()
        raw = raw.strip()
        if key in {"position", "daypart"} and raw:
            out[key] = raw
    return out


@dataclass
class OverrideSet:
    """A loaded, scoped set of operator overrides.

    Build with :meth:`from_csv` (the real file) or directly from a list of
    :class:`Override` objects (tests). Every method is deterministic and reads
    only what is stored, never inventing a constraint.
    """

    overrides: list[Override] = field(default_factory=list)

    @classmethod
    def from_csv(cls, path: str | Path | None = None) -> "OverrideSet":
        target = Path(path) if path else DEFAULT_OVERRIDES_PATH
        return cls(overrides=_load_overrides(target))

    def _valid(self) -> list[Override]:
        return [o for o in self.overrides if o.is_valid()]

    def segment_constraints(self) -> dict[str, dict[str, object]]:
        """Constraints per segment id the optimizer must honour.

        Each value is a dict that may carry ``pin`` (int, exact count), ``min``
        (int, floor), ``forbid`` (bool, force 0) and ``gold`` (bool, tag breaks
        gold). When several overrides target the same segment they merge: the
        last pin wins, the largest min wins, any forbid sticks, any gold sticks.
        A ``gold`` override (or the ``gold`` flag on any override) sets gold.
        """
        out: dict[str, dict[str, object]] = {}
        for override in self._valid():
            if override.scope != SEGMENT:
                continue
            entry = out.setdefault(override.target_id, {})
            if override.gold:
                entry["gold"] = True
            if override.kind == PIN:
                count = _to_int(override.value)
                if count is not None:
                    entry["pin"] = count
            elif override.kind == FORCE:
                count = _to_int(override.value)
                if count is not None:
                    entry["min"] = max(int(entry.get("min", 0)), count)
            elif override.kind == FORBID:
                entry["forbid"] = True
            elif override.kind == GOLD:
                entry["gold"] = True
        return out

    def spot_overrides(self) -> dict[str, dict[str, object]]:
        """Overrides per spot id the daily pricing path must honour.

        Each value may carry ``lock`` (bool, keep the spot exactly as-is) and
        ``move`` (dict, the re-tag target parsed from the value token, with any of
        ``position`` and ``daypart``). When several overrides target the same spot
        they merge: any lock sticks, the last move's parsed target wins.
        """
        out: dict[str, dict[str, object]] = {}
        for override in self._valid():
            if override.scope != SPOT:
                continue
            entry = out.setdefault(override.target_id, {})
            if override.kind == LOCK:
                entry["lock"] = True
            elif override.kind == MOVE:
                entry["move"] = _parse_move_token(override.value)
        return out


def _load_overrides(path: Path) -> list[Override]:
    """Read manual_overrides.csv into Override objects.

    A missing file or a header-only file yields an empty list, the honest answer
    for the seeded state. Malformed rows are kept but flagged invalid by
    :meth:`Override.is_valid`, so the loader never silently drops a row it read.
    """
    if not path.exists():
        return []
    out: list[Override] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        for row in reader:
            override_id = str(row.get("override_id", "")).strip()
            if not override_id:
                continue
            # New columns are absent in an older CSV; row.get returns "" and the
            # blank-safe defaults below keep an old file loading unchanged
            # (source -> manual, status -> active, anchors -> blank).
            out.append(Override(
                override_id=override_id,
                scope=str(row.get("scope", "")).strip().lower(),
                target_id=str(row.get("target_id", "")).strip(),
                kind=str(row.get("kind", "")).strip().lower(),
                value=str(row.get("value", "")).strip(),
                gold=_to_bool(row.get("gold")),
                notes=str(row.get("notes", "")),
                created_at=str(row.get("created_at", "")),
                source=str(row.get("source", "")).strip().lower() or DEFAULT_SOURCE,
                rec_id=str(row.get("rec_id", "")).strip(),
                status=str(row.get("status", "")).strip().lower() or DEFAULT_STATUS,
                anchor_date=str(row.get("anchor_date", "")).strip(),
                anchor_start=str(row.get("anchor_start", "")).strip(),
                anchor_title=str(row.get("anchor_title", "")).strip(),
            ))
    return out
