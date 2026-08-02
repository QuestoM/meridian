"""The plan target: the number a plan window is measured against.

"Is this week on plan" is the first question a general manager asks and the
only one this product could never answer, because nothing in the data is a
target. Discovery measured it: ``/api/overview`` carries no ``goal``, ``target``,
``budget``, ``on_plan`` or ``variance`` key, and no budget or quota entity
exists anywhere in the model.

A target cannot be derived from the plan, because a plan compared against
itself is always exactly on plan. So it is supplied by a person, stored here,
and until somebody supplies one the surface says so and offers the control that
supplies it. Nothing in this module invents a number.

What a target record is:

- **The window it measures.** A channel plus a start and an end date, keyed on
  the same window the saved plan reports (``summary.week.date_from`` and
  ``date_to``), so the target and the projection are never two different spans.
- **The quantity.** ``projected_revenue`` in ILS today, which is the only
  quantity the plan already computes at that grain. The ``metric`` column
  exists so a rating-point target is representable without a migration.
- **The threshold**, supplied with the target and never defaulted. Three states
  need a boundary between them, and picking that boundary here would be this
  module inventing the business rule it exists to record.
- **Who set it and when**, so a figure on a screen has an author.

The store follows the doctrine the other stores already use: a module lock, a
pre-write backup, a temp file plus ``os.replace``, and a version snapshot before
the write, which is a safe no-op until the version store registers the
``plan_targets`` logical name.
"""

from __future__ import annotations

import csv
import logging
import os
import shutil
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import HTTPException, Request

from kairos_api.affiliation_wall import READ_ONLY_ROLE_DETAIL, Wall

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
TARGETS_PATH = DATA_DIR / "plan_targets.csv"
PATH_ENV = "KAIROS_PLAN_TARGETS_PATH"

COLUMNS = (
    "channel",
    "period_start",
    "period_end",
    "metric",
    "amount_ils",
    "at_risk_band_percent",
    "set_by",
    "set_at",
    "note",
)

# The only quantity the saved plan computes at window grain today. A target in
# any other unit would have nothing to be compared against, so it is refused
# rather than stored where it would read as measurable.
SUPPORTED_METRICS = ("projected_revenue",)
DEFAULT_METRIC = "projected_revenue"

MAX_AMOUNT_ILS = 1_000_000_000.0
MAX_BAND_PERCENT = 100.0

# Role decides this one, not affiliation. A target is the operator's own
# commercial number, so any account on their side of the line reads it and a
# write role sets it.
TARGET_WALL = Wall(detail=READ_ONLY_ROLE_DETAIL, company_only=False)

# The wall answers in the product's first language, and its string is used
# verbatim so the refusal a person reads before the click and the one the 403
# would carry cannot drift. Today also ships an English screen, and the reader
# this destination is built for is on it, so the same rule is carried in both
# languages: the Hebrew half is the wall's own constant, byte for byte, and the
# English half is this surface's rendering of the identical rule. A refusal this
# map does not know is repeated as it arrived rather than translated into words
# nobody sent.
REFUSAL_WORDS: dict[str, tuple[str, str]] = {
    READ_ONLY_ROLE_DETAIL: ("A viewing account has no edit permission", READ_ONLY_ROLE_DETAIL),
}

_LOCK = threading.RLock()


def path() -> Path:
    """The store's path, overridable by environment for tests and deployments."""
    override = os.environ.get(PATH_ENV, "").strip()
    return Path(override) if override else TARGETS_PATH


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: Any) -> str:
    return str(value if value is not None else "").strip()


def _float_or_none(value: Any) -> Optional[float]:
    text = _clean(value)
    if not text:
        return None
    try:
        number = float(text)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _iso_date(value: Any) -> Optional[str]:
    text = _clean(value)[:10]
    if not text:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError:
        return None


def _record(row: dict[str, Any]) -> dict[str, Any]:
    """One stored row as the payload shape, with nothing coerced into existence."""
    return {
        "channel": _clean(row.get("channel")),
        "period_start": _iso_date(row.get("period_start")),
        "period_end": _iso_date(row.get("period_end")),
        "metric": _clean(row.get("metric")) or DEFAULT_METRIC,
        "amount_ils": _float_or_none(row.get("amount_ils")),
        "at_risk_band_percent": _float_or_none(row.get("at_risk_band_percent")),
        "set_by": _clean(row.get("set_by")),
        "set_at": _clean(row.get("set_at")),
        "note": _clean(row.get("note")),
    }


def read_all() -> list[dict[str, Any]]:
    """Every stored target, oldest row first. Missing file reads as no targets."""
    target = path()
    if not target.exists():
        return []
    try:
        with target.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = [_record(row) for row in csv.DictReader(handle)]
    except OSError:  # pragma: no cover - defensive, a read must not 500 a page
        logger.exception("plan target store unreadable at %s", target)
        return []
    return [row for row in rows if row["channel"] and row["period_start"] and row["amount_ils"] is not None]


def target_for(channel: str, period_start: str, period_end: str) -> Optional[dict[str, Any]]:
    """The target for exactly this window, or None.

    The match is exact on all three keys. A target set for a different span is
    deliberately not returned as this window's: comparing a seven-day
    projection against a thirty-day number is the kind of quiet mismatch the
    scope rule exists to stop.
    """
    owned = _clean(channel)
    start = _iso_date(period_start)
    end = _iso_date(period_end)
    if not owned or not start or not end:
        return None
    for row in read_all():
        if row["channel"] == owned and row["period_start"] == start and row["period_end"] == end:
            return row
    return None


def targets_for_channel(channel: str) -> list[dict[str, Any]]:
    """Every target on this channel, newest window first."""
    owned = _clean(channel)
    rows = [row for row in read_all() if row["channel"] == owned]
    return sorted(rows, key=lambda row: row["period_start"] or "", reverse=True)


def _validate(
    channel: str,
    period_start: str,
    period_end: str,
    metric: str,
    amount_ils: Any,
    at_risk_band_percent: Any,
) -> dict[str, Any]:
    owned = _clean(channel)
    if not owned:
        raise HTTPException(status_code=400, detail="A target needs the channel it measures.")
    start = _iso_date(period_start)
    end = _iso_date(period_end)
    if start is None or end is None:
        raise HTTPException(status_code=400, detail="A target needs an ISO start and end date.")
    if end < start:
        raise HTTPException(status_code=400, detail="A target window ends before it starts.")
    chosen = _clean(metric) or DEFAULT_METRIC
    if chosen not in SUPPORTED_METRICS:
        raise HTTPException(status_code=400, detail=f"Unsupported target metric '{chosen}'.")
    amount = _float_or_none(amount_ils)
    if amount is None or amount <= 0 or amount > MAX_AMOUNT_ILS:
        raise HTTPException(status_code=400, detail="A target amount must be a positive number of shekels.")
    band = _float_or_none(at_risk_band_percent)
    if band is None or band < 0 or band > MAX_BAND_PERCENT:
        raise HTTPException(status_code=400, detail="An at-risk band must be a percentage between 0 and 100.")
    return {
        "channel": owned,
        "period_start": start,
        "period_end": end,
        "metric": chosen,
        "amount_ils": round(amount, 2),
        "at_risk_band_percent": round(band, 2),
    }


def _backup(target: Path) -> None:
    """Copy the store beside itself before a write.

    The backup directory is derived from the store's own path rather than from
    the module constant, so a redirected store (a test, a second deployment)
    backs up into its own tree and never writes into the repository's data
    directory. A store that does not exist yet has nothing to back up.
    """
    if not target.exists():
        return
    backups = target.parent / BACKUP_DIR.name
    backups.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    shutil.copy2(target, backups / f"plan_targets_{stamp}.csv")


def _write_rows(rows: list[dict[str, Any]]) -> None:
    """Backup, then write atomically through a temp file and ``os.replace``."""
    target = path()
    _backup(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _serialize(row.get(column)) for column in COLUMNS})
    os.replace(tmp, target)


def _serialize(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.2f}".rstrip("0").rstrip(".") if value % 1 else str(int(value))
    return str(value)


def _snapshot_before_write(request: Optional[Request]) -> None:
    """Version the store before a manual edit writes it. A safe no-op until the
    version store registers the ``plan_targets`` logical name."""
    from kairos_api import version_store

    version_store.snapshot_manual_edit(request, "plan_targets")


def _actor(request: Optional[Request]) -> str:
    from kairos_api.affiliation_wall import session_for

    session = session_for(request)
    return str((session or {}).get("username", "")) or "unknown"


def save_target(
    channel: str,
    period_start: str,
    period_end: str,
    amount_ils: Any,
    at_risk_band_percent: Any,
    metric: str = DEFAULT_METRIC,
    note: str = "",
    request: Optional[Request] = None,
) -> dict[str, Any]:
    """Set or replace the target for one window. Returns the stored record."""
    validated = _validate(channel, period_start, period_end, metric, amount_ils, at_risk_band_percent)
    record = {
        **validated,
        "set_by": _actor(request),
        "set_at": _now_iso(),
        "note": _clean(note)[:280],
    }
    with _LOCK:
        _snapshot_before_write(request)
        kept = [
            row
            for row in read_all()
            if not (
                row["channel"] == record["channel"]
                and row["period_start"] == record["period_start"]
                and row["period_end"] == record["period_end"]
            )
        ]
        _write_rows(kept + [record])
    return record


def delete_target(
    channel: str,
    period_start: str,
    period_end: str,
    request: Optional[Request] = None,
) -> bool:
    """Remove one window's target. True when a row was removed."""
    owned = _clean(channel)
    start = _iso_date(period_start)
    end = _iso_date(period_end)
    with _LOCK:
        rows = read_all()
        kept = [
            row
            for row in rows
            if not (row["channel"] == owned and row["period_start"] == start and row["period_end"] == end)
        ]
        if len(kept) == len(rows):
            return False
        _snapshot_before_write(request)
        _write_rows(kept)
    return True


def refusal_words(reason: Any) -> tuple[str, str]:
    """One refusal in both languages: English first, then the wall's own Hebrew."""
    text = _clean(reason)
    return REFUSAL_WORDS.get(text, (text, text))


def threshold_sentence(band: Optional[float], locale: str = "en") -> str:
    """The published rule, in the words the surface prints beside the verdict."""
    if band is None:
        return ""
    reading = f"{band:g}"
    if locale == "he":
        return f"על התוכנית ביעד או מעליו, בסיכון עד {reading} אחוז מתחת ליעד, מתחת לכך בפיגור"
    return f"On plan at or above the target, at risk up to {reading} percent below it, behind under that"


def verdict(projected: Any, target: Optional[dict[str, Any]]) -> dict[str, Any]:
    """The three-state read on a window, with the threshold it was decided by.

    ``unavailable`` is a fourth state and it is not a verdict: it is the honest
    answer when there is no target, or the plan projects nothing for the window.
    It carries the reason and never a number.
    """
    amount = None if target is None else target.get("amount_ils")
    band = None if target is None else target.get("at_risk_band_percent")
    value = _float_or_none(projected)
    if amount is None or band is None:
        return {
            "state": "unavailable",
            "reason": "no_target",
            "variance_ils": None,
            "variance_percent": None,
            "threshold_en": "",
            "threshold_he": "",
        }
    if value is None:
        return {
            "state": "unavailable",
            "reason": "no_projection",
            "variance_ils": None,
            "variance_percent": None,
            "threshold_en": threshold_sentence(band, "en"),
            "threshold_he": threshold_sentence(band, "he"),
        }
    variance = round(value - float(amount), 2)
    variance_percent = round(variance / float(amount) * 100, 2) if float(amount) else None
    # The band is compared in shekels, not on the rounded percentage, so the
    # boundary sits exactly where the arithmetic puts it. Rounding first moved
    # it by up to half a basis point, which is real money at this scale.
    at_risk_floor = -float(amount) * float(band) / 100.0
    if variance >= 0:
        state = "on_plan"
    elif variance >= at_risk_floor:
        state = "at_risk"
    else:
        state = "behind"
    return {
        "state": state,
        "reason": None,
        "variance_ils": variance,
        "variance_percent": variance_percent,
        "threshold_en": threshold_sentence(band, "en"),
        "threshold_he": threshold_sentence(band, "he"),
    }


def payload(
    channel: str,
    period_start: str,
    period_end: str,
    request: Optional[Request] = None,
) -> dict[str, Any]:
    """The target block a surface renders, with ``can_edit`` stamped on it.

    When the answer is no, the refusal travels in both languages beside the
    wall's own verbatim string, because a surface that prints only the string it
    was sent prints Hebrew under an English paragraph.
    """
    record = target_for(channel, period_start, period_end)
    body: dict[str, Any] = {
        "state": "set" if record else "unset",
        "metric": DEFAULT_METRIC,
        "currency": "ILS",
        "channel": _clean(channel) or None,
        "period_start": _iso_date(period_start),
        "period_end": _iso_date(period_end),
        "amount_ils": None if record is None else record["amount_ils"],
        "at_risk_band_percent": None if record is None else record["at_risk_band_percent"],
        "set_by": None if record is None else record["set_by"],
        "set_at": None if record is None else record["set_at"],
        "note": None if record is None else record["note"],
        "other_windows": [row for row in targets_for_channel(channel) if row != record],
        "store_path": str(path().relative_to(ROOT)) if path().is_relative_to(ROOT) else str(path()),
    }
    stamped = TARGET_WALL.stamp(body, request)
    if not stamped.get("can_edit"):
        english, hebrew = refusal_words(stamped.get("can_edit_reason"))
        stamped["can_edit_reason_en"] = english
        stamped["can_edit_reason_he"] = hebrew
    return stamped
