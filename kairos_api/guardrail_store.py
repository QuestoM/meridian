"""The regulatory guardrails, in their own store, with a date and a record.

The four limits below are the broadcast licence expressed as numbers. They sat
in ``KairosSettings`` beside the revenue slider, which meant any account that
could move the slider could move the licence, with no effective date, no record
of who changed what, and no way for a compliance owner to attest that nothing
had changed since the last review.

This module is their home. It gives them the three things the settings
document could not:

- **An effective date.** A change carries the date it takes force. The store
  answers what the limits were, or will be, on any given day, so a change can
  be recorded before it applies and an attestation can name the day it covers.
- **A change record.** Every change appends who changed it, when it was
  recorded, which values moved, what they were before, and why. The log is
  append-only; nothing rewrites history.
- **A distinct permission.** Changing a limit is an admin act, not an operator
  act. The revenue slider stays where it is: an operator may move it. That is
  section 4.5's rule applied literally, affiliation for the side of the line
  and role for what may change on it.

The values here are today's shipped values, and a test pins them against the
``KairosSettings`` defaults so the two cannot silently diverge while both
exist. Nothing reads this store yet: the cutover is one line in the piece that
owns the rules surface, and :func:`settings_overlay` is that line. It is an
exact identity while the two agree, which is also a test.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import Request

from kairos_api.affiliation_wall import ADMIN_ROLES, Wall

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PATH = ROOT / "data" / "regulatory_guardrails.json"
PATH_ENV = "KAIROS_GUARDRAILS_PATH"

# The four numbers that are the licence. Every other optimizer knob, including
# the retention floor and the daily cap, is sales policy and stays in settings.
GUARDRAIL_KEYS = (
    "max_ad_minutes_per_hour",
    "max_breaks_per_hour",
    "min_break_spacing_minutes",
    "protected_program_max_ad_minutes_per_hour",
)

# key: (python type, minimum, maximum). Mirrors the KairosSettings bounds, so a
# value refused there is refused here and the two stores cannot disagree about
# what is even representable.
BOUNDS: dict[str, tuple[type, float, float]] = {
    "max_ad_minutes_per_hour": (float, 0.0, 60.0),
    "max_breaks_per_hour": (int, 1, 20),
    "min_break_spacing_minutes": (int, 0, 120),
    "protected_program_max_ad_minutes_per_hour": (float, 0.0, 60.0),
}

GUARDRAIL_ADMIN_ONLY_DETAIL = "עריכת מגבלות הרגולציה שמורה למנהל המערכת"

# Role decides this one, not affiliation: the limits are the operator's own
# licence, so a channel account reads them and an admin changes them.
GUARDRAIL_WALL = Wall(
    detail=GUARDRAIL_ADMIN_ONLY_DETAIL,
    company_only=False,
    roles=ADMIN_ROLES,
    role_detail=GUARDRAIL_ADMIN_ONLY_DETAIL,
)

_LOCK = threading.RLock()


class GuardrailError(ValueError):
    """Raised when a proposed guardrail value or date is not usable."""


def store_path() -> Path:
    """The store file, relocatable with an env knob so tests never touch data."""
    value = os.getenv(PATH_ENV, "").strip()
    if not value:
        return DEFAULT_PATH
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _settings_defaults() -> dict[str, Any]:
    """The four values as the settings model declares them.

    The seed and the fallback both come from here rather than from literals in
    this file, so a store that has never been written is exactly the shipped
    behaviour and never a second opinion about the licence.
    """
    from kairos_api.core import KairosSettings

    defaults = KairosSettings()
    return {key: getattr(defaults, key) for key in GUARDRAIL_KEYS}


def _seed_record() -> dict[str, Any]:
    from kairos_api.core import KairosSettings

    defaults = KairosSettings()
    return {
        "profile_name": defaults.profile_name,
        "source_url": defaults.regulatory_source_url,
        "baseline": {
            "effective_date": defaults.effective_date,
            "values": _settings_defaults(),
        },
        "changes": [],
    }


def load_record() -> dict[str, Any]:
    """The whole store, or the seed when the file is absent or unreadable.

    An unreadable file degrades to the seed and logs, exactly as the settings
    loader does: a transient read failure must never be answered with a
    fabricated limit, and the seed is the shipped licence.
    """
    path = store_path()
    with _LOCK:
        if not path.is_file():
            return _seed_record()
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("regulatory guardrail store unreadable (%s); serving the shipped baseline", exc)
            return _seed_record()
    if not isinstance(record, dict) or not isinstance(record.get("baseline"), dict):
        logger.warning("regulatory guardrail store malformed; serving the shipped baseline")
        return _seed_record()
    record.setdefault("changes", [])
    return record


def save_record(record: dict[str, Any]) -> dict[str, Any]:
    """Atomic write, so a reader never sees a half-written licence."""
    path = store_path()
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(record, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    return record


def _as_day(value: Any, fallback: Optional[date] = None) -> date:
    text = str(value or "").strip().split(" ")[0].split("T")[0]
    try:
        return date.fromisoformat(text)
    except ValueError:
        if fallback is not None:
            return fallback
        raise GuardrailError(f"The effective date must be an ISO date, got {value!r}.")


def values_on(day: Optional[date] = None, record: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The limits in force on ``day``, the baseline plus every change due by it.

    Changes are applied in effective-date order, so a change recorded today for
    next month does not move a number today. Default day is the real today.
    """
    record = load_record() if record is None else record
    when = day or date.today()
    values = dict(_settings_defaults())
    baseline = record.get("baseline") or {}
    values.update({
        key: value for key, value in (baseline.get("values") or {}).items()
        if key in GUARDRAIL_KEYS
    })
    # A change whose date cannot be parsed is treated as not yet in force. A
    # corrupt record must never be the reason a limit reads looser than the
    # licence, so the unreadable case fails toward the baseline.
    due = [
        change for change in record.get("changes") or []
        if _as_day(change.get("effective_date"), date.max) <= when
    ]
    for change in sorted(due, key=lambda item: _as_day(item.get("effective_date"), date.max)):
        values.update({
            key: value for key, value in (change.get("values") or {}).items()
            if key in GUARDRAIL_KEYS
        })
    return {key: values[key] for key in GUARDRAIL_KEYS}


def current_values(record: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The limits in force today."""
    return values_on(None, record)


def effective_date(day: Optional[date] = None, record: Optional[dict[str, Any]] = None) -> str:
    """The date the limits in force on ``day`` took effect."""
    record = load_record() if record is None else record
    when = day or date.today()
    baseline = record.get("baseline") or {}
    latest = str(baseline.get("effective_date") or "")
    latest_day = _as_day(latest, date.min)
    for change in record.get("changes") or []:
        change_day = _as_day(change.get("effective_date"), date.max)
        if change_day <= when and change_day >= latest_day:
            latest_day = change_day
            latest = str(change.get("effective_date"))
    return latest


def scheduled_changes(day: Optional[date] = None, record: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Recorded changes that have not taken force yet, newest date last.

    This is the alert the compliance owner never had: a limit that is about to
    move is visible before the day it moves.
    """
    record = load_record() if record is None else record
    when = day or date.today()
    pending = [
        change for change in record.get("changes") or []
        if _as_day(change.get("effective_date"), date.max) > when
    ]
    return sorted(pending, key=lambda item: _as_day(item.get("effective_date"), date.max))


def changes(record: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """The whole append-only change log, oldest first."""
    record = load_record() if record is None else record
    return list(record.get("changes") or [])


def changed_since(since: date, record: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    """Changes recorded on or after ``since``, which is the attestation answer.

    An empty list is the evidence a compliance owner needs: no guardrail moved
    since the last review.
    """
    record = load_record() if record is None else record
    out = []
    for change in record.get("changes") or []:
        recorded = str(change.get("recorded_at") or "")[:10]
        if _as_day(recorded, date.min) >= since:
            out.append(change)
    return out


def _clean_values(values: dict[str, Any]) -> dict[str, Any]:
    if not values:
        raise GuardrailError("A guardrail change must name at least one limit.")
    cleaned: dict[str, Any] = {}
    for key, raw in values.items():
        if key not in BOUNDS:
            raise GuardrailError(f"{key} is not a regulatory guardrail.")
        kind, low, high = BOUNDS[key]
        try:
            value = kind(raw)
        except (TypeError, ValueError) as exc:
            raise GuardrailError(f"{key} must be a number.") from exc
        if not low <= value <= high:
            raise GuardrailError(f"{key} must be between {low} and {high}.")
        cleaned[key] = value
    return cleaned


def require_guardrail_editor(request: Optional[Request]) -> None:
    """The distinct permission: raise 403 unless this session may change a limit."""
    GUARDRAIL_WALL.require(request)


def record_change(
    values: dict[str, Any],
    effective: str,
    actor: str = "",
    reason: str = "",
    day: Optional[date] = None,
) -> dict[str, Any]:
    """Append one change to the log and return it. The permission is the caller's.

    Validates the values and the date, records what the limits were before the
    change on its own effective day, and writes atomically. Raises
    :class:`GuardrailError` on a value or date the licence cannot hold.
    """
    cleaned = _clean_values(values)
    effective_day = _as_day(effective)
    with _LOCK:
        record = load_record()
        before = {key: values_on(effective_day, record)[key] for key in cleaned}
        change = {
            "effective_date": effective_day.isoformat(),
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "actor": str(actor or "").strip() or "unknown",
            "reason": str(reason or "").strip(),
            "values": cleaned,
            "before": before,
        }
        record.setdefault("changes", []).append(change)
        save_record(record)
    return change


def settings_overlay(settings: Any, day: Optional[date] = None) -> Any:
    """The cutover in one line: settings with the store's limits applied.

    Returns a copy carrying the four values in force, and leaves every other
    field alone. While the store and the settings model agree, which a test
    pins, this is an exact identity and no number moves.
    """
    values = values_on(day)
    if isinstance(settings, dict):
        return {**settings, **values}
    if hasattr(settings, "model_copy"):
        return settings.model_copy(update=values)
    updated = settings.copy() if hasattr(settings, "copy") else settings
    for key, value in values.items():
        setattr(updated, key, value)
    return updated


def payload(request: Optional[Request] = None, day: Optional[date] = None) -> dict[str, Any]:
    """The honest read: the limits, their date, the log, and who may edit.

    ``can_edit`` and its reason come from the same wall the write path uses, so
    the refusal is legible before the click rather than a 403 after it.
    """
    record = load_record()
    when = day or date.today()
    body = {
        "profile_name": record.get("profile_name", ""),
        "source_url": record.get("source_url", ""),
        "effective_date": effective_date(when, record),
        "values": values_on(when, record),
        "changes": changes(record),
        "scheduled_changes": scheduled_changes(when, record),
    }
    return GUARDRAIL_WALL.stamp(body, request)
