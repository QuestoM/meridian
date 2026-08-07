"""The regulatory guardrails, in their own store, with a date and a record.

The four limits below are the broadcast licence expressed as numbers. They sat
in ``KairosSettings`` beside the revenue slider, which meant any account that
could move the slider could move the licence, with no effective date, no record
of who changed what, and no way for a compliance owner to attest that nothing
had changed since the last review.

This module is their home. It gives them the three things the settings
document could not:

- **An effective date.** A change carries the date it takes force, so the store
  answers what the limits were, or will be, on any given day.
- **A change record.** Every change appends who changed it, when, which values
  moved, what they were before, and why. Append-only; nothing rewrites history.
- **A distinct permission.** Changing a limit is company staff only, by the
  owner's ruling of 2026-08-01, and an admin act on top of that. The revenue
  slider stays where it is. Reading is not gated at all, because the licence is
  the broadcaster's own and the person who attests to it works for it.

The values here are today's shipped values, and a test pins them against the
``KairosSettings`` defaults so the two cannot silently diverge while both
exist. :func:`settings_overlay` is the cutover onto the engine, an exact
identity while the two agree, which is also a test. **Every sentence this store
serves a screen is authored in both languages at once**, through :func:`say`
and :data:`WORDS`: the reader may be reading either, and a refusal nobody can
read is not a refusal.
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

from kairos_api.affiliation_wall import ADMIN_ROLES, Wall, has_role
# The process's one first-strong isolate pair, not a second copy of it.
from kairos_api.uploads_messages import ISOLATE_END, ISOLATE_START

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

# The four limits as the person accountable for them says them, both languages
# from one entry. A sentence below that names a limit takes the pair and each
# half resolves its own, so a Hebrew refusal never names the engine's key.
LIMIT_NAMES: dict[str, tuple[str, str]] = {
    "max_ad_minutes_per_hour": ("ad minutes per broadcast hour", "דקות פרסום לשעת שידור"),
    "max_breaks_per_hour": ("breaks per hour", "ברייקים בשעה"),
    "min_break_spacing_minutes": ("the minimum spacing between breaks", "המרווח המינימלי בין ברייקים"),
    "protected_program_max_ad_minutes_per_hour": ("ad minutes per hour in protected content", "דקות פרסום לשעה בתוכן מוגן"),
}

# Every sentence this store serves a screen, one entry each. Measured before
# this: the wall's refusal was authored in Hebrew alone and ``can_edit_reason``
# carried it verbatim, so with the product in English the licence section printed
# ``שינוי מגבלות הרגולציה שמור לצוות החברה`` above four English fields; and the
# mirror of it sat under the save button, where every :class:`GuardrailError` was
# authored in English alone, reached a Hebrew screen through the 400 the route
# raises, and named the engine's own key while it did it. The Hebrew halves of
# ``company_only`` and ``admin_only`` are the wall's own two details, taken from
# here rather than restated, which is why the 403 body and the reason a control
# renders before the click are still one string and still byte-identical.
WORDS: dict[str, dict[str, str]] = {
    "company_only": {"en": "Only company staff change the regulatory limits.",
                     "he": "שינוי מגבלות הרגולציה שמור לצוות החברה"},
    "admin_only": {"en": "Only an administrator changes the regulatory limits.",
                   "he": "עריכת מגבלות הרגולציה שמורה למנהל המערכת"},
    "bad_date": {"en": "The effective date has to be a calendar date, and {value} is not one.",
                 "he": "תאריך התוקף חייב להיות תאריך בלוח השנה, ו-{value} אינו כזה."},
    "no_limits": {"en": "A licence change has to name at least one limit to move.",
                  "he": "שינוי רישיון חייב לנקוב לפחות במגבלה אחת שזזה."},
    "not_a_guardrail": {"en": "There is no regulatory limit called {name}.",
                        "he": "אין מגבלת רגולציה בשם {name}."},
    "not_a_number": {"en": "The limit for {name} has to be a number.",
                     "he": "המגבלה על {name} חייבת להיות מספר."},
    "out_of_bounds": {"en": "The limit for {name} has to be between {low} and {high}.",
                      "he": "המגבלה על {name} חייבת להיות בין {low} ל-{high}."},
}

GUARDRAIL_ADMIN_ONLY_DETAIL = WORDS["admin_only"]["he"]
GUARDRAIL_COMPANY_ONLY_DETAIL = WORDS["company_only"]["he"]


def _rendered(fields: dict[str, object], index: int, isolate: bool) -> dict[str, str]:
    """One language's view of the fields, runs in the other direction isolated.

    A two-item tuple carries its own two languages, which is how the name of a
    limit stays in the language of the sentence around it. Without the isolate
    the Hebrew sentence's punctuation lands on the wrong side of a digit run.
    """
    out: dict[str, str] = {}
    for name, value in fields.items():
        text = str(value[index] if isinstance(value, tuple) and len(value) == 2 else value)
        out[name] = f"{ISOLATE_START}{text}{ISOLATE_END}" if isolate and text else text
    return out


def say(code: str, **fields: object) -> tuple[str, str]:
    """This code's sentence in English and in Hebrew, together or not at all. A
    code this table lacks renders two blanks, never one half nobody can read."""
    words = WORDS.get(code)
    if words is None:
        return "", ""
    try:
        return words["en"].format(**_rendered(fields, 0, False)), words["he"].format(**_rendered(fields, 1, True))
    except (KeyError, IndexError):
        return "", ""


class GuardrailWall(Wall):
    """The licence wall, and both halves of its refusal from one call.

    ``can_edit_reason`` is Hebrew because the 403 detail and the reason a control
    renders before the click are one string by contract, and rendered verbatim it
    printed a Hebrew sentence above four English fields. The pair rides beside it
    now, resolved from the gate that actually closed rather than by translating
    the sentence that gate produced, so the two cannot drift: :meth:`refusal`
    walks the same two gates in the same order ``Wall.reason`` does, and a test
    pins that its Hebrew half is that reason for every affiliation and role.
    """

    def refusal(self, request: Optional[Request]) -> tuple[str, str]:
        """Why this requester may not CHANGE a limit, both languages, or two blanks."""
        if self.read_reason(request) is not None:
            return say("company_only")
        if self.roles and not has_role(request, self.roles):
            return say("admin_only")
        return "", ""

    def stamp(self, payload: dict[str, Any], request: Optional[Request]) -> dict[str, Any]:
        stamped = super().stamp(payload, request)
        english, hebrew = self.refusal(request)
        for key in ("can_edit_reason_en", "can_edit_reason_he"):
            stamped.pop(key, None)
        if english and hebrew:
            stamped["can_edit_reason_en"], stamped["can_edit_reason_he"] = english, hebrew
        return stamped


# Both gates, by the owner's ruling. Changing a licence number is company staff
# only until a real owner for it is named at the broadcaster, so affiliation is
# the outer gate and role the inner one. The READ is deliberately not gated: the
# licence is the broadcaster's own and so is the person who attests to it, so
# they read every limit, every change and the whole attestation, and what they
# cannot do is move a number. Nothing here calls ``guard`` or ``require_read``,
# so the read stays open by construction and ``stamp`` answers before the click.
GUARDRAIL_WALL = GuardrailWall(
    detail=GUARDRAIL_COMPANY_ONLY_DETAIL,
    company_only=True,
    roles=ADMIN_ROLES,
    role_detail=GUARDRAIL_ADMIN_ONLY_DETAIL,
)

_LOCK = threading.RLock()


class GuardrailError(ValueError):
    """A value or date the licence cannot hold, said in both languages at once.

    ``str()`` is the English half, because the route that answers this is frozen
    and puts exactly that into the 400's ``detail``. ``hebrew`` is the same
    sentence from the same entry, so a surface that knows its reader's language
    has it without translating anything.
    """

    def __init__(self, code: str, **fields: object) -> None:
        self.code = code
        self.fields = fields
        self.english, self.hebrew = say(code, **fields)
        super().__init__(self.english)


def store_path() -> Path:
    """The store file, relocatable with an env knob so tests never touch data."""
    value = os.getenv(PATH_ENV, "").strip()
    if not value:
        return DEFAULT_PATH
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _settings_defaults() -> dict[str, Any]:
    """The four values as the settings model declares them.

    The seed and the fallback both come from here rather than from literals, so
    an unwritten store is the shipped behaviour and never a second opinion.
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
        raise GuardrailError("bad_date", value=text or ("nothing at all", "ערך ריק"))


def values_on(day: Optional[date] = None, record: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The limits in force on ``day``, the baseline plus every change due by it.

    Changes are applied in effective-date order, so a change recorded today for
    next month does not move a number today. Default day is the real today.
    """
    record = load_record() if record is None else record
    when = day or date.today()
    values = dict(_settings_defaults())
    baseline = record.get("baseline") or {}
    values.update({key: value for key, value in (baseline.get("values") or {}).items() if key in GUARDRAIL_KEYS})
    # A change whose date cannot be parsed is treated as not yet in force. A
    # corrupt record must never be the reason a limit reads looser than the
    # licence, so the unreadable case fails toward the baseline.
    due = [item for item in record.get("changes") or [] if _as_day(item.get("effective_date"), date.max) <= when]
    for change in sorted(due, key=lambda item: _as_day(item.get("effective_date"), date.max)):
        values.update({key: value for key, value in (change.get("values") or {}).items() if key in GUARDRAIL_KEYS})
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
    """Recorded changes not in force yet, newest last: the alert a compliance
    owner never had, because a limit about to move is visible before it moves."""
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
    """Changes recorded on or after ``since``, the attestation answer. An empty
    list is the evidence: no guardrail moved since the last review."""
    record = load_record() if record is None else record
    out = []
    for change in record.get("changes") or []:
        recorded = str(change.get("recorded_at") or "")[:10]
        if _as_day(recorded, date.min) >= since:
            out.append(change)
    return out


def _clean_values(values: dict[str, Any]) -> dict[str, Any]:
    if not values:
        raise GuardrailError("no_limits")
    cleaned: dict[str, Any] = {}
    for key, raw in values.items():
        name = LIMIT_NAMES.get(key, (str(key), str(key)))
        if key not in BOUNDS:
            raise GuardrailError("not_a_guardrail", name=name)
        kind, low, high = BOUNDS[key]
        try:
            value = kind(raw)
        except (TypeError, ValueError) as exc:
            raise GuardrailError("not_a_number", name=name) from exc
        if not low <= value <= high:
            raise GuardrailError("out_of_bounds", name=name, low=low, high=high)
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

    Records what the limits were before the change on its own effective day, and
    writes atomically. Raises :class:`GuardrailError` on a value it cannot hold.
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

    A copy carrying the four values in force, every other field left alone.
    While the two agree, which a test pins, this is an exact identity.
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
    the refusal is legible before the click rather than a 403 after it, and the
    reason arrives in both languages together.

    ``bounds`` is the other refusal a person can reach here. The route that
    answers a rejected change is frozen and forwards ``str(exc)`` alone, so only
    a :class:`GuardrailError`'s English half can travel on it. The numbers travel
    instead, from the same :data:`BOUNDS` the write path validates against, so
    the surface refuses in its reader's own language before it sends.
    """
    record = load_record()
    when = day or date.today()
    body = {
        "profile_name": record.get("profile_name", ""),
        "source_url": record.get("source_url", ""),
        "effective_date": effective_date(when, record),
        "values": values_on(when, record),
        "bounds": {key: {"min": low, "max": high} for key, (_, low, high) in BOUNDS.items()},
        "changes": changes(record),
        "scheduled_changes": scheduled_changes(when, record),
    }
    return GUARDRAIL_WALL.stamp(body, request)
