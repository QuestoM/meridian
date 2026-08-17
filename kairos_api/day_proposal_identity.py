"""What a broadcast day IS, and whether it moved under somebody's proposal.

Split out of the proposal store because it answers a different question. The
store answers "who proposed what and who decided"; this answers "is the day
these people are arguing about still the day they were looking at". The second
question is the one that makes the first one safe.

Three pieces.

**Canonical bytes.** A day's rows sorted by segment with a fixed column order,
so a sha over them answers "is this the same day" and not "was this file written
the same way". A re-run that produced an identical day therefore reads as
unchanged rather than as a change.

**The baseline reference.** A day's identity as a proposal recorded it: the
canonical sha of its rows, the engine's run stamp, the whole plan file's sha, and
the operator settings in force. It carries ``basis`` because two different bases
exist for the same date - the engine's live re-plan of the channel-day and the
committed weekly plan file - and comparing across them silently is the mistake
this whole surface exists to prevent.

**Staleness.** The comparison of an authored reference against the day as it
stands now, field by field, so a refusal can name what moved rather than only
that something did. An unavailable current reference is reported as UNKNOWN and
never as fresh: a proposal whose baseline cannot be read is not thereby safe to
adopt. This is the server-side form of the fingerprint rule the browser-local day
drafts already enforce.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ProposalRefused(ValueError):
    """A refusal the operator reads, in both languages.

    Store-level refusals in this product reach a person through an HTTP detail,
    so they carry the Hebrew sentence with them instead of leaving the route to
    invent one from an English string.
    """

    def __init__(self, reason: str, reason_he: str, code: str = "refused") -> None:
        super().__init__(reason)
        self.reason = reason
        self.reason_he = reason_he
        self.code = code


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_of(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def channel_slug(channel: str) -> str:
    """A directory name for a channel, Hebrew letters kept and separators refused.

    ``\\w`` is unicode-aware here, so ``רשת 13`` stays legible as ``רשת-13``
    rather than becoming a hash nobody can read in a directory listing. The
    manifest still records the channel verbatim and every read filters on that,
    so two channels that happened to slug alike could never be confused.
    """
    text = re.sub(r"[^\w-]+", "-", str(channel or "").strip(), flags=re.UNICODE).strip("-")
    if not text or text in {".", ".."}:
        raise ProposalRefused(
            f"channel {channel!r} has no usable directory name",
            "לערוץ אין שם תיקייה תקין",
            code="bad_channel",
        )
    return text


def clean_date(value: Any) -> str:
    text = str(value or "").strip()
    if not _DATE_RE.fullmatch(text):
        raise ProposalRefused(
            f"a broadcast day reads YYYY-MM-DD, got {value!r}",
            "יום שידור נכתב בתבנית YYYY-MM-DD",
            code="bad_date",
        )
    return text


def canonical_bytes(frame: pd.DataFrame) -> bytes:
    """One day's rows as comparable bytes: sorted by segment, column order fixed."""
    if frame is None or frame.empty:
        return b""
    work = frame.copy()
    if "segment_id" in work.columns:
        work = work.sort_values("segment_id", kind="stable")
    work = work.reindex(sorted(work.columns), axis=1)
    return work.to_csv(index=False).encode("utf-8")


def baseline_ref(
    frame: pd.DataFrame,
    *,
    basis: str,
    computed_at: Optional[str] = None,
    plan_sha256: Optional[str] = None,
    settings_basis: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """The identity of the day a proposal was authored against.

    ``settings_basis`` defaults to the week freeze's own reading of the operator
    decision in force, so a day proposal and a week version describe the settings
    they were made under with the same field names and the same values.
    """
    if settings_basis is None:
        from kairos_api.plan_version_store import _settings_basis

        settings_basis = _settings_basis()
    return {
        "basis": str(basis),
        "day_sha256": sha256_of(canonical_bytes(frame)),
        "plan_sha256": plan_sha256,
        "computed_at": computed_at,
        "segments": 0 if frame is None else int(len(frame)),
        "settings_basis": settings_basis,
        "captured_at": now_iso(),
    }


def staleness(manifest: dict[str, Any], current: Optional[dict[str, Any]]) -> dict[str, Any]:
    """Whether the day moved under a proposal, and exactly what moved."""
    authored = (manifest or {}).get("baseline_ref") or {}
    if not isinstance(current, dict) or not current:
        return {
            "known": False, "stale": False,
            "reason": "the day's current baseline could not be read, so staleness is unknown",
            "reason_he": "לא ניתן לקרוא את בסיס היום הנוכחי, ולכן לא ידוע אם ההצעה מעודכנת",
            "moved": [],
        }
    moved: list[dict[str, Any]] = []
    if str(authored.get("basis") or "") != str(current.get("basis") or ""):
        moved.append({
            "field": "basis",
            "before": authored.get("basis"), "after": current.get("basis"),
            "reason_he": "ההצעה נכתבה על בסיס אחר מזה שנקרא כעת",
        })
    if str(authored.get("day_sha256") or "") != str(current.get("day_sha256") or ""):
        moved.append({
            "field": "day_sha256",
            "before": authored.get("day_sha256"), "after": current.get("day_sha256"),
            "reason_he": "שורות היום השתנו מאז שההצעה נכתבה",
        })
    before_settings = authored.get("settings_basis") or {}
    after_settings = current.get("settings_basis") or {}
    for key in sorted(set(before_settings) | set(after_settings)):
        if before_settings.get(key) != after_settings.get(key):
            moved.append({
                "field": f"settings.{key}",
                "before": before_settings.get(key), "after": after_settings.get(key),
                "reason_he": f"הגדרת {key} השתנתה מאז שההצעה נכתבה",
            })
    return {
        "known": True,
        "stale": bool(moved),
        "reason": "" if not moved else "the baseline this proposal was authored against has moved",
        "reason_he": "" if not moved else "הבסיס שעליו נכתבה ההצעה השתנה",
        "moved": moved,
    }
