"""What the acting account may actually change, asked before Kai proposes it.

Rule 7 of Kai's contract is that it never reaches a control the person
themselves cannot reach, and that a refusal legible to them before the click is
legible to Kai before the proposal. Two of the fields Kai may propose are not
ordinary settings at all: they are the operator's broadcast licence.

``kairos_api.guardrail_store`` holds the four regulatory limits with an
effective date, an append-only change record and a permission of their own, and
this module is the seam that carries that fact into a proposal. A settings
proposal touching one of the four gains a ``permission`` block naming the limit,
the date the limits in force took effect, whether this account may change them
and, when it may not, the same Hebrew reason the store's own refusal uses. The
proposal card prints it before the approval.

**It discloses and refuses through the same owner.** The four values still live
on ``KairosSettings`` because the optimizer reads them there, but the generic
``PUT /api/settings`` rejects a moved regulatory value. A real change goes
through ``/api/rules/guardrails`` with the store's permission, effective date
and append-only record; this module carries that same permission into a Kai
proposal before approval.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

GUARDRAIL_BASIS_HE = "מגבלת רגולציה: אחת מארבע המגבלות של רישיון השידור"
GUARDRAIL_BASIS_EN = "Regulatory limit: one of the four broadcast licence limits"
GUARDRAIL_OWNER_HE = "שינוי המגבלות נרשם ביומן השינויים עם תאריך תחילה"
GUARDRAIL_OWNER_EN = "A change to the limits is recorded with the date it takes effect"


def guardrail_fields(changes: Any) -> list[str]:
    """The licence limits this settings change would move, sorted. Empty for an
    ordinary settings change, which is every other field."""
    from kairos_api import guardrail_store

    if not isinstance(changes, dict):
        return []
    return sorted(set(changes) & set(guardrail_store.GUARDRAIL_KEYS))


def actor_may_change_guardrails(user: "str | None") -> tuple[bool, "str | None"]:
    """Whether this account may change the licence limits, and why not.

    The answer comes from the store's own wall rather than from a rule restated
    here, so the two can never disagree. Unknown identity is permitted, exactly
    as the wall itself is: with authentication off there is no account to refuse.
    """
    from kairos_api import auth, auth_store, guardrail_store

    if not auth.auth_active():
        return True, None
    try:
        record = auth_store.get_user(str(user or ""))
    except Exception:  # noqa: BLE001 - an unreadable store must not open the gate
        logger.exception("reading the account %r failed", user)
        return False, guardrail_store.GUARDRAIL_ADMIN_ONLY_DETAIL
    if record is None:
        return True, None
    if str(record.get("role") or "") in guardrail_store.GUARDRAIL_WALL.roles:
        return True, None
    return False, guardrail_store.GUARDRAIL_ADMIN_ONLY_DETAIL


def guardrail_permission(changes: Any, user: "str | None") -> "dict[str, Any] | None":
    """The permission block for a settings change, or None when it moves no limit."""
    from kairos_api import guardrail_store

    fields = guardrail_fields(changes)
    if not fields:
        return None
    may_change, reason = actor_may_change_guardrails(user)
    block: dict[str, Any] = {
        "fields": fields,
        "may_change": may_change,
        "basis_he": GUARDRAIL_BASIS_HE,
        "basis_en": GUARDRAIL_BASIS_EN,
        "record_he": GUARDRAIL_OWNER_HE,
        "record_en": GUARDRAIL_OWNER_EN,
    }
    if reason:
        block["reason"] = reason
    try:
        block["effective_date"] = guardrail_store.effective_date()
    except Exception:  # noqa: BLE001 - an absent date is absent, never invented
        logger.exception("reading the guardrail effective date failed")
    return block


def refusal(name: str, args: dict[str, Any], user: "str | None") -> "str | None":
    """The refusal a guardrail-touching proposal would carry, or None.

    Not called from the propose path today, for the reason in this module's
    docstring. It is the one line that turns the disclosure into a refusal the
    day the licence limits leave the unguarded settings document.
    """
    if name != "propose_settings_change":
        return None
    permission = guardrail_permission((args or {}).get("changes"), user)
    if permission is None or permission.get("may_change"):
        return None
    return str(permission.get("reason") or "")
