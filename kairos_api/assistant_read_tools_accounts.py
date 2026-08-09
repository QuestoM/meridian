"""The account administrator's persona, which had no tool at all.

The coverage audit scored accounts, roles and affiliation as read *none* and
propose *n/a*, and the n/a is a ruling rather than an omission: creating an
account, resetting a password and moving somebody's affiliation are credential
acts, and a review-first assistant that could stage them would be staging a way
into the product. **Nothing here proposes anything.** This is the read half, and
the read half is the whole of what the persona was missing.

Four things it answers, each sourced from the module that owns it rather than
restated:

* **The roles and what each may change.** ``auth_store.ROLES`` and the two role
  sets in ``affiliation_wall`` are the same objects the walls consult, so the
  answer Kai gives and the answer a click gets cannot drift.
* **The affiliation rule.** Company or channel, the outer gate, which decides
  what somebody may SEE and never what they may change. It is the distinction
  ``affiliation_wall`` exists to keep separate and the one most often asked
  about backwards.
* **The four licence limits, stated.** This closes a named finding: the propose
  path already REFUSES a settings change touching a broadcast-licence limit
  through ``assistant_permissions``, and no tool listed the four, so Kai could
  decline to move a number it could not name. It now names them, with their
  current values, the date the limits in force took effect, and who may move
  them. Reading them is deliberately open, which is ``guardrail_store``'s own
  ruling: the licence is the broadcaster's own and so is the person who attests
  to it, and what they cannot do is move a number.
* **The roster, to an administrator only.** ``GET /api/auth/users`` is admin
  gated, so this tool is too, and a non-admin gets every rule above and no
  names. A refusal states itself rather than returning an empty list, because
  an empty roster and a walled one are different facts and one of them is a lie.

No password material reaches this module: the roster is built through
``auth._public_user``, the same projection the route serves, which is where the
credential fields are dropped.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

ROSTER_ADMIN_ONLY_EN = "The account roster is administrator-only, so it is not included here. Every rule below applies whoever is asking."
ROSTER_ADMIN_ONLY_HE = "רשימת החשבונות שמורה למנהל מערכת בלבד, ולכן אינה נכללת כאן. כל הכללים שלמטה חלים על כל שואל."
AUTH_OFF_EN = "Authentication is switched off in this deployment, so there are no accounts and every surface is open."
AUTH_OFF_HE = "האימות כבוי בפריסה הזו, ולכן אין חשבונות וכל המסכים פתוחים."

# What each role may do, in the terms the walls actually enforce. Read from the
# role sets rather than written out, so a role added to either set appears here.
_ROLE_NOTE_EN = {
    "admin": "May change anything a role gates, including the broadcast licence limits, and is the only role that manages accounts.",
    "operator": "May change the operational stores (settings, pricing, constraints, overrides, campaigns, pacing decisions) but not the licence limits and not accounts.",
    "viewer": "Reads its side of the affiliation line and changes nothing.",
}
_ROLE_NOTE_HE = {
    "admin": "רשאי לשנות כל מה שתפקיד חוסם, כולל מגבלות רישיון השידור, והוא התפקיד היחיד שמנהל חשבונות.",
    "operator": "רשאי לשנות את מאגרי התפעול (הגדרות, תמחור, אילוצים, עקיפות, קמפיינים והחלטות בספר ההחלטות) אך לא את מגבלות רישיון השידור ולא חשבונות.",
    "viewer": "קורא את המסכים הפתוחים לצוות שלו ואינו משנה דבר.",
}
_AFFILIATION_NOTE_EN = {
    "company": "Company staff. Sees the surfaces reserved to the company side, including the model training and adoption views.",
    "channel": "Broadcaster staff. Sees the operational product and is walled out of the company-only surfaces; affiliation decides seeing, never changing.",
}
_AFFILIATION_NOTE_HE = {
    "company": "צוות החברה. רואה את המסכים השמורים לצוות החברה, כולל מסכי אימון המודל והאימוץ.",
    "channel": "צוות הערוץ. רואה את מוצר התפעול, והמסכים השמורים לצוות החברה סגורים בפניו; הקביעה הזו מחליטה מה נראה ולא מה ניתן לשנות.",
}


def _roles() -> list[dict[str, Any]]:
    from kairos_api import auth_store
    from kairos_api.affiliation_wall import ADMIN_ROLES, WRITE_ROLES

    return [
        {
            "role": role,
            "may_change_anything_role_gates": role in WRITE_ROLES,
            "manages_accounts": role in ADMIN_ROLES,
            "may_change_licence_limits": role in ADMIN_ROLES,
            "note_en": _ROLE_NOTE_EN.get(role, ""),
            "note_he": _ROLE_NOTE_HE.get(role, ""),
        }
        for role in auth_store.ROLES
    ]


def _affiliations() -> list[dict[str, Any]]:
    from kairos_api import auth_store

    return [
        {"affiliation": name, "note_en": _AFFILIATION_NOTE_EN.get(name, ""),
         "note_he": _AFFILIATION_NOTE_HE.get(name, "")}
        for name in auth_store.AFFILIATIONS
    ]


def _licence_limits() -> dict[str, Any]:
    """The four numbers that are the licence, with who may move them.

    Every figure comes from the guardrail store, which is the register of record
    for them; a value it cannot produce is reported absent rather than filled in
    from the settings document, because the two can legitimately differ while a
    dated change is pending and reporting one as the other would hide exactly
    that.
    """
    from kairos_api import guardrail_store
    from kairos_api.affiliation_wall import ADMIN_ROLES

    out: dict[str, Any] = {
        "keys": list(guardrail_store.GUARDRAIL_KEYS),
        "may_change_roles": sorted(ADMIN_ROLES),
        "company_staff_only": bool(guardrail_store.GUARDRAIL_WALL.company_only),
        "read_is_open": True,
    }
    try:
        out["values"] = guardrail_store.current_values()
        out["effective_date"] = guardrail_store.effective_date()
    except Exception:  # noqa: BLE001 - an unreadable register is absent, never invented
        logger.exception("reading the guardrail register failed")
        out["values_unavailable"] = "the guardrail register could not be read; the limits in force are unknown rather than default"
    return out


def _roster(user: str | None) -> dict[str, Any]:
    """The accounts, to an administrator; a stated refusal to anybody else."""
    from kairos_api import auth, auth_store

    if not auth.auth_active():
        return {"available": False, "reason_en": AUTH_OFF_EN, "reason_he": AUTH_OFF_HE}
    record = auth_store.get_user(str(user or "")) or {}
    if str(record.get("role") or "") not in {"admin"}:
        return {"available": False, "reason_en": ROSTER_ADMIN_ONLY_EN, "reason_he": ROSTER_ADMIN_ONLY_HE}
    accounts = [auth._public_user(entry) for entry in auth_store.load_users()]
    return {"available": True, "count": len(accounts), "accounts": accounts}


def _read_get_accounts(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    del args
    return {
        "roles": _roles(),
        "affiliations": _affiliations(),
        "licence_limits": _licence_limits(),
        "roster": _roster(user),
        "acting_account": str(user or "") or None,
        "proposing_account_changes": "not available: creating an account, resetting a password and changing an affiliation are credential acts and no assistant tool stages them",
    }


ACCOUNTS_READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_accounts",
        "description": (
            "Read how access works in this product: the roles and exactly what each may "
            "change, the company/channel affiliation rule and what it decides (seeing, never "
            "changing), the four broadcast-licence limits with their current values, the date "
            "the limits in force took effect and which role may move them, and the account "
            "roster. The roster is administrator-only and says so plainly when the asker is "
            "not one, rather than coming back empty. Call this when the operator asks who can "
            "do what, why a control is refused or greyed out, what the difference between "
            "company and channel is, which numbers are the licence rather than policy, or "
            "anything about accounts. There is no tool that changes an account: creating one, "
            "resetting a password and moving an affiliation are credential acts and the "
            "assistant stages none of them."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]

_ACCOUNTS_READ_EXECUTORS = {"get_accounts": _read_get_accounts}

# Where the answer came from, in the two registers an operator already knows by
# name: the account list the admin dialog manages, and the four limits that are
# the broadcast licence. It does not enumerate the fields, which is what pushed
# the earlier wording into needing a Hebrew noun for "affiliation" that this
# product does not have.
ACCOUNTS_SOURCE_BY_TOOL = {
    "get_accounts": "the account list and the broadcast licence limits",
}


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    """Merge this executor and its source label into the shared registry."""
    executors.update(_ACCOUNTS_READ_EXECUTORS)
    sources.update(ACCOUNTS_SOURCE_BY_TOOL)
