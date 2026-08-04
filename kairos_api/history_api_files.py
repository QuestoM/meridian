"""The nine logical files a restore writes, and who may put each one back.

The five ``/api/versions`` routes gated on role from the day they were written:
an operator or an administrator may restore, a viewer may not. Nothing gated on
affiliation, and one of the nine logical files is the settings document, which a
restore writes whole.

**Measured on a channel-affiliated operator's own session against a live
server.** That account is refused ``PUT /api/rules/model-activation`` with 403
and the Hebrew detail הפעלת מודל הקהל שמורה לצוות החברה. The same account was
served ``can_edit: true`` by ``GET /api/versions``, posted
``{"files": ["settings"]}`` to a restore, was answered 200, and with that one
write flipped ``audience_model_activation`` from false to true, moved
``max_breaks_per_hour`` to 9 with no effective date and no change record, and
wrote a rival channel into ``operator_channel``, which is the anchor every
scoped figure in this product is computed from. The restore route was a second
door into three settings the product locks on the front one.

This module is that door's lock, and it states the rule as a property of the
FILE rather than of the route, because the same fact holds wherever a restore is
offered.

- **A file whose direct writes are company-only is company-only to put back.**
  Two of the nine are. ``settings`` carries the three things above, and
  ``events`` has all three of its write routes calling ``require_company_editor``
  today, so a channel account that cannot add one event cannot put back the
  whole calendar either.
- **A field that answers to a stricter wall than its file answers to it on the
  way back too.** A regulatory limit is company staff and an administrator, by
  ``guardrail_store.GUARDRAIL_WALL`` and the owner's ruling of 2026-08-01; the
  channel declaration is an administrator of either affiliation, by
  ``compliance_api_licence.CHANNEL_WALL``. So a restore that would move one is
  refused unless the caller passes that same wall, and a restore that moves
  neither is the ordinary restore it has always been.
- **The refusal is legible before the click.** :func:`permissions` stamps the
  same ``can_edit`` and the same reason string per logical file that the 403
  would carry, so a control renders as state rather than failing after the act.

What is deliberately not applied here is the schedule validation that
``PUT /api/rules/operator-channel`` runs on a new declaration. A restore puts
back a value this deployment recorded rather than declaring a new one, and
loading the whole programme table to second-guess a recorded value would refuse
a legitimate undo. The permission is the question this module answers.
"""

from __future__ import annotations

import importlib
from typing import Any, Iterable, Optional

from fastapi import HTTPException, Request

from kairos_api import version_store
from kairos_api.affiliation_wall import (
    ADMIN_ROLES,
    COMPANY_ONLY_DETAIL,
    READ_ONLY_ROLE_DETAIL,
    Wall,
)

# The nine, in the store's own order, so this module cannot hold a tenth or miss
# one the store adds.
LOGICAL_FILES = version_store._LOGICAL_ORDER

SETTINGS_RESTORE_DETAIL = "שחזור קובץ ההגדרות שמור לצוות החברה"

# Used only if the owning module's wall cannot be resolved; see :func:`_wall`.
LICENCE_RESTORE_DETAIL = "שחזור מגבלות הרגולציה שמור לצוות החברה ולמנהל המערכת"
CHANNEL_RESTORE_DETAIL = "שחזור ערוץ המפעיל שמור למנהל המערכת"

# Eight of the nine: role decides, affiliation does not. This is the wall the
# whole surface reads with, so the listing, the diff and the timeline answer the
# same question with one object.
RESTORE_WALL = Wall(detail=READ_ONLY_ROLE_DETAIL, company_only=False)

# The two whose direct writes are company-only, so putting them back is too.
FILE_WALLS: dict[str, Wall] = {
    "settings": Wall(detail=SETTINGS_RESTORE_DETAIL, company_only=True),
    "events": Wall(detail=COMPANY_ONLY_DETAIL, company_only=True),
}

__all__ = [
    "CHANNEL_RESTORE_DETAIL",
    "FILE_WALLS",
    "LICENCE_RESTORE_DETAIL",
    "LOGICAL_FILES",
    "RESTORE_WALL",
    "SETTINGS_RESTORE_DETAIL",
    "guarded_fields",
    "permissions",
    "public_entry",
    "require_restore",
    "settings_move_reason",
    "withheld",
]


def _wall(module_name: str, attribute: str, fallback: Wall) -> Wall:
    """The owning module's wall, or a fallback that is never weaker than it.

    The field walls belong to the modules that own the fields, so a restore and
    a direct write answer to one rule rather than to two that can drift. They
    resolve at call time rather than at import, because those modules are route
    layers that import this package back. A missing or renamed symbol falls back
    to a wall with the same two gates rather than to no gate at all, and
    ``tests/test_p8_restore.py`` asserts that the real ones resolve today, so the
    fallback is a floor and never the live answer.
    """
    try:
        found = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError):  # pragma: no cover - pinned by the test above
        return fallback
    return found if isinstance(found, Wall) else fallback


def _channel_wall() -> Wall:
    return _wall(
        "kairos_api.compliance_api_licence", "CHANNEL_WALL",
        Wall(detail=CHANNEL_RESTORE_DETAIL, company_only=False, roles=ADMIN_ROLES,
             role_detail=CHANNEL_RESTORE_DETAIL),
    )


def _guardrail_wall() -> Wall:
    return _wall(
        "kairos_api.guardrail_store", "GUARDRAIL_WALL",
        Wall(detail=LICENCE_RESTORE_DETAIL, company_only=True, roles=ADMIN_ROLES,
             role_detail=LICENCE_RESTORE_DETAIL),
    )


def _activation_wall() -> Wall:
    return _wall(
        "kairos_api.model_activation", "ACTIVATION_WALL",
        Wall(detail=SETTINGS_RESTORE_DETAIL, company_only=True,
             role_detail=SETTINGS_RESTORE_DETAIL),
    )


def _field_walls() -> tuple[tuple[frozenset[str], Wall], ...]:
    """Each guarded settings field beside the wall that owns it, in check order.

    The order is the order a refusal is reported in, so a point that moves two
    of them names the first rather than an aggregate nobody can act on.
    """
    from kairos_api import guardrail_store, model_activation

    return (
        (frozenset({model_activation.SETTINGS_FIELD}), _activation_wall()),
        (frozenset(guardrail_store.GUARDRAIL_KEYS), _guardrail_wall()),
        (frozenset({"operator_channel"}), _channel_wall()),
    )


def guarded_fields(logical: str) -> tuple[str, ...]:
    """The fields inside a logical file that carry a wall of their own.

    Field names, never sentences: the surface says what they are in the reader's
    own language, which is the rule every payload on this destination follows.
    """
    if logical != "settings":
        return ()
    from kairos_api import guardrail_store, model_activation

    return (model_activation.SETTINGS_FIELD, *guardrail_store.GUARDRAIL_KEYS, "operator_channel")


def wall_for(logical: str) -> Wall:
    """The wall that decides whether this caller may put this file back."""
    return FILE_WALLS.get(str(logical), RESTORE_WALL)


def withheld(request: Optional[Request]) -> tuple[str, ...]:
    """The files this caller is on the wrong side of the line for.

    Affiliation only. A viewer is refused every file on role, which the payload's
    own ``can_edit`` already says once, and printing nine locks over one sentence
    would say it nine times.
    """
    return tuple(name for name, wall in FILE_WALLS.items() if wall.read_reason(request) is not None)


def settings_move_reason(changed: Optional[Iterable[dict[str, Any]]],
                         request: Optional[Request]) -> Optional[str]:
    """Why this caller may not put back these settings changes, or None.

    ``changed`` is the settings diff the store already computes: one row per
    field, with the value now and the value at that point. A field that does not
    move is not a change and asks no permission, which is what keeps an ordinary
    restore ordinary.
    """
    moved = {str(row.get("field") or "") for row in (changed or ())}
    for fields, wall in _field_walls():
        if moved & fields:
            reason = wall.reason(request)
            if reason is not None:
                return reason
    return None


def permissions(request: Optional[Request],
                settings_changed: Optional[Iterable[dict[str, Any]]] = None) -> dict[str, Any]:
    """Per logical file: may this caller put it back, and the reason if not.

    ``settings_changed`` refines the settings answer with the fields one named
    version would actually move, which is what the diff route knows and the
    listing does not. Without it the answer is the file's own wall, which is the
    strictest thing that can be said without reading a version.
    """
    body: dict[str, Any] = {}
    for logical in LOGICAL_FILES:
        wall = wall_for(logical)
        reason = wall.reason(request)
        if reason is None and logical == "settings" and settings_changed is not None:
            reason = settings_move_reason(settings_changed, request)
        entry: dict[str, Any] = {
            "can_edit": reason is None,
            "company_only": wall.company_only,
            "guards": list(guarded_fields(logical)),
        }
        if reason is not None:
            entry["can_edit_reason"] = reason
        body[logical] = entry
    return body


def public_entry(manifest: dict[str, Any], block: Optional[str],
                 withheld_files: Iterable[str] = ()) -> dict[str, Any]:
    """One version as a surface reads it.

    ``restorable`` and ``restore_block`` are the tri-state a restore control
    needs before it renders: restorable, blocked with a named reason, or, when
    the manifest names no known logical file, nothing to put back. They are
    properties of the version. ``withheld_files`` is the other question, and it
    is a property of the reader: which of the files this point covers are on the
    other side of the line from the account asking.
    """
    covered = [item.get("logical") for item in manifest.get("files", [])]
    blocked = set(withheld_files)
    return {
        "version_id": manifest.get("version_id"),
        "created_at": manifest.get("created_at"),
        "actor": manifest.get("actor"),
        "source": manifest.get("source"),
        "label": manifest.get("label"),
        "batch_id": manifest.get("batch_id"),
        "files": covered,
        "restorable": block is None,
        "restore_block": block,
        "withheld_files": [name for name in covered if name in blocked],
    }


def require_restore(version_id: str, selected: Iterable[str],
                    request: Optional[Request]) -> None:
    """Every gate this restore must pass, raised before anything is written.

    Called after the selection is resolved and before the pre-restore safety
    point, so a refused restore leaves the store exactly as it found it: no
    safety version, no audit line, nothing put back.
    """
    chosen = [str(name) for name in selected]
    for logical in chosen:
        if logical in FILE_WALLS:
            FILE_WALLS[logical].require(request)
    if "settings" not in chosen:
        return
    diff = version_store._diff_logical(str(version_id), "settings") or {}
    reason = settings_move_reason(diff.get("changed") or [], request)
    if reason is not None:
        raise HTTPException(status_code=403, detail=reason)
