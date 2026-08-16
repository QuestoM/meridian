"""The licence: the four regulatory numbers, who may move them, and the proof.

The compliance verdict has always been good at its first half. It names seven
checks, an observed figure against a limit, a profile, an effective date and a
source, and it refuses to overstate any of it. Its second half was unbuildable:
the limits it judged against were ordinary settings fields, editable with the
same permission as the revenue slider, with no effective date, no record of who
moved what and no way for the person accountable to prove that nothing had.

This module closes that. It serves the limits from
:mod:`kairos_api.guardrail_store`, which carries the date, the log and the
distinct permission, and it serves an attestation that answers the compliance
owner's actual question in one read: here are the seven checks, here is the
licence they were judged against, here is when it took effect, here is every
change since the day you last signed, and here is any change already recorded
for a future date.

An empty change list is the evidence, not the absence of it, and the payload
says so in words rather than leaving a reader to infer it from a blank.

The activation switch lives here for the reason section 4.1 of the specification
gives: throwing it writes settings, not a file under ``models/``, so it is a
configuration act on the run side and its surface is Rules. The model console
mirrors the state and owns no control.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from kairos_api import guardrail_store, model_activation
from kairos_api.affiliation_wall import ADMIN_ROLES, Wall
from kairos_api.core import _load_settings

logger = logging.getLogger(__name__)

router = APIRouter()

OPERATOR_CHANNEL_ADMIN_ONLY_DETAIL = "בחירת ערוץ המפעיל שמורה למנהל המערכת"

OPERATOR_CHANNEL_OMITTED_DETAIL = "This write leaves out operator_channel, and saving it would clear the declared channel and un-scope every figure in the product. Send the settings model whole, or move the channel through PUT /api/rules/operator-channel."

# The channel is the operator's own identity, so a channel account reads it and
# an admin changes it. The same shape as the licence wall beside it: affiliation
# is not the question here, role is.
CHANNEL_WALL = Wall(
    detail=OPERATOR_CHANNEL_ADMIN_ONLY_DETAIL,
    company_only=False,
    roles=ADMIN_ROLES,
    role_detail=OPERATOR_CHANNEL_ADMIN_ONLY_DETAIL,
)


class GuardrailChange(BaseModel):
    """One recorded change to the licence, with the day it takes force."""

    values: dict[str, Any] = Field(default_factory=dict)
    effective_date: str
    reason: str = ""


class ActivationRequest(BaseModel):
    """The audience model activation switch."""

    active: bool


class OperatorChannelRequest(BaseModel):
    """Which of the channels in the loaded schedule this operator owns."""

    operator_channel: str


def _actor(request: Optional[Request]) -> str:
    from kairos_api.affiliation_wall import session_for

    session = session_for(request)
    return str((session or {}).get("username") or "").strip()


def _require_channel_in_schedule(wanted: str) -> None:
    """Refuse a channel the loaded schedule does not carry, and say what to do.

    A channel nobody broadcasts is not a harmless typo: every scoped figure in
    the product silently becomes an empty set with no explanation.
    """
    from kairos_api._constraint_options import channel_options

    if wanted and wanted not in channel_options():
        raise HTTPException(
            status_code=400,
            detail=f"'{wanted}' is not a channel in the loaded schedule. Load its programmes first.",
        )


def guard_channel_move(incoming: Any, request: Optional[Request] = None) -> None:
    """The declaration rule, for any settings write that is not the declaration route.

    The scoping this workspace ships is exactly as strong as the weakest writer
    of ``operator_channel``, and the weakest one is ``PUT /api/settings``, which
    takes the whole settings model and carries no permission and no validation.
    Measured on the real app with real sessions: an operator-role account of
    either affiliation is refused by ``PUT /api/rules/operator-channel`` and
    allowed by ``PUT /api/settings``, after which every scoped surface serves a
    rival channel as the operator's own. That does not leak a rival's data, it
    inverts the boundary, which is worse.

    So the rule lives here, beside the wall it uses, and the settings route
    calls it in two lines. Three cases, in the order they can bite:

    - The write does not move the channel. Nothing to check, and this is the
      case every shipped client takes, because both of them send the settings
      model whole.
    - The body leaves the field out entirely. Pydantic would default it to the
      empty string and the save would clear the declaration, which un-scopes
      every money figure in the product. ``model_fields_set`` is the only way to
      tell "set it to nothing" from "did not mention it", so the two get
      different answers rather than the same silent one. A plain mapping is
      accepted too and answers the same question with its own keys, so a caller
      that has not parsed the model yet cannot get the tolerant answer by
      accident.
    - The body moves it. Then it is a declaration, and it answers to the same
      role gate and the same schedule validation the declaration route uses,
      because one rule enforced in two places is one rule only while it is the
      same code.
    """
    mapping = incoming if isinstance(incoming, dict) else None
    raw = mapping.get("operator_channel") if mapping is not None else getattr(incoming, "operator_channel", "")
    current = str(getattr(_load_settings(), "operator_channel", "") or "").strip()
    wanted = str(raw or "").strip()
    if wanted == current:
        return
    declared = set(mapping) if mapping is not None else set(getattr(incoming, "model_fields_set", None) or ())
    if "operator_channel" not in declared:
        raise HTTPException(status_code=400, detail=OPERATOR_CHANNEL_OMITTED_DETAIL)
    CHANNEL_WALL.require(request)
    _require_channel_in_schedule(wanted)


@router.get("/api/rules/guardrails", tags=["dashboard"])
def read_guardrails(request: Request = None, on: str = "") -> dict[str, Any]:
    """The limits in force, their date, the whole change log and who may edit."""
    day = None
    if str(on or "").strip():
        try:
            day = date.fromisoformat(str(on).strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"on must be an ISO date, got {on!r}.") from exc
    return guardrail_store.payload(request, day)


@router.post("/api/rules/guardrails", tags=["dashboard"])
def change_guardrails(change: GuardrailChange, request: Request = None) -> dict[str, Any]:
    """Record a change to the licence. The permission is the store's, not settings'.

    A change that is already in force is written through to the settings
    document as well, because the optimizer reads its guardrails from there
    through a frozen seam. Without the write-through the licence and the engine
    would silently disagree, which is the exact defect this store exists to end.
    A change dated in the future is recorded and not written through, and
    :func:`apply_due_guardrails` is the named path that lands it on its day.
    """
    guardrail_store.require_guardrail_editor(request)
    try:
        guardrail_store.record_change(
            change.values,
            effective=change.effective_date,
            actor=_actor(request),
            reason=change.reason,
        )
    except guardrail_store.GuardrailError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _write_through_if_due(request)
    return guardrail_store.payload(request)


def _engine_values() -> dict[str, Any]:
    """The four limits as the engine reads them today, straight off settings."""
    settings = _load_settings()
    return {key: getattr(settings, key, None) for key in guardrail_store.GUARDRAIL_KEYS}


def _write_through_if_due(request: Optional[Request]) -> bool:
    """Materialise the limits in force into settings, so the engine agrees."""
    from kairos_api import core, version_store

    with core._SETTINGS_LOCK:
        # Read the registry inside the same transaction that writes the engine
        # settings. Otherwise two due changes can interleave so an older caller
        # captures stale limits, resumes last, and makes registry and engine
        # disagree even though both individual files were written atomically.
        in_force = guardrail_store.values_on()
        settings = _load_settings()
        current = {key: getattr(settings, key, None) for key in guardrail_store.GUARDRAIL_KEYS}
        if current == in_force:
            return False
        version_store.snapshot_manual_edit(request, "settings")
        for key, value in in_force.items():
            setattr(settings, key, value)
        core._save_settings(settings)
        return True


@router.post("/api/rules/guardrails/apply", tags=["dashboard"])
def apply_due_guardrails(request: Request = None) -> dict[str, Any]:
    """Land every limit that has reached its effective date on the engine.

    The store answers what is in force on a day; the optimizer reads settings.
    This is the one act that makes the second agree with the first, and it is
    explicit rather than implicit because it marks the saved plan out of date
    and the operator has to be told that rather than discover it.
    """
    guardrail_store.require_guardrail_editor(request)
    moved = _write_through_if_due(request)
    body = guardrail_store.payload(request)
    body["applied"] = moved
    return body


@router.get("/api/rules/attestation", tags=["dashboard"])
def attestation(request: Request = None, since: str = "") -> dict[str, Any]:
    """Everything a compliance owner needs to sign, in one read.

    The verdict comes from the same builder ``GET /api/compliance`` uses, so
    there is one set of seven checks in the product and not two that could
    disagree, and it is the scoped one: what a compliance owner signs is a
    verdict on the channel this operator owns, and it carries the scope note
    that says so. ``since`` is the day of the last attestation; without it the
    whole change log is returned and the payload says that is what happened.
    """
    from kairos_api.compliance_api import compliance as scoped_compliance

    verdict = scoped_compliance()
    record = guardrail_store.load_record()
    today = date.today()
    since_day: Optional[date] = None
    if str(since or "").strip():
        try:
            since_day = date.fromisoformat(str(since).strip())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"since must be an ISO date, got {since!r}.") from exc
    changes = guardrail_store.changed_since(since_day, record) if since_day else guardrail_store.changes(record)
    scheduled = guardrail_store.scheduled_changes(today, record)
    in_force = guardrail_store.values_on(today, record)
    engine = _engine_values()
    body = {
        "compliance": verdict,
        "licence": {
            "profile_name": record.get("profile_name", ""),
            "source_url": record.get("source_url", ""),
            "effective_date": guardrail_store.effective_date(today, record),
            "values": in_force,
        },
        "engine_values": engine,
        # The verdict above judges the plan against what the engine ran with. If
        # a recorded change has reached its date and has not been landed, the two
        # differ, and that is a finding rather than something to average away.
        "engine_matches_licence": engine == in_force,
        "since": since_day.isoformat() if since_day else "",
        "since_is_whole_log": since_day is None,
        "changes_since": changes,
        "unchanged": not changes,
        "scheduled_changes": scheduled,
        "checked_on": today.isoformat(),
    }
    return guardrail_store.GUARDRAIL_WALL.stamp(body, request)


@router.get("/api/rules/model-activation", tags=["dashboard"])
def read_activation(request: Request = None) -> dict[str, Any]:
    """The audience model activation switch, its basis and its consequence."""
    return model_activation.payload(request)


@router.put("/api/rules/model-activation", tags=["dashboard"])
def set_activation(payload: ActivationRequest, request: Request = None) -> dict[str, Any]:
    """Throw the switch. Company staff only, refused before the click elsewhere."""
    return model_activation.set_active(payload.active, request)


@router.get("/api/rules/operator-channel", tags=["dashboard"])
def read_operator_channel(request: Request = None) -> dict[str, Any]:
    """Which channel this operator owns, and the channels it may be declared from.

    The declaration is the one act that cannot be performed inside the boundary:
    before it is made the product does not know which channel to hide, so the
    person making it sees every name the loaded schedule carries. No figure and
    no schedule travels with the list, only names and the current selection.

    Everybody else sees the declaration and not the ballot. A reader who cannot
    change the channel has no use for the other names, so the list they get is
    the declared channel alone, which is the same rule the rest of the product
    runs on and leaves the exception exactly as wide as the act that needs it.
    """
    from kairos_api._constraint_options import channel_options

    settings = _load_settings()
    current = str(getattr(settings, "operator_channel", "") or "")
    options = channel_options()
    declarable = CHANNEL_WALL.allows(request)
    body = {
        "operator_channel": current,
        "available_channels": options if declarable else [name for name in options if name == current],
        "is_declared": bool(current),
        "is_in_schedule": bool(current) and current in options,
        "lists_every_channel": declarable,
    }
    return CHANNEL_WALL.stamp(body, request)


@router.put("/api/rules/operator-channel", tags=["dashboard"])
def set_operator_channel(payload: OperatorChannelRequest, request: Request = None) -> dict[str, Any]:
    """Declare the operator's channel, validated against the loaded schedule.

    Two things the free settings write never did. It refuses a channel the
    loaded schedule does not carry, because a channel nobody broadcasts turns
    every scoped figure into an empty set with no explanation. And it is
    role-gated, because setting this to a rival's channel does not leak a
    competitor's data, it inverts the boundary: the product would then hide the
    operator's own channel and treat somebody else's as owned.

    This is the declared writer, not the only one. The settings model still
    carries the field, so :func:`guard_channel_move` puts the same two checks on
    that seam, and until it is called there this route's refusal is a door in a
    wall with a second door beside it.
    """
    from kairos_api.core import _mutate_settings

    CHANNEL_WALL.require(request)
    wanted = str(payload.operator_channel or "").strip()
    _require_channel_in_schedule(wanted)
    from kairos_api import version_store

    def apply(settings: Any) -> None:
        version_store.snapshot_manual_edit(request, "settings")
        settings.operator_channel = wanted

    _mutate_settings(apply)
    return read_operator_channel(request)
