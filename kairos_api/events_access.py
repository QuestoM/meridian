"""Company-only edit access and the training-gate disclosure for the events surface.

Event management belongs to company staff. Accounts carry an ``affiliation``
of company or channel (kairos_api/auth_store.py; missing or malformed stored
values read as unresolved and fail closed); a
channel-affiliated account reads the calendar freely but every event write
(POST/PUT/DELETE /api/events*) and the event pricing activation switch
(PUT /api/pricing carrying ``pricing_activation.events``) answer 403 with a
clear Hebrew detail. Split out of events_api.py to keep that module under the
file-size cap; pricing_api.py imports the same guard so both surfaces enforce
one rule.

The guard is deliberately tolerant where a request identity cannot be resolved: with
auth disabled, with no request object (direct in-process calls, bare-router
tests), or with no resolvable session (the server middleware already walls
unauthenticated API requests with 401 before any route runs) the requester
reads as company. A resolved session whose stored affiliation is channel,
missing or malformed is denied; the latter two normalize to ``unknown`` until
an admin resolves the legacy record.

``training_gate`` turns the ``event_layer_gate`` key that the model rebuild
writes into the coefficients metadata into a tri-state honest block: absent or
malformed metadata reads as verdict unknown, never as a fabricated verdict.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import HTTPException, Request

COMPANY_ONLY_DETAIL = "עריכת אירועים שמורה לצוות החברה"
EVENT_PRICING_COMPANY_ONLY_DETAIL = "הפעלת תמחור אירועים שמורה לצוות החברה"

TRAINING_GATE_UNKNOWN_REASON = "המודל טרם נבנה מחדש עם שכבת האירועים"


def requester_is_company(request: Optional[Request]) -> bool:
    """Whether the requester may manage events. See the module docstring for
    why the unknown-identity paths read True."""
    from kairos_api import auth, auth_store

    if not auth.auth_active():
        return True
    if request is None:
        return True
    session = auth_store.resolve_session(request.cookies.get(auth_store.COOKIE_NAME))
    if session is None:
        return True
    return auth_store.is_company_user(session["username"])


def require_company_editor(request: Optional[Request],
                           detail: str = COMPANY_ONLY_DETAIL) -> None:
    """Raise 403 with a Hebrew detail when a channel-affiliated session tries
    to use a company-only surface."""
    if not requester_is_company(request):
        raise HTTPException(status_code=403, detail=detail)


# The assistant proposal kinds that only company staff may propose; the apply
# side must mirror the propose-time gate, or a channel account could approve a
# pending company-only item it could never have created.
COMPANY_ONLY_PROPOSAL_KINDS = frozenset(
    {"event_change", "agency_change", "agency_link_change", "agency_condition_change"})


def assistant_apply_block(username: str, items: "list[dict[str, Any]]") -> Optional[str]:
    """The Hebrew refusal detail when a non-company actor tries to APPLY a
    company-only proposal item; None when the apply may proceed."""
    from kairos_api import auth, auth_store

    if not auth.auth_active():
        return None
    checker = getattr(auth_store, "is_company_user", None)
    if checker is None or checker(str(username)):
        return None
    for item in items:
        kind = str(item.get("kind", ""))
        if kind in COMPANY_ONLY_PROPOSAL_KINDS:
            return COMPANY_ONLY_DETAIL
        if kind == "pricing_change":
            changes = (item.get("payload") or {}).get("changes") or {}
            activation = changes.get("pricing_activation")
            if isinstance(activation, dict) and "events" in activation:
                return EVENT_PRICING_COMPANY_ONLY_DETAIL
    return None


def training_gate(metadata: "dict[str, Any] | None") -> dict[str, Any]:
    """The event-layer gate verdict from the coefficients metadata.

    Tri-state honest: a metadata block without a valid ``event_layer_gate``
    key (the model has not been rebuilt with the event layer yet) reads as
    verdict unknown with a Hebrew reason, never as on or off.
    """
    gate = (metadata or {}).get("event_layer_gate")
    if not isinstance(gate, dict) or str(gate.get("verdict", "")).strip() not in ("on", "off"):
        return {"verdict": "unknown", "reason": TRAINING_GATE_UNKNOWN_REASON,
                "held_out_delta_pct": None, "measured_at": None}
    delta = gate.get("held_out_delta_pct")
    return {
        "verdict": str(gate["verdict"]).strip(),
        "reason": str(gate.get("reason", "")),
        "held_out_delta_pct": (
            float(delta) if isinstance(delta, (int, float)) and not isinstance(delta, bool)
            else None
        ),
        "measured_at": str(gate.get("measured_at", "")) or None,
    }
