"""The audience model activation switch: run-side configuration, company-gated.

Whether runs consume the trained audience model is a configuration act, not a
training act. Throwing it writes ``data/kairos_settings.json``, and by the
classification rule an act is training only when its output is a file under
``models/``. It also changes the freshness fingerprint, so the saved plan goes
stale and the operator is asked to run it again. Both of those are run-side
facts, which is why the control lives on the rules surface and never in the
model console.

What it is gated on is a different question from where it lives. It moves
money, so only a company-affiliated account may throw it. The product already
does exactly this for the event pricing activation switch, refused with its own
Hebrew detail in ``pricing_api.py``, so this reuses that guard rather than
inventing a second one.

The read side is deliberately thin. It returns the switch, the three-state
basis the engine already computes, and ``can_edit`` with its reason. It carries
no gate verdict, no coverage, no coefficient and no p-value, so it passes the
lexicon check on every operator surface, and keeping it that way is a duty of
whoever owns this module next, not an accident of the current shape.

The three states, which are the artifact's own and not this module's opinion:
``off`` means forward-dated ratings are the historical baseline, ``on`` names
the model version's date, and ``on_no_artifact`` means the switch is on and
nothing is trained, so the numbers are still historical.
"""

from __future__ import annotations

from typing import Any, Optional

from fastapi import Request

from kairos_api.affiliation_wall import Wall

SETTINGS_FIELD = "audience_model_activation"

AUDIENCE_MODEL_COMPANY_ONLY_DETAIL = "הפעלת מודל הקהל שמורה לצוות החברה"

# Affiliation is the outer gate, so a channel administrator may not throw it.
# Role is still the inner one, so a company viewer sees the state and not the
# control. That is section 4.5's sentence with no exception.
ACTIVATION_WALL = Wall(
    detail=AUDIENCE_MODEL_COMPANY_ONLY_DETAIL,
    company_only=True,
    role_detail=AUDIENCE_MODEL_COMPANY_ONLY_DETAIL,
)

# The consequence, stated before the click in the operator's own language.
CONSEQUENCE_HE = "שינוי המתג משנה את מקור הרייטינג לתאריכים עתידיים, ולכן התוכנית השמורה תסומן כלא עדכנית ותידרש הרצה מחדש."
CONSEQUENCE_EN = "Flipping the switch changes where forward-dated ratings come from, so the saved plan is marked out of date and will need a run."

__all__ = [
    "ACTIVATION_WALL",
    "AUDIENCE_MODEL_COMPANY_ONLY_DETAIL",
    "CONSEQUENCE_EN",
    "CONSEQUENCE_HE",
    "SETTINGS_FIELD",
    "is_active",
    "payload",
    "require_activation_editor",
    "set_active",
]


def is_active() -> bool:
    """The saved switch. An absent key reads off, which is the shipped default."""
    from kairos_api.core import _load_settings

    return bool(getattr(_load_settings(), SETTINGS_FIELD, False))


def state() -> dict[str, Any]:
    """The engine's own basis note: ``{state, computed_at}`` and nothing else."""
    from kairos_api.core import _audience_model_note_safe

    return _audience_model_note_safe()


def require_activation_editor(request: Optional[Request]) -> None:
    """Raise 403 with the Hebrew denial unless this session may throw the switch."""
    ACTIVATION_WALL.require(request)


def payload(request: Optional[Request] = None) -> dict[str, Any]:
    """The honest read: the switch, its basis, its consequence and can_edit.

    A channel account gets the same body with ``can_edit`` false and the reason
    it would be refused, so the control renders as state before the click
    instead of failing after it.
    """
    basis = state()
    body = {
        "field": SETTINGS_FIELD,
        "active": is_active(),
        "state": basis.get("state"),
        "computed_at": basis.get("computed_at"),
        "consequence_he": CONSEQUENCE_HE,
        "consequence_en": CONSEQUENCE_EN,
    }
    return ACTIVATION_WALL.stamp(body, request)


def set_active(active: bool, request: Optional[Request] = None) -> dict[str, Any]:
    """Throw the switch, after the gate. Returns the payload the read returns.

    Saves through the same settings seam every other configuration act uses, so
    the freshness fingerprint sees the flip and the saved plan is marked out of
    date exactly as it is today.
    """
    require_activation_editor(request)
    from kairos_api.core import _load_settings, _save_settings

    settings = _load_settings()
    setattr(settings, SETTINGS_FIELD, bool(active))
    _save_settings(settings)
    return payload(request)
