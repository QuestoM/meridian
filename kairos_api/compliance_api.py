"""Rules: the regulatory compliance verdict on the operator's own plan.

The route moved verbatim from dashboard_api.py as part of the wave-zero router
split. The verdict itself is a frozen plan read
(:mod:`kairos_api.plan_read_compliance`), because Today prints it in the overview
payload and Sources counts its checks in the reports row, so no single surface
can own it. The seven checks, the profile, the effective date and the source url
are unchanged, and both of the frozen module's own halves are still what compute
them: its geometry function supplies the breaks, its verdict function grades
them, and its whole builder is still the fallback when the geometry does not
join.

What this module adds is the population. The frozen builder grades the FULL
committed plan, and the committed plan is the whole market, because the retention
model is measured against the competitive lineup. Measured on the reference plan:
9,026 breaks graded, of which 6,635, 73.5 percent, belong to three channels this
operator does not own. The break that set the printed retention figure was
``כאן 11``, 2024-11-01, hour 20, and 136 of the 200 violations in the payload the
operator's browser held named a rival. No wrong number was on screen, because the
operator's own worst break is numerically identical today, but the figure was
drawn from the wrong population and the payload carried a rival's identity.

One graded population, one scope note printed beside the figures, and no rival
name anywhere in the body. The other two readers of the verdict still call the
frozen builder directly and are still unscoped; adopting this is one line for
each of them and the route below is the entry point.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from kairos_api import channel_scope, plan_read_compliance, plan_read_guardrails
from kairos_api.core import _load_break_schedule, _load_settings

logger = logging.getLogger(__name__)

router = APIRouter()

# The act that supplies the missing input, named on the empty state rather than
# described, so a reader who cannot see a verdict is told where to go.
DECLARE_CHANNEL_ROUTE = "PUT /api/rules/operator-channel"

NO_CHANNEL_EN = "No channel is declared, so there is no population to judge. Declare the channel this operator owns and the verdict is computed for it."
NO_CHANNEL_HE = "לא הוצהר ערוץ, ולכן אין אוכלוסייה לשפוט. הצהירו על הערוץ שבבעלות המפעיל וחוות הדעת תחושב עבורו."


def _envelope(settings: Any, scope: dict[str, Any]) -> dict[str, Any]:
    """The licence half of the payload, which is the same whatever was graded."""
    return {
        "profile": settings.profile_name,
        "effective_date": settings.effective_date,
        "source_url": settings.regulatory_source_url,
        "disclaimer": settings.notes,
        "scope": scope,
    }


@router.get("/api/compliance", tags=["dashboard"])
def compliance() -> dict[str, Any]:
    """The seven checks over the operator's own breaks, with the scope on them.

    Judged against the limits the engine itself runs with, deliberately, and not
    through the guardrail store's overlay. The optimizer reads its guardrails
    from the settings document through a frozen seam, so overlaying a
    future-dated licence here would move the verdict while the plan it judges was
    still built on the old number. The licence store is the record of what is in
    force and its own surface names any divergence; this route keeps answering
    for the plan that exists.

    With no channel declared the boundary cannot be applied, so no verdict is
    served. The pass-through form, the whole market with a note, is deliberately
    not offered: serving the market as if it were the operator's is the exact
    thing this closes, and an honest empty state that names the missing input is
    the only other true answer.
    """
    settings = _load_settings()
    channel = channel_scope.operator_channel(settings)
    items = plan_read_guardrails.plan_guardrail_items()
    channels_in = len({str(item.channel).strip() for item in items if str(item.channel).strip()})

    if not channel:
        scope = channel_scope.scope_note("", len(items), 0, channels_in, scoped=False)
        scope["supply_route"] = DECLARE_CHANNEL_ROUTE
        scope["reason_en"] = NO_CHANNEL_EN
        scope["reason_he"] = NO_CHANNEL_HE
        return {
            **_envelope(settings, scope),
            "checks": [],
            "violations": [],
            "status": "unknown",
            "graded_breaks": 0,
        }

    owned = [item for item in items if str(item.channel).strip() == channel]
    scope = channel_scope.scope_note(channel, len(items), len(owned), channels_in, scoped=True)
    verdict = plan_read_guardrails.guardrail_compliance_from_breaks(owned, settings)
    if verdict is not None:
        return {
            **_envelope(settings, scope),
            "checks": verdict["checks"],
            "violations": verdict["violations"],
            "status": verdict["status"],
            # Which optional caps ran and which did not. Without this the
            # verdict would read as though every expressible rule had been
            # applied, and a plan would carry a badge nobody's rule earned.
            "optional_caps": verdict["optional_caps"],
            "cap_states": verdict["cap_states"],
            "graded_breaks": len(owned),
        }

    # The break geometry could not be joined, which the frozen module reports as
    # an empty tuple and leaves the caller to fall back on. The fallback is the
    # frozen builder over the saved plan, and the plan is scoped the same way
    # before it is handed over, so neither path can grade a rival.
    frame, frame_scope = channel_scope.scope_frame(_load_break_schedule(), settings=settings)
    fallback = plan_read_compliance.build_compliance(frame, settings)
    scope = {**scope, "rows_in": frame_scope["rows_in"], "rows_out": frame_scope["rows_out"]}
    scope["basis"] = "saved plan summary, because the break geometry did not join"
    return {
        **_envelope(settings, scope),
        "checks": fallback["checks"],
        "violations": fallback["violations"],
        "status": fallback["status"],
        "optional_caps": fallback["optional_caps"],
        "cap_states": fallback["cap_states"],
        "graded_breaks": 0,
    }


# The licence, the attestation, the activation switch and the operator channel
# ride on this router: they are the Rules surface's own controls and they share
# this module's mount, so no registration below the marker in server.py changes.
from kairos_api.compliance_api_licence import router as _licence_router  # noqa: E402

router.include_router(_licence_router)
