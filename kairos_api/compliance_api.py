"""Rules: the regulatory compliance verdict on the committed plan.

The route moved verbatim from dashboard_api.py as part of the wave-zero router
split. The verdict itself is a frozen plan read
(:mod:`kairos_api.plan_read_compliance`), because Today prints it in the overview
payload and Sources counts its checks in the reports row, so no single surface
can own it. Behaviour is unchanged: the same seven checks, the same profile,
effective date and source url.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from kairos_api import plan_read_compliance
from kairos_api.core import _load_break_schedule, _load_settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/compliance", tags=["dashboard"])
def compliance() -> dict[str, Any]:
    # Judged against the limits the engine itself runs with, deliberately, and
    # not through the guardrail store's overlay. The optimizer reads its
    # guardrails from the settings document through a frozen seam, so overlaying
    # a future-dated licence here would move the verdict while the plan it
    # judges was still built on the old number. The licence store is the record
    # of what is in force and its own surface names any divergence; this route
    # keeps answering for the plan that exists.
    return plan_read_compliance.build_compliance(_load_break_schedule(), _load_settings())


# The licence, the attestation, the activation switch and the operator channel
# ride on this router: they are the Rules surface's own controls and they share
# this module's mount, so no registration below the marker in server.py changes.
from kairos_api.compliance_api_licence import router as _licence_router  # noqa: E402

router.include_router(_licence_router)
