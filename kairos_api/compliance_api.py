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
    return plan_read_compliance.build_compliance(_load_break_schedule(), _load_settings())
