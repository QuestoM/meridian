"""Model console: the honest tri-state status of the trained audience model.

Moved verbatim from insights_api.py as part of the wave-zero router split. The
read itself stays in :mod:`kairos_api.audience_api`, the single reader of the
audience artifact, so this route and the forecast basis note can never disagree.

**Walled.** This route serves eight held-out gate verdicts with their reasons,
which is training content, and section 4.5 of the rebuild specification puts
training content behind ``affiliation = company`` on the read as well as the
write. It is one of the four open reads that section names. Before this wave it
answered every account on the product, which is what made the gate table render
on an operator's calendar page.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter

from kairos_api.model_console_api import MODEL_WALL

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/model/audience", tags=["insights"])
@MODEL_WALL.guard()
def model_audience() -> dict[str, Any]:
    """Honest tri-state status of the trained audience model.

    Frozen payload ``{available, computed_at, activation, gates, base_summary}``;
    the read itself lives in :mod:`kairos_api.audience_api` so this route and
    the forecast basis note can never disagree.
    """
    from kairos_api.audience_api import build_audience_model_payload

    return build_audience_model_payload()
