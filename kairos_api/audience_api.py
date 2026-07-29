"""Audience-model status reads: the artifact, the API payload, the basis note.

One small module owns every read of ``models/audience_model.json`` on the API
side, so the ``GET /api/model/audience`` payload (mounted through the insights
router) and the basis note the forecast surfaces attach can never disagree.
Everything here is read-only and tolerant: an absent or unreadable artifact is
an honest ``available: false`` with the reason, never an error and never an
invented gate verdict. The activation flag itself lives on the operator
settings (``audience_model_activation``, default off), exactly like the
pricing-overrides activation pattern.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

ARTIFACT_NAME = "audience_model.json"


def _read_artifact() -> Optional[dict[str, Any]]:
    """The parsed artifact dict, or None when absent or unreadable (logged)."""
    from kairos_api.core import MODELS_DIR

    path = MODELS_DIR / ARTIFACT_NAME
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("audience model artifact at %s is unreadable; treating as absent", path)
        return None
    return payload if isinstance(payload, dict) else None


def _activation() -> bool:
    """The operator's saved activation flag (absent reads False)."""
    from kairos_api.core import _load_settings

    return bool(getattr(_load_settings(), "audience_model_activation", False))


def scalar_base_summary(base: Any) -> "dict[str, Any] | None":
    """The competitor-safe view of the artifact's base block: scalars only.

    The full base carries per-channel maps keyed by every channel name in the
    training data, including rival channels. Those names must never reach the
    assistant context or an API payload (the competitor boundary), so every
    nested dict or list is dropped and only scalar facts (kind, observation
    count, the owned channel, shrinkage constants) ship."""
    if not isinstance(base, dict):
        return None
    return {key: value for key, value in base.items()
            if isinstance(value, (str, int, float, bool)) or value is None}


def build_audience_model_payload() -> dict[str, Any]:
    """The frozen ``GET /api/model/audience`` contract payload.

    ``{available, computed_at, activation, gates, base_summary}``: honest
    tri-state when the artifact is absent (available false with the reason,
    activation still reported truthfully), the artifact's own gate verdicts and
    pooled-base summary when present. Nothing is recomputed here; this is a
    faithful read of what the rebuild wrote.
    """
    activation = _activation()
    artifact = _read_artifact()
    if artifact is None:
        return {
            "available": False,
            "computed_at": None,
            "activation": activation,
            "gates": {},
            "base_summary": None,
            "reason": "No trained audience-model artifact on disk (models/audience_model.json); run the rebuild to train it.",
        }
    gates = artifact.get("gates")
    computed_at = artifact.get("computed_at")
    return {
        "available": True,
        "computed_at": str(computed_at) if computed_at else None,
        "activation": activation,
        "gates": gates if isinstance(gates, dict) else {},
        "base_summary": scalar_base_summary(artifact.get("base")),
    }


def audience_model_note() -> dict[str, Any]:
    """The basis note the overview/forecast payloads carry beside their numbers.

    Names the audience-model state so the dashboard can label forecast figures
    honestly: ``off`` (forecast ratings are the historical baseline),
    ``on`` with the artifact's ``computed_at`` (forward-dated ratings come from
    the model), or ``on_no_artifact`` (the flag is on but nothing is trained,
    so the numbers are still historical). Never guesses a state.
    """
    activation = _activation()
    artifact = _read_artifact()
    computed_at = None
    if artifact is not None and artifact.get("computed_at"):
        computed_at = str(artifact.get("computed_at"))
    # The state enum plus computed_at carry the whole meaning (off = forecast
    # ratings are the historical baseline; on_no_artifact = flag on, nothing
    # trained, still historical; on = forward dates come from the model). No
    # prose field: the note rides every overview payload including the
    # assistant's budgeted context, where each byte competes with data rows.
    if not activation:
        return {"state": "off", "computed_at": computed_at}
    if computed_at is None:
        return {"state": "on_no_artifact", "computed_at": None}
    return {"state": "on", "computed_at": computed_at}
