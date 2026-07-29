"""The audience-model disclosure for the assistant: one honest artifact summary.

:func:`audience_model_summary` powers the ``get_audience_model`` READ tool and
the ``audience_model`` keyword grounding section. It reads the rebuild artifact
``models/audience_model.json`` (frozen contract: computed_at, activation_default,
base, per-family gates, source_fingerprints) plus the live
``audience_model_activation`` settings flag, and reports every factor family
with its measured held-out gate verdict. Tri-state honesty throughout: an
absent or unparsable artifact reads available false with the reason, a family
the artifact does not record reads verdict unknown, and nothing is ever
invented or asserted.

The two-model distinction is load-bearing and rides on every payload as
``basis``: expected rating (this audience model, predicting TVR for
forward-dated segments) and predicted retention (the break coefficient model)
are different models, and these gates never touch a retention coefficient.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from kairos_api.audience_api import scalar_base_summary

logger = logging.getLogger(__name__)

AUDIENCE_ARTIFACT = "models/audience_model.json"

# The frozen factor-family vocabulary of the rebuild artifact, in disclosure order.
AUDIENCE_FAMILIES = (
    "weekday_slot",
    "series",
    "calendar_school_and_chol_hamoed",
    "calendar_hanukkah",
    "calendar_religious_blackout",
    "season",
    "operator_events",
    "competitor_lineup",
)

AUDIENCE_FAMILY_LABELS_HE = {
    "weekday_slot": "יום ורצועה",
    "series": "סדרה",
    "calendar_school_and_chol_hamoed": "חול המועד וחופשות",
    "calendar_hanukkah": "חנוכה",
    "calendar_religious_blackout": "שבתות וימים טובים",
    "season": "עונה",
    "operator_events": "אירועי מפעיל",
    "competitor_lineup": "ליינאפ מתחרים",
}

AUDIENCE_BASIS = (
    "expected rating (the audience model's predicted TVR for forward-dated "
    "segments) and predicted retention (the break coefficient model) are "
    "DIFFERENT models: these gates govern expected rating only and never touch "
    "a retention coefficient; with every gate off the forward prediction "
    "equals the historical mean path (baseline_tvr)"
)

AUDIENCE_ALL_OFF_HE = "המודל טרם למד גורמי לוח מהנתונים; ההיסטוריה הנוכחית קצרה מדי, האימון הדו-שנתי יכריע"

AUDIENCE_MISSING_FAMILY_REASON = "the artifact carries no gate record for this family"

AUDIENCE_SOURCE = "audience model artifact (models/audience_model.json) plus the activation flag"


def _audience_model_path():
    from kairos_api.core import MODELS_DIR

    return MODELS_DIR / "audience_model.json"


def _audience_activation() -> dict[str, Any]:
    """The live activation flag, contract-shaped: absent reads False."""
    from kairos_api.core import _load_settings

    try:
        enabled = bool(getattr(_load_settings(), "audience_model_activation", False))
    except Exception:  # noqa: BLE001 - unreadable settings honestly read as the default
        logger.exception("audience_model_activation could not be read from settings")
        enabled = False
    return {
        "flag": "audience_model_activation",
        "enabled": enabled,
        "default_off": True,
        "note": (
            "ON replaces baseline_tvr for forward-dated segments only, with the "
            "basis recorded per segment; OFF keeps every number byte-identical to "
            "today, and historical measurement paths never use predictions"
        ),
    }


def audience_model_summary() -> dict[str, Any]:
    """The audience-model artifact summarized honestly: activation, computed_at
    and the per-family held-out gate verdicts. Shared by the get_audience_model
    read tool and the keyword grounding section; an absent or unparsable
    artifact reads available false with the reason, never an invented gate."""
    payload: dict[str, Any] = {
        "artifact": AUDIENCE_ARTIFACT,
        "activation": _audience_activation(),
        "basis": AUDIENCE_BASIS,
    }
    path = _audience_model_path()
    if not path.exists():
        payload["available"] = False
        payload["reason"] = (
            f"{AUDIENCE_ARTIFACT} does not exist; the audience model has not been "
            "rebuilt yet, so no factor family carries a measured verdict"
        )
        return payload
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload["available"] = False
        payload["reason"] = f"{AUDIENCE_ARTIFACT} exists but could not be parsed"
        return payload
    raw_gates = artifact.get("gates") if isinstance(artifact.get("gates"), dict) else {}
    gates: dict[str, Any] = {}
    for family in AUDIENCE_FAMILIES:
        record = raw_gates.get(family)
        record = record if isinstance(record, dict) else {}
        gates[family] = {
            "verdict": record.get("verdict") or "unknown",
            "reason": record.get("reason") or AUDIENCE_MISSING_FAMILY_REASON,
            "held_out_delta_pct": record.get("held_out_delta_pct"),
            "measured_at": record.get("measured_at"),
            "label_he": AUDIENCE_FAMILY_LABELS_HE[family],
        }
    families_on = sorted(name for name, gate in gates.items() if gate["verdict"] == "on")
    payload.update({
        "available": True,
        "computed_at": artifact.get("computed_at"),
        "gates": gates,
        "families_on": families_on,
        "families_on_count": len(families_on),
        "base_summary": scalar_base_summary(artifact.get("base")),
    })
    if not families_on:
        payload["all_off_headline_he"] = AUDIENCE_ALL_OFF_HE
    return payload


def _read_get_audience_model(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    return audience_model_summary()


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    """Merge the audience-model executor and its source label into the shared registry."""
    executors["get_audience_model"] = _read_get_audience_model
    sources["get_audience_model"] = AUDIENCE_SOURCE
