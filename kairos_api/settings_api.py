"""Settings and health endpoints for the Kairos dashboard.

Thin domain router over the shared kernel (:mod:`kairos_api.core`): the health
probe, the operator settings read/write, and the settings-controls schema that
drives the dashboard's controls panel (lever labels, help text, bounds, and the
named templates in one authoritative place). Extracted from server.py as part
of the modular-monolith carve-up; behavior is unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from kairos_api.core import (
    MODELS_DIR,
    KairosSettings,
    _load_break_schedule,
    _load_settings,
    _model_dump,
    _save_settings,
)

router = APIRouter(tags=["settings"])


def _require_canonical_protected_writes(
    incoming: KairosSettings, current: KairosSettings | None = None,
) -> None:
    """Keep protected settings behind the routes that own their consequences.

    The generic settings client sends the whole model, so unchanged protected
    values must pass. A moved value cannot: those routes add permission,
    validation and (for the licence) the effective-date record that this body
    cannot supply.
    """
    from kairos_api import guardrail_store, model_activation

    current = current or _load_settings()
    moved_limits = [
        key for key in guardrail_store.GUARDRAIL_KEYS
        if getattr(incoming, key, None) != getattr(current, key, None)
    ]
    if moved_limits:
        raise HTTPException(
            status_code=409,
            detail=(
                "Regulatory limits must be changed through /api/rules/guardrails "
                "with an effective date, reason and append-only change record."
            ),
        )
    if getattr(incoming, model_activation.SETTINGS_FIELD, False) != getattr(
        current, model_activation.SETTINGS_FIELD, False
    ):
        raise HTTPException(
            status_code=409,
            detail="Audience model activation must be changed through /api/rules/model-activation.",
        )
    if incoming.pricing_overrides != current.pricing_overrides:
        raise HTTPException(
            status_code=409,
            detail="Pricing overrides must be changed through /api/pricing so the edit is validated.",
        )


@router.get("/api/health")
def health() -> dict[str, Any]:
    schedule = _load_break_schedule()
    return {
        "status": "ok",
        "project": "Kairos",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "has_schedule": not schedule.empty,
        "has_model": (MODELS_DIR / "tv_break_posterior.pkl").exists(),
    }


@router.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return _model_dump(_load_settings())


@router.put("/api/settings")
def update_settings(settings: KairosSettings, request: Request = None) -> dict[str, Any]:
    # The operator channel is the scoping declaration the whole product hangs
    # on, and this route takes the settings model whole, so before P5 measured
    # it an operator of either affiliation could move it here after being
    # refused at the declaration route. That does not leak a rival's data, it
    # inverts the boundary, which is worse. The rule lives with the wall it
    # uses and is called in one line, so the two enforcement points cannot
    # drift apart. A write that does not move the channel, which is what both
    # shipped clients send, returns immediately.
    from kairos_api.compliance_api_licence import guard_channel_move
    from kairos_api import version_store

    # The protected-field comparison and the replacement are one transaction.
    # Without the outer lock, a canonical licence/model/pricing write could land
    # between the comparison and this whole-document save, letting a stale
    # generic body put the protected subtree back without its required record.
    from kairos_api import core

    with core._SETTINGS_LOCK:
        current = _load_settings()
        declared_fields = getattr(settings, "model_fields_set", None)
        if declared_fields is None:  # Pydantic v1 compatibility
            declared_fields = getattr(settings, "__fields_set__", None)
        declared = set(declared_fields or ())
        # FastAPI/Pydantic materializes defaults for omitted fields. Treat this
        # whole-document endpoint conservatively as a merge of fields the
        # caller actually sent, so a partial client cannot reset unrelated
        # settings simply by not knowing they exist.
        merged = _model_dump(current)
        incoming = _model_dump(settings)
        for field in declared:
            if field in incoming:
                merged[field] = incoming[field]
        candidate = KairosSettings(**merged)
        guard_channel_move(candidate, request)
        _require_canonical_protected_writes(candidate, current)
        if _model_dump(candidate) == _model_dump(current):
            return _model_dump(current)
        version_store.snapshot_manual_edit(request, "settings")
        return _model_dump(_save_settings(candidate))


@router.get("/api/settings/controls")
def settings_controls() -> dict[str, Any]:
    """Describe the operator-tunable optimizer levers and the named templates.

    The dashboard renders its controls panel from this schema so every knob has a
    clear label (Hebrew and English), help text, a default, and bounds, and so the
    presets ("setups"/templates) stay in one authoritative place instead of being
    hardcoded in the frontend. The ``current`` block is the operator's saved value,
    so the panel opens on the real state, not a guess.
    """
    saved = _load_settings()
    levers = [
        {
            "key": "revenue_weight",
            "label_he": "איזון הכנסה מול צפייה",
            "label_en": "Revenue vs retention balance",
            "help_he": (
                "הלֶבֶר המרכזי: כמה לרדוף אחרי הכנסת פרסום מול שמירה על הצופים. "
                "0 שומר על הצפייה בלבד (כמעט בלי ברייקים), 100 ממקסם הכנסה עד גבול הרגולציה, "
                "60 הוא איזון נוטה-להכנסה. הערך הזה מניע את הלוח השבועי, גבול היעילות והתחזיות."
            ),
            "help_en": (
                "The central lever: how hard to chase ad revenue versus protecting viewers. "
                "0 protects retention only (almost no breaks), 100 maximizes revenue up to the "
                "regulatory guardrails, 60 is a revenue-leaning balance. This drives the weekly "
                "schedule, the efficiency frontier, and the forecasts."
            ),
            "default": 60,
            "min": 0,
            "max": 100,
            "step": 5,
            "unit": "%",
        },
        {
            "key": "risk_lambda",
            "label_he": "זהירות מול אי-ודאות",
            "label_en": "Uncertainty caution",
            "help_he": (
                "קובע איזו עלות צפייה נכנסת למספרים המדווחים: 0 מדווח לפי האומדן הנקודתי, "
                "1 מתמחר את עלות הצפייה הסבירה הגרועה ביותר בטווח המדידה. זהו כלי דיווח שקוף, "
                "ובנתונים הנוכחיים הוא אינו משנה את התוכנית שנבחרת."
            ),
            "help_en": (
                "Sets which retention cost the reported numbers carry: 0 reports at the point "
                "estimate, 1 prices the worst plausible cost in the measured interval. A "
                "reporting lever: on current data it does not change which plan is chosen."
            ),
            "default": 0.0,
            "min": 0.0,
            "max": 1.0,
            "step": 0.1,
            "unit": "",
        },
        {
            "key": "min_retention_floor",
            "label_he": "רצפת צפייה מינימלית",
            "label_en": "Minimum retention floor",
            "help_he": (
                "אף ברייק לא ייבחר אם הוא מוריד את הצפייה הצפויה מתחת לרצפה הזו. "
                "מגן על הנכס לטווח ארוך גם כשהלֶבֶר נוטה להכנסה."
            ),
            "help_en": (
                "No break is chosen if it pushes predicted retention below this floor. Protects the "
                "long-term asset even when the balance lever leans toward revenue."
            ),
            "default": 0.72,
            "min": 0.0,
            "max": 1.0,
            "step": 0.01,
            "unit": "",
        },
        {
            "key": "max_breaks_per_hour",
            "label_he": "מקסימום ברייקים לשעה",
            "label_en": "Max breaks per hour",
            "help_he": "תקרת מספר הברייקים בכל שעת שידור. כבול לרגולציה ולמדיניות המכירה.",
            "help_en": "Ceiling on breaks in any broadcast hour. Bound by regulation and sales policy.",
            "default": 4,
            "min": 1,
            "max": 20,
            "step": 1,
            "unit": "",
        },
    ]
    templates = [
        {
            "key": "balanced",
            "label_he": "מאוזן",
            "label_en": "Balanced",
            "description_he": "ברירת המחדל: נוטה-להכנסה אך שומר על הצופים.",
            "description_en": "The default: revenue-leaning but protective of viewers.",
            "values": {"revenue_weight": 60, "risk_lambda": 0.0, "min_retention_floor": 0.72},
        },
        {
            "key": "revenue_priority",
            "label_he": "עדיפות להכנסה",
            "label_en": "Revenue priority",
            "description_he": "ממקסם הכנסת פרסום עד גבול הרגולציה. לשבועות מכירה חזקים.",
            "description_en": "Maximizes ad revenue up to the guardrails. For strong sales weeks.",
            "values": {"revenue_weight": 85, "risk_lambda": 0.0, "min_retention_floor": 0.70},
        },
        {
            "key": "retention_guardrail",
            "label_he": "שמירה על צפייה",
            "label_en": "Retention guardrail",
            "description_he": "מגן על הצופים: פחות ברייקים, רצפת צפייה גבוהה.",
            "description_en": "Protects viewers: fewer breaks, a higher retention floor.",
            "values": {"revenue_weight": 35, "risk_lambda": 0.0, "min_retention_floor": 0.78},
        },
        {
            "key": "conservative_uncertainty",
            "label_he": "זהיר באי-ודאות",
            "label_en": "Conservative under uncertainty",
            "description_he": "מדווח את המספרים לפי עלות הצפייה הסבירה הגרועה ביותר. בנתונים הנוכחיים התוכנית עצמה אינה משתנה.",
            "description_en": "Reports the numbers at the worst plausible retention cost. On current data the plan itself is unchanged.",
            "values": {"revenue_weight": 60, "risk_lambda": 1.0, "min_retention_floor": 0.74},
        },
    ]
    current = {
        "revenue_weight": saved.revenue_weight,
        "risk_lambda": saved.risk_lambda,
        "min_retention_floor": saved.min_retention_floor,
        "max_breaks_per_hour": saved.max_breaks_per_hour,
    }
    return {"levers": levers, "templates": templates, "current": current}
