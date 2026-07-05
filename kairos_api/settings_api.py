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

from fastapi import APIRouter

from kairos_api.core import (
    MODELS_DIR,
    KairosSettings,
    _load_break_schedule,
    _load_settings,
    _model_dump,
    _save_settings,
)

router = APIRouter(tags=["settings"])


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
def update_settings(settings: KairosSettings) -> dict[str, Any]:
    return _model_dump(_save_settings(settings))


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
                "0 שומר על הצפייה בלבד (כמעט בלי הפסקות), 100 ממקסם הכנסה עד גבול הרגולציה, "
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
                "אף הפסקה לא תיבחר אם היא מורידה את הצפייה הצפויה מתחת לרצף הזה. "
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
            "label_he": "מקסימום הפסקות לשעה",
            "label_en": "Max breaks per hour",
            "help_he": "תקרת מספר ההפסקות בכל שעת שידור. כבול לרגולציה ולמדיניות המכירה.",
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
            "description_he": "מגן על הצופים: פחות הפסקות, רצפת צפייה גבוהה.",
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
