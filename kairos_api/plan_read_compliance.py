"""The composed compliance verdict over the committed plan.

Frozen helper of :mod:`kairos_api.plan_read`, split out under the 450-line law.
Three owners read this verdict and none of them owns it: Today prints it in the
overview payload, Rules serves it at /api/compliance, and Sources counts its
checks in the reports row.

The geometry it grades lives in :mod:`kairos_api.plan_read_guardrails` and is
reached through the module rather than through a bound name, so a caller can
still substitute the geometry and exercise the summary fallback path.

Moved verbatim from dashboard_api.py with the leading underscore dropped. The
pre-split name keeps resolving from :mod:`kairos_api.dashboard_api` and
:mod:`kairos_api.server`, against this same object.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from kairos_api import plan_read_guardrails
from kairos_api.core import KairosSettings, _summarize_schedule

logger = logging.getLogger(__name__)


def build_compliance(
    schedule: pd.DataFrame,
    settings: KairosSettings,
    operations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # The verdict is computed from the FULL committed plan's break geometry, not
    # from the break-operations display board (which is truncated to the first
    # programmes per channel for the editor and would silently grade under one
    # percent of the plan). The operations argument is kept for signature
    # compatibility but no longer feeds the verdict.
    del operations
    guardrail_items = plan_read_guardrails.plan_guardrail_items()
    break_level = plan_read_guardrails.guardrail_compliance_from_breaks(guardrail_items, settings)
    if break_level is not None:
        return {
            "profile": settings.profile_name,
            "effective_date": settings.effective_date,
            "source_url": settings.regulatory_source_url,
            "checks": break_level["checks"],
            "violations": break_level["violations"],
            "status": break_level["status"],
            "disclaimer": settings.notes,
        }

    summary = _summarize_schedule(schedule)
    hourly_seconds = plan_read_guardrails.infer_hourly_ad_seconds(schedule)
    hourly_breaks = plan_read_guardrails.infer_hourly_break_counts(schedule)
    max_hourly_minutes = round(float(hourly_seconds.max() / 60), 2) if not hourly_seconds.empty else 0.0
    max_hourly_breaks = int(hourly_breaks.max()) if not hourly_breaks.empty else 0

    protected_minutes = 0.0
    if not schedule.empty and "program_type" in schedule.columns:
        protected_types = {item.lower() for item in settings.protected_program_types}
        protected = schedule[schedule["program_type"].astype(str).str.lower().isin(protected_types)].copy()
        if not protected.empty:
            protected["ad_seconds"] = pd.to_numeric(
                protected.get("total_break_time", protected.get("break_length", 0)),
                errors="coerce",
            ).fillna(0)
            protected_minutes = round(float(protected["ad_seconds"].max() / 60), 2)

    checks = [
        {
            "id": "hourly_ad_load",
            "label_en": "Ad minutes per broadcast hour",
            "label_he": "דקות פרסום לשעת שידור",
            "status": "compliant" if max_hourly_minutes <= settings.max_ad_minutes_per_hour else "at_risk",
            "observed": max_hourly_minutes,
            "limit": settings.max_ad_minutes_per_hour,
            "unit": "minutes/hour",
        },
        {
            "id": "break_density",
            "label_en": "Breaks per hour",
            "label_he": "מספר ברייקים בשעה",
            "status": "compliant" if max_hourly_breaks <= settings.max_breaks_per_hour else "at_risk",
            "observed": max_hourly_breaks,
            "limit": settings.max_breaks_per_hour,
            "unit": "breaks/hour",
        },
        {
            "id": "retention_floor",
            "label_en": "Viewer retention floor",
            "label_he": "רף שימור צפייה",
            # average_retention is None when no schedule has been computed yet;
            # report an honest unknown rather than comparing None to the floor.
            "status": (
                "unknown"
                if summary["average_retention"] is None
                else "compliant"
                if summary["average_retention"] >= settings.min_retention_floor * 100
                else "at_risk"
            ),
            "observed": summary["average_retention"],
            "limit": round(settings.min_retention_floor * 100, 1),
            "unit": "%",
        },
        {
            "id": "protected_programs",
            "label_en": "Protected programme ad load",
            "label_he": "עומס פרסום בתוכן מוגן",
            "status": "compliant"
            if protected_minutes <= settings.protected_program_max_ad_minutes_per_hour
            else "at_risk",
            "observed": protected_minutes,
            "limit": settings.protected_program_max_ad_minutes_per_hour,
            "unit": "minutes/hour",
        },
    ]

    return {
        "profile": settings.profile_name,
        "effective_date": settings.effective_date,
        "source_url": settings.regulatory_source_url,
        "checks": checks,
        "violations": [],
        "status": "at_risk" if any(check["status"] == "at_risk" for check in checks) else "compliant",
        "disclaimer": settings.notes,
    }
