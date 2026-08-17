"""Rating-forecast routes: the number, its range, its drivers, its accuracy.

Four reads, two depths, one wall.

``GET /api/forecast/programme`` and ``.../schedule`` are DECISION surfaces: they
answer what a programme is expected to rate, with the range and the drivers, so
the planner can see the forecast where the plan is made. Any signed-in account
reads them.

``GET /api/forecast/accuracy`` and ``.../drivers`` are MEASUREMENT surfaces: the
walk-forward backtest and the model's own decomposition with its held-out gate
verdicts. That is training content, and section 4.5 of the rebuild specification
puts training content behind ``affiliation = company`` on the read as well as
the write, so both ride the model console's wall.

The same rule scrubs the decision payloads: a family's held-out percentage and
verdict prose are measurement, so a non-company account sees which families
applied and which did not, without the measurements behind the verdicts.

**The channel wall.** The forecast serves the operator's own channel only. The
audience model trains on every channel in the file, including rivals, and their
names must never reach a payload (the boundary
:func:`kairos_api.audience_api.scalar_base_summary` exists to hold). A request
naming another channel is refused without echoing the name back, and the
competitor factor ships as a scalar pressure figure, never as a rival lineup.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request

from kairos_api.affiliation_wall import is_company
from kairos_api.model_console_api import MODEL_WALL

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/forecast", tags=["forecast"])

ACCURACY_ARTIFACT = "forecast_accuracy.json"

# The command that writes the accuracy artifact, named on the honest absence so
# the surface tells the operator how to make the number exist.
ACCURACY_COMMAND = "PYTHONUTF8=1 python scripts/compute_forecast_accuracy.py"

# Measurement keys stripped from a decision payload for a non-company account.
_TRAINING_KEYS = ("held_out_delta_pct", "reason", "measured_at", "verdict")

_MAX_SCHEDULE_DAYS = 7


def _operator_channel() -> str:
    from kairos_api.core import _load_settings

    return str(getattr(_load_settings(), "operator_channel", "") or "").strip()


def _resolve_channel(requested: str) -> str:
    """The operator's own channel, or a refusal that names no other channel."""
    operator = _operator_channel()
    if not operator:
        raise HTTPException(
            status_code=422,
            detail="לא הוגדר ערוץ מפעיל בהגדרות; תחזית רייטינג נמסרת לערוץ המפעיל בלבד",
        )
    wanted = str(requested or "").strip()
    if wanted and wanted != operator:
        raise HTTPException(
            status_code=422,
            detail=(
                "משטח התחזית משרת את ערוץ המפעיל בלבד; רייטינג של ערוץ אחר אינו נמסר כאן"
            ),
        )
    return operator


def _seconds(raw: str) -> float:
    """Accept ``HH:MM``, ``HH:MM:SS`` or a plain second count."""
    text = str(raw or "").strip()
    if not text:
        return 0.0
    if ":" in text:
        parts = [p.strip() or "0" for p in text.split(":")]
        try:
            values = [float(p) for p in parts[:3]]
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"שעת התחלה לא קריאה: {raw}") from exc
        while len(values) < 3:
            values.append(0.0)
        return values[0] * 3600 + values[1] * 60 + values[2]
    try:
        return float(text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"שעת התחלה לא קריאה: {raw}") from exc


def _level(raw: float) -> float:
    value = float(raw)
    if not 0.5 <= value <= 0.99:
        raise HTTPException(
            status_code=422,
            detail="רמת הביטחון חייבת להיות בין 0.5 ל-0.99",
        )
    return value


def _scrub(payload: dict[str, Any], request: Optional[Request]) -> dict[str, Any]:
    """Strip held-out measurement from a decision payload for a non-company read."""
    if is_company(request):
        return payload
    out = dict(payload)
    scrubbed_note = (
        "מדידות מוחזקות-חוץ של המודל מוצגות במשטח המודל, הפתוח לחשבונות החברה בלבד"
    )
    if isinstance(out.get("not_applied"), list):
        out["not_applied"] = [
            {**{k: v for k, v in entry.items() if k not in _TRAINING_KEYS},
             "applied": False, "disclosure_he": scrubbed_note}
            for entry in out["not_applied"]
        ]
    if isinstance(out.get("drivers"), list):
        out["drivers"] = [
            {k: v for k, v in driver.items() if k != "held_out_delta_pct"}
            for driver in out["drivers"]
        ]
    return out


def _service():
    from kairos.model.forecast import default_service

    try:
        return default_service()
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail=(
                "אין מודל קהל מאומן בדיסק (models/audience_model.json); "
                "יש להריץ את האימון לפני שתחזית רייטינג תהיה זמינה"
            ),
        ) from None


@router.get("/programme")
def programme(
    request: Request,
    title: str,
    date: str,
    start: str = "",
    duration_seconds: float = 0.0,
    channel: str = "",
    audience: str = "",
    level: float = 0.80,
) -> dict[str, Any]:
    """One programme's expected rating, with its range, drivers and provenance."""
    resolved = _resolve_channel(channel)
    payload = _service().forecast_programme(
        channel=resolved, program_title=title, day=date,
        start_seconds=_seconds(start), duration_seconds=float(duration_seconds or 0.0),
        level=_level(level), audience=audience,
    )
    return {"channel": resolved, **_scrub(payload, request)}


@router.get("/schedule")
def schedule(
    request: Request,
    date: str,
    days: int = 1,
    channel: str = "",
    audience: str = "",
    level: float = 0.80,
) -> dict[str, Any]:
    """Every programme of a channel-day (or up to a week), forecast for planning.

    Reads the programme schedule the plan itself is built from, so the forecast
    a planner sees on this surface is a forecast of the same programmes the
    optimizer prices, not a parallel list.
    """
    import pandas as pd

    from kairos.model.audience_frame import PREDICTION_COLUMNS
    from kairos_api.core import _load_programmes

    resolved = _resolve_channel(channel)
    span = max(1, min(int(days or 1), _MAX_SCHEDULE_DAYS))
    try:
        first = pd.Timestamp(date).normalize()
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=f"תאריך לא קריא: {date}") from exc
    wanted = {(first + pd.Timedelta(days=offset)).date() for offset in range(span)}

    frame = _load_programmes()
    if frame is None or frame.empty or "start_dt" not in frame.columns:
        return {"available": False, "channel": resolved, "days": [],
                "reason_he": "אין לוח תוכניות טעון; אין מה לחזות"}
    sliced = frame[(frame["Channel"].astype(str) == resolved) & frame["start_dt"].notna()]
    sliced = sliced[sliced["start_dt"].dt.date.isin(wanted)].sort_values("start_dt")
    if sliced.empty:
        return {
            "available": False, "channel": resolved, "days": [],
            "requested": {"from": first.date().isoformat(), "days": span},
            "reason_he": "בלוח התוכניות אין תוכניות לערוץ ולתאריכים המבוקשים",
        }

    rows = pd.DataFrame({
        "date": sliced["start_dt"].dt.strftime("%Y-%m-%d"),
        "channel": resolved,
        "program_title": sliced["Title"].astype(str),
        "start_seconds": (
            sliced["start_dt"].dt.hour * 3600 + sliced["start_dt"].dt.minute * 60
            + sliced["start_dt"].dt.second
        ).astype(float),
        "duration_seconds": pd.to_numeric(sliced["Duration"], errors="coerce").fillna(0.0),
    }, columns=list(PREDICTION_COLUMNS)).reset_index(drop=True)

    payloads = [_scrub(p, request) for p in _service().forecast_rows(
        rows, level=_level(level), audience=audience)]
    by_day: dict[str, list[dict[str, Any]]] = {}
    for (_, row), payload in zip(rows.iterrows(), payloads):
        clock = int(row["start_seconds"])
        entry = {
            "title": row["program_title"],
            "start_clock": f"{clock // 3600:02d}:{clock % 3600 // 60:02d}",
            "start_seconds": clock,
            **payload,
        }
        by_day.setdefault(str(row["date"]), []).append(entry)

    days_out = []
    for day, entries in sorted(by_day.items()):
        served = [e for e in entries if e.get("available")]
        banded = [e for e in served if (e.get("interval") or {}).get("available")]
        days_out.append({
            "date": day,
            "programmes": entries,
            "summary": {
                "n": len(entries), "n_forecast": len(served), "n_with_band": len(banded),
                "mean_expected_tvr": (
                    round(sum(e["expected_tvr"] for e in served) / len(served), 4)
                    if served else None
                ),
                "mean_historical_tvr": (
                    round(sum(e["history"]["historical_tvr"] for e in served) / len(served), 4)
                    if served else None
                ),
            },
        })
    return {
        "available": any(day["summary"]["n_forecast"] for day in days_out),
        "channel": resolved,
        "requested": {"from": first.date().isoformat(), "days": span},
        "days": days_out,
        "audience_basis": next(
            (p["audience_basis"] for p in payloads if p.get("audience_basis")), None),
    }


@router.get("/accuracy")
@MODEL_WALL.guard()
def accuracy() -> dict[str, Any]:
    """The walk-forward record as the measurement wrote it, or an honest absence."""
    from kairos_api.core import MODELS_DIR

    path = MODELS_DIR / ACCURACY_ARTIFACT
    if not path.exists():
        return {
            "available": False, "computed_at": None,
            "reason_he": "טרם נמדד דיוק התחזית; אין קובץ מדידה בדיסק",
            "reason_en": (
                "no forecast-accuracy measurement on disk "
                f"(models/{ACCURACY_ARTIFACT}); run the measurement to create it"
            ),
            "command": ACCURACY_COMMAND,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        logger.warning("forecast accuracy artifact at %s is unreadable", path)
        return {"available": False, "computed_at": None,
                "reason_en": f"the accuracy artifact at models/{ACCURACY_ARTIFACT} is unreadable",
                "command": ACCURACY_COMMAND}
    if not isinstance(payload, dict):
        return {"available": False, "computed_at": None,
                "reason_en": "the accuracy artifact is not an object", "command": ACCURACY_COMMAND}
    return {"command": ACCURACY_COMMAND, **payload}


@router.get("/drivers")
@MODEL_WALL.guard()
def drivers() -> dict[str, Any]:
    """How the forecast is built, what it is measured on, and what it refuses.

    The disclosure surface for the forecast stage: the multiplication in order,
    every family with the verdict its own held-out measurement returned, the
    scatter behind the published ranges, and the two windows outside which the
    surface refuses to answer. Scalars only from the base block, so no channel
    but the operator's own can appear.
    """
    from kairos.model.forecast_basis import (
        DEFAULT_LEVEL,
        FAMILY_LABELS_HE,
        MAX_HORIZON_DAYS,
        audience_basis_block,
        calendar_span,
    )
    from kairos_api.audience_api import _activation, _read_artifact, scalar_base_summary

    artifact = _read_artifact()
    if artifact is None:
        return {"available": False, "activation": _activation(),
                "reason_he": "אין מודל קהל מאומן בדיסק",
                "reason_en": "no trained audience model on disk"}
    service = _service()
    gates = artifact.get("gates") if isinstance(artifact.get("gates"), dict) else {}
    first, last = calendar_span()
    return {
        "available": True,
        "computed_at": artifact.get("computed_at"),
        "activation": _activation(),
        "activation_note_he": (
            "הדגל קובע אם התחזית מחליפה את הבסיס ההיסטורי בתמחור; המשטח הזה קורא בלבד"
        ),
        "audience_basis": audience_basis_block(),
        "base": scalar_base_summary(artifact.get("base")),
        "decomposition": [
            {"step": 1, "key": "global", "label_he": "רמת הבסיס של כל המדידה",
             "explain_he": "ממוצע הלוג של כל התצפיות; נקודת ההתחלה"},
            {"step": 2, "key": "channel", "label_he": "רמת הערוץ",
             "explain_he": "הפרש רמת הערוץ מהבסיס, מכווץ לפי מספר התצפיות"},
            {"step": 3, "key": "genre_or_slot", "label_he": "ז'אנר, ובהיעדרו רצועה",
             "explain_he": "הרמה שבתוך הערוץ; בהיעדר ז'אנר נצפה נופלים לרצועה ואז לערוץ"},
            {"step": 4, "key": "families", "label_he": "משפחות מופעלות",
             "explain_he": "כל משפחה שעברה את השער מוסיפה מקדם כפלי אחד"},
        ],
        "families": [
            {"family": family, "label_he": FAMILY_LABELS_HE.get(family, family),
             **{key: gate.get(key) for key in
                ("verdict", "reason", "held_out_delta_pct", "measured_at")}}
            for family, gate in sorted(gates.items())
        ],
        "interval": {
            "default_level": DEFAULT_LEVEL,
            "method_en": (
                "empirical-Bayes predictive band in log space from the scatter the "
                "model was fitted from: within-cell spread, the between-cell spread "
                "the shrinkage trades against, and the sampling error of the cell mean"
            ),
            "levels": service.dispersion.summary() if service.dispersion else None,
            "unavailable_reason": service.dispersion_reason or None,
        },
        "windows": {
            "measured": service.measured_window(),
            "max_horizon_days": MAX_HORIZON_DAYS,
            "calendar_from": first.isoformat() if first else None,
            "calendar_to": last.isoformat() if last else None,
            "refusal_he": (
                "תאריך מחוץ ללוח השנה המובנה, או רחוק מסוף חלון המדידה מעל התקרה, "
                "מקבל תשובת אין-זמין עם הסיבה"
            ),
        },
        "accuracy_command": ACCURACY_COMMAND,
    }
