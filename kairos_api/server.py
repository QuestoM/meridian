"""FastAPI server for the Kairos revenue optimization dashboard."""

from __future__ import annotations

import math
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from kairos.optimize.guardrails import Break as GuardrailBreak
from kairos.optimize.guardrails import Guardrails, evaluate as evaluate_guardrails
from kairos.optimize.objective import break_revenue as cpp_break_revenue
from kairos.optimize.objective import retention_adjusted_revenue

# The real optimization engine is imported defensively: if its dependencies are
# absent the rest of the API still boots, and the engine-backed endpoints report
# that honestly instead of crashing.
try:
    from dataclasses import asdict as _asdict

    from kairos.data.loaders import CHANNELS as KAIROS_CHANNELS
    from kairos.optimize.pricing import OptimizerAssumptions, PricingModel
    from kairos.service import guardrails_from_settings, optimize_day_plan, run_scenario

    _ENGINE_AVAILABLE = True
except Exception:  # pragma: no cover - engine optional at import time
    _ENGINE_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"
MODELS_DIR = ROOT / "models"
SETTINGS_PATH = DATA_DIR / "kairos_settings.json"


class OptimizeRequest(BaseModel):
    """Parameters for a traffic-ready optimization run."""

    model_path: str = Field(default="models/tv_break_posterior.pkl")
    programs_path: str = Field(default="data/Programmes.csv")
    spots_inventory: str | None = Field(default=None)
    output_path: str = Field(default="optimization_results.csv")
    min_retention: float = Field(default=0.75, ge=0.0, le=1.0)
    max_breaks_per_hour: int = Field(default=3, ge=1, le=12)
    budget: float = Field(default=100_000.0, gt=0)


class ScenarioRequest(BaseModel):
    """Lightweight scenario controls used by the dashboard simulation."""

    revenue_weight: int = Field(default=60, ge=0, le=100)
    retention_floor: float = Field(default=0.72, ge=0.0, le=1.0)
    max_breaks_per_hour: int = Field(default=3, ge=1, le=12)


class BreakDecisionRequest(BaseModel):
    """Operator decision captured from the dashboard command surface."""

    action: Literal["approve", "reject", "apply_similar"]
    recommendation_id: str | None = Field(default=None)
    break_id: str | None = Field(default=None)
    program_type: str | None = Field(default=None)
    scenario: str | None = Field(default=None)
    note: str | None = Field(default=None, max_length=500)


class KairosSettings(BaseModel):
    """Operational controls for market, regulatory, and UX behavior.

    These values are deliberately configurable because Israeli TV rules,
    internal sales policy, and customer contracts can change.
    """

    profile_name: str = "Israel commercial TV"
    locale: Literal["he", "en"] = "he"
    direction: Literal["rtl", "ltr"] = "rtl"
    chart_direction: Literal["ltr"] = "ltr"
    timezone: str = "Asia/Jerusalem"
    currency: str = "ILS"
    effective_date: str = "2026-06-14"
    regulatory_source_url: str = "https://www.rashut2.org.il/"
    max_ad_minutes_per_hour: float = Field(default=12.0, ge=0, le=60)
    max_breaks_per_hour: int = Field(default=4, ge=1, le=20)
    min_break_spacing_minutes: int = Field(default=7, ge=0, le=120)
    min_retention_floor: float = Field(default=0.72, ge=0, le=1)
    max_daily_ad_minutes: int = Field(default=160, ge=0, le=1440)
    protected_program_types: list[str] = Field(default_factory=lambda: ["News", "Kids", "Children"])
    protected_program_max_ad_minutes_per_hour: float = Field(default=8.0, ge=0, le=60)
    sponsorships_enabled: bool = True
    gold_breaks_enabled: bool = True
    gold_breaks_max_per_day: int = Field(default=3, ge=0, le=50)
    require_manual_approval: bool = True
    notes: str = "Configurable baseline. Validate with current counsel and broadcaster policy before production use."


def _safe_path(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    if ROOT not in path.parents and path != ROOT:
        raise HTTPException(status_code=400, detail="Path is outside project root")
    return path


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if not kwargs:
        stat = path.stat()
        return _read_csv_cached(str(path), stat.st_mtime_ns, stat.st_size).copy()
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


@lru_cache(maxsize=64)
def _read_csv_cached(path: str, mtime_ns: int, size: int) -> pd.DataFrame:
    del mtime_ns, size
    return pd.read_csv(Path(path), encoding="utf-8-sig")


def _safe_number(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(numeric) or math.isinf(numeric):
        return default
    return numeric


def _model_dump(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _load_settings() -> KairosSettings:
    if not SETTINGS_PATH.exists():
        return KairosSettings()
    try:
        with SETTINGS_PATH.open("r", encoding="utf-8") as handle:
            return KairosSettings(**json.load(handle))
    except (OSError, ValueError, TypeError):
        return KairosSettings()


def _save_settings(settings: KairosSettings) -> KairosSettings:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(_model_dump(settings), handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return settings


def _settings_to_guardrails(settings: KairosSettings) -> Guardrails:
    return Guardrails(
        max_ad_seconds_per_hour=settings.max_ad_minutes_per_hour * 60,
        max_breaks_per_hour=settings.max_breaks_per_hour,
        min_break_spacing_seconds=settings.min_break_spacing_minutes * 60,
        min_retention_floor=settings.min_retention_floor,
        max_daily_ad_seconds=settings.max_daily_ad_minutes * 60,
        protected_program_types=tuple(settings.protected_program_types),
        protected_max_ad_seconds_per_hour=settings.protected_program_max_ad_minutes_per_hour * 60,
        gold_breaks_max_per_day=settings.gold_breaks_max_per_day,
    )


def _percent(value: Any) -> float:
    numeric = _safe_number(value, 0.0)
    if numeric <= 1.5:
        return numeric * 100
    return numeric


def _ratio(value: Any) -> float:
    numeric = _safe_number(value, 0.0)
    if numeric > 1.5:
        return numeric / 100
    return numeric


def _money(value: Any) -> float:
    return round(_safe_number(value, 0.0), 2)


def _time_to_seconds(value: Any) -> float:
    text = str(value or "").strip()
    if not text:
        return 0.0
    try:
        parts = [int(float(part)) for part in text.split(":")[:3]]
    except ValueError:
        parsed = pd.to_datetime(text, errors="coerce")
        if pd.isna(parsed):
            return 0.0
        return float(parsed.hour * 3600 + parsed.minute * 60 + parsed.second)
    if len(parts) == 1:
        return float(parts[0] * 3600)
    if len(parts) == 2:
        hour, minute = parts
        second = 0
    else:
        hour, minute, second = parts
    return float(hour * 3600 + minute * 60 + second)


def _day_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    aliases = {
        "monday": "Mon",
        "mon": "Mon",
        "tuesday": "Tue",
        "tue": "Tue",
        "wednesday": "Wed",
        "wed": "Wed",
        "thursday": "Thu",
        "thu": "Thu",
        "friday": "Fri",
        "fri": "Fri",
        "saturday": "Sat",
        "sat": "Sat",
        "sunday": "Sun",
        "sun": "Sun",
    }
    return aliases.get(text.lower(), text[:3].title())


def _program_datetime_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize programme CSV date/time variants into start_dt and end_dt."""

    result = frame.copy()
    if "Start_datetime" in result.columns:
        starts = pd.to_datetime(result["Start_datetime"], errors="coerce")
    elif {"Date", "Start time"}.issubset(result.columns):
        starts = pd.to_datetime(
            result["Date"].astype(str) + " " + result["Start time"].astype(str),
            errors="coerce",
            dayfirst=True,
        )
    else:
        starts = pd.to_datetime(result.get("Start time"), errors="coerce")

    if "End_datetime" in result.columns:
        ends = pd.to_datetime(result["End_datetime"], errors="coerce")
    elif {"Date", "End time"}.issubset(result.columns):
        ends = pd.to_datetime(
            result["Date"].astype(str) + " " + result["End time"].astype(str),
            errors="coerce",
            dayfirst=True,
        )
    else:
        ends = pd.to_datetime(result.get("End time"), errors="coerce")

    result["start_dt"] = starts
    duration = pd.to_numeric(result.get("Duration", 0), errors="coerce").fillna(0)
    result["end_dt"] = ends.where(ends.notna(), result["start_dt"] + pd.to_timedelta(duration, unit="s"))
    result.loc[result["end_dt"] <= result["start_dt"], "end_dt"] = (
        result["start_dt"] + pd.to_timedelta(duration.clip(lower=1800), unit="s")
    )
    return result


def _load_break_schedule() -> pd.DataFrame:
    candidates = [
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
    ]
    for path in candidates:
        frame = _read_csv(path)
        if not frame.empty:
            return frame
    return pd.DataFrame()


def _load_programmes() -> pd.DataFrame:
    return _read_csv(DATA_DIR / "Programmes.csv")


def _load_spots() -> pd.DataFrame:
    return _read_csv(DATA_DIR / "Spots.csv")


def _load_impact(path: Path) -> list[dict[str, Any]]:
    frame = _read_csv(path)
    records: list[dict[str, Any]] = []
    for raw in frame.replace({pd.NA: None}).where(pd.notna(frame), None).to_dict("records"):
        record: dict[str, Any] = {}
        for key, value in raw.items():
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = None
            else:
                record[key] = value
        records.append(record)
    return records


def _segment_key(channel_name: str) -> tuple[str, str, str] | None:
    parts = str(channel_name or "").split("_")
    if len(parts) < 3:
        return None
    return "_".join(parts[:-2]), parts[-2], parts[-1]


def _weighted_impact_rows(items: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        segment = str(item.get(key) or "")
        coefficient = _safe_number(item.get("coefficient"), math.nan)
        if not segment or not math.isfinite(coefficient):
            continue
        grouped.setdefault(segment, []).append(item)

    rows: list[dict[str, Any]] = []
    for segment, values in grouped.items():
        total_weight = 0
        weighted_coefficient = 0.0
        weighted_raw = 0.0
        ci_low: list[float] = []
        ci_high: list[float] = []
        for item in values:
            sample_count = max(1, int(_safe_number(item.get("n"), 1)))
            coefficient = _safe_number(item.get("coefficient"), 0.0)
            raw_delta = _safe_number(item.get("raw_delta"), coefficient)
            weighted_coefficient += coefficient * sample_count
            weighted_raw += raw_delta * sample_count
            total_weight += sample_count
            low = _safe_number(item.get("ci_low"), math.nan)
            high = _safe_number(item.get("ci_high"), math.nan)
            if math.isfinite(low):
                ci_low.append(low)
            if math.isfinite(high):
                ci_high.append(high)
        if total_weight <= 0:
            continue
        rows.append(
            {
                "segment": segment,
                "average_coefficient": round(weighted_coefficient / total_weight, 6),
                "average_raw_delta": round(weighted_raw / total_weight, 6),
                "sample_count": total_weight,
                "channel_count": len(values),
                "ci_low": round(min(ci_low), 6) if ci_low else None,
                "ci_high": round(max(ci_high), 6) if ci_high else None,
            }
        )
    return sorted(rows, key=lambda row: abs(float(row["average_coefficient"])), reverse=True)


def _load_measured_impact_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "source": "legacy_csv",
            "program_type": [],
            "position": [],
            "length": [],
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {
            "source": "legacy_csv",
            "program_type": [],
            "position": [],
            "length": [],
        }

    details = payload.get("detail", {})
    items: list[dict[str, Any]] = []
    for name, raw in details.items():
        if not isinstance(raw, dict):
            continue
        segment = _segment_key(str(raw.get("channel_name") or name))
        if not segment:
            continue
        program_type, position, length = segment
        items.append(
            {
                **raw,
                "program_type": program_type,
                "position": position,
                "length": length,
            }
        )

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    return {
        "source": payload.get("method") or "measured_coefficients",
        "metadata": metadata,
        "program_type": _weighted_impact_rows(items, "program_type"),
        "position": _weighted_impact_rows(items, "position"),
        "length": _weighted_impact_rows(items, "length"),
    }


def _summarize_schedule(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {
            "total_breaks": 0,
            "total_ad_seconds": 0,
            "projected_revenue": 0,
            "average_retention": 0,
            "risk_score": 0,
        }

    num_breaks = pd.to_numeric(schedule.get("num_breaks", 1), errors="coerce").fillna(1)
    break_time = pd.to_numeric(
        schedule.get("total_break_time", schedule.get("break_length", 0)),
        errors="coerce",
    ).fillna(0)
    revenue = pd.to_numeric(
        schedule.get("predicted_revenue", schedule.get("revenue_ils", 0)),
        errors="coerce",
    ).fillna(0)
    retention = pd.to_numeric(schedule.get("predicted_retention", 0), errors="coerce")
    retention = retention[retention > 0]
    avg_retention = retention.mean() if not retention.empty else 0.0
    risk_score = max(0.0, min(100.0, (0.78 - avg_retention) * 220 + len(schedule) * 0.8))

    return {
        "total_breaks": int(num_breaks.sum()),
        "total_ad_seconds": int(break_time.sum()),
        "projected_revenue": _money(revenue.sum()),
        "average_retention": round(_percent(avg_retention), 1),
        "risk_score": round(risk_score, 1),
    }


def _build_schedule_canvas(programmes: pd.DataFrame, schedule: pd.DataFrame) -> list[dict[str, Any]]:
    if programmes.empty:
        return []

    frame = _program_datetime_columns(programmes)
    frame = frame.dropna(subset=["start_dt"])
    frame["day"] = frame["start_dt"].dt.strftime("%a")
    frame["hour"] = frame["start_dt"].dt.hour
    frame["program_type"] = frame.get("program_type", frame.get("programme_type", "Other")).fillna("Other")
    frame["viewing_points"] = pd.to_numeric(frame.get("TVR", 1.0), errors="coerce").fillna(1.0)

    schedule_by_type: dict[str, dict[str, float]] = {}
    if not schedule.empty and "program_type" in schedule.columns:
        for program_type, group in schedule.groupby("program_type"):
            schedule_by_type[str(program_type)] = _summarize_schedule(group)

    rows: list[dict[str, Any]] = []
    for channel, channel_df in frame.sort_values("start_dt").groupby("Channel"):
        programs = []
        for _, row in channel_df.head(18).iterrows():
            type_summary = schedule_by_type.get(str(row["program_type"]), {})
            retention = type_summary.get("average_retention", 0.0)
            revenue = type_summary.get("projected_revenue", 0.0)
            break_count = max(1, min(8, int(type_summary.get("total_breaks", 2) / 4) or 2))
            programs.append(
                {
                    "title": row.get("Title", "Untitled"),
                    "program_type": row["program_type"],
                    "day": row["day"],
                    "time": row["start_dt"].strftime("%H:%M"),
                    "duration_minutes": round(_safe_number(row.get("Duration"), 3600) / 60),
                    "revenue": _money(revenue),
                    "retention": round(_safe_number(retention, 74.0), 1),
                    "break_markers": break_count,
                    "selected": len(programs) == 1 and len(rows) == 0,
                }
            )
        rows.append({"channel": channel, "programs": programs})

    return rows[:6]


def _schedule_lookup(schedule: pd.DataFrame) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, dict[str, Any]]]:
    by_type_day: dict[tuple[str, str], dict[str, Any]] = {}
    by_type: dict[str, dict[str, Any]] = {}
    if schedule.empty:
        return by_type_day, by_type

    frame = schedule.copy()
    frame["program_type"] = frame.get("program_type", "Other").fillna("Other").astype(str)
    frame["day_key"] = frame.get("day", "").map(_day_key) if "day" in frame.columns else ""
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0.0), errors="coerce").fillna(0.0)
    frame["num_breaks"] = pd.to_numeric(frame.get("num_breaks", 1), errors="coerce").fillna(1)
    frame["break_length"] = pd.to_numeric(frame.get("break_length", frame.get("total_break_time", 120)), errors="coerce").fillna(120)

    for _, row in frame.sort_values("predicted_revenue", ascending=False).iterrows():
        key = str(row["program_type"]).lower()
        record = row.to_dict()
        by_type.setdefault(key, record)
        if row.get("day_key"):
            by_type_day.setdefault((key, str(row["day_key"])), record)

    return by_type_day, by_type


def _build_break_operations(programmes: pd.DataFrame, schedule: pd.DataFrame) -> dict[str, Any]:
    if programmes.empty:
        return {"programs": [], "breaks": [], "summary": {"programs": 0, "breaks": 0, "ad_seconds": 0, "revenue": 0}}

    frame = _program_datetime_columns(programmes)
    frame = frame.dropna(subset=["start_dt", "end_dt"]).copy()
    if frame.empty:
        return {"programs": [], "breaks": [], "summary": {"programs": 0, "breaks": 0, "ad_seconds": 0, "revenue": 0}}

    frame["program_type"] = frame.get("program_type", frame.get("programme_type", "Other")).fillna("Other").astype(str)
    frame["viewing_points"] = pd.to_numeric(frame.get("TVR", 1.0), errors="coerce").fillna(1.0)
    frame["day_key"] = frame["start_dt"].dt.strftime("%a")
    frame["duration_seconds"] = (frame["end_dt"] - frame["start_dt"]).dt.total_seconds().clip(lower=0)
    frame = frame.sort_values("start_dt").groupby("Channel", dropna=False).head(12).reset_index(drop=True)

    by_type_day, by_type = _schedule_lookup(schedule)
    programs: list[dict[str, Any]] = []
    breaks: list[dict[str, Any]] = []

    for row_index, row in frame.iterrows():
        channel = str(row.get("Channel") or row.get("channel") or "Channel")
        program_type = str(row.get("program_type") or "Other")
        day = str(row.get("day_key") or "")
        program_id = str(row.get("programme_id") or row.get("id") or f"program-{row_index}")
        program_key = f"{channel}-{program_id}-{row['start_dt'].strftime('%H%M')}"
        duration_seconds = int(_safe_number(row.get("duration_seconds"), 0))
        duration_minutes = round(duration_seconds / 60, 1)
        schedule_row = by_type_day.get((program_type.lower(), day)) or by_type.get(program_type.lower()) or {}
        planned_breaks = int(max(0, _safe_number(schedule_row.get("num_breaks"), 1 if duration_minutes >= 30 else 0)))
        capacity_breaks = int(max(0, duration_minutes // 18))
        break_count = max(0, min(5, planned_breaks, capacity_breaks if duration_minutes >= 18 else 0))
        revenue_total = _money(schedule_row.get("predicted_revenue", 0.0))
        retention = round(_percent(schedule_row.get("predicted_retention", 0.0)), 1)
        break_seconds = int(max(30, min(360, _safe_number(schedule_row.get("break_length"), 120))))
        lane = f"{channel} / {day}"

        programs.append(
            {
                "id": program_id,
                "key": program_key,
                "lane": lane,
                "channel": channel,
                "title": row.get("Title", "Untitled"),
                "program_type": program_type,
                "day": day,
                "date": row["start_dt"].date().isoformat(),
                "start_time": row["start_dt"].strftime("%H:%M"),
                "end_time": row["end_dt"].strftime("%H:%M"),
                "duration_minutes": duration_minutes,
                "revenue": revenue_total,
                "retention": retention,
                "break_markers": break_count,
            }
        )

        if break_count == 0:
            continue

        for break_index in range(1, break_count + 1):
            candidate = row["start_dt"] + pd.Timedelta(seconds=int((duration_seconds / (break_count + 1)) * break_index))
            min_start = row["start_dt"] + pd.Timedelta(minutes=2)
            max_start = row["end_dt"] - pd.Timedelta(seconds=break_seconds + 60)
            if max_start > min_start:
                if candidate < min_start:
                    candidate = min_start
                if candidate > max_start:
                    candidate = max_start
            break_end = candidate + pd.Timedelta(seconds=break_seconds)
            is_prime = 20 <= int(candidate.hour) <= 23
            is_gold = bool(is_prime and break_index == 1 and revenue_total >= 20_000)
            reference_revenue = _money(revenue_total / max(break_count, 1))
            rating_points = _safe_number(row.get("viewing_points"), 1.0)
            cpp = _safe_number(schedule_row.get("base_rate"), 1000)
            premium = 1.25 if is_gold else 1.0
            try:
                cpp_revenue = cpp_break_revenue(rating_points, break_seconds, cpp, premium=premium)
                break_revenue = _money(retention_adjusted_revenue(cpp_revenue, retention / 100))
            except ValueError:
                break_revenue = reference_revenue
            breaks.append(
                {
                    "id": f"{program_key}-br-{break_index}",
                    "program_id": program_id,
                    "program_key": program_key,
                    "program_title": row.get("Title", "Untitled"),
                    "lane": lane,
                    "channel": channel,
                    "day": day,
                    "date": row["start_dt"].date().isoformat(),
                    "program_type": program_type,
                    "position": schedule_row.get("position", "middle"),
                    "break_type": schedule_row.get("break_type", "regular"),
                    "break_num_in_program": break_index,
                    "breaks_in_program": break_count,
                    "start_time": candidate.strftime("%H:%M"),
                    "end_time": break_end.strftime("%H:%M"),
                    "duration_sec": break_seconds,
                    "sponsorships_count": 1 if is_gold else 0,
                    "is_gold": is_gold,
                    "source": "Model",
                    "rating_predicted": round(_safe_number(row.get("viewing_points"), 1.0), 2),
                    "cpp": _money(_safe_number(schedule_row.get("base_rate"), 1000)),
                    "revenue_reference": reference_revenue,
                    "revenue_premium": premium,
                    "revenue_calculated": break_revenue,
                    "retention": retention,
                    "status": "at_risk" if retention < 72 else "ready",
                }
            )

    return {
        "programs": programs,
        "breaks": breaks,
        "summary": {
            "programs": len(programs),
            "breaks": len(breaks),
            "ad_seconds": int(sum(item["duration_sec"] for item in breaks)),
            "revenue": _money(sum(item["revenue_calculated"] for item in breaks)),
        },
    }


def _build_optimizer_plan(request: ScenarioRequest | None = None) -> dict[str, Any]:
    request = request or ScenarioRequest()
    if not _ENGINE_AVAILABLE:
        return {
            "summary": {
                **_summarize_schedule(_load_break_schedule()),
                "is_compliant": False,
            },
            "controls": _model_dump(request),
            "engine": "unavailable",
        }
    payload = run_scenario(
        revenue_weight=request.revenue_weight,
        retention_floor=request.retention_floor,
        max_breaks_per_hour=request.max_breaks_per_hour,
    )
    summary = payload.setdefault("summary", {})
    summary["is_compliant"] = bool(summary.get("is_compliant", summary.get("compliant", False)))
    return payload


def _build_recommendations(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    if schedule.empty:
        return []

    frame = schedule.copy()
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0.0), errors="coerce").fillna(0.0)
    frame = frame.sort_values(["predicted_revenue", "predicted_retention"], ascending=[False, True])

    actions = []
    for idx, row in frame.head(5).iterrows():
        retention = _percent(row.get("predicted_retention", 0.0))
        revenue = _money(row.get("predicted_revenue", 0))
        risk = "High" if retention < 70 else "Medium" if retention < 74 else "Low"
        actions.append(
            {
                "id": f"rec-{idx}",
                "title": f"Review {row.get('position', 'middle')} {row.get('break_type', 'medium')} break",
                "title_he": f"בדיקת ברייק {row.get('break_type', 'בינוני')} במיקום {row.get('position', 'אמצע')}",
                "program_type": row.get("program_type", "Other"),
                "impact": revenue,
                "retention": round(retention, 1),
                "risk": risk,
                "rationale": "Revenue opportunity is strong while guardrails remain within the selected scenario.",
                "rationale_he": "פוטנציאל ההכנסה גבוה, והבקרות עדיין עומדות בתרחיש שנבחר.",
            }
        )
    return actions


def _build_frontier(summary: dict[str, Any]) -> list[dict[str, float]]:
    revenue = _safe_number(summary.get("projected_revenue"), 0)
    retention = _safe_number(summary.get("average_retention"), 74)
    points = []
    for offset in range(-3, 4):
        points.append(
            {
                "retention": round(retention + offset * 1.15, 1),
                "revenue": round(max(0, revenue * (1 - offset * 0.035)), 2),
                "selected": offset == 0,
            }
        )
    return points


def _infer_hourly_ad_seconds(schedule: pd.DataFrame) -> pd.Series:
    if schedule.empty:
        return pd.Series(dtype=float)

    frame = schedule.copy()
    frame["ad_seconds"] = pd.to_numeric(
        frame.get("total_break_time", frame.get("break_length", 0)),
        errors="coerce",
    ).fillna(0)

    if "hour" not in frame.columns:
        candidate = None
        for column in ["start_time", "time", "break_start", "Start time"]:
            if column in frame.columns:
                candidate = pd.to_datetime(frame[column], errors="coerce")
                break
        if candidate is not None:
            frame["hour"] = candidate.dt.hour
        else:
            frame["hour"] = 0

    group_columns = [column for column in ["date", "Channel", "channel", "hour"] if column in frame.columns]
    if not group_columns:
        group_columns = ["hour"]
    return frame.groupby(group_columns)["ad_seconds"].sum()


def _infer_hourly_break_counts(schedule: pd.DataFrame) -> pd.Series:
    if schedule.empty:
        return pd.Series(dtype=float)
    frame = schedule.copy()
    frame["break_count"] = pd.to_numeric(frame.get("num_breaks", 1), errors="coerce").fillna(1)
    group_columns = [column for column in ["date", "Channel", "channel", "hour"] if column in frame.columns]
    if not group_columns:
        group_columns = ["program_type"] if "program_type" in frame.columns else []
    if not group_columns:
        return frame["break_count"]
    return frame.groupby(group_columns)["break_count"].sum()


def _guardrail_breaks_from_operations(operations: dict[str, Any]) -> list[GuardrailBreak]:
    out: list[GuardrailBreak] = []
    for item in operations.get("breaks", []):
        start_seconds = _time_to_seconds(item.get("start_time"))
        duration_seconds = _safe_number(item.get("duration_sec"), 0)
        if duration_seconds <= 0:
            continue
        out.append(
            GuardrailBreak(
                channel=str(item.get("channel") or "Channel"),
                day=str(item.get("day") or ""),
                hour=int(start_seconds // 3600),
                start_seconds=start_seconds,
                duration_seconds=duration_seconds,
                program_type=str(item.get("program_type") or "Other"),
                retention=_ratio(item.get("retention")),
                is_gold=bool(item.get("is_gold")),
            )
        )
    return out


def _max_group_sum(items: list[GuardrailBreak], key_fn: Any, value_fn: Any) -> float:
    grouped: dict[Any, float] = {}
    for item in items:
        key = key_fn(item)
        grouped[key] = grouped.get(key, 0.0) + float(value_fn(item))
    return max(grouped.values(), default=0.0)


def _max_group_count(items: list[GuardrailBreak], key_fn: Any) -> int:
    grouped: dict[Any, int] = {}
    for item in items:
        key = key_fn(item)
        grouped[key] = grouped.get(key, 0) + 1
    return max(grouped.values(), default=0)


def _min_break_spacing_seconds(items: list[GuardrailBreak]) -> float | None:
    grouped: dict[tuple[str, str], list[GuardrailBreak]] = {}
    for item in items:
        grouped.setdefault((item.channel, item.day), []).append(item)
    gaps: list[float] = []
    for breaks in grouped.values():
        ordered = sorted(breaks, key=lambda item: item.start_seconds)
        for previous, current in zip(ordered, ordered[1:]):
            gaps.append(current.start_seconds - (previous.start_seconds + previous.duration_seconds))
    return min(gaps) if gaps else None


def _guardrail_compliance_from_breaks(items: list[GuardrailBreak], settings: KairosSettings) -> dict[str, Any] | None:
    if not items:
        return None

    guardrails = _settings_to_guardrails(settings)
    violations = evaluate_guardrails(items, guardrails)
    violation_counts: dict[str, int] = {}
    for violation in violations:
        violation_counts[violation.code] = violation_counts.get(violation.code, 0) + 1

    protected_types = {item.lower() for item in settings.protected_program_types}
    protected_items = [item for item in items if item.program_type.lower() in protected_types]
    max_hourly_seconds = _max_group_sum(items, lambda item: (item.channel, item.day, item.hour), lambda item: item.duration_seconds)
    max_protected_seconds = _max_group_sum(
        protected_items,
        lambda item: (item.channel, item.day, item.hour),
        lambda item: item.duration_seconds,
    )
    min_spacing = _min_break_spacing_seconds(items)
    observed_spacing = min_spacing if min_spacing is not None else settings.min_break_spacing_minutes * 60
    max_daily_seconds = _max_group_sum(items, lambda item: (item.channel, item.day), lambda item: item.duration_seconds)
    max_gold_breaks = _max_group_count(
        [item for item in items if item.is_gold],
        lambda item: (item.channel, item.day),
    )
    min_retention = min((item.retention for item in items), default=0.0)

    checks = [
        {
            "id": "hourly_ad_load",
            "violation_code": "hourly_ad_load",
            "label_en": "Ad minutes per broadcast hour",
            "label_he": "דקות פרסום לשעת שידור",
            "observed": round(max_hourly_seconds / 60, 2),
            "limit": settings.max_ad_minutes_per_hour,
            "unit": "minutes/hour",
        },
        {
            "id": "break_density",
            "violation_code": "breaks_per_hour",
            "label_en": "Breaks per hour",
            "label_he": "מספר ברייקים בשעה",
            "observed": _max_group_count(items, lambda item: (item.channel, item.day, item.hour)),
            "limit": settings.max_breaks_per_hour,
            "unit": "breaks/hour",
        },
        {
            "id": "retention_floor",
            "violation_code": "retention_floor",
            "label_en": "Viewer retention floor",
            "label_he": "רף שימור צפייה",
            "observed": round(min_retention * 100, 1),
            "limit": round(settings.min_retention_floor * 100, 1),
            "unit": "%",
        },
        {
            "id": "protected_programs",
            "violation_code": "hourly_ad_load",
            "label_en": "Protected programme ad load",
            "label_he": "עומס פרסום בתוכן מוגן",
            "observed": round(max_protected_seconds / 60, 2),
            "limit": settings.protected_program_max_ad_minutes_per_hour,
            "unit": "minutes/hour",
        },
        {
            "id": "break_spacing",
            "violation_code": "break_spacing",
            "label_en": "Minimum break spacing",
            "label_he": "מרווח מינימלי בין ברייקים",
            "observed": round(observed_spacing / 60, 2),
            "limit": settings.min_break_spacing_minutes,
            "unit": "minutes",
        },
        {
            "id": "daily_ad_load",
            "violation_code": "daily_ad_load",
            "label_en": "Daily ad load",
            "label_he": "עומס פרסום יומי",
            "observed": round(max_daily_seconds / 60, 2),
            "limit": settings.max_daily_ad_minutes,
            "unit": "minutes/day",
        },
        {
            "id": "gold_breaks",
            "violation_code": "gold_breaks",
            "label_en": "Gold breaks per day",
            "label_he": "ברייקי זהב ביום",
            "observed": max_gold_breaks,
            "limit": settings.gold_breaks_max_per_day,
            "unit": "breaks/day",
        },
    ]

    for check in checks:
        count = violation_counts.get(check["violation_code"], 0)
        if check["id"] == "protected_programs":
            count = sum(
                1
                for violation in violations
                if violation.code == "hourly_ad_load" and "protected programme" in violation.detail
            )
        check["status"] = "at_risk" if count else "compliant"
        check["violations"] = count

    return {
        "checks": checks,
        "violations": [
            {
                "code": violation.code,
                "scope": violation.scope,
                "observed": violation.observed,
                "limit": violation.limit,
                "detail": violation.detail,
            }
            for violation in violations[:200]
        ],
        "status": "at_risk" if violations else "compliant",
    }


def _build_compliance(
    schedule: pd.DataFrame,
    settings: KairosSettings,
    operations: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if operations is None:
        operations = _build_break_operations(_load_programmes(), schedule)
    guardrail_items = _guardrail_breaks_from_operations(operations)
    break_level = _guardrail_compliance_from_breaks(guardrail_items, settings)
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
    hourly_seconds = _infer_hourly_ad_seconds(schedule)
    hourly_breaks = _infer_hourly_break_counts(schedule)
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
            "status": "compliant" if summary["average_retention"] >= settings.min_retention_floor * 100 else "at_risk",
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


def _records(frame: pd.DataFrame, limit: int = 200) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    cleaned = frame.head(limit).replace({pd.NA: None}).where(pd.notna(frame.head(limit)), None)
    return cleaned.to_dict("records")


def _build_inventory(spots: pd.DataFrame) -> dict[str, Any]:
    if spots.empty:
        return {"summary": {"spots": 0, "revenue": 0, "seconds": 0}, "by_channel": [], "by_hour": []}

    frame = spots.copy()
    frame["revenue_ils"] = pd.to_numeric(frame.get("revenue_ils", 0), errors="coerce").fillna(0)
    frame["Duration"] = pd.to_numeric(frame.get("Duration", 0), errors="coerce").fillna(0)
    frame["hour_of_day"] = pd.to_numeric(frame.get("hour_of_day", 0), errors="coerce").fillna(0).astype(int)
    frame["target"] = frame.get("is_target_channel", False).astype(str).str.lower().isin(["true", "1", "yes"])
    valid_hours = frame[(frame["hour_of_day"] >= 0) & (frame["hour_of_day"] <= 23)]

    by_channel = (
        frame.groupby("Channel", dropna=False)
        .agg(spots=("Campaign", "count"), seconds=("Duration", "sum"), revenue=("revenue_ils", "sum"), target_spots=("target", "sum"))
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(12)
    )
    by_hour = (
        valid_hours.groupby("hour_of_day", dropna=False)
        .agg(spots=("Campaign", "count"), seconds=("Duration", "sum"), revenue=("revenue_ils", "sum"))
        .reset_index()
        .sort_values("hour_of_day")
    )

    return {
        "summary": {
            "spots": int(len(frame)),
            "revenue": _money(frame["revenue_ils"].sum()),
            "seconds": int(frame["Duration"].sum()),
        },
        "by_channel": _records(by_channel),
        "by_hour": _records(by_hour, 24),
    }


def _build_campaigns(spots: pd.DataFrame) -> dict[str, Any]:
    if spots.empty:
        return {"campaigns": []}

    frame = spots.copy()
    frame["revenue_ils"] = pd.to_numeric(frame.get("revenue_ils", 0), errors="coerce").fillna(0)
    frame["Duration"] = pd.to_numeric(frame.get("Duration", 0), errors="coerce").fillna(0)
    grouped = (
        frame.groupby(["Campaign", "advertiser_id"], dropna=False)
        .agg(
            spots=("Campaign", "count"),
            seconds=("Duration", "sum"),
            revenue=("revenue_ils", "sum"),
            channels=("Channel", "nunique"),
            last_airing=("Date", "max"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(50)
    )
    return {"campaigns": _records(grouped)}


def _build_break_library(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {"breaks": []}

    frame = schedule.copy()
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0), errors="coerce").fillna(0)
    frame["total_break_time"] = pd.to_numeric(frame.get("total_break_time", 0), errors="coerce").fillna(0)
    frame["priority"] = frame["predicted_revenue"] * frame["predicted_retention"].clip(lower=0.1)
    frame = frame.sort_values("priority", ascending=False).head(80)
    frame["status"] = frame["predicted_retention"].map(lambda value: "at_risk" if _percent(value) < 72 else "ready")
    return {"breaks": _records(frame)}


def _build_forecasts(schedule: pd.DataFrame) -> dict[str, Any]:
    if schedule.empty:
        return {"by_day": [], "scenarios": []}

    frame = schedule.copy()
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0), errors="coerce").fillna(0)
    by_day = (
        frame.groupby("day", dropna=False)
        .agg(revenue=("predicted_revenue", "sum"), retention=("predicted_retention", "mean"), breaks=("num_breaks", "sum"))
        .reset_index()
    )
    summary = _summarize_schedule(schedule)
    scenarios = [
        {"name": "Retention guardrail", "revenue": round(summary["projected_revenue"] * 0.94, 2), "retention": max(summary["average_retention"], 74.0)},
        {"name": "Balanced", "revenue": summary["projected_revenue"], "retention": summary["average_retention"]},
        {"name": "Revenue priority", "revenue": round(summary["projected_revenue"] * 1.08, 2), "retention": max(0, summary["average_retention"] - 1.6)},
    ]
    return {"by_day": _records(by_day), "scenarios": scenarios}


def _build_reports(schedule: pd.DataFrame, settings: KairosSettings) -> dict[str, Any]:
    summary = _summarize_schedule(schedule)
    compliance = _build_compliance(schedule, settings)
    return {
        "reports": [
            {"id": "weekly-plan", "title": "Weekly traffic plan", "status": "ready", "rows": int(len(schedule)), "owner": "Traffic"},
            {"id": "compliance", "title": "Compliance and guardrails", "status": compliance["status"], "rows": len(compliance["checks"]), "owner": "Legal / Ops"},
            {"id": "revenue", "title": "Revenue forecast", "status": "ready", "rows": summary["total_breaks"], "owner": "Revenue"},
            {"id": "data-quality", "title": "Source file audit", "status": "ready", "rows": 8, "owner": "Data"},
        ]
    }


def _signature(paths: list[Path]) -> tuple[tuple[str, int, int], ...]:
    result = []
    for path in paths:
        if path.exists():
            stat = path.stat()
            result.append((str(path), stat.st_mtime_ns, stat.st_size))
        else:
            result.append((str(path), 0, 0))
    return tuple(result)


@lru_cache(maxsize=16)
def _overview_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    schedule = _load_break_schedule()
    programmes = _load_programmes()
    spots = _load_spots()
    summary = _summarize_schedule(schedule)
    settings = _load_settings()
    break_operations = _build_break_operations(programmes, schedule)
    return {
        "brand": "Kairos",
        "workspace": "KAI Network",
        "data_freshness": datetime.fromtimestamp(
            max(
                [
                    path.stat().st_mtime
                    for path in [OUTPUT_DIR / "weekly_break_schedule.csv", DATA_DIR / "Programmes.csv", DATA_DIR / "Spots.csv"]
                    if path.exists()
                ]
                or [time.time()]
            ),
            tz=timezone.utc,
        ).isoformat(),
        "summary": summary,
        "source_counts": {
            "programmes": int(len(programmes)),
            "spots": int(len(spots)),
            "planned_break_rows": int(len(schedule)),
        },
        "recommendations": _build_recommendations(schedule),
        "frontier": _build_frontier(summary),
        "settings": _model_dump(settings),
        "compliance": _build_compliance(schedule, settings, break_operations),
    }


@lru_cache(maxsize=16)
def _schedule_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    programmes = _load_programmes()
    break_schedule = _load_break_schedule()
    return {
        "rows": _build_schedule_canvas(programmes, break_schedule),
        "break_operations": _build_break_operations(programmes, break_schedule),
        "break_schedule": break_schedule.head(200).replace({pd.NA: None}).where(pd.notna(break_schedule.head(200)), None).to_dict("records"),
    }


@lru_cache(maxsize=16)
def _inventory_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_inventory(_load_spots())


@lru_cache(maxsize=16)
def _break_library_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_break_library(_load_break_schedule())


@lru_cache(maxsize=16)
def _campaigns_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_campaigns(_load_spots())


@lru_cache(maxsize=16)
def _forecasts_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_forecasts(_load_break_schedule())


@lru_cache(maxsize=16)
def _reports_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_reports(_load_break_schedule(), _load_settings())


@lru_cache(maxsize=16)
def _break_operations_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return _build_break_operations(_load_programmes(), _load_break_schedule())


@lru_cache(maxsize=16)
def _impact_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return {
        "program_type_impacts": _load_impact(OUTPUT_DIR / "program_type_impacts.csv"),
        "position_impacts": _load_impact(OUTPUT_DIR / "position_impacts.csv"),
        "length_impacts": _load_impact(OUTPUT_DIR / "length_impacts.csv"),
        "coefficient_impacts": _load_measured_impact_summary(MODELS_DIR / "tv_break_coefficients.json"),
    }


def _decisions_path() -> Path:
    return DATA_DIR / "kairos_decisions.json"


def _load_decisions() -> list[dict[str, Any]]:
    path = _decisions_path()
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError, TypeError):
        return []
    return payload if isinstance(payload, list) else []


def _save_decision(request: BreakDecisionRequest) -> dict[str, Any]:
    decisions = _load_decisions()
    record = {
        "id": f"decision-{int(time.time() * 1000)}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        **_model_dump(request),
    }
    decisions.insert(0, record)
    path = _decisions_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(decisions[:500], handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return record


app = FastAPI(
    title="Kairos API",
    version="0.1.0",
    description="Operational API for TV ad break revenue optimization.",
)

allowed_origins = os.getenv(
    "KAIROS_CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in allowed_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Operational capabilities live in focused modules to keep this file lean.
from kairos_api.advertiser_conditions import router as advertiser_conditions_router  # noqa: E402
from kairos_api.advertisers import router as advertisers_router  # noqa: E402
from kairos_api.exporters import router as exporters_router  # noqa: E402
from kairos_api.overrides import router as overrides_router  # noqa: E402
from kairos_api.uploads import router as uploads_router  # noqa: E402

app.include_router(uploads_router)
app.include_router(advertisers_router)
app.include_router(advertiser_conditions_router)
app.include_router(exporters_router)
app.include_router(overrides_router)


@app.get("/api/health")
def health() -> dict[str, Any]:
    schedule = _load_break_schedule()
    return {
        "status": "ok",
        "project": "Kairos",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "has_schedule": not schedule.empty,
        "has_model": (MODELS_DIR / "tv_break_posterior.pkl").exists(),
    }


@app.get("/api/settings")
def get_settings() -> dict[str, Any]:
    return _model_dump(_load_settings())


@app.put("/api/settings")
def update_settings(settings: KairosSettings) -> dict[str, Any]:
    return _model_dump(_save_settings(settings))


@app.get("/api/compliance")
def compliance() -> dict[str, Any]:
    return _build_compliance(_load_break_schedule(), _load_settings())


@app.get("/api/guardrails")
def guardrails() -> dict[str, Any]:
    return compliance()


@app.get("/api/optimizer-plan")
def optimizer_plan() -> dict[str, Any]:
    return _build_optimizer_plan()


@app.post("/api/optimizer-plan")
def create_optimizer_plan(request: ScenarioRequest) -> dict[str, Any]:
    return _build_optimizer_plan(request)


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    return _overview_cached(
        _signature([OUTPUT_DIR / "weekly_break_schedule.csv", DATA_DIR / "Programmes.csv", DATA_DIR / "Spots.csv", SETTINGS_PATH])
    )


@app.get("/api/schedule")
def schedule() -> dict[str, Any]:
    return _schedule_cached(_signature([DATA_DIR / "Programmes.csv", OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv"]))


@app.get("/api/break-operations")
def break_operations() -> dict[str, Any]:
    return _break_operations_cached(_signature([DATA_DIR / "Programmes.csv", OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv"]))


@app.get("/api/break-decisions")
def break_decisions() -> dict[str, Any]:
    return {"decisions": _load_decisions()}


@app.post("/api/break-decisions")
def create_break_decision(request: BreakDecisionRequest) -> dict[str, Any]:
    return {"decision": _save_decision(request)}


@app.get("/api/impact")
def impact() -> dict[str, Any]:
    return _impact_cached(
        _signature(
            [
                OUTPUT_DIR / "program_type_impacts.csv",
                OUTPUT_DIR / "position_impacts.csv",
                OUTPUT_DIR / "length_impacts.csv",
                MODELS_DIR / "tv_break_coefficients.json",
            ]
        )
    )


@app.get("/api/inventory")
def inventory() -> dict[str, Any]:
    return _inventory_cached(_signature([DATA_DIR / "Spots.csv"]))


@app.get("/api/break-library")
def break_library() -> dict[str, Any]:
    return _break_library_cached(_signature([OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv"]))


@app.get("/api/campaigns")
def campaigns() -> dict[str, Any]:
    return _campaigns_cached(_signature([DATA_DIR / "Spots.csv"]))


@app.get("/api/forecasts")
def forecasts() -> dict[str, Any]:
    return _forecasts_cached(_signature([OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv"]))


@app.get("/api/reports")
def reports() -> dict[str, Any]:
    return _reports_cached(
        _signature([OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv", DATA_DIR / "Programmes.csv", SETTINGS_PATH])
    )


@app.get("/api/files")
def files() -> dict[str, Any]:
    paths = [
        DATA_DIR / "Dayparts.csv",
        DATA_DIR / "Programmes.csv",
        DATA_DIR / "Spots.csv",
        DATA_DIR / "rate_card_premiums.csv",
        DATA_DIR / "advertiser_rules.csv",
        OUTPUT_DIR / "weekly_break_schedule.csv",
        ROOT / "optimization_results.csv",
        MODELS_DIR / "tv_break_posterior.pkl",
    ]
    return {
        "files": [
            {
                "path": str(path.relative_to(ROOT)),
                "exists": path.exists(),
                "size": path.stat().st_size if path.exists() else 0,
                "modified": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
                if path.exists()
                else None,
            }
            for path in paths
        ]
    }


def _risk_from_retention(average_retention_percent: float, total_breaks: int) -> float:
    """A transparent risk score: lower retention and more breaks raise it."""
    return round(max(0.0, min(100.0, (78 - average_retention_percent) * 2.2 + total_breaks * 0.8)), 1)


@lru_cache(maxsize=128)
def _scenario_cached(revenue_weight: int, retention_floor: float, max_breaks_per_hour: int) -> dict[str, Any]:
    result = run_scenario(
        revenue_weight=revenue_weight,
        retention_floor=retention_floor,
        max_breaks_per_hour=max_breaks_per_hour,
    )
    summary = result["summary"]
    return {
        "summary": {
            "total_breaks": summary["total_breaks"],
            "total_ad_seconds": summary["total_ad_seconds"],
            "projected_revenue": summary["projected_revenue"],
            "average_retention": summary["average_retention"],
            "risk_score": _risk_from_retention(summary["average_retention"], summary["total_breaks"]),
        },
        "controls": result["controls"],
        "guardrails": result["guardrails"],
        "channel": result["channel"],
        "day": result["day"],
        "compliant": summary["compliant"],
        "engine": "kairos",
    }


@app.post("/api/scenario")
def scenario(request: ScenarioRequest) -> dict[str, Any]:
    """Run a real optimization for the scenario controls (no placeholder math).

    Falls back to the stored schedule summary only if the engine or its data is
    unavailable, reporting that honestly instead of inventing numbers.
    """
    if _ENGINE_AVAILABLE:
        try:
            return _scenario_cached(
                request.revenue_weight, request.retention_floor, request.max_breaks_per_hour
            )
        except Exception as exc:  # pragma: no cover - data/environment dependent
            return {
                "summary": _summarize_schedule(_load_break_schedule()),
                "controls": _model_dump(request),
                "engine": "unavailable",
                "detail": str(exc)[:300],
            }
    return {
        "summary": _summarize_schedule(_load_break_schedule()),
        "controls": _model_dump(request),
        "engine": "unavailable",
    }


class OptimizePlanRequest(BaseModel):
    """Controls for a real, in-process optimization of one channel-day."""

    channel: str | None = Field(default=None)
    day: str | None = Field(default=None)
    revenue_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    # When set, the day's real daily plan (the Wally csv) drives the decision
    # instead of the Programmes EPG; channel and day are read from the file.
    daily_input: str | None = Field(default=None)


@app.post("/api/optimize-plan")
def optimize_plan(request: OptimizePlanRequest) -> dict[str, Any]:
    """Serve a real optimal break plan, driven by the saved settings.

    This is the engine-backed counterpart to /api/optimize (which shells out to
    the trained Meridian model). It uses the live KairosSettings as guardrails,
    so the dashboard's settings page controls the optimizer directly.
    """
    if not _ENGINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Optimization engine is unavailable")
    try:
        return optimize_day_plan(
            channel=request.channel,
            day=request.day,
            revenue_weight=request.revenue_weight,
            daily_input_path=request.daily_input,
            settings=_model_dump(_load_settings()),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Reference data not found: {exc}")
    except Exception as exc:  # pragma: no cover - data/environment dependent
        raise HTTPException(status_code=503, detail=f"Optimization failed: {exc}")


@app.get("/api/parameters")
def parameters() -> dict[str, Any]:
    """Every adjustable parameter the optimizer uses, in one place.

    Surfaces the guardrails (derived from the saved settings), the declared
    optimizer assumptions, the pricing model, and the known channels, so the
    dashboard can show and edit each one.
    """
    settings = _load_settings()
    payload: dict[str, Any] = {"settings": _model_dump(settings)}
    if not _ENGINE_AVAILABLE:
        payload["engine"] = "unavailable"
        return payload
    payload["guardrails"] = _asdict(guardrails_from_settings(_model_dump(settings)))
    payload["assumptions"] = _asdict(OptimizerAssumptions())
    payload["channels"] = list(KAIROS_CHANNELS)
    try:
        pricing = PricingModel.from_yaml()
        payload["pricing"] = {
            "base_price_per_second_per_tvr_point": pricing.base_price,
            "program_type_premiums": pricing.program_type_premiums,
            "ad_type_premiums": pricing.ad_type_premiums,
            "position_premiums": {str(k): v for k, v in pricing.position_premiums.items()},
            "day_of_week_premiums": {str(k): v for k, v in pricing.day_of_week_premiums.items()},
        }
    except Exception as exc:  # pragma: no cover - config dependent
        payload["pricing"] = {"error": str(exc)[:200]}
    return payload


@app.post("/api/optimize")
def optimize(request: OptimizeRequest) -> dict[str, Any]:
    model_path = _safe_path(request.model_path)
    programs_path = _safe_path(request.programs_path)
    output_path = _safe_path(request.output_path)
    command = [
        sys.executable,
        str(ROOT / "run_optimization.py"),
        "--model-path",
        str(model_path),
        "--programs-path",
        str(programs_path),
        "--output-path",
        str(output_path),
        "--min-retention",
        str(request.min_retention),
        "--max-breaks-per-hour",
        str(request.max_breaks_per_hour),
        "--budget",
        str(request.budget),
    ]
    if request.spots_inventory:
        command.extend(["--spots-inventory", str(_safe_path(request.spots_inventory))])

    started = time.time()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=int(os.getenv("KAIROS_OPTIMIZE_TIMEOUT_SECONDS", "300")),
        check=False,
    )
    return {
        "status": "success" if completed.returncode == 0 else "error",
        "return_code": completed.returncode,
        "duration_seconds": round(time.time() - started, 2),
        "stdout": completed.stdout[-4000:],
        "stderr": completed.stderr[-4000:],
        "output_path": str(output_path.relative_to(ROOT)),
    }


# Serve the built dashboard (Vite `dist/`) from the same container in production.
# Mounted last so it never shadows the `/api/*` routes above; only active when a
# build is present, so local API-only runs are unaffected.
_DASHBOARD_DIST = ROOT / "tv-break-dashboard" / "dist"
if _DASHBOARD_DIST.is_dir():
    from fastapi.staticfiles import StaticFiles  # noqa: E402

    app.mount("/", StaticFiles(directory=str(_DASHBOARD_DIST), html=True), name="dashboard")
