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


def _percent(value: Any) -> float:
    numeric = _safe_number(value, 0.0)
    if numeric <= 1.5:
        return numeric * 100
    return numeric


def _money(value: Any) -> float:
    return round(_safe_number(value, 0.0), 2)


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
    return frame.replace({pd.NA: None}).where(pd.notna(frame), None).to_dict("records")


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

    frame = programmes.copy()
    if "Start_datetime" in frame.columns:
        starts = pd.to_datetime(frame["Start_datetime"], errors="coerce")
    elif {"Date", "Start time"}.issubset(frame.columns):
        starts = pd.to_datetime(
            frame["Date"].astype(str) + " " + frame["Start time"].astype(str),
            errors="coerce",
            dayfirst=True,
        )
    else:
        starts = pd.to_datetime(frame.get("Start time"), errors="coerce")

    frame["start_dt"] = starts
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
            retention = type_summary.get("average_retention", 74 + (row["viewing_points"] % 5))
            revenue = type_summary.get("projected_revenue", row["viewing_points"] * 45000)
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


def _build_recommendations(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    if schedule.empty:
        return []

    frame = schedule.copy()
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0.74), errors="coerce").fillna(0.74)
    frame = frame.sort_values(["predicted_revenue", "predicted_retention"], ascending=[False, True])

    actions = []
    for idx, row in frame.head(5).iterrows():
        retention = _percent(row.get("predicted_retention", 0.74))
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


def _build_compliance(schedule: pd.DataFrame, settings: KairosSettings) -> dict[str, Any]:
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
        "compliance": _build_compliance(schedule, settings),
    }


@lru_cache(maxsize=16)
def _schedule_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    programmes = _load_programmes()
    break_schedule = _load_break_schedule()
    return {
        "rows": _build_schedule_canvas(programmes, break_schedule),
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
def _impact_cached(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Any]:
    del signature
    return {
        "program_type_impacts": _load_impact(OUTPUT_DIR / "program_type_impacts.csv"),
        "position_impacts": _load_impact(OUTPUT_DIR / "position_impacts.csv"),
        "length_impacts": _load_impact(OUTPUT_DIR / "length_impacts.csv"),
    }


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


@app.get("/api/overview")
def overview() -> dict[str, Any]:
    return _overview_cached(
        _signature([OUTPUT_DIR / "weekly_break_schedule.csv", DATA_DIR / "Programmes.csv", DATA_DIR / "Spots.csv", SETTINGS_PATH])
    )


@app.get("/api/schedule")
def schedule() -> dict[str, Any]:
    return _schedule_cached(_signature([DATA_DIR / "Programmes.csv", OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv"]))


@app.get("/api/impact")
def impact() -> dict[str, Any]:
    return _impact_cached(
        _signature([OUTPUT_DIR / "program_type_impacts.csv", OUTPUT_DIR / "position_impacts.csv", OUTPUT_DIR / "length_impacts.csv"])
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
    return _reports_cached(_signature([OUTPUT_DIR / "weekly_break_schedule.csv", ROOT / "optimization_results.csv", SETTINGS_PATH]))


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


@app.post("/api/scenario")
def scenario(request: ScenarioRequest) -> dict[str, Any]:
    summary = _summarize_schedule(_load_break_schedule())
    revenue_factor = 0.85 + request.revenue_weight / 100 * 0.3
    retention_penalty = max(0, request.revenue_weight - 50) * 0.035
    return {
        "summary": {
            **summary,
            "projected_revenue": _money(summary["projected_revenue"] * revenue_factor),
            "average_retention": round(max(request.retention_floor * 100, summary["average_retention"] - retention_penalty), 1),
            "risk_score": round(min(100, summary["risk_score"] + retention_penalty * 4), 1),
        },
        "controls": _model_dump(request),
    }


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
