"""Persist a real optimized weekly break schedule for the dashboard.

The dashboard renders its schedule canvas, break-operations board and
recommendation list from ``output/weekly_break_schedule.csv``. When that file is
missing the API falls back to placeholder revenue and retention numbers. This
module removes that fallback in practice: it runs the real engine across every
channel-day in the source and writes one row per programme segment with the
columns the dashboard consumes, so every number on those screens is computed,
not invented.

Each row is one programme segment. ``num_breaks`` is the optimizer's decision,
``predicted_revenue`` and ``predicted_retention`` come straight from the
:class:`~kairos.optimize.optimizer.SegmentPlan`, ``base_rate`` is the segment's
effective per-second rate (CPP times premium) and ``break_type`` is derived from
the break length. Nothing on the row is fabricated: a programme the optimizer
left without breaks earns zero and keeps its full baseline retention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional

import pandas as pd

from kairos.data import ProgramClassifier
from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import OptimizerAssumptions, PricingModel
from kairos.service import guardrails_from_settings

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_PATH = ROOT / "output" / "weekly_break_schedule.csv"

SECONDS_PER_MINUTE = 60

# Column order the dashboard's schedule readers expect (extra columns are
# ignored by them, missing ones trigger the placeholder fallback we are killing).
COLUMNS = [
    "channel",
    "date",
    "day",
    "program_type",
    "start_time",
    "num_breaks",
    "break_length",
    "total_break_time",
    "predicted_revenue",
    "predicted_retention",
    "position",
    "break_type",
    "base_rate",
]


def _clock(start_seconds: float) -> str:
    total_minutes = int(start_seconds // SECONDS_PER_MINUTE)
    return f"{(total_minutes // 60) % 24:02d}:{total_minutes % 60:02d}"


def _weekday_abbrev(day: str) -> str:
    """Map a ``YYYY-MM-DD`` date to the dashboard's day key (``Mon`` .. ``Sun``)."""
    return pd.Timestamp(day).strftime("%a")


def _break_type(break_seconds: float) -> str:
    """Label a break by its length, the way the operations board groups them."""
    if break_seconds < 90:
        return "short"
    if break_seconds < 180:
        return "medium"
    return "long"


def _channel_days(programmes: pd.DataFrame) -> list[tuple[str, str]]:
    if "start_dt" not in programmes.columns:
        raise ValueError("programmes frame must have a start_dt column (use load_programmes)")
    valid = programmes[programmes["start_dt"].notna()]
    if valid.empty:
        return []
    pairs = (
        valid.assign(_day=valid["start_dt"].dt.strftime("%Y-%m-%d"))[["Channel", "_day"]]
        .drop_duplicates()
        .sort_values(["_day", "Channel"])
    )
    return [(str(channel), str(day)) for channel, day in pairs.itertuples(index=False, name=None)]


def build_weekly_schedule(
    programmes: Optional[pd.DataFrame] = None,
    *,
    programmes_path: Optional[str] = None,
    pricing: Optional[PricingModel] = None,
    assumptions: Optional[OptimizerAssumptions] = None,
    settings: Optional[Mapping[str, Any]] = None,
    revenue_weight: Optional[float] = None,
    classifier: Optional[ProgramClassifier] = None,
) -> pd.DataFrame:
    """Optimise every channel-day and return one schedule row per segment.

    ``settings`` are the dashboard's KairosSettings (mapped onto guardrails);
    ``revenue_weight`` overrides the assumptions default. The frame is sorted by
    day then channel so the output is deterministic.
    """
    pricing = pricing or PricingModel.from_yaml()
    assumptions = assumptions or OptimizerAssumptions()
    classifier = classifier or ProgramClassifier.from_yaml()
    guardrails = guardrails_from_settings(settings) if settings else Guardrails()
    weight = revenue_weight if revenue_weight is not None else assumptions.revenue_weight
    if programmes is None:
        programmes = load_programmes(programmes_path)

    rows: list[dict[str, Any]] = []
    for channel, day in _channel_days(programmes):
        segments = build_segments_from_programmes(
            programmes, classifier, pricing, assumptions=assumptions, channel=channel, day=day,
        )
        if not segments:
            continue
        result = optimize_breaks(segments, guardrails, revenue_weight=weight)
        plans = {plan.segment_id: plan for plan in result.segments}
        for segment in segments:
            plan = plans.get(segment.segment_id)
            num_breaks = plan.num_breaks if plan else 0
            # A 0-break segment keeps its baseline retention and earns nothing.
            retention = plan.retention if plan else segment.retention_baseline
            revenue = plan.revenue if plan else 0.0
            rows.append(
                {
                    "channel": segment.channel,
                    "date": segment.day,
                    "day": _weekday_abbrev(segment.day),
                    "program_type": segment.program_type,
                    "start_time": _clock(segment.start_seconds),
                    "num_breaks": num_breaks,
                    "break_length": round(segment.break_length_seconds, 1),
                    "total_break_time": round(num_breaks * segment.break_length_seconds, 1),
                    "predicted_revenue": round(revenue, 2),
                    "predicted_retention": round(retention, 4),
                    "position": "middle",
                    "break_type": _break_type(segment.break_length_seconds),
                    "base_rate": round(segment.cpp * segment.premium, 4),
                }
            )

    return pd.DataFrame(rows, columns=COLUMNS)


def write_weekly_schedule(
    path: Optional[str | Path] = None,
    *,
    frame: Optional[pd.DataFrame] = None,
    **kwargs: Any,
) -> Path:
    """Build (unless ``frame`` is supplied) and write the schedule CSV.

    Returns the path written. Extra keyword arguments are forwarded to
    :func:`build_weekly_schedule`.
    """
    if frame is None:
        frame = build_weekly_schedule(**kwargs)
    target = Path(path) if path is not None else DEFAULT_OUTPUT_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8")
    return target
