"""Today: the operator's one read on the week, and the decisions it implies.

The overview payload and the priority-decision list, moved verbatim from
dashboard_api.py as part of the wave-zero router split. Behaviour is unchanged:
the same cache object, the same background frontier, the same honest
fresh/stale/unknown freshness verdict.

The shared reads it composes live in the frozen plan-read layer
(:mod:`kairos_api.plan_read_compliance` for the verdict,
:mod:`kairos_api.plan_read_frontier` for the curve) and are reached through the
module, never copied. The decision plane sits beside this file in
:mod:`kairos_api.overview_api_decisions`.

``GET /api/today`` is the Today surface's own read, added in wave one. It is a
projection of the same cached body ``/api/overview`` serves plus the plan
target, so the two can never disagree, and it is one round trip because the
surface it feeds has a five-second bar with zero clicks in it. The composition
lives in :mod:`kairos_api.overview_api_today`; the target and its verdict live
in :mod:`kairos_api.target_store`; the second level of the money drill lives in
:mod:`kairos_api.overview_api_drill`; the target's own three routes live in
:mod:`kairos_api.overview_api_target`, mounted through this module's router.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from fastapi import APIRouter, Query, Request

from kairos_api import (
    overview_api_drill,
    overview_api_target,
    overview_api_today,
    plan_read_compliance,
    plan_read_frontier,
    target_store,
)
from kairos_api.core import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    ROOT,
    SETTINGS_PATH,
    _augment_segment_ids,
    _load_break_schedule,
    _load_programmes,
    _load_settings,
    _load_spots,
    _model_dump,
    _money,
    _percent,
    _row_anchor,
    _signature,
    _summarize_schedule,
)

logger = logging.getLogger(__name__)

router = APIRouter()
# The plan target's three routes live beside this file under the file-size law
# and are mounted through this router, so server.py still mounts exactly one.
router.include_router(overview_api_target.router)


def _proposed_kind(risk: str, num_breaks: int, is_gold: bool) -> str | None:
    """The override a review of this break honestly implies, or None when the break
    is inherently advisory (nothing concrete to change).

    A break below the retention floor (High risk) should shed load: lower its count
    when it carries more than one break, or forbid it when it carries a single break.
    A healthy break (at or above the floor) that already earns is worth protecting:
    tag it gold, or pin its count when it is already gold. A zero-break healthy
    segment has nothing to act on.
    """
    if risk == "High":
        if num_breaks > 1:
            return "lower_count"
        if num_breaks == 1:
            return "forbid"
        return None
    if num_breaks <= 0:
        return None
    return "pin" if is_gold else "gold"


def _build_recommendations(schedule: pd.DataFrame) -> list[dict[str, Any]]:
    if schedule.empty:
        return []

    settings = _load_settings()
    owned = str(settings.operator_channel or "").strip()
    frame = _augment_segment_ids(schedule)
    frame["predicted_revenue"] = pd.to_numeric(frame.get("predicted_revenue", 0), errors="coerce").fillna(0)
    frame["predicted_retention"] = pd.to_numeric(frame.get("predicted_retention", 0.0), errors="coerce").fillna(0.0)
    frame["num_breaks"] = pd.to_numeric(frame.get("num_breaks", 0), errors="coerce").fillna(0).astype(int)

    # Competitor boundary: only the operator's OWNED channel produces
    # recommendations, and only owned-channel segments are ever offered as targets.
    # With no channel configured yet, recommendations stay advisory (channel null, no
    # segment mapping) rather than binding to a channel the operator may not own.
    scoped = frame[frame["channel"].astype(str).str.strip() == owned] if owned else frame
    if scoped.empty:
        return []
    scoped = scoped.sort_values(["predicted_revenue", "predicted_retention"], ascending=[False, True])

    # Risk labels are sourced from the operator's configured retention floor, not
    # fixed literals. Below the floor is High; within a 2-point band above it is
    # Medium; clear of the band is Low. This mirrors the honest _risk_from_retention
    # scale so the recommendation risk and the headline risk score agree.
    floor_percent = round(settings.min_retention_floor * 100, 1)

    def _risk_label(retention_percent: float) -> str:
        if retention_percent < floor_percent:
            return "High"
        if retention_percent < floor_percent + 2.0:
            return "Medium"
        return "Low"

    # Identity and grouping key on the segment's REAL distinguishing facts (its
    # programme type, clock, weekday and canonical daypart), never on the
    # constant position/break_type template the CSV once stamped on every row.
    from kairos.data.dayparts import daypart_for_hour

    def _row_hour(value: Any) -> int | None:
        text = str(value or "").strip()
        return int(text[:2]) if len(text) >= 4 and text[:2].isdigit() else None

    scoped = scoped.copy()
    scoped["_daypart"] = scoped["start_time"].map(lambda v: daypart_for_hour(_row_hour(v)) or "")
    scoped["_risk"] = scoped["predicted_retention"].map(lambda v: _risk_label(_percent(v)))

    weekday_he = {"Mon": "ביום שני", "Tue": "ביום שלישי", "Wed": "ביום רביעי", "Thu": "ביום חמישי", "Fri": "ביום שישי", "Sat": "בשבת", "Sun": "ביום ראשון"}
    # Honest per-risk rationale: the copy states what the risk band actually
    # means against the configured floor, instead of one constant sentence.
    rationale_by_risk = {
        "High": ("Predicted retention is below the configured floor; review whether this segment should shed break load.", "השימור החזוי נמוך מרצפת השימור שהוגדרה; כדאי לבחון הפחתת עומס ברייקים במקטע הזה."),
        "Medium": ("Predicted retention is within two points of the floor; review before committing.", "השימור החזוי קרוב לרצפת השימור, בטווח שתי נקודות; מומלץ לבדוק לפני אישור."),
        "Low": ("Revenue is strong and retention clears the floor; consider protecting this placement.", "ההכנסה גבוהה והשימור מעל הרצפה; שקלו להגן על השיבוץ הזה."),
    }

    actions = []
    for idx, row in scoped.head(5).iterrows():
        retention = _percent(row.get("predicted_retention", 0.0))
        revenue = _money(row.get("predicted_revenue", 0))
        num_breaks = int(row.get("num_breaks", 0))
        risk = str(row.get("_risk", "Low"))
        program_type = str(row.get("program_type", "Other"))
        daypart = str(row.get("_daypart", "")).strip()
        start_clock = str(row.get("start_time", "")).strip()
        date = str(row.get("date", "")).strip()
        weekday = str(row.get("day", "")).strip()
        segment_id = str(row.get("segment_id", "")).strip()
        anchor = _row_anchor(row)
        proposed_kind = _proposed_kind(risk, num_breaks, bool(row.get("is_gold", False)))
        # The title carries the real programme identity (type, clock, weekday,
        # date), so five recommendations can never render one generic label.
        unit = "break" if num_breaks == 1 else "breaks" if num_breaks > 1 else "segment"
        title = " ".join(part for part in ["Review the", start_clock, program_type, unit, "on", weekday, date] if part)
        unit_he = "הברייק" if num_breaks == 1 else "הברייקים" if num_breaks > 1 else "המקטע"
        day_phrase_he = weekday_he.get(weekday, "")
        title_he = f"בדיקת {unit_he} בתוכנית {program_type} בשעה {start_clock} {day_phrase_he}, {date}".replace("  ", " ")
        rationale, rationale_he = rationale_by_risk[risk]
        # Candidate owned-channel segments this review resolves to, grouped on the
        # same real facts (programme type, daypart, risk band). Only produced for
        # the owned channel with a real segment_id; never fabricated for an
        # aggregate advisory.
        candidates: list[dict[str, Any]] = []
        if owned and segment_id:
            same = scoped[
                (scoped["program_type"].astype(str) == program_type)
                & (scoped["_daypart"] == daypart)
                & (scoped["_risk"] == risk)
            ]
            candidates = [
                {"segment_id": str(cand.get("segment_id", "")).strip(), "anchor": _row_anchor(cand)}
                for _, cand in same.head(12).iterrows()
            ]
        actionable = bool(owned and segment_id and proposed_kind)
        actions.append(
            {
                "id": f"rec-{idx}",
                "title": title,
                "title_he": title_he,
                "program_type": program_type,
                # Real identity fields behind the title, so consumers can group
                # or render without re-parsing display copy.
                "start_clock": start_clock,
                "date": date,
                "weekday": weekday,
                "daypart": daypart or None,
                "num_breaks": num_breaks,
                "impact": revenue,
                "retention": round(retention, 1),
                "risk": risk,
                "rationale": rationale,
                "rationale_he": rationale_he,
                # Decision-plane enrichment: the owned channel, the concrete
                # owned-channel segment(s) this resolves to, and the override kind the
                # review implies. non_actionable marks an advisory that cannot honestly
                # bind to a segment (no owned channel, or nothing to change).
                "channel": owned or None,
                "actionable": actionable,
                "non_actionable": not actionable,
                "segment_id": segment_id if actionable else None,
                "anchor": anchor if actionable else None,
                "proposed_kind": proposed_kind if actionable else None,
                "candidates": candidates,
            }
        )
    return actions


@lru_cache(maxsize=16)
def _overview_cached(signature: tuple[tuple[str, int, int], ...], scope: str | None = None) -> dict[str, Any]:
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
                    for path in [
                        OUTPUT_DIR / "weekly_break_schedule.csv",
                        DATA_DIR / "reference" / "Programmes.xlsx",
                        DATA_DIR / "reference" / "Spots.xlsx",
                        DATA_DIR / "Programmes.csv",
                        DATA_DIR / "Spots.csv",
                    ]
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
        "frontier_scope": scope or None,
        "settings": _model_dump(settings),
        # plan_read_compliance.build_compliance grades the FULL committed plan geometry and ignores
        # its operations argument entirely, so the cold-overview path no longer
        # computes the truncated break-operations board just to discard it.
        "compliance": plan_read_compliance.build_compliance(schedule, settings),
    }


@router.get("/api/overview", tags=["dashboard"])
def overview(
    scope: str | None = Query(
        default=None,
        description="Optional frontier scope: 'channel:<id>' or 'day:<date>'. Only the owned channel is selectable. Omit for the whole-default frontier.",
    ),
) -> dict[str, Any]:
    body = dict(_overview_cached(
        _signature([
            OUTPUT_DIR / "weekly_break_schedule.csv",
            DATA_DIR / "reference" / "Programmes.xlsx",
            DATA_DIR / "reference" / "Spots.xlsx",
            DATA_DIR / "Programmes.csv",
            DATA_DIR / "Spots.csv",
            SETTINGS_PATH,
        ]),
        scope or None,
    ))
    # The frontier is a slow optimizer sweep, computed in the background so the
    # overview never blocks on it. Merge its current state into the response.
    overview_settings = _load_settings()
    points, net_bundle, status = plan_read_frontier.frontier_state(overview_settings, scope or None)
    body["frontier"] = points
    body["frontier_status"] = status
    # Additive: the single net-focused scenario computed beside the sweep (the
    # same runner and saved guardrails under objective_mode='revenue_net'),
    # carrying the frontier-point fields plus its label id. Null while the
    # background sweep is computing or when the run failed; never fabricated.
    body["frontier_net_point"] = (net_bundle or {}).get("net_point") if status == "ready" else None
    # Honest disclosure of what the frontier curve actually measures. Each point is
    # a single representative-day REFINED optimum (run_scenario refine=True) on the
    # owned channel, swept across the retention floor at the saved revenue weight,
    # not the whole-week plan behind the projected_revenue headline it sits next to.
    # Surfaced as structured metadata so the dashboard can label the curve and the
    # operator never reads a one-day estimate as the saved weekly total.
    owned_channel = str(overview_settings.operator_channel or "").strip()
    body["frontier_basis"] = {
        "scope": "representative_day",
        "channel": owned_channel or None,
        "method": "refined_floor_sweep",
        "swept": "retention_floor",
        "disclosure": "This frontier sweeps the retention floor at your saved revenue weight, each point a refined single representative-day optimum for the owned channel, not the saved weekly plan total.",
    }
    # Schedule freshness: is the saved schedule the dashboard renders still in
    # step with its inputs? Computed FRESH here (never inside _overview_cached),
    # because a settings/constraints/pricing edit does not clear the overview
    # cache, so a cached verdict would lie. Honest fresh/stale/unknown; never
    # fabricated. Guarded so a freshness failure never breaks the overview.
    try:
        from kairos.export.schedule_freshness import schedule_freshness

        body["schedule_freshness"] = schedule_freshness(ROOT)
    except Exception:  # pragma: no cover - defensive, never blocks the overview
        body["schedule_freshness"] = {"status": "unknown", "computed_at": None, "changed": []}
    return body


def _plan_freshness() -> dict[str, Any]:
    """The saved plan's honest fresh/stale/unknown verdict, never cached here."""
    try:
        from kairos.export.schedule_freshness import schedule_freshness

        return schedule_freshness(ROOT)
    except Exception:  # pragma: no cover - defensive, never blocks a read
        return {"status": "unknown", "computed_at": None, "changed": []}


def _model_trained_at() -> Optional[str]:
    """The date the model version in use was trained, and nothing else about it.

    One field crosses the line here: when it was trained. No gate verdict, no
    coverage, no fitted value, so this payload passes the lexicon test that
    every run surface is checked against.
    """
    try:
        from kairos.model.measure import read_coefficients_metadata

        metadata = read_coefficients_metadata(Path(MODELS_DIR) / "tv_break_coefficients.json")
    except Exception:  # pragma: no cover - defensive
        return None
    stamp = metadata.get("computed_at") if isinstance(metadata, dict) else None
    text = str(stamp or "").strip()
    return text or None


@router.get("/api/today", tags=["dashboard"])
def today(request: Request) -> dict[str, Any]:
    """The three answers Today lands on, in one round trip.

    A projection of the same cached body ``/api/overview`` serves, so every
    figure here is the figure there, plus the plan target and the per-day rows
    the money figure resolves to.
    """
    body = _overview_cached(
        _signature([
            OUTPUT_DIR / "weekly_break_schedule.csv",
            DATA_DIR / "reference" / "Programmes.xlsx",
            DATA_DIR / "reference" / "Spots.xlsx",
            DATA_DIR / "Programmes.csv",
            DATA_DIR / "Spots.csv",
            SETTINGS_PATH,
        ]),
        None,
    )
    settings = _load_settings()
    channel = overview_api_today.owned_channel(settings)
    summary = body.get("summary") if isinstance(body.get("summary"), dict) else {}
    window = overview_api_today.window_from_summary(summary)
    rows, boundary = overview_api_today.day_rows(_load_break_schedule(), window)
    money = overview_api_today.money_block(summary, window, rows, boundary, getattr(settings, "timezone", ""))
    target = target_store.payload(channel, window["date_from"] or "", window["date_to"] or "", request)
    freshness = _plan_freshness()
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "channel": channel or None,
        "window": window,
        "money": money,
        "target": target,
        "verdict": target_store.verdict(money["amount_ils"], target_store.target_for(channel, window["date_from"] or "", window["date_to"] or "")),
        "health": overview_api_today.health_block(body, freshness, _model_trained_at(), channel),
        "decisions": overview_api_today.decisions_block(body, channel),
        "plan_run_at": freshness.get("computed_at"),
        "model_trained_at": _model_trained_at(),
    }


@router.get("/api/today/day/{iso_date}", tags=["dashboard"])
def today_day(iso_date: str) -> dict[str, Any]:
    """The plan rows behind one day of the window figure.

    The second level of the drill, fetched when a day is opened rather than
    shipped with the first paint, so the surface that has to answer in five
    seconds carries seven rows and not five hundred.
    """
    summary = _summarize_schedule(_load_break_schedule())
    window = overview_api_today.window_from_summary(summary)
    rows, _ = overview_api_today.day_rows(_load_break_schedule(), window)
    return overview_api_drill.day_detail(
        _load_break_schedule(),
        iso_date,
        [str(row["date"]) for row in rows],
    )
