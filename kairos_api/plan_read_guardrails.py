"""The committed plan's break geometry, and the guardrail verdict over it.

Frozen helper of :mod:`kairos_api.plan_read`, split out under the 450-line law.
The geometry is rebuilt from the reference EPG and joined to the saved weekly CSV
by segment_id, so the verdict covers the FULL committed plan rather than the
truncated display board. Three owners read the verdict built on it, so it belongs
to none of them.

Moved verbatim from dashboard_api.py with the leading underscore dropped on the
shared names. The old names keep resolving from :mod:`kairos_api.dashboard_api`
and :mod:`kairos_api.server`, against these same objects, including the single
lru_cache instance.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from functools import lru_cache
from typing import Any

import pandas as pd

from kairos.optimize.guardrails import Break as GuardrailBreak
from kairos.optimize.guardrails import evaluate as evaluate_guardrails
from kairos.optimize.guardrails import (
    CAP_ENFORCED,
    Guardrails,
    cap_state,
    check_day_fraction_ad_load,
    check_window_ad_load,
)

from kairos_api.core import (
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    SETTINGS_PATH,
    KairosSettings,
    _augment_segment_ids,
    _load_break_schedule,
    _load_settings,
    _model_dump,
    _ratio,
    _safe_number,
    _settings_to_guardrails,
    _signature,
    _time_to_seconds,
)

logger = logging.getLogger(__name__)


def infer_hourly_ad_seconds(schedule: pd.DataFrame) -> pd.Series:
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


def infer_hourly_break_counts(schedule: pd.DataFrame) -> pd.Series:
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


@lru_cache(maxsize=2)
def plan_guardrail_items_cached(signature: tuple[tuple[str, int], ...]) -> tuple[GuardrailBreak, ...]:
    """The exact break geometry of the FULL committed plan, for compliance.

    Rebuilds the engine's own segments from the reference EPG, joins them to the
    saved weekly CSV by segment_id, and lays each row's breaks with the same
    _segment_break_objects the optimizer's guardrail check uses, carrying the
    row's true is_gold. This covers every channel-day of the plan (the previous
    source, the break-operations board, truncated to the first 12 programmes per
    channel and synthesized gold flags, so the compliance verdict watched under
    one percent of the plan). Cached on the plan+EPG signature; empty result
    means the geometry could not be joined and callers fall back honestly.
    """
    del signature  # cache key only
    schedule = _load_break_schedule()
    if schedule.empty or "segment_id" not in schedule.columns:
        return ()
    try:
        from kairos.data import ProgramClassifier
        from kairos.data.loaders import load_programmes as _load_prog
        from kairos.data.transform import build_segments_from_programmes
        from kairos.model.impact import load_impact_model
        from kairos.optimize._segment_math import _segment_break_objects
        from kairos.optimize.pricing import OptimizerAssumptions
        from kairos.service import pricing_from_settings

        # segment_id indexes reset per channel-day build, so segments must be
        # rebuilt per (channel, date) pair exactly like the export loop, with
        # the shared resources loaded once.
        programmes = _load_prog()
        settings_map = _model_dump(_load_settings())
        pricing = pricing_from_settings(settings_map)
        assumptions = OptimizerAssumptions()
        impact = load_impact_model(MODELS_DIR / "tv_break_posterior.pkl", assumptions=assumptions)
        classifier = ProgramClassifier.from_yaml()
        pairs = (
            schedule[["channel", "date"]].astype(str).drop_duplicates().itertuples(index=False)
        )
        by_id: dict[str, Any] = {}
        for channel_name, date_str in pairs:
            day_segments = build_segments_from_programmes(
                programmes, classifier, pricing,
                assumptions=assumptions, impact_model=impact,
                channel=channel_name, day=date_str,
            )
            for segment in day_segments:
                by_id[segment.segment_id] = segment
    except Exception:
        logger.exception("plan guardrail geometry unavailable")
        return ()
    frame = _augment_segment_ids(schedule)
    items: list[GuardrailBreak] = []
    joined = 0
    for row in frame.itertuples(index=False):
        segment = by_id.get(str(getattr(row, "segment_id", "")))
        if segment is None:
            continue
        joined += 1
        count = int(_safe_number(getattr(row, "num_breaks", 0)))
        if count <= 0:
            continue
        gold = str(getattr(row, "is_gold", "")).strip().lower() in ("true", "1", "yes")
        items.extend(_segment_break_objects(segment, count, is_gold=gold))
    if joined < len(frame) * 0.99:
        # The EPG no longer matches the saved plan (a re-ingest happened without
        # a recompute). A partial verdict would be dishonest; report nothing and
        # let the caller fall back, with the freshness banner telling the story.
        logger.warning(
            "plan guardrail geometry joined %s of %s rows; falling back", joined, len(frame)
        )
        return ()
    return tuple(items)


def plan_guardrail_items() -> list[GuardrailBreak]:
    return list(plan_guardrail_items_cached(_signature([
        OUTPUT_DIR / "weekly_break_schedule.csv",
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "Programmes.csv",
        SETTINGS_PATH,
    ])))


def guardrail_breaks_from_operations(operations: dict[str, Any]) -> list[GuardrailBreak]:
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


def _optional_cap_checks(items: list[GuardrailBreak], guardrails: Guardrails) -> list[dict[str, Any]]:
    """One row per optional cap, saying plainly whether the rule actually ran.

    An absent cap reports no observation and no limit: there is no window and no
    fraction to measure against, so the honest answer is that the quantity is
    unavailable, never that it is zero and never that the plan is compliant with
    a rule nobody configured. A configured-but-off cap CAN be measured, so it
    reports what it would have found and whether it would have bitten, under a
    status that is not a compliance verdict. Only an enforced cap grades.
    """
    specs = (
        ("window_ad_load", guardrails.airtime_caps.window, check_window_ad_load,
         "Ad minutes in the capped window", "דקות פרסום בחלון המוגבל", "minutes/window"),
        ("day_fraction_ad_load", guardrails.airtime_caps.day_fraction, check_day_fraction_ad_load,
         "Ad time as a share of the broadcast day", "זמן פרסום כשיעור מיממת השידור", "minutes/day"),
    )
    rows: list[dict[str, Any]] = []
    for check_id, cap, check_fn, label_en, label_he, unit in specs:
        state = cap_state(cap)
        row: dict[str, Any] = {
            "id": check_id,
            "violation_code": check_id,
            "label_en": label_en,
            "label_he": label_he,
            "unit": unit,
            "cap_state": state,
            "observed": None,
            "limit": None,
            "violations": 0,
            "status": state,
        }
        if cap is not None:
            # The worst channel-day under this cap, measured whether or not the
            # cap is switched on: a configured cap is a question we can answer.
            if check_id == "window_ad_load":
                scoped = [b for b in items if cap.start_hour <= b.hour < cap.end_hour]
            else:
                scoped = items
            observed = _max_group_sum(
                scoped, lambda item: (item.channel, item.day), lambda item: item.duration_seconds,
            )
            # Run the cap's own check with the switch forced on, so the verdict
            # reported is the one the rule itself would reach rather than a
            # second implementation of the same comparison.
            forced = replace(cap, enabled=True)
            field_name = "window" if check_id == "window_ad_load" else "day_fraction"
            found = check_fn(items, replace(
                guardrails,
                airtime_caps=replace(guardrails.airtime_caps, **{field_name: forced}),
            ))
            row["limit"] = round(cap.max_ad_seconds / 60, 2)
            row["observed"] = round(observed / 60, 2)
            row["violations"] = len(found) if state == CAP_ENFORCED else 0
            row["would_breach"] = bool(found)
            if state == CAP_ENFORCED:
                row["status"] = "at_risk" if found else "compliant"
        rows.append(row)
    return rows


def guardrail_compliance_from_breaks(items: list[GuardrailBreak], settings: KairosSettings) -> dict[str, Any] | None:
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
        # Every check above always runs, so it grades a rule that was applied.
        # The optional caps below say for themselves whether they did.
        check["cap_state"] = CAP_ENFORCED

    # ``checks`` stays the list of rules that ACTUALLY RAN against this plan, so
    # a caller counting it is counting graded rules and nothing else. An
    # optional cap joins it only once enforced; absent and available caps live
    # in ``optional_caps`` alone, where their state is the headline. That is what
    # stops a plan from carrying a badge earned by a rule nobody ran.
    optional = _optional_cap_checks(items, guardrails)
    checks.extend(row for row in optional if row["cap_state"] == CAP_ENFORCED)

    return {
        "checks": checks,
        "optional_caps": optional,
        "cap_states": guardrails.airtime_caps.states(),
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
