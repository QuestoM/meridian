"""Regulatory and policy guardrails for the Kairos optimizer.

These encode the constraints that protect programmes from over-monetisation.
The everyday tension (marketing wants more and longer breaks, programme owners
want fewer and shorter) is resolved here as hard limits, so the optimizer can
chase revenue only inside a safe envelope.

Every check is a pure function that returns a list of Violations, so a candidate
schedule can be rejected or repaired without side effects. Defaults follow the
KairosSettings baseline and should be confirmed against current broadcaster
policy before production use.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class Guardrails:
    """Configurable limits. Seconds are used throughout for precision."""

    max_ad_seconds_per_hour: float = 720.0          # 12 minutes
    max_breaks_per_hour: int = 4
    min_break_spacing_seconds: float = 420.0        # 7 minutes end-to-start
    min_retention_floor: float = 0.72
    max_daily_ad_seconds: float = 9600.0            # 160 minutes
    protected_program_types: tuple[str, ...] = ("News", "Children", "Kids")
    protected_max_ad_seconds_per_hour: float = 480.0  # 8 minutes
    gold_breaks_max_per_day: int = 3


@dataclass(frozen=True)
class Break:
    """A single candidate break, enough to evaluate every guardrail."""

    channel: str
    day: str
    hour: int
    start_seconds: float      # seconds from the start of the broadcast day
    duration_seconds: float
    program_type: str
    retention: float
    is_gold: bool = False


@dataclass(frozen=True)
class Violation:
    code: str
    scope: str
    observed: float
    limit: float
    detail: str


def _is_protected(program_type: str, guardrails: Guardrails) -> bool:
    lowered = {p.lower() for p in guardrails.protected_program_types}
    return str(program_type).lower() in lowered


def check_retention_floor(breaks: Iterable[Break], guardrails: Guardrails) -> list[Violation]:
    out: list[Violation] = []
    for item in breaks:
        if item.retention < guardrails.min_retention_floor:
            out.append(Violation(
                code="retention_floor",
                scope=f"{item.channel}/{item.day} {item.program_type}",
                observed=round(item.retention, 3),
                limit=guardrails.min_retention_floor,
                detail="predicted retention below floor",
            ))
    return out


def check_breaks_per_hour(breaks: Iterable[Break], guardrails: Guardrails) -> list[Violation]:
    counts: dict[tuple[str, str, int], int] = defaultdict(int)
    for item in breaks:
        counts[(item.channel, item.day, item.hour)] += 1
    out: list[Violation] = []
    for (channel, day, hour), count in counts.items():
        if count > guardrails.max_breaks_per_hour:
            out.append(Violation(
                code="breaks_per_hour",
                scope=f"{channel}/{day} {hour:02d}:00",
                observed=count,
                limit=guardrails.max_breaks_per_hour,
                detail="too many breaks in the hour",
            ))
    return out


def check_hourly_ad_load(breaks: Iterable[Break], guardrails: Guardrails) -> list[Violation]:
    seconds: dict[tuple[str, str, int], float] = defaultdict(float)
    protected: dict[tuple[str, str, int], bool] = defaultdict(bool)
    for item in breaks:
        key = (item.channel, item.day, item.hour)
        seconds[key] += item.duration_seconds
        protected[key] = protected[key] or _is_protected(item.program_type, guardrails)
    out: list[Violation] = []
    for key, total in seconds.items():
        limit = (guardrails.protected_max_ad_seconds_per_hour
                 if protected[key] else guardrails.max_ad_seconds_per_hour)
        if total > limit:
            channel, day, hour = key
            out.append(Violation(
                code="hourly_ad_load",
                scope=f"{channel}/{day} {hour:02d}:00",
                observed=round(total, 1),
                limit=limit,
                detail="ad seconds in the hour exceed the limit"
                       + (" (protected programme)" if protected[key] else ""),
            ))
    return out


def check_break_spacing(breaks: Iterable[Break], guardrails: Guardrails) -> list[Violation]:
    by_channel_day: dict[tuple[str, str], list[Break]] = defaultdict(list)
    for item in breaks:
        by_channel_day[(item.channel, item.day)].append(item)
    out: list[Violation] = []
    for (channel, day), items in by_channel_day.items():
        ordered = sorted(items, key=lambda b: b.start_seconds)
        for previous, current in zip(ordered, ordered[1:]):
            gap = current.start_seconds - (previous.start_seconds + previous.duration_seconds)
            if gap < guardrails.min_break_spacing_seconds:
                out.append(Violation(
                    code="break_spacing",
                    scope=f"{channel}/{day}",
                    observed=round(gap, 1),
                    limit=guardrails.min_break_spacing_seconds,
                    detail="breaks are too close together",
                ))
    return out


def check_daily_ad_load(breaks: Iterable[Break], guardrails: Guardrails) -> list[Violation]:
    seconds: dict[tuple[str, str], float] = defaultdict(float)
    for item in breaks:
        seconds[(item.channel, item.day)] += item.duration_seconds
    out: list[Violation] = []
    for (channel, day), total in seconds.items():
        if total > guardrails.max_daily_ad_seconds:
            out.append(Violation(
                code="daily_ad_load",
                scope=f"{channel}/{day}",
                observed=round(total, 1),
                limit=guardrails.max_daily_ad_seconds,
                detail="daily ad seconds exceed the limit",
            ))
    return out


def check_gold_breaks(breaks: Iterable[Break], guardrails: Guardrails) -> list[Violation]:
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for item in breaks:
        if item.is_gold:
            counts[(item.channel, item.day)] += 1
    out: list[Violation] = []
    for (channel, day), count in counts.items():
        if count > guardrails.gold_breaks_max_per_day:
            out.append(Violation(
                code="gold_breaks",
                scope=f"{channel}/{day}",
                observed=count,
                limit=guardrails.gold_breaks_max_per_day,
                detail="too many gold breaks in the day",
            ))
    return out


def evaluate(breaks: Iterable[Break], guardrails: Guardrails) -> list[Violation]:
    """Run every guardrail and return all violations (empty list = compliant)."""
    items = list(breaks)
    violations: list[Violation] = []
    violations.extend(check_retention_floor(items, guardrails))
    violations.extend(check_breaks_per_hour(items, guardrails))
    violations.extend(check_hourly_ad_load(items, guardrails))
    violations.extend(check_break_spacing(items, guardrails))
    violations.extend(check_daily_ad_load(items, guardrails))
    violations.extend(check_gold_breaks(items, guardrails))
    return violations


def is_compliant(breaks: Iterable[Break], guardrails: Guardrails) -> bool:
    return not evaluate(breaks, guardrails)
