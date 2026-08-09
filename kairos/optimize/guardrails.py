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
from functools import lru_cache
from typing import Any, Iterable, Mapping, Optional

# The three states a cap can be in, and the only three this module reports. A
# cap that is not enforced must never be presented as though it were, so "did
# not run" is a first-class answer rather than a silent pass.
CAP_ABSENT = "absent"        # not configured at all: the rule does not exist here
CAP_AVAILABLE = "available"  # configured and switched off: expressible, not applied
CAP_ENFORCED = "enforced"    # configured and switched on: applied to this plan

SECONDS_PER_CALENDAR_DAY = 86400.0


@dataclass(frozen=True)
class WindowAdCap:
    """A cap on total ad seconds inside a wall-clock window of one channel-day.

    ``start_hour`` and ``end_hour`` are wall-clock hours on the broadcast day,
    half-open as ``[start_hour, end_hour)``, so an end of 24 means "up to
    midnight". A break belongs to the window when ITS OWN start falls inside it,
    which is the same attribution the hourly cap uses (see
    :func:`check_hourly_ad_load`): a break is counted whole, where it starts.

    Nothing here is a regulatory figure. The hours and the limit are supplied by
    whoever configures the cap; this type only knows how to apply them.
    """

    start_hour: int
    end_hour: int
    max_ad_seconds: float
    enabled: bool = False


@dataclass(frozen=True)
class DayFractionAdCap:
    """A cap on a channel-day's ad seconds as a fraction of the broadcast day.

    ``day_seconds`` is the denominator the fraction is taken against and defaults
    to a full calendar day. It is an explicit field because "the broadcast day"
    is an assumption, not a constant: an operator that does not broadcast around
    the clock has a shorter one, and the honest thing is to make the caller say
    so rather than to bake a number in.
    """

    max_fraction: float
    day_seconds: float = SECONDS_PER_CALENDAR_DAY
    enabled: bool = False

    @property
    def max_ad_seconds(self) -> float:
        return self.max_fraction * self.day_seconds


@dataclass(frozen=True)
class AirtimeCaps:
    """Optional caps beyond the hourly guardrail, each absent unless configured.

    Both fields default to ``None``, which means the cap DOES NOT EXIST rather
    than that its limit is zero. That distinction is the whole point: an absent
    cap contributes no violations and grades as :data:`CAP_ABSENT`, so a plan
    built without one never carries a verdict it did not earn.
    """

    window: Optional[WindowAdCap] = None
    day_fraction: Optional[DayFractionAdCap] = None

    def states(self) -> dict[str, str]:
        """Which state each cap is in, for the compliance read to disclose."""
        return {
            "window_ad_load": cap_state(self.window),
            "day_fraction_ad_load": cap_state(self.day_fraction),
        }


def cap_state(cap: Optional[WindowAdCap | DayFractionAdCap]) -> str:
    """Absent, available or enforced. Never a bare boolean."""
    if cap is None:
        return CAP_ABSENT
    return CAP_ENFORCED if cap.enabled else CAP_AVAILABLE


def airtime_caps_from_mapping(settings: Optional[Mapping[str, Any]]) -> AirtimeCaps:
    """Build the optional caps from a settings mapping; missing keys stay absent.

    The single translation both settings paths use, so the API model and the
    plain-dict service path cannot drift into disagreeing about what a stored
    cap means. A key that is absent, ``None`` or empty leaves its cap absent.
    """
    if not settings:
        return AirtimeCaps()
    window_raw = settings.get("window_ad_cap")
    day_raw = settings.get("day_fraction_ad_cap")
    window = None
    if window_raw:
        window = WindowAdCap(
            start_hour=int(window_raw["start_hour"]),
            end_hour=int(window_raw["end_hour"]),
            max_ad_seconds=float(window_raw["max_ad_minutes"]) * 60.0,
            enabled=bool(window_raw.get("enabled", False)),
        )
    day_fraction = None
    if day_raw:
        day_fraction = DayFractionAdCap(
            max_fraction=float(day_raw["max_fraction_of_day"]),
            day_seconds=float(day_raw.get("day_seconds", SECONDS_PER_CALENDAR_DAY)),
            enabled=bool(day_raw.get("enabled", False)),
        )
    return AirtimeCaps(window=window, day_fraction=day_fraction)


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
    # Optional caps, absent by default. An engine built without them behaves
    # exactly as it did before they existed, which is what makes "off by
    # default" a structural property rather than a promise.
    airtime_caps: AirtimeCaps = field(default_factory=AirtimeCaps)


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


@lru_cache(maxsize=None)
def _lowered_protected(protected_program_types: tuple[str, ...]) -> frozenset[str]:
    """Lowercased protected-type set, memoised once per unique type tuple.

    Guardrails is a frozen dataclass, so its protected_program_types tuple is
    hashable and stable. Caching on that tuple keeps the matching semantics
    identical while avoiding a fresh set build on every _is_protected call.
    """
    return frozenset(p.lower() for p in protected_program_types)


def _is_protected(program_type: str, guardrails: Guardrails) -> bool:
    lowered = _lowered_protected(guardrails.protected_program_types)
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


def check_window_ad_load(breaks: Iterable[Break], guardrails: Guardrails) -> list[Violation]:
    """Total ad seconds inside the configured wall-clock window of a channel-day.

    Returns no violations when the cap is absent or switched off, so this is a
    genuine no-op until an operator asks for it. A break counts toward the window
    when its own start falls inside ``[start_hour, end_hour)``; a break that
    starts inside and runs past the end is still counted whole, exactly as the
    hourly cap counts it.
    """
    cap = guardrails.airtime_caps.window
    if cap is None or not cap.enabled:
        return []
    seconds: dict[tuple[str, str], float] = defaultdict(float)
    for item in breaks:
        if cap.start_hour <= item.hour < cap.end_hour:
            seconds[(item.channel, item.day)] += item.duration_seconds
    out: list[Violation] = []
    for (channel, day), total in seconds.items():
        if total > cap.max_ad_seconds:
            out.append(Violation(
                code="window_ad_load",
                scope=f"{channel}/{day} {cap.start_hour:02d}:00-{cap.end_hour:02d}:00",
                observed=round(total, 1),
                limit=cap.max_ad_seconds,
                detail="ad seconds in the window exceed the limit",
            ))
    return out


def check_day_fraction_ad_load(breaks: Iterable[Break], guardrails: Guardrails) -> list[Violation]:
    """A channel-day's ad seconds against a fraction of the broadcast day.

    Returns no violations when the cap is absent or switched off. This is a
    separate rule from :func:`check_daily_ad_load`, which caps the same total
    against an absolute duration; both can be configured and they do not replace
    one another.
    """
    cap = guardrails.airtime_caps.day_fraction
    if cap is None or not cap.enabled:
        return []
    limit = cap.max_ad_seconds
    seconds: dict[tuple[str, str], float] = defaultdict(float)
    for item in breaks:
        seconds[(item.channel, item.day)] += item.duration_seconds
    out: list[Violation] = []
    for (channel, day), total in seconds.items():
        if total > limit:
            out.append(Violation(
                code="day_fraction_ad_load",
                scope=f"{channel}/{day}",
                observed=round(total, 1),
                limit=round(limit, 1),
                detail=f"ad seconds exceed {cap.max_fraction:.4g} of the broadcast day",
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
    violations.extend(check_window_ad_load(items, guardrails))
    violations.extend(check_day_fraction_ad_load(items, guardrails))
    violations.extend(check_gold_breaks(items, guardrails))
    return violations


def is_compliant(breaks: Iterable[Break], guardrails: Guardrails) -> bool:
    """Return True only when no guardrail is violated.

    Logically identical to (evaluate(breaks, guardrails) == []) but stops at the
    first rule that reports a violation, in the same rule order evaluate() uses,
    so the common compliant-checking path does not build every violation list.
    """
    items = list(breaks)
    checks = (
        check_retention_floor,
        check_breaks_per_hour,
        check_hourly_ad_load,
        check_break_spacing,
        check_daily_ad_load,
        check_window_ad_load,
        check_day_fraction_ad_load,
        check_gold_breaks,
    )
    for check in checks:
        if check(items, guardrails):
            return False
    return True
