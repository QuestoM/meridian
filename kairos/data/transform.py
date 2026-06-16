"""Turn the real programme grid into optimizer inputs.

This is the bridge between the data layer and the optimizer. It takes the loaded
programmes (Title, Channel, Date, Duration in seconds, TVR), classifies each
title for its genre, derives the prime-time pricing class, applies the configured
premiums, and emits one :class:`~kairos.optimize.optimizer.ProgramSegment` per
programme.

Two honesty rules shape it:

  * ``baseline_tvr`` is the real rating from the data, never invented. A row with
    no rating contributes a segment the optimizer simply will not load (it earns
    nothing), rather than a fabricated number.
  * The retention impact, baseline and break defaults come from
    :class:`~kairos.optimize.pricing.OptimizerAssumptions`. They are declared
    assumptions until the Meridian impact model is trained, and they travel with
    the segment so the source of every number is clear.

The pricing class is positional, not a genre: the news programme is ``News``, the
first main show after it is ``PrimeShow1``, the second ``PrimeShow2``, and
everything else ``Other``. That mirrors how the channel actually prices breaks.
"""

from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.model.impact import ImpactModel, RetentionEstimate
from kairos.model.spec import DEFAULT_BREAK_POSITIONS
from kairos.optimize.optimizer import ProgramSegment
from kairos.optimize.pricing import OptimizerAssumptions, PricingModel

logger = logging.getLogger(__name__)

SECONDS_PER_HOUR = 3600
DEFAULT_NEWS_KEYWORDS = ("חדשות",)
DEFAULT_NUM_MAIN_SHOWS = 2
# Break-length buckets matching the trained model's break_length vocabulary
# (kairos.model.spec.DEFAULT_BREAK_LENGTHS). A segment carries one representative
# length, so its impact coefficient is read at this bucket.
_SHORT_MAX_SECONDS = 90.0
_STANDARD_MAX_SECONDS = 180.0


def _seconds_from_midnight(timestamp: pd.Timestamp) -> float:
    return float(timestamp.hour * SECONDS_PER_HOUR + timestamp.minute * 60 + timestamp.second)


def _length_bucket(break_length_seconds: float) -> str:
    """Map a break length in seconds to the model's short/standard/long bucket."""
    if break_length_seconds < _SHORT_MAX_SECONDS:
        return "short"
    if break_length_seconds < _STANDARD_MAX_SECONDS:
        return "standard"
    return "long"


_CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}
_RANK_CONFIDENCE = {0: "low", 1: "medium", 2: "high"}


def _lowest_confidence(labels: Iterable[str]) -> str:
    """The most conservative confidence label among ``labels`` (low < medium < high)."""
    ranks = [_CONFIDENCE_RANK.get(label, 0) for label in labels]
    if not ranks:
        return "low"
    return _RANK_CONFIDENCE[min(ranks)]


def _segment_impact_estimate(
    impact_model: ImpactModel,
    pricing_class: str,
    break_length_seconds: float,
) -> RetentionEstimate:
    """Read one representative retention estimate (point + interval + n + confidence).

    The optimizer applies a single per-break coefficient per segment, but the
    trained model is keyed by break position too. We average the point coefficient
    and the credible bounds across the position buckets at the segment's length,
    take the rounded mean sample size, and the most conservative (lowest) confidence
    label, so a segment never reports more certainty than its weakest constituent
    cell. For the assumption fallback every position returns the same degenerate
    estimate, so the average is that estimate.
    """
    length = _length_bucket(break_length_seconds)
    estimates = [
        impact_model.estimate_for(pricing_class, position, length)
        for position in DEFAULT_BREAK_POSITIONS
    ]
    count = len(estimates)
    return RetentionEstimate(
        coefficient=sum(e.coefficient for e in estimates) / count,
        ci_low=sum(e.ci_low for e in estimates) / count,
        ci_high=sum(e.ci_high for e in estimates) / count,
        n=round(sum(e.n for e in estimates) / count),
        confidence=_lowest_confidence(e.confidence for e in estimates),
    )


def _segment_impact_coefficient(
    impact_model: ImpactModel,
    pricing_class: str,
    break_length_seconds: float,
) -> float:
    """The representative point coefficient for a segment (see _segment_impact_estimate)."""
    return _segment_impact_estimate(impact_model, pricing_class, break_length_seconds).coefficient


def _segment_impact_kwargs(
    impact_model: ImpactModel | None,
    pricing_class: str,
    assumptions: OptimizerAssumptions,
) -> tuple[float, dict[str, object]]:
    """The segment's point coefficient and the optional uncertainty fields to pass on.

    With no impact model the segment carries only the declared assumption point and
    no interval, so the optimizer's risk preference correctly has nothing to bite on.
    With a model the segment also carries the credible interval, sample size and
    confidence so the decision can be uncertainty-aware and the plan can report how
    trustworthy each segment's retention cost was.
    """
    if impact_model is None:
        return assumptions.retention_impact_per_break, {}
    estimate = _segment_impact_estimate(
        impact_model, pricing_class, assumptions.default_break_length_seconds,
    )
    return estimate.coefficient, {
        "impact_ci_low": estimate.ci_low,
        "impact_ci_high": estimate.ci_high,
        "impact_n": estimate.n,
        "impact_confidence": estimate.confidence,
    }


SECONDS_PER_MINUTE = 60
# Fallback programme length when the daily input gives no next programme to
# bound the last one (the daily plan carries no explicit programme duration).
_DEFAULT_PROGRAMME_SECONDS = 3600.0
# The reference daily-input file is a single channel (Reshet 13); the canonical
# daily csv carries no channel column, so the channel is supplied or defaulted.
_DEFAULT_DAILY_CHANNEL = "רשת 13"


def _hhmm_to_seconds(value: object) -> float | None:
    """Parse a 'HH:MM' programme start into seconds from midnight, or None."""
    text = str(value).strip()
    if not text or ":" not in text:
        return None
    try:
        hours, minutes = text.split(":")[:2]
        return float(int(hours) * SECONDS_PER_HOUR + int(minutes) * SECONDS_PER_MINUTE)
    except ValueError:
        return None


def build_segments_from_daily_input(
    daily: pd.DataFrame,
    classifier: ProgramClassifier,
    pricing: PricingModel,
    *,
    assumptions: OptimizerAssumptions | None = None,
    impact_model: ImpactModel | None = None,
    channel: str = _DEFAULT_DAILY_CHANNEL,
    news_keywords: Iterable[str] = DEFAULT_NEWS_KEYWORDS,
    num_main_shows: int = DEFAULT_NUM_MAIN_SHOWS,
) -> list[ProgramSegment]:
    """Build optimizer segments from a real daily optimization input (Wally csv).

    The daily input is the channel's plan for one day: one row per aired spot,
    grouped into programmes by ``program`` and ``program_start``. This collapses
    it to one :class:`~kairos.optimize.optimizer.ProgramSegment` per programme so
    the optimizer decides break placement on the real day rather than on the
    Programmes EPG. ``baseline_tvr`` is the mean planned break rating of the
    programme (a real value from the plan, never invented; a programme with no
    planned rating becomes a zero-value segment). Programme length is inferred
    from the gap to the next programme, with a documented fallback for the last.
    ``channel`` is supplied because the canonical daily csv carries no channel
    column. ``impact_model`` supplies each segment's retention coefficient, as in
    :func:`build_segments_from_programmes`.
    """
    assumptions = assumptions or OptimizerAssumptions()
    frame = daily[daily["program"].notna() & daily["program_start"].notna()].copy()
    if frame.empty:
        return []
    frame["planned_tvr"] = pd.to_numeric(frame.get("planned_tvr"), errors="coerce")
    frame["_start_seconds"] = frame["program_start"].map(_hhmm_to_seconds)
    frame = frame[frame["_start_seconds"].notna()]
    if frame.empty:
        return []

    day = _daily_day(frame)
    grouped = (
        frame.groupby(["program", "program_start"], sort=False)
        .agg(start_seconds=("_start_seconds", "first"), baseline_tvr=("planned_tvr", "mean"))
        .reset_index()
        .sort_values("start_seconds")
        .reset_index(drop=True)
    )

    titles_categories = [
        (str(title), classifier.classify(title).category) for title in grouped["program"]
    ]
    classes = _pricing_classes(
        titles_categories, news_keywords=news_keywords, num_main_shows=num_main_shows,
    )

    starts = grouped["start_seconds"].tolist()
    segments: list[ProgramSegment] = []
    for index, (row, classification, pricing_class) in enumerate(
        zip(grouped.itertuples(index=False), titles_categories, classes)
    ):
        start_seconds = float(getattr(row, "start_seconds"))
        # Length runs to the next programme's start; the last keeps the fallback.
        next_start = starts[index + 1] if index + 1 < len(starts) else None
        duration = (next_start - start_seconds) if next_start and next_start > start_seconds else _DEFAULT_PROGRAMME_SECONDS
        baseline = getattr(row, "baseline_tvr")
        baseline_tvr = 0.0 if pd.isna(baseline) else max(0.0, float(baseline))
        impact_coefficient, impact_fields = _segment_impact_kwargs(
            impact_model, pricing_class, assumptions,
        )
        segments.append(ProgramSegment(
            segment_id=f"{day}|{channel}|{index:03d}",
            channel=channel,
            day=day,
            start_seconds=start_seconds,
            duration_seconds=duration,
            program_type=classification[1],
            baseline_tvr=baseline_tvr,
            cpp=pricing.base_price,
            unit_seconds=1.0,
            impact_coefficient=impact_coefficient,
            **impact_fields,
            retention_baseline=assumptions.retention_baseline,
            premium=pricing.segment_premium(pricing_class=pricing_class, weekday_iso=_daily_weekday(day)),
            max_breaks=assumptions.default_max_breaks,
            break_length_seconds=assumptions.default_break_length_seconds,
        ))
    return segments


def _daily_day(frame: pd.DataFrame) -> str:
    """Read the broadcast day (YYYY-MM-DD) from the daily input date column."""
    if "date" in frame.columns:
        dates = pd.to_datetime(frame["date"], errors="coerce").dropna()
        if not dates.empty:
            return dates.iloc[0].strftime("%Y-%m-%d")
    return "unknown"


def _daily_weekday(day: str) -> int:
    """ISO weekday (1..7) for a YYYY-MM-DD string, defaulting to Sunday."""
    stamp = pd.to_datetime(day, errors="coerce")
    return int(stamp.isoweekday()) if pd.notna(stamp) else 7


def _is_news(title: str, category: str, news_keywords: Iterable[str]) -> bool:
    if category == "News":
        return True
    lowered = str(title)
    return any(keyword in lowered for keyword in news_keywords)


def _pricing_classes(
    titles_categories: list[tuple[str, str]],
    *,
    news_keywords: Iterable[str],
    num_main_shows: int,
) -> list[str]:
    """Assign each programme (in air order) its positional pricing class."""
    news_keywords = tuple(news_keywords)
    classes: list[str] = []
    since_news: int | None = None    # shows counted since the last news programme
    for title, category in titles_categories:
        if _is_news(title, category, news_keywords):
            classes.append("News")
            since_news = 0
        elif since_news is not None and since_news < num_main_shows:
            since_news += 1
            classes.append(f"PrimeShow{since_news}")
        else:
            classes.append("Other")
    return classes


def build_segments_from_programmes(
    programmes: pd.DataFrame,
    classifier: ProgramClassifier,
    pricing: PricingModel,
    *,
    assumptions: OptimizerAssumptions | None = None,
    impact_model: ImpactModel | None = None,
    channel: str | None = None,
    day: str | None = None,
    news_keywords: Iterable[str] = DEFAULT_NEWS_KEYWORDS,
    num_main_shows: int = DEFAULT_NUM_MAIN_SHOWS,
) -> list[ProgramSegment]:
    """Build optimizer segments from a loaded programmes frame.

    ``programmes`` must carry the columns produced by
    :func:`kairos.data.loaders.load_programmes` (``Title``, ``Channel``,
    ``start_dt``, ``Duration``, ``TVR``). ``channel`` filters to one channel and
    ``day`` (``YYYY-MM-DD``) to one broadcast date; both default to the whole
    frame. Rows missing a start time or a positive duration are skipped.

    ``impact_model`` supplies each segment's retention impact coefficient. When
    omitted, the declared :attr:`OptimizerAssumptions.retention_impact_per_break`
    is used (the honest assumption fallback); a trained
    :class:`~kairos.model.impact.PosteriorImpactModel` makes the coefficient
    per-channel and measured. Either way the segment carries one number, so the
    output is identical in shape.
    """
    assumptions = assumptions or OptimizerAssumptions()
    frame = programmes
    if channel is not None:
        frame = frame[frame["Channel"] == channel]
    if "start_dt" not in frame.columns:
        raise ValueError("programmes frame must have a start_dt column (use load_programmes)")
    frame = frame[frame["start_dt"].notna()]
    if day is not None:
        frame = frame[frame["start_dt"].dt.strftime("%Y-%m-%d") == day]
    frame = frame.sort_values("start_dt").reset_index(drop=True)

    titles_categories: list[tuple[str, str]] = []
    classifications = []
    for title in frame["Title"]:
        result = classifier.classify(title)
        classifications.append(result)
        titles_categories.append((str(title), result.category))
    classes = _pricing_classes(
        titles_categories, news_keywords=news_keywords, num_main_shows=num_main_shows,
    )

    segments: list[ProgramSegment] = []
    for index, (row, classification, pricing_class) in enumerate(
        zip(frame.itertuples(index=False), classifications, classes)
    ):
        start = getattr(row, "start_dt")
        duration = float(getattr(row, "Duration", 0.0) or 0.0)
        if duration <= 0:
            continue
        tvr = getattr(row, "TVR", 0.0)
        baseline_tvr = 0.0 if pd.isna(tvr) else max(0.0, float(tvr))
        segment_date = start.strftime("%Y-%m-%d")
        impact_coefficient, impact_fields = _segment_impact_kwargs(
            impact_model, pricing_class, assumptions,
        )
        segments.append(ProgramSegment(
            segment_id=f"{segment_date}|{getattr(row, 'Channel')}|{index:03d}",
            channel=str(getattr(row, "Channel")),
            day=segment_date,
            start_seconds=_seconds_from_midnight(start),
            duration_seconds=duration,
            program_type=classification.category,
            baseline_tvr=baseline_tvr,
            cpp=pricing.base_price,
            unit_seconds=1.0,                       # base price is quoted per second
            impact_coefficient=impact_coefficient,
            **impact_fields,
            retention_baseline=assumptions.retention_baseline,
            premium=pricing.segment_premium(
                pricing_class=pricing_class,
                weekday_iso=start.isoweekday(),
            ),
            max_breaks=assumptions.default_max_breaks,
            break_length_seconds=assumptions.default_break_length_seconds,
        ))
    return segments
