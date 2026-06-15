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
from kairos.model.impact import ImpactModel
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


def _segment_impact_coefficient(
    impact_model: ImpactModel,
    pricing_class: str,
    break_length_seconds: float,
) -> float:
    """Read one representative retention coefficient for a segment.

    The optimizer applies a single per-break coefficient per segment, but the
    trained model is keyed by break position too. We average the model's
    coefficient across the position buckets at the segment's length, which is the
    most faithful single-number summary. For the assumption fallback every
    position returns the same declared number, so the average is that number.
    """
    length = _length_bucket(break_length_seconds)
    coefficients = [
        impact_model.coefficient_for(pricing_class, position, length)
        for position in DEFAULT_BREAK_POSITIONS
    ]
    return sum(coefficients) / len(coefficients)


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
        if impact_model is None:
            impact_coefficient = assumptions.retention_impact_per_break
        else:
            impact_coefficient = _segment_impact_coefficient(
                impact_model, pricing_class, assumptions.default_break_length_seconds,
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
            retention_baseline=assumptions.retention_baseline,
            premium=pricing.segment_premium(
                pricing_class=pricing_class,
                weekday_iso=start.isoweekday(),
            ),
            max_breaks=assumptions.default_max_breaks,
            break_length_seconds=assumptions.default_break_length_seconds,
        ))
    return segments
