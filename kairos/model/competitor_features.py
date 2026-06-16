"""Competitor-context features for the retention model, with a strict info boundary.

Stage 3 of the retention model (see docs/model/retention-model.md): a break does
not shed audience in a vacuum. Whether a viewer who leaves during a break comes
back depends on what the rival channels are airing in the same minutes. This
module turns "what the competitors are doing against this slot" into numeric
features the estimator can use, while enforcing the one rule that makes the model
deployable honestly: the information boundary.

The information boundary (load-bearing, see docs/model/retention-model.md Stage 3).
The channel knows its competitors' PROGRAMMES for the coming week, because the
rival EPG is published in advance. It does NOT know its competitors' BREAKS and
ADS until after the fact, from the historical aired-spots logs. Therefore:

  FORWARD features (legitimately available when the plan is made):
    - competitor_strength: the typical rival audience opposite the slot, from the
      rival channels' historical minute-level audience curve. A break opposite a
      strong rival show is a more dangerous place to lose a viewer (there is
      somewhere good to go).
    - competitor_genre_contrast: the fraction of rival channels airing the same
      programme genre as the slot's own programme, from the rival EPG. A substitute
      genre (news against news, drama against drama) raises the switching hazard.

  TRAINING-ONLY feature (known only after the fact, from rival ad logs):
    - competitor_in_break: the fraction of the slot's minutes where a rival was
      itself in a break. When everyone breaks together shedding is nearly free;
      when only we break the viewer has a clean alternative. This may be used to
      TRAIN the model (to estimate or de-confound the forward betas) but must NEVER
      be a forward input to a live decision.

The boundary is enforced in code, not just in docs: :data:`FORWARD_FEATURES` and
:data:`TRAINING_ONLY_FEATURES` are disjoint, and :func:`assert_forward_only` raises
if a training-only key reaches the inference path. Pure pandas and numpy, so it
imports and unit-tests without Meridian.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from kairos.data.classifier import ProgramClassifier
from kairos.data.loaders import CHANNELS
from kairos.model.measure import _baseline_levels, _broadcast_minute, _dayparts_frame
from kairos.model.prepare import identify_breaks

logger = logging.getLogger(__name__)

# The two forward-usable competitor features (from the rival EPG, published ahead).
FORWARD_FEATURES: tuple[str, ...] = ("competitor_strength", "competitor_genre_contrast")
# The training-only feature (from rival aired-spots, known only historically).
TRAINING_ONLY_FEATURES: tuple[str, ...] = ("competitor_in_break",)
# Every competitor feature this module produces.
ALL_FEATURES: tuple[str, ...] = FORWARD_FEATURES + TRAINING_ONLY_FEATURES


class ForwardBoundaryError(ValueError):
    """Raised when a training-only competitor feature reaches the inference path.

    Using a training-only feature (anything derived from rival ad placement) as a
    forward input would leak information that does not exist when the plan is made,
    producing a model that cannot be deployed honestly. This error makes that
    mistake loud instead of silent.
    """


def assert_forward_only(feature_names) -> None:
    """Raise :class:`ForwardBoundaryError` if any name is a training-only feature.

    Call this at the feature-engineering seam of any forward/live decision path to
    enforce the information boundary: only :data:`FORWARD_FEATURES` may flow into a
    plan made before the week airs.
    """
    leaked = [name for name in feature_names if name in TRAINING_ONLY_FEATURES]
    if leaked:
        raise ForwardBoundaryError(
            "training-only competitor feature(s) reached the forward path: "
            f"{sorted(leaked)}. Only {list(FORWARD_FEATURES)} may be used to make a "
            "live decision; rival ad placement is known only historically."
        )


def _rivals(channel: str) -> tuple[str, ...]:
    """The other reference channels that compete against ``channel``."""
    return tuple(c for c in CHANNELS if c != channel)


def _break_minutes(break_start: pd.Timestamp, break_end: pd.Timestamp) -> list[pd.Timestamp]:
    """The whole-minute timestamps spanned by a break (at least the start minute)."""
    start = pd.Timestamp(break_start).floor("min")
    end = pd.Timestamp(break_end).floor("min")
    if end < start:
        end = start
    count = int((end - start).total_seconds() // 60) + 1
    return [start + pd.Timedelta(minutes=k) for k in range(count)]


def _programme_category_lookup(
    programmes: pd.DataFrame, classifier: ProgramClassifier
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]]:
    """Map channel -> air-ordered ``(start, end, classifier_category)`` records.

    The category is the genre the classifier assigns to the programme title, which
    is what ``competitor_genre_contrast`` compares across channels.
    """
    frame = programmes[programmes["start_dt"].notna()].copy()
    lookup: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]] = {}
    for channel, group in frame.groupby("Channel", sort=False):
        records: list[tuple[pd.Timestamp, pd.Timestamp, str]] = []
        for row in group.sort_values("start_dt").itertuples(index=False):
            start = getattr(row, "start_dt")
            end = getattr(row, "end_dt")
            if pd.isna(start) or pd.isna(end):
                continue
            category = classifier.classify(str(getattr(row, "Title"))).category
            records.append((start, end, str(category)))
        lookup[str(channel)] = records
    return lookup


def _category_at(
    lookup: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]],
    channel: str,
    timestamp: pd.Timestamp,
) -> Optional[str]:
    """The genre category the channel is airing at ``timestamp`` (None if none)."""
    for start, end, category in lookup.get(channel, []):
        if start <= timestamp <= end:
            return category
    return None


def _rival_break_intervals(spots: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    """Map channel -> historical ``(break_start, break_end)`` intervals.

    Built from the aired-spots log via the same break detection the model uses, so
    a rival "in break" minute means the rival really aired a commercial break then.
    """
    breaks = identify_breaks(spots)
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if breaks.empty:
        return intervals
    for row in breaks.itertuples(index=False):
        intervals.setdefault(str(getattr(row, "channel")), []).append(
            (pd.Timestamp(getattr(row, "break_start")), pd.Timestamp(getattr(row, "break_end")))
        )
    return intervals


def _strength(
    minutes: list[pd.Timestamp], rivals: tuple[str, ...], baseline: dict[tuple[str, int], float]
) -> float:
    """Mean over break minutes of the summed rival typical audience at that minute.

    Forward-usable: it reads only the rivals' historical average audience curve,
    which is known by time-of-day before the week airs.
    """
    if not minutes:
        return 0.0
    per_minute: list[float] = []
    for stamp in minutes:
        mod = _broadcast_minute(stamp)
        total = sum(baseline.get((rival, mod), 0.0) for rival in rivals)
        per_minute.append(total)
    return float(np.mean(per_minute))


def _genre_contrast(
    minutes: list[pd.Timestamp],
    own_category: Optional[str],
    rivals: tuple[str, ...],
    category_lookup: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, str]]],
) -> float:
    """Fraction of rivals airing the same genre as the slot's own programme.

    Forward-usable: it reads only the rival EPG (published in advance). Returns 0.0
    when the own genre is unknown (no matched programme), so an unknown slot adds no
    spurious contrast.
    """
    if own_category is None or not minutes or not rivals:
        return 0.0
    anchor = minutes[len(minutes) // 2]  # the break's middle minute
    same = 0
    for rival in rivals:
        rival_category = _category_at(category_lookup, rival, anchor)
        if rival_category is not None and rival_category == own_category:
            same += 1
    return float(same) / float(len(rivals))


def _in_break(
    minutes: list[pd.Timestamp],
    rivals: tuple[str, ...],
    intervals: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]],
) -> float:
    """Fraction of break minutes where at least one rival was itself in a break.

    TRAINING-ONLY: derived from rival ad placement, known only historically. Never
    a forward input (see :func:`assert_forward_only`).
    """
    if not minutes:
        return 0.0
    covered = 0
    for stamp in minutes:
        minute_end = stamp + pd.Timedelta(minutes=1)
        hit = False
        for rival in rivals:
            for start, end in intervals.get(rival, ()):  # rival break spans
                if start < minute_end and end > stamp:  # overlaps this minute
                    hit = True
                    break
            if hit:
                break
        covered += 1 if hit else 0
    return float(covered) / float(len(minutes))


def attach_competitor_features(
    breaks: pd.DataFrame,
    programmes: pd.DataFrame,
    dayparts: pd.DataFrame,
    spots: pd.DataFrame,
    classifier: ProgramClassifier,
) -> pd.DataFrame:
    """Add the competitor-context columns to a keyed-breaks frame.

    ``breaks`` is the frame from :func:`kairos.model.prepare.keyed_breaks` (one row
    per break with ``channel``, ``break_start``, ``break_end``). Returns a copy with
    three added columns: the two :data:`FORWARD_FEATURES` and the one
    :data:`TRAINING_ONLY_FEATURES`. The forward columns read only the rival EPG and
    the rivals' historical audience curve; the training-only column reads the rival
    aired-spots log and is tagged as such by its name so the estimator can keep it
    out of the forward path. An empty input returns the same columns, empty.
    """
    out = breaks.copy()
    if out.empty:
        for name in ALL_FEATURES:
            out[name] = pd.Series(dtype=float)
        return out

    baseline = _baseline_levels(_dayparts_frame(dayparts))
    category_lookup = _programme_category_lookup(programmes, classifier)
    rival_intervals = _rival_break_intervals(spots)

    strength: list[float] = []
    contrast: list[float] = []
    in_break: list[float] = []
    for row in out.itertuples(index=False):
        channel = str(getattr(row, "channel"))
        rivals = _rivals(channel)
        minutes = _break_minutes(getattr(row, "break_start"), getattr(row, "break_end"))
        own_category = _category_at(category_lookup, channel, minutes[len(minutes) // 2])
        strength.append(_strength(minutes, rivals, baseline))
        contrast.append(_genre_contrast(minutes, own_category, rivals, category_lookup))
        in_break.append(_in_break(minutes, rivals, rival_intervals))

    out["competitor_strength"] = strength
    out["competitor_genre_contrast"] = contrast
    out["competitor_in_break"] = in_break
    return out
