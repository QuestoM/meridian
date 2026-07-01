"""Incremental recompute and commit-path override resolution for the weekly CSV.

Support machinery for :mod:`kairos.export.schedule`, split out to keep that
module under its size budget. Two concerns live here:

1. Incremental recompute. ``build_weekly_schedule(only_days=...)`` re-optimizes
   only the named channel-days and merges the fresh rows into the saved CSV.
   The merge is byte-comparable to a full rebuild because it works in the CSV's
   own text space: the saved file is read with ``dtype=str`` so untouched rows
   are preserved verbatim, fresh rows are serialized through the same
   ``DataFrame.to_csv`` a full build uses, and the day blocks are reassembled
   in the full build's ordering (date ascending, then channel; rows inside a
   day keep their build order).

2. Commit-path anchor guard. :func:`resolve_commit_overrides` applies the same
   semantic-anchor check the ``/api/overrides/effect`` preview uses, so an
   override whose stored anchor no longer matches the segment carrying its
   target_id is SKIPPED at commit time instead of silently bending the plan.

THE LAW-9 HAZARD this module is designed against: an incremental run that skips
a channel-day it should have rebuilt presents a stale number as current. Two
defenses: :func:`classify_change` is a conservative allowlist, and every
precondition failure in :func:`incremental_weekly_frame` falls back to a full
rebuild rather than guessing. Callers of ``only_days`` are responsible for
deriving the day list from :func:`classify_change`; a hand-picked list can
leave stale days looking current.
"""

from __future__ import annotations

import io
import logging
from datetime import date as _date
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import pandas as pd

from kairos.optimize.overrides import SEGMENT, STATUS_ACTIVE, OverrideSet

logger = logging.getLogger(__name__)

# The one column whose CSV text depends on whole-frame dtype context: it holds
# Optional[int] values, which pandas renders as "229" when the frame has no
# missing value in the column (int64) but as "229.0" when any row is missing
# (float64); :func:`retention_n_style_conflict` guards the mix explicitly.
_OPTIONAL_INT_COLUMN = "retention_n"

SECONDS_PER_MINUTE = 60


def _clock(start_seconds: float) -> str:
    """A segment start (seconds past midnight) as the HH:MM the CSV stores."""
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


def classify_change(kind: str, payload: dict) -> Union[str, list[tuple[str, str]]]:
    """Map a change event to the channel-days whose schedule rows it can move.

    Conservative allowlist, per governance: the ONLY change that maps to a
    single channel-day is a segment-scope override whose ``target_id`` parses as
    the engine's ``day|channel|index`` segment id; it returns exactly
    ``[(channel, day)]``. EVERYTHING else (settings, pricing, constraints file,
    coefficients, reference data, spot overrides, malformed targets, unknown
    kinds) returns ``'all'``: a wrong 'all' costs a full rebuild, a wrong
    narrow answer shows a stale number as current.
    """
    if str(kind or "").strip().lower() != "override":
        return "all"
    if not isinstance(payload, dict):
        return "all"
    if str(payload.get("scope", "")).strip().lower() != SEGMENT:
        return "all"
    target_id = str(payload.get("target_id", "")).strip()
    parts = target_id.split("|")
    if len(parts) != 3:
        return "all"
    day, channel, index = (part.strip() for part in parts)
    if not channel or not index.isdigit():
        return "all"
    try:
        parsed = _date.fromisoformat(day)
    except ValueError:
        return "all"
    # fromisoformat accepts several ISO spellings; require strict YYYY-MM-DD.
    if parsed.isoformat() != day:
        return "all"
    return [(channel, day)]


def read_schedule_text(
    path: Union[str, Path], expected_columns: Sequence[str]
) -> Optional[pd.DataFrame]:
    """The saved schedule CSV as an all-string frame, or None when unusable.

    ``dtype=str`` with ``keep_default_na=False`` keeps every cell verbatim, so
    rows that are not recomputed round-trip byte-identically. Returns None (the
    caller's full-rebuild signal) when the file is missing, unparseable, or its
    column list differs from the current schema (covers the pre-segment_id layout).
    """
    target = Path(path)
    if not target.exists():
        return None
    try:
        frame = pd.read_csv(target, dtype=str, keep_default_na=False, encoding="utf-8")
    except Exception:
        logger.warning("Could not parse %s; falling back to a full rebuild.", target, exc_info=True)
        return None
    if list(frame.columns) != list(expected_columns):
        return None
    return frame


def frame_text_form(frame: pd.DataFrame) -> pd.DataFrame:
    """A typed frame re-expressed as the exact strings ``to_csv`` writes for it.

    Serializing through ``to_csv`` and re-reading with ``dtype=str`` guarantees
    the fresh rows carry the same text a direct CSV write would produce (float
    shortest-repr, blanks for missing, True/False for bools), so they
    concatenate with the saved file's verbatim rows and stay byte-comparable.
    """
    text = frame.to_csv(index=False)
    return pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False)


def _number_styles(values: Iterable[str]) -> set[str]:
    styles: set[str] = set()
    for value in values:
        text = str(value)
        if not text:
            continue
        styles.add("decimal" if "." in text else "integer")
    return styles


def retention_n_style_conflict(
    existing: pd.DataFrame, fresh_blocks: Iterable[pd.DataFrame]
) -> bool:
    """True when saved and fresh rows render the Optional[int] column differently.

    See :data:`_OPTIONAL_INT_COLUMN`: when the saved CSV and the fresh blocks
    disagree on integer-vs-decimal style, a merged file could not equal a full
    rebuild byte for byte, so the caller must fall back and rebuild.
    """
    styles = _number_styles(existing[_OPTIONAL_INT_COLUMN])
    for block in fresh_blocks:
        styles |= _number_styles(block[_OPTIONAL_INT_COLUMN])
    return len(styles) > 1


def merge_day_blocks(
    pairs: Sequence[tuple[str, str]],
    existing: pd.DataFrame,
    fresh_blocks: dict[tuple[str, str], pd.DataFrame],
    columns: Sequence[str],
) -> pd.DataFrame:
    """Assemble the merged all-string schedule in the full build's ordering.

    Ordering guarantee: a full build emits one contiguous block per (channel,
    day) in ``_channel_days``'s pair order (date ascending, then channel). This
    walks the SAME ``pairs`` list, taking the fresh block for a recomputed pair
    and the saved CSV's rows (in stored order) for every other pair, so the
    merged frame's row order equals a full rebuild's.
    """
    blocks: list[pd.DataFrame] = []
    for channel, day in pairs:
        if (channel, day) in fresh_blocks:
            block = fresh_blocks[(channel, day)]
        else:
            mask = (existing["channel"] == channel) & (existing["date"] == day)
            block = existing.loc[mask]
        if len(block):
            blocks.append(block[list(columns)])
    if not blocks:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(blocks, ignore_index=True)


def incremental_weekly_frame(
    *,
    pairs: Sequence[tuple[str, str]],
    requested: Sequence[tuple[str, str]],
    existing_csv_path: Union[str, Path],
    columns: Sequence[str],
    day_rows: Callable[[str, str], list[dict[str, Any]]],
    has_segments: Callable[[str, str], bool],
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Optional[pd.DataFrame]:
    """Re-optimize only ``requested`` channel-days and merge into the saved CSV.

    ``pairs`` is the full build's channel-day list for the CURRENT source data,
    ``day_rows`` runs one channel-day through the real engine, ``has_segments``
    answers whether a pair would produce any rows, and ``progress_cb(done,
    total)`` fires once per recomputed channel-day.

    Returns the merged all-string frame, or None when any precondition fails,
    which the caller MUST answer with a full rebuild (the honest escape hatch).
    The preconditions exist so a merge can never quietly keep a stale day: the
    saved CSV must exist with the current schema, every requested pair must be
    in the current source, the CSV must not carry pairs the source lost, and a
    source pair absent from both the CSV and the request must be row-less.
    """
    target = Path(existing_csv_path)
    existing = read_schedule_text(target, columns)
    if existing is None:
        logger.warning(
            "Incremental rebuild fell back to a full build: %s is missing, unreadable, "
            "or has a stale schema.", target,
        )
        return None

    requested_set = set(requested)
    pair_set = set(pairs)
    if not requested_set <= pair_set:
        unknown = sorted(requested_set - pair_set)
        logger.warning(
            "Incremental rebuild fell back to a full build: requested channel-days "
            "%s are not in the current source data.", unknown,
        )
        return None
    csv_pairs = set(zip(existing["channel"], existing["date"]))
    if not csv_pairs <= pair_set:
        logger.warning(
            "Incremental rebuild fell back to a full build: the saved CSV carries "
            "channel-days the source data no longer has."
        )
        return None
    for channel, day in pairs:
        if (channel, day) in csv_pairs or (channel, day) in requested_set:
            continue
        # In the CSV's absence a pair is legitimate only when a full build would
        # also write no rows for it; otherwise the CSV predates a source change.
        if has_segments(channel, day):
            logger.warning(
                "Incremental rebuild fell back to a full build: the saved CSV has no "
                "rows for %s %s but the source produces segments for it.", channel, day,
            )
            return None

    fresh_blocks: dict[tuple[str, str], pd.DataFrame] = {}
    total = len(requested_set)
    done = 0
    for channel, day in pairs:  # pairs order keeps the recompute deterministic
        if (channel, day) not in requested_set:
            continue
        rows = day_rows(channel, day)
        fresh_blocks[(channel, day)] = frame_text_form(
            pd.DataFrame(rows, columns=list(columns))
        )
        done += 1
        if progress_cb is not None:
            progress_cb(done, total)

    if retention_n_style_conflict(existing, fresh_blocks.values()):
        logger.warning(
            "Incremental rebuild fell back to a full build: the saved CSV and the "
            "fresh rows render %s in different numeric styles.", _OPTIONAL_INT_COLUMN,
        )
        return None
    return merge_day_blocks(pairs, existing, fresh_blocks, columns)


def resolve_commit_overrides(
    overrides: Optional[OverrideSet],
    segments: Sequence[Any],
    *,
    seen_segment_ids: set[str],
    skipped: list[dict[str, Any]],
) -> Optional[OverrideSet]:
    """Anchor-guard the overrides for one channel-day at COMMIT time.

    The same semantics as the ``/api/overrides/effect`` preview
    (kairos_api/overrides.py): each current segment's semantic anchor is the
    trio ``(day, start clock, program type)``, and an active segment override
    with a stored anchor binds only when that anchor still matches the segment
    now carrying its target_id. A mismatch means a re-ingest reordered the
    build-order ids, so the override is SKIPPED (reported into ``skipped``,
    never applied) instead of silently moving revenue on the wrong break; a
    blank-anchor (legacy) override still binds by target_id alone. Segment
    overrides targeting other channel-days are dropped from the returned set
    for THIS day; they never matched this day's segment ids, so the optimizer
    outcome is unchanged, and their own day performs their check.
    ``seen_segment_ids`` accumulates this day's ids so the caller can report,
    after a full run, anchored overrides that matched no day at all.
    """
    if overrides is None or not overrides.overrides:
        return overrides
    anchors = {
        segment.segment_id: (
            str(segment.day),
            _clock(segment.start_seconds),
            str(segment.program_type),
        )
        for segment in segments
    }
    seen_segment_ids.update(anchors)
    relevant = OverrideSet(overrides=[
        override for override in overrides.overrides
        if override.scope != SEGMENT or override.target_id in anchors
    ])
    resolved, stale = relevant.resolve_against_segments(anchors)
    skipped.extend(stale)
    return resolved


def unmatched_anchor_reports(
    overrides: Optional[OverrideSet], seen_segment_ids: set[str]
) -> list[dict[str, Any]]:
    """Skip reports for anchored overrides that matched no channel-day at all.

    Only meaningful after a FULL run, when ``seen_segment_ids`` covers every
    optimized segment: an active, anchored segment override whose target_id
    appeared nowhere points at nothing in the current schedule, so it was not
    applied and is reported in the same shape
    :meth:`OverrideSet.resolve_against_segments` uses. Blank-anchor legacy
    overrides are not reported; they bind by target_id wherever it exists.
    """
    if overrides is None:
        return []
    reports: list[dict[str, Any]] = []
    for override in overrides.overrides:
        if override.scope != SEGMENT or not override.is_valid():
            continue
        if override.status != STATUS_ACTIVE:
            continue
        anchored = bool(
            str(override.anchor_date or "").strip()
            or str(override.anchor_start or "").strip()
            or str(override.anchor_title or "").strip()
        )
        if not anchored or override.target_id in seen_segment_ids:
            continue
        reports.append({
            "override_id": override.override_id,
            "segment_id": override.target_id,
            "kind": override.kind,
            "reason": "anchor target segment_id is not in the current schedule",
            "expected": {
                "date": str(override.anchor_date or "").strip(),
                "start_clock": str(override.anchor_start or "").strip(),
                "program": str(override.anchor_title or "").strip(),
            },
            "found": None,
        })
    return reports


def rows_from_result(segments: Sequence[Any], result: Any) -> list[dict[str, Any]]:
    """One schedule row per programme segment, from one channel-day's plan.

    The single row construction the full and incremental paths share. Nothing
    is fabricated: a segment left without breaks earns zero and keeps its full
    baseline retention; risk CI fields stay blank without a measured CI.
    """
    rows: list[dict[str, Any]] = []
    plans = {plan.segment_id: plan for plan in result.segments}
    for segment in segments:
        plan = plans.get(segment.segment_id)
        num_breaks = plan.num_breaks if plan else 0
        # A 0-break segment keeps its baseline retention and earns nothing.
        retention = plan.retention if plan else segment.retention_baseline
        revenue = plan.revenue if plan else 0.0

        # Risk-adjusted retention fields: surface the per-segment uncertainty
        # the optimizer used so the weekly CSV carries the full risk decision.
        # ``plan.retention`` is already the risk-adjusted value (computed with
        # the conservative coefficient when risk_lambda > 0 and a CI exists);
        # the CI columns translate the per-break coefficient interval into
        # retention bounds at ``num_breaks`` breaks. All four auxiliary fields
        # are None (blank in CSV) when the segment has no measured CI.
        if plan is not None:
            ret_ci_low: Optional[float] = None
            ret_ci_high: Optional[float] = None
            if (
                plan.retention_cost_ci_low is not None
                and plan.retention_cost_ci_high is not None
                and num_breaks > 0
            ):
                from kairos.optimize.objective import predicted_retention as _pred_ret

                ret_ci_low = round(
                    _pred_ret(
                        segment.retention_baseline,
                        plan.retention_cost_ci_low,
                        num_breaks,
                    ),
                    4,
                )
                ret_ci_high = round(
                    _pred_ret(
                        segment.retention_baseline,
                        plan.retention_cost_ci_high,
                        num_breaks,
                    ),
                    4,
                )
            retention_n: Optional[int] = plan.retention_cost_n if plan.retention_cost_n else None
            retention_confidence: Optional[str] = plan.retention_confidence or None
        else:
            ret_ci_low = None
            ret_ci_high = None
            retention_n = None
            retention_confidence = None

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
                "retention_used": round(retention, 4),
                "retention_ci_low": ret_ci_low,
                "retention_ci_high": ret_ci_high,
                "retention_n": retention_n,
                "retention_confidence": retention_confidence,
                "segment_id": segment.segment_id,
                # Gold is a per-break flag on the plan's placements; a segment
                # is gold when the optimizer emitted a gold break for it. False
                # (no gold break, honest) for a 0-break or non-gold segment.
                "is_gold": bool(plan is not None and any(p.is_gold for p in plan.placements)),
            }
        )

    return rows
