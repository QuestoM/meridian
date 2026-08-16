"""The owned channel-day scope every preview surface shares.

Frozen helper of :mod:`kairos_api.plan_read`, split out under the 450-line law.
Revenue is only ever projected for the operator's own channel, scoped to one
representative broadcast day so a run stays interactive. The frontier, the saved
optimizer plan, the scenario slider and the named forecasts all resolve their
scope here, so they can never disagree and none of them can be redirected to a
competitor channel.

Moved verbatim from dashboard_api.py with the leading underscore dropped. The old
names keep resolving from :mod:`kairos_api.dashboard_api` and
:mod:`kairos_api.server`, against these same objects.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from kairos.optimize import inventory as inventory_input
from kairos_api.core import (
    DATA_DIR,
    OUTPUT_DIR,
    KairosSettings,
    _load_programmes,
)

logger = logging.getLogger(__name__)


def parse_frontier_scope(scope: str | None, settings: KairosSettings) -> dict[str, str | None]:
    """Parse a ``scope=channel:<id>`` or ``scope=day:<date>`` query into run_scenario kwargs.

    Returns ``{"channel": ..., "day": ...}`` to forward to :func:`run_scenario`.
    No scope (None/empty) returns both None, which preserves the current
    whole-default behaviour (run_scenario auto-detects the first channel-day),
    making the unscoped frontier byte-identical to before. Only the operator's
    OWNED channel is selectable: a channel scope that does not match the configured
    operator_channel is rejected (treated as no scope) so no competitor channel can
    ever be requested. An unrecognised prefix is ignored (honest no-op).
    """
    result: dict[str, str | None] = {"channel": None, "day": None}
    text = str(scope or "").strip()
    if not text or ":" not in text:
        return result
    prefix, _, value = text.partition(":")
    prefix = prefix.strip().lower()
    value = value.strip()
    if not value:
        return result
    if prefix == "channel":
        owned = str(settings.operator_channel or "").strip()
        # When an owned channel is configured, only it is selectable. When it is
        # not configured yet, accept the requested channel (no competitor boundary
        # to enforce against) so the feature is usable in the unconfigured state.
        if owned and value != owned:
            return result
        result["channel"] = value
    elif prefix == "day":
        result["day"] = value
    return result


def frontier_data_signature() -> tuple[tuple[str, int], ...]:
    """A cheap hashable signature of the data files the frontier depends on, so
    the frontier cache invalidates automatically when programmes, spots or the
    planned schedule change on disk."""
    candidates = [
        OUTPUT_DIR / "weekly_break_schedule.csv",
        DATA_DIR / "reference" / "Programmes.xlsx",
        DATA_DIR / "reference" / "Spots.xlsx",
        DATA_DIR / "Programmes.csv",
        DATA_DIR / "Spots.csv",
        inventory_input.DEFAULT_INVENTORY_PATH,
    ]
    sig: list[tuple[str, int]] = []
    for path in candidates:
        try:
            sig.append((str(path), path.stat().st_mtime_ns))
        except OSError:
            continue
    return tuple(sig)


@lru_cache(maxsize=8)
def owned_representative_day(signature: tuple[tuple[str, int], ...], owned: str) -> str | None:
    """The owned channel's busiest broadcast day (most programmes), as YYYY-MM-DD.

    The frontier is traced on this single representative day, not across every
    broadcast day of the channel. A real owned channel spans dozens of broadcast
    days and each day is a full refined optimization per retention-floor step, so a
    whole-channel sweep is many minutes of compute; one full day is interactive.
    The busiest day gives the richest, most distinct curve (a thin day collapses
    the Pareto points on top of each other). Ties break to the latest date for a
    deterministic, recent forecast. Returns ``None`` when the channel has no dated
    programmes.
    """
    del signature  # cache key only
    try:
        programmes = _load_programmes()
    except Exception:
        logger.exception("representative-day load failed")
        return None
    if programmes.empty or "Channel" not in programmes.columns or "start_dt" not in programmes.columns:
        return None
    owned_rows = programmes[programmes["Channel"].astype(str) == owned]
    owned_rows = owned_rows[owned_rows["start_dt"].notna()]
    if owned_rows.empty:
        return None
    days = owned_rows["start_dt"].dt.strftime("%Y-%m-%d")
    counts = days.value_counts()
    busiest = counts[counts == counts.max()].index
    return max(busiest)  # YYYY-MM-DD sorts lexicographically; latest of the busiest


def owned_scope(settings: KairosSettings) -> tuple[str | None, str | None]:
    """The operator's owned channel and its representative broadcast day.

    The single scope selector every operator-facing preview surface (the scenario
    slider, the saved optimizer plan, the one-day optimize default) shares with the
    frontier: revenue is only ever projected for ``settings.operator_channel``,
    scoped to that channel's busiest broadcast day (:func:`owned_representative_day`)
    so a run stays interactive rather than sweeping every owned day. Returns
    ``(None, None)`` when no owned channel is configured, so callers degrade to the
    runner's documented default (or an honest no-channel state) instead of silently
    optimizing a competitor channel-day.
    """
    owned = str(settings.operator_channel or "").strip()
    if not owned:
        return None, None
    return owned, owned_representative_day(frontier_data_signature(), owned)
