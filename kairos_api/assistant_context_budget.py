"""The character budget on Kai's grounding context, and how a cut is disclosed.

Split out of kairos_api.assistant_context so that module stays under the
file-size cap. Nothing changed on the move: kairos_api.assistant_context
imports every name below and re-exports it, so ``assistant_context.enforce_budget``
and ``assistant_context.BUDGET_ENV`` still resolve for every caller and every
test that sets the environment variable through them.

The rule this file enforces is one sentence: the serialized context that ships
to the model is never larger than the budget, and any row that had to go is
counted in the payload rather than dropped silently.
"""

from __future__ import annotations

import json
import os
from typing import Any

DEFAULT_CONTEXT_BUDGET = 60000
BUDGET_ENV = "KAIROS_ASSISTANT_CONTEXT_BUDGET"
DAY_DETAIL_PREFIX = "day_detail"


def _context_budget() -> int:
    raw = os.environ.get(BUDGET_ENV, "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            value = 0
        if value > 0:
            return value
    return DEFAULT_CONTEXT_BUDGET


def _serialized_size(context: dict[str, Any]) -> int:
    return len(json.dumps(context, ensure_ascii=False, separators=(",", ":"), default=str))


def enforce_budget(context: dict[str, Any]) -> None:
    """Drop day-detail rows, lowest revenue first, until the context fits.

    Rows are already ordered by revenue descending, so popping from the end
    always removes the least valuable row. The section with the most remaining
    rows gives one up first (ties resolve to the latest date), keeping
    multi-day answers balanced and the whole procedure deterministic. Base
    sections and the per-day table are never trimmed; matched full rows go last
    because they are the data the question asked for most specifically. Any cut
    anywhere, including the per-day row caps applied at build time, raises the
    top-level ``day_detail_truncated`` flag the system prompt tells the model
    to disclose.
    """
    day_keys = [key for key in context if key.startswith(f"{DAY_DETAIL_PREFIX} ")]
    if day_keys:
        budget = _context_budget()
        # The disclosure flag itself costs bytes. Raise it before measuring
        # whenever any section is already truncated (build-time row caps), and
        # again on every trim, so the size the loop checks is the size that
        # actually ships; otherwise adding the flag after the loop could push
        # the final payload past the budget it just enforced.
        if any(bool(context[key].get("truncated")) for key in day_keys):
            context["day_detail_truncated"] = True
        while _serialized_size(context) > budget:
            # Floor of one row per section: an answer with zero data rows is
            # worthless, so the top-revenue row always survives even when the
            # base sections leave almost no budget for day detail.
            pools = [
                (len(context[key]["segments"]), key, "segments")
                for key in day_keys
                if len(context[key].get("segments") or []) > 1
            ] or [
                (len(context[key]["matched_full_rows"]), key, "matched_full_rows")
                for key in day_keys
                if len(context[key].get("matched_full_rows") or []) > 1
            ]
            if not pools:
                break
            _, key, field = max(pools)
            section = context[key]
            section[field].pop()
            section["rows_omitted"] = int(section.get("rows_omitted", 0)) + 1
            section["truncated"] = True
            context["day_detail_truncated"] = True
        _trim_recommendations(context, budget)
    if any(bool(context[key].get("truncated")) for key in day_keys):
        context["day_detail_truncated"] = True


def _trim_recommendations(context: dict[str, Any], budget: int) -> None:
    """Last resort when the day-detail floor still leaves the payload oversize.

    The day-detail loop keeps one row per section on purpose, so a budget the
    base sections alone nearly fill cannot be reached by that loop at all, and
    before this existed the contract "the shipped size respects the budget" held
    only by a dozen characters of slack. The recommendation list is the one base
    section that is a ranked list rather than a single fact, so it is the honest
    thing to shorten: rows go lowest-ranked first, at least one always survives,
    and ``recommendations_omitted`` records how many went, so a cut is never
    silent. Nothing else is ever trimmed.
    """
    rows = context.get("recommendations")
    if not isinstance(rows, list) or len(rows) <= 1:
        return
    while len(rows) > 1 and _serialized_size(context) > budget:
        rows.pop()
        context["recommendations_omitted"] = int(context.get("recommendations_omitted", 0)) + 1
