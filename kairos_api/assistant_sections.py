"""The grounding context Kai answers from, and the cache that makes it fast.

Split out of kairos_api.assistant so that module stays under the file-size cap.

Every section reuses a real dashboard builder, so Kai reads exactly what the
operator's own pages render and nothing else. A failing section is omitted and
listed in sources with an absent marker, never substituted and never faked.

Two things beyond the move:

**The competitor boundary reaches the counts section.** Measured before this
change, on the reference data with operator_channel = רשת 13: the counts section
reported 8,704 segments and 9,026 breaks across all four channels, while the
overview section beside it reported the operator's own 2,391 breaks. Two
different break counts in one context, one of them built from rival schedules
and neither carrying its scope. It is now scoped through
kairos_api.channel_scope with the scope note attached, which is the duty section
8.3 of the rebuild specification gives this piece.

**The base sections are cached on a fingerprint.** Measured in a fresh process:
composing them costs 11.13 s cold and 0.034 s warm, a 327x gap that the operator
pays as dead time before the first token. The cache is the frozen
kairos_api.read_cache, keyed on the signatures of every file the sections read
plus the saved settings, so a changed input is a miss rather than a stale
answer. Only the six base sections are cached: the day, keyword and location
sections depend on the question or the caller and are rebuilt every ask.
"""

from __future__ import annotations

import copy
from typing import Any, Callable

from kairos_api import (
    assistant_context,
    assistant_keywords,
    assistant_page_context,
    channel_scope,
    read_cache,
)

CACHE_NAMESPACE = "assistant_context_base"

# Why a count carries no owned-channel scope, and the code each reason is said
# by. The English is the record an API reader gets; the code is what the run
# trace says the reason from, because English prose printed raw on a Hebrew
# screen is a defect. Both strings are named rather than repeated, so a reason
# that is reworded cannot silently stop matching its code.
EMPTY_PLAN_REASON = "the saved weekly plan is empty"
SCOPE_REASON_CODES = {
    channel_scope.NO_OPERATOR_CHANNEL_REASON: "no_operator_channel",
    EMPTY_PLAN_REASON: "empty_plan",
}


def _server() -> Any:
    """Lazy handle to server.py helpers (avoids an import cycle at module load)."""
    from kairos_api import server

    return server


def _overview_body() -> dict[str, Any]:
    server = _server()
    return server._overview_cached(
        server._signature(
            [
                server.OUTPUT_DIR / "weekly_break_schedule.csv",
                server.DATA_DIR / "reference" / "Programmes.xlsx",
                server.DATA_DIR / "reference" / "Spots.xlsx",
                server.DATA_DIR / "Programmes.csv",
                server.DATA_DIR / "Spots.csv",
                server.SETTINGS_PATH,
            ]
        ),
        None,
    )


def _section_overview_summary() -> dict[str, Any]:
    return dict(_overview_body()["summary"])


def _section_schedule_freshness() -> dict[str, Any]:
    from kairos.export.schedule_freshness import schedule_freshness

    return dict(schedule_freshness(_server().ROOT))


def _section_yield_totals() -> dict[str, Any]:
    # The operator-channel-scoped payload the dashboard route serves, so the
    # assistant and the yield page quote the same money, never a whole-network
    # figure relabeled as ours. scope_channel and n_channels_total disclose the
    # scope; the retention-cost band keys ride along when the net is available.
    from kairos_api.insights_api import scoped_yield_payload

    payload = scoped_yield_payload()
    keys = (
        "available",
        "reason",
        "currency",
        "scope_channel",
        "n_channels_total",
        "totals",
        "revenue_net_available",
        "revenue_net_ils",
        "retention_cost_ils",
        "retention_cost_low",
        "retention_cost_high",
        "revenue_ils",
        "revenue_net_reason",
    )
    return {key: payload[key] for key in keys if key in payload}


def _section_recommendations() -> list[dict[str, Any]]:
    server = _server()
    rows = server._build_recommendations(server._load_break_schedule())
    return [
        {
            "title": row.get("title"),
            "title_he": row.get("title_he"),
            "severity": row.get("risk"),
            "segment": row.get("segment_id"),
            "program_type": row.get("program_type"),
            "impact_ils": row.get("impact"),
            "retention_pct": row.get("retention"),
        }
        for row in rows[:5]
    ]


def _section_settings() -> dict[str, Any]:
    settings = _server()._load_settings()
    return {
        "revenue_weight": settings.revenue_weight,
        "min_retention_floor": settings.min_retention_floor,
        "objective_mode": settings.objective_mode,
        "operator_channel": settings.operator_channel or None,
    }


def _section_counts() -> dict[str, Any]:
    """Segment and break counts, scoped to the operator's own channel.

    The competitor boundary applies here exactly as it applies to a plan
    projection: rival rows never reach the payload, and what they contributed is
    disclosed as an unnamed count so the total is legible rather than silently
    smaller than the file.
    """
    import pandas as pd

    from kairos_api import channel_scope

    frame = _server()._load_break_schedule()
    if frame.empty:
        return {"segments": 0, "breaks": 0, "scoped": False,
                "reason": EMPTY_PLAN_REASON}
    scoped, note = channel_scope.scope_frame(frame)
    if "segment_id" in scoped.columns:
        segments = int(scoped["segment_id"].nunique())
    else:
        segments = int(len(scoped))
    if "num_breaks" in scoped.columns:
        breaks = int(pd.to_numeric(scoped["num_breaks"], errors="coerce").fillna(1).sum())
    else:
        breaks = int(len(scoped))
    # scope_channel is the whole disclosure the model needs and the whole
    # disclosure it may have. How many rival rows were dropped is a fact about
    # rivals, and the fewer of those in the context the better: the count is
    # asserted at the channel_scope seam by the piece's own test instead. A null
    # scope_channel with a reason is the honest unscoped state.
    payload: dict[str, Any] = {"segments": segments, "breaks": breaks,
                               "scope_channel": note.get("scope_channel")}
    if not note.get("scoped") and note.get("reason"):
        payload["reason"] = note["reason"]
    return payload


_SECTIONS: tuple[tuple[str, Callable[[], Any]], ...] = (
    ("overview_summary", _section_overview_summary),
    ("schedule_freshness", _section_schedule_freshness),
    ("yield_totals", _section_yield_totals),
    ("recommendations", _section_recommendations),
    ("settings", _section_settings),
    ("counts", _section_counts),
)


def _fingerprint() -> Any:
    """Every input the six base sections read, as one comparable summary.

    File signatures cover the plan, the reference workbooks and the settings
    file; the settings body itself is folded in because a settings write can
    land inside the same second and leave the file the same size, which a
    (path, mtime, size) signature cannot see. The builders themselves ride in by
    identity, so a code change is a miss rather than a stale answer.
    """
    server = _server()
    paths = [
        server.OUTPUT_DIR / "weekly_break_schedule.csv",
        server.DATA_DIR / "reference" / "Programmes.xlsx",
        server.DATA_DIR / "reference" / "Spots.xlsx",
        server.DATA_DIR / "Programmes.csv",
        server.DATA_DIR / "Spots.csv",
        server.SETTINGS_PATH,
    ]
    try:
        settings_digest = repr(server._load_settings())
    except Exception:  # noqa: BLE001 - an unreadable settings file is its own fingerprint
        settings_digest = "settings-unreadable"
    # The seams by identity, this module's and server.py's alike, so a replaced
    # builder is a miss. Suites monkeypatch these, and a cache that survived one
    # would serve the real plan's figures into a test that asked for another.
    seams = tuple(build for _, build in _SECTIONS) + tuple(
        getattr(server, name, None)
        for name in ("_overview_cached", "_build_recommendations", "_load_break_schedule",
                     "_load_settings", "_signature")
    )
    return (read_cache.file_signatures(paths), settings_digest, seams)


def _build_base() -> tuple[dict[str, Any], list[str]]:
    context: dict[str, Any] = {}
    sources: list[str] = []
    for name, build in _SECTIONS:
        try:
            context[name] = build()
            sources.append(name)
        except Exception:  # noqa: BLE001 - an absent section is named, never invented
            sources.append(f"{name} (absent)")
    return context, sources


def base_context(use_cache: bool = True) -> tuple[dict[str, Any], list[str]]:
    """The six base sections plus their source list, deep-copied per caller.

    The cache holds one immutable pair per fingerprint; every caller gets its own
    copy because the composer mutates the context in place afterwards.
    """
    if not use_cache:
        return _build_base()
    cached = read_cache.cached(CACHE_NAMESPACE, "base", _fingerprint(), _build_base)
    return copy.deepcopy(cached[0]), list(cached[1])


def warm() -> dict[str, Any]:
    """Build the base context now so the next ask does not pay for it.

    Returns the honest outcome: how many sections are present, which are absent,
    and whether this call was already served from the cache.
    """
    before = read_cache.stats(CACHE_NAMESPACE)
    _, sources = base_context()
    after = read_cache.stats(CACHE_NAMESPACE)
    hit = int(after.get("hits", 0)) > int(before.get("hits", 0))
    return {
        "sections": len([name for name in sources if not name.endswith("(absent)")]),
        "absent": [name for name in sources if name.endswith("(absent)")],
        "already_warm": hit,
    }


def compose_context(question: str, page_context: "dict[str, Any] | None" = None,
                    user: "str | None" = None) -> tuple[dict[str, Any], list[str]]:
    """Build the grounding context from the real payload builders.

    The base sections come from the fingerprinted cache. assistant_context then
    adds the always-on per-day owned-channel table and any day_detail sections
    the question's dates resolve to, assistant_keywords attaches the compact
    keyword-matched sections (walled by the acting account, so a channel account
    never grounds on model internals), assistant_page_context attaches the
    current_location section when the dock sent a valid page context, and the
    serialized character budget is enforced last.
    """
    context, sources = base_context()
    assistant_context.extend_with_day_grounding(context, sources, question)
    assistant_keywords.extend_with_keyword_sections(context, sources, question, user)
    assistant_page_context.extend_with_current_location(context, sources, page_context)
    assistant_context.enforce_budget(context)
    return context, sources
