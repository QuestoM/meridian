"""The engine seam: one channel-day as plan rows, and one adopted day published.

A day proposal has to hold real plan rows, not a summary, because adopting one
publishes them into the plan of record. So the rows a proposal freezes are built
by :func:`kairos.export.incremental.rows_from_result` - the ONE row construction
the full export and the incremental export already share. Nothing here is a
second way of writing a schedule row: a proposal's frozen bytes and a recompute's
bytes come out of the same function, which is what makes adoption a publish
rather than a translation.

Two bases exist for the same date and this module keeps them apart by name.

- ``engine-day-plan`` is this channel-day re-planned live against current
  settings, constraints, overrides and models, through
  :mod:`kairos_api.break_store`. It is what a person on the day board is looking
  at and what proposals are authored against.
- ``committed-weekly-plan`` is the row the saved ``weekly_break_schedule.csv``
  actually holds for this channel-day. It moves only when something saves.

They part company whenever config or a model moves under a saved plan, which is
why :func:`kairos_api.break_store.committed_totals` exists at all. A comparison
across the two is offered as reference and refuses to attribute a delta between
them; a comparison within one basis attributes every agora.

Editing goes through the day board's own path: the moves become the same
``PlacementConstraint`` rows a save would write
(:func:`kairos_api.break_api_board.candidate_restrictions`), the day is planned
again with them in force and nothing written, and the resulting plan's rows are
what the proposal freezes. So a proposal's money is what the engine would really
produce for those edits, including the re-planning of everything around them -
measured on ``רשת 13 / 2024-11-01`` at about one second, which is why creating a
proposal is a deliberate act and never a keystroke.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from kairos_api import break_api_board as board
from kairos_api import break_store, channel_scope, day_proposal_store as store

ENGINE_BASIS = "engine-day-plan"
COMMITTED_BASIS = "committed-weekly-plan"


def schedule_columns() -> list[str]:
    from kairos.export.schedule import COLUMNS

    return list(COLUMNS)


def rows_from_day_plan(plan: break_store.DayPlan) -> pd.DataFrame:
    """One channel-day's plan as schedule rows, by the export's own construction."""
    from kairos.export.incremental import rows_from_result

    records = rows_from_result(list(plan.segments), plan.result)
    return pd.DataFrame(records, columns=schedule_columns())


def caps_from_settings() -> dict[str, Any]:
    """The licence ceilings this day is measured against, or None where unset."""
    from kairos_api.core import _load_settings, _settings_to_guardrails

    guardrails = _settings_to_guardrails(_load_settings())
    return {
        "max_daily_ad_seconds": float(guardrails.max_daily_ad_seconds),
        "max_ad_seconds_per_hour": float(guardrails.max_ad_seconds_per_hour),
        "max_breaks_per_hour": int(guardrails.max_breaks_per_hour),
    }


def _guardrail_verdict(plan: break_store.DayPlan) -> dict[str, Any]:
    counts, pins = break_store.arrangement(plan)
    items = board._breaks_for_guardrails(plan, pins)
    verdict = board.compliance(items, plan.guardrails)
    verdict["available"] = True
    verdict["basis"] = "the engine's own guardrail run on this arrangement at authoring time"
    return verdict


def _engine_block(plan: break_store.DayPlan, elapsed_ms: Optional[float] = None) -> dict[str, Any]:
    from kairos.optimize.evaluate import score

    counts, pins = break_store.arrangement(plan)
    items = board._breaks_for_guardrails(plan, pins)
    evaluation = score(plan.basis, counts, revenue_weight=plan.revenue_weight, placements=pins)
    return {
        "totals": board.totals(plan, evaluation, items),
        "compliance": _guardrail_verdict(plan),
        "hours": board.hour_load(items, plan.guardrails),
        "basis": board.basis(plan),
        "engine_ms": None if elapsed_ms is None else round(elapsed_ms, 3),
    }


def plan_with_moves(plan: break_store.DayPlan,
                    moves: list[dict[str, Any]]) -> tuple[break_store.DayPlan, float]:
    """The day re-planned with these edits pinned, nothing written. One optimization."""
    import time

    from kairos.optimize.day_core import _optimize_one_day
    from kairos_api.overrides import _resolved_store_overrides, _stored_constraints

    candidates = board.candidate_restrictions(plan, moves)
    active, _stale = _resolved_store_overrides(list(plan.segments))
    started = time.perf_counter()
    result = _optimize_one_day(
        list(plan.segments),
        constraints=list(_stored_constraints()) + candidates,
        overrides=active if active.overrides else None,
        **plan.engine_kwargs,
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return break_store.DayPlan(
        channel=plan.channel, day=plan.day, segments=plan.segments, result=result,
        basis=plan.basis, engine_kwargs=plan.engine_kwargs,
    ), elapsed_ms


def baseline_for_day(day: str) -> dict[str, Any]:
    """The day everybody proposes against, with its identity and its ceilings.

    Raises :class:`LookupError` exactly as :func:`kairos_api.break_store.day_plan`
    does, so a route answers the same honest 404 it already answers elsewhere.
    """
    plan = break_store.day_plan(day)
    rows = rows_from_day_plan(plan)
    live = live_plan_identity()
    ref = store.baseline_ref(
        rows,
        basis=ENGINE_BASIS,
        computed_at=live.get("computed_at"),
        plan_sha256=live.get("plan_sha256"),
    )
    return {
        "channel": plan.channel,
        "day": plan.day,
        "plan": plan,
        "rows": rows,
        "ref": ref,
        "caps": caps_from_settings(),
        "engine": _engine_block(plan),
    }


def proposal_rows(day: str, moves: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """The rows one proposal should freeze, and the engine verdict that produced them.

    With no moves this is the day as the engine plans it now, which is the honest
    way to open a competing version: an author starts from what everyone else is
    looking at. With moves it is the day re-planned with those edits pinned.
    """
    plan = break_store.day_plan(day)
    elapsed: Optional[float] = None
    edited = plan
    if moves:
        edited, elapsed = plan_with_moves(plan, moves)
    rows = rows_from_day_plan(edited)
    live = live_plan_identity()
    return {
        "channel": plan.channel,
        "day": plan.day,
        "rows": rows,
        "engine": _engine_block(edited, elapsed),
        "rows_source": (ENGINE_BASIS + "-with-edits") if moves else ENGINE_BASIS,
        "baseline_ref": store.baseline_ref(
            rows_from_day_plan(plan), basis=ENGINE_BASIS,
            computed_at=live.get("computed_at"), plan_sha256=live.get("plan_sha256"),
        ),
    }


# --------------------------------------------------------------- the live plan

def plan_path() -> Path:
    """The committed weekly plan, resolved at call time so a test can relocate it."""
    from kairos_api import plan_version_store

    return plan_version_store.plan_path()


def shipped_plan_path() -> Path:
    from kairos.export.schedule import DEFAULT_OUTPUT_PATH

    return Path(DEFAULT_OUTPUT_PATH)


def live_plan_identity() -> dict[str, Any]:
    from kairos_api import plan_version_store

    state = plan_version_store.live_state()
    return {"plan_sha256": state.get("sha256"), "computed_at": state.get("computed_at")}


def live_day_rows(channel: str, day: str) -> Optional[pd.DataFrame]:
    """This channel-day's rows as the committed weekly plan actually holds them."""
    path = plan_path()
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty or "channel" not in frame.columns or "date" not in frame.columns:
        return None
    mine = frame[
        (frame["channel"].astype(str).str.strip() == str(channel).strip())
        & (frame["date"].astype(str).str.strip() == str(day).strip())
    ]
    return None if mine.empty else mine.reset_index(drop=True)


def publish_day(channel: str, day: str, rows: pd.DataFrame, *, actor: str,
                proposal_name: str) -> dict[str, Any]:
    """Put an adopted day into the plan of record, freezing what it replaced first.

    The freeze is not politeness. Adoption replaces rows that every export, the
    week board and the overview read, so the plan as it stood is captured as a
    named version before a byte moves and the adoption is reversible by the
    restore path that already exists.

    The read-only wall is checked FIRST, before the freeze and before a byte is
    read, because a refusal that has already written something is not a refusal.
    Measured the hard way while building this: with the guard after the freeze, a
    single guard test on a read-only tree wrote twelve real ``pre_day_adoption``
    versions of the operator's own plan into ``data/plan_versions`` before being
    turned away. :mod:`kairos.export.plan_guard` states the rule in its own
    docstring - the wall is the first shipped-path decision, ahead of any
    provenance write - and this is what obeying it looks like.
    """
    from kairos.export.plan_guard import (record_shipped_plan_write,
                                          require_shipped_plan_writable)
    from kairos_api import plan_version_store

    target = plan_path()
    shipped = shipped_plan_path()
    require_shipped_plan_writable(target, shipped)
    if not target.exists():
        raise FileNotFoundError(str(target))
    safety = plan_version_store.freeze(
        name=f"before adopting {proposal_name}"[:80],
        actor=actor,
        note=f"automatic freeze taken before publishing a day proposal for {channel} on {day}",
        source="pre_day_adoption",
    )
    record_shipped_plan_write(target, shipped)
    existing = pd.read_csv(target)
    keep = existing[~(
        (existing["channel"].astype(str).str.strip() == str(channel).strip())
        & (existing["date"].astype(str).str.strip() == str(day).strip())
    )]
    replaced = int(len(existing) - len(keep))
    merged = pd.concat([keep, rows.reindex(columns=list(existing.columns))],
                       ignore_index=True)
    if "channel" in merged.columns and "date" in merged.columns:
        merged = merged.sort_values(["date", "channel", "start_time"], kind="stable"
                                    ) if "start_time" in merged.columns else merged.sort_values(
            ["date", "channel"], kind="stable")
    tmp = target.with_name(target.name + ".tmp")
    merged.to_csv(tmp, index=False, encoding="utf-8")
    os.replace(tmp, target)

    from kairos_api.core import _read_csv_cached

    _read_csv_cached.cache_clear()
    break_store.invalidate()
    owned, note = channel_scope.scope_frame(rows)
    return {
        "ok": True,
        "rows_written": int(len(rows)),
        "rows_replaced": replaced,
        "plan_rows_after": int(len(merged)),
        "safety_version_id": safety.get("version_id"),
        "safety_version_name": safety.get("name"),
        "path": str(target),
        "scope": note,
        "revenue_published": round(float(pd.to_numeric(
            owned.get("predicted_revenue", 0), errors="coerce").fillna(0).sum()), 2),
    }
