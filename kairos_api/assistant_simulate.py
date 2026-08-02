"""Side-effect-free settings simulation for the assistant.

Given a subset of the mutable settings allowlist, this computes the owned
channel's before and after WITHOUT applying anything: it deep-reads the saved
settings, applies the changes to a copy, and runs the SAME scenario runner both
ways scoped to the operator's channel through the shared owned-scope selector,
then prices each side with the SAME per-break retention-cost model the frontier
net comparison uses. The result is ``{before, after, delta}`` where, on each
side, ``net == gross - retention_cost`` on the same basis the committed plan
would show. It writes nothing: the runner reads the reference EPG and optimizes
in memory, and the priced results are memoized. When the engine is down or no
owned channel is configured it returns an honest ``{status: 'unavailable',
reason}`` instead of guessing.

This is the primitive the ``simulate_settings_change`` read tool exposes, the
one the goal-seeker climbs, and the source of the effect block attached to a
settings-change proposal.
"""

from __future__ import annotations

import json
from datetime import date
from functools import lru_cache
from typing import Any

from kairos_api.assistant_tools import ALLOWED_SETTINGS_FIELDS

_MONEY_KEYS = ("gross", "retention_cost", "net", "breaks")


def _unavailable(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason}


@lru_cache(maxsize=64)
def _priced_side(
    signature: tuple[tuple[str, int], ...],
    channel: str,
    day: str,
    revenue_weight: int,
    retention_floor: float,
    max_breaks_per_hour: int,
    risk_lambda: float,
    objective_mode: str,
    today_iso: str,
    settings_json: str,
) -> dict[str, Any]:
    """Run and price one scenario side, memoized on every input that moves it.

    Reuses ``run_scenario`` (the live optimizer seam), ``_plan_segment_index``
    (the segments that priced the plan) and ``_scenario_plan_money`` (the
    frontier net-comparison money builder), so a simulated figure is the same
    figure the committed plan and the net comparison would show. Memoized so the
    saved-settings side is computed once and reused across every what-if and
    every goal-seek step in the process. Returns the money builder's
    ``{available, gross, retention_cost, net, breaks}`` or its honest
    ``{available: False, reason}``.
    """
    del signature  # part of the memo key only; keeps results honest across data edits
    from kairos_api.core import _plan_segment_index, run_scenario
    from kairos_api.dashboard_api import _scenario_plan_money

    settings_map = json.loads(settings_json)
    today = None
    if today_iso:
        try:
            today = date.fromisoformat(today_iso)
        except ValueError:
            today = None
    payload = run_scenario(
        revenue_weight=revenue_weight,
        retention_floor=retention_floor,
        max_breaks_per_hour=max_breaks_per_hour,
        risk_lambda=risk_lambda,
        channel=channel,
        day=day,
        objective_mode=objective_mode,
        today=today,
        settings=settings_map,
    )
    segments = list(_plan_segment_index(((channel, str(day)),), settings_map).values())
    return _scenario_plan_money(payload, segments, risk_lambda)


def _side_for(signature: tuple[tuple[str, int], ...], channel: str, day: str, settings: Any) -> dict[str, Any]:
    """Derive the runner controls from one settings object and price that side."""
    from kairos_api.core import _model_dump, _reference_today

    settings_map = _model_dump(settings)
    return _priced_side(
        signature,
        channel,
        day,
        int(settings.revenue_weight),
        float(settings.min_retention_floor),
        int(settings.max_breaks_per_hour),
        float(settings.risk_lambda),
        str(getattr(settings, "objective_mode", "blend") or "blend"),
        _reference_today(settings).isoformat(),
        json.dumps(settings_map, sort_keys=True, ensure_ascii=False, default=str),
    )


def simulate_settings_change(changes: dict[str, Any]) -> dict[str, Any]:
    """Owned-channel before/after for a proposed settings change, applying nothing.

    ``changes`` is a subset of the mutable settings allowlist. Returns
    ``{status: 'ok', channel, day, before, after, delta}`` on success (each of
    ``before``/``after`` carries ``gross, retention_cost, net, breaks``; ``net``
    equals ``gross - retention_cost`` on both sides), or ``{status:
    'unavailable', reason}`` when the engine is down, no owned channel is
    configured, the changes are not applicable, or a side cannot be priced.
    Never raises and never writes.
    """
    try:
        from kairos_api.core import (
            KairosSettings,
            _ENGINE_AVAILABLE,
            _load_settings,
            _model_dump,
        )
        from kairos_api.dashboard_api import _frontier_data_signature, _owned_scope

        if not _ENGINE_AVAILABLE:
            return _unavailable("the optimization engine is unavailable")
        if not isinstance(changes, dict) or not changes:
            return _unavailable("changes must be a non-empty object of settings fields")
        forbidden = sorted(set(changes) - ALLOWED_SETTINGS_FIELDS)
        if forbidden:
            return _unavailable(f"fields not allowed for simulation: {', '.join(forbidden)}")

        saved = _load_settings()
        try:
            modified = KairosSettings(**{**_model_dump(saved), **changes})
        except Exception as exc:  # noqa: BLE001 - a bad value is an honest unavailable, not a crash
            return _unavailable(f"invalid settings values: {str(exc)[:200]}")

        channel, day = _owned_scope(saved)
        if not channel:
            return _unavailable(
                "no operator channel is configured; pick your channel in settings first"
            )
        if not day:
            return _unavailable("the owned channel has no dated programmes to scope the simulation")

        signature = _frontier_data_signature()
        before = _side_for(signature, channel, day, saved)
        after = _side_for(signature, channel, day, modified)
    except Exception as exc:  # noqa: BLE001 - the primitive is honest, never fatal
        return _unavailable(f"simulation failed ({type(exc).__name__}): {str(exc)[:200]}")

    if not before.get("available"):
        return _unavailable(str(before.get("reason") or "the saved-settings plan could not be priced"))
    if not after.get("available"):
        return _unavailable(str(after.get("reason") or "the proposed-settings plan could not be priced"))

    before_block = {key: before[key] for key in _MONEY_KEYS}
    after_block = {key: after[key] for key in _MONEY_KEYS}
    delta = {
        "gross": round(after_block["gross"] - before_block["gross"], 2),
        "retention_cost": round(after_block["retention_cost"] - before_block["retention_cost"], 2),
        "net": round(after_block["net"] - before_block["net"], 2),
        "breaks": int(after_block["breaks"] - before_block["breaks"]),
    }
    return {
        "status": "ok",
        "channel": channel,
        "day": day,
        "before": before_block,
        "after": after_block,
        "delta": delta,
    }


# The basis of the figures and the figures themselves, kept as two named groups
# because the surface that prints the money has to print what it covers. The
# money is one representative channel-day of the owned channel, never a weekly
# total, and a reader who cannot see which channel and which day cannot tell
# those two apart.
EFFECT_BASIS_KEYS = ("channel", "day")
EFFECT_MONEY_KEYS = ("before", "after", "delta")


def settings_effect(changes: Any) -> dict[str, Any]:
    """The owned-channel before/after a settings change would produce, with its basis.

    ``{channel, day, before, after, delta}`` when the simulation succeeds, else
    an honest ``{status: 'unavailable', reason}``. ``channel`` is the operator's
    own channel and ``day`` is the representative broadcast day both sides were
    optimized on, so the approval surface can name the basis of the money it
    prints. Used as the additive effect on a settings-change proposal item; the
    apply engine ignores it. Never raises.
    """
    sim = simulate_settings_change(dict(changes or {}))
    if sim.get("status") == "ok":
        return {key: sim[key] for key in EFFECT_BASIS_KEYS + EFFECT_MONEY_KEYS}
    return {"status": "unavailable", "reason": str(sim.get("reason") or "simulation unavailable")}
