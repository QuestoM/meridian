"""Tool registry for the assistant's action plane.

READ tools execute immediately in-loop, each reusing a real dashboard builder
(plus simulate_settings_change, the side-effect-free owned-channel what-if), and
a failing executor returns an honest ``{"error": ...}`` rather than fabricating
or crashing; every read result is stamped with a provenance source. PROPOSE
tools are NEVER executed in-loop: each is validated through the SAME validators
the manual UI path uses and captured as a ``pending`` (or honestly ``rejected``)
proposal item, and a settings change also carries its simulated effect. Nothing
mutates until the operator approves and kairos_api.assistant_actions replays it
through the real seam. The settings allowlist below is the contract for
propose_settings_change: only operator-tunable levers may be proposed.
"""

from __future__ import annotations

import json
import re
import uuid
import logging
from typing import Any

from fastapi import HTTPException

# The mutable KairosSettings fields the model may propose to change. Everything
# else (operator_channel, locale, currency, profile identity, notes, raw
# pricing_overrides, protected_program_types) is rejected with an honest reason.
ALLOWED_SETTINGS_FIELDS = frozenset(
    {
        "revenue_weight",
        "objective_mode",
        "risk_lambda",
        "max_ad_minutes_per_hour",
        "max_breaks_per_hour",
        "min_break_spacing_minutes",
        "min_retention_floor",
        "max_daily_ad_minutes",
        "protected_program_max_ad_minutes_per_hour",
        "gold_breaks_enabled",
        "gold_breaks_max_per_day",
        "sponsorships_enabled",
        "pacing_enabled",
        "pacing_reference_date",
        "pacing_urgency_k",
        "pacing_urgency_max",
        "pacing_ahead_k",
        "pacing_weight_floor",
        "pacing_epsilon",
    }
)

_ISO_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_REASON_PROPERTY = {
    "type": "string",
    "description": "Why this change is being proposed, in the operator's language.",
}


logger = logging.getLogger(__name__)


def _tool(name: str, description: str, properties: dict[str, Any] | None = None,
          required: list[str] | None = None) -> dict[str, Any]:
    """One Anthropic tool schema in the messages.create format."""
    schema: dict[str, Any] = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = required
    return {"name": name, "description": description, "input_schema": schema}


def _propose(name: str, description: str, key: str, key_description: str,
             typed: bool = True) -> dict[str, Any]:
    """A propose-family tool schema: one payload property plus required reason."""
    prop: dict[str, Any] = {"description": key_description}
    if typed:
        prop["type"] = "object"
    return _tool(name, description, {key: prop, "reason": _REASON_PROPERTY}, [key, "reason"])


READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _tool(
        "get_settings",
        "Read the saved operator settings: the tunable optimizer levers "
        "(revenue_weight, risk_lambda, guardrails, pacing) plus operator_channel.",
    ),
    _tool(
        "get_day_detail",
        "Read one saved plan day of the operator's own channel: every segment "
        "with breaks, revenue and retention, ordered by revenue descending.",
        {"date": {"type": "string", "description": "Plan day, YYYY-MM-DD."}},
        ["date"],
    ),
    _tool("list_constraints", "List the stored placement constraints with their ids."),
    _tool("list_overrides", "List the stored manual overrides with their ids, grouped by scope."),
    _tool(
        "get_pricing",
        "Read the pricing hierarchy: base CPP, the premium layers with their "
        "values and live/off state, and whether operator overrides exist.",
    ),
    _tool(
        "get_net_comparison",
        "Read the saved-plan objective versus a net-focused plan (gross, "
        "retention cost, net, breaks and the deltas), or its honest status.",
    ),
    _tool("get_compliance", "Read the regulatory compliance verdict summary for the saved plan."),
    _tool(
        "simulate_settings_change",
        "Simulate a settings change on the owned channel WITHOUT applying it: returns "
        "before/after gross, retention cost, net and breaks plus the deltas, on the plan's "
        "own basis. Use it for what-ifs and to search toward a goal. Allowed fields only: "
        + ", ".join(sorted(ALLOWED_SETTINGS_FIELDS)) + ".",
        {"changes": {"type": "object", "description": "Allowed settings fields to simulate."}},
        ["changes"],
    ),
    _tool(
        "get_recommendations",
        "Read the real recommendations the overview computes for the owned channel: the "
        "top segments to review, each with its risk band, revenue impact, retention and "
        "the override kind the review implies. Call this when the operator asks what to "
        "fix or review, or where the plan is at risk.",
    ),
    _tool(
        "get_frontier",
        "Read the owned-channel revenue-vs-retention frontier: the Pareto sweep points, "
        "the net-focused point, and which point is the current plan. Call this when the "
        "operator asks about the revenue and retention tradeoff, alternative retention "
        "floors, or where the current plan sits. A status of 'computing' means the sweep "
        "is still running in the background; say so instead of inventing points.",
    ),
    _tool(
        "get_audience_stability",
        "Read the measured weekly level drift of the retention coefficient base, the "
        "audience-stability check. Call this when the operator asks whether the audience "
        "or retention model is stable, drifting or fresh. Reports an honest unavailable "
        "when the coefficients artifact carries no measurement.",
    ),
    _tool(
        "get_plan_days",
        "Read the committed weekly plan per day for the owned channel: date, breaks, "
        "revenue and, when derivable, the modeled retention cost and net. Call this when "
        "the operator asks which day earns most, about per-day totals, or for a week "
        "overview that includes retention cost.",
    ),
]

PROPOSE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    _propose(
        "propose_settings_change",
        "Propose changing saved operator settings. Allowed fields only: "
        + ", ".join(sorted(ALLOWED_SETTINGS_FIELDS))
        + ". The operator must approve before anything is saved.",
        "changes",
        "Field-to-new-value map of allowed settings fields.",
    ),
    _propose(
        "propose_constraint",
        "Propose a new scoped placement constraint (scope_type, effect, optional "
        "where predicate), validated against the frozen constraint contract.",
        "constraint",
        "Constraint fields as the constraints API accepts them.",
    ),
    _propose(
        "propose_override",
        "Propose a new manual override (scope, target_id, kind, value), "
        "validated against the overrides schema.",
        "override",
        "Override fields as the overrides API accepts them.",
    ),
    _propose(
        "propose_pricing_change",
        "Propose a rate-card edit: a pricing_overrides deep-merge patch in the "
        "YAML shape (base_price_per_second_per_tvr_point, premiums, pricing_activation).",
        "changes",
        "Partial pricing_overrides patch to deep-merge.",
    ),
    _propose(
        "propose_recompute",
        "Propose recomputing the saved weekly schedule so approved changes take "
        "effect. scope is the string 'full' or {\"days\": [\"YYYY-MM-DD\", ...]}.",
        "scope",
        "'full' or an object {\"days\": [\"YYYY-MM-DD\", ...]}.",
        typed=False,
    ),
]

READ_TOOL_NAMES = frozenset(schema["name"] for schema in READ_TOOL_SCHEMAS)
PROPOSE_TOOL_NAMES = frozenset(schema["name"] for schema in PROPOSE_TOOL_SCHEMAS)

# Proposal item kind per propose tool; the apply engine dispatches on kind.
KIND_BY_TOOL = {
    "propose_settings_change": "settings",
    "propose_constraint": "constraint",
    "propose_override": "override",
    "propose_pricing_change": "pricing",
    "propose_recompute": "recompute",
}


def anthropic_tools(include_propose: bool = True) -> list[dict[str, Any]]:
    """The full tool list for the messages.create tools parameter."""
    return [*READ_TOOL_SCHEMAS, *PROPOSE_TOOL_SCHEMAS] if include_propose else list(READ_TOOL_SCHEMAS)


# READ executors. Each one calls the real builder of the owning module.
def _read_get_settings(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api.core import _load_settings, _model_dump

    saved = _model_dump(_load_settings())
    subset = {field: saved.get(field) for field in sorted(ALLOWED_SETTINGS_FIELDS)}
    subset["operator_channel"] = saved.get("operator_channel") or None
    return subset


def _read_get_day_detail(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api import assistant_context

    date = str(args.get("date", "")).strip()
    if not _ISO_DAY_RE.fullmatch(date):
        return {"error": f"date must be YYYY-MM-DD, got {date!r}"}
    frame, owned, _competitors, reason = assistant_context._owned_frame()
    if frame is None:
        return {"error": reason or "no owned-channel plan is available"}
    if date not in set(frame["date_text"]):
        return {"error": f"the saved plan has no rows for {date} on {owned}"}
    return assistant_context._day_detail_section(frame, date, [], [])


def _read_list_constraints(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api.constraints import list_constraints

    payload = list_constraints()
    return {"constraints": payload["constraints"]}


def _read_list_overrides(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api.overrides import list_overrides

    payload = list_overrides()
    return {"overrides": payload["overrides"]}


def _read_get_pricing(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api.pricing_api import get_pricing

    state = get_pricing()
    layers = [
        {
            "name": layer["name"],
            "live_today": layer["live_today"],
            **({"value": layer["value"]} if layer.get("kind") == "base" else {"values": layer.get("values", {})}),
        }
        for layer in state["layers"]
    ]
    return {
        "currency": state["currency"],
        "units": state["units"],
        "base": state["base"],
        "layers": layers,
        "activation": state["activation"],
        "has_operator_overrides": state["has_overrides"],
    }


def _read_get_net_comparison(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api.scenario_api import optimizer_net_comparison

    return optimizer_net_comparison()


def _read_get_compliance(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api.dashboard_api import compliance

    verdict = compliance()
    checks = [
        {key: check.get(key) for key in ("id", "status", "observed", "limit", "unit")}
        for check in verdict.get("checks", [])
    ]
    violations = verdict.get("violations") or []
    return {
        "status": verdict.get("status"),
        "profile": verdict.get("profile"),
        "checks": checks,
        "violations_total": len(violations),
        "violations_sample": violations[:5],
    }


def _read_simulate_settings_change(args: dict[str, Any]) -> dict[str, Any]:
    # The primitive validates the changes against the SAME allowlist and returns an
    # honest unavailable for a forbidden/unknown field or a bad value, never crashing.
    from kairos_api import assistant_simulate

    return assistant_simulate.simulate_settings_change(args.get("changes"))


def _read_get_recommendations(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api.core import _load_break_schedule
    from kairos_api.dashboard_api import _build_recommendations

    rows = _build_recommendations(_load_break_schedule())
    slim = [
        {
            "title": row.get("title"),
            "title_he": row.get("title_he"),
            "risk": row.get("risk"),
            "channel": row.get("channel"),
            "segment_id": row.get("segment_id"),
            "program_type": row.get("program_type"),
            "date": row.get("date"),
            "weekday": row.get("weekday"),
            "start_clock": row.get("start_clock"),
            "num_breaks": row.get("num_breaks"),
            "impact_ils": row.get("impact"),
            "retention_pct": row.get("retention"),
            "rationale": row.get("rationale"),
            "proposed_kind": row.get("proposed_kind"),
            "actionable": row.get("actionable"),
        }
        for row in rows[:5]
    ]
    payload: dict[str, Any] = {"recommendations": slim, "count": len(slim)}
    if not slim:
        payload["note"] = "the overview builder produced no recommendations for the owned channel"
    return payload


def _read_get_frontier(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api.core import _load_settings
    from kairos_api.dashboard_api import _frontier_state

    points, net_bundle, status = _frontier_state(_load_settings())
    payload: dict[str, Any] = {"status": status, "points": [dict(point) for point in points]}
    if status == "no_channel":
        payload["reason"] = "no operator channel is configured in settings"
    elif status == "computing":
        payload["reason"] = "the frontier sweep is still computing in the background; try again shortly"
    net_point = (net_bundle or {}).get("net_point")
    current = next((dict(point) for point in points if point.get("selected")), None)
    if isinstance(net_point, dict) and net_point.get("selected"):
        current = dict(net_point)
    payload["current_plan_point"] = current
    payload["net_focused_point"] = dict(net_point) if isinstance(net_point, dict) else None
    return payload


def _read_get_audience_stability(args: dict[str, Any]) -> dict[str, Any]:
    from kairos_api.catalog_api import impact

    drift = impact().get("drift")
    if not isinstance(drift, dict) or not drift:
        return {
            "status": "unavailable",
            "reason": "the coefficients artifact carries no level-drift measurement",
        }
    payload = {key: value for key, value in drift.items() if key != "weekly_levels"}
    levels = drift.get("weekly_levels")
    if isinstance(levels, list):
        payload["weekly_levels"] = levels[-12:]
        if len(levels) > 12:
            payload["weekly_levels_truncated"] = True
            payload["weekly_levels_total"] = len(levels)
    return payload


def _read_get_plan_days(args: dict[str, Any]) -> dict[str, Any]:
    from kairos.optimize.revenue_net import frame_revenue_net
    from kairos_api import assistant_context

    frame, owned, _competitors, reason = assistant_context._owned_frame()
    if frame is None:
        return {"error": reason or "no owned-channel plan is available"}
    days: list[dict[str, Any]] = []
    cost_missing_reason: str | None = None
    for date_text, group in frame.groupby("date_text", sort=True):
        entry: dict[str, Any] = {
            "date": str(date_text),
            "weekday": assistant_context._weekday_label(str(date_text), group),
            "breaks": int(group["num_breaks"].sum()),
            "revenue_ils": int(round(float(group["predicted_revenue"].sum()))),
        }
        money = frame_revenue_net(group)
        if money.get("available"):
            entry["retention_cost_ils"] = round(float(money["retention_cost_ils"]), 2)
            entry["revenue_net_ils"] = round(float(money["revenue_net_ils"]), 2)
        else:
            entry["retention_cost_ils"] = None
            cost_missing_reason = cost_missing_reason or str(money.get("reason") or "")
        days.append(entry)
    payload: dict[str, Any] = {"channel": owned, "days_total": len(days), "days": days[:31]}
    if len(days) > 31:
        payload["truncated"] = True
        payload["days_omitted"] = len(days) - 31
    if cost_missing_reason:
        payload["retention_cost_note"] = cost_missing_reason
    return payload


_READ_EXECUTORS = {
    "get_settings": _read_get_settings,
    "get_day_detail": _read_get_day_detail,
    "list_constraints": _read_list_constraints,
    "list_overrides": _read_list_overrides,
    "get_pricing": _read_get_pricing,
    "get_net_comparison": _read_get_net_comparison,
    "get_compliance": _read_get_compliance,
    "simulate_settings_change": _read_simulate_settings_change,
    "get_recommendations": _read_get_recommendations,
    "get_frontier": _read_get_frontier,
    "get_audience_stability": _read_get_audience_stability,
    "get_plan_days": _read_get_plan_days,
}

# The provenance stamp for each read tool result: the endpoint or dataset the
# figures came from. Attached uniformly in execute_read_tool and surfaced on the trace.
SOURCE_BY_TOOL = {
    "get_settings": "saved settings",
    "get_day_detail": "saved weekly plan, owned channel",
    "list_constraints": "stored placement constraints",
    "list_overrides": "stored manual overrides",
    "get_pricing": "pricing hierarchy (rate card and operator overrides)",
    "get_net_comparison": "owned-channel scenario runner, net comparison",
    "get_compliance": "compliance verdict over the committed plan",
    "simulate_settings_change": "owned-channel scenario runner, representative day",
    "get_recommendations": "overview recommendations, owned channel",
    "get_frontier": "owned-channel frontier sweep",
    "get_audience_stability": "measured coefficients artifact, level-drift monitor",
    "get_plan_days": "saved weekly plan, owned channel",
}


def execute_read_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run one READ tool, stamping the result with its provenance source.

    Failures come back as {"error": ...}, never raised. Every result dict carries
    a non-empty "source" so the model can name where each figure came from.
    """
    executor = _READ_EXECUTORS.get(name)
    if executor is None:
        return {"error": f"unknown read tool {name!r}", "source": "unknown tool"}
    try:
        result = executor(args)
    except HTTPException as exc:
        result = {"error": str(exc.detail)}
    except Exception as exc:  # noqa: BLE001 - surfaced honestly, without internals
        logger.exception("assistant read tool %s failed", name)
        result = {"error": f"{name} failed ({type(exc).__name__}); details are in the server log"}
    if isinstance(result, dict):
        result.setdefault("source", SOURCE_BY_TOOL.get(name, name))
    return result


# PROPOSE validation. Same validators as the manual paths; nothing mutates.
def _validate_settings_change(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api.core import KairosSettings, _load_settings, _model_dump

    changes = args.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise ValueError("changes must be a non-empty object of settings fields")
    forbidden = sorted(set(changes) - ALLOWED_SETTINGS_FIELDS)
    if forbidden:
        raise ValueError(
            f"fields not allowed for assistant proposals: {', '.join(forbidden)}. "
            f"Allowed: {', '.join(sorted(ALLOWED_SETTINGS_FIELDS))}"
        )
    current = _model_dump(_load_settings())
    try:
        KairosSettings(**{**current, **changes})
    except Exception as exc:
        raise ValueError(f"invalid settings values: {str(exc)[:300]}") from exc
    parts = [f"{field} {current.get(field)} -> {changes[field]}" for field in sorted(changes)]
    return {"changes": dict(changes)}, "settings: " + ", ".join(parts)


def _validate_constraint(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api.constraints import ConstraintCreate, _validate_effect, _validate_scope
    from kairos_api._constraint_options import validate_where

    constraint = args.get("constraint")
    if not isinstance(constraint, dict) or not constraint:
        raise ValueError("constraint must be a non-empty object")
    try:
        model = ConstraintCreate(**constraint)
        scope_type = _validate_scope(model.scope_type)
        effect = _validate_effect(model.effect)
        validate_where(model.where)
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc
    except Exception as exc:
        raise ValueError(f"invalid constraint: {str(exc)[:300]}") from exc
    scope_text = f"{scope_type}" + (f"={model.scope_value}" if model.scope_value else "")
    where_text = " with predicate" if model.where else ""
    return {"constraint": dict(constraint)}, f"constraint: {effect} on {scope_text}{where_text}"


def _validate_override(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api.overrides import OverrideCreate, _validate

    override = args.get("override")
    if not isinstance(override, dict) or not override:
        raise ValueError("override must be a non-empty object")
    try:
        model = OverrideCreate(**override)
        scope, kind = _validate(model.scope, model.kind)
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc
    except Exception as exc:
        raise ValueError(f"invalid override: {str(exc)[:300]}") from exc
    if not str(model.target_id or "").strip():
        raise ValueError("target_id is required")
    value_text = f"={model.value}" if model.value else ""
    return {"override": dict(override)}, f"override: {kind}{value_text} on {scope} {model.target_id}"


def _validate_pricing_change(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos.optimize.pricing import PricingModel, _deep_merge
    from kairos_api.core import _load_settings

    changes = args.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise ValueError("changes must be a non-empty pricing_overrides patch")
    current = dict(getattr(_load_settings(), "pricing_overrides", None) or {})
    try:
        PricingModel.from_config(_deep_merge(current, changes))
    except ValueError as exc:
        raise ValueError(f"invalid pricing edit: {str(exc)[:300]}") from exc
    return {"changes": dict(changes)}, "pricing: edit " + ", ".join(sorted(changes))


def _validate_recompute(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    scope = args.get("scope")
    if scope == "full":
        return {"scope": "full"}, "recompute: full week"
    if isinstance(scope, dict) and set(scope) == {"days"} and isinstance(scope["days"], list):
        days = [str(day).strip() for day in scope["days"]]
        bad = sorted(day for day in days if not _ISO_DAY_RE.fullmatch(day))
        if not days:
            raise ValueError("scope.days must name at least one YYYY-MM-DD day")
        if bad:
            raise ValueError(f"scope.days entries must be YYYY-MM-DD, got: {', '.join(bad)}")
        days = sorted(set(days))
        return {"scope": {"days": days}}, "recompute: days " + ", ".join(days)
    raise ValueError("scope must be the string 'full' or an object {\"days\": [\"YYYY-MM-DD\", ...]}")


_PROPOSE_VALIDATORS = {
    "propose_settings_change": _validate_settings_change,
    "propose_constraint": _validate_constraint,
    "propose_override": _validate_override,
    "propose_pricing_change": _validate_pricing_change,
    "propose_recompute": _validate_recompute,
}


def build_proposal_item(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Validate one PROPOSE tool call and shape it as a proposal item.

    A valid call becomes a ``pending`` item; an invalid one becomes a
    ``rejected`` item whose ``error`` carries the honest reason. Never raises
    and never mutates state: the apply engine is the only writer.
    """
    kind = KIND_BY_TOOL.get(name)
    reason = str(args.get("reason", "") or "").strip()
    item: dict[str, Any] = {
        "id": uuid.uuid4().hex[:12],
        "kind": kind or name,
        "summary": "",
        "payload": {},
        "reason": reason,
        "status": "pending",
    }
    if kind is None:
        item["status"] = "rejected"
        item["error"] = f"unknown propose tool {name!r}"
        return item
    if not reason:
        item["status"] = "rejected"
        item["error"] = "reason is required: state why this change is being proposed"
        return item
    try:
        payload, summary = _PROPOSE_VALIDATORS[name](args)
    except ValueError as exc:
        item["status"] = "rejected"
        item["error"] = str(exc)
        return item
    except Exception as exc:  # noqa: BLE001 - a validator crash is still an honest rejection
        item["status"] = "rejected"
        item["error"] = f"validation failed ({type(exc).__name__}): {str(exc)[:200]}"
        return item
    item["payload"] = payload
    item["summary"] = summary
    # Additive: attach the simulated owned-channel effect of a settings change so
    # the operator sees the before/after before approving. The apply engine ignores it.
    if kind == "settings":
        from kairos_api import assistant_simulate

        item["effect"] = assistant_simulate.settings_effect(payload.get("changes"))
    return item


def handle_tool_use(block: Any, trace: list[dict[str, Any]], items: list[dict[str, Any]], propose_allowed: bool = True) -> dict[str, Any]:
    """Dispatch one tool_use block and return its tool_result message block.

    READ tools run now; PROPOSE tools are captured into ``items`` untouched by
    execution. Every call lands in ``trace`` as {tool, ok} (names only, for the
    UI). The returned content is what the model sees, so a rejected proposal
    reports its honest reason back to the model too.
    """
    name = str(getattr(block, "name", ""))
    args_raw = getattr(block, "input", None)
    args = dict(args_raw) if isinstance(args_raw, dict) else {}
    source: str | None = None
    if name in READ_TOOL_NAMES:
        payload = execute_read_tool(name, args)
        ok = "error" not in payload
        # Surface the read result's provenance on the trace step so the response's
        # source trail names, for every figure, where it came from.
        source = payload.get("source") if isinstance(payload, dict) else None
    elif name in PROPOSE_TOOL_NAMES and not propose_allowed:
        result = {"error": "the account role does not allow proposing changes"}
        trace.append({"tool": name, "ok": False})
        return {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result, ensure_ascii=False)}
    elif name in PROPOSE_TOOL_NAMES:
        item = build_proposal_item(name, args)
        items.append(item)
        ok = item["status"] == "pending"
        payload = {"captured": ok, "item_id": item["id"], "status": item["status"],
                   "summary": item["summary"]}
        if not ok:
            payload["reason"] = item.get("error")
    else:
        payload, ok = {"error": f"unknown tool {name!r}"}, False
    step: dict[str, Any] = {"tool": name, "ok": ok}
    if source:
        step["source"] = source
    trace.append(step)
    return {
        "type": "tool_result",
        "tool_use_id": str(getattr(block, "id", "")),
        "content": json.dumps(payload, ensure_ascii=False, default=str),
    }
