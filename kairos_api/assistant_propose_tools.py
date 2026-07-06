"""PROPOSE-tool validators and per-field diffs for the assistant's action plane.

Each validator runs the SAME pydantic models and checks the manual UI path uses,
so a proposal is validated exactly as a manual edit would be; nothing here
mutates state (the apply engine is the only writer). Split out of
kairos_api.assistant_tools so both files stay under the size cap; the validator
map and diff helpers are imported back into assistant_tools for build_proposal_item.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import HTTPException

_ISO_DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _validate_settings_change(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api.assistant_tools import ALLOWED_SETTINGS_FIELDS
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


def _advertiser_current(name: str) -> dict[str, Any] | None:
    """The current advertiser record for ``name``, or None when it does not exist."""
    from kairos_api.advertisers import _load_frame, _row_to_record

    frame = _load_frame()
    mask = frame["advertiser_id"].astype(str) == str(name)
    if not mask.any():
        return None
    return _row_to_record(frame.loc[frame.index[mask][0]])


def _validate_advertiser_change(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api.advertisers import AdvertiserCreate, AdvertiserUpdate

    name = str(args.get("advertiser_name", "") or "").strip()
    if not name:
        raise ValueError("advertiser_name is required")
    create = bool(args.get("create"))
    changes = args.get("changes")
    if not isinstance(changes, dict) or not changes:
        raise ValueError("changes must be a non-empty object of advertiser fields")
    current = _advertiser_current(name)
    try:
        if create:
            if current is not None:
                raise ValueError(f"advertiser {name!r} already exists; propose an edit, not a create")
            AdvertiserCreate(advertiser_id=name, **changes)
        else:
            if current is None:
                raise ValueError(f"no advertiser {name!r}; set create true to add it")
            AdvertiserUpdate(**changes)
    except ValueError:
        raise
    except Exception as exc:  # pydantic and unexpected shapes surface honestly
        raise ValueError(f"invalid advertiser change: {str(exc)[:300]}") from exc
    verb = "create" if create else "edit"
    payload = {"advertiser_name": name, "create": create, "changes": dict(changes)}
    return payload, f"advertiser: {verb} {name} ({', '.join(sorted(changes))})"


_PROPOSE_VALIDATORS = {
    "propose_settings_change": _validate_settings_change,
    "propose_constraint": _validate_constraint,
    "propose_override": _validate_override,
    "propose_pricing_change": _validate_pricing_change,
    "propose_recompute": _validate_recompute,
    "propose_advertiser_change": _validate_advertiser_change,
}


def _settings_diff(changes: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Per-field {field, before (current saved), after (proposed)} for a settings change."""
    from kairos_api.core import _load_settings, _model_dump

    current = _model_dump(_load_settings())
    return [
        {"field": field, "before": current.get(field), "after": (changes or {})[field]}
        for field in sorted(changes or {})
    ]


def _advertiser_diff(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Per-field {field, before, after} for an advertiser change; before null on create."""
    changes = payload.get("changes") or {}
    current = None if payload.get("create") else _advertiser_current(str(payload.get("advertiser_name")))
    return [
        {"field": field,
         "before": None if current is None else current.get(field),
         "after": changes[field]}
        for field in sorted(changes)
    ]
