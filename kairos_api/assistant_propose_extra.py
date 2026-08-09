"""Calendar-event and agency PROPOSE validators, appliers and registration.

The propose_event_change and propose_agency_change tools flow through the SAME
review-first machinery as every other proposal: validators here run the exact
checks the manual routes run and only shape a pending item (nothing mutates at
capture time), and the appliers replay approved items through the real route
functions (events_api, agencies, agency_conditions) with ``request=None``.

Registration keeps the frozen action-plane modules untouched:
:func:`register_action_plane` (called once when kairos_api.assistant loads)
merges the appliers into ``assistant_actions._APPLIERS``, maps each new item
kind to its logical file in ``version_store._LOGICAL_FOR_KIND`` (so an
assistant apply lands in the unified version timeline exactly like the built-in
kinds), and wraps ``assistant_actions._state_files_for`` so the pre-apply
restore point copies the events and agency CSVs byte-for-byte.

Money honesty: an event proposal that carries a non-neutral price multiplier,
and a pricing proposal that touches ``pricing_activation.events``, both state
the forecast revenue effect on the saved plan's event days in the summary,
labeled as an estimate from the operator-asserted multiplier, and disclose
whether the layer is currently on or off.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

# Ordered so the Hebrew field list in a summary is deterministic.
_EVENT_FIELDS = ("name", "type", "start_date", "end_date", "intensity",
                 "notes", "active", "price_multiplier")
_EVENT_ACTIONS = ("create", "update", "deactivate")
_AGENCY_ACTIONS = ("create", "update", "deactivate", "link_advertiser",
                   "unlink_advertiser", "add_condition", "update_condition",
                   "delete_condition")

# Item kind per agency action: each kind maps to exactly the store file the
# apply mutates, so restore points and the version timeline stay precise.
_AGENCY_KIND_BY_ACTION = {
    "create": "agency_change", "update": "agency_change", "deactivate": "agency_change",
    "link_advertiser": "agency_link_change", "unlink_advertiser": "agency_link_change",
    "add_condition": "agency_condition_change", "update_condition": "agency_condition_change",
    "delete_condition": "agency_condition_change",
}


def agency_change_kind(payload: dict[str, Any]) -> str:
    return _AGENCY_KIND_BY_ACTION.get(str(payload.get("action") or ""), "agency_change")


# --- event-day forecast notes (honest estimates, never silent money moves) --------
def _plan_day_revenue() -> dict[str, float]:
    """Planned gross revenue per owned-channel plan day, empty when no plan."""
    from kairos_api import assistant_context

    frame, _owned, _competitors, _reason = assistant_context._owned_frame()
    if frame is None:
        return {}
    return {str(day): float(group["predicted_revenue"].sum())
            for day, group in frame.groupby("date_text", sort=True)}


def _events_layer_enabled() -> bool:
    from kairos.optimize.pricing import PricingModel
    from kairos_api.core import _load_settings

    overrides = getattr(_load_settings(), "pricing_overrides", None) or {}
    return bool(PricingModel.from_config(overrides).enable_events)


def _delta_text(delta: float) -> str:
    rounded = int(round(delta))
    return f"+{rounded:,}" if rounded >= 0 else f"{rounded:,}"


def events_activation_note(turning_on: bool) -> str:
    """The forecast statement for a pricing_activation.events proposal."""
    day_revenue = _plan_day_revenue()
    if not day_revenue:
        return "אין תוכנית שבועית שמורה, כך שלא ניתן לאמוד שינוי הכנסה על ימי אירוע"
    from kairos.optimize.pricing import load_event_day_multipliers

    multipliers = load_event_day_multipliers()
    affected = [(day, multipliers[day], revenue) for day, revenue in sorted(day_revenue.items())
                if multipliers.get(day, 1.0) != 1.0]
    if not affected:
        return "אף יום בתוכנית השמורה אינו מכוסה באירוע עם מכפיל שונה מ-1.0, כך שההכנסה הצפויה לא תשתנה מהצעד הזה"
    delta = sum(revenue * (multiplier - 1.0) for _day, multiplier, revenue in affected)
    days_text = ", ".join(day for day, _m, _r in affected[:7])
    verb = "הפעלת" if turning_on else "כיבוי"
    if not turning_on:
        delta = -delta
    return (f"{verb} שכבת תמחור האירועים משנה הכנסה צפויה מוערכת של {_delta_text(delta)} ILS "
            f"על {len(affected)} ימי תוכנית ({days_text}); האומדן מכפיל את הכנסת היום במכפיל המוצהר, ברוטו")


def event_multiplier_note(record: dict[str, Any]) -> "str | None":
    """The forecast statement for one event proposal carrying a non-neutral
    multiplier: which saved plan days it covers and the estimated gross delta."""
    try:
        multiplier = float(record.get("price_multiplier", 1.0))
    except (TypeError, ValueError):
        return None
    if multiplier == 1.0:
        return None
    from kairos_api import events_api

    layer_text = ("שכבת תמחור האירועים פעילה" if _events_layer_enabled()
                  else "שכבת תמחור האירועים כבויה כרגע, כך שהמכפיל לא ישפיע על התחזית עד הפעלתה")
    day_revenue = _plan_day_revenue()
    overlap = events_api._plan_overlap_dates(
        {"start_date": record.get("start_date"), "end_date": record.get("end_date") or None},
        sorted(day_revenue),
    )
    if not overlap:
        return f"{layer_text}; האירוע אינו חופף אף יום בתוכנית השמורה, ללא שינוי הכנסה צפויה"
    delta = sum(day_revenue[day] * (multiplier - 1.0) for day in overlap)
    return (f"{layer_text}; האירוע חופף {len(overlap)} ימי תוכנית, שינוי הכנסה צפויה מוערך "
            f"{_delta_text(delta)} ILS בהפעלה (מכפיל {multiplier} על הכנסת היום, ברוטו)")


def _with_note(summary: str, note: "str | None") -> str:
    return f"{summary}. {note}" if note else summary


# --- propose_event_change validator -----------------------------------------------
def _clean_event_fields(event: Any) -> dict[str, Any]:
    if not isinstance(event, dict) or not event:
        raise ValueError("event must be a non-empty object of event fields")
    unknown = sorted(set(event) - set(_EVENT_FIELDS))
    if unknown:
        raise ValueError(f"unknown event fields: {', '.join(unknown)}. Allowed: {', '.join(_EVENT_FIELDS)}")
    return dict(event)


def _validated_event(merged: dict[str, Any]) -> dict[str, Any]:
    """Run the events store's own validator on a full field set."""
    from kairos_api import events_api

    try:
        return events_api._validate(
            merged.get("name"), merged.get("type"), merged.get("start_date"),
            merged.get("end_date") or "", merged.get("intensity", 3),
            merged.get("price_multiplier", 1.0),
        )
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid event: {str(exc)[:200]}") from exc


def _event_record(event_id: str) -> dict[str, Any]:
    from kairos_api import events_api

    frame = events_api._load_frame()
    try:
        return events_api._record(frame.loc[events_api._locate(frame, event_id)])
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc


def _validate_event_change(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    action = str(args.get("action", "") or "").strip().lower()
    if action not in _EVENT_ACTIONS:
        raise ValueError(f"action must be one of {list(_EVENT_ACTIONS)}, got {action!r}")
    if action == "create":
        event = _clean_event_fields(args.get("event"))
        validated = _validated_event(event)
        span = validated["start_date"] + (f" עד {validated['end_date']}" if validated["end_date"] else ", ללא תאריך סיום")
        summary = (f"אירוע: יצירת {validated['name']} ({validated['type']}) מ-{span}, "
                   f"עוצמה {validated['intensity']}, מכפיל מחיר {validated['price_multiplier']}")
        merged_for_note = {**event, **validated}
        return ({"action": "create", "event": event}, _with_note(summary, event_multiplier_note(merged_for_note)))
    event_id = str(args.get("event_id", "") or "").strip()
    if not event_id:
        raise ValueError(f"event_id is required for action {action!r}")
    current = _event_record(event_id)
    if action == "deactivate":
        summary = f"אירוע: כיבוי {current['name']} ({event_id}); האירוע נשמר אך מפסיק להשפיע"
        return {"action": "deactivate", "event_id": event_id}, summary
    event = _clean_event_fields(args.get("event"))
    merged = {**current, "end_date": current.get("end_date") or "", **event}
    _validated_event(merged)
    fields = ", ".join(field for field in _EVENT_FIELDS if field in event)
    summary = f"אירוע: עדכון {current['name']} ({fields})"
    return ({"action": "update", "event_id": event_id, "event": event},
            _with_note(summary, event_multiplier_note(merged) if "price_multiplier" in event else None))


# --- propose_agency_change validator ----------------------------------------------
def _agency_record(agency_id: str) -> dict[str, Any]:
    from kairos_api import agencies

    frame = agencies._load_frame()
    try:
        return agencies._row_to_record(frame.loc[agencies._locate(frame, agency_id)])
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc


def _validated_agency_values(agency_id: str, changes: Any, create: bool) -> dict[str, str]:
    from kairos_api import agencies

    if not isinstance(changes, dict) or not changes:
        raise ValueError("changes must be a non-empty object of agency fields")
    try:
        if create:
            payload = agencies.AgencyCreate(agency_id=agency_id, **changes)
        else:
            payload = agencies.AgencyUpdate(**changes)
        return agencies._apply_validated(payload, partial=not create)
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc
    except ValueError:
        raise
    except Exception as exc:  # pydantic and unexpected shapes surface honestly
        raise ValueError(f"invalid agency change: {str(exc)[:300]}") from exc


def _validate_agency_record_action(agency_id: str, action: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api import agencies

    if action == "create":
        exists = (agencies._load_frame()["agency_id"].astype(str) == agency_id).any()
        if exists:
            raise ValueError(f"agency {agency_id!r} already exists; propose an update, not a create")
        changes = dict(args.get("changes") or {})
        if not str(changes.get("name", "") or "").strip():
            raise ValueError("a new agency needs a name (the exact daily-file string)")
        _validated_agency_values(agency_id, changes, create=True)
        summary = f"סוכנות: יצירת {changes['name']} ({agency_id})"
        return {"agency_id": agency_id, "action": "create", "changes": changes}, summary
    current = _agency_record(agency_id)
    if action == "deactivate":
        summary = f"סוכנות: השעיית {current['name']} ({agency_id}); התנאים וההנחה מפסיקים להשפיע על הלדג'ר היומי"
        return {"agency_id": agency_id, "action": "deactivate"}, summary
    changes = dict(args.get("changes") or {})
    _validated_agency_values(agency_id, changes, create=False)
    parts = [f"{field} {current.get(field)} -> {changes[field]}" for field in sorted(changes)]
    summary = f"סוכנות: עדכון {current['name']} ({agency_id}): " + ", ".join(parts)
    return {"agency_id": agency_id, "action": "update", "changes": changes}, summary


def _validate_agency_link_action(agency_id: str, action: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api import agency_conditions

    _agency_record(agency_id)  # must exist
    advertiser = str(args.get("advertiser", "") or "").strip()
    if not advertiser:
        raise ValueError(f"advertiser is required for action {action!r}")
    links = agency_conditions._load_csv(agency_conditions.LINKS_PATH, agency_conditions.LINK_COLUMNS)
    manual = links[(links["source"] == "manual") & (links["advertiser"].astype(str) == advertiser)]
    if action == "link_advertiser":
        if not manual.empty:
            holder = str(manual.iloc[0]["agency_id"])
            raise ValueError(f"{advertiser!r} already has a manual link to agency {holder!r}; propose unlinking it first")
        summary = f"סוכנות: קישור ידני של {advertiser} אל {agency_id}"
    else:
        if manual.empty or not (manual["agency_id"].astype(str) == agency_id).any():
            raise ValueError(f"no manual link from {advertiser!r} to agency {agency_id!r}")
        summary = f"סוכנות: הסרת הקישור הידני של {advertiser} מ-{agency_id}"
    return {"agency_id": agency_id, "action": action, "advertiser": advertiser}, summary


def _validate_agency_condition_action(agency_id: str, action: str, args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api import agency_conditions

    _agency_record(agency_id)  # must exist
    condition = args.get("condition")
    if not isinstance(condition, dict) or not str(condition.get("rule_id", "") or "").strip():
        raise ValueError("condition must be an object carrying at least rule_id")
    rule_id = str(condition["rule_id"]).strip()
    existing = {record["rule_id"] for record in agency_conditions.conditions_for(agency_id)}
    if action == "add_condition":
        if rule_id in existing:
            raise ValueError(f"rule {rule_id!r} already exists for agency {agency_id!r}; propose an update")
        try:
            model = agency_conditions.ConditionCreate(**condition)
            agency_conditions._validate_effect(model.effect)
        except HTTPException as exc:
            raise ValueError(str(exc.detail)) from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"invalid condition: {str(exc)[:300]}") from exc
        summary = f"סוכנות: תנאי חדש {rule_id} ({model.effect}) עבור {agency_id}"
    elif action == "delete_condition":
        if rule_id not in existing:
            raise ValueError(f"no rule {rule_id!r} for agency {agency_id!r}")
        summary = f"סוכנות: מחיקת תנאי {rule_id} של {agency_id}"
    else:
        if rule_id not in existing:
            raise ValueError(f"no rule {rule_id!r} for agency {agency_id!r}")
        fields = {key: value for key, value in condition.items() if key != "rule_id"}
        if not fields:
            raise ValueError("an update_condition needs at least one field beside rule_id")
        try:
            if "effect" in fields:
                agency_conditions._validate_effect(fields["effect"])
            agency_conditions.ConditionUpdate(**fields)
        except HTTPException as exc:
            raise ValueError(str(exc.detail)) from exc
        except ValueError:
            raise
        except Exception as exc:
            raise ValueError(f"invalid condition: {str(exc)[:300]}") from exc
        summary = f"סוכנות: עדכון תנאי {rule_id} של {agency_id} ({', '.join(sorted(fields))})"
    return {"agency_id": agency_id, "action": action, "condition": dict(condition)}, summary


def _validate_agency_change(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    agency_id = str(args.get("agency_id", "") or "").strip()
    if not agency_id:
        raise ValueError("agency_id is required")
    action = str(args.get("action", "") or "").strip().lower()
    if action not in _AGENCY_ACTIONS:
        raise ValueError(f"action must be one of {list(_AGENCY_ACTIONS)}, got {action!r}")
    if action in ("create", "update", "deactivate"):
        return _validate_agency_record_action(agency_id, action, args)
    if action in ("link_advertiser", "unlink_advertiser"):
        return _validate_agency_link_action(agency_id, action, args)
    return _validate_agency_condition_action(agency_id, action, args)


EXTRA_PROPOSE_VALIDATORS = {
    "propose_event_change": _validate_event_change,
    "propose_agency_change": _validate_agency_change,
}


# --- appliers: replay approved items through the real route functions -------------
def _apply_event_change(payload: dict[str, Any], actor: str) -> dict[str, Any]:
    from kairos_api import events_api

    action = str(payload.get("action") or "")
    if action == "create":
        record = events_api.create_event(events_api.EventCreate(**dict(payload.get("event") or {})), request=None)
    elif action == "deactivate":
        record = events_api.update_event(str(payload.get("event_id") or ""),
                                         events_api.EventUpdate(active=False), request=None)
    else:
        record = events_api.update_event(str(payload.get("event_id") or ""),
                                         events_api.EventUpdate(**dict(payload.get("event") or {})),
                                         request=None)
    return {"event_id": record.get("event_id"), "action": action}


def _apply_agency_change(payload: dict[str, Any], actor: str) -> dict[str, Any]:
    from kairos_api import agencies, agency_conditions

    agency_id = str(payload.get("agency_id") or "")
    action = str(payload.get("action") or "")
    if action == "create":
        record = agencies.create_agency(
            agencies.AgencyCreate(agency_id=agency_id, **dict(payload.get("changes") or {})), request=None)
        return {"agency_id": record.get("agency_id"), "action": action}
    if action == "update":
        record = agencies.update_agency(
            agency_id, agencies.AgencyUpdate(**dict(payload.get("changes") or {})), request=None)
        return {"agency_id": record.get("agency_id"), "action": action}
    if action == "deactivate":
        record = agencies.deactivate_agency(agency_id, request=None)
        return {"agency_id": record.get("agency_id"), "action": action, "status": record.get("status")}
    if action == "link_advertiser":
        result = agency_conditions.create_link(
            agency_id, agency_conditions.LinkCreate(advertiser=str(payload.get("advertiser") or "")), request=None)
        return {"agency_id": agency_id, "action": action, "linked": result.get("linked")}
    if action == "unlink_advertiser":
        result = agency_conditions.delete_link(agency_id, str(payload.get("advertiser") or ""), request=None)
        return {"agency_id": agency_id, "action": action, "unlinked": result.get("unlinked")}
    condition = dict(payload.get("condition") or {})
    rule_id = str(condition.get("rule_id") or "")
    if action == "add_condition":
        record = agency_conditions.create_condition(
            agency_id, agency_conditions.ConditionCreate(**condition), request=None)
        return {"agency_id": agency_id, "action": action, "rule_id": record.get("rule_id")}
    if action == "delete_condition":
        agency_conditions.delete_condition(agency_id, rule_id, request=None)
        return {"agency_id": agency_id, "action": action, "rule_id": rule_id}
    fields = {key: value for key, value in condition.items() if key != "rule_id"}
    record = agency_conditions.update_condition(
        agency_id, rule_id, agency_conditions.ConditionUpdate(**fields), request=None)
    return {"agency_id": agency_id, "action": action, "rule_id": record.get("rule_id")}


EXTRA_APPLIERS = {
    "event_change": _apply_event_change,
    "agency_change": _apply_agency_change,
    "agency_link_change": _apply_agency_change,
    "agency_condition_change": _apply_agency_change,
}

# Each new kind maps to exactly one logical version-store file and one restore
# state file, so both timelines cover assistant applies on the new stores.
EXTRA_LOGICAL_FOR_KIND = {
    "event_change": "events",
    "agency_change": "agencies",
    "agency_link_change": "agency_links",
    "agency_condition_change": "agency_conditions",
}


def _extra_state_paths(kinds: set[str]) -> list[Any]:
    from pathlib import Path

    from kairos_api import agencies, agency_conditions, events_api

    paths: list[Any] = []
    if "event_change" in kinds:
        paths.append(Path(events_api.EVENTS_PATH))
    if "agency_change" in kinds:
        paths.append(Path(agencies.AGENCIES_PATH))
    if "agency_link_change" in kinds:
        paths.append(Path(agency_conditions.LINKS_PATH))
    if "agency_condition_change" in kinds:
        paths.append(Path(agency_conditions.CONDITIONS_PATH))
    if "pacing_decision" in kinds:
        from kairos_api import makegood_store

        paths.append(Path(makegood_store.MAKE_GOODS_PATH))
    return paths


def register_action_plane() -> None:
    """Idempotently extend the apply, restore and version machinery in place.

    Called when kairos_api.assistant loads (which also mounts the action-plane
    router), so registration always precedes any HTTP apply. The state-files
    seam is wrapped, not replaced: built-in kinds keep their exact behavior.
    """
    from kairos_api import assistant_actions, assistant_pacing_propose, version_store

    assistant_actions._APPLIERS.update(EXTRA_APPLIERS)
    version_store._LOGICAL_FOR_KIND.update(EXTRA_LOGICAL_FOR_KIND)
    # The pacing decisions register themselves the same way, through the same
    # call, so there is one moment the action plane is extended and not two.
    assistant_pacing_propose.register()
    if getattr(assistant_actions._state_files_for, "_kairos_extra_kinds", False):
        return
    original = assistant_actions._state_files_for

    def _state_files_extended(kinds: set[str]) -> list[Any]:
        return [*original(kinds), *_extra_state_paths(kinds)]

    _state_files_extended._kairos_extra_kinds = True  # type: ignore[attr-defined]
    assistant_actions._state_files_for = _state_files_extended
