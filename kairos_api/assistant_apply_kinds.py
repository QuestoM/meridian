"""The appliers: one per proposal kind, replaying an approved item for real.

Split out of :mod:`kairos_api.assistant_actions` under the file-size law. The
registry object itself is imported back there, so
``assistant_actions._APPLIERS`` is this dict and
``assistant_propose_extra.register_action_plane`` still extends the same
mapping it always did.

**Every applier takes the approving account.** It used to take only the
payload, and every applier reaches its store with ``request=None`` so the
store's own ``_actor(request)`` resolves to nobody. For the stores that carry
an actor column that is not a cosmetic gap: a make-good raised through an
approved proposal would have recorded a shortfall, a state and a blank name.
The account that matters is the one that APPROVED the item, not the one that
asked Kai for it, because approval is where the accountability sits in a
review-first plane, and it is the only one of the two that had to hold a write
role. It is passed to every applier for one signature, and used by the ones
whose store records who acted.
"""

from __future__ import annotations

from typing import Any

from kairos_api import assistant_tools


def _apply_settings(payload: dict[str, Any], actor: str) -> dict[str, Any]:
    from kairos_api import settings_api
    from kairos_api.core import KairosSettings, _load_settings, _model_dump

    changes = dict(payload.get("changes") or {})
    forbidden = sorted(set(changes) - assistant_tools.ALLOWED_SETTINGS_FIELDS)
    if not changes or forbidden:
        raise ValueError(f"settings changes not applicable: forbidden fields {forbidden}")
    merged = {**_model_dump(_load_settings()), **changes}
    settings_api.update_settings(KairosSettings(**merged))
    return {"changed": changes}


def _apply_constraint(payload: dict[str, Any], actor: str) -> dict[str, Any]:
    from kairos_api.constraints import ConstraintCreate, create_constraint

    record = create_constraint(ConstraintCreate(**dict(payload.get("constraint") or {})))
    return {"constraint_id": record.get("constraint_id")}


def _apply_override(payload: dict[str, Any], actor: str) -> dict[str, Any]:
    from kairos_api.overrides import OverrideCreate, create_override

    record = create_override(OverrideCreate(**dict(payload.get("override") or {})))
    return {"override_id": record.get("override_id")}


def _apply_pricing(payload: dict[str, Any], actor: str) -> dict[str, Any]:
    from kairos_api.pricing_api import PricingUpdate, put_pricing

    changes = dict(payload.get("changes") or {})
    if not changes:
        raise ValueError("pricing changes are empty")
    state = put_pricing(PricingUpdate(overrides=changes))
    return {"has_overrides": bool(state.get("has_overrides"))}


def _expand_days(days: list[str]) -> list[dict[str, str]]:
    """Every (channel, day) pair the saved plan carries for the named days."""
    from kairos_api.core import _load_break_schedule

    frame = _load_break_schedule()
    if frame.empty or "channel" not in frame.columns or "date" not in frame.columns:
        raise ValueError("no saved weekly plan to scope a recompute by days; use scope 'full'")
    dates = frame["date"].astype(str).str.strip()
    channels = frame["channel"].astype(str).str.strip()
    pairs: list[dict[str, str]] = []
    for day in days:
        day_channels = sorted(set(channels[dates == day]) - {""})
        if not day_channels:
            raise ValueError(f"the saved plan has no rows for {day}; use scope 'full'")
        pairs.extend({"channel": channel, "day": day} for channel in day_channels)
    return pairs


def _apply_advertiser(payload: dict[str, Any], actor: str) -> dict[str, Any]:
    """Create or edit one advertiser through the real advertisers store. request is
    None (programmatic), so the store skips its own manual-edit snapshot; the assistant
    restore point taken before this apply covers the advertiser rules file."""
    from kairos_api.advertisers import (
        AdvertiserCreate,
        AdvertiserUpdate,
        create_advertiser,
        update_advertiser,
    )

    name = str(payload.get("advertiser_name") or "").strip()
    changes = dict(payload.get("changes") or {})
    if not name or not changes:
        raise ValueError("advertiser change needs an advertiser_name and non-empty changes")
    if payload.get("create"):
        record = create_advertiser(AdvertiserCreate(advertiser_id=name, **changes), request=None)
    else:
        record = update_advertiser(name, AdvertiserUpdate(**changes), request=None)
    return {"advertiser_id": record.get("advertiser_id")}


def _apply_recompute(payload: dict[str, Any], actor: str) -> dict[str, Any]:
    from kairos_api.recompute_api import RecomputeJobRequest, start_recompute_job

    scope = payload.get("scope")
    if scope == "full":
        response = start_recompute_job(None)
    else:
        days = [str(day) for day in (scope or {}).get("days", [])]
        response = start_recompute_job(RecomputeJobRequest(scope=_expand_days(days)))
    return {"job_id": response["job_id"], "already_running": bool(response.get("already_running"))}


_APPLIERS = {"settings": _apply_settings, "constraint": _apply_constraint,
             "override": _apply_override, "pricing": _apply_pricing,
             "recompute": _apply_recompute, "advertiser_change": _apply_advertiser}
