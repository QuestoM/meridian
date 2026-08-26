"""Merging two records that turned out to be one party.

A duplicate is not a cosmetic problem. An agency carries a rebate and priced
conditions; when the same party sits on two ids, its rules apply to half its
spots, its net is split across two rows that never reconcile, and every number
downstream looks correct on its own. So a merge is a MONEY act, and it is built
like every other money act in this product: proposed, shown field by field,
approved by a human, and reversible.

Three properties carry the design:

**Nothing is deleted.** The duplicate keeps its row, its status becomes
``suspended`` and its name strings fold into the survivor's aliases, so the
daily file's own spelling still resolves to the surviving record. A merge is
therefore undoable by hand from the record itself - and the version store's
restore covers it like any other approved apply.

**Conflicts are stated, never silently resolved.** :func:`plan_merge` returns a
row per differing field: the survivor's value, the duplicate's value, and which
one a merge would keep. The default rule is conservative and explicit - the
SURVIVOR wins every populated field, and the duplicate only fills fields the
survivor left empty - because a merge must never quietly change a rebate the
operator already agreed. Where the duplicate would win (an empty survivor
field), the row says so, and the operator can override any single field before
approving.

**Every attachment moves, and the plan counts them first.** Links, conditions
and campaigns are re-pointed to the survivor. The counts are computed in the
plan, so the approval card states exactly how many rows will move, and the
apply re-reads and re-counts rather than trusting the plan it was handed.
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

# Fields that identify rather than describe: they never merge by value, they
# fold into the survivor's alias list so both spellings keep resolving.
IDENTITY_FIELDS = ("name", "display_name", "aliases")

# Fields a merge must never touch on the survivor, because they are the
# survivor's own identity or lifecycle, not a describable property.
FROZEN_FIELDS = ("agency_id", "status")

SURVIVOR = "survivor"
DUPLICATE = "duplicate"


def merge_evidence(survivor_id: str, duplicate_id: str) -> dict[str, Any]:
    """WHY we believe these two records are one party, computed from the stores.

    The conflict table answers "what changes". This answers the question that
    comes before it and matters more: are these one company at all? It resolves
    the duplicate's own identity against the roster and reports what the
    survivor scored - the signals, the model's ruling if one was needed, and the
    tier. Computed here rather than taken from whoever asked for the merge,
    because a proposer's claim about identity is not evidence of it.
    """
    from kairos_api import entity_resolution

    survivor = _record(survivor_id)
    duplicate = _record(duplicate_id)
    resolution = entity_resolution.resolve_counterparty(
        "agency",
        str(duplicate.get("name") or duplicate_id),
        vat_id=str(duplicate.get("vat_id") or ""),
        aliases=_alias_tokens(duplicate)[1:],
        limit=20,
    )
    scored = next((c for c in resolution["candidates"] if c["entity_id"] == survivor_id), None)
    signals = (scored or {}).get("signals", {})
    # The deterministic half: a signal no model produced. A merge that rests on
    # the model alone rests on one witness, and that witness also proposed it.
    deterministic = bool(signals.get("normalized_exact") or signals.get("vat_match")
                         or signals.get("alias_hit"))
    return {
        "verdict": resolution["verdict"] if scored is not None else entity_resolution.NONE,
        "signals": signals,
        "model_verdict": (scored or {}).get("model_verdict"),
        "model_confidence": (scored or {}).get("model_confidence"),
        "model_reason": (scored or {}).get("model_reason"),
        "model_used": resolution["model_used"],
        "deterministic_signal": deterministic,
        "survivor_name": survivor.get("name"),
        "duplicate_name": duplicate.get("name"),
        "scored": scored is not None,
    }


def _agencies_frame() -> pd.DataFrame:
    from kairos_api import agencies

    return agencies._load_frame()


def _record(agency_id: str) -> dict[str, Any]:
    from kairos_api import agencies

    frame = _agencies_frame()
    matches = frame[frame["agency_id"].astype(str).str.strip() == agency_id]
    if matches.empty:
        raise ValueError(f"agency {agency_id!r} does not exist")
    return agencies._row_to_record(matches.iloc[0])


def _alias_tokens(record: dict[str, Any]) -> list[str]:
    """Every name string one record answers to, de-duplicated, order kept."""
    raw = [str(record.get("name", "")), str(record.get("display_name", "")),
           *str(record.get("aliases", "")).split("|")]
    out: list[str] = []
    for token in raw:
        token = token.strip()
        if token and token not in out:
            out.append(token)
    return out


def _attachment_counts(agency_id: str) -> dict[str, int]:
    """How many rows in each store point at this agency. Read tolerantly: a
    missing side store is zero attachments, never an exception that would make
    a merge plan unavailable."""
    counts = {"links": 0, "conditions": 0, "campaigns": 0}
    try:
        from kairos_api import agency_conditions

        # ``manual`` is a list of advertiser NAMES, so its length is the count.
        counts["links"] = len(agency_conditions.links_for(agency_id).get("manual") or [])
        counts["conditions"] = len(agency_conditions.conditions_for(agency_id))
    except Exception:  # noqa: BLE001 - a side store must not break the plan
        pass
    try:
        from kairos_api import campaigns_api_store

        frame = campaigns_api_store.load_frame()
        if "agency_id" in frame.columns:
            rows = frame[frame["agency_id"].astype(str).str.strip() == agency_id]
            counts["campaigns"] = int((rows["record_type"].astype(str) == "campaign").sum()
                                      if "record_type" in rows.columns else len(rows))
    except Exception:  # noqa: BLE001
        pass
    return counts


def plan_merge(survivor_id: str, duplicate_id: str,
               overrides: Optional[dict[str, str]] = None) -> dict[str, Any]:
    """What a merge WOULD do, field by field and row by row. Writes nothing.

    ``overrides`` maps a field name to ``'survivor'`` or ``'duplicate'`` and
    lets the operator flip any single conflict before approving.
    """
    survivor_id = str(survivor_id or "").strip()
    duplicate_id = str(duplicate_id or "").strip()
    if not survivor_id or not duplicate_id:
        raise ValueError("both a survivor and a duplicate agency id are required")
    if survivor_id == duplicate_id:
        raise ValueError("an agency cannot be merged into itself")
    survivor = _record(survivor_id)
    duplicate = _record(duplicate_id)
    overrides = {str(k): str(v) for k, v in (overrides or {}).items()}

    conflicts: list[dict[str, Any]] = []
    resolved: dict[str, str] = {}
    for field in sorted(set(survivor) | set(duplicate)):
        if field in FROZEN_FIELDS or field in IDENTITY_FIELDS:
            continue
        left = str(survivor.get(field, "") or "").strip()
        right = str(duplicate.get(field, "") or "").strip()
        if left == right:
            continue
        # The conservative default: a populated survivor value stands; the
        # duplicate only fills what the survivor left empty.
        default = DUPLICATE if (not left and right) else SURVIVOR
        winner = overrides.get(field, default)
        if winner not in (SURVIVOR, DUPLICATE):
            raise ValueError(f"override for {field!r} must be 'survivor' or 'duplicate'")
        value = right if winner == DUPLICATE else left
        conflicts.append({
            "field": field,
            "survivor_value": left or None,
            "duplicate_value": right or None,
            "winner": winner,
            "kept_value": value or None,
            "overridden": field in overrides,
        })
        if winner == DUPLICATE:
            resolved[field] = value

    # Identity strings never conflict: they accumulate, so both spellings keep
    # resolving to the surviving record after the merge.
    survivor_aliases = _alias_tokens(survivor)
    folded = [token for token in _alias_tokens(duplicate) if token not in survivor_aliases]
    aliases_after = [*survivor_aliases[1:], *folded] if survivor_aliases else folded

    attachments = _attachment_counts(duplicate_id)
    return {
        "survivor": {"agency_id": survivor_id, "name": survivor.get("name")},
        "duplicate": {"agency_id": duplicate_id, "name": duplicate.get("name")},
        "conflicts": conflicts,
        "field_changes": resolved,
        "aliases_folded": folded,
        "aliases_after": aliases_after,
        "attachments_to_move": attachments,
        "attachments_total": sum(attachments.values()),
        "duplicate_after": "suspended",
        # Precise about HOW, because "reversible" alone would be a comfortable
        # word doing no work. Nothing is deleted, so the duplicate can be
        # reactivated by hand; and the approval takes a restore point across all
        # four stores it touches, which is the only way back for a survivor
        # field an override overwrote.
        "reversible": True,
        "reversal": (
            "The duplicate is suspended, not deleted, so it can be reactivated "
            "from its own record. A field the operator flipped to the "
            "duplicate's value overwrites the survivor's, and comes back only "
            "through the approval's restore point, which covers the agencies, "
            "links, conditions and campaigns stores together."
        ),
        "basis": (
            "The survivor keeps every populated field unless the operator flips "
            "it; the duplicate fills only what the survivor left empty. Its name "
            "strings fold into the survivor's aliases and its row is suspended, "
            "never deleted, so the merge is reversible by hand."
        ),
    }


def apply_merge(survivor_id: str, duplicate_id: str,
                overrides: Optional[dict[str, str]] = None,
                actor: str = "") -> dict[str, Any]:
    """Perform the merge. Re-plans from the CURRENT stores rather than trusting
    a plan computed when the proposal was written, so an agency edited between
    proposal and approval cannot be merged on stale values."""
    from kairos_api import agencies, agency_conditions

    plan = plan_merge(survivor_id, duplicate_id, overrides)
    survivor_id = plan["survivor"]["agency_id"]
    duplicate_id = plan["duplicate"]["agency_id"]

    # 1. The surviving record: the fields the plan resolved to the duplicate,
    #    plus the folded aliases so both spellings still resolve.
    changes = dict(plan["field_changes"])
    if plan["aliases_after"]:
        changes["aliases"] = "|".join(plan["aliases_after"])
    if changes:
        agencies.update_agency(survivor_id, agencies.AgencyUpdate(**changes), request=None)

    # 2. The attachments. Links and conditions are re-created on the survivor
    #    and removed from the duplicate; a link that already exists on the
    #    survivor is left alone rather than duplicated.
    # A half-done merge is worse than none: an attachment that fails to move is
    # RECORDED and returned, never swallowed, so the operator sees that the
    # duplicate still holds something and can finish it by hand.
    moved = {"links": 0, "conditions": 0, "campaigns": 0}
    problems: list[str] = []
    # links_for returns manual links as plain advertiser NAMES, not row dicts.
    existing_links = set(agency_conditions.links_for(survivor_id).get("manual") or [])
    for advertiser in list(agency_conditions.links_for(duplicate_id).get("manual") or []):
        advertiser = str(advertiser).strip()
        if not advertiser:
            continue
        try:
            # RELEASE BEFORE CLAIM. The links store enforces one manual link per
            # advertiser, so creating on the survivor while the duplicate still
            # holds it is refused with a 409. Delete first, then create - and if
            # the create fails, hand the link back rather than losing it.
            agency_conditions.delete_link(duplicate_id, advertiser, request=None)
            if advertiser in existing_links:
                continue  # the survivor already claims it; releasing was the whole move
            try:
                agency_conditions.create_link(
                    survivor_id, agency_conditions.LinkCreate(advertiser=advertiser), request=None)
                moved["links"] += 1
            except Exception:
                agency_conditions.create_link(
                    duplicate_id, agency_conditions.LinkCreate(advertiser=advertiser), request=None)
                raise
        except Exception as exc:  # noqa: BLE001 - one link's failure is not the merge's
            problems.append(f"link {advertiser}: {type(exc).__name__}")
    for condition in list(agency_conditions.conditions_for(duplicate_id)):
        rule_id = str(condition.get("rule_id") or "")
        try:
            fields = {k: v for k, v in condition.items() if k not in ("rule_id", "agency_id")}
            agency_conditions.create_condition(
                survivor_id, agency_conditions.ConditionCreate(**fields), request=None)
            agency_conditions.delete_condition(duplicate_id, rule_id, request=None)
            moved["conditions"] += 1
        except Exception as exc:  # noqa: BLE001
            problems.append(f"condition {rule_id}: {type(exc).__name__}")

    # 3. Campaigns are re-pointed in place: the campaign is the same booking,
    #    it was simply filed under the duplicate's id.
    try:
        from kairos_api import campaigns_api_store

        # The store's own lock and its backup-then-atomic write, so a merge is
        # exactly as safe as a manual campaign edit.
        with campaigns_api_store.lock():
            frame = campaigns_api_store.load_frame()
            if "agency_id" in frame.columns:
                mask = frame["agency_id"].astype(str).str.strip() == duplicate_id
                moved["campaigns"] = int(mask.sum())
                if moved["campaigns"]:
                    frame.loc[mask, "agency_id"] = survivor_id
                    campaigns_api_store.write_frame(frame)
    except Exception as exc:  # noqa: BLE001
        problems.append(f"campaigns: {type(exc).__name__}")

    # 4. The duplicate stops pricing anything. Suspended, not deleted.
    agencies.deactivate_agency(duplicate_id, request=None)

    result = {
        "survivor": survivor_id,
        "duplicate": duplicate_id,
        "duplicate_status": "suspended",
        "fields_taken_from_duplicate": sorted(plan["field_changes"]),
        "aliases_folded": plan["aliases_folded"],
        "moved": moved,
        "actor": actor,
        "reversible": True,
        "reversal": plan["reversal"],
    }
    if problems:
        result["incomplete"] = problems
    return result
