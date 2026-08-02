"""The machine-readable terms behind a proposal's one-line summary.

An item's ``summary`` is a record, not display copy. It is written into the
audit trail, returned to the model as the tool result the next turn reasons
over, and pinned by tests outside this piece, so it stays exactly as the
validator wrote it. Printing that English sentence verbatim on a Hebrew
operator surface is the defect this module exists to close.

So every item the six older validators produce also carries ``summary_terms``:
a stable code plus the values the sentence was built from, copied from the
payload the validator already accepted. The surface says the sentence in the
reader's own language from those terms, and because both come from one payload
the two can never disagree.

Nothing here computes or estimates anything. A kind with no entry carries no
terms at all and the surface falls back to the summary, which is what the
calendar-event and agency validators want: those summaries are already Hebrew.
"""

from __future__ import annotations

from typing import Any

# The prefix kairos_api.assistant_propose_tools writes before the optional
# Hebrew activation note on a rate-card proposal. The note is real measured
# disclosure and belongs on the card, so the terms carry it rather than losing
# it; it is taken from the summary only when the summary actually starts with
# the prefix built from the same payload, never guessed.
_PRICING_PREFIX = "pricing: edit "


def _token(value: Any) -> str:
    """A vocabulary token as the validators normalise it: stripped, lowered."""
    return str(value or "").strip().lower()


def _names(values: Any) -> list[str]:
    """Sorted field names from a changes mapping, or an empty list."""
    return sorted(str(name) for name in values) if isinstance(values, dict) else []


def _settings_terms(payload: dict[str, Any], summary: str) -> dict[str, Any] | None:
    fields = _names(payload.get("changes"))
    return {"code": "settings", "fields": fields} if fields else None


def _recompute_terms(payload: dict[str, Any], summary: str) -> dict[str, Any] | None:
    scope = payload.get("scope")
    if scope == "full":
        return {"code": "recompute", "scope": "full"}
    if isinstance(scope, dict) and isinstance(scope.get("days"), list):
        return {"code": "recompute", "scope": "days",
                "days": [str(day) for day in scope["days"]]}
    return None


def _constraint_terms(payload: dict[str, Any], summary: str) -> dict[str, Any] | None:
    constraint = payload.get("constraint")
    if not isinstance(constraint, dict):
        return None
    return {
        "code": "constraint",
        "effect": _token(constraint.get("effect")),
        "scope_type": _token(constraint.get("scope_type")),
        "scope_value": str(constraint.get("scope_value") or ""),
        "predicate": bool(constraint.get("where")),
    }


def _override_terms(payload: dict[str, Any], summary: str) -> dict[str, Any] | None:
    override = payload.get("override")
    if not isinstance(override, dict):
        return None
    return {
        "code": "override",
        "kind": _token(override.get("kind")),
        "scope": _token(override.get("scope")),
        "target_id": str(override.get("target_id") or ""),
        "value": str(override.get("value") or ""),
    }


def _pricing_terms(payload: dict[str, Any], summary: str) -> dict[str, Any] | None:
    keys = _names(payload.get("changes"))
    if not keys:
        return None
    terms: dict[str, Any] = {"code": "pricing", "keys": keys}
    prefix = _PRICING_PREFIX + ", ".join(keys)
    if summary.startswith(prefix):
        note = summary[len(prefix):].lstrip(". ").strip()
        if note:
            terms["note"] = note
    return terms


def _advertiser_terms(payload: dict[str, Any], summary: str) -> dict[str, Any] | None:
    name = str(payload.get("advertiser_name") or "")
    if not name:
        return None
    return {
        "code": "advertiser",
        "action": "create" if payload.get("create") else "edit",
        "name": name,
        "fields": _names(payload.get("changes")),
    }


_TERMS_BY_KIND = {
    "settings": _settings_terms,
    "recompute": _recompute_terms,
    "constraint": _constraint_terms,
    "override": _override_terms,
    "pricing": _pricing_terms,
    "advertiser_change": _advertiser_terms,
}

# Every code this module can emit. The surface maps each one; a test asserts the
# two lists match, so a code added here without a Hebrew reading fails there
# rather than on an operator's screen.
CODES = ("settings", "recompute", "constraint", "override", "pricing", "advertiser")


def terms_for(kind: str, payload: dict[str, Any], summary: str) -> dict[str, Any] | None:
    """The terms behind this item's summary, or None when it has none.

    None is the honest answer for a kind whose summary is already written in
    the reader's language, and for a payload that does not carry the values the
    sentence needs. The surface prints the summary itself in both cases.
    """
    builder = _TERMS_BY_KIND.get(str(kind or ""))
    if builder is None or not isinstance(payload, dict):
        return None
    return builder(payload, str(summary or ""))


def terms_for_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """The terms for a stored proposal item, derived at read time when the item
    was written before this module existed.

    Measured on the batch store: every item already on disk carries its
    validated payload and none carries terms, so every one of them fell back to
    the English record on both surfaces. Deriving here is not a migration and
    not a guess. It is the same pure function over the same payload the
    validator accepted, so a stored item and a fresh one say the same sentence.
    """
    if not isinstance(item, dict):
        return None
    stored = item.get("summary_terms")
    if stored:
        return stored
    return terms_for(item.get("kind", ""), item.get("payload") or {}, item.get("summary", ""))
