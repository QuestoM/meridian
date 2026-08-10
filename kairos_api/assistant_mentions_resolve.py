"""Turning a typed reference into a grounding section: ``mentioned_objects``.

WHAT ARRIVES. The composer sends, beside the prose, a list of
``{type, id, label}`` records for the things the operator pointed at. The prose
still carries the human-readable label, so the question reads as a Hebrew
sentence and a copy-paste of it is readable; the typed record is what says which
of two same-named things was meant.

WHY THIS IS A CONTEXT SECTION AND NOT A SYNTHESIZED TOOL CALL, and there are two
reasons that each settle it alone.

Honest math forces the first. Every figure Kai quotes must name the section or
tool it came from and the scope it covers. A bare identifier carries neither, so
a mention that reached the model as an identifier would arrive as a figure with
no basis, which is the one thing the prompt rules forbid outright. The card
below carries ``basis`` and ``scope_channel`` per reference, exactly as the
scoped sections already do.

The second is the run trace. It is an operator-visible audit surface and
``assistant_claimed_action`` treats it as THE authority on what happened, which
is the whole reason it exists: it is the defence against the model claiming an
act it never performed. Injecting a synthetic tool call would corrupt the one
artifact that proves what Kai did. Mentions ride in CONTEXT with their own
source stamp and the trace records real calls only.

RESOLUTION IS AT SEND TIME, NEVER AT INSERTION TIME, and here that matters more
than it does in either reference product: a RUN rewrites the plan underneath the
operator. A snapshot taken when the picker was open would let Kai quote a
pre-run figure as current.

THE FOUR STATES, AND SILENT DROP IS FORBIDDEN.

The reference product that inspired this drops a dead reference silently and
leaves the text behind as prose, and three of one month's changelog entries are
bugs of exactly that shape, because the failure is invisible by design. Here it
would be worse than a bug: Kai would see a Hebrew label in the question with no
data behind it, and the rule that every figure must name its basis would push it
to answer from the label. That is fabrication. So every reference reaches the
model, including the ones that did not resolve.

``resolved``      the object was read, with its figures, basis and scope.
``changed``       the object was read and its own name is no longer the name the
                  operator inserted, so the question and the store disagree
                  about what this thing is called. The CURRENT name is sent and
                  the disagreement is stated.
``gone``          the store was read and holds no such object. No figures.
``unavailable``   the store could not be read at all. Distinct from ``gone``,
                  because "we looked and it is not there" and "we could not
                  look" are different claims, which is the tri-state doctrine
                  this product already applies to delivery and to the model.

THE BOUNDARY. A reference is a client-supplied identifier, so it is the one
place a rival name could be pushed in from outside the picker. Every card is
built through the resolution path the page context already uses, and the
programme path reads ``_owned_frame`` and nothing else, so a rival programme
resolves to ``gone`` -- the same state, with the same bytes, as a typo. Nothing
in a ``gone`` card is derived from the store; its only variable field is the
identifier the caller itself sent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from kairos_api import assistant_mentions_words as words

SECTION_NAME = "mentioned_objects"

# The sub-budget. Eight is the popup's own row cap and it is reused here for a
# different reason: it is more things than an operator points at in one
# sentence, so a list longer than this is a client fault rather than an ask.
REFS_CAP = 8
LABEL_MAX = 200
ID_MAX = 200
# How many of a day's programme names ride on the day's own card. Eager for the
# thing, lazy for its contents: the names say what the day holds, and the rows
# behind them come from a real read tool if the model wants them.
PROGRAMMES_CAP = 12

STATE_RESOLVED = "resolved"
STATE_CHANGED = "changed"
STATE_GONE = "gone"
STATE_UNAVAILABLE = "unavailable"

# Where each kind's card is read from, named in the card so the model can say it.
BASIS = {
    "day": "the saved weekly plan, scoped to the operator's own channel",
    "program": "the saved weekly plan, scoped to the operator's own channel",
    "break": "the live day plan for the operator's own channel, one addressable break",
    "advertiser": "the advertiser rules store",
    "agency": "the agencies store, and agency terms bite on the daily ledger's reporting net rather than the weekly plan",
    "event": "the calendar events store",
}


def _clip(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def parse_mentions(raw: Any) -> list[dict[str, str]]:
    """A validated list of references, or an empty list to degrade.

    Conservative in exactly the way the page-context parser is conservative:
    anything that is not the contract shape is dropped, and dropping everything
    means the ask behaves precisely as it does without the field. A duplicate
    reference collapses, because pointing at the same thing twice in one
    sentence is one thing.
    """
    if not isinstance(raw, list):
        return []
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        kind = _clip(item.get("type"), 40)
        ident = _clip(item.get("id"), ID_MAX)
        if kind not in words.KINDS or not ident:
            continue
        key = (kind, ident)
        if key in seen:
            continue
        seen.add(key)
        out.append({"type": kind, "id": ident, "label": _clip(item.get("label"), LABEL_MAX)})
        if len(out) >= REFS_CAP:
            break
    return out


# --- the card ----------------------------------------------------------------------
def _day_card(day_id: str) -> dict[str, Any]:
    """One broadcast day of the operator's own channel.

    This kind has no builder in the page-context table, and it is not added
    there: that table's entity types are a frozen contract with the dock. The
    day is read from the same scoped frame the rest of the plan spine is read
    from, which is the thing that actually matters.
    """
    from kairos_api import assistant_context

    frame, owned, _competitors, reason = assistant_context._owned_frame()
    if frame is None:
        return {"status": "unavailable", "reason": reason or "no owned-channel plan is available"}
    matched = frame[frame["date_text"].astype(str) == day_id]
    if matched.empty:
        return {"status": "not_found", "reason": "the saved plan holds no such day"}
    column = "program_title" if "program_title" in frame.columns else "program_type"
    programmes = sorted({str(value).strip() for value in matched[column] if str(value).strip()})
    return {
        "channel": owned,
        "date": day_id,
        "segments_total": int(len(matched)),
        "breaks": int(matched["num_breaks"].sum()),
        "revenue_ils": int(round(float(matched["predicted_revenue"].sum()))),
        "avg_retention_pct": assistant_context._retention_pct(matched["predicted_retention"].mean()),
        # Eager for the day itself, lazy for its contents: the names of what is
        # on it, and the rows themselves from get_day_detail if the model wants
        # them. A container card that inlined its children would spend the whole
        # budget on one mention.
        "programmes": programmes[:PROGRAMMES_CAP],
        "programmes_count": len(programmes),
        "contents_note": "read the day's own rows with get_day_detail; this card carries the day, not its breaks",
    }


def _break_card(break_id: str) -> dict[str, Any]:
    """One addressable break, rebuilt only inside the operator's own day plan."""
    from kairos_api import break_store
    from kairos_api.break_api_detail import _clock

    try:
        segment_id, _ordinal = break_store.parse_break_id(break_id)
    except ValueError:
        return {"status": "not_found"}
    day = segment_id.split("|", 1)[0]
    try:
        records = break_store.break_records(break_store.day_plan(day))
    except LookupError:
        return {"status": "not_found"}
    record = next((row for row in records if row.get("break_id") == break_id), None)
    if record is None:
        return {"status": "not_found"}
    return {
        "break_id": break_id,
        "day": record.get("day"),
        "channel": record.get("channel"),
        "programme": record.get("programme"),
        "start_clock": _clock(float(record.get("start_seconds") or 0)),
        "duration_seconds": record.get("duration_seconds"),
        "ordinal": record.get("ordinal"),
        "breaks_in_programme": record.get("breaks_in_segment"),
        "is_gold": record.get("is_gold"),
        "projected_revenue": record.get("projected_revenue"),
        "segment_retention": record.get("segment_retention"),
        "detail_tool": "get_break reads this break whole",
    }


def _build_card(kind: str, ident: str) -> dict[str, Any]:
    if kind == "day":
        return _day_card(ident)
    if kind == "break":
        return _break_card(ident)
    from kairos_api.assistant_page_context import _ENTITY_BUILDERS

    return _ENTITY_BUILDERS[kind](ident)


def _current_label(kind: str, ident: str, data: dict[str, Any]) -> str:
    """The name the STORE holds for this object right now.

    For every kind here the identifier is the store's own key, so the current
    name is the identifier unless a display name overrides it. The advertiser
    store is the one that carries a separate display name, and it is the one
    place the two can drift apart.
    """
    record = data.get("record") if isinstance(data.get("record"), dict) else None
    if record:
        for field in ("display_name", "name"):
            value = _clip(record.get(field), LABEL_MAX)
            if value:
                return value
    for field in ("name", "date", "start_clock"):
        value = _clip(data.get(field), LABEL_MAX)
        if value:
            return value
    return ident


def resolve_one(ref: dict[str, str]) -> dict[str, Any]:
    """One reference as its card, in one of the four states."""
    kind, ident = ref["type"], ref["id"]
    card: dict[str, Any] = {
        "type": kind,
        "type_label_he": words.label(kind, "he"),
        "type_label_en": words.label(kind, "en"),
        "id": ident,
        "label": ref.get("label") or ident,
    }
    try:
        data = _build_card(kind, ident)
    except Exception:  # noqa: BLE001 - an unreadable store is unavailable, never gone
        card["state"] = STATE_UNAVAILABLE
        card["reason"] = "the object's store could not be read"
        return card
    status = data.get("status") if isinstance(data, dict) else None
    if status == "not_found":
        # The reason the builder gives echoes the identifier back into a
        # sentence, and this route does not: a gone card's only variable field
        # is the identifier the caller itself sent, so a rival's object and a
        # typo produce the same bytes apart from what the caller typed.
        card["state"] = STATE_GONE
        card["reason"] = "the store was read and holds no object with this identifier"
        return card
    if status == "unavailable":
        card["state"] = STATE_UNAVAILABLE
        card["reason"] = "the object's store could not be read"
        return card

    current = _current_label(kind, ident, data)
    inserted = _clip(ref.get("label"), LABEL_MAX)
    # A day's label is rendered by the client from the ISO identifier, so the
    # two are different notations of one value and comparing them would report a
    # change on every single day mention. The kinds below carry the store's own
    # key as their label, so a difference there is a real difference.
    drifted = bool(kind != "day" and inserted and current and inserted != current)
    card["state"] = STATE_CHANGED if drifted else STATE_RESOLVED
    if drifted:
        card["current_label"] = current
        card["changed_note"] = ("the question carries the name this object had when it was "
                                "pointed at; the store now holds the name in current_label, "
                                "so quote the current one and say it changed")
    card["basis"] = BASIS.get(kind, "")
    card["data"] = data
    return card


def build_section(refs: list[dict[str, str]]) -> dict[str, Any]:
    from kairos_api import channel_scope

    objects = [resolve_one(ref) for ref in refs]
    return {
        "as_of": datetime.now(timezone.utc).isoformat(),
        "scope_channel": channel_scope.operator_channel() or None,
        "note": ("the operator pointed at these objects in the question itself. Each is "
                 "resolved at send time against the current saved state. A state other than "
                 "resolved means the reference did not come back clean, and it must be said "
                 "plainly rather than answered from the label in the question."),
        "states": [STATE_RESOLVED, STATE_CHANGED, STATE_GONE, STATE_UNAVAILABLE],
        "objects": objects,
    }


def public_states(section: dict[str, Any]) -> list[dict[str, str]]:
    """What the OPERATOR is shown about each reference, from the same resolution
    the model was given.

    This exists because of the finding against ``page_context``: it is well built
    and the binding is INVISIBLE, so nothing tells the operator that "it"
    resolved, or to what. A state the model can read and the operator cannot is
    the same defect one layer down. So the four states come back on the ask, the
    console prints them under the question, and the two never disagree because
    there is only one resolution.

    It carries no figures. The card the model reads holds the data; this holds
    the binding, which is the type, the identifier, the name the sentence used,
    the name the store holds when those differ, and the state. A ``gone`` row
    here has exactly the fields a ``gone`` card has, so nothing about a rival's
    object is disclosed by the shape of the answer either.
    """
    rows: list[dict[str, str]] = []
    for card in section.get("objects", []):
        row = {
            "type": str(card.get("type") or ""),
            "id": str(card.get("id") or ""),
            "label": str(card.get("label") or ""),
            "state": str(card.get("state") or ""),
            "kind_he": str(card.get("type_label_he") or ""),
            "kind_en": str(card.get("type_label_en") or ""),
            # The kind's navigational identity, the same key the picker row
            # carries, so the chip in the answer wears the glyph the chip in the
            # composer wore. Without it every kind would draw the same glyph,
            # which is worse than none: it would say the wrong thing quietly.
            "icon": words.icon(str(card.get("type") or "")),
            "state_he": words.state_label(str(card.get("state") or ""), "he"),
            "state_en": words.state_label(str(card.get("state") or ""), "en"),
        }
        if card.get("current_label"):
            row["current_label"] = str(card["current_label"])
        rows.append(row)
    return rows


def extend_with_mentioned_objects(context: dict[str, Any], sources: list[str], raw: Any) -> list[dict[str, str]]:
    """Attach the section when valid references arrived. Mutates in place.

    An absent or unparseable list adds nothing at all, which is today's
    behaviour; a list whose objects cannot be read still attaches, carrying the
    honest state of each, because the silent drop is the failure this whole
    module exists to prevent.

    Returns what the operator is shown, so the caller can put it on the answer.
    A resolution that itself failed returns an empty list rather than a guess.
    """
    refs = parse_mentions(raw)
    if not refs:
        return []
    try:
        section = build_section(refs)
        context[SECTION_NAME] = section
        sources.append(SECTION_NAME)
        return public_states(section)
    except Exception:  # noqa: BLE001 - the section is advisory, never fail the ask
        sources.append(f"{SECTION_NAME} (absent)")
        return []
