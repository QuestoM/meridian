"""The remedy, proposed: Kai can now act on the pacing problem it could read.

Kai gained the pacing board, one campaign's pacing and the decision ledger as
READ tools, and gained no way to record anything. ``get_campaign_pacing``'s own
description ended "Both are acts a person takes on the board; no tool records
them", which was true and is the gap this module closes.

**A proposal is not an act.** Nothing here writes. Each action is validated
through the same functions the routes validate with and captured as a pending
item by the machinery in :mod:`kairos_api.assistant_tools`; the ledger is
touched only when an operator approves it and
:mod:`kairos_api.assistant_actions` replays it. That is the same review-first
path the other eight propose tools take, extended rather than duplicated, and
:func:`register` extends the apply, restore and version machinery exactly as
:func:`kairos_api.assistant_propose_extra.register_action_plane` does.

**Kai cannot state a figure here, and that is the point.** The board's own write
path refuses a shortfall from its caller: a raise names a campaign, and the
goal, the counted value and the deficit are measured from the board at the
instant the row is written. So a proposal carries a campaign and a reason and
no number at all, and the number the ledger ends up holding is one this product
computed after the approval, not one a model produced before it. The single
figure a person may set is the OFFER, which the ledger already documents as a
human's number that reserves nothing, and a proposal carrying one says its value
and unit in the summary so the approval is against a stated amount.

**A proposal that would be refused is refused at capture.** The validator calls
``raisable_deficit``, ``acceptance_figures``, ``open_for`` and
``transition_allowed`` — the same four the routes call — so a raise Kai offers is
one the API would accept, and one it would not becomes a ``rejected`` item
carrying the board's own published Hebrew sentence rather than a pending item
that fails later in front of the operator.

**The competitor boundary rides the read.** Every action resolves its campaign
through ``pacing_alerts_api_read.board_payload``, which scopes through
``channel_scope``, so a campaign on any other channel is not found here for the
same reason it is not on the board.
"""

from __future__ import annotations

from typing import Any

from fastapi import HTTPException

ACTIONS = ("raise_make_good", "accept_risk", "move_make_good")

# The item kind, and the one logical file it mutates. Registered into the
# version store's vocabulary and its path resolver by register() below, because
# a kind whose logical name that store does not know raises rather than being
# silently versioned nowhere.
KIND = "pacing_decision"
LOGICAL = "make_goods"

_REASON = {
    "type": "string",
    "description": "Why this decision is being proposed, in the operator's language.",
}


def _detail(exc: HTTPException) -> str:
    """The Hebrew half of a pacing refusal, which is what the operator reads.

    The routes on this spine refuse with ``{message_en, message_he}`` rather than
    a string, so a validator that stringified the detail would put a Python dict
    repr on a proposal card. Falls back to the English half and then to the raw
    detail, so an unexpected shape still says something true.
    """
    detail = exc.detail
    if isinstance(detail, dict):
        return str(detail.get("message_he") or detail.get("message_en") or detail)
    return str(detail)


def _board_row(campaign_id: str) -> tuple[dict[str, Any], dict[str, Any], Any]:
    """The board, one campaign's row on it, and the day it was counted at.

    Raises ValueError with the board's own sentence when the operator's channel
    carries no such campaign, which is the same refusal the route sends.
    """
    from kairos_api import pacing_alerts_api_board as board
    from kairos_api import pacing_alerts_api_read as read
    from kairos_api import pacing_alerts_api_wire as wire

    if not campaign_id:
        raise ValueError("campaign_id is required")
    view = wire.expand(read.board_payload())
    row = read.find_row(view, campaign_id)
    if row is None:
        raise ValueError(read.UNKNOWN_CAMPAIGN_HE)
    return view, row, board.parse_date(view.get("as_of", {}).get("instant"))


def _no_open_record(campaign_id: str, kind: str, duplicate_he: str) -> None:
    from kairos_api import makegood_store as ledger

    if ledger.open_for(ledger.load_frame(), campaign_id, kind):
        raise ValueError(duplicate_he)


def _validate_raise(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api import makegood_store as ledger
    from kairos_api import pacing_alerts_api as api
    from kairos_api import pacing_alerts_api_read as read

    campaign_id = str(args.get("campaign_id", "") or "").strip()
    view, row, as_of = _board_row(campaign_id)
    deficit, why = read.raisable_deficit(row, as_of)
    if deficit is None:
        raise ValueError(api.REFUSED_RAISE.get(why, (read.NOTHING_TO_RAISE_EN, read.NOTHING_TO_RAISE_HE))[1])
    _no_open_record(campaign_id, ledger.MAKE_GOOD, api.DUPLICATE_HE)
    note = str(args.get("note", "") or "").strip()
    summary = (f"החלטה: פתיחת פיצוי שידור מול {row['name']} ({campaign_id}), "
               f"חוסר נמדד {deficit['deficit_value']} {deficit['unit']} ({deficit['deficit_kind']}); "
               "הנתונים נמדדים מהלוח ברגע האישור ואינם נלקחים מההצעה")
    return {"action": "raise_make_good", "campaign_id": campaign_id, "note": note}, summary


def _validate_accept(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api import makegood_store as ledger
    from kairos_api import pacing_alerts_api as api
    from kairos_api import pacing_alerts_api_read as read
    from kairos_api import pacing_alerts_api_words as words

    campaign_id = str(args.get("campaign_id", "") or "").strip()
    view, row, as_of = _board_row(campaign_id)
    if read.acceptance_figures(row, as_of) is None:
        raise ValueError(words.ACCEPT_NOT_AT_RISK_HE)
    _no_open_record(campaign_id, ledger.ACCEPTANCE, api.DUPLICATE_ACCEPT_HE)
    note = str(args.get("note", "") or "").strip()
    # "בקמפיין" and not a bare "ב" prefix: a campaign name here begins with its
    # year on the shipped data, and a one-letter prefix glued to a digit reads as
    # a typo to the only people who will read it.
    summary = (f"החלטה: הסיכון בקמפיין {row['name']} ({campaign_id}) נשאר כפי שהוא; "
               "הרישום אינו משנה נתון ואינו שומר מלאי")
    return {"action": "accept_risk", "campaign_id": campaign_id, "note": note}, summary


def _move_payload(args: dict[str, Any]) -> Any:
    """The move as the route's own model, so the validator checks what it checks."""
    from kairos_api.pacing_alerts_api import MoveMakeGood

    try:
        return MoveMakeGood(
            state=str(args.get("state", "") or "").strip(),
            offer_value=args.get("offer_value"),
            offer_window_start=str(args.get("offer_window_start", "") or ""),
            offer_window_end=str(args.get("offer_window_end", "") or ""),
            reason=str(args.get("close_reason", "") or ""),
            note=str(args.get("note", "") or ""),
        )
    except Exception as exc:  # noqa: BLE001 - a bad shape is an honest rejection
        raise ValueError(f"invalid move: {str(exc)[:200]}") from exc


def _validate_move(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    from kairos_api import makegood_store as ledger
    from kairos_api import pacing_alerts_api as api
    from kairos_api import pacing_alerts_api_write as write

    make_good_id = str(args.get("make_good_id", "") or "").strip()
    if not make_good_id:
        raise ValueError("make_good_id is required to move a ledger record")
    frame = ledger.load_frame()
    index = ledger.locate(frame, make_good_id)
    if index < 0:
        raise ValueError(api.UNKNOWN_MAKE_GOOD_HE)
    payload = _move_payload(args)
    current = str(frame.at[index, "state"] or ledger.RAISED)
    kind = str(frame.at[index, "kind"] or ledger.MAKE_GOOD)
    if not ledger.transition_allowed(current, payload.state):
        raise ValueError(_detail(write.refuse_transition(current, payload.state, kind)))
    # The same two checks apply_move runs before it writes, run here so a move
    # that would be refused never becomes a pending item. Neither one writes.
    try:
        write.check_reason(payload.state, payload.reason, payload.note.strip())
    except HTTPException as exc:
        raise ValueError(_detail(exc)) from exc
    _check_offer(payload, frame, index, ledger, write)
    record = ledger.record(frame.loc[index])
    _, want_he = write.state_words(payload.state)
    money = ""
    if payload.state == ledger.OFFERED and payload.offer_value is not None:
        money = f", הצעה {round(float(payload.offer_value), 2)} {record['shortfall']['unit']}"
    summary = (f"החלטה: העברת {make_good_id} ({record['campaign_name']}) למצב {want_he}{money}")
    return {"action": "move_make_good", "make_good_id": make_good_id,
            "move": _move_fields(payload)}, summary


def _check_offer(payload: Any, frame: Any, index: int, ledger: Any, write: Any) -> None:
    """The offer rules the writer enforces, checked without touching the frame."""
    if payload.state == ledger.OFFERED:
        value = payload.offer_value
        if value is None or float(value) <= 0:
            raise ValueError(write.OFFER_VALUE_HE)
        start, end = payload.offer_window_start.strip(), payload.offer_window_end.strip()
        if start and end:
            from kairos_api import pacing_alerts_api_board as board

            parsed_start, parsed_end = board.parse_date(start), board.parse_date(end)
            if parsed_start and parsed_end and parsed_end < parsed_start:
                raise ValueError(write.OFFER_ORDER_HE)
    elif payload.state in ledger.NEEDS_OFFER:
        if not str(frame.at[index, "offer_value"] or "").strip():
            raise ValueError(write.NEEDS_OFFER_HE)


def _move_fields(payload: Any) -> dict[str, Any]:
    return {"state": payload.state, "offer_value": payload.offer_value,
            "offer_window_start": payload.offer_window_start,
            "offer_window_end": payload.offer_window_end,
            "reason": payload.reason, "note": payload.note}


_VALIDATORS = {"raise_make_good": _validate_raise, "accept_risk": _validate_accept,
               "move_make_good": _validate_move}


def validate_pacing_decision(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
    action = str(args.get("action", "") or "").strip().lower()
    if action not in ACTIONS:
        raise ValueError(f"action must be one of {list(ACTIONS)}, got {action!r}")
    return _VALIDATORS[action](args)


def apply_pacing_decision(payload: dict[str, Any], actor: str) -> dict[str, Any]:
    """Replay one approved decision through the pacing module's own writers.

    ``actor`` is the account that approved it, which is what the ledger records.
    The two private writers are called rather than the route functions because a
    route derives its actor from an HTTP session there is none of here, and a
    ledger row naming nobody is the defect this parameter exists to prevent.
    """
    from kairos_api import makegood_store as ledger
    from kairos_api import pacing_alerts_api as api

    action = str(payload.get("action") or "")
    if action == "move_make_good":
        move = api.MoveMakeGood(**dict(payload.get("move") or {}))
        result = api._move_decision(str(payload.get("make_good_id") or ""), move, actor)
        return {"action": action, "make_good_id": result["make_good"]["make_good_id"],
                "state": result["make_good"]["state"]}
    kind = ledger.MAKE_GOOD if action == "raise_make_good" else ledger.ACCEPTANCE
    result = api._write_decision(kind, str(payload.get("campaign_id") or ""),
                                 str(payload.get("note") or ""), actor)
    return {"action": action, "make_good_id": result["make_good"]["make_good_id"],
            "state": result["make_good"]["state"]}


def pending_index() -> dict[str, list[dict[str, Any]]]:
    """Which campaigns carry a pacing decision that is proposed and not yet approved.

    A pending proposal was visible only inside the assistant dock, so an account
    manager could open the board, see a row offering a raise, and have no way to
    know that a raise on it was already recorded and waiting for somebody's
    approval. The apply path would refuse the second one as a duplicate, which is
    honest and far too late: by then two people have decided the same thing.

    So the board publishes the same shape it already publishes for open ledger
    records. It is keyed by campaign id, and each entry names the batch and item
    an approver has to open. **It is not a ledger entry and must never render as
    one**: the count of what is owed comes from ``make_goods``, and a proposal
    has changed nothing.

    Tolerant by construction. The proposal store is the assistant's runtime state
    and a board must not fail to load because it is missing or half-written, so
    an unreadable store yields an empty index rather than an error; the honest
    consequence is only that a row offers a control the apply path will refuse.
    """
    from kairos_api import assistant_actions

    index: dict[str, list[dict[str, Any]]] = {}
    try:
        with assistant_actions._LOCK:
            store = assistant_actions._load_store()
        batches = store.get("batches") or []
    except Exception:  # noqa: BLE001 - the board outranks the index; see the docstring
        return index
    for batch in batches:
        for item in batch.get("items") or []:
            if item.get("kind") != KIND or item.get("status") != "pending":
                continue
            payload = item.get("payload") or {}
            campaign_id = str(payload.get("campaign_id") or "")
            if not campaign_id:
                continue
            index.setdefault(campaign_id, []).append({
                "batch_id": batch.get("batch_id"), "item_id": item.get("id"),
                "action": payload.get("action"), "proposed_by": batch.get("created_by"),
                "proposed_at": batch.get("created_at"),
            })
    return index


PACING_PROPOSE_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "propose_pacing_decision",
        "description": (
            "Propose one of the three decisions the pacing board records: raise a make-good "
            "against a campaign's measured shortfall (raise_make_good), record that the risk "
            "on a campaign stands as it is (accept_risk), or move an existing ledger record "
            "to its next state (move_make_good, for example to offered, settled, declined or "
            "withdrawn). You never supply the shortfall: the goal, the counted value and the "
            "deficit are measured from the board at the instant the approved decision is "
            "written, so name the campaign and the reason and no figure. The one figure a "
            "decision may carry is an offer_value when moving to offered, which is a human's "
            "number in the shortfall's own unit and reserves no inventory. A decision the "
            "board would refuse is rejected here with the board's own reason, so read "
            "get_campaign_pacing first and propose only what its remedy block says is "
            "available. The operator must approve before anything is written to the ledger."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "description": "'raise_make_good', 'accept_risk' or 'move_make_good'."},
                "campaign_id": {"type": "string", "description": "For raise_make_good and accept_risk: the campaign id from get_pacing_board."},
                "make_good_id": {"type": "string", "description": "For move_make_good: the ledger record id from get_make_good_ledger, for example MG_0007."},
                "state": {"type": "string", "description": "For move_make_good: the target state. get_make_good_ledger publishes next_states for each record."},
                "offer_value": {"type": "number", "description": "For a move to offered: the offer's value in the shortfall's own unit, above zero."},
                "offer_window_start": {"type": "string", "description": "For a move to offered: ISO date the offer window opens."},
                "offer_window_end": {"type": "string", "description": "For a move to offered: ISO date the offer window closes."},
                "close_reason": {"type": "string", "description": "For a move that closes a record without a delivery: one of the ledger's published close reasons."},
                "note": {"type": "string", "description": "The note recorded on the ledger row itself, in the operator's language."},
                "reason": _REASON,
            },
            "required": ["action", "reason"],
        },
    },
]


def register() -> None:
    """Extend the propose registries, the apply engine and both timelines.

    Idempotent, and additive at every point. Only the APPLY-side wiring is here:
    the schema is registered at import by assistant_tool_schemas and the
    validator at import by assistant_propose_tools, because both are needed to
    capture a proposal and capture does not require the router. The applier and
    the logical file are apply-side, and applying always goes through the router
    this call is made from.
    """
    from kairos_api import assistant_actions, version_store

    assistant_actions._APPLIERS[KIND] = apply_pacing_decision
    version_store._LOGICAL_FOR_KIND[KIND] = LOGICAL
