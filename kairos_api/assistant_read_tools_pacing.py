"""READ tool executors for pacing against goal, and the remedy when it is behind.

The ``campaign_manager`` persona is told to answer "in terms of campaigns,
flights and pacing against goal, and name the remedy when something is behind".
The only tool it had was ``get_make_good_alerts``, the older projection over
``campaign_flights.csv``, which is a header-only seed and answers
``data_available: false``. So Kai could name a remedy and could not read the thing
the remedy is for: the pacing board, the days behind one campaign's figures, and
the ledger of what has actually been decided.

All three tools call the pacing routes' own functions in
:mod:`kairos_api.pacing_alerts_api_read` and :mod:`kairos_api.pacing_alerts_api`
rather than the two stores underneath them, so the tool and the board answer one
campaign the same way. Four rules carry through from those modules, and the money
on these surfaces is why they matter more here than they did for the pod.

**A refusal to state a pace is passed through with its own reason.** The board
refuses in four separate places, and each refusal sends a reader somewhere
different: the flight has not started, no broadcast day of it carries a per-spot
source at all, some elapsed day is missing so the figure to date is a floor
rather than a total, or the campaign carries no goal or an unmeasurable one.
Nothing here flattens those into one word, and none of them becomes a zero.

**Counted is not delivered.** The counted figures are the planned break rating
the traffic log carries, engine-priced per spot; nothing is invoiced and nothing
is a post-campaign panel report. That sentence rides every payload in the board's
own words rather than being restated here.

**What may be raised is not what is measured.** A gap against the pace reference
is a measured figure and is not a debt. The remedy block calls
``raisable_deficit``, the function the write path itself calls, and refuses in
the API's own sentences, so a raise Kai offers is one the API would accept.

**The competitor boundary is applied at the read.** ``board_payload`` and
``days_payload`` both scope through ``channel_scope``, so a campaign on any other
channel never reaches a row, a day, a count or a make-good, and the ``scope``
block states what it scoped. ``tests/test_assistant_pacing_tools.py`` proves that
with a positive control rather than trusting this paragraph.

Split into its own module under the size cap, beside
kairos_api.assistant_read_tools_break, and registered by
kairos_api.assistant_read_tools_catalog with every other read tool.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

MAX_ROWS = 30
DEFAULT_ROWS = 15
MAX_DAYS = 14
MAX_DAY_ROWS = 45
MAX_DECISIONS = 25
MAX_LIST_ROWS = 20

# The four strings a pace, forward or headline block carries. A block that states
# a verdict carries them empty, and an empty string is not a reason.
PROSE = ("reason_en", "reason_he", "path_forward_en", "path_forward_he")
DAY_LISTS = ("unsourced_elapsed_days", "unsourced_remaining_days")


def _cap(payload: dict[str, Any], key: str, limit: int) -> None:
    """Cap ``payload[key]`` to ``limit`` rows in place, recording the overflow."""
    rows = list(payload.get(key) or [])
    payload[key] = rows[:limit]
    if len(rows) > limit:
        payload[f"{key}_total"] = len(rows)
        payload[f"{key}_omitted"] = len(rows) - limit


def _block(raw: Any) -> Any:
    """One pace, forward or headline block: its verdict, and its reason when it refuses.

    The four prose keys are dropped only when they are empty, which is the
    payload's own way of saying there is nothing to explain because a verdict was
    stated. A block that refuses keeps every word of why, in both languages, and
    keeps the days it names.
    """
    if not isinstance(raw, dict):
        return raw
    block = {key: value for key, value in raw.items()
             if key not in PROSE or str(value or "").strip()}
    for key in DAY_LISTS:
        if isinstance(block.get(key), list):
            _cap(block, key, MAX_DAYS)
    return block


def _line(raw: Any) -> Any:
    """One goal line whole: the counted figures, the reference, the pace, the forward.

    The even-share reference rule is one sentence and it is written on both goal
    lines of every row, so it rides the payload once instead, exactly as the
    board's own wire collapse does with it. Nothing else is lifted off a line.
    """
    if not isinstance(raw, dict):
        return raw
    line = dict(raw)
    reference = line.get("reference")
    if isinstance(reference, dict):
        line["reference"] = {key: value for key, value in reference.items()
                             if key not in ("rule_en", "rule_he")}
    line["pace"] = _block(line.get("pace"))
    line["forward"] = _block(line.get("forward"))
    return line


def _line_summary(raw: Any) -> Any:
    """One goal line as a board-scan row: the figures, the verdict, the reason.

    Every figure is the board's own, passed exactly as it computed it; the line
    whole rides ``get_campaign_pacing``, which is where the reference block's own
    prose belongs.
    """
    if not isinstance(raw, dict):
        return raw
    reference = raw.get("reference")
    summary = {
        "unit": raw.get("unit"),
        "goal": raw.get("goal"),
        "counted": raw.get("counted"),
        "expected_through_counted_day": (
            reference.get("expected_through_counted_day") if isinstance(reference, dict) else None),
        "pace": _block(raw.get("pace")),
        "forward": _block(raw.get("forward")),
    }
    if raw.get("audience"):
        summary["audience"] = raw["audience"]
    return summary


def _flight(raw: Any) -> Any:
    if not isinstance(raw, dict):
        return raw
    flight = dict(raw)
    for key in DAY_LISTS:
        _cap(flight, key, MAX_DAYS)
    return flight


def _row_digest(row: dict[str, Any], make_goods: dict[str, list[str]],
                acceptances: dict[str, list[str]]) -> dict[str, Any]:
    """One campaign as a board row: who it is, its flight, and both goal lines."""
    campaign_id = row.get("campaign_id", "")
    return {
        "campaign_id": campaign_id,
        "name": row.get("name"),
        "advertiser": row.get("advertiser"),
        "agency_id": row.get("agency_id"),
        "status": row.get("status"),
        "goal_kind": row.get("goal_kind"),
        "is_demo": row.get("is_demo"),
        "days_available": row.get("days_available"),
        "flight": _flight(row.get("flight")),
        "headline": _block(row.get("headline")),
        "rating": _line_summary(row.get("rating")),
        "money": _line_summary(row.get("money")),
        # Which endings this campaign already carries, so an answer never offers
        # a raise on a campaign that has an open one.
        "open_make_goods": make_goods.get(campaign_id, []),
        "open_risk_acceptances": acceptances.get(campaign_id, []),
    }


def _expanded_board() -> dict[str, Any]:
    """The board in the full shape, which is the shape a write measures from.

    ``board_payload`` collapses the repeated prose for the wire and the surface
    expands it back the instant it lands. The model has no such expander, so the
    board's own inverse is called here and every row arrives carrying its own
    words. This is also the exact call ``_write_decision`` makes, so what Kai
    reads is what an act would be decided on.
    """
    from kairos_api import pacing_alerts_api_read as read
    from kairos_api import pacing_alerts_api_wire as wire

    return wire.expand(read.board_payload())


def _scan_board() -> dict[str, Any]:
    """The board for a 52-row scan: prose published once, day lists back on the rows.

    The board's own wire form already publishes each reason once, keyed by the
    code or the state the row carries, and a block whose words differ from the
    published ones keeps its own. Nothing is lost by it and it is 72 KB smaller
    than the expanded form, so the scan reads it as it is served.

    One thing in that form is a hazard here and only here: a day list identical
    to the flight's collapses to an explicit null, and a model reading
    ``unsourced_remaining_days: null`` would take it for no missing days, which is
    the opposite of what it says. So the wire's own per-row inverse for exactly
    those two lists is applied, and nothing else is.
    """
    from kairos_api import pacing_alerts_api_read as read
    from kairos_api import pacing_alerts_api_wire as wire

    payload = read.board_payload()
    for row in payload.get("rows", []):
        wire._expand_days(row)
    return payload


def _open_records() -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """The open make-goods and the open risk acceptances, keyed by campaign."""
    from kairos_api import makegood_store as ledger
    from kairos_api.pacing_alerts_api import _open_index

    frame = ledger.load_frame()
    return _open_index(frame, ledger.MAKE_GOOD), _open_index(frame, ledger.ACCEPTANCE)


def _read_get_pacing_board(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import pacing_alerts_api_words as words

    verdict = str(args.get("verdict", "") or "").strip().lower()
    allowed = (words.BEHIND, words.AT_RISK, words.ON_PACE, words.UNKNOWN)
    if verdict and verdict not in allowed:
        return {"error": f"verdict must be one of {list(allowed)}, got {verdict!r}"}
    try:
        limit = int(args.get("limit") or DEFAULT_ROWS)
    except (TypeError, ValueError):
        limit = DEFAULT_ROWS
    limit = max(1, min(limit, MAX_ROWS))

    payload = _scan_board()
    payload.pop("source", None)
    # The wire marker is bookkeeping for a client that expands, and the
    # vocabulary is the label table a screen renders chips from. Neither is a
    # figure or a reason, and both are 2 KB the answer cannot use.
    for key in ("wire", "vocabulary"):
        payload.pop(key, None)
    make_goods, acceptances = _open_records()
    rows = [row for row in payload.get("rows", [])
            if not verdict or (row.get("headline") or {}).get("verdict") == verdict]
    payload["rows_matching"] = len(rows)
    payload["rows"] = [_row_digest(row, make_goods, acceptances) for row in rows[:limit]]
    if len(rows) > limit:
        payload["rows_omitted"] = len(rows) - limit
    if verdict:
        payload["filtered_by"] = {"verdict": verdict}
    payload["needs_a_decision"] = list(words.NEEDS_A_DECISION)
    payload["open_make_good_campaigns"] = len(make_goods)
    payload["open_risk_acceptance_campaigns"] = len(acceptances)
    payload["order"] = "worst pacing first: behind, then at risk, then unknown with a reason, then on pace"
    payload["how_to_read_a_reason"] = (
        "a headline or pace block that could not state a verdict names a code, and a forward "
        "block names a state; the reason and the path forward for each are published once in "
        "reasons and forward_reasons. A block that carries its own words carries them inline "
        "instead. get_campaign_pacing returns one campaign with every word already in place."
    )
    payload["detail_tool"] = "get_campaign_pacing reads one campaign whole, with the broadcast days behind its figures and whether a make-good may be raised"
    payload["ledger_tool"] = "get_make_good_ledger reads what has actually been decided"
    return payload


def _remedy(row: dict[str, Any], as_of_day: Any) -> dict[str, Any]:
    """Whether a make-good may be raised on this campaign, in the product's own words.

    The rule for what is owed is not restated here. ``raisable_deficit`` is the
    same function the write path calls and ``REFUSED_RAISE`` holds the same two
    sentences the API refuses with, so a raise this block offers is one the API
    would accept, and one it will not is refused in the board's own words.
    """
    from kairos_api import pacing_alerts_api_read as read
    from kairos_api import pacing_alerts_api_words as words
    from kairos_api.pacing_alerts_api import REFUSED_RAISE

    deficit, why = read.raisable_deficit(row, as_of_day)
    accepted = read.acceptance_figures(row, as_of_day)
    block: dict[str, Any] = {
        "make_good_can_be_raised": deficit is not None,
        "owed_deficit": deficit,
        "risk_can_be_taken_on": accepted is not None,
        # What an acceptance would stamp: the measured state at the counted day,
        # which is a figure and not a debt. The two are deliberately separate.
        "measured_figures_an_acceptance_would_record": accepted,
        "rule_en": words.RAISE_RULE_EN,
        "rule_he": words.RAISE_RULE_HE,
    }
    if deficit is None:
        refusal = REFUSED_RAISE.get(why, (read.NOTHING_TO_RAISE_EN, read.NOTHING_TO_RAISE_HE))
        block["why_not"] = why
        block["reason_en"] = refusal[0]
        block["reason_he"] = refusal[1]
    if accepted is None:
        block["acceptance_reason_en"] = words.ACCEPT_NOT_AT_RISK_EN
        block["acceptance_reason_he"] = words.ACCEPT_NOT_AT_RISK_HE
    # Neither act is a tool. Kai reads the board and the person acts on it, so
    # the answer must not read as though the model could raise anything.
    block["how_it_is_recorded"] = "both are acts a person takes on the pacing board; no assistant tool raises a make-good or takes a risk on"
    return block


def _read_get_campaign_pacing(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import makegood_store as ledger
    from kairos_api import pacing_alerts_api_board as board
    from kairos_api import pacing_alerts_api_read as read

    campaign_id = str(args.get("campaign_id", "") or "").strip()
    if not campaign_id:
        return {"error": "provide the campaign id; list the campaigns with get_pacing_board"}
    view = _expanded_board()
    row = read.find_row(view, campaign_id)
    if row is None:
        # The board's own refusal. A campaign on another channel is not a
        # different error from one that does not exist, and it must not be:
        # telling the two apart would disclose that a rival's campaign exists.
        return {"error": read.UNKNOWN_CAMPAIGN_EN, "reason_he": read.UNKNOWN_CAMPAIGN_HE,
                "campaign_id": campaign_id,
                "campaigns": [str(one.get("campaign_id", "")) for one in view.get("rows", [])][:MAX_LIST_ROWS]}
    as_of_day = board.parse_date((view.get("as_of") or {}).get("instant"))
    payload: dict[str, Any] = {key: value for key, value in row.items()
                               if key not in ("rating", "money", "flight", "headline")}
    payload["flight"] = _flight(row.get("flight"))
    payload["headline"] = _block(row.get("headline"))
    payload["rating"] = _line(row.get("rating"))
    payload["money"] = _line(row.get("money"))
    payload["as_of"] = view.get("as_of")
    payload["scope"] = view.get("scope")
    payload["trigger"] = view.get("trigger")
    payload["reference_rule"] = view.get("reference_rule")
    payload["counted_is_planned_en"] = view.get("counted_is_planned_en")
    payload["counted_is_planned_he"] = view.get("counted_is_planned_he")
    payload["remedy"] = _remedy(row, as_of_day)

    days = read.days_payload(campaign_id) or {}
    # A flight day with no per-spot source is a row that says so, not a missing
    # row and not a zero, so the day list passes through exactly as the drill
    # serves it and the count beside it stays the true one.
    detail = {"days": list(days.get("days") or []), "count": days.get("count"),
              "sources": (days.get("sources") or [])[:MAX_LIST_ROWS],
              "booking_rules": days.get("booking_rules") or {}}
    _cap(detail, "days", MAX_DAY_ROWS)
    payload["broadcast_days"] = detail

    records = [record for record in ledger.records(ledger.load_frame())
               if record.get("campaign_id") == campaign_id]
    payload["decisions"] = records[:MAX_DECISIONS]
    payload["decisions_count"] = len(records)
    return payload


def _read_get_make_good_ledger(args: dict[str, Any], user: str | None = None) -> dict[str, Any]:
    from kairos_api import makegood_store as ledger
    from kairos_api import pacing_alerts_api_read as read

    payload = dict(read.ledger_payload(ledger.load_frame()))
    payload.pop("source", None)
    for key in ("decisions", "make_goods", "acceptances"):
        _cap(payload, key, MAX_DECISIONS)
    if not payload.get("count"):
        # An empty ledger is a real state and it is the state on disk today: no
        # make-good and no risk acceptance has been recorded yet. It is not the
        # same fact as a campaign having nothing owed, and it is not a zero
        # anybody measured, so it says which it is.
        payload["note"] = "no make-good and no risk acceptance has been recorded yet; this is an empty ledger, not a finding that nothing is owed"
    payload["board_tool"] = "get_pacing_board reads which campaigns are behind, and get_campaign_pacing whether a make-good may be raised on one"
    return payload


_PACING_READ_EXECUTORS = {
    "get_pacing_board": _read_get_pacing_board,
    "get_campaign_pacing": _read_get_campaign_pacing,
    "get_make_good_ledger": _read_get_make_good_ledger,
}

# Provenance stamps, same vocabulary as SOURCE_BY_TOOL in assistant_read_tools:
# the dataset each figure came from, surfaced on the run trace. The first two
# name both stores, because a pacing figure is a campaign's booked goal measured
# against the delivery ledger and neither store alone produces it.
PACING_SOURCE_BY_TOOL = {
    "get_pacing_board": "the pacing board: the campaign store and the delivery ledger, owned channel",
    "get_campaign_pacing": "the pacing board and the delivery ledger, the broadcast days behind one campaign",
    "get_make_good_ledger": "the make-good decision ledger",
}

# The schemas live beside their executors rather than in the schema module, so a
# description cannot drift from what the executor returns.
# kairos_api.assistant_tool_schemas extends its own list with these before
# READ_TOOL_NAMES freezes, so the model still sees one flat tool list.
PACING_READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_pacing_board",
        "description": (
            "Read the pacing board: every campaign on the operator's channel paced against "
            "its booked goal, worst first. Each row carries the flight window with the "
            "broadcast days that carry no per-spot source, both goal lines (rating points and "
            "money) with what was counted, what an even share of the goal would be by the "
            "counted day, the pace verdict and ratio, the gap to the reference, and what the "
            "rest of the flight says. A pace that cannot be stated says why in words rather "
            "than as a number: the flight has not started, no day carries a source, an "
            "elapsed day is missing, or there is no goal in that unit. Counted means the "
            "planned break rating the traffic log holds, engine-priced; nothing is invoiced. "
            "Optional verdict filters to behind, at_risk, on_pace or unknown, and limit sets "
            "how many rows. Call this when the campaign manager asks which campaigns are "
            "behind, then get_campaign_pacing for one."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "verdict": {"type": "string", "description": "Only rows with this headline verdict: behind, at_risk, on_pace or unknown."},
                "limit": {"type": "integer", "description": "How many rows to return (1-30, default 15). The board is ordered worst first."},
            },
        },
    },
    {
        "name": "get_campaign_pacing",
        "description": (
            "Read one campaign's pacing whole: both goal lines with every counted figure, the "
            "reference, the verdict and the named reason when there is not one, plus every "
            "broadcast day of the flight as its own row with what aired, what is booked and "
            "not yet aired, and which days carry no per-spot source at all and are therefore "
            "unknown rather than empty. It also carries the remedy: whether a make-good may "
            "actually be raised against a shortfall that is owed, or the reason it may not "
            "yet, whether the risk may instead be taken on, and the decisions already recorded "
            "against this campaign. Both are acts a person takes on the board, and propose_pacing_decision "
            "records either one for their approval; read this first and propose only what the remedy block "
            "says is available. Call this when the campaign manager asks about one campaign, why it is behind, or what can be done about it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"campaign_id": {"type": "string", "description": "The campaign id from get_pacing_board, for example CMP_D040."}},
            "required": ["campaign_id"],
        },
    },
    {
        "name": "get_make_good_ledger",
        "description": (
            "Read the decision ledger behind the pacing board: every make-good raised against "
            "a measured shortfall and every recorded decision to take the risk on as it "
            "stands, with the figures each was measured from, its state, any offer and its "
            "window, and who acted and when. It also carries what this product was never "
            "given: the commercial rule for what a make-good may be offered against and who "
            "signs one off, so it records who acted and refuses to derive an entitlement. An "
            "empty ledger means nothing has been recorded yet, which is not a finding that "
            "nothing is owed. Call this when the campaign manager asks what make-goods are "
            "open, what was offered, or what was decided."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
]


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    """Merge these executors and their source labels into the shared registry."""
    executors.update(_PACING_READ_EXECUTORS)
    sources.update(PACING_SOURCE_BY_TOOL)
