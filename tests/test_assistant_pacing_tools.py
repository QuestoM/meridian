"""The assistant's three pacing read tools, against the real stores on disk.

The campaign-manager persona is told to answer in campaigns, flights and pacing
against goal, and to name the remedy when something is behind. Until these tools
existed the only pacing tool was ``get_make_good_alerts``, the older projection
over a header-only seed, so Kai could name a remedy and could not read the thing
the remedy is for. These tests hold the three new tools to the surface they
report:

* every figure is the board's own, never re-rounded and never re-derived here,
* each of the board's separate refusals to state a pace arrives with its own
  reason rather than flattened into one word or into a zero,
* a day with no per-spot source is unknown and is never counted as no delivery,
* what may be raised is exactly what the write path would accept,
* and no rival channel name, and no rival campaign, reaches any of the three.

Nothing is mocked: the executors run their real code against the campaign store,
the delivery ledger and the make-good ledger.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kairos_api import assistant_tools as tools
from kairos_api import channel_scope
from kairos_api import pacing_alerts_api_board as board
from kairos_api import pacing_alerts_api_read as read
from kairos_api import pacing_alerts_api_words as words
from kairos_api.assistant_read_tools import SOURCE_BY_TOOL, _READ_EXECUTORS, execute_read_tool

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"
PACING_TOOLS = ("get_pacing_board", "get_campaign_pacing", "get_make_good_ledger")
UNKNOWN_CAMPAIGN = "CMP_DOES_NOT_EXIST"


def _board() -> dict:
    payload = execute_read_tool("get_pacing_board", {"limit": 30}, None)
    if not payload.get("rows"):
        pytest.skip("no campaign on the operator channel, so there is no board to read")
    return payload


def _rival_channels() -> set[str]:
    """Every channel in the saved weekly plan that the operator does not own."""
    owned = channel_scope.operator_channel()
    frame = pd.read_csv(PLAN_PATH)
    channels = {str(value).strip() for value in frame["channel"].dropna().unique()}
    return {name for name in channels if name and name != owned}


def _names_found(payload: object, names: set[str]) -> set[str]:
    """Which of these names appear anywhere in a payload, serialized as the model
    would see it. Substring, not equality: a rival hidden inside a campaign name
    or a reason would still be a rival reaching the assistant's context."""
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    return {name for name in names if name in blob}


# --- the registry -----------------------------------------------------------------
def test_all_three_pacing_tools_are_registered_with_a_named_source() -> None:
    for name in PACING_TOOLS:
        assert name in tools.READ_TOOL_NAMES, name
        assert name in _READ_EXECUTORS, name
        # A tool whose source is not named is a tool whose answer cannot be
        # checked, so the catalogue entry must name the data behind it.
        assert "ledger" in SOURCE_BY_TOOL[name], name
    assert not (set(PACING_TOOLS) & set(tools.PROPOSE_TOOL_NAMES))
    # The older projection stays: Bar 3 forbids removing a working answer, and it
    # is the signal the optimizer's own pacing weights read.
    assert "get_make_good_alerts" in tools.READ_TOOL_NAMES


def test_the_pacing_schemas_describe_the_board_the_drill_and_the_ledger() -> None:
    by_name = {schema["name"]: schema for schema in tools.READ_TOOL_SCHEMAS}
    listing = by_name["get_pacing_board"]
    assert set(listing["input_schema"]["properties"]) == {"verdict", "limit"}
    assert not listing["input_schema"].get("required")
    assert by_name["get_campaign_pacing"]["input_schema"]["required"] == ["campaign_id"]
    assert not by_name["get_make_good_ledger"]["input_schema"]["properties"]
    for name in PACING_TOOLS:
        text = by_name[name]["description"]
        assert "campaign" in text or "make-good" in text


# --- the board: the figures arrive intact -----------------------------------------
def test_the_board_returns_the_boards_own_figures_worst_first() -> None:
    payload = _board()
    truth = read.board_payload()
    assert payload["available"] is True
    assert payload["counts"] == truth["counts"]
    assert payload["rows_matching"] == len(truth["rows"])
    by_id = {row["campaign_id"]: row for row in truth["rows"]}
    for row in payload["rows"]:
        source = by_id[row["campaign_id"]]
        assert row["headline"]["verdict"] == source["headline"]["verdict"]
        assert row["headline"].get("ratio") == source["headline"].get("ratio")
        for unit in ("rating", "money"):
            line, origin = row.get(unit), source.get(unit)
            if origin is None:
                assert line is None
                continue
            # Identity, not approximate equality: the tool may not move a goal,
            # a counted figure or a ratio on its way to the model.
            assert line["goal"] == origin["goal"], unit
            assert line["counted"] == origin["counted"], unit
            assert line["pace"]["verdict"] == origin["pace"]["verdict"], unit
            assert line["pace"].get("ratio") == origin["pace"].get("ratio"), unit
    verdicts = [row["headline"]["verdict"] for row in payload["rows"]]
    rank = {words.BEHIND: 0, words.AT_RISK: 1, words.UNKNOWN: 2, words.ON_PACE: 3}
    assert [rank[v] for v in verdicts] == sorted(rank[v] for v in verdicts)


def test_the_board_states_the_trigger_and_that_counted_is_not_delivered() -> None:
    """Two claims decide what every verdict on this board means, and an answer
    that quotes a pace without them is quoting a number with no basis."""
    payload = _board()
    assert payload["trigger"]["on_pace_ratio"] == words.ON_PACE_RATIO
    assert payload["trigger"]["at_risk_ratio"] == words.AT_RISK_RATIO
    assert payload["counted_is_planned_en"] == words.COUNTED_IS_PLANNED_EN
    assert payload["counted_is_planned_he"] == words.COUNTED_IS_PLANNED_HE
    assert payload["as_of"]["instant"], "a pacing figure with no counted instant cannot be reproduced"
    assert payload["raise_rule"]["rule_en"] == words.RAISE_RULE_EN


def test_the_filter_narrows_and_still_states_how_many_the_board_holds() -> None:
    payload = _board()
    verdict = payload["rows"][0]["headline"]["verdict"]
    narrowed = execute_read_tool("get_pacing_board", {"verdict": verdict, "limit": 2}, None)
    assert narrowed["filtered_by"] == {"verdict": verdict}
    assert {row["headline"]["verdict"] for row in narrowed["rows"]} == {verdict}
    assert narrowed["rows_matching"] == payload["counts"][verdict]
    # The count of what came back never hides how many rows matched.
    assert len(narrowed["rows"]) <= 2
    if narrowed["rows_matching"] > 2:
        assert narrowed["rows_omitted"] == narrowed["rows_matching"] - 2
    assert "error" in execute_read_tool("get_pacing_board", {"verdict": "sideways"}, None)


# --- honest refusals: four of them, each with its own reason -----------------------
def test_every_refusal_to_state_a_pace_arrives_with_its_own_reason() -> None:
    """The board refuses in four separate places and they send a reader to four
    different places. A row that cannot state a pace must never come back as a
    bare unknown, and must never come back as a zero."""
    payload = execute_read_tool("get_pacing_board", {"limit": 30}, None)
    published = payload["reasons"]
    unknowns = [row for row in payload["rows"]
                if row["headline"]["verdict"] == words.UNKNOWN]
    if not unknowns:
        pytest.skip("no row on the shipped board fails to state a pace")
    for row in unknowns:
        code = row["headline"].get("code")
        assert code, f"{row['campaign_id']} is unknown with no code at all"
        words_for_it = published.get(code) or row["headline"]
        assert str(words_for_it["reason_en"]).strip()
        assert str(words_for_it["reason_he"]).strip()
        assert row["headline"]["ratio"] is None, "an unknown pace must not carry a ratio"


def test_the_four_reason_codes_the_board_can_refuse_with_all_have_words() -> None:
    """Every refusal the arithmetic can produce, checked against the published
    words rather than against the rows that happen to be on disk today."""
    for code in ("not_started", "no_source", "gap_in_elapsed", "no_goal", "unmeasurable",
                 "no_flight_dates"):
        published = words.reason(code)
        assert published["reason_en"].strip() and published["reason_he"].strip(), code
        assert published["path_forward_en"].strip(), code


def test_a_day_with_no_source_is_unknown_and_is_never_counted_as_no_delivery() -> None:
    payload = _board()
    detail = execute_read_tool("get_campaign_pacing",
                               {"campaign_id": payload["rows"][0]["campaign_id"]}, None)
    days = detail["broadcast_days"]["days"]
    assert days, "a campaign on the board must have its flight days readable"
    for day in days:
        if day["air_state"] in ("aired", "scheduled"):
            continue
        # Not a zero and not a missing row: the day is present and says why it
        # holds no figure, which is the difference the whole board rests on.
        assert day["rating_points_planned"] is None
        assert day["spend_ils"] is None
        assert str(day["note"]).strip()


def test_the_day_lists_are_never_the_wire_null_that_means_same_as_the_flight() -> None:
    """The board's wire form collapses a day list identical to the flight's to an
    explicit null. A model reading that null would take it for no missing days,
    which is the opposite of what it says, so the scan restores those two lists."""
    payload = _board()
    for row in payload["rows"]:
        for unit in ("rating", "money"):
            line = row.get(unit)
            if not line:
                continue
            assert line["forward"].get("unsourced_remaining_days") is not None or \
                "unsourced_remaining_days" not in line["forward"]
            assert line["pace"].get("unsourced_elapsed_days") is not None or \
                "unsourced_elapsed_days" not in line["pace"]


# --- one campaign, and the remedy -------------------------------------------------
def test_one_campaign_arrives_whole_with_its_days_and_its_reasons_inline() -> None:
    payload = _board()
    campaign_id = payload["rows"][0]["campaign_id"]
    detail = execute_read_tool("get_campaign_pacing", {"campaign_id": campaign_id}, None)
    assert "error" not in detail
    assert detail["campaign_id"] == campaign_id
    truth = read.days_payload(campaign_id)
    assert detail["broadcast_days"]["count"] == truth["count"]
    assert detail["broadcast_days"]["days"][:5] == truth["days"][:5]
    # A block that refuses carries its own words here, with no table to consult.
    for unit in ("rating", "money"):
        line = detail.get(unit)
        if line and line["pace"].get("code"):
            assert line["pace"]["reason_en"].strip()
            assert line["pace"]["reason_he"].strip()
    assert detail["reference_rule"]["rule_en"] == words.EVEN_REFERENCE_EN


def test_the_remedy_is_exactly_what_the_write_path_would_accept() -> None:
    """The campaign manager is told to name the remedy. What this says can be
    raised is computed by the same function the raise route calls, so it cannot
    offer a raise the API would refuse, and it refuses in the API's own words."""
    payload = _board()
    view = read.board_payload()
    as_of_day = board.parse_date(view["as_of"]["instant"])
    checked = 0
    for row in payload["rows"][:8]:
        detail = execute_read_tool("get_campaign_pacing", {"campaign_id": row["campaign_id"]}, None)
        remedy = detail["remedy"]
        truth_row = read.find_row(view, row["campaign_id"])
        deficit, why = read.raisable_deficit(truth_row, as_of_day)
        assert remedy["make_good_can_be_raised"] is (deficit is not None)
        assert remedy["owed_deficit"] == deficit
        if deficit is None:
            assert remedy["why_not"] == why
            assert remedy["reason_en"].strip() and remedy["reason_he"].strip()
        assert remedy["risk_can_be_taken_on"] is (
            read.acceptance_figures(truth_row, as_of_day) is not None)
        assert remedy["rule_en"] == words.RAISE_RULE_EN
        checked += 1
    assert checked, "the remedy was never exercised"


def test_a_measured_gap_to_date_is_not_offered_as_a_debt() -> None:
    """The third rung of the ladder is a measured figure and is not owed. Holding
    both rules would let a client put a figure in the ledger this product says is
    not owed, so the tool states the gap and refuses the raise."""
    payload = _board()
    view = read.board_payload()
    as_of_day = board.parse_date(view["as_of"]["instant"])
    for row in payload["rows"]:
        truth_row = read.find_row(view, row["campaign_id"])
        full = read.deficit_for(truth_row, as_of_day)
        if full is None or full["deficit_kind"] in read.RAISABLE_KINDS:
            continue
        detail = execute_read_tool("get_campaign_pacing", {"campaign_id": row["campaign_id"]}, None)
        remedy = detail["remedy"]
        assert remedy["make_good_can_be_raised"] is False
        assert remedy["owed_deficit"] is None
        assert remedy["why_not"] == "not_owed_yet"
        # The measured gap is still stated, on the acceptance side where it belongs.
        assert remedy["measured_figures_an_acceptance_would_record"] == full
        return
    pytest.skip("no row on the shipped board reaches only the gap-to-date rung")


def test_an_unknown_campaign_is_the_boards_own_refusal_not_an_empty_row() -> None:
    payload = execute_read_tool("get_campaign_pacing", {"campaign_id": UNKNOWN_CAMPAIGN}, None)
    assert payload["error"] == read.UNKNOWN_CAMPAIGN_EN
    assert payload["reason_he"] == read.UNKNOWN_CAMPAIGN_HE
    assert "rating" not in payload and "remedy" not in payload
    missing = execute_read_tool("get_campaign_pacing", {}, None)
    assert "get_pacing_board" in missing["error"]


# --- the ledger -------------------------------------------------------------------
def test_the_ledger_reports_what_was_decided_and_what_was_never_configured() -> None:
    payload = execute_read_tool("get_make_good_ledger", {}, None)
    from kairos_api import makegood_store as ledger

    truth = read.ledger_payload(ledger.load_frame())
    assert payload["count"] == truth["count"]
    assert payload["open_count"] == truth["open_count"]
    assert payload["accepted_count"] == truth["accepted_count"]
    # The rule for what may be offered and who signs it off was never supplied,
    # and the ledger says so rather than deriving an entitlement.
    assert payload["sign_off"]["configured"] is False
    assert payload["sign_off"]["reason_en"].strip()
    if not payload["count"]:
        # An empty ledger is not a finding that nothing is owed, and the payload
        # must not let those two be read as the same fact.
        assert "empty ledger" in payload["note"]


# --- the competitor boundary ------------------------------------------------------
def test_no_rival_channel_name_reaches_any_pacing_tool() -> None:
    rivals = _rival_channels()
    assert rivals, "the scan is vacuous unless the product's data really carries rivals"
    payload = execute_read_tool("get_pacing_board", {"limit": 30}, None)
    assert _names_found(payload, rivals) == set()
    assert payload["scope"]["scope_channel"] == channel_scope.operator_channel()
    assert payload["scope"]["scoped"] is True
    for row in payload["rows"][:10]:
        detail = execute_read_tool("get_campaign_pacing", {"campaign_id": row["campaign_id"]}, None)
        assert _names_found(detail, rivals) == set(), row["campaign_id"]
        assert detail["channel"] == channel_scope.operator_channel()
    assert _names_found(execute_read_tool("get_make_good_ledger", {}, None), rivals) == set()
    # Not even the refusal, which lists the campaigns that can be read.
    assert _names_found(
        execute_read_tool("get_campaign_pacing", {"campaign_id": UNKNOWN_CAMPAIGN}, None),
        rivals) == set()


def test_the_boundary_scan_bites_on_an_unscoped_source() -> None:
    """The positive control for the test above.

    A scan that has never flagged anything has not been shown to work, so the
    same scan is pointed at the saved weekly plan read WITHOUT the channel scope,
    which is exactly the unscoped source every operator surface is forbidden to
    serve. It must find rivals there. If this ever passes empty, the scan above
    is asleep.
    """
    rivals = _rival_channels()
    unscoped = pd.read_csv(PLAN_PATH).head(500).to_dict("records")
    assert _names_found(unscoped, rivals), "the scan failed to flag an unscoped plan read"


def test_the_scope_the_pacing_read_applies_really_drops_a_rival_campaign() -> None:
    """The second control, on the mechanism rather than on the payload.

    Today's campaign store happens to hold only the operator's own channel, so
    the scan above would pass even if the scope had been removed. This feeds the
    same scoping function the pacing read applies a record on a real rival
    channel and asserts it never comes out, so the boundary is shown to bite and
    not merely to be unexercised.
    """
    rivals = _rival_channels()
    rival = sorted(rivals)[0]
    records = [{"campaign_id": "CMP_OWNED", "channel": channel_scope.operator_channel()},
               {"campaign_id": "CMP_RIVAL", "channel": rival}]
    scoped, scope = channel_scope.scope_records(records, key="channel")
    assert [one["campaign_id"] for one in scoped] == ["CMP_OWNED"]
    assert scope["competitor_rows_excluded"] == 1
    assert _names_found(scoped, rivals) == set()
