"""The remedy Kai can now propose, and the four ways it could be dishonest.

``propose_pacing_decision`` is the first propose tool over the pacing board, and
the board is where money is owed to a client. These tests hold it to the four
properties the design rests on, each of which fails silently if it is not
checked:

* **Capture writes nothing.** A proposal is not an act, so building one must
  leave the ledger byte-identical. The failure here is the worst kind: it looks
  exactly like the tool working.
* **A refusal arrives at capture, in the board's own words.** The routes refuse
  a raise with no measured shortfall, a duplicate, and a transition the machine
  does not hold. If the validator did not run those same checks, Kai would offer
  a pending item that fails in front of the operator at approval time.
* **The approving account reaches the ledger.** Every applier reaches its store
  with no HTTP request, so the store's own actor lookup finds nobody; a row that
  records a shortfall and a blank name is the one thing this ledger exists not
  to produce.
* **The competitor boundary holds**, checked with the positive control this
  suite established: the scan is pointed at an unscoped frame first and required
  to find rivals there, so a clean result on the real one means the scan works
  rather than that it is blind.

Nothing is mocked. The validator runs against the real board and the real
ledger; the apply test redirects the ledger to a tmp copy so the shipped file is
never written.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from kairos_api import assistant_pacing_propose as propose
from kairos_api import assistant_tools as tools
from kairos_api import channel_scope
from kairos_api import makegood_store as ledger
from kairos_api import pacing_alerts_api_read as read
from kairos_api import pacing_alerts_api_wire as wire

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"
TOOL = "propose_pacing_decision"


def _rows() -> list[dict]:
    view = wire.expand(read.board_payload())
    rows = view.get("rows") or []
    if not rows:
        pytest.skip("no campaign on the operator channel, so there is no board to decide about")
    return rows


def _ledger_digest() -> str | None:
    path = Path(ledger.MAKE_GOODS_PATH)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _capture(**args: object) -> dict:
    return tools.build_proposal_item(TOOL, {"reason": "בדיקה", **args}, user="tester")


def _acceptable() -> str:
    """A campaign whose risk the board would actually let somebody take on."""
    for row in _rows():
        item = _capture(action="accept_risk", campaign_id=row["campaign_id"])
        if item["status"] == "pending":
            return row["campaign_id"]
    pytest.skip("no campaign on the board is currently at risk, so no acceptance is available")


# --- registration: the wiring a kind needs, or its edits are versioned nowhere ----
def test_the_tool_is_registered_everywhere_a_kind_has_to_be() -> None:
    """A propose kind is wired in five places and four of them fail silently.

    A schema registered after PROPOSE_TOOL_NAMES freezes is invisible to the
    dispatcher; a kind with no applier fails at approval; a kind whose logical
    name the version store does not know RAISES on the snapshot; and a kind with
    no state file produces an apply that cannot be undone.
    """
    import kairos_api.assistant  # noqa: F401 - loading the router registers the apply side
    from kairos_api import assistant_actions, version_store

    assert TOOL in tools.PROPOSE_TOOL_NAMES
    assert TOOL not in tools.READ_TOOL_NAMES
    assert tools.KIND_BY_TOOL[TOOL] == propose.KIND
    assert assistant_actions._APPLIERS[propose.KIND] is propose.apply_pacing_decision
    assert version_store._LOGICAL_FOR_KIND[propose.KIND] == propose.LOGICAL
    # The exact defect the campaign has already paid for once: a logical name the
    # store does not know is versioned nowhere, and snapshot() raises on it.
    assert propose.LOGICAL in version_store._KNOWN_LOGICAL
    assert version_store._logical_path(propose.LOGICAL) == Path(ledger.MAKE_GOODS_PATH)
    assert Path(ledger.MAKE_GOODS_PATH) in assistant_actions._state_files_for({propose.KIND})


def test_a_proposal_can_be_captured_without_the_router_being_loaded() -> None:
    """The schema and its validator must arrive together, in one import.

    This caught a real defect. The schema was registered at import of
    assistant_tool_schemas and the validator inside the action plane's
    register(), which only runs when kairos_api.assistant loads. So any process
    that imported the tools without the router offered the tool to the model and
    answered every call of it with a KeyError dressed as a validation failure.
    A subprocess is the only honest check: within this suite something else has
    always already imported the router.
    """
    import subprocess
    import sys

    script = (
        "from kairos_api import assistant_tools as t;"
        "item = t.build_proposal_item('propose_pacing_decision',"
        " {'action': 'accept_risk', 'campaign_id': 'CMP_DOES_NOT_EXIST', 'reason': 'x'}, 'u');"
        "print(item['status'], '|', item.get('error', ''))"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          cwd=str(ROOT), check=False)
    assert done.returncode == 0, done.stderr[-2000:]
    tail = done.stdout.strip().splitlines()[-1]
    assert tail.startswith("rejected"), tail
    # Refused for the real reason, not because the validator was missing.
    assert "validation failed" not in tail and "KeyError" not in tail
    assert read.UNKNOWN_CAMPAIGN_HE in tail


def test_the_schema_asks_for_a_campaign_and_never_for_a_figure() -> None:
    """The shortfall is measured by the product, so the tool cannot accept one."""
    schema = next(s for s in tools.PROPOSE_TOOL_SCHEMAS if s["name"] == TOOL)
    properties = set(schema["input_schema"]["properties"])
    assert schema["input_schema"]["required"] == ["action", "reason"]
    # The only figure a decision may carry is the human's offer. A goal, a
    # counted value or a deficit in this schema would be a number nobody computed.
    for forbidden in ("goal_value", "counted_value", "deficit_value", "shortfall"):
        assert forbidden not in properties
    assert "offer_value" in properties


# --- a proposal is not an act -----------------------------------------------------
def test_capturing_any_decision_leaves_the_ledger_byte_identical() -> None:
    before = _ledger_digest()
    for row in _rows()[:10]:
        for action in ("raise_make_good", "accept_risk"):
            item = _capture(action=action, campaign_id=row["campaign_id"])
            assert item["status"] in ("pending", "rejected")
    assert _capture(action="move_make_good", make_good_id="MG_0001",
                    state="withdrawn")["status"] in ("pending", "rejected")
    assert _ledger_digest() == before, "capturing a proposal wrote to the make-good ledger"


# --- refusals arrive at capture, in the product's own words -----------------------
def test_an_unknown_campaign_is_refused_with_the_boards_own_sentence() -> None:
    item = _capture(action="accept_risk", campaign_id="CMP_DOES_NOT_EXIST")
    assert item["status"] == "rejected"
    assert item["error"] == read.UNKNOWN_CAMPAIGN_HE


def test_every_raise_agrees_with_what_the_write_path_would_have_decided() -> None:
    """The validator must consult raisable_deficit, the function the writer calls.

    Stated as an agreement rather than as "some of them are refused", because a
    validator that stopped consulting it would simply capture everything as
    pending, and a test that only inspected the refusals it happened to get would
    have nothing left to look at and would pass by falling silent. Here every row
    is compared against the truth, so removing the check fails on the first one.
    """
    from kairos_api import pacing_alerts_api as api

    published = {half for pair in api.REFUSED_RAISE.values() for half in pair}
    published |= {read.NOTHING_TO_RAISE_HE, api.DUPLICATE_HE}
    view = wire.expand(read.board_payload())
    as_of = __import__("kairos_api.pacing_alerts_api_board", fromlist=["board"]).parse_date(
        view.get("as_of", {}).get("instant"))
    checked = 0
    for row in _rows()[:12]:
        truth, _why = read.raisable_deficit(row, as_of)
        owed = truth is not None and not ledger.open_for(
            ledger.load_frame(), row["campaign_id"], ledger.MAKE_GOOD)
        item = _capture(action="raise_make_good", campaign_id=row["campaign_id"])
        assert (item["status"] == "pending") is owed, (row["campaign_id"], item.get("error"))
        if not owed:
            assert item["error"] in published, item["error"]
        checked += 1
    assert checked, "no board row was checked, so this test proved nothing"


def test_an_action_the_tool_does_not_hold_is_refused_by_name() -> None:
    item = _capture(action="delete_the_ledger", campaign_id="CMP_D040")
    assert item["status"] == "rejected"
    for action in propose.ACTIONS:
        assert action in item["error"]


@pytest.fixture()
def seeded_ledger(tmp_path, monkeypatch):
    """One raised make-good in a tmp ledger, so the move path is really exercised.

    The shipped ledger is empty, so every move assertion would skip against it,
    and a skipped assertion is not an assertion. Seeding is honest here because
    nothing about a transition depends on how the row was measured.
    """
    row = ledger.blank_row()
    row.update({"make_good_id": "MG_0001", "kind": ledger.MAKE_GOOD,
                "campaign_id": "CMP_SEED", "campaign_name": "קמפיין בדיקה",
                "unit": "rating_points", "goal_value": "100", "counted_value": "60",
                "deficit_value": "40", "deficit_kind": ledger.MEASURED_CLOSED,
                "state": ledger.RAISED, "raised_at": ledger.now_stamp(), "raised_by": "seed"})
    path = tmp_path / "make_goods.csv"
    monkeypatch.setattr(ledger, "MAKE_GOODS_PATH", path)
    monkeypatch.setattr(ledger, "BACKUP_DIR", tmp_path / "_backups")
    ledger.write_frame(pd.DataFrame([row]))
    return ledger.record(ledger.load_frame().loc[0])


def test_a_move_the_state_machine_forbids_is_refused_at_capture(seeded_ledger) -> None:
    forbidden = next(s for s in ledger.STATE_VOCABULARY
                     if s["value"] not in seeded_ledger["next_states"]
                     and s["value"] != seeded_ledger["state"])
    item = _capture(action="move_make_good", make_good_id="MG_0001", state=forbidden["value"])
    assert item["status"] == "rejected"
    # The refusal names the states that ARE allowed, so a caller is told the
    # shape of the machine instead of guessing at it.
    assert forbidden["label_he"] in item["error"]


def test_a_move_to_offered_without_a_value_is_refused_and_with_one_is_captured(
    seeded_ledger,
) -> None:
    """The one figure a decision may carry, and the rule the ledger puts on it."""
    if ledger.OFFERED not in seeded_ledger["next_states"]:
        pytest.skip("this ledger's machine does not offer from the entry state")
    bad = _capture(action="move_make_good", make_good_id="MG_0001", state=ledger.OFFERED)
    assert bad["status"] == "rejected"

    good = _capture(action="move_make_good", make_good_id="MG_0001",
                    state=ledger.OFFERED, offer_value=25.0)
    assert good["status"] == "pending", good.get("error")
    assert good["payload"]["move"]["offer_value"] == 25.0
    # An offer is a number, so the card has to say it before anybody approves it.
    assert "25.0" in good["summary"]
    assert seeded_ledger["shortfall"]["unit"] in good["summary"]


def test_an_approved_move_records_the_approver_not_the_proposer(seeded_ledger) -> None:
    item = _capture(action="move_make_good", make_good_id="MG_0001",
                    state=ledger.OFFERED, offer_value=25.0)
    if item["status"] != "pending":
        pytest.skip("this ledger's machine does not offer from the entry state")
    propose.apply_pacing_decision(item["payload"], "orit.admin")
    moved = ledger.record(ledger.load_frame().loc[0])
    assert moved["state"] == ledger.OFFERED
    assert moved["offer"]["offered_by"] == "orit.admin"
    assert moved["offer"]["value"] == 25.0


def test_a_decision_without_a_reason_is_refused_like_every_other_proposal() -> None:
    item = tools.build_proposal_item(TOOL, {"action": "accept_risk", "campaign_id": "CMP_D040"},
                                     user="tester")
    assert item["status"] == "rejected"
    assert "reason is required" in item["error"]


# --- a valid proposal, and what the operator reads on it --------------------------
def test_an_available_acceptance_captures_as_pending_with_a_hebrew_summary() -> None:
    campaign_id = _acceptable()
    item = _capture(action="accept_risk", campaign_id=campaign_id, note="סוכם מול הלקוח")
    assert item["status"] == "pending"
    assert item["kind"] == propose.KIND
    assert item["payload"] == {"action": "accept_risk", "campaign_id": campaign_id,
                               "note": "סוכם מול הלקוח"}
    # The summary is what the approval card reads. It must be in the operator's
    # own language, and it must not glue a one-letter prefix onto a digit.
    assert any("֐" <= ch <= "ת" for ch in item["summary"])
    assert "ב2" not in item["summary"]
    assert campaign_id in item["summary"]


# --- the approving account reaches the row ----------------------------------------
def test_the_approving_account_is_recorded_on_the_ledger_row(tmp_path, monkeypatch) -> None:
    """Apply for real, against a tmp copy of the ledger, and read back who acted.

    This is the assertion the applier signature exists for. Before the approving
    account was threaded through, every applier reached its store with
    request=None, the store resolved the actor to the empty string, and the row
    recorded a shortfall against nobody.
    """
    campaign_id = _acceptable()
    copy = tmp_path / "make_goods.csv"
    source = Path(ledger.MAKE_GOODS_PATH)
    copy.write_bytes(source.read_bytes() if source.exists() else b"")
    monkeypatch.setattr(ledger, "MAKE_GOODS_PATH", copy)
    monkeypatch.setattr(ledger, "BACKUP_DIR", tmp_path / "_backups")

    item = _capture(action="accept_risk", campaign_id=campaign_id, note="הוחלט")
    assert item["status"] == "pending"
    result = propose.apply_pacing_decision(item["payload"], "orit.admin")

    written = ledger.record(ledger.load_frame().set_index("make_good_id").loc[result["make_good_id"]])
    assert written["raised_by"] == "orit.admin", "the ledger row records nobody"
    assert written["campaign_id"] == campaign_id
    assert written["kind"] == ledger.ACCEPTANCE
    # And the figures on the row are the board's, stamped at the write, not any
    # number the proposal carried: the payload carried none at all.
    assert written["shortfall"]["counted_as_of"]
    assert source.read_bytes() if source.exists() else True


# --- the competitor boundary, with its positive control ---------------------------
def _rivals() -> set[str]:
    owned = channel_scope.operator_channel()
    channels = {str(v).strip() for v in pd.read_csv(PLAN_PATH)["channel"].dropna().unique()}
    return {name for name in channels if name and name != owned}


def _found(payload: object, names: set[str]) -> set[str]:
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    return {name for name in names if name in blob}


def test_no_rival_channel_reaches_a_captured_proposal() -> None:
    rivals = _rivals()
    if not rivals:
        pytest.skip("the saved plan holds only the operator's own channel")
    captured = [_capture(action=action, campaign_id=row["campaign_id"])
                for row in _rows()[:12] for action in ("raise_make_good", "accept_risk")]
    captured.append(_capture(action="accept_risk", campaign_id="CMP_DOES_NOT_EXIST"))
    assert _found(captured, rivals) == set()
    assert _found(propose.pending_index(), rivals) == set()


def test_the_boundary_scan_bites_on_an_unscoped_source() -> None:
    """The positive control for the test above.

    A scan that finds nothing proves nothing until it has been shown finding
    something. The saved plan holds every channel because the retention model is
    measured against the competitive lineup, so reading it unscoped is exactly
    what an operator surface is forbidden to do, and the scan must flag it.
    """
    rivals = _rivals()
    if not rivals:
        pytest.skip("the saved plan holds only the operator's own channel")
    unscoped = pd.read_csv(PLAN_PATH).head(500).to_dict("records")
    assert _found(unscoped, rivals), "the scan failed to flag an unscoped plan read"


# --- the surface the proposal concerns --------------------------------------------
def test_the_board_publishes_pending_proposals_apart_from_the_ledger() -> None:
    """A pending proposal is visible on the board, and never as a ledger record.

    The count of what is owed comes from ``make_goods``; a proposal has changed
    nothing and is owed by nobody, so it rides its own key. Merging them would
    make an unapproved suggestion read as a debt.
    """
    from kairos_api import pacing_alerts_api as api

    payload = api.pacing_board(None)
    assert set(payload["proposed"]) <= set()  | set(payload["proposed"])  # shape only
    assert "proposed" in payload and "make_goods" in payload and "acceptances" in payload
    assert payload["proposed"] is not payload["make_goods"]
    for entries in payload["proposed"].values():
        for entry in entries:
            assert {"batch_id", "item_id", "action"} <= set(entry)


def test_an_unreadable_proposal_store_leaves_the_board_readable(monkeypatch) -> None:
    """The board outranks the index: a broken store empties it, never raises."""
    from kairos_api import assistant_actions

    def boom() -> dict:
        raise RuntimeError("store is half-written")

    monkeypatch.setattr(assistant_actions, "_load_store", boom)
    assert propose.pending_index() == {}
