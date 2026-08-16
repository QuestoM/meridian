"""The promise a save makes, held against what the save actually did.

The day board answers a drag in under a millisecond because it scores the
arrangement on screen against the plan's own basis while holding the break counts
the plan already chose. That answer is a PREDICTION, and a prediction that is
never checked is a claim.

Measured on ``רשת 13 / 2024-11-01``, one process, same inputs, seconds apart:
writing exactly the restriction the board writes for one break moves the
committed day from 1,067,845.55 to 1,037,270.00, while the prediction for that
same act is 0.00. The gap is not a price the move carries. Pinning a break at the
offset, duration and gold flag the plan had already given it leaves the unpinned
arrangement feasible, and the engine's own objective still falls from
0.541628827 to 0.537917737, so the gap is the rebuild's search landing elsewhere.

So this file asserts the product tells the truth about that: either the two
figures agree, or the surface names the divergence with both numbers in it. Which
of the two happens is measured here rather than assumed, and the shipped
classifier is executed rather than described, so a critic reading this file is
reading the same code the operator sees.

The second half of the file holds the gold act, which had the same shape of
defect: a route that reported what it asked for instead of what the plan came
back with.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# See the helper's own docstring: the shared settings file can lose the operator
# channel while these tests run, and a skipped money test proves nothing.
from test_p3_break_store import declare_operator_channel

pytestmark = pytest.mark.realdata

ROOT = Path(__file__).resolve().parents[1]
SETTLEMENT_JS = ROOT / "tv-break-dashboard" / "src" / "plan" / "day" / "day-board-settlement.js"
BOARD_JSX = ROOT / "tv-break-dashboard" / "src" / "plan" / "day" / "DayBoard.jsx"
PANEL_JSX = ROOT / "tv-break-dashboard" / "src" / "plan" / "day" / "DayBoardSettlement.jsx"


@pytest.fixture(scope="module", autouse=True)
def owned_channel():
    patch = declare_operator_channel()
    yield
    if patch is not None:
        patch.undo()


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Every store this test writes points at a temporary directory.

    The restriction store is in the list because a save writes one, and a test
    that left a restriction behind would change the plan every later test reads.
    """
    from kairos_api import break_store, break_store_pins
    from kairos_api import constraints as constraints_api
    from kairos_api import overrides as override_api

    monkeypatch.setattr(override_api, "OVERRIDES_PATH", tmp_path / "manual_overrides.csv")
    monkeypatch.setattr(override_api, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(override_api, "_snapshot_before_write", lambda request: None)
    monkeypatch.setattr(break_store_pins, "BREAKS_PATH", tmp_path / "breaks.csv")
    monkeypatch.setattr(break_store_pins, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", tmp_path / "kairos_constraints.csv")
    monkeypatch.setattr(constraints_api, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(constraints_api, "_snapshot_before_write", lambda request: None)
    break_store.invalidate()
    yield tmp_path
    break_store.invalidate()


@pytest.fixture()
def client(isolated):
    from kairos_api.break_api import router as break_router
    from kairos_api.constraints import router as constraint_router

    app = FastAPI()
    app.include_router(break_router)
    app.include_router(constraint_router)
    return TestClient(app)


@pytest.fixture()
def opened_day(client):
    days = client.get("/api/plan/days").json()
    if not days["available"]:
        pytest.skip(days["reason"])
    board = client.get("/api/plan/day", params={"day": days["days"][0]}).json()
    if not board["breaks"]:
        pytest.skip("this day carries no breaks")
    return board


def node_settlement(payload: dict) -> dict:
    """Run the shipped settlement module on this payload and return its verdict.

    The classifier the operator's browser runs is the one asserted here. A python
    re-implementation of it would only prove that two pieces of test code agree.
    """
    script = (
        f"const m = await import({json.dumps(SETTLEMENT_JS.as_uri())});"
        f"const out = m.settlementOf({json.dumps(payload)});"
        "process.stdout.write(JSON.stringify(out));"
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def board_predicate(programme: dict) -> dict:
    """The predicate ``day-board-actions.js`` writes, on the frozen contract."""
    return {
        "combinator": "and",
        "conditions": [
            {"field": "date", "operator": "is", "value": programme["day"]},
            {"field": "programme", "operator": "is", "value": programme["title"]},
            {"field": "hour", "operator": "eq", "value": int(programme["start_seconds"] // 3600) % 24},
        ],
    }


def test_the_saved_plan_is_measured_against_the_preview_that_promised_it(client, opened_day):
    """Write what the board writes, then compare the two figures out loud."""
    day = opened_day["day"]
    before = opened_day["totals"]
    target = opened_day["breaks"][0]
    programme = next(row for row in opened_day["programmes"] if row["segment_id"] == target["segment_id"])

    preview = client.post("/api/plan/day/score", json={"day": day, "moves": [{
        "break_id": target["break_id"],
        "offset_seconds": target["offset_seconds"],
        "duration_seconds": target["duration_seconds"],
        "is_gold": target["is_gold"],
    }]}).json()
    predicted = preview["delta"]["revenue"]

    created = client.post("/api/constraints", json={
        "scope_type": "always",
        "effect": "fix_offset",
        "offset_seconds": round(target["offset_seconds"]),
        "duration_seconds": round(target["duration_seconds"]),
        "order_index": target["ordinal"],
        "where": board_predicate(programme),
    })
    assert created.status_code == 201
    constraint_id = created.json()["constraint_id"]
    recorded = client.post(f"/api/breaks/{quote(target['break_id'], safe='')}/placement", json={
        "constraint_id": constraint_id,
        "offset_seconds": target["offset_seconds"],
        "duration_seconds": target["duration_seconds"],
        "is_gold": bool(target["is_gold"]),
    })
    assert recorded.status_code == 201

    after_board = client.get("/api/plan/day", params={"day": day}).json()
    after = after_board["totals"]
    realised = round(after["revenue"] - before["revenue"], 2)

    settlement = node_settlement({
        "act": "save",
        "basis": after_board["basis"],
        "before": before,
        "after": after,
        "beforeBreaks": opened_day["breaks"],
        "afterBreaks": after_board["breaks"],
        "predictedRevenue": predicted,
    })
    assert settlement["realised"]["revenue"] == pytest.approx(realised, abs=0.01)
    assert settlement["predicted"] == pytest.approx(predicted, abs=1e-9)

    if abs(realised - predicted) <= 0.005:
        assert settlement["verdict"] == "agreed", settlement
    else:
        # The measured case today. The surface must name it, with the realised
        # figure in hand, rather than re-basing itself and printing a zero.
        assert settlement["verdict"] == "diverged", settlement
        assert settlement["difference"] == pytest.approx(realised - predicted, abs=0.01)
        assert abs(settlement["realised"]["revenue"]) > 0.005
        assert settlement["rearranged"]["changed"] >= 1, "a divergence this size did not come from nothing"

    # The undo half, asserted on the same figures: the day goes back to itself.
    client.delete(f"/api/constraints/{quote(str(constraint_id), safe='')}")
    client.delete(f"/api/breaks/{quote(target['break_id'], safe='')}/placement")
    restored = client.get("/api/plan/day", params={"day": day}).json()["totals"]
    assert restored["revenue"] == pytest.approx(before["revenue"], abs=0.005)
    assert restored["breaks"] == before["breaks"]


def _strand_a_placement(client, board):
    """Save one break so the engine re-plans its programme and drops that id.

    A save writes a restriction and the engine then plans the whole day again with
    it in force, and on that second run it chooses how many breaks each programme
    gets. When it chooses fewer, every ordinal after the survivors stops existing,
    because identity is segment plus ordinal. That is the case this file is here
    for, so it is produced rather than hoped for: candidates are tried in order and
    the first one whose id the re-plan removes is returned.
    """
    counts: dict[str, int] = {}
    for row in board["breaks"]:
        counts[row["segment_id"]] = counts.get(row["segment_id"], 0) + 1
    crowded = sorted(counts, key=lambda key: (-counts[key], key))
    for segment_id in crowded[:3]:
        if counts[segment_id] < 2:
            continue
        target = [row for row in board["breaks"] if row["segment_id"] == segment_id][-1]
        programme = next(row for row in board["programmes"] if row["segment_id"] == segment_id)
        created = client.post("/api/constraints", json={
            "scope_type": "always",
            "effect": "fix_offset",
            "offset_seconds": round(target["offset_seconds"]) + 60,
            "duration_seconds": round(target["duration_seconds"]),
            "order_index": target["ordinal"],
            "where": board_predicate(programme),
        })
        assert created.status_code == 201
        constraint_id = created.json()["constraint_id"]
        recorded = client.post(f"/api/breaks/{quote(target['break_id'], safe='')}/placement", json={
            "constraint_id": constraint_id,
            "offset_seconds": target["offset_seconds"] + 60,
            "duration_seconds": target["duration_seconds"],
            "is_gold": bool(target["is_gold"]),
        })
        assert recorded.status_code == 201
        after = client.get("/api/plan/day", params={"day": board["day"]}).json()
        if target["break_id"] not in {row["break_id"] for row in after["breaks"]}:
            return target, constraint_id, after
        client.delete(f"/api/constraints/{quote(str(constraint_id), safe='')}")
        client.delete(f"/api/breaks/{quote(target['break_id'], safe='')}/placement")
    return None, None, None


def test_a_save_that_removes_its_own_break_id_still_offers_exactly_one_inverse(client, opened_day):
    """The money a save spends is reversible even when the save deletes the chip.

    Measured on ``רשת 13 / 2024-11-01`` before this closed: select
    ``2024-11-01|רשת 13|001~2``, one ArrowRight, Save. The engine re-plans that
    programme from four breaks down to one, the day falls from 1,067,845.55 to
    1,020,401.35 and the count from 80 to 78, so the id the record names stops
    existing. Zero breaks came back carrying ``saved_placement``, no chip rendered
    as saved, and after a reload the board offered no Undo, no Discard and no
    Remove anywhere, while ``data/breaks.csv`` and ``data/kairos_constraints.csv``
    both still held the row. 47,444.20 ILS spent with no route back from the
    surface that spent it.

    So the partition is asserted here rather than described: every saved record is
    either reachable on its chip or served as an unbound placement, exactly one of
    the two, and taking the inverse the surface offers returns the day's own
    figures to what they were before the save.
    """
    day = opened_day["day"]
    before = opened_day["totals"]
    target, constraint_id, after_board = _strand_a_placement(client, opened_day)
    if target is None:
        pytest.skip("no save on this day re-planned its programme to fewer breaks, so the case cannot be produced")

    stranded = after_board["unbound_placements"]
    live_saved = [row["break_id"] for row in after_board["breaks"] if row.get("saved_placement")]
    assert target["break_id"] not in {row["break_id"] for row in after_board["breaks"]}
    assert live_saved == [], "the re-plan removed the id, so no chip can be carrying the record"
    assert len(stranded) + len(live_saved) == 1, "one saved record, one inverse, never two and never none"
    record = stranded[0]
    assert record["break_id"] == target["break_id"]
    assert record["constraint_id"] == constraint_id, "the row addresses the restriction it has to delete"
    assert record["state"] in {"segment_replanned", "segment_absent"}
    assert record["reason"] and record["reason_he"], "the reason travels in both languages"
    assert record["restriction"]["state"] == "in_force", "the restriction is why the money moved"
    assert after_board["totals"]["revenue"] != pytest.approx(before["revenue"], abs=0.005), (
        "a save that changed nothing would not be evidence of anything"
    )

    # The inverse the surface offers, taken by the record's own two ids and
    # nothing else: no break, no session, no memory of the save.
    client.delete(f"/api/constraints/{quote(str(record['constraint_id']), safe='')}")
    client.delete(f"/api/breaks/{quote(record['break_id'], safe='')}/placement")
    restored = client.get("/api/plan/day", params={"day": day}).json()
    assert restored["totals"]["revenue"] == pytest.approx(before["revenue"], abs=0.005)
    assert restored["totals"]["breaks"] == before["breaks"]
    assert restored["unbound_placements"] == [], "the record is gone, so the row goes with it"


def test_the_surface_renders_a_route_back_for_a_record_with_no_break_to_hang_from():
    """The payload alone is not the fix. The board has to draw the way back."""
    readout = (ROOT / "tv-break-dashboard" / "src" / "plan" / "day" / "DayBoardReadout.jsx").read_text(encoding="utf-8")
    assert "export function StrandedPlacements" in readout
    assert "onClick={() => setPendingRemoval(record)}" in readout, "each row opens review for its own record"
    assert "onRemove(record);" in readout, "only the reviewed record reaches the removal callback"
    assert "Remove the saved placement" in readout and "הסרת הנעיצה השמורה" in readout
    assert "record.reason_he : record.reason" in readout, "the reason is read in the operator's language"
    assert readout.count("{stranded}") == 2, "a record is as real before the first score as after it"

    board = BOARD_JSX.read_text(encoding="utf-8")
    assert "unbound={board.unbound_placements}" in board
    assert "onRemoveUnbound={removeUnboundPlacement}" in board
    assert "function removeUnboundPlacement(record)" in board
    assert board.count("predicted: inverseOf(settlement),") == 3, "it settles like the save it reverses"

    acts = (ROOT / "tv-break-dashboard" / "src" / "plan" / "day" / "day-board-writes.js").read_text(encoding="utf-8")
    assert "export async function removeUnboundPlacement({" in acts
    assert "const inverse = inverseOfRecord(record);" in acts
    assert "await undoBreakPlacement(inverse);" in acts, "the same inverse the chip control performs"


def test_the_settlement_reports_agreement_and_divergence_from_the_same_rule():
    """Both verdicts, on figures chosen so each branch is genuinely exercised."""
    agreed = node_settlement({
        "act": "save",
        "basis": {"channel": "רשת 13", "day": "2024-11-01"},
        "before": {"revenue": 1067845.55, "retention": 0.947698, "breaks": 80, "ad_seconds": 9600.0, "gold_breaks": 0},
        "after": {"revenue": 1067845.55, "retention": 0.947698, "breaks": 80, "ad_seconds": 9600.0, "gold_breaks": 0},
        "beforeBreaks": [], "afterBreaks": [], "predictedRevenue": 0.0,
    })
    assert agreed["verdict"] == "agreed"
    assert agreed["realised"]["revenue"] == 0

    diverged = node_settlement({
        "act": "save",
        "basis": {"channel": "רשת 13", "day": "2024-11-01"},
        "before": {"revenue": 1067845.55, "retention": 0.947698, "breaks": 80, "ad_seconds": 9600.0, "gold_breaks": 0},
        "after": {"revenue": 1037270.00, "retention": 0.950056, "breaks": 80, "ad_seconds": 9600.0, "gold_breaks": 0},
        "beforeBreaks": [{"break_id": "a", "segment_id": "s1", "start_seconds": 100, "duration_seconds": 120}],
        "afterBreaks": [{"break_id": "a", "segment_id": "s1", "start_seconds": 400, "duration_seconds": 120}],
        "predictedRevenue": 0.0,
    })
    assert diverged["verdict"] == "diverged"
    assert diverged["realised"]["revenue"] == pytest.approx(-30575.55, abs=0.01)
    assert diverged["difference"] == pytest.approx(-30575.55, abs=0.01)
    assert diverged["rearranged"] == {"moved": 1, "added": 0, "removed": 0, "programmes": 1, "changed": 1}

    unknown = node_settlement({
        "act": "undo",
        "basis": {"channel": "רשת 13", "day": "2024-11-01"},
        "before": {"revenue": 1037270.00, "retention": 0.950056, "breaks": 80, "ad_seconds": 9600.0, "gold_breaks": 0},
        "after": {"revenue": 1067845.55, "retention": 0.947698, "breaks": 80, "ad_seconds": 9600.0, "gold_breaks": 0},
        "beforeBreaks": [], "afterBreaks": [], "predictedRevenue": None,
    })
    assert unknown["verdict"] == "unknown", "an absent prediction is not a prediction of zero"


def test_the_board_keeps_the_totals_it_had_so_a_re_read_cannot_hide_the_cost():
    """The defect this closes was structural, so the structure is asserted."""
    board = BOARD_JSX.read_text(encoding="utf-8")
    assert "const priorTotals = board.totals;" in board
    assert "const fresh = await load(board.day);" in board
    assert "setSettlement(settlementOf({" in board
    assert "<DayBoardSettlement" in board
    assert "return payload;" in board, "the loader has to hand the fresh day back to the caller"
    # The settlement is its own state, so the score that re-bases after a reload
    # cannot clear it. Only opening another day or dismissing it does.
    assert board.count("setSettlement(null)") == 2
    panel = PANEL_JSX.read_text(encoding="utf-8")
    assert "Realised change" in panel and "שינוי בפועל" in panel
    assert "Predicted before the act" in panel
    assert panel.count("day-figure-scope") >= 4, "every figure on this panel prints its own scope"
    assert "exactCurrency(" in panel and "formatCurrency(" not in panel


def test_marking_a_break_gold_reports_the_count_the_rebuilt_plan_actually_carries(client, opened_day):
    """The act's own second defect: a success answered before anything was read.

    Measured before the anchor carried its clock: the route stored the override,
    the engine's re-ingest guard read the blank start as a mismatch and refused
    to bind it, every one of the 80 breaks came back ``is_gold: false``, revenue
    moved by 0.00, and the route answered ``breaks_marked: 4``.
    """
    from kairos_api.break_api import _gold_enabled

    if not _gold_enabled():
        pytest.skip("gold breaks are switched off in settings, so the act is refused by design")
    day = opened_day["day"]
    counts: dict[str, int] = {}
    for row in opened_day["breaks"]:
        counts[row["segment_id"]] = counts.get(row["segment_id"], 0) + 1
    segment_id = max(counts, key=lambda key: counts[key])
    target = next(row for row in opened_day["breaks"] if row["segment_id"] == segment_id)

    response = client.post(f"/api/breaks/{quote(target['break_id'], safe='')}/gold")
    assert response.status_code == 201
    body = response.json()
    assert body["override"]["anchor_start"], "an anchor without its clock is an anchor the engine refuses"
    assert body["override"]["anchor_date"] and body["override"]["anchor_title"]

    rebuilt = client.get("/api/plan/day", params={"day": day}).json()
    actually_gold = sum(1 for row in rebuilt["breaks"] if row["segment_id"] == segment_id and row["is_gold"])
    assert body["breaks_marked"] == actually_gold, "the route reported a count the plan does not carry"
    assert body["bound"] is (actually_gold > 0)
    if body["bound"]:
        assert rebuilt["gold"]["count"] >= actually_gold >= 1
    else:
        assert body["reason"] and body["reason_he"], "a mark that reached nothing has to say so"

    client.delete(f"/api/breaks/{quote(target['break_id'], safe='')}/gold")
    back = client.get("/api/plan/day", params={"day": day}).json()
    assert back["gold"]["count"] == opened_day["gold"]["count"]
    assert back["totals"]["revenue"] == pytest.approx(opened_day["totals"]["revenue"], abs=0.005)
