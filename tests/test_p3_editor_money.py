"""The money the editor's own save moves, on the surface a scheduler reaches.

The defect this file closes was measured by a critic on the shipped product.
The week board's tab יום אחד, לעריכה is where the scheduler door lands, and its
panel contained zero occurrences of the shekel sign. One chip dragged 02:29:00 to
02:22:00, then שמור כנעיצה, 605 ms: the day fell from 1,062,669.88 to
1,037,270.00, which is 25,399.88 ILS and 2.39 per cent of the day, with no figure
on that screen before or after, no preview of what the save would do, and no
inverse control anywhere on it. JS-3's own Done clause is that the money it cost
or earned is on screen and that it can be undone, and neither half was true.

Nothing new was invented to close it. The editor now drives the day board's own
three seams: ``scoreDay`` for what the arrangement is worth, ``saveEffect`` for
what the save would do measured before it is written, and two reads of the day
either side of the write for what it did. This file asserts the property that
makes such figures worth printing: the preview equals the written save, on the
editor's own rows, to the cent and to the break.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import pytest

# The editor's real rows and the real anchors behind them, built from the two
# routes the surface itself calls. See that module for why this is not a fixture
# anybody typed.
from test_p3_save_scope import editor_rows, node_pin_bodies, read

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
MONEY_JS = SRC / "plan" / "day" / "schedule-editor-money.js"


@pytest.fixture(scope="module", autouse=True)
def owned_channel():
    from test_p3_break_store import declare_operator_channel

    patch = declare_operator_channel()
    yield
    if patch is not None:
        patch.undo()


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    from kairos_api import break_store, break_store_pins
    from kairos_api import constraints as constraints_api

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
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kairos_api.break_api import router as break_router
    from kairos_api.constraints import router as constraint_router

    app = FastAPI()
    app.include_router(break_router)
    app.include_router(constraint_router)
    return TestClient(app)


def node_money(body: str) -> dict:
    """Run the shipped money module in node and return what it computed.

    The module the operator's browser imports is the module asserted here. A
    python re-implementation of the move list would only prove that two pieces of
    test code agree with each other.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    script = f"const m = await import({json.dumps(MONEY_JS.as_uri())});\n{body}"
    result = subprocess.run(
        # shell/bidi and shell/dates are real shell primitives the shipped module
        # imports; this loader hook resolves them to the real compiled files.
        ["node", "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def editor_moves(rows: list[dict]) -> dict:
    """The move list the shipped module builds from real editor rows.

    ``stateOf`` and ``targetFor`` are the editor's own two functions, handed in
    exactly as the component hands them in: the first restates the dragged
    position and the second resolves the row to its addressable break through the
    anchors the page already loaded.
    """
    lanes = [{"items": [row["item"] for row in rows]}]
    edits = {row["item"]["id"]: True for row in rows}
    payload = {
        "lanes": lanes,
        "edits": edits,
        "state": {row["item"]["id"]: [row["start_sec"], row["duration_sec"]] for row in rows},
        "segments": {row["item"]["id"]: row["segment_id"] for row in rows},
    }
    return node_money(
        f"const p = {json.dumps(payload, ensure_ascii=False)};"
        "const pin = await import(new URL('./schedule-editor-pin.js', "
        f"{json.dumps(MONEY_JS.as_uri())}));"
        "const stateOf = (item) => ({ startSec: p.state[item.id][0], durationSec: p.state[item.id][1] });"
        "const targetFor = (item, startSec, durationSec) => {"
        "  const segment = p.segments[item.id];"
        "  return segment ? pin.pinTarget(item, startSec, durationSec, segment) : null;"
        "};"
        "process.stdout.write(JSON.stringify(m.pendingMoves(p.lanes, p.edits, stateOf, targetFor)));"
    )


@pytest.mark.realdata
def test_the_editor_s_own_rows_become_the_score_endpoint_s_move_list():
    """An editor row is a display key, and the plan addresses a break by id.

    The bridge is the segment anchors the page already loads, so the move list
    the shipped module builds has to carry real ``<segment_id>~<ordinal>`` ids
    and the offset the drag produced, or every figure on the panel is about a
    break nobody moved.
    """
    rows = editor_rows()
    if not rows:
        pytest.skip("the editor has no breaks to draw, so there is nothing to score")
    measured = editor_moves(rows[:1])
    assert measured["day"] == rows[0]["item"]["date"]
    assert len(measured["moves"]) == 1
    move = measured["moves"][0]
    assert move["break_id"].startswith(f"{rows[0]['item']['date']}|")
    assert "~" in move["break_id"], "the plan's identity is segment id and ordinal"
    assert move["offset_seconds"] == rows[0]["start_sec"] - rows[0]["item"]["program_start_sec"]
    assert move["duration_seconds"] == rows[0]["duration_sec"]
    assert move["is_gold"] is None, "the editor has no gold act, so it asserts nothing about gold"
    assert measured["unaddressable"] == [] and measured["otherDays"] == []


def test_a_row_the_plan_cannot_address_is_named_rather_than_dropped():
    """A figure that silently left an edit out would be a wrong figure.

    Two cases produce one: a row whose anchor resolves no segment, and a row on a
    second broadcast day, because a score is one day's answer. Both come back by
    name so the surface can say what is not in the total.
    """
    measured = node_money("""
      const lanes = [{ items: [
        { id: 'a', date: '2024-11-01', program_title: 'קליפים', program_start_sec: 3600, break_num_in_program: 1 },
        { id: 'b', date: '2024-11-01', program_title: 'ללא מקטע', program_start_sec: 7200, break_num_in_program: 1 },
        { id: 'c', date: '2024-11-02', program_title: 'יום אחר', program_start_sec: 3600, break_num_in_program: 1 },
      ] }];
      const edits = { a: true, b: true, c: true };
      const stateOf = () => ({ startSec: 3900, durationSec: 120 });
      const targetFor = (item) => (item.id === 'b' ? null : {
        item: { break_id: `${item.date}|x|001~1`, ordinal: 1 },
        programme: { day: item.date, title: item.program_title, start_seconds: item.program_start_sec },
        live: { offsetSeconds: 300, durationSeconds: 120, isGold: false },
      });
      process.stdout.write(JSON.stringify(m.pendingMoves(lanes, edits, stateOf, targetFor)));
    """)
    assert measured["day"] == "2024-11-01"
    assert [move["break_id"] for move in measured["moves"]] == ["2024-11-01|x|001~1"]
    assert measured["unaddressable"] == ["ללא מקטע"], "the row that cannot be saved is named"
    assert measured["otherDays"] == ["2024-11-02"], "and so is the day the figures do not cover"


@pytest.mark.realdata
def test_the_preview_the_editor_shows_equals_the_save_it_performs(client):
    """The property that makes a pre-save figure worth printing.

    Driven on the editor's own first row: ``POST /api/plan/day/save-effect`` with
    the move list the shipped module builds, then the real two-step save with the
    body the shipped module builds, then the day read back. The preview and the
    written save have to be the same number, or the panel is guessing.

    The day plan's cache is fingerprinted on the real restriction file and this
    test writes a temporary one, so each store write is followed by the same
    invalidation the placement route performs in the product.
    """
    from kairos_api import break_store

    rows = editor_rows()
    if not rows:
        pytest.skip("the editor has no breaks to draw, so there is nothing to save")
    day = rows[0]["item"]["date"]
    before = client.get("/api/plan/day", params={"day": day}).json()
    if not before["breaks"]:
        pytest.skip("this day carries no breaks")

    moves = editor_moves(rows[:1])["moves"]
    preview = client.post("/api/plan/day/save-effect", json={"day": day, "moves": moves}).json()
    assert preview["measured"] is True
    assert preview["before"]["revenue"] == pytest.approx(before["totals"]["revenue"], abs=0.005)
    assert client.get("/api/constraints").json()["constraints"] == [], (
        "a preview that wrote a restriction would not be a preview"
    )

    made = node_pin_bodies(rows[:1])[0]
    break_id = made["target"]["item"]["break_id"]
    assert break_id == moves[0]["break_id"], "one row, one break id, whichever module names it"
    created = client.post("/api/constraints", json=made["body"])
    assert created.status_code == 201, created.text
    constraint_id = created.json()["constraint_id"]
    recorded = client.post(f"/api/breaks/{quote(break_id, safe='')}/placement", json={
        "constraint_id": str(constraint_id),
        "offset_seconds": made["body"]["offset_seconds"],
        "duration_seconds": made["body"]["duration_seconds"],
    })
    assert recorded.status_code == 201, recorded.text
    break_store.invalidate()

    after = client.get("/api/plan/day", params={"day": day}).json()
    assert after["totals"]["revenue"] == pytest.approx(preview["after"]["revenue"], abs=0.005), (
        "the preview said one thing and the save did another"
    )
    assert after["totals"]["breaks"] == preview["after"]["breaks"]

    # And the inverse the settlement panel offers, by the two ids the record
    # carries, which is what the undo control sends.
    client.delete(f"/api/constraints/{quote(str(constraint_id), safe='')}")
    client.delete(f"/api/breaks/{quote(break_id, safe='')}/placement")
    break_store.invalidate()
    restored = client.get("/api/plan/day", params={"day": day}).json()["totals"]
    assert restored["revenue"] == pytest.approx(before["totals"]["revenue"], abs=0.005)
    assert restored["breaks"] == before["totals"]["breaks"]


def test_the_editor_carries_the_same_three_answers_the_day_board_carries():
    """One vocabulary and one implementation, so the two cannot diverge again.

    The critic's fix note named the exports: the board's own ``scoreDay``,
    ``saveEffect`` and ``undoBreakPlacement``. They are imported here rather than
    reimplemented, and the settlement panel with its inverse control is the
    board's own component rendered by the editor's readout.
    """
    hook = read("plan/day/schedule-editor-money.js")
    assert "import { fetchDay, saveEffect, scoreDay, undoBreakPlacement } from './day-board-actions.js';" in hook
    assert "import { inverseOf, settlementOf } from './day-board-settlement.js';" in hook
    assert "import { predictionFor } from './day-board-forecast.js';" in hook
    assert "() => undoBreakPlacement({ breakId: lastSave.breakId, constraintId: lastSave.constraintId })," in hook, (
        "the inverse is addressed by both ids the saved record carries"
    )
    assert "predictionFor(forecast, score)" in hook, "a save settles against the measured check when there was one"

    readout = read("plan/day/ScheduleEditorReadout.jsx")
    assert "<ScheduleEditorMoney money={money} locale={locale}" in readout
    assert "onUndo={money.undoLastSave}" in readout and "canUndo={money.canUndo}" in readout

    editor = read("plan/day/ScheduleEditor.jsx")
    assert "const money = useEditorMoney({ pending, locale, notify, onGlobalRefresh });" in editor
    assert "await money.saveAndSettle(item.id, target);" in editor, "the save the row performs is the settled one"
    assert "money={money}" in editor


def test_the_editor_prints_no_figure_before_the_engine_has_answered():
    """Honest math on the one screen with nothing to show yet.

    Between opening the surface and the first score there is no figure for this
    day, and a zero standing in for one would be a fabricated total. The panel
    says what it is doing instead, in both languages, and every figure it does
    print carries the channel and day it was computed on.
    """
    money = read("plan/day/ScheduleEditorMoney.jsx")
    idle = money.split("if (!score) {")[1].split("const { basis")[0]
    assert "Reading what this day is worth." in idle and "קורא כמה שווה היום הזה." in idle
    assert "exactCurrency" not in idle and "formatPercent" not in idle, (
        "the idle state may not print a figure of any kind"
    )
    # The scope is the channel and the day AND the plan the figure was computed
    # on, because this panel prices the live re-plan while the sentence above the
    # timeline counts the saved weekly plan. The basis wording itself is asserted
    # as a class in test_p3_editor_coverage.py; here it only has to be present.
    assert 'scopeWithBasis(`${basis.channel} / ${basis.day}`, LIVE_PLAN, locale)' in money
    assert money.count('className="day-figure-scope"') >= 4, "every figure names its own scope"
    assert "negative" not in money.lower()

    settlement = read("plan/day/day-board-settlement.js")
    inverse = settlement.split("export function inverseOf")[1]
    assert "return inverse === 0 ? 0 : inverse;" in inverse, (
        "an undo of a save that cost nothing predicted minus zero, which reads as a loss"
    )
