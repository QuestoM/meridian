"""What a saved move writes, what it binds, and what the surface says it binds.

The shipped editor wrote a placement scoped to the whole broadcast date. The
restriction resolver matches a date scope against every segment on that date, so
one dragged break bound the whole day: measured through the engine's own resolver
on the 82 real segments of 2024-11-01, 82 of 82, and driven through the running
product one drag cost 789,576.18 ILS, 74.3 per cent of the day, with no record
written anywhere and so no Remove control and no route back.

Both timelines now build one body through one module and write it through one
two-step save. This file measures that body on the engine, drives the save and its
inverse through the routes, and holds the words the surface prints about what it
binds, which is not always one airing and no longer says it is.

The scaffolding here is imported by ``test_p3_direct_manipulation.py``, so the
headline binding assertion and the save that follows it measure one thing one way.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
BOARD_MODEL_JS = SRC / "plan" / "day" / "day-board-model.js"
PIN_JS = SRC / "plan" / "day" / "schedule-editor-pin.js"


def read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def node_board_model(body: str) -> dict:
    """Run the shipped board model in node and return what it computed.

    The module the operator's browser imports is the module asserted here. A
    python re-implementation would only prove that two pieces of test code agree
    with each other, which is what let the defects below ship.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    script = f"const m = await import({json.dumps(BOARD_MODEL_JS.as_uri())});\n{body}"
    result = subprocess.run(
        # shell/bidi and shell/dates are real shell primitives the shipped module
        # imports; this loader hook resolves them to the real compiled files.
        ["node", "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


# The scope a saved move binds, measured on the engine's own resolver.
#
# The stores are redirected into a temporary directory, so nothing below writes
# the operator's real placements or restrictions.


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
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kairos_api.break_api import router as break_router
    from kairos_api.constraints import router as constraint_router

    app = FastAPI()
    app.include_router(break_router)
    app.include_router(constraint_router)
    return TestClient(app)


def clock_to_seconds(clock: str) -> int:
    """The editor's own timeToSeconds, on a minute-resolution wire clock."""
    hour, _, minute = str(clock or "00:00").partition(":")
    return (int(hour) * 60 + int(minute)) * 60


def editor_rows() -> list[dict]:
    """The editor's real rows, each with the segment its anchor resolves to.

    Built from the two routes the surface itself calls, so this is the editor's
    own input and not a fixture somebody typed: ``/api/break-operations`` for the
    rows and ``/api/schedule/segments`` for the anchors. The key is the one
    ``useSegmentAnchors`` builds in the browser, channel, date and start clock.
    """
    from kairos_api.day_api import _break_operations_cached, break_operations, schedule_segments

    _break_operations_cached.cache_clear()
    operations = break_operations()
    anchors: dict[str, str] = {}
    for segment in schedule_segments()["segments"]:
        anchor = segment.get("anchor") or {}
        key = f"{segment.get('channel', '')}|{anchor.get('date') or segment.get('day', '')}|{anchor.get('start_clock', '')}"
        anchors[key] = segment["segment_id"]
    programmes = {row["key"]: row for row in operations["programs"]}
    rows = []
    for item in operations["breaks"]:
        programme = programmes.get(item["program_key"])
        if programme is None:
            continue
        start_sec = clock_to_seconds(programme["start_time"])
        rows.append({
            "item": {
                "id": item["id"],
                "date": item["date"],
                "channel": item["channel"],
                "program_title": item["program_title"],
                "program_start_sec": start_sec,
                "break_num_in_program": item["break_num_in_program"],
                "is_gold": item["is_gold"],
            },
            "segment_id": anchors.get(f"{programme['channel']}|{programme['date']}|{programme['start_time']}"),
            "start_sec": clock_to_seconds(item["start_time"]) + 60,
            "duration_sec": item["duration_sec"],
        })
    return rows


def node_pin_bodies(rows: list[dict]) -> list[dict]:
    """Run the shipped save module in node and return the bodies it produces."""
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped module cannot be executed")
    script = (
        f"const m = await import({json.dumps(PIN_JS.as_uri())});"
        f"const rows = {json.dumps(rows, ensure_ascii=False)};"
        "const out = rows.map((row) => {"
        "  const target = m.pinTarget(row.item, row.start_sec, row.duration_sec, row.segment_id);"
        "  return { target, body: m.pinBody(target) };"
        "});"
        "process.stdout.write(JSON.stringify(out));"
    )
    result = subprocess.run(
        # shell/bidi and shell/dates are real shell primitives the shipped module
        # imports; this loader hook resolves them to the real compiled files.
        ["node", "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def as_constraint(body: dict, constraint_id: str):
    """The body as the store's own row, which is what the resolver reads.

    The effect is lower-cased because that is what the write route does to it
    before the row reaches the store, so a body carrying FIX_OFFSET and a body
    carrying fix_offset are the same row on disk and must be the same row here.
    """
    from kairos.optimize.constraints_store import PlacementConstraint

    return PlacementConstraint(
        constraint_id=constraint_id,
        scope_type=body["scope_type"],
        scope_value=body.get("scope_value", ""),
        channel=body.get("channel", ""),
        effect=str(body["effect"]).strip().lower(),
        offset_seconds=float(body["offset_seconds"]),
        duration_seconds=float(body["duration_seconds"]),
        order_index=int(body["order_index"]),
        where=body.get("where"),
    )


def bound_segments(plan, body: dict) -> list[str]:
    from kairos.optimize.constraints_store import resolve_constraints

    pins, _counts, _forbids, _skipped = resolve_constraints(
        list(plan.segments), [as_constraint(body, "measure")], operator_channel=plan.channel,
    )
    return sorted(pins)


def same_airing_key(segment) -> tuple:
    """The finest thing the frozen predicate contract can say about an airing."""
    return (str(segment.day), str(segment.program_title), int(float(segment.start_seconds) // 3600) % 24)


def test_the_scope_a_save_binds_is_stated_rather_than_claimed_to_be_one():
    """The sentence beside the selection used to end in the word only.

    Measured on רשת 13 / 2024-11-01, that word was false for 18 of the 48
    break-carrying programmes, because the frozen contract cannot tell two airings
    of one title inside one hour apart. The board is served the whole day, so it
    counts and says the number; a surface that draws part of a day states the rule
    and passes no count rather than under-reporting one.
    """
    measured = node_board_model("""
      const day = [
        { segment_id: 'a', title: 'קליפים', start_seconds: 14922 },
        { segment_id: 'b', title: 'קליפים', start_seconds: 16383 },
        { segment_id: 'c', title: 'מילון היופי', start_seconds: 11425 },
        { segment_id: 'd', title: 'קליפים', start_seconds: 25000 },
      ];
      process.stdout.write(JSON.stringify({
        repeated: m.airingsBound(day, day[0]),
        alone: m.airingsBound(day, day[2]),
        nothing: m.airingsBound(day, null),
      }));
    """)
    assert measured["repeated"] == 2, "two airings of one title in one hour are two, and the surface says two"
    assert measured["alone"] == 1
    assert measured["nothing"] == 0

    actions = read("plan/day/day-board-actions.js")
    sentence = actions.split("export function scopeSentence")[1].split("export function")[0]
    assert "only" not in sentence.split("const rule =")[0], "the claim was in the words, so the words changed"
    assert "and it is the only such airing that day" in sentence, "a true only is still allowed to be said"
    assert "והיא רצועת השידור היחידה כזו ביום" in sentence
    assert "together with ${others} more airings of it in the same hour" in sentence
    assert "ואיתה עוד רצועת שידור אחת של אותה תוכנית באותה שעה" in sentence, (
        "one other airing is one airing, in a language that inflects it"
    )
    toolbar = read("plan/day/DayBoardToolbar.jsx")
    assert "scopeSentence(programme, locale, airingsBound(board.programmes, programme))" in toolbar


@pytest.mark.realdata
def test_the_editor_s_save_leaves_the_day_standing_and_its_own_way_back(client):
    """What the two bodies cost, driven through the product rather than described.

    The same drag, saved twice: once with the body the shipped module produces and
    once with the whole-date body it used to produce. The first keeps the day and
    leaves a record the Remove control reads; the second takes about three
    quarters of it and leaves no record anywhere, which is why there was no route
    back from it.

    The day plan's cache is fingerprinted on the real restriction file, and this
    test writes a temporary one, so each store write is followed by the same
    invalidation the placement route performs. In the product the fingerprint sees
    the write itself; here it cannot, and a cached day would make this test pass
    while measuring nothing.
    """
    from kairos_api import break_store

    rows = editor_rows()
    if not rows:
        pytest.skip("the editor has no breaks to draw, so there is nothing to save")
    day = rows[0]["item"]["date"]
    before = client.get("/api/plan/day", params={"day": day}).json()
    if not before["breaks"]:
        pytest.skip("this day carries no breaks")
    made = node_pin_bodies(rows[:1])[0]
    break_id = made["target"]["item"]["break_id"]

    created = client.post("/api/constraints", json=made["body"])
    assert created.status_code == 201, created.text
    constraint_id = created.json()["constraint_id"]
    recorded = client.post(f"/api/breaks/{quote(break_id, safe='')}/placement", json={
        "constraint_id": str(constraint_id),
        "offset_seconds": made["body"]["offset_seconds"],
        "duration_seconds": made["body"]["duration_seconds"],
        "is_gold": bool(made["target"]["live"]["isGold"]),
    })
    assert recorded.status_code == 201, recorded.text

    after = client.get("/api/plan/day", params={"day": day}).json()
    bound = after["restrictions"]["count"]
    assert 1 <= bound <= 4, f"the saved restriction binds {bound} of the day's programmes"
    kept = after["totals"]["revenue"] / before["totals"]["revenue"]
    assert kept > 0.95, f"one saved break cost {round(100 * (1 - kept), 1)} per cent of the day"
    saved = [row["break_id"] for row in after["breaks"] if row["saved_placement"]]
    assert saved == [break_id], "the chip has to come back saved, or the Remove control has nothing to hang off"
    assert after["unbound_placements"] == []

    client.delete(f"/api/constraints/{quote(str(constraint_id), safe='')}")
    client.delete(f"/api/breaks/{quote(break_id, safe='')}/placement")
    restored = client.get("/api/plan/day", params={"day": day}).json()["totals"]
    assert restored["revenue"] == pytest.approx(before["totals"]["revenue"], abs=0.005)
    assert restored["breaks"] == before["totals"]["breaks"]

    # The same drag at the scope this surface used to send, with nothing else changed.
    whole_date = client.post("/api/constraints", json={
        "scope_type": "date",
        "scope_value": day,
        "channel": rows[0]["item"]["channel"],
        "effect": "FIX_OFFSET",
        "offset_seconds": made["body"]["offset_seconds"],
        "duration_seconds": made["body"]["duration_seconds"],
        "order_index": made["body"]["order_index"],
    })
    assert whole_date.status_code == 201, whole_date.text
    break_store.invalidate()
    wrecked = client.get("/api/plan/day", params={"day": day}).json()
    assert wrecked["restrictions"]["count"] == len(before["programmes"]) == 82, (
        "the whole-date scope binds every programme on the day, which is the defect"
    )
    survived = wrecked["totals"]["revenue"] / before["totals"]["revenue"]
    assert survived < 0.30, "the measured cost of the old default was 74.3 per cent of the day"
    assert [row["break_id"] for row in wrecked["breaks"] if row["saved_placement"]] == [], (
        "and it wrote no record, so nothing on the board offered to reverse it"
    )
    client.delete(f"/api/constraints/{quote(str(whole_date.json()['constraint_id']), safe='')}")
    break_store.invalidate()
    back = client.get("/api/plan/day", params={"day": day}).json()["totals"]
    assert back["revenue"] == pytest.approx(before["totals"]["revenue"], abs=0.005)


def test_the_editor_offers_no_scope_wider_than_the_airing_it_is_looking_at():
    """The selector that carried the default is gone, and its words with it."""
    toolbar = read("plan/day/ScheduleEditorToolbar.jsx")
    for gone in ("scopeChoice", "Pin scope", "היקף הנעיצה", "This date", "תאריך זה"):
        assert gone not in toolbar, f"the editor still offers {gone}"
    editor = read("plan/day/ScheduleEditor.jsx")
    assert "useState('date')" not in editor and "savePin(item, 'date')" not in editor
    assert "requestPinReview(item)" in editor, "Enter reviews the one scope there is before saving"
    assert "onSave={requestPinReview}" in editor, "the row Save control enters the same review"
    assert "scopeFor={scopeFor}" in editor and "return target ? scopeSentence(target.programme, locale) : '';" in editor, (
        "what a save binds is on the row that carries the Save button"
    )
    pin = read("plan/day/schedule-editor-pin.js")
    assert "import { placementBody, saveBreakPlacement } from './day-board-actions.js';" in pin, (
        "one body builder and one write path for both timelines, never a second copy"
    )
    assert "savePinPlacement" in pin
    row = read("plan/day/ScheduleEditorRow.jsx")
    # Isolation moved to tv-break-dashboard/src/shell/bidi.jsx and stopped using
    # dir on the way. A dir attribute on an inline run fixes its internal order
    # and also re-anchors its own alignment, which drags it off the line its
    # neighbours sit on. The scope sentence is interface copy that already knows
    # its language, so it carries no dir; the figures beside it isolate through
    # the primitive. A correction, not a rename: do not put dir back on this row.
    assert '{scope && <span className="editor-row-scope">{scope}</span>}' in row
    assert "import { Figure } from '../../shell/bidi';" in row
    assert "dir=" not in row, "no run on this row sets a base direction of its own"


def test_the_editor_prints_a_currency_figure_and_an_inverse_control_after_a_save():
    """The panel that spent 25,399.88 ILS carried neither, which was the defect.

    Held at the address it was reported at. The drive is in test_p3_editor_money.
    """
    readout = read("plan/day/ScheduleEditorReadout.jsx")
    assert "<ScheduleEditorMoney money={money} locale={locale}" in readout
    assert "onUndo={money.undoLastSave}" in readout
    assert "exactCurrency(current.revenue, locale)" in read("plan/day/ScheduleEditorMoney.jsx")
    assert "ביטול השמירה הזו" in read("plan/day/DayBoardSettlement.jsx")


def test_a_restriction_that_lands_without_its_record_is_taken_back_out():
    """Half a saved move is money spent with no inverse, so it is not left standing."""
    actions = read("plan/day/day-board-actions.js")
    save = actions.split("export async function saveBreakPlacement")[1].split("export async function")[0]
    assert "await undoBreakPlacement({ breakId: item.break_id, constraintId: String(constraintId) });" in save
    assert "throw error;" in save, "and the surface is told, rather than reporting a save that half happened"


def test_a_hebrew_surface_names_the_weekday_in_hebrew():
    """Two surfaces printed the wire's English abbreviation at a person.

    The programme record read ``2024-11-01 (Fri)`` and the editor lane read
    ``רשת 13 / Fri``. The wire carries ``day`` as ``%a`` by design; the ISO date
    beside it is the honest source for a weekday in either language.
    """
    measured = node_board_model("""
      process.stdout.write(JSON.stringify({
        friday: m.weekdayName('2024-11-01', 'he'),
        fridayEn: m.weekdayName('2024-11-01', 'en'),
        saturday: m.weekdayName('2024-11-02', 'he'),
        sunday: m.weekdayName('2024-11-03', 'he'),
        nothing: m.weekdayName('', 'he'),
      }));
    """)
    assert measured["friday"] == "שישי" and measured["fridayEn"] == "Friday"
    assert measured["saturday"] == "שבת" and measured["sunday"] == "ראשון"
    assert measured["nothing"] == "", "an unparseable date names no day rather than guessing one"

    inspector = read("plan/day/ScheduleInspector.jsx")
    assert "value={dateLine(id, locale)}" in inspector
    assert "`(${id.day})`" not in inspector, "the wire's abbreviation is not a label"
    assert "const named = weekdayName(date, locale);" in inspector
    editor = read("plan/day/ScheduleEditor.jsx")
    assert "label: laneLabel(entry.channel, entry.date, laneKey, locale)," in editor
    assert "{lane.label || lane.lane}" in editor
    fmt = read("plan/day/schedule-editor-format.js")
    assert "export function laneLabel" in fmt


@pytest.mark.realdata
def test_saving_the_same_break_twice_leaves_one_restriction_and_not_two(client):
    """An update replaces the record, so it has to take the old restriction with it.

    One break carries at most one saved placement, and the record is the only
    thing that names the restriction carrying it. A second save writes a second
    restriction and replaces the record, so before this the first restriction
    stayed in force with nothing on any surface addressing it: the same class of
    defect as the whole-date scope, one size smaller. Measured here through the
    routes, which is where both surfaces meet.
    """
    rows = editor_rows()
    if not rows:
        pytest.skip("the editor has no breaks to draw, so there is nothing to save")
    made = node_pin_bodies(rows[:1])[0]
    break_id = made["target"]["item"]["break_id"]

    first = client.post("/api/constraints", json=made["body"]).json()["constraint_id"]
    opening = client.post(f"/api/breaks/{quote(break_id, safe='')}/placement", json={
        "constraint_id": str(first),
        "offset_seconds": made["body"]["offset_seconds"],
        "duration_seconds": made["body"]["duration_seconds"],
    }).json()
    assert opening["replaced"] is None, "the first save of a break replaces nothing"

    second = client.post("/api/constraints", json={**made["body"], "offset_seconds": made["body"]["offset_seconds"] + 60}).json()["constraint_id"]
    update = client.post(f"/api/breaks/{quote(break_id, safe='')}/placement", json={
        "constraint_id": str(second),
        "offset_seconds": made["body"]["offset_seconds"] + 60,
        "duration_seconds": made["body"]["duration_seconds"],
    }).json()
    assert update["replaced"]["constraint_id"] == str(first), (
        "the route has to report what it replaced, or the caller cannot reverse it"
    )
    assert len(client.get("/api/constraints").json()["constraints"]) == 2, (
        "both restrictions exist at this point, which is why the surface deletes the first"
    )

    actions = (SRC / "plan" / "day" / "day-board-actions.js").read_text(encoding="utf-8")
    save = actions.split("export async function saveBreakPlacement")[1].split("export async function")[0]
    assert "const replaced = record && record.replaced;" in save
    assert "await call(`/api/constraints/${encodeURIComponent(replaced.constraint_id)}`, { method: 'DELETE' });" in save

    # The surface's own step, performed here, and the store is back to one rule.
    client.delete(f"/api/constraints/{quote(str(first), safe='')}")
    remaining = client.get("/api/constraints").json()["constraints"]
    assert [row["constraint_id"] for row in remaining] == [second]
    client.delete(f"/api/constraints/{quote(str(second), safe='')}")
    client.delete(f"/api/breaks/{quote(break_id, safe='')}/placement")
