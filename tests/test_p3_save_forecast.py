"""The prediction a scheduler reads before the click, held against the real save.

The day board answers a drag in under a millisecond because it scores the
arrangement on screen while holding the break counts the plan already chose. That
answer cannot see what a save does. A save writes a restriction, the engine then
plans the whole day again with it in force, and on that second run it is free to
place the rest of the day differently.

Measured on ``רשת 13 / 2024-11-01`` over HTTP before this route existed: pinning a
break at exactly the offset and duration the plan had already given it reads 0.00
on the cheap score and moves the day from 1,067,845.55 to 1,037,270.00, a fall of
30,575.55 ILS. One ArrowRight on ``001~2`` reads 0.00 and costs 47,444.20, with the
day falling from 80 breaks to 78. Nothing on the surface said so before the click.

``POST /api/plan/day/save-effect`` runs that second plan with the restrictions a
save would write and with nothing written. This file asserts the only property
that makes such a figure worth showing: it equals what the written save actually
does, to the cent and to the break, and the check itself leaves every store as it
found it.
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
SRC = ROOT / "tv-break-dashboard" / "src" / "plan" / "day"
SETTLEMENT_JS = SRC / "day-board-settlement.js"


@pytest.fixture(scope="module", autouse=True)
def owned_channel():
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


def client_predicate(programme: dict) -> dict:
    """The predicate ``day-board-actions.js`` writes, on the frozen contract."""
    return {
        "combinator": "and",
        "conditions": [
            {"field": "date", "operator": "is", "value": programme["day"]},
            {"field": "programme", "operator": "is", "value": programme["title"]},
            {"field": "hour", "operator": "eq", "value": int(programme["start_seconds"] // 3600) % 24},
        ],
    }


def node_settlement(payload: dict) -> dict:
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


def test_the_unwritten_prediction_equals_the_written_save_to_the_cent(client, opened_day):
    """The one property that makes a pre-save figure worth putting on a screen."""
    day = opened_day["day"]
    before = opened_day["totals"]
    target = opened_day["breaks"][0]
    programme = next(row for row in opened_day["programmes"] if row["segment_id"] == target["segment_id"])
    moves = [{
        "break_id": target["break_id"],
        "offset_seconds": target["offset_seconds"],
        "duration_seconds": target["duration_seconds"],
        "is_gold": target["is_gold"],
    }]

    cheap = client.post("/api/plan/day/score", json={"day": day, "moves": moves}).json()
    measured = client.post("/api/plan/day/save-effect", json={"day": day, "moves": moves}).json()
    assert measured["measured"] is True
    assert measured["engine_ms"] > 0
    assert measured["before"] == before, "the forecast starts from the day the board is showing"
    assert measured["restrictions"][0]["where"] == client_predicate(programme), (
        "the airing the forecast prices and the airing the save binds must be the same one"
    )

    # The check writes nothing. Both operator stores are still exactly as the
    # fixture left them, which is what makes this a forecast and not a save.
    assert client.get("/api/constraints").json()["constraints"] == []
    assert client.get("/api/plan/day", params={"day": day}).json()["unbound_placements"] == []

    created = client.post("/api/constraints", json={
        "scope_type": "always",
        "effect": "fix_offset",
        "offset_seconds": round(target["offset_seconds"]),
        "duration_seconds": round(target["duration_seconds"]),
        "order_index": target["ordinal"],
        "where": client_predicate(programme),
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

    after = client.get("/api/plan/day", params={"day": day}).json()["totals"]
    assert after["revenue"] == pytest.approx(measured["after"]["revenue"], abs=0.005)
    assert after["breaks"] == measured["after"]["breaks"]
    assert after["retention"] == pytest.approx(measured["after"]["retention"], abs=1e-6)
    assert after["ad_seconds"] == pytest.approx(measured["after"]["ad_seconds"], abs=0.05)

    realised = round(after["revenue"] - before["revenue"], 2)
    # The cheap score is not wrong, it answers a different question, and this is
    # the gap that made a scheduler click Save with no warning. Asserted rather
    # than described, so a future engine change that closes it is noticed.
    if abs(realised) > 0.005:
        assert abs(cheap["delta"]["revenue"] - realised) > 0.005, (
            "the cheap score now agrees with the save, so the warning copy wants re-measuring"
        )

    # With the measured figure as the prediction, the settlement the operator sees
    # after the save reports agreement rather than naming a divergence.
    settled = node_settlement({
        "act": "save",
        "basis": measured["basis"],
        "before": before,
        "after": after,
        "beforeBreaks": [],
        "afterBreaks": [],
        "predictedRevenue": measured["delta"]["revenue"],
    })
    assert settled["verdict"] == "agreed", settled

    client.delete(f"/api/constraints/{quote(str(constraint_id), safe='')}")
    client.delete(f"/api/breaks/{quote(target['break_id'], safe='')}/placement")
    restored = client.get("/api/plan/day", params={"day": day}).json()["totals"]
    assert restored["revenue"] == pytest.approx(before["revenue"], abs=0.005)
    assert restored["breaks"] == before["breaks"]


def test_the_forecast_refuses_a_break_that_is_not_in_this_day(client, opened_day):
    """An edit the plan cannot resolve is a 404, never a figure for something else."""
    response = client.post("/api/plan/day/save-effect", json={
        "day": opened_day["day"],
        "moves": [{"break_id": f"{opened_day['day']}|{opened_day['operator_channel']}|999~1"}],
    })
    assert response.status_code == 404


def test_the_surface_says_a_save_re_plans_the_day_before_the_click():
    """The warning stands where the money is, and the check replaces it in place."""
    readout = (SRC / "DayBoardReadout.jsx").read_text(encoding="utf-8")
    assert "export function SaveForecast" in readout
    assert "Check what saving would do" in readout and "בדיקה מה תעשה השמירה" in readout
    assert "The change above holds the break counts the plan already chose." in readout[:readout.index("export function violationLabel")]
    assert "plans the whole day again with it in force" in readout
    assert "Nothing has been written yet." in readout, "a forecast has to say it is one"
    assert "מספר הברייקים שהתוכנית כבר בחרה" in readout, "the warning is in Hebrew too"

    forecast = (SRC / "day-board-forecast.js").read_text(encoding="utf-8")
    assert "useEffect(() => { setForecast(null); }, [edits, board]);" in forecast, (
        "a forecast describes one arrangement and must die with it"
    )
    assert "export function predictionFor" in forecast

    board = (SRC / "DayBoard.jsx").read_text(encoding="utf-8")
    assert "predicted: predictionFor(forecast, score)," in board, (
        "the save settles against the measured prediction when there is one"
    )
    assert "onCheck={checkSaveEffect}" in board

    actions = (SRC / "day-board-actions.js").read_text(encoding="utf-8")
    assert "/api/plan/day/save-effect" in actions
