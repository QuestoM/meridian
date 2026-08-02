"""The two acts a scheduler performs on a break, and the exact undo of each.

Both write, so both run against temporary stores: the shipped
``data/manual_overrides.csv`` and the new ``data/breaks.csv`` are never touched
by this file.

The gold act is the one that needed saying out loud. The engine carries gold on
the programme segment, not on one break inside it, so marking a break gold marks
every break in its programme. The route reports how many that is instead of
letting a person discover it from the board afterwards.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

# See the helper's own docstring: the shared settings file can lose the operator
# channel while these tests run, and a skipped act test proves nothing.
from test_p3_break_store import declare_operator_channel

pytestmark = pytest.mark.realdata


@pytest.fixture(scope="module", autouse=True)
def owned_channel():
    patch = declare_operator_channel()
    yield
    if patch is not None:
        patch.undo()


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Point both operator stores at a temporary directory for this test only."""
    from kairos_api import break_store, break_store_pins
    from kairos_api import overrides as override_api

    overrides_path = tmp_path / "manual_overrides.csv"
    monkeypatch.setattr(override_api, "OVERRIDES_PATH", overrides_path)
    monkeypatch.setattr(override_api, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(override_api, "_snapshot_before_write", lambda request: None)
    monkeypatch.setattr(break_store_pins, "BREAKS_PATH", tmp_path / "breaks.csv")
    monkeypatch.setattr(break_store_pins, "BACKUP_DIR", tmp_path / "_backups")
    break_store.invalidate()
    yield tmp_path
    break_store.invalidate()


@pytest.fixture()
def client(isolated):
    from kairos_api.break_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def first_break(client):
    days = client.get("/api/plan/days").json()
    if not days["available"]:
        pytest.skip(days["reason"])
    board = client.get("/api/plan/day", params={"day": days["days"][0]}).json()
    if not board["breaks"]:
        pytest.skip("this day carries no breaks")
    return board["breaks"][0]


def path_for(break_id: str, suffix: str = "") -> str:
    return f"/api/breaks/{quote(break_id, safe='')}{suffix}"


def test_marking_a_break_gold_says_how_many_breaks_it_actually_marks(client, first_break):
    response = client.post(path_for(first_break["break_id"], "/gold"))
    assert response.status_code == 201
    body = response.json()
    assert body["scope"] == "programme"
    assert body["breaks_marked"] >= 1
    assert body["override"]["kind"] == "gold"
    assert body["override"]["target_id"] == first_break["segment_id"]


def test_the_gold_mark_comes_off_again_and_a_second_removal_is_refused(client, first_break):
    client.post(path_for(first_break["break_id"], "/gold"))
    removed = client.delete(path_for(first_break["break_id"], "/gold"))
    assert removed.status_code == 200
    assert removed.json()["removed"]
    again = client.delete(path_for(first_break["break_id"], "/gold"))
    assert again.status_code == 404


def test_a_saved_placement_records_the_restriction_that_carries_it(client, first_break):
    """The record is what makes an undo exact instead of a guess."""
    from kairos_api import break_store_pins

    response = client.post(path_for(first_break["break_id"], "/placement"), json={
        "constraint_id": "c-123",
        "offset_seconds": first_break["offset_seconds"] + 60,
        "duration_seconds": first_break["duration_seconds"],
        "note": "moved for the programming representative",
    })
    assert response.status_code == 201
    record = response.json()
    assert record["constraint_id"] == "c-123"
    assert record["break_id"] == first_break["break_id"]
    assert record["saved_at"]

    saved = break_store_pins.for_day(first_break["day"])
    assert first_break["break_id"] in saved

    dropped = client.delete(path_for(first_break["break_id"], "/placement"))
    assert dropped.status_code == 200
    assert dropped.json()["forgotten"]["constraint_id"] == "c-123"
    assert not break_store_pins.for_day(first_break["day"])


def test_saving_the_same_break_twice_keeps_one_record_not_two(client, first_break):
    from kairos_api import break_store_pins

    for constraint in ("c-1", "c-2"):
        client.post(path_for(first_break["break_id"], "/placement"), json={
            "constraint_id": constraint,
            "offset_seconds": 10,
            "duration_seconds": first_break["duration_seconds"],
        })
    records = break_store_pins.records()
    assert len(records) == 1
    assert records[0]["constraint_id"] == "c-2"


def test_the_board_marks_a_saved_break_as_the_operators_own_placement(client, first_break):
    client.post(path_for(first_break["break_id"], "/placement"), json={
        "constraint_id": "c-9",
        "offset_seconds": first_break["offset_seconds"],
        "duration_seconds": first_break["duration_seconds"],
    })
    board = client.get("/api/plan/day", params={"day": first_break["day"]}).json()
    row = next(item for item in board["breaks"] if item["break_id"] == first_break["break_id"])
    assert row["placement_source"] == "operator"
    assert row["saved_placement"]["constraint_id"] == "c-9"
    others = [item for item in board["breaks"] if item["break_id"] != first_break["break_id"]]
    assert all(item["placement_source"] == "plan" for item in others)


def test_the_inverse_of_a_save_survives_the_session_that_performed_it(client, first_break):
    """A reload must not take away the only route back from an act that spent money.

    The board's undo stack lives in one browser tab. The restriction and the
    record live on disk, so after a reload the surface has to rebuild the offer
    from the payloads alone. Both payloads are checked through a client that
    never saw the save: the board row a chip is drawn from, and the drawer a
    break opens into.
    """
    from fastapi import FastAPI as FreshApp
    from fastapi.testclient import TestClient as FreshClient

    from kairos_api.break_api import router as fresh_router

    saved = client.post(path_for(first_break["break_id"], "/placement"), json={
        "constraint_id": "0679cf29dc86",
        "offset_seconds": first_break["offset_seconds"] + 60,
        "duration_seconds": first_break["duration_seconds"],
    })
    assert saved.status_code == 201

    app = FreshApp()
    app.include_router(fresh_router)
    reloaded = FreshClient(app)

    board = reloaded.get("/api/plan/day", params={"day": first_break["day"]}).json()
    row = next(item for item in board["breaks"] if item["break_id"] == first_break["break_id"])
    assert row["saved_placement"]["constraint_id"] == "0679cf29dc86", "the chip cannot offer what the payload omits"

    detail = reloaded.get(path_for(first_break["break_id"])).json()
    assert detail["placement"]["source"] == "operator"
    assert detail["placement"]["saved_placement"]["constraint_id"] == "0679cf29dc86"

    # And the inverse itself still works from that fresh session.
    assert reloaded.delete(path_for(first_break["break_id"], "/placement")).status_code == 200
    back = reloaded.get("/api/plan/day", params={"day": first_break["day"]}).json()
    freed = next(item for item in back["breaks"] if item["break_id"] == first_break["break_id"])
    assert freed["placement_source"] == "plan"
    assert not freed["saved_placement"]


def test_a_placement_on_a_break_that_does_not_exist_is_refused(client):
    response = client.post(path_for("2024-11-01|nowhere|999~1", "/placement"), json={
        "constraint_id": "c-0", "offset_seconds": 0, "duration_seconds": 120,
    })
    assert response.status_code in {404, 503}
