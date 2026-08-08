"""The error list and the lock: verification, then finalising.

Every write here runs against a temporary order register, so the shipped
``data/break_pod_order.csv`` is never created, read or changed by this file.
"""

from __future__ import annotations

from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

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
    from kairos_api import break_api_pod_order as order_store

    monkeypatch.setattr(order_store, "ORDER_PATH", tmp_path / "break_pod_order.csv")
    monkeypatch.setattr(order_store, "BACKUP_DIR", tmp_path / "_backups")
    return tmp_path


@pytest.fixture()
def client(isolated):
    from kairos_api.break_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture()
def day_pods(client):
    payload = client.get("/api/breaks/pods").json()
    if not payload["available"]:
        pytest.skip(payload["reason"])
    return payload


def _biggest(day_pods):
    return max(day_pods["pods"], key=lambda pod: pod["arithmetic"]["spot_count"])


def _by_clock(day_pods, clock):
    for record in day_pods["pods"]:
        if record["break_start_clock"] == clock:
            return record
    pytest.skip(f"the traffic file declares no break at {clock}")
    return None


def test_the_file_order_carries_zero_position_violations_on_every_pod(day_pods):
    """The check must never manufacture a false alarm on data nobody reordered."""
    for record in day_pods["pods"]:
        assert record["positions"]["violation_count"] == 0, record["pod_id"]
        assert record["positions"]["violations"] == []


def test_a_reorder_that_moves_a_bought_position_is_reported_as_a_violation(client, day_pods):
    """Position 1 is a rank among the priced spots, not the pod's own sequence.

    On the shipped break at 20:40:09 six unpositioned billboards air before
    position 1, so the spot bought as position 1 is ranked first only among
    the spots that carry a real position. Moving it out of that rank is the one
    error a traffic log exists to catch, and the surface must say so rather
    than accept it silently.
    """
    target = _by_clock(day_pods, "20:40:09")
    keys = [spot["spot_key"] for spot in target["spots"]]
    ranked = [spot for spot in target["spots"] if spot["position"]["kind"] in ("ordinal", "last")]
    assert len(ranked) >= 2, "this pod needs at least two positioned spots to drive this test"
    first_ranked_key = ranked[0]["spot_key"]
    second_ranked_key = ranked[1]["spot_key"]
    wanted = [second_ranked_key if key == first_ranked_key else (first_ranked_key if key == second_ranked_key else key) for key in keys]
    encoded = quote(target["pod_id"], safe="")

    response = client.put(f"/api/breaks/pod/{encoded}/order", json={"spot_keys": wanted})
    assert response.status_code == 200, response.text
    served = response.json()["pod"]
    assert served["positions"]["violation_count"] >= 2, served["positions"]
    kinds = {item["kind"] for item in served["verification"]["errors"]}
    assert "position_order" in kinds
    assert served["verification"]["count"] >= served["positions"]["violation_count"]


def test_locking_freezes_the_current_order_and_refuses_a_further_write(client, day_pods):
    target = _biggest(day_pods)
    encoded = quote(target["pod_id"], safe="")

    response = client.put(f"/api/breaks/pod/{encoded}/lock")
    assert response.status_code == 200, response.text
    locked_pod = response.json()["pod"]
    assert locked_pod["order"]["locked"] is True
    assert locked_pod["order"]["locked_at"]

    reread = client.get(f"/api/breaks/pod/{encoded}").json()
    assert reread["order"]["locked"] is True

    keys = [spot["spot_key"] for spot in target["spots"]]
    refused = client.put(f"/api/breaks/pod/{encoded}/order", json={"spot_keys": list(reversed(keys))})
    assert refused.status_code == 423, refused.text

    relocked = client.put(f"/api/breaks/pod/{encoded}/lock")
    assert relocked.status_code == 409

    unlocked = client.delete(f"/api/breaks/pod/{encoded}/lock")
    assert unlocked.status_code == 200, unlocked.text
    assert client.get(f"/api/breaks/pod/{encoded}").json()["order"]["locked"] is False
    assert client.delete(f"/api/breaks/pod/{encoded}/lock").status_code == 404


def test_unlocking_leaves_the_frozen_order_exactly_as_it_stood(client, day_pods):
    """A pod an operator genuinely reordered keeps that order, and its own note
    and save time, straight through a lock and an unlock. Locking must touch
    only the lock columns of a row an operator already wrote.
    """
    target = _biggest(day_pods)
    keys = [spot["spot_key"] for spot in target["spots"]]
    if len(keys) < 2:
        pytest.skip("the largest pod on this day holds fewer than two spots")
    encoded = quote(target["pod_id"], safe="")
    saved = client.put(
        f"/api/breaks/pod/{encoded}/order",
        json={"spot_keys": list(reversed(keys)), "note": "swapped for the client's own request"},
    ).json()
    saved_at = saved["pod"]["order"]["saved_at"]
    assert saved_at
    client.put(f"/api/breaks/pod/{encoded}/lock")
    client.delete(f"/api/breaks/pod/{encoded}/lock")
    reread = client.get(f"/api/breaks/pod/{encoded}").json()
    assert [spot["spot_key"] for spot in reread["spots"]] == list(reversed(keys))
    assert reread["order"]["locked"] is False
    assert reread["order"]["state"] == "operator"
    assert reread["order"]["saved_at"] == saved_at, "locking must not restate when the operator's own order was saved"
    assert reread["order"]["note"] == "swapped for the client's own request", "locking must not erase the operator's own note"


def test_a_locked_pod_refuses_the_revert_as_firmly_as_it_refuses_a_write(client, day_pods):
    """Back to the traffic file order is a change of order like any other.

    It is the larger one: dropping the row takes the frozen order away and
    clears the lock with it, so allowing it on a locked pod would be a way to
    unfinalise a pod without ever pressing unlock.
    """
    target = _biggest(day_pods)
    keys = [spot["spot_key"] for spot in target["spots"]]
    if len(keys) < 2:
        pytest.skip("the largest pod on this day holds fewer than two spots")
    encoded = quote(target["pod_id"], safe="")
    client.put(f"/api/breaks/pod/{encoded}/order", json={"spot_keys": list(reversed(keys)), "note": "kept for the client"})
    client.put(f"/api/breaks/pod/{encoded}/lock")

    refused = client.delete(f"/api/breaks/pod/{encoded}/order")
    assert refused.status_code == 423, refused.text

    still = client.get(f"/api/breaks/pod/{encoded}").json()
    assert still["order"]["locked"] is True
    assert still["order"]["state"] == "operator"
    assert still["order"]["note"] == "kept for the client"
    assert [spot["spot_key"] for spot in still["spots"]] == list(reversed(keys))

    client.delete(f"/api/breaks/pod/{encoded}/lock")
    assert client.delete(f"/api/breaks/pod/{encoded}/order").status_code == 200


def test_locking_an_untouched_pod_never_manufactures_an_operator_order(client, day_pods):
    """A lock and an unlock with no reorder ever performed must leave the pod
    exactly as the file declares it, and leave the register holding no row at
    all, rather than attributing a decision to an operator nobody was.
    """
    from kairos_api import break_api_pod_order as order_store

    target = _biggest(day_pods)
    encoded = quote(target["pod_id"], safe="")
    assert order_store.stored(target["pod_id"]) is None, "an untouched pod must start with no saved order"

    locked = client.put(f"/api/breaks/pod/{encoded}/lock").json()
    assert locked["pod"]["order"]["state"] == "file"
    assert locked["pod"]["order"]["locked"] is True
    assert [spot["spot_key"] for spot in locked["pod"]["spots"]] == [spot["spot_key"] for spot in target["spots"]]

    client.delete(f"/api/breaks/pod/{encoded}/lock")
    reread = client.get(f"/api/breaks/pod/{encoded}").json()
    assert reread["order"]["state"] == "file"
    assert reread["order"]["locked"] is False
    assert order_store.stored(target["pod_id"]) is None, "unlocking a pod nobody reordered must leave no row behind"


def test_locking_a_stale_operator_order_keeps_it_rather_than_destroying_it(client, day_pods):
    """A saved order whose fingerprint has moved underneath it is not applied,
    because a stale order over a pod that changed would put a spot in a
    position nobody chose. But the operator's keys, note and save time are a
    real decision that was made, and locking a pod in that state must freeze
    the file's own order without throwing that decision away. Unlocking
    afterwards must leave it exactly where it stood, the same as it does for
    a fresh operator order.
    """
    from kairos_api import break_api_pod_order as order_store

    target = _biggest(day_pods)
    keys = [spot["spot_key"] for spot in target["spots"]]
    if len(keys) < 2:
        pytest.skip("the largest pod on this day holds fewer than two spots")
    encoded = quote(target["pod_id"], safe="")
    client.put(
        f"/api/breaks/pod/{encoded}/order",
        json={"spot_keys": list(reversed(keys)), "note": "recorded before the file moved under it"},
    )
    saved_before = order_store.stored(target["pod_id"])
    assert saved_before is not None
    saved_at = saved_before["saved_at"]

    # Simulate the daily traffic file changing under the saved order, the way
    # tomorrow's file genuinely would, without touching any file on disk.
    frame = order_store._load_frame()
    mask = frame["pod_id"].astype(str) == target["pod_id"]
    frame.loc[mask, "fingerprint"] = "a-fingerprint-this-pod-no-longer-carries"
    order_store._write_frame(frame)

    stale = client.get(f"/api/breaks/pod/{encoded}").json()
    assert stale["order"]["state"] == "stale", "the moved fingerprint must be honoured before the lock too"

    locked = client.put(f"/api/breaks/pod/{encoded}/lock").json()["pod"]
    assert locked["order"]["state"] == "stale", "a stale order must not be applied just because it was locked"
    assert locked["order"]["locked"] is True
    assert [spot["spot_key"] for spot in locked["spots"]] == keys, "the file's own order is what a stale lock freezes"

    row = order_store.stored(target["pod_id"])
    assert row is not None, "locking a stale row must not delete the operator's own decision"
    assert row["spot_keys"] == order_store.KEY_SEPARATOR.join(reversed(keys))
    assert row["saved_at"] == saved_at
    assert row["note"] == "recorded before the file moved under it"

    client.delete(f"/api/breaks/pod/{encoded}/lock")
    after_unlock = order_store.stored(target["pod_id"])
    assert after_unlock is not None, "unlocking a stale row must not delete the operator's own decision either"
    assert after_unlock["spot_keys"] == order_store.KEY_SEPARATOR.join(reversed(keys))
    assert after_unlock["saved_at"] == saved_at
    assert after_unlock["note"] == "recorded before the file moved under it"
    assert after_unlock["locked"] == ""
    reread = client.get(f"/api/breaks/pod/{encoded}").json()
    assert reread["order"]["state"] == "stale"
    assert reread["order"]["locked"] is False


def test_a_locked_stale_order_still_says_when_it_was_locked_and_by_whom(client, day_pods):
    """The audit line of the finalising act must survive the stale branch.

    The two other branches of ``applied`` already carry ``locked_at`` and
    ``locked_by``; the stale one carried ``locked`` alone, so a pod finalised
    at a real moment rendered as a lock with no moment and no actor while the
    register held both. Measured on the shipped file before this was pinned:
    the row read ``locked_at 2026-08-08T18:27:52`` and the banner printed the
    word locked and nothing else.
    """
    from kairos_api import break_api_pod_order as order_store

    target = _biggest(day_pods)
    keys = [spot["spot_key"] for spot in target["spots"]]
    if len(keys) < 2:
        pytest.skip("the largest pod on this day holds fewer than two spots")
    encoded = quote(target["pod_id"], safe="")
    client.put(
        f"/api/breaks/pod/{encoded}/order",
        json={"spot_keys": list(reversed(keys)), "note": "the reason this order was chosen"},
    )
    frame = order_store._load_frame()
    mask = frame["pod_id"].astype(str) == target["pod_id"]
    frame.loc[mask, "fingerprint"] = "a-fingerprint-this-pod-no-longer-carries"
    order_store._write_frame(frame)

    locked = client.put(f"/api/breaks/pod/{encoded}/lock").json()["pod"]
    order = locked["order"]
    assert order["state"] == "stale"
    assert order["locked"] is True
    row = order_store.stored(target["pod_id"])
    assert order["locked_at"] == row["locked_at"] and order["locked_at"], "a lock with no moment cannot be audited"
    assert order["locked_by"] == row["locked_by"]
    assert order["note"] == "the reason this order was chosen", "the operator's own reason is readable in every state"

    client.delete(f"/api/breaks/pod/{encoded}/lock")
    cleared = client.get(f"/api/breaks/pod/{encoded}").json()["order"]
    assert cleared["locked"] is False
    assert cleared["locked_at"] == "" and cleared["locked_by"] == ""


def test_the_2253_pod_verification_names_the_one_real_disagreement(day_pods):
    record = _by_clock(day_pods, "22:53:49")
    errors = record["verification"]["errors"]
    copy_errors = [item for item in errors if item["kind"] == "copy_length"]
    assert len(copy_errors) == 1
    assert copy_errors[0]["detail"] and copy_errors[0]["detail_he"]
    assert record["verification"]["count"] == 1


def test_the_boundary_note_names_the_break_start_column(day_pods):
    for record in day_pods["pods"][:1]:
        assert record["boundary"]["value"] == "שעת התחלת ברייק"
        assert record["boundary"]["basis"] and record["boundary"]["basis_he"]
