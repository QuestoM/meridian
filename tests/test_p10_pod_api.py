"""The four pod routes, the reorder and its inverse, and the competitor boundary.

The reorder writes, so it runs against a temporary register: the shipped
``data/break_pod_order.csv`` is never created, read or changed by this file.

One route ordering fact is asserted rather than assumed. ``/api/breaks/pods`` and
``/api/breaks/{break_id}`` share a prefix, and a router that registered the second
first would answer 422 on the first with the word ``pods`` parsed as a break id.
The include happens at the top of ``break_api`` for exactly that reason, so a test
holds it there.
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


def test_the_pods_route_is_not_swallowed_by_the_break_id_route(client):
    """The shared prefix, held. Without the include order this answers 422."""
    response = client.get("/api/breaks/pods")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "pods" in body and "covered_days" in body
    assert "a break id reads" not in response.text


def test_the_day_serves_every_pod_the_traffic_file_declares(day_pods):
    assert day_pods["count"] == len(day_pods["pods"]) > 0
    assert day_pods["day"] in day_pods["covered_days"]
    clocks = [pod["break_start_clock"] for pod in day_pods["pods"]]
    assert clocks == sorted(clocks), "pods came back out of time order"
    assert len(set(pod["pod_id"] for pod in day_pods["pods"])) == len(clocks)


def test_a_day_with_no_traffic_file_is_a_state_and_names_the_days_that_are_covered(client):
    payload = client.get("/api/breaks/pods", params={"day": "1999-01-01"}).json()
    assert payload["available"] is False
    assert payload["pods"] == [] and payload["count"] == 0
    assert payload["reason"] and payload["reason_he"]
    assert payload["path_forward"] and payload["path_forward_he"]
    assert payload["covered_days"]


def test_one_pod_opens_by_its_own_id_and_a_pod_that_does_not_exist_answers_404(client, day_pods):
    wanted = day_pods["pods"][0]
    response = client.get(f"/api/breaks/pod/{quote(wanted['pod_id'], safe='')}")
    assert response.status_code == 200, response.text
    assert response.json()["pod_id"] == wanted["pod_id"]
    assert client.get(f"/api/breaks/pod/{quote('1999-01-01~00:00:00', safe='')}").status_code == 404
    assert client.get(f"/api/breaks/pod/{quote('nonsense', safe='')}").status_code == 422


def _biggest(day_pods):
    return max(day_pods["pods"], key=lambda pod: pod["arithmetic"]["spot_count"])


def test_a_saved_order_is_applied_and_the_sequence_is_restated_with_it(client, day_pods):
    """The reorder, written and read back. The positions travel with the spots."""
    target = _biggest(day_pods)
    keys = [spot["spot_key"] for spot in target["spots"]]
    if len(keys) < 3:
        pytest.skip("the largest pod on this day holds fewer than three spots")
    wanted = [keys[2], keys[0], keys[1]] + keys[3:]
    encoded = quote(target["pod_id"], safe="")

    response = client.put(f"/api/breaks/pod/{encoded}/order", json={"spot_keys": wanted, "note": "traffic"})
    assert response.status_code == 200, response.text
    served = response.json()["pod"]
    assert [spot["spot_key"] for spot in served["spots"]] == wanted
    assert [spot["sequence"] for spot in served["spots"]] == list(range(1, len(wanted) + 1))
    assert served["order"]["state"] == "operator"
    assert served["order"]["reason"] and served["order"]["reason_he"]

    # The arithmetic does not move, because reordering a pod changes no length.
    assert served["arithmetic"]["declared_load"] == target["arithmetic"]["declared_load"]

    reread = client.get(f"/api/breaks/pod/{encoded}").json()
    assert [spot["spot_key"] for spot in reread["spots"]] == wanted


def test_the_inverse_of_a_saved_order_puts_the_pod_back_in_the_file_order(client, day_pods):
    target = _biggest(day_pods)
    keys = [spot["spot_key"] for spot in target["spots"]]
    if len(keys) < 2:
        pytest.skip("the largest pod on this day holds fewer than two spots")
    encoded = quote(target["pod_id"], safe="")
    reversed_keys = list(reversed(keys))

    client.put(f"/api/breaks/pod/{encoded}/order", json={"spot_keys": reversed_keys})
    assert [spot["spot_key"] for spot in client.get(f"/api/breaks/pod/{encoded}").json()["spots"]] == reversed_keys

    response = client.delete(f"/api/breaks/pod/{encoded}/order")
    assert response.status_code == 200, response.text
    back = response.json()["pod"]
    assert [spot["spot_key"] for spot in back["spots"]] == keys
    assert back["order"]["state"] == "file"
    assert client.delete(f"/api/breaks/pod/{encoded}/order").status_code == 404


def test_an_order_that_is_not_exactly_this_pod_keys_is_refused_rather_than_half_applied(client, day_pods):
    target = _biggest(day_pods)
    keys = [spot["spot_key"] for spot in target["spots"]]
    if len(keys) < 2:
        pytest.skip("the largest pod on this day holds fewer than two spots")
    encoded = quote(target["pod_id"], safe="")
    for bad in ([keys[0]], keys + ["s999"], [keys[0]] * len(keys), []):
        assert client.put(f"/api/breaks/pod/{encoded}/order", json={"spot_keys": bad}).status_code == 422
    assert client.get(f"/api/breaks/pod/{encoded}").json()["order"]["state"] == "file"


def test_a_saved_order_whose_pod_changed_underneath_it_is_stale_and_is_not_applied(client, day_pods):
    """The traffic file is replaced every day, so a saved order can outlive its pod.

    Applying it anyway would put an advertiser in a position nobody chose, so the
    file's own order is served and the record is reported stale.
    """
    from kairos_api import break_api_pod_order as order_store

    target = _biggest(day_pods)
    keys = [spot["spot_key"] for spot in target["spots"]]
    if len(keys) < 2:
        pytest.skip("the largest pod on this day holds fewer than two spots")
    encoded = quote(target["pod_id"], safe="")
    client.put(f"/api/breaks/pod/{encoded}/order", json={"spot_keys": list(reversed(keys))})

    stored = order_store.stored(target["pod_id"])
    assert stored is not None and stored["fingerprint"]
    order_store.save(target["pod_id"], list(reversed(keys)), "a-different-pod-entirely")

    served = client.get(f"/api/breaks/pod/{encoded}").json()
    assert served["order"]["state"] == "stale"
    assert served["order"]["reason"] and served["order"]["reason_he"]
    assert [spot["spot_key"] for spot in served["spots"]] == keys


def test_no_rival_channel_name_reaches_this_surface(client, day_pods):
    """The boundary, greped over the whole serialized payload rather than a field."""
    from kairos_api import break_store
    from kairos_api.core import _load_break_schedule

    owned = break_store.operator_channel()
    if not owned:
        pytest.skip("no operator channel is configured, so there is no boundary to test")
    schedule = _load_break_schedule()
    if schedule.empty or "channel" not in schedule.columns:
        pytest.skip("no saved plan, so no rival channel names to check against")
    rivals = {str(value).strip() for value in schedule["channel"].tolist() if str(value).strip()} - {owned}
    if not rivals:
        pytest.skip("the saved plan carries only the operator's own channel")
    body = client.get("/api/breaks/pods").text
    for rival in rivals:
        assert rival not in body, f"the pod payload named the rival channel {rival}"
    assert day_pods["channel"]["value"] == owned
    assert "no channel column" in day_pods["channel"]["basis"]


def test_a_planned_break_carries_its_contents_from_the_same_pod_module(client):
    """The break detail's contents field, which was a fixed unavailable state.

    The plan and the traffic file overlap on no date today, so the honest answer
    is still a state. What is asserted here is that it is a state this piece
    computed from coverage it read, naming the days that are covered, rather than
    the constant it used to be.
    """
    days = client.get("/api/plan/days").json()
    if not days["available"]:
        pytest.skip(days["reason"])
    board = client.get("/api/plan/day", params={"day": days["days"][0]}).json()
    if not board["breaks"]:
        pytest.skip("this day carries no breaks")
    detail = client.get(f"/api/breaks/{quote(board['breaks'][0]['break_id'], safe='')}").json()
    contents = detail["contents"]
    assert contents["state"] in {"real", "unavailable"}
    assert "covered_days" in contents, "contents no longer names which days a traffic file covers"
    assert contents["reason"] and contents["reason_he"]
    if contents["state"] == "unavailable":
        assert contents["spots"] == []
        assert "not modelled yet" not in contents["reason"]


def test_every_file_this_piece_owns_is_inside_the_size_law():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    owned = [
        root / "kairos_api" / "break_api.py",
        root / "kairos_api" / "break_api_pod.py",
        root / "kairos_api" / "break_api_pod_order.py",
        root / "kairos_api" / "break_api_pod_math.py",
        root / "kairos_api" / "break_api_pod_spots.py",
        root / "kairos_api" / "break_api_detail.py",
    ] + sorted((root / "tv-break-dashboard" / "src" / "plan" / "break").glob("*")) + sorted(
        (root / "tests").glob("test_p10_*.py")
    )
    oversize = {
        path.name: len(path.read_text(encoding="utf-8").splitlines())
        for path in owned
        if path.is_file() and len(path.read_text(encoding="utf-8").splitlines()) > 450
    }
    assert not oversize, f"over the 450-line law: {oversize}"
