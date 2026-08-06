"""The console names who may run a rebuild, before the click.

The owner's third gap: the console showed a Train button to every company
account that reached it, whether or not a POST behind that button would ever
succeed. A company viewer could open the training panel, fill in the flags and
press Train, and only then read the 403 the wall had already decided at page
load. Section 4.5's ``can_edit`` contract exists exactly for this: the same
``Wall.reason`` the write route consults is stamped onto the read, so the
control renders its own permission rather than discovering it after a click.

Two routes carry this stamp, ``GET /api/model/console`` (the header, where the
owner's "who may run a rebuild" question is answered at the point the stuck
gate counts live) and ``GET /api/model/training`` (the panel with the button
itself). Both share the model wall's role gate, so both refuse a viewer for
the same reason and admit an admin for the same reason.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from kairos_api import auth_store
from kairos_api.affiliation_wall import READ_ONLY_ROLE_DETAIL

ADMIN_PASSWORD = "rootpass-1234"
VIEWER_PASSWORD = "viewerpass-123"
ROUTES = ("/api/model/console", "/api/model/training")


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.setenv("KAIROS_MODEL_RELEASES_DIR", str(tmp_path / "releases"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    yield tmp_path
    auth_store.reset_runtime_state()


def _clients(auth_env) -> dict[str, TestClient]:
    from kairos_api.server import app

    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
    admin = TestClient(app)
    signed = admin.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD})
    assert signed.status_code == 200, signed.text
    created = admin.post("/api/auth/users", json={
        "username": "view1", "password": VIEWER_PASSWORD, "role": "viewer",
        "display_name": "view1", "must_change_password": False, "affiliation": "company",
    })
    assert created.status_code == 201, created.text
    viewer = TestClient(app)
    response = viewer.post("/api/auth/login", json={"username": "view1", "password": VIEWER_PASSWORD})
    assert response.status_code == 200, response.text
    return {"admin": admin, "viewer": viewer}


def test_an_admin_reads_can_edit_true_with_no_reason(auth_env) -> None:
    clients = _clients(auth_env)
    for path in ROUTES:
        body = clients["admin"].get(path).json()
        assert body["can_edit"] is True, f"{path}: {body}"
        assert "can_edit_reason" not in body, f"{path}: {body}"


def test_a_company_viewer_reads_can_edit_false_with_the_wall_s_own_reason(auth_env) -> None:
    clients = _clients(auth_env)
    for path in ROUTES:
        body = clients["viewer"].get(path).json()
        assert body["can_edit"] is False, f"{path}: {body}"
        assert body["can_edit_reason"] == READ_ONLY_ROLE_DETAIL, f"{path}: {body}"


def test_the_stamped_reason_is_the_wall_s_own_role_detail(auth_env) -> None:
    """The whole point of the contract: the reason cannot drift from the wall's.

    The server's own middleware refuses a viewer's write first and with its own
    words, exactly as ``test_a_company_viewer_cannot_record_a_decision`` in
    ``test_p7_model_wall.py`` already measures; that is a different, outer
    gate. This asserts the wall's own gate directly, on the same session,
    the way that file's ``test_the_walls_own_role_gate_refuses_the_same_viewer``
    does, so the stamp is checked against the mechanism it actually reads.
    """
    from starlette.requests import Request

    from kairos_api.model_console_api import MODEL_WALL

    clients = _clients(auth_env)
    read = clients["viewer"].get("/api/model/console").json()["can_edit_reason"]
    cookie = clients["viewer"].cookies[auth_store.COOKIE_NAME]
    scope = {
        "type": "http", "method": "POST", "path": "/api/model/training",
        "headers": [(b"cookie", f"{auth_store.COOKIE_NAME}={cookie}".encode())],
    }
    assert MODEL_WALL.reason(Request(scope)) == read == READ_ONLY_ROLE_DETAIL


def test_the_activation_mirror_still_carries_no_can_edit_of_its_own(auth_env) -> None:
    """The console's stamp is on the whole payload; the sub-block stays as it was.

    ``activation`` is state the console renders and never controls (the switch
    lives on Rules), and it must not gain a claim about a write this console
    does not offer. Guards the deliberate absence pinned in
    ``test_p7_model_console.py`` against this same change.
    """
    clients = _clients(auth_env)
    body = clients["admin"].get("/api/model/console").json()
    assert body["can_edit"] is True
    assert "can_edit" not in body["activation"]
