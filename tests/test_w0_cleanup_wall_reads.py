"""The wall's role gate belongs on the write, not on the read.

W0-4 shipped ``Wall`` with ``roles`` defaulting to the write roles and a
``guard()`` that enforced every gate on every method, and its contract
demonstrates the one-liner ``Wall(detail=..., company_only=True)``. Adopted the
way it is demonstrated, that decorator answered 403 to a company **viewer**
asking for a read, with the read-only-role refusal. Eighteen later pieces copy
that line, so the failure would have been eighteen surfaces a viewer cannot open.

The two walls wave zero actually shipped both document the behaviour these tests
assert, which is the evidence that the guard was wrong rather than the walls:
``model_activation.ACTIVATION_WALL`` says "a company viewer sees the state and
not the control", and ``guardrail_store.GUARDRAIL_WALL`` says "a channel account
reads them and an admin changes them". Under the shipped guard neither sentence
was true of its own route.

The rule now: **affiliation gates every method, role gates only the write.** A
surface that genuinely needs a role-gated read asks for it in one argument,
``guard(roles_on_read=True)``, so the strict form stays reachable and stays
visible at the call site.

This module postpones its annotations for the same load-bearing reason
``tests/test_w0_4_affiliation_wall.py`` does: 79 of the 80 modules in
``kairos_api`` do, so a walled route's parameters reach FastAPI as strings, and a
test module that did not postpone them would pass while every real adopter broke.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from kairos_api import affiliation_wall as wall_module
from kairos_api import auth_store
from kairos_api.affiliation_wall import Wall, company_only

ADMIN_PASSWORD = "rootpass-1234"
CHANNEL_PASSWORD = "channelpass-123"
VIEWER_PASSWORD = "viewerpass-123"


class WalledPayload(BaseModel):
    """A body model declared where a route module would declare it."""

    name: str


# The wall exactly as docs/ux-gauntlet/contracts/W0-4.md demonstrates it, so
# what these tests measure is the line a later piece copies, not a variant.
CONTRACT_WALL = Wall(detail="התצוגה הזו שמורה לצוות החברה", company_only=True)


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    yield tmp_path
    auth_store.reset_runtime_state()


def _walled_app() -> FastAPI:
    """One read and one write behind the contract's own wall, plus a strict read."""
    app = FastAPI()

    @app.get("/read")
    @CONTRACT_WALL.guard()
    def read() -> dict:
        return {"seen": "read"}

    @app.get("/read-declared")
    @CONTRACT_WALL.guard()
    def read_declared(request: Request) -> dict:
        return {"seen": request.url.path}

    @app.post("/write")
    @CONTRACT_WALL.guard()
    def write(payload: WalledPayload) -> dict:
        return {"name": payload.name}

    # The surface that genuinely wants the strict old behaviour asks for it.
    @app.get("/read-strict")
    @CONTRACT_WALL.guard(roles_on_read=True)
    def read_strict() -> dict:
        return {"seen": "strict"}

    # The one-line company form, on a read and on a write.
    @app.get("/company-read")
    @company_only("company only")
    def company_read() -> dict:
        return {"seen": "company"}

    @app.post("/company-write")
    @company_only("company only")
    def company_write(payload: WalledPayload) -> dict:
        return {"name": payload.name}

    return app


def _accounts(app: FastAPI) -> dict[str, str]:
    """Seed an admin, a company viewer and a channel operator; return their tokens."""
    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
    admin = TestClient(app)
    signed = admin.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD})
    assert signed.status_code == 200, signed.text
    for username, password, role, affiliation in (
        ("view1", VIEWER_PASSWORD, "viewer", "company"),
        ("chan1", CHANNEL_PASSWORD, "operator", "channel"),
    ):
        created = admin.post("/api/auth/users", json={
            "username": username, "password": password, "role": role,
            "display_name": username, "must_change_password": False,
            "affiliation": affiliation,
        })
        assert created.status_code == 201, created.text
    tokens = {"admin": admin.cookies[auth_store.COOKIE_NAME]}
    for username, password in (("view1", VIEWER_PASSWORD), ("chan1", CHANNEL_PASSWORD)):
        client = TestClient(app)
        response = client.post(
            "/api/auth/login", json={"username": username, "password": password},
        )
        assert response.status_code == 200, response.text
        tokens[username] = client.cookies[auth_store.COOKIE_NAME]
    return tokens


def _as(app: FastAPI, token: str) -> TestClient:
    client = TestClient(app)
    client.cookies.set(auth_store.COOKIE_NAME, token)
    return client


@pytest.fixture()
def walled(auth_env):
    """The walled app plus a client per account, on real resolved sessions."""
    from kairos_api.server import app as server_app

    tokens = _accounts(server_app)
    walled_app = _walled_app()
    return {name: _as(walled_app, token) for name, token in tokens.items()}


# ---------------------------------------------------------------------------
# The gap: a viewer could not read a walled surface
# ---------------------------------------------------------------------------

def test_a_viewer_reads_a_walled_surface_and_still_cannot_write_it(walled) -> None:
    """The contract's own one-liner, measured on a real viewer session.

    Before the fix the two reads answered 403 with the read-only-role refusal,
    which is the whole defect: the role gate closed a read.
    """
    viewer = walled["view1"]

    assert viewer.get("/read").status_code == 200
    assert viewer.get("/read").json() == {"seen": "read"}
    assert viewer.get("/read-declared").status_code == 200

    refused = viewer.post("/write", json={"name": "kai"})
    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"] == wall_module.READ_ONLY_ROLE_DETAIL


def test_a_channel_account_is_refused_the_read_and_the_write(walled) -> None:
    """Affiliation is the outer gate and it still closes both methods."""
    channel = walled["chan1"]

    denied_read = channel.get("/read")
    assert denied_read.status_code == 403, denied_read.text
    assert denied_read.json()["detail"] == CONTRACT_WALL.detail

    denied_write = channel.post("/write", json={"name": "kai"})
    assert denied_write.status_code == 403, denied_write.text
    assert denied_write.json()["detail"] == CONTRACT_WALL.detail


def test_a_company_admin_reads_and_writes(walled) -> None:
    """The control: the same routes, an account both gates permit."""
    admin = walled["admin"]
    assert admin.get("/read").status_code == 200
    assert admin.get("/read-declared").status_code == 200
    written = admin.post("/write", json={"name": "kai"})
    assert written.status_code == 200, written.text
    assert written.json() == {"name": "kai"}


def test_a_surface_that_wants_a_role_gated_read_asks_for_one(walled) -> None:
    """The strict form stays reachable, and it is visible at the call site."""
    assert walled["view1"].get("/read-strict").status_code == 403
    assert walled["view1"].get("/read-strict").json()["detail"] == (
        wall_module.READ_ONLY_ROLE_DETAIL
    )
    assert walled["admin"].get("/read-strict").status_code == 200
    assert walled["chan1"].get("/read-strict").status_code == 403


def test_the_one_line_company_form_gates_the_write_on_role_too(walled) -> None:
    """``company_only`` shipped with no role gate at all, so a viewer could POST.

    It is the same wall as the decorated form now: affiliation on every method,
    role on the write.
    """
    assert walled["view1"].get("/company-read").status_code == 200
    refused = walled["view1"].post("/company-write", json={"name": "kai"})
    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"] == wall_module.READ_ONLY_ROLE_DETAIL
    assert walled["chan1"].get("/company-read").status_code == 403
    assert walled["admin"].post("/company-write", json={"name": "kai"}).status_code == 200


# ---------------------------------------------------------------------------
# The mechanism, without a client
# ---------------------------------------------------------------------------

def test_read_reason_asks_only_the_affiliation_question(auth_env) -> None:
    """``reason`` is the write question and stays the source of ``can_edit``."""
    from kairos_api.server import app as server_app

    tokens = _accounts(server_app)

    class _Req:
        def __init__(self, token: str, method: str = "GET") -> None:
            self.cookies = {auth_store.COOKIE_NAME: token}
            self.method = method

    viewer = _Req(tokens["view1"])
    channel = _Req(tokens["chan1"])
    admin = _Req(tokens["admin"])

    assert CONTRACT_WALL.read_reason(viewer) is None
    assert CONTRACT_WALL.reason(viewer) == wall_module.READ_ONLY_ROLE_DETAIL
    assert CONTRACT_WALL.read_reason(channel) == CONTRACT_WALL.detail
    assert CONTRACT_WALL.reason(channel) == CONTRACT_WALL.detail
    assert CONTRACT_WALL.read_reason(admin) is None
    assert CONTRACT_WALL.reason(admin) is None

    # can_edit is the write question, so a viewer's payload still says false
    # with the reason a POST would carry. That is what a control renders from.
    stamped = CONTRACT_WALL.stamp({"value": 1}, viewer)
    assert stamped["can_edit"] is False
    assert stamped["can_edit_reason"] == wall_module.READ_ONLY_ROLE_DETAIL


def test_unknown_identity_is_permitted_on_a_read_as_it_is_on_a_write() -> None:
    """No store, no request and no session all stay open, on both questions."""
    assert CONTRACT_WALL.read_reason(None) is None
    assert CONTRACT_WALL.reason(None) is None
    assert CONTRACT_WALL.allows_read(None) is True
    assert CONTRACT_WALL.allows(None) is True


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS"])
def test_the_safe_methods_are_exactly_the_ones_that_skip_the_role_gate(method) -> None:
    assert method in wall_module.SAFE_METHODS
    for unsafe in ("POST", "PUT", "PATCH", "DELETE"):
        assert unsafe not in wall_module.SAFE_METHODS
