"""The model surface is closed to a channel account, on the read as well as the write.

Wave zero built the affiliation wall and wired it to nothing: measured at the
close of that wave, ``guard()`` and ``company_only()`` appeared on zero routes,
and the two model routes that existed took no request parameter at all, so no
affiliation check was even reachable on them. Its own nine unit tests prove the
primitive works; they do not prove a door is locked. This file is that proof,
and it is deliberately written against the real application rather than a
fixture app, because the question is whether the shipped product refuses, not
whether a decorator can.

Four properties, each asserted on real resolved sessions through a live client:

1. **Affiliation closes every method.** A channel-affiliated operator is refused
   every route under ``/api/model``, with the shipped Hebrew denial, before the
   handler runs.
2. **Role closes only the write.** A company viewer reads every one of them and
   cannot record a decision, which is section 4.5's sentence exactly.
3. **Nothing leaks in the refusal.** A refused body carries zero hits of section
   4.2's training lexicon, so the refusal itself cannot be the leak.
4. **The wall closes content without polluting unrelated routes.** ``GET
   /api/impact`` still answers a channel account with the unavailable tri-state,
   but the route-scoped shell no longer fetches company-model impact data on
   operator pages. Partial state is derived only from resources selected for the
   active destination.

The route list is enumerated from the application's own route table rather than
typed here, so a route added to this surface later without the wall fails this
file instead of shipping open.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from kairos_api import auth_store
from kairos_api.affiliation_wall import COMPANY_SURFACE_DETAIL, READ_ONLY_ROLE_DETAIL

REPO_ROOT = Path(__file__).resolve().parents[1]
SHELL_DATA_HOOK = REPO_ROOT / "tv-break-dashboard" / "src" / "shell" / "use-kairos-data.js"

# The compatibility answering route. It answers rather than leaking or refusing,
# but it is no longer part of unrelated shell route profiles.
ANSWERING_PATH = "/api/impact"

ADMIN_PASSWORD = "rootpass-1234"
CHANNEL_PASSWORD = "channelpass-123"
VIEWER_PASSWORD = "viewerpass-123"

# Section 4.2's lexicon. A run-side response returns zero hits; a refusal must
# return zero hits too, or the 403 body is itself the disclosure.
TRAINING_LEXICON = (
    "gate", "held_out", "tau", "drift", "coefficient", "pooling",
    "p_value", "training_window", "wartime",
)


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.setenv("KAIROS_MODEL_RELEASES_DIR", str(tmp_path / "releases"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    yield tmp_path
    auth_store.reset_runtime_state()


def _app():
    from kairos_api.server import app

    return app


def _clients(auth_env) -> dict[str, TestClient]:
    """An admin, a company viewer and a channel operator, on real sessions."""
    app = _app()
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
    out = {"admin": admin}
    for name, password in (("viewer", VIEWER_PASSWORD), ("channel", CHANNEL_PASSWORD)):
        client = TestClient(app)
        username = "view1" if name == "viewer" else "chan1"
        response = client.post("/api/auth/login", json={"username": username, "password": password})
        assert response.status_code == 200, response.text
        out[name] = client
    return out


def _model_routes() -> "list[tuple[str, str]]":
    """Every published route on the model surface, from the app's own table."""
    routes: list[tuple[str, str]] = []
    for route in _app().routes:
        if not isinstance(route, APIRoute):
            continue
        if not (route.path.startswith("/api/model") or route.path == "/api/impact"):
            continue
        for method in sorted(route.methods):
            if method == "HEAD":
                continue
            routes.append((method, route.path))
    return sorted(set(routes))


def _refusing_routes() -> "list[tuple[str, str]]":
    """The surface minus the one route the shell needs an answer from."""
    return [row for row in _model_routes() if row[1] != ANSWERING_PATH]


# A valid body per write, because FastAPI validates the body before the route
# runs and a malformed one answers 422 without ever reaching the wall. The point
# of this file is what the wall does, so every call must reach it.
VALID_BODIES = {
    "/api/model/decisions": {"decision": "not_shipped", "reason": "probe"},
    "/api/model/training": {"artifact": "audience"},
    "/api/model/versions": {},
    "/api/model/candidates/{candidate_id}/measure": None,
}


def _call(client: TestClient, method: str, path: str):
    concrete = path.replace("{candidate_id}", "spotclip")
    if method == "GET":
        return client.get(concrete)
    body = VALID_BODIES.get(path, {})
    return client.post(concrete) if body is None else client.post(concrete, json=body)


def test_the_surface_has_routes_to_wall() -> None:
    """A guard against the test passing because it found nothing to check."""
    routes = _model_routes()
    assert len(routes) >= 12, f"expected the model surface to publish routes, found {routes}"
    assert ("GET", "/api/impact") in routes
    assert ("GET", "/api/model/audience") in routes
    assert ("GET", "/api/model/gates") in routes


def test_a_channel_account_is_refused_every_route_on_the_model_surface(auth_env) -> None:
    clients = _clients(auth_env)
    refused = []
    for method, path in _refusing_routes():
        response = _call(clients["channel"], method, path)
        refused.append((method, path, response.status_code, response.json().get("detail")))
    open_doors = [row for row in refused if row[2] != 403]
    assert open_doors == [], f"a channel account reached the model surface: {open_doors}"
    wrong_words = [row for row in refused if row[3] != COMPANY_SURFACE_DETAIL]
    assert wrong_words == [], f"a refusal used words the product does not ship: {wrong_words}"


def test_no_body_on_the_surface_carries_the_training_lexicon_to_a_channel_account(auth_env) -> None:
    """Every route, refusing or answering, measured on the wire.

    The answering route is in this sweep deliberately: turning a 403 into a 200
    is exactly the move that could leak, so its body is greped with the same
    nine words as every refusal.
    """
    clients = _clients(auth_env)
    for method, path in _model_routes():
        body = json.dumps(_call(clients["channel"], method, path).json(), ensure_ascii=False).lower()
        hits = [word for word in TRAINING_LEXICON if word in body]
        assert hits == [], f"{method} {path} leaked {hits} to a channel account"


def test_a_company_viewer_reads_the_whole_surface(auth_env) -> None:
    """Affiliation decides seeing; a read-only account on the right side reads."""
    clients = _clients(auth_env)
    blocked = []
    for method, path in _model_routes():
        if method != "GET":
            continue
        response = _call(clients["viewer"], method, path)
        if response.status_code != 200:
            blocked.append((path, response.status_code, response.text[:120]))
    assert blocked == [], f"a company viewer could not read its own side: {blocked}"


def test_a_company_viewer_cannot_record_a_decision(auth_env) -> None:
    """Role decides changing, and a verdict is a change.

    Two gates would refuse this and the outer one answers first: the server's
    own middleware refuses every write from a viewer session before any route
    runs, so the message a viewer reads is the middleware's. The wall's role
    gate is the second line and is asserted directly beneath, on the same
    session, so neither is taken on trust.
    """
    clients = _clients(auth_env)
    response = clients["viewer"].post("/api/model/decisions", json={
        "decision": "not_shipped", "subject": "current", "reason": "a viewer should not land this",
    })
    assert response.status_code == 403
    assert response.json()["detail"] in ("A viewer session is read-only.", READ_ONLY_ROLE_DETAIL)


def test_the_walls_own_role_gate_refuses_the_same_viewer(auth_env) -> None:
    """The second line, measured on the wall itself with the viewer's session.

    The middleware answers first on the live route, so without this the wall's
    role gate would be untested on this surface and a middleware change would
    silently open the write.
    """
    from starlette.requests import Request

    from kairos_api.model_console_api import MODEL_WALL

    clients = _clients(auth_env)
    cookie = clients["viewer"].cookies[auth_store.COOKIE_NAME]
    scope = {
        "type": "http", "method": "POST", "path": "/api/model/decisions",
        "headers": [(b"cookie", f"{auth_store.COOKIE_NAME}={cookie}".encode())],
    }
    request = Request(scope)
    assert MODEL_WALL.allows_read(request) is True
    assert MODEL_WALL.allows(request) is False
    assert MODEL_WALL.reason(request) == READ_ONLY_ROLE_DETAIL


def test_an_admin_records_a_decision_and_the_wall_lets_it_through(auth_env) -> None:
    clients = _clients(auth_env)
    response = clients["admin"].post("/api/model/decisions", json={
        "decision": "not_shipped", "subject": "current",
        "reason": "the window has no contrast for five of thirteen factors",
    })
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["decision"] == "not_shipped"
    assert record["actor"] == "admin"
    assert record["model_version_id"].startswith("mv-")


def test_the_operator_surfaces_no_longer_serve_the_gate_verdicts(auth_env) -> None:
    """The calendar read is the one an operator page actually renders.

    ``/api/model/audience`` is what ``CalendarAudienceModel`` fetches, and it is
    refused here. This asserts the refusal on the exact path that page calls, so
    the Bar 3 row "no operator surface may keep showing them" is checked on the
    wire rather than in the component.
    """
    clients = _clients(auth_env)
    response = clients["channel"].get("/api/model/audience")
    assert response.status_code == 403
    assert "verdict" not in response.text


# --------------------------------------------------------------------------- #
# The answering route: unavailable content is explicit and not loaded globally
# --------------------------------------------------------------------------- #
def test_the_shell_does_not_fetch_company_impact_on_unrelated_routes() -> None:
    """Route profiles own partial state; company impact is not global chrome."""
    source = SHELL_DATA_HOOK.read_text(encoding="utf-8")
    assert f"fetchJson('{ANSWERING_PATH}'" not in source
    assert "kairosDataResources(activeView" in source
    assert "currentRecords.some((record) => !record.online)" in source


def test_a_channel_account_gets_an_answer_from_the_answering_route(auth_env) -> None:
    """200 with the tri-state, so no page reads "Some data failed to load".

    This is the Bar 3 regression the wall introduced and this piece owns. The
    body is asserted key by key rather than by status alone, because a 200 that
    carried the measurement would be a far worse failure than the 403 was.
    """
    clients = _clients(auth_env)
    response = clients["channel"].get(ANSWERING_PATH)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["state"] == "unavailable"
    assert body["state_reason"] == COMPANY_SURFACE_DETAIL
    assert set(body) == {"state", "state_reason"}, f"the walled body grew keys: {sorted(body)}"


def test_a_company_account_still_gets_the_measurement_from_the_answering_route(auth_env) -> None:
    """The other side of the same route, so the fix cannot have closed it for everyone."""
    clients = _clients(auth_env)
    body = clients["viewer"].get(ANSWERING_PATH).json()
    assert body["state"] in {"real", "unknown"}
    assert "coefficient_impacts" in body
    assert "drift" in body
    if body["state"] == "real":
        assert body["state_reason"] is None
        axes = body["coefficient_impacts"]
        assert any(axes[axis] for axis in ("program_type", "position", "length"))
    else:
        assert body["state_reason"], "an unknown state must name what is missing"
