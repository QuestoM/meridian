"""The affiliation wall: one mechanism, three uses, no behaviour change.

Pins that the generalized wall answers exactly what the events guard already
answers (so the five shipped call sites cannot drift), that its decorator keeps
FastAPI injecting the request whether or not the route declares one, and that
``can_edit`` carries the same string the 403 would carry.

This module declares ``from __future__ import annotations`` deliberately, and
the line is load-bearing rather than habit: 79 of the 80 modules in
``kairos_api`` postpone their annotations, so a walled route's parameters reach
FastAPI as strings. A test module that did not postpone them would pass while
every real adopter broke.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI, Query, Request
from fastapi.testclient import TestClient
from pydantic import BaseModel

from kairos_api import affiliation_wall as wall_module
from kairos_api import auth, auth_store, events_access
from kairos_api.affiliation_wall import Wall

ADMIN_PASSWORD = "rootpass-1234"
CHANNEL_PASSWORD = "channelpass-123"
VIEWER_PASSWORD = "viewerpass-123"


class WalledPayload(BaseModel):
    """A body model declared where a route module would declare it.

    It has to live at module level, because that is the namespace the wall
    evaluates a postponed annotation against.
    """

    name: str
    minutes: float = 1.0


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    yield tmp_path
    auth_store.reset_runtime_state()


def _walled_app() -> FastAPI:
    """Every parameter shape a real route uses, behind the same wall.

    The two no-parameter routes are the easy case. The rest are the ones that
    matter: a request body, a body beside a declared ``Request``, a body model
    that is itself called ``request``, and a path parameter with a query
    parameter. All of them arrive as postponed annotations, so each is a route
    that a wall which republishes raw strings would turn into a 422 or a broken
    schema.
    """
    app = FastAPI()
    company = Wall(detail="company only", company_only=True, roles=frozenset())

    @app.get("/declared")
    @company.guard()
    def declared(request: Request) -> dict:
        return {"seen": request.url.path}

    @app.get("/injected")
    @company.guard()
    def injected() -> dict:
        return {"seen": "no request parameter"}

    @app.post("/body")
    @company.guard()
    def body(payload: WalledPayload) -> dict:
        return {"name": payload.name, "minutes": payload.minutes}

    @app.post("/body-declared")
    @company.guard()
    def body_declared(request: Request, payload: WalledPayload) -> dict:
        return {"name": payload.name, "seen": request.url.path}

    # The shape that made the wall fail open. The parameter is called
    # ``request`` and is a body model, which is how 8 of the app's 115 routes
    # are written, all of them writes. See the dedicated test below.
    @app.post("/shadowed")
    @company.guard()
    def shadowed(request: WalledPayload) -> dict:
        return {"name": request.name, "minutes": request.minutes}

    @app.get("/scoped/{channel}")
    @company.guard()
    def scoped(channel: str, day: str = Query(default="")) -> dict:
        return {"channel": channel, "day": day}

    @app.post("/returns-model")
    @company.guard()
    def returns_model(payload: WalledPayload) -> WalledPayload:
        return WalledPayload(name=payload.name.upper(), minutes=payload.minutes)

    # The wrapper has a separate coroutine branch. No body-carrying route in
    # kairos_api is async today, measured, so only a test covers it, and a later
    # piece writing an async route needs it covered before it does.
    @app.post("/async-body")
    @company.guard()
    async def async_body(payload: WalledPayload) -> dict:
        return {"name": payload.name, "async": True}

    # The unwalled control, identical but for the decorator, so the walled
    # route's published contract can be compared against something.
    @app.post("/body-open")
    def body_open(payload: WalledPayload) -> dict:
        return {"name": payload.name, "minutes": payload.minutes}

    return app


def _sign_in(app: FastAPI, username: str, password: str) -> TestClient:
    client = TestClient(app)
    response = client.post("/api/auth/login", json={"username": username, "password": password})
    assert response.status_code == 200, response.text
    return client


# ---------------------------------------------------------------------------
# The gates
# ---------------------------------------------------------------------------

def test_unknown_identity_is_permitted_exactly_as_the_events_guard_is(auth_env):
    """No store, no request and no session all read as company, which is what
    keeps a deployment without login fully open."""
    assert wall_module.is_company(None) is True
    assert events_access.requester_is_company(None) is True
    assert wall_module.has_role(None, {"admin"}) is True
    assert wall_module.session_for(None) is None
    assert Wall(detail="x").allows(None) is True
    assert Wall(detail="x").reason(None) is None


def test_reason_names_the_gate_that_closed(auth_env):
    from kairos_api.server import app

    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
    admin = _sign_in(app, "admin", ADMIN_PASSWORD)
    created = admin.post("/api/auth/users", json={
        "username": "chan1", "password": CHANNEL_PASSWORD, "role": "operator",
        "display_name": "chan1", "must_change_password": False, "affiliation": "channel",
    })
    assert created.status_code == 201, created.text
    viewer = admin.post("/api/auth/users", json={
        "username": "view1", "password": VIEWER_PASSWORD, "role": "viewer",
        "display_name": "view1", "must_change_password": False, "affiliation": "company",
    })
    assert viewer.status_code == 201, viewer.text

    company_wall = Wall(detail="company detail", company_only=True)
    role_wall = Wall(detail="company detail", company_only=False, role_detail="role detail")

    class _Req:
        def __init__(self, token):
            self.cookies = {auth_store.COOKIE_NAME: token}

    def token_for(username: str, password: str) -> str:
        client = _sign_in(app, username, password)
        return client.cookies[auth_store.COOKIE_NAME]

    channel_request = _Req(token_for("chan1", CHANNEL_PASSWORD))
    viewer_request = _Req(token_for("view1", VIEWER_PASSWORD))
    admin_request = _Req(token_for("admin", ADMIN_PASSWORD))

    # Affiliation is the outer gate: a channel operator is refused on
    # affiliation even though its role could write.
    assert company_wall.reason(channel_request) == "company detail"
    # A company viewer passes affiliation and is refused on role, with the
    # default role refusal because this wall declared no other one.
    assert company_wall.reason(viewer_request) == wall_module.READ_ONLY_ROLE_DETAIL
    # A role-only wall lets the channel operator through.
    assert role_wall.reason(channel_request) is None
    assert role_wall.reason(viewer_request) == "role detail"
    assert company_wall.reason(admin_request) is None

    assert wall_module.session_for(channel_request)["affiliation"] == "channel"
    assert wall_module.session_for(admin_request)["role"] == "admin"


def test_write_roles_match_the_shipped_enforcement_rule():
    """The wall must not invent a second opinion about who may write."""
    assert set(wall_module.WRITE_ROLES) == set(auth.WRITE_ROLES)
    assert set(wall_module.ADMIN_ROLES) == {"admin"}


# ---------------------------------------------------------------------------
# The three uses
# ---------------------------------------------------------------------------

def test_stamp_writes_can_edit_and_the_reason_the_refusal_would_use():
    permissive = Wall(detail="denied", company_only=True, roles=frozenset())
    body = permissive.stamp({"value": 1}, None)
    assert body == {"value": 1, "can_edit": True}

    class _AlwaysChannel(Wall):
        def reason(self, request):
            return self.detail

    stamped = _AlwaysChannel(detail="denied").stamp({"value": 1}, None)
    assert stamped["can_edit"] is False
    assert stamped["can_edit_reason"] == "denied"
    # A payload that becomes editable again drops the stale reason.
    assert "can_edit_reason" not in permissive.stamp(stamped, None)


def test_require_raises_403_with_the_hebrew_detail():
    from fastapi import HTTPException

    class _AlwaysChannel(Wall):
        def reason(self, request):
            return self.detail

    with pytest.raises(HTTPException) as caught:
        _AlwaysChannel(detail=events_access.COMPANY_ONLY_DETAIL).require(None)
    assert caught.value.status_code == 403
    assert caught.value.detail == events_access.COMPANY_ONLY_DETAIL


def test_guard_lets_fastapi_inject_the_request_either_way(monkeypatch):
    """The decorator must not break a route that declares no Request, must not
    double-inject one that does, and must leave a body a body.

    The body cases are the ones with teeth. The wrapper is defined in
    ``affiliation_wall``, so FastAPI would resolve a republished string
    annotation against that module's namespace, where ``WalledPayload`` does not
    exist. The parameter would fall through to a query parameter and the route
    would answer 422 to the body it was sent, and the schema would not build at
    all. Measured against the real package before this was fixed: 48 of 133
    route and path pairs were misclassified, 44 of them a body turning into a
    query parameter.
    """
    client = TestClient(_walled_app())
    assert client.get("/declared").json() == {"seen": "/declared"}
    assert client.get("/injected").json() == {"seen": "no request parameter"}

    sent = {"name": "kai", "minutes": 2.5}
    body = client.post("/body", json=sent)
    assert body.status_code == 200, body.text
    assert body.json() == {"name": "kai", "minutes": 2.5}

    body_declared = client.post("/body-declared", json=sent)
    assert body_declared.status_code == 200, body_declared.text
    assert body_declared.json() == {"name": "kai", "seen": "/body-declared"}

    shadowed = client.post("/shadowed", json=sent)
    assert shadowed.status_code == 200, shadowed.text
    assert shadowed.json() == {"name": "kai", "minutes": 2.5}

    scoped = client.get("/scoped/keshet-12", params={"day": "2024-11-01"})
    assert scoped.status_code == 200, scoped.text
    assert scoped.json() == {"channel": "keshet-12", "day": "2024-11-01"}

    # The return annotation is postponed too, and FastAPI reads it to infer the
    # response model, so it has to be resolved by the same pass.
    returned = client.post("/returns-model", json=sent)
    assert returned.status_code == 200, returned.text
    assert returned.json() == {"name": "KAI", "minutes": 2.5}

    awaited = client.post("/async-body", json=sent)
    assert awaited.status_code == 200, awaited.text
    assert awaited.json() == {"name": "kai", "async": True}

    # The schema builds, which is the bar every later piece's server.py append
    # is measured against, and the walled route publishes the same contract as
    # the identical unwalled one.
    schema = client.get("/openapi.json")
    assert schema.status_code == 200, schema.text
    paths = schema.json()["paths"]
    assert paths["/body"]["post"]["requestBody"] == paths["/body-open"]["post"]["requestBody"]
    assert paths["/shadowed"]["post"]["requestBody"] == paths["/body-open"]["post"]["requestBody"]
    assert paths["/body"]["post"].get("parameters", []) == []
    assert paths["/shadowed"]["post"].get("parameters", []) == []
    assert {p["name"] for p in paths["/scoped/{channel}"]["get"]["parameters"]} == {"channel", "day"}
    returned_schema = paths["/returns-model"]["post"]["responses"]["200"]
    assert returned_schema["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/WalledPayload"
    }

    # And it closes when the wall closes, on a read and on a write.
    monkeypatch.setattr(wall_module, "is_company", lambda request: False)
    denied = client.get("/injected")
    assert denied.status_code == 403
    assert denied.json()["detail"] == "company only"
    assert client.get("/declared").status_code == 403
    refused_write = client.post("/body", json=sent)
    assert refused_write.status_code == 403
    assert refused_write.json()["detail"] == "company only"
    assert client.post("/body-declared", json=sent).status_code == 403
    assert client.post("/shadowed", json=sent).status_code == 403
    assert client.get("/scoped/keshet-12").status_code == 403
    assert client.post("/async-body", json=sent).status_code == 403
    # The unwalled control is unaffected, so the 403s above are the wall and not
    # the harness.
    assert client.post("/body-open", json=sent).status_code == 200


def test_a_body_model_named_request_does_not_open_the_wall(monkeypatch):
    """The wall resolves identity by type. A parameter's name decides nothing.

    This is the one refusal shape that discriminates. The blanket refusal used
    above also refuses unknown identity, so it cannot tell a wall that resolved
    a session from one that failed to. Here unknown identity stays permitted,
    exactly as the module documents, and only a resolved session is refused. A
    wall that took the declared-``Request`` branch on ``request: WalledPayload``
    found a Pydantic model, read the identity as unknown and answered 200 to an
    account the identical route ``/body`` answers 403 to.

    Measured on the live app: 8 of 115 routes have this shape, and all 8 are
    writes: ``POST /api/assistant/ask``, ``/api/assistant/ask/stream``,
    ``/api/scenario``, ``/api/optimizer-plan``, ``/api/optimal-plan``,
    ``/api/scenario-compare``, ``/api/jobs/recompute``, ``/api/break-decisions``.
    """
    client = TestClient(_walled_app())
    sent = {"name": "kai", "minutes": 2.5}

    # Unknown identity is permitted, so a refusal below is the wall closing on a
    # resolved session and not the harness refusing everything.
    monkeypatch.setattr(wall_module, "is_company", lambda request: request is None)

    refused = client.post("/shadowed", json=sent)
    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"] == "company only"
    # The control: the same wall, the same account, a parameter with any other
    # name. If this were the only assertion the fail-open would have passed.
    assert client.post("/body", json=sent).status_code == 403
    assert client.post("/body-declared", json=sent).status_code == 403

    # And the same account permitted again opens both, so the 403s are the gate
    # and not a wrapper that broke the route.
    monkeypatch.setattr(wall_module, "is_company", lambda request: True)
    assert client.post("/shadowed", json=sent).json() == {"name": "kai", "minutes": 2.5}
    assert client.post("/body", json=sent).status_code == 200


def test_guard_publishes_a_signature_with_nothing_left_to_resolve():
    """The root cause, pinned at the mechanism rather than through a route.

    FastAPI evaluates a published signature's remaining strings against the
    callable's ``__globals__``, which for a wrapper defined in
    ``affiliation_wall`` is the wrong module. So the contract is not "the route
    happens to work", it is "the wall leaves no string for FastAPI to resolve".
    """
    import inspect

    company = Wall(detail="company only", company_only=True, roles=frozenset())

    def route(payload: WalledPayload, channel: str, day: str = "") -> dict:
        return {}

    def route_with_request(request: Request, payload: WalledPayload) -> dict:
        return {}

    def route_named_request(request: WalledPayload, channel: str) -> dict:
        return {}

    for func in (route, route_with_request, route_named_request):
        # The premise: this module postpones its annotations, so the raw ones
        # really are strings. Without this the test would prove nothing.
        raw = inspect.signature(func)
        assert all(isinstance(p.annotation, str) for p in raw.parameters.values())

        published = inspect.signature(company.guard()(func))
        for name, parameter in published.parameters.items():
            assert not isinstance(parameter.annotation, str), name
        assert not isinstance(published.return_annotation, str)

    # Whatever a parameter is called, it keeps the type it was declared with.
    assert inspect.signature(company.guard()(route)).parameters["payload"].annotation is WalledPayload
    with_request = inspect.signature(company.guard()(route_with_request)).parameters
    assert with_request["payload"].annotation is WalledPayload
    assert with_request["request"].annotation is Request

    # The injected Request is appended under a reserved name that no route can
    # collide with, and a declared Request is never duplicated.
    assert wall_module._INJECTED_REQUEST == "kairos_wall_request"
    assert list(inspect.signature(company.guard()(route)).parameters) == [
        "payload",
        "channel",
        "day",
        "kairos_wall_request",
    ]
    assert list(inspect.signature(company.guard()(route_with_request)).parameters) == [
        "request",
        "payload",
    ]
    # The shape that used to raise ``ValueError: duplicate parameter name`` the
    # moment the wall stopped trusting the name: the route's own ``request`` is
    # a body model, and the injected Request sits beside it.
    named = inspect.signature(company.guard()(route_named_request))
    assert list(named.parameters) == ["request", "channel", "kairos_wall_request"]
    assert named.parameters["request"].annotation is WalledPayload
    assert named.parameters["kairos_wall_request"].annotation is Request


def test_a_module_that_does_not_postpone_its_annotations_is_left_alone():
    """The resolution only ever replaces a string, so an eagerly annotated
    module gets back the signature it already had. That is what keeps the fix
    from widening or rewriting an annotation nobody asked it to touch."""
    import inspect

    from kairos_api.affiliation_wall import _resolved_hints, _typed_signature

    def eager(payload, channel, day=""):
        return {}

    eager.__annotations__ = {
        "payload": WalledPayload,
        "channel": str,
        "day": str,
        "return": dict,
    }
    signature = inspect.signature(eager)
    assert _typed_signature(signature, _resolved_hints(eager)) == signature
