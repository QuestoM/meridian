"""P5: who may move the operator's channel, by any route at all.

The competitor boundary has two halves. The read half is scoping, and it is
tested elsewhere: no rival name and no rival figure reaches an operator surface.
This file is the write half, and it is the half that inverts the boundary rather
than leaking through it. Setting ``operator_channel`` to a rival's channel does
not disclose a competitor's data. It makes the product treat the competitor's
schedule as the operator's own and hide the operator's, and every scoped figure
in the product then reads for somebody else.

The scoping is therefore exactly as strong as the weakest writer of that field,
so the assertion here is deliberately not about one route. It enumerates every
write route in the running app whose declared request body can carry the field,
sends each one a body that carries it, and asserts the channel on disk did not
move. A route added later that can carry the field is swept in automatically.

The sweep found one open route, ``PUT /api/settings``, in a file frozen to this
piece, so P5 built the rule, filed the measurement and the two-line patch in
``docs/ux-gauntlet/contracts/P5.md``, and marked the bar below as a strict
expected failure. The call has since landed at ``settings_api.py:56-59``, the
marker came off, and the route that carried the hole keeps a case of its own.

These drive the real app with real resolved sessions. The affiliation wall's own
unit tests prove the lock works; they cannot prove that a door is locked.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

import kairos_api.core as core
import kairos_api.version_store as vs
from kairos_api.compliance_api_licence import (
    OPERATOR_CHANNEL_ADMIN_ONLY_DETAIL,
    OPERATOR_CHANNEL_OMITTED_DETAIL,
    guard_channel_move,
)

ROOT = Path(__file__).resolve().parents[1]

# The channel the deployed document declares. Pinned into the copy rather than
# read out of it, because the field this file is about is writable by anything
# that can reach PUT /api/settings and cannot be leaned on as a fixture.
OPERATOR_CHANNEL = "רשת 13"


@pytest.fixture(autouse=True)
def relocated(tmp_path, monkeypatch):
    """Settings, versions and the audit trail all move into tmp before anything writes."""
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    settings_copy = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", settings_copy)
    document = json.loads(settings_copy.read_text(encoding="utf-8"))
    document["operator_channel"] = OPERATOR_CHANNEL
    settings_copy.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_PATH", settings_copy)
    return tmp_path


def _logged_in(app, username, password, role, affiliation, admin):
    from kairos_api import auth_store  # noqa: F401  (imported for the reset in `walled`)

    assert admin.post("/api/auth/users", json={
        "username": username, "password": password, "role": role,
        "display_name": username, "must_change_password": False,
        "affiliation": affiliation,
    }).status_code == 201
    client = TestClient(app)
    assert client.post("/api/auth/login", json={
        "username": username, "password": password}).status_code == 200
    return client


@pytest.fixture()
def walled(tmp_path, monkeypatch):
    """The real app, real auth, and the three accounts the question needs.

    Both non-admins carry the operator role, so the middleware's read-only rule
    lets them write and the only thing that can stop them is the declaration
    rule itself. One sits on each side of the affiliation line, because the
    permission here is about role and a company account must be refused too.
    """
    from kairos_api import auth_store
    from kairos_api.server import app

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    try:
        auth_store.seed_initial_admin(password="Company-Admin-1")
        admin = TestClient(app)
        assert admin.post("/api/auth/login", json={
            "username": "admin", "password": "Company-Admin-1"}).status_code == 200
        yield {
            "app": app,
            "admin": admin,
            "channel": _logged_in(app, "planchan", "Planner-Pass-1", "operator", "channel", admin),
            "company": _logged_in(app, "plancomp", "Planner-Pass-2", "operator", "company", admin),
        }
    finally:
        auth_store.reset_runtime_state()


@pytest.fixture()
def rival(walled) -> str:
    """A channel in the loaded schedule that is not the operator's own."""
    from kairos_api._constraint_options import channel_options

    others = [name for name in channel_options() if name != OPERATOR_CHANNEL]
    if not others:
        pytest.skip("the loaded schedule carries one channel, so no move can be attempted")
    return others[0]


def _request_for(client) -> Any:
    """A real Request carrying this client's live session cookie.

    Not a stand-in for a session: the cookie is the one the login handed out and
    the wall resolves it through the real store, which is the whole point.
    """
    from starlette.requests import Request

    cookie = "; ".join(f"{name}={value}" for name, value in client.cookies.items())
    return Request({
        "type": "http",
        "method": "PUT",
        "path": "/api/settings",
        "query_string": b"",
        "headers": [(b"cookie", cookie.encode("utf-8"))],
    })


def _settings_body(admin, **overrides) -> dict[str, Any]:
    body = dict(admin.get("/api/settings").json())
    body.update(overrides)
    return body


def _restore(admin) -> None:
    admin.put("/api/settings", json=_settings_body(admin, operator_channel=OPERATOR_CHANNEL))
    assert core._load_settings().operator_channel == OPERATOR_CHANNEL


# ---------------------------------------------------------------------------
# The declared writer, and the permission it carries.


def test_the_declaration_route_refuses_a_non_admin_from_either_side_of_the_line(walled, rival):
    for side in ("channel", "company"):
        refused = walled[side].put("/api/rules/operator-channel", json={"operator_channel": rival})
        assert refused.status_code == 403, refused.text
        assert refused.json()["detail"] == OPERATOR_CHANNEL_ADMIN_ONLY_DETAIL
        assert core._load_settings().operator_channel == OPERATOR_CHANNEL, (
            "a refused declaration still moved the channel"
        )


def test_an_administrator_declares_it_and_an_unbroadcast_channel_is_still_refused(walled, rival):
    allowed = walled["admin"].put("/api/rules/operator-channel", json={"operator_channel": rival})
    assert allowed.status_code == 200
    assert core._load_settings().operator_channel == rival
    _restore(walled["admin"])

    nonsense = walled["admin"].put("/api/rules/operator-channel", json={"operator_channel": "Channel 9"})
    assert nonsense.status_code == 400
    assert "not a channel in the loaded schedule" in nonsense.json()["detail"]
    assert core._load_settings().operator_channel == OPERATOR_CHANNEL


# ---------------------------------------------------------------------------
# The rule itself, resolved against real sessions rather than against a mock.


def test_the_rule_refuses_a_non_admin_and_lets_an_administrator_through(walled, rival):
    moved = core.KairosSettings(**_settings_body(walled["admin"], operator_channel=rival))
    for side in ("channel", "company"):
        with pytest.raises(Exception) as raised:
            guard_channel_move(moved, _request_for(walled[side]))
        assert getattr(raised.value, "status_code", None) == 403
        assert raised.value.detail == OPERATOR_CHANNEL_ADMIN_ONLY_DETAIL
    guard_channel_move(moved, _request_for(walled["admin"]))


def test_the_rule_lets_through_a_write_that_does_not_move_the_channel(walled):
    """Every shipped client sends the settings model whole, so this is the common path."""
    unchanged = core.KairosSettings(**_settings_body(walled["admin"], revenue_weight=65))
    guard_channel_move(unchanged, _request_for(walled["channel"]))


def test_a_body_that_leaves_the_field_out_is_refused_instead_of_clearing_it(walled):
    """An omitted field defaults to empty, and an empty channel un-scopes the product.

    Measured before the rule existed: a settings write carrying one lever cleared
    the declaration outright, which turns every scoped figure from the operator's
    own channel into the whole loaded market with no notice anywhere.
    """
    partial = core.KairosSettings(revenue_weight=65)
    assert "operator_channel" not in partial.model_fields_set
    with pytest.raises(Exception) as raised:
        guard_channel_move(partial, _request_for(walled["admin"]))
    assert getattr(raised.value, "status_code", None) == 400
    assert raised.value.detail == OPERATOR_CHANNEL_OMITTED_DETAIL


# ---------------------------------------------------------------------------
# Every route, not one route.


def _routes_that_can_carry_the_channel(app) -> list[tuple[str, str]]:
    """Every write operation in the live app whose declared body can carry the field.

    Read off the app's own OpenAPI document rather than a hand-kept list, so a
    route added later is swept in without anyone remembering to add it here.
    """
    schema = app.openapi()
    components = schema.get("components", {}).get("schemas", {})

    def carries(node: Any, seen: frozenset[str] = frozenset(), depth: int = 0) -> bool:
        if depth > 6 or not isinstance(node, dict):
            return False
        reference = node.get("$ref")
        if reference:
            name = reference.rsplit("/", 1)[-1]
            return False if name in seen else carries(components.get(name, {}), seen | {name}, depth + 1)
        if "operator_channel" in (node.get("properties") or {}):
            return True
        branches = [sub for key in ("allOf", "anyOf", "oneOf") for sub in (node.get(key) or [])]
        branches += list((node.get("properties") or {}).values())
        if node.get("items"):
            branches.append(node["items"])
        return any(carries(sub, seen, depth + 1) for sub in branches)

    found = []
    for path, item in schema["paths"].items():
        for method, operation in item.items():
            if method.upper() not in {"POST", "PUT", "PATCH"}:
                continue
            body = (operation.get("requestBody") or {}).get("content", {}).get("application/json", {}).get("schema")
            if body and carries(body):
                found.append((method.upper(), path))
    return sorted(found)


def _routes_a_non_admin_can_move_it_through(walled, rival) -> list[str]:
    """Probe each candidate and report the ones where the channel on disk moved.

    One body serves every route: the whole settings document with the channel
    changed. A narrower request model ignores the extra keys, so the probe is
    the same shape everywhere and no route needs a special case.
    """
    open_routes = []
    for method, path in _routes_that_can_carry_the_channel(walled["app"]):
        for side in ("channel", "company"):
            _restore(walled["admin"])
            walled[side].request(method, path, json=_settings_body(walled["admin"], operator_channel=rival))
            if core._load_settings().operator_channel != OPERATOR_CHANNEL:
                open_routes.append(f"{method} {path}")
                break
    _restore(walled["admin"])
    return sorted(set(open_routes))


def test_no_route_lets_a_non_admin_move_the_operator_channel(walled, rival):
    """The bar, and it now holds.

    This carried a strict xfail while the settings route was open, because the
    piece that measured the hole did not own the file that had to close it. The
    strictness was the point: the day the guard landed, the marker reported
    itself stale rather than passing quietly, so a filed blocker could not
    outlive its cause. The lead called guard_channel_move from
    kairos_api/settings_api.py and the marker came off in the same change.
    """
    assert _routes_a_non_admin_can_move_it_through(walled, rival) == []


def test_the_route_that_was_open_refuses_the_move_and_still_saves_the_rest(walled, rival):
    """The route that carried the hole, pinned by name and by both its answers.

    The sweep above is derived from the OpenAPI document, so it would still pass
    if this route stopped accepting settings at all. This one asserts the pair
    that has to hold together: a non-admin of either affiliation is refused the
    move in the declaration route's own words and the channel on disk does not
    budge, an administrator still makes the move, and a non-admin still saves
    every other lever through the same route, which is the capability Bar 3
    protects and the one a blunter guard would have taken away.
    """
    for side in ("channel", "company"):
        refused = walled[side].put(
            "/api/settings", json=_settings_body(walled["admin"], operator_channel=rival),
        )
        assert refused.status_code == 403, refused.text
        assert refused.json()["detail"] == OPERATOR_CHANNEL_ADMIN_ONLY_DETAIL
        assert core._load_settings().operator_channel == OPERATOR_CHANNEL

    kept = walled["channel"].put(
        "/api/settings", json=_settings_body(walled["admin"], revenue_weight=65),
    )
    assert kept.status_code == 200, kept.text
    assert core._load_settings().revenue_weight == 65
    assert core._load_settings().operator_channel == OPERATOR_CHANNEL

    moved = walled["admin"].put(
        "/api/settings", json=_settings_body(walled["admin"], operator_channel=rival),
    )
    assert moved.status_code == 200, moved.text
    assert core._load_settings().operator_channel == rival
    _restore(walled["admin"])


def test_the_sweep_actually_reaches_the_declaration_route(walled):
    """A sweep that found nothing would pass for the wrong reason."""
    swept = _routes_that_can_carry_the_channel(walled["app"])
    assert ("PUT", "/api/rules/operator-channel") in swept
    assert ("PUT", "/api/settings") in swept
