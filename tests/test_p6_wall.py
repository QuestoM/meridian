"""P6 Sources: the door that accepts a file is locked, not merely labelled.

``GET /api/uploads/status`` stamps ``can_edit`` and the Sources surface renders
the commit button disabled from it. That is the label. Until the guard went on
``POST /api/uploads/{kind}`` there was no lock behind it: a viewer session read
``can_edit`` false and then wrote ``data/daily_input/Wally_*.csv``, which is the
live daily input the whole plan is computed from.

Every account here is a REAL resolved session, seeded and logged in through the
auth store, not a monkeypatched identity. A test that patches the wall's own
lookup proves the wall works; only a real session proves the door is shut. Every
writable path the uploads module owns is relocated under ``tmp_path/sources``
before anything is posted, so the assertion "no file was written" is measured on
a directory this test owns and nothing in the repository is touched.

**What is measured here is this module's own gate**, on the router mounted
alone, which is how this suite mounts it and how any other assembly would. The
deployed server also carries a blanket middleware rule that refuses a viewer
every mutating method, measured answering ``A viewer session is read-only.``,
and it fires first. That rule does not travel with the router, it is not this
surface's reason, and it is not in this piece's hands; the route's own refusal
is, and it is the sentence ``can_edit_reason`` already prints.

This module postpones its annotations for the reason
``tests/test_w0_cleanup_wall_reads.py`` gives: 79 of the 80 modules in
``kairos_api`` do, so a walled route's parameters reach FastAPI as strings, and a
test module that did not postpone them would pass while every real adopter broke.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.uploads as uploads
from kairos.data.loaders import DAILY_COLUMN_MAP
from kairos_api import affiliation_wall, auth_store

ADMIN_PASSWORD = "rootpass-1234"
VIEWER_PASSWORD = "viewerpass-123"
OPERATOR_PASSWORD = "operpass-1234"

# The one route on this router that answers a write method and writes nothing at
# all. It is the door's own dry run: it parses, it validates, it returns the
# verdict and it stores no report, which tests/test_p6_sources.py measures by
# comparing the whole directory tree before and after. A viewer may run it, and
# the test below re-measures that on a real viewer session rather than trusting
# the note.
WRITES_NOTHING = frozenset({"/api/uploads/{kind}/check"})

LIVE_DAILY = "Wally_2025-04-27.csv"


def _daily_bytes() -> bytes:
    """One valid row of the real daily input, in the engine's own column names."""
    row = {name: "" for name in DAILY_COLUMN_MAP}
    row["תאריך"] = "4/27/2025"
    row["שעה"] = "18:01:00"
    row["שעת התחלת ברייק"] = "18:00:00"
    row["מפרסם"] = "Acme"
    row["קמפיין"] = "Acme Summer"
    row["תוכנית מוזמנת"] = "Evening Show"
    row["שעת התחלת תוכנית"] = "18:00"
    row["אורך תשדיר"] = "30"
    row["מיקום בברייק"] = "1"
    row["רייטינג ברייקים מתוכנן"] = "5.5"
    return pd.DataFrame([row], columns=list(DAILY_COLUMN_MAP)).to_csv(index=False).encode("utf-8")


@pytest.fixture()
def sources_dir(tmp_path, monkeypatch) -> Path:
    """Every writable path this module owns, relocated under one directory."""
    root = tmp_path / "sources"
    monkeypatch.setattr(uploads, "DATA_DIR", root / "data")
    monkeypatch.setattr(uploads, "DAILY_DIR", root / "data" / "daily_input")
    monkeypatch.setattr(uploads, "BACKUP_DIR", root / "data" / "_backups")
    monkeypatch.setattr(uploads, "VALIDATION_REPORTS_PATH", root / "output" / "reports.json")
    return root


@pytest.fixture()
def accounts(tmp_path, monkeypatch) -> "dict[str, str]":
    """A live admin, a company viewer and a channel operator, with their cookies."""
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    from kairos_api.server import app as server_app

    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
    admin = TestClient(server_app)
    signed = admin.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD})
    assert signed.status_code == 200, signed.text
    for username, password, role, affiliation in (
        ("view1", VIEWER_PASSWORD, "viewer", "company"),
        ("oper1", OPERATOR_PASSWORD, "operator", "channel"),
    ):
        created = admin.post("/api/auth/users", json={
            "username": username, "password": password, "role": role,
            "display_name": username, "must_change_password": False,
            "affiliation": affiliation,
        })
        assert created.status_code == 201, created.text
    tokens = {"admin": admin.cookies[auth_store.COOKIE_NAME]}
    for username, password in (("view1", VIEWER_PASSWORD), ("oper1", OPERATOR_PASSWORD)):
        client = TestClient(server_app)
        response = client.post("/api/auth/login", json={"username": username, "password": password})
        assert response.status_code == 200, response.text
        tokens[username] = client.cookies[auth_store.COOKIE_NAME]
    yield tokens
    auth_store.reset_runtime_state()


@pytest.fixture()
def sources_app(sources_dir) -> FastAPI:
    """The uploads router alone, so a failure here is this piece's."""
    app = FastAPI()
    app.include_router(uploads.router)
    return app


def _as(app: FastAPI, token: "str | None" = None) -> TestClient:
    client = TestClient(app)
    if token:
        client.cookies.set(auth_store.COOKIE_NAME, token)
    return client


def _written(root: Path) -> "list[str]":
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _post_daily(client: TestClient):
    return client.post("/api/uploads/daily", files={"file": (LIVE_DAILY, _daily_bytes(), "text/csv")})


# --- the lock, on the live daily input ----------------------------------------
def test_a_viewer_is_refused_the_upload_and_writes_no_file(sources_app, sources_dir, accounts) -> None:
    """The critic's measurement, inverted: 403, and an empty directory after it."""
    viewer = _as(sources_app, accounts["view1"])

    refused = _post_daily(viewer)

    assert refused.status_code == 403, refused.text
    assert refused.json()["detail"] == affiliation_wall.READ_ONLY_ROLE_DETAIL
    assert _written(sources_dir) == [], "a viewer wrote a file the plan is computed from"


def test_the_refusal_is_the_same_sentence_the_status_printed(sources_app, sources_dir, accounts) -> None:
    """The label and the lock are one string, so neither can become a lie alone."""
    viewer = _as(sources_app, accounts["view1"])

    status = viewer.get("/api/uploads/status")
    assert status.status_code == 200, status.text
    body = status.json()
    assert body["can_edit"] is False
    assert body["can_edit_reason"] == _post_daily(viewer).json()["detail"]


def test_a_viewer_still_reads_every_state_and_may_still_check_a_file(sources_app, sources_dir, accounts) -> None:
    """A read-only account stays a usable account: it reads and it validates.

    Checking writes nothing, so the role gate has nothing to protect there, and
    this measures that rather than asserting it. On the deployed server the
    blanket middleware rule above refuses this account the check as well, which
    is that rule's choice and not this module's.
    """
    viewer = _as(sources_app, accounts["view1"])

    assert viewer.get("/api/uploads/status").status_code == 200
    assert viewer.get("/api/uploads/daily/preview?limit=1").status_code == 200
    checked = viewer.post("/api/uploads/daily/check", files={"file": (LIVE_DAILY, _daily_bytes(), "text/csv")})
    assert checked.status_code == 200, checked.text
    assert checked.json()["accepted"] is True
    assert checked.json()["can_edit"] is False
    assert _written(sources_dir) == [], "the check wrote something"


# --- and the door still opens for the people whose job it is ------------------
def test_an_operator_commits_the_identical_file(sources_app, sources_dir, accounts) -> None:
    """The guard closed a door on one account, not on the surface.

    The operator is channel-affiliated on purpose: uploading is not a
    company-only act, so what refused the viewer must be role and only role.
    """
    accepted = _post_daily(_as(sources_app, accounts["oper1"]))

    assert accepted.status_code == 200, accepted.text
    assert accepted.json()["rows"] == 1
    assert _written(sources_dir) == [f"data/daily_input/{LIVE_DAILY}", "output/reports.json"]


def test_an_unresolved_session_is_still_permitted(sources_app, sources_dir, accounts) -> None:
    """Unknown identity stays tolerant, which is what the wall documents.

    Without it a blanket refusal would pass every test above while refusing
    everyone. In the real app the server middleware answers 401 before any route
    runs, so this is the login-less deployment staying open, not a hole.
    """
    assert _post_daily(_as(sources_app)).status_code == 200
    assert f"data/daily_input/{LIVE_DAILY}" in _written(sources_dir)


# --- the invariant, so the next write route cannot arrive undressed -----------
def test_every_write_route_on_this_router_carries_the_wall(sources_app) -> None:
    for route in uploads.router.routes:
        writes = set(getattr(route, "methods", set())) - affiliation_wall.SAFE_METHODS
        if not writes or route.path in WRITES_NOTHING:
            continue
        wall = getattr(route.endpoint, "kairos_wall", None)
        assert wall is uploads.UPLOAD_WALL, f"{route.path} answers {sorted(writes)} with no wall"
