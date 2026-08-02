"""P5: the condition builder's value picker holds one channel, the operator's.

The composer's programme picker was scoped to the operator's own channel from
the day it shipped. The condition builder beside it read a second, older option
route, and that one was never scoped: measured on the reference EPG on
2026-08-01 with the operator channel set to ``רשת 13``,
``GET /api/constraints/options`` served 418 programme titles, of which 106 are
the operator's own and 312 are three rivals' entire lineups (149 on one, 104 on
another, 59 on the third), plus all four channel names. So the surface that
authors a restriction offered a representative three competitors' schedules.

These tests measure the payload against the reference EPG itself rather than
against a fixed list, so they keep holding when the data changes. The rival set
is derived here the same way the breach was: every title on a channel that is
not the declared one.

Every store this touches is relocated into tmp, so no test writes ``data/``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.constraints as constraints_api
import kairos_api.core as core
import kairos_api.version_store as vs

ROOT = Path(__file__).resolve().parents[1]

CHANNEL = "רשת 13"


def _settings_copy(tmp_path: Path, channel: str) -> Path:
    """A private settings document with the operator channel pinned to ``channel``."""
    path = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", path)
    document = json.loads(path.read_text(encoding="utf-8"))
    document["operator_channel"] = channel
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def relocated(tmp_path, monkeypatch):
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", tmp_path / "kairos_constraints.csv")
    monkeypatch.setattr(constraints_api, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(core, "SETTINGS_PATH", _settings_copy(tmp_path, CHANNEL))
    return tmp_path


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(constraints_api.router)
    return TestClient(app)


@pytest.fixture()
def undeclared(tmp_path, monkeypatch) -> TestClient:
    """The same route with no channel declared, which is the honest empty state."""
    monkeypatch.setattr(core, "SETTINGS_PATH", _settings_copy(tmp_path, ""))
    app = FastAPI()
    app.include_router(constraints_api.router)
    return TestClient(app)


def _epg_titles() -> "tuple[set[str], set[str]]":
    """(the operator's own titles, every title that is only on another channel)."""
    from kairos.data.loaders import load_programmes

    frame = load_programmes()
    channels = frame["Channel"].astype(str).str.strip()
    owned = {
        str(title).strip()
        for title in frame[channels == CHANNEL]["Title"].dropna()
        if str(title).strip()
    }
    every = {str(title).strip() for title in frame["Title"].dropna() if str(title).strip()}
    return owned, every - owned


def _rival_channels() -> "set[str]":
    from kairos.data.loaders import load_programmes

    names = {
        str(name).strip()
        for name in load_programmes()["Channel"].dropna()
        if str(name).strip()
    }
    return names - {CHANNEL}


# ---------------------------------------------------------------------------
# The competitor boundary on the condition builder's own route.


def test_the_options_payload_holds_no_rival_title_and_no_rival_channel_name(client):
    """The bar, stated the way the breach was measured: zero of either."""
    owned, rivals = _epg_titles()
    assert rivals, "the reference EPG has to carry rival titles or this proves nothing"
    body = client.get("/api/constraints/options").json()

    offered = set(body["programmes"])
    assert offered <= owned, sorted(offered - owned)[:8]
    assert not (offered & rivals), sorted(offered & rivals)[:8]

    text = json.dumps(body, ensure_ascii=False)
    assert [name for name in _rival_channels() if name in text] == [], (
        "a rival channel name reached an operator surface"
    )
    assert body["channels"] == [CHANNEL]
    assert body["available_channels"] == [CHANNEL]


def test_the_picker_offers_every_programme_the_operator_broadcasts(client):
    """Scoped, not truncated: the operator loses no title of their own."""
    owned, _rivals = _epg_titles()
    body = client.get("/api/constraints/options").json()
    assert set(body["programmes"]) == owned
    assert len(body["programmes"]) == len(owned)


def test_the_scope_note_discloses_the_channel_and_what_it_dropped(client):
    """A scoped payload says so, and the excluded count carries no rival's name."""
    body = client.get("/api/constraints/options").json()
    scope = body["scope"]
    assert scope["scoped"] is True
    assert scope["scope_channel"] == CHANNEL
    assert scope["reason"] is None
    assert scope["rows_out"] < scope["rows_in"]
    assert scope["competitor_rows_excluded"] == scope["rows_in"] - scope["rows_out"]
    assert scope["competitor_channels_excluded"] == len(_rival_channels())
    assert "channel" not in {key for key in scope if key.endswith("_name")}


def test_with_no_channel_declared_the_lists_are_empty_and_name_the_route(undeclared):
    """The pass-through form of the scope helper is the breach, so it is refused.

    Nothing is declared, so nothing can be scoped, and the honest answer is the
    empty list plus the reason and the route that supplies the missing input.
    """
    body = undeclared.get("/api/constraints/options").json()
    assert body["programmes"] == []
    assert body["genres"] == []
    assert body["channels"] == []
    assert body["available_channels"] == []
    assert body["operator_channel"] == ""
    scope = body["scope"]
    assert scope["scoped"] is False
    assert scope["reason"] == "operator channel is not configured in settings"
    assert scope["supply_route"] == "PUT /api/rules/operator-channel"
    text = json.dumps(body, ensure_ascii=False)
    assert [name for name in _rival_channels() if name in text] == []


# ---------------------------------------------------------------------------
# The two pickers on this surface answer with the same lineup.


def test_the_composer_and_the_builder_offer_the_same_channels_programmes(client):
    """One surface, one lineup. The composer reads the plan window and the builder
    reads the reference schedule, so the composer's titles are a subset, and a
    title in the composer that the builder cannot offer would mean the two
    pickers disagree about whose programme it is."""
    composer = client.get("/api/constraints/restrictions/titles").json()
    if not composer["titles"]:
        pytest.skip("the plan window holds no airings, so there is nothing to join")
    assert composer["channel"] == CHANNEL
    builder = set(client.get("/api/constraints/options").json()["programmes"])
    offered = {row["title"] for row in composer["titles"]}
    assert offered <= builder, sorted(offered - builder)[:8]


# ---------------------------------------------------------------------------
# Bar 3: the builder still gets every list it reads.


def test_the_payload_keeps_every_key_the_condition_builder_reads(client):
    """Scoping the values changed no key, so the grammar and the pickers stand."""
    body = client.get("/api/constraints/options").json()
    for key in (
        "scope_types", "effects", "programmes", "genres", "channels", "weekdays",
        "dayparts", "predicate_fields", "operator_channel", "available_channels",
    ):
        assert key in body, key
    assert body["scope_types"] and body["effects"]
    assert [field["field"] for field in body["predicate_fields"]] == [
        "programme", "genre", "daypart", "weekday", "date", "hour",
    ]
    assert [day["key"] for day in body["weekdays"]][0] == "7", "the week starts on Sunday"
    assert body["dayparts"], "the daypart vocabulary is unchanged"
