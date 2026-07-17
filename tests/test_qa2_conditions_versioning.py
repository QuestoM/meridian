"""Advertiser-condition edits participate in the version timeline.

Before this wave, the conditions CRUD (the pricing-rule store) never snapshotted
to the version store, so a rule edit was not undoable and invisible in history.
These tests prove the 'conditions' logical file end to end: every mutation
records a version BEFORE the write, the diff keys rows on rule_id, a restore
puts the exact pre-edit bytes back, and the operator's full manual snapshot now
covers the conditions store too.

Everything is relocated into tmp_path (KAIROS_VERSIONS_DIR plus the store PATH
module globals); nothing touches real data/.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.advertiser_conditions as conditions_api
import kairos_api.version_store as vs


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    conditions_path = tmp_path / "advertiser_conditions.csv"
    monkeypatch.setattr(conditions_api, "CONDITIONS_PATH", conditions_path)
    monkeypatch.setattr(conditions_api, "BACKUP_DIR", tmp_path / "_backups")

    app = FastAPI()
    app.include_router(conditions_api.router)
    app.include_router(vs.router)
    return SimpleNamespace(client=TestClient(app), path=conditions_path)


def _versions(client: TestClient) -> list[dict]:
    return client.get("/api/versions").json()["entries"]


def test_create_update_delete_each_record_a_conditions_version(env) -> None:
    client = env.client

    created = client.post("/api/advertisers/ACME/conditions", json={
        "rule_id": "r1", "effect": "premium", "value": 1.2, "mode": "multiplier",
    })
    assert created.status_code == 201
    entries = _versions(client)
    assert len(entries) == 1
    assert entries[0]["source"] == "manual_edit"
    assert entries[0]["files"] == ["conditions"], "the edit must version the conditions file"

    updated = client.put("/api/advertisers/ACME/conditions/r1", json={"value": 1.5})
    assert updated.status_code == 200
    entries = _versions(client)
    assert len(entries) == 2 and entries[0]["files"] == ["conditions"]

    deleted = client.delete("/api/advertisers/ACME/conditions/r1")
    assert deleted.status_code == 200
    assert len(_versions(client)) == 3


def test_diff_reports_the_conditions_change_keyed_on_rule_id(env) -> None:
    client = env.client
    client.post("/api/advertisers/ACME/conditions", json={
        "rule_id": "r1", "effect": "premium", "value": 1.2,
    })
    client.put("/api/advertisers/ACME/conditions/r1", json={"value": 1.5})

    # The newest version captured the pre-update state (value 1.2); the diff is
    # CURRENT (1.5) versus that version, keyed on the rule_id column.
    version_id = _versions(client)[0]["version_id"]
    body = client.get(f"/api/versions/{version_id}/diff").json()
    assert "conditions" in body["diff"]
    changed = body["diff"]["conditions"]["changed"]
    assert any(entry["id"] == "r1" and entry["field"] == "value"
               and entry["from"] == "1.5" and entry["to"] == "1.2"
               for entry in changed), changed


def test_restore_puts_the_pre_edit_conditions_back(env) -> None:
    client = env.client
    client.post("/api/advertisers/ACME/conditions", json={
        "rule_id": "r1", "effect": "premium", "value": 1.2,
    })
    before_update = env.path.read_bytes()
    client.put("/api/advertisers/ACME/conditions/r1", json={"value": 9.9})
    assert env.path.read_bytes() != before_update

    version_id = _versions(client)[0]["version_id"]
    restore = client.post(f"/api/versions/{version_id}/restore", json={"files": ["conditions"]})
    assert restore.status_code == 200
    assert restore.json()["restored"] == ["conditions"]
    assert env.path.read_bytes() == before_update, "restore must be byte-for-byte"
    record = conditions_api.conditions_for("ACME")[0]
    assert record["value"] == pytest.approx(1.2)


def test_restore_undoes_a_delete(env) -> None:
    client = env.client
    client.post("/api/advertisers/ACME/conditions", json={
        "rule_id": "r1", "effect": "forbid", "value": 1.0,
    })
    with_row = env.path.read_bytes()
    client.delete("/api/advertisers/ACME/conditions/r1")
    assert conditions_api.conditions_for("ACME") == []

    version_id = _versions(client)[0]["version_id"]
    client.post(f"/api/versions/{version_id}/restore", json={"files": ["conditions"]})
    assert env.path.read_bytes() == with_row
    assert len(conditions_api.conditions_for("ACME")) == 1


def test_manual_full_snapshot_covers_conditions(env) -> None:
    client = env.client
    client.post("/api/advertisers/ACME/conditions", json={
        "rule_id": "r1", "effect": "premium", "value": 1.2,
    })
    snapshot = client.post("/api/versions/snapshot", json={"label": "point"}).json()
    assert "conditions" in snapshot["files"]


def test_bad_version_ids_are_404_not_traversal(env) -> None:
    client = env.client
    for bad in ("no-hex-here!", "abc", "ABCDEF123456", "%2e%2e%2fescape"):
        assert client.get(f"/api/versions/{bad}/diff").status_code == 404
        assert client.post(f"/api/versions/{bad}/restore", json={}).status_code == 404
        assert client.patch(f"/api/versions/{bad}", json={"label": "x"}).status_code == 404
