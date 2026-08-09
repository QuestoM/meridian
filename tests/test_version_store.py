"""Contract tests for the operation-state version history.

Every test relocates the four operation-state files and the version store into a
tmp tree (KAIROS_VERSIONS_DIR plus the store PATH module globals), so nothing
touches the repo. The store is exercised through its own helpers and its HTTP
endpoints, and the manual-edit hooks are proven through the real domain routers.
Role enforcement runs the real auth store.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.advertisers as advertisers_api
import kairos_api.assistant_actions as actions
import kairos_api.constraints as constraints_api
import kairos_api.overrides as overrides_api
import kairos_api.version_store as vs
from kairos_api import core

ROOT = Path(__file__).resolve().parents[1]

from kairos.optimize.constraints_store import COLUMNS as CONSTRAINT_COLUMNS
from kairos.optimize.overrides import COLUMNS as OVERRIDE_COLUMNS


# --- fixtures -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def relocated(tmp_path, monkeypatch):
    """Relocate the version store and all four operation-state files to tmp."""
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)

    settings_path = tmp_path / "kairos_settings.json"
    import shutil
    shutil.copy(ROOT / "data" / "kairos_settings.json", settings_path)
    monkeypatch.setattr(core, "SETTINGS_PATH", settings_path)
    # Canonicalize the settings file to the full model dump so a later single-field
    # edit produces a single-field diff (the shipped file omits defaulted keys).
    core._save_settings(core._load_settings())

    constraints_path = tmp_path / "kairos_constraints.csv"
    overrides_path = tmp_path / "manual_overrides.csv"
    advertisers_path = tmp_path / "advertiser_rules.csv"
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", constraints_path)
    monkeypatch.setattr(constraints_api, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(overrides_api, "OVERRIDES_PATH", overrides_path)
    monkeypatch.setattr(overrides_api, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(advertisers_api, "RULES_PATH", advertisers_path)
    monkeypatch.setattr(advertisers_api, "BACKUP_DIR", tmp_path / "_backups")
    return {
        "settings": settings_path, "constraints": constraints_path,
        "overrides": overrides_path, "advertisers": advertisers_path,
    }


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _set_setting(field: str, value: Any) -> None:
    settings = core._load_settings()
    setattr(settings, field, value)
    core._save_settings(settings)


def _app(*routers) -> TestClient:
    app = FastAPI()
    for router in routers:
        app.include_router(router)
    return TestClient(app)


# --- snapshot on manual edit and identical-content skip -------------------------
def test_manual_edit_through_settings_router_records_a_version(relocated) -> None:
    import kairos_api.settings_api as settings_api

    client = _app(settings_api.router, vs.router)
    payload = core._model_dump(core._load_settings())
    payload["revenue_weight"] = 61 if payload["revenue_weight"] != 61 else 62
    assert client.put("/api/settings", json=payload).status_code == 200

    entries = client.get("/api/versions").json()["entries"]
    assert len(entries) == 1
    assert entries[0]["source"] == "manual_edit"
    assert entries[0]["files"] == ["settings"]
    assert entries[0]["actor"] == "auth-disabled"


def test_identical_content_snapshot_is_skipped(relocated) -> None:
    first = vs.snapshot("manual_edit", "t", ["settings"])
    second = vs.snapshot("manual_edit", "t", ["settings"])
    assert first is not None and second == first, "an unchanged capture reuses the newest id"
    assert len(vs._all_manifests()) == 1

    _set_setting("revenue_weight", 77 if core._load_settings().revenue_weight != 77 else 78)
    third = vs.snapshot("manual_edit", "t", ["settings"])
    assert third != first and len(vs._all_manifests()) == 2


def test_naming_nothing_is_a_no_op_and_naming_the_unknown_is_refused(relocated) -> None:
    """This asserted that BOTH answer None, and that is what hid a real defect.

    An unknown name was filtered out and the call returned None, which at every
    call site reads exactly like "nothing had changed". Two callers were passing
    a name the store had never heard of, for months, with no evidence anywhere:
    campaigns_api_store said it versioned the campaigns store and versioned
    nothing, and target_store did the same for the plan targets.

    So the two cases are now different answers, because they are different
    facts. Naming nothing is a caller asking for nothing. Naming something the
    store cannot capture is a caller asking for something that does not exist,
    and it raises.
    """
    assert vs.snapshot("manual_edit", "t", []) is None
    with pytest.raises(ValueError) as raised:
        vs.snapshot("manual_edit", "t", ["recompute", "nonsense"])
    assert "does not know" in str(raised.value)
    assert vs._all_manifests() == []


# --- snapshot on assistant apply ------------------------------------------------
def test_assistant_apply_records_a_version(relocated, monkeypatch) -> None:
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(relocated["settings"].parent / "assistant"))
    import kairos_api.assistant_tools as tools

    old = core._load_settings().revenue_weight
    new = 85 if old != 85 else 90
    item = tools.build_proposal_item(
        "propose_settings_change", {"changes": {"revenue_weight": new}, "reason": "lift revenue"})
    batch = actions.create_batch("lift the revenue weight", [item], "auth-disabled", "m")
    actions.apply_proposals(batch["batch_id"], actions.ItemIdsRequest(item_ids=[item["id"]]), None)

    apply_versions = [m for m in vs._all_manifests() if m["source"] == "assistant_apply"]
    assert len(apply_versions) == 1
    assert apply_versions[0]["batch_id"] == batch["batch_id"]
    assert apply_versions[0]["label"] == "lift the revenue weight"
    assert [f["logical"] for f in apply_versions[0]["files"]] == ["settings"]


# --- diff correctness: settings -------------------------------------------------
def test_settings_diff_reports_current_versus_version(relocated) -> None:
    old = core._load_settings().revenue_weight
    new = 88 if old != 88 else 89
    version_id = vs.snapshot("manual_snapshot", "t", ["settings"], force=True)
    _set_setting("revenue_weight", new)

    diff = vs._diff_logical(version_id, "settings")
    changed = {c["field"]: (c["from"], c["to"]) for c in diff["changed"]}
    assert changed == {"revenue_weight": (new, old)}, "from=current, to=version (restore target)"


# --- diff correctness: constraints (added / removed / changed) ------------------
def test_constraints_diff_added_removed_changed(relocated) -> None:
    path = relocated["constraints"]
    _write_csv(path, list(CONSTRAINT_COLUMNS), [
        {"constraint_id": "c1", "scope_type": "weekday", "effect": "force"},
        {"constraint_id": "c2", "scope_type": "weekday", "effect": "forbid"},
    ])
    version_id = vs.snapshot("manual_snapshot", "t", ["constraints"], force=True)
    _write_csv(path, list(CONSTRAINT_COLUMNS), [
        {"constraint_id": "c1", "scope_type": "weekday", "effect": "forbid"},
        {"constraint_id": "c3", "scope_type": "channel", "effect": "force"},
    ])

    diff = vs._diff_logical(version_id, "constraints")
    assert [r["constraint_id"] for r in diff["added"]] == ["c2"], "restore re-adds c2"
    assert [r["constraint_id"] for r in diff["removed"]] == ["c3"], "restore removes c3"
    assert {"id": "c1", "field": "effect", "from": "forbid", "to": "force"} in diff["changed"]


# --- diff correctness: overrides ------------------------------------------------
def test_overrides_diff_changed_field(relocated) -> None:
    path = relocated["overrides"]
    _write_csv(path, list(OVERRIDE_COLUMNS), [
        {"override_id": "o1", "scope": "segment", "target_id": "d|c|1", "kind": "pin", "value": "2"},
    ])
    version_id = vs.snapshot("manual_snapshot", "t", ["overrides"], force=True)
    _write_csv(path, list(OVERRIDE_COLUMNS), [
        {"override_id": "o1", "scope": "segment", "target_id": "d|c|1", "kind": "pin", "value": "3"},
    ])
    diff = vs._diff_logical(version_id, "overrides")
    assert diff["added"] == [] and diff["removed"] == []
    assert {"id": "o1", "field": "value", "from": "3", "to": "2"} in diff["changed"]


# --- diff correctness: advertisers (names) --------------------------------------
def test_advertisers_diff_uses_names(relocated) -> None:
    path = relocated["advertisers"]
    _write_csv(path, advertisers_api.COLUMNS, [
        {"advertiser_id": "Acme", "default_premium": "1.0"},
        {"advertiser_id": "Globex", "default_premium": "1.2"},
    ])
    version_id = vs.snapshot("manual_snapshot", "t", ["advertisers"], force=True)
    _write_csv(path, advertisers_api.COLUMNS, [
        {"advertiser_id": "Acme", "default_premium": "1.5"},
        {"advertiser_id": "Initech", "default_premium": "1.1"},
    ])
    diff = vs._diff_logical(version_id, "advertisers")
    assert diff["added"] == ["Globex"], "restore re-adds Globex by name"
    assert diff["removed"] == ["Initech"]
    assert {"advertiser": "Acme", "field": "default_premium", "from": "1.5", "to": "1.0"} in diff["changed"]


# --- partial restore + pre_restore safety + undo --------------------------------
def test_partial_restore_only_selected_files_and_writes_safety_first(relocated) -> None:
    old_weight = core._load_settings().revenue_weight
    new_weight = 66 if old_weight != 66 else 67
    _write_csv(relocated["constraints"], list(CONSTRAINT_COLUMNS),
               [{"constraint_id": "c1", "scope_type": "weekday", "effect": "force"}])
    version_id = vs.snapshot("manual_snapshot", "t", ["settings", "constraints"], force=True)

    _set_setting("revenue_weight", new_weight)
    _write_csv(relocated["constraints"], list(CONSTRAINT_COLUMNS),
               [{"constraint_id": "c1", "scope_type": "weekday", "effect": "forbid"}])

    result = vs.restore_version(version_id, None, vs.RestoreRequest(files=["settings"]))
    assert result["restored"] == ["settings"]
    safety = result["safety_version_id"]
    assert safety, "a pre_restore safety version is written first"

    # Only settings came back; the constraints edit stands.
    assert core._load_settings().revenue_weight == old_weight
    assert "forbid" in relocated["constraints"].read_text(encoding="utf-8-sig")

    # The safety version captured only the restored file, at its pre-restore value.
    safety_manifest = next(m for m in vs._all_manifests() if m["version_id"] == safety)
    assert safety_manifest["source"] == "pre_restore"
    assert [f["logical"] for f in safety_manifest["files"]] == ["settings"]

    # Restoring the safety version undoes the restore (Google-Sheets style).
    vs.restore_version(safety, None, vs.RestoreRequest(files=["settings"]))
    assert core._load_settings().revenue_weight == new_weight


# --- rename ---------------------------------------------------------------------
def test_rename_updates_label(relocated) -> None:
    version_id = vs.snapshot("manual_snapshot", "t", ["settings"], force=True)
    updated = vs.rename_version(version_id, vs.LabelRequest(label="Before the sales week"), None)
    assert updated["label"] == "Before the sales week"
    assert vs._read_manifest(version_id)["label"] == "Before the sales week"


# --- prune ----------------------------------------------------------------------
def test_prune_keeps_the_newest_max(relocated) -> None:
    ids = []
    for index in range(vs.MAX_VERSIONS + 5):
        _set_setting("revenue_weight", index % 100)
        version = vs.snapshot("manual_edit", "t", ["settings"], force=True)
        ids.append(version)
    manifests = vs._all_manifests()
    assert len(manifests) == vs.MAX_VERSIONS
    survivors = {m["version_id"] for m in manifests}
    assert ids[-1] in survivors and ids[0] not in survivors


# --- role enforcement -----------------------------------------------------------
@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    from kairos_api import auth_store

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    auth_store.seed_initial_admin(password="rootpass-1234")
    auth_store.add_user("viewer1", "viewerpass-123", "viewer", "V", must_change_password=False)
    auth_store.add_user("operator1", "operatorpass-123", "operator", "O", must_change_password=False)
    yield auth_store
    auth_store.reset_runtime_state()


def _session_client(auth_store, username: str, role: str) -> TestClient:
    client = _app(vs.router)
    client.cookies.set(auth_store.COOKIE_NAME, auth_store.create_session(username, role))
    return client


def test_role_gates_writer_only_for_mutations(auth_env, relocated) -> None:
    version_id = vs.snapshot("manual_snapshot", "seed", ["settings"], force=True)
    viewer = _session_client(auth_env, "viewer1", "viewer")
    operator = _session_client(auth_env, "operator1", "operator")

    # GET is open to any authenticated role.
    assert viewer.get("/api/versions").status_code == 200
    assert viewer.get(f"/api/versions/{version_id}/diff").status_code == 200

    # Mutations are writer-only for the viewer.
    assert viewer.post("/api/versions/snapshot", json={"label": "x"}).status_code == 403
    assert viewer.post(f"/api/versions/{version_id}/restore", json={}).status_code == 403
    assert viewer.patch(f"/api/versions/{version_id}", json={"label": "x"}).status_code == 403

    # The operator may snapshot, restore and rename.
    assert operator.post("/api/versions/snapshot", json={"label": "point"}).status_code == 200
    assert operator.post(f"/api/versions/{version_id}/restore", json={}).status_code == 200
    assert operator.patch(f"/api/versions/{version_id}", json={"label": "renamed"}).status_code == 200

    # An anonymous session cannot even read.
    anon = _app(vs.router)
    assert anon.get("/api/versions").status_code == 401


# --- endpoint diff shape --------------------------------------------------------
def test_diff_endpoint_returns_per_logical_shape(relocated) -> None:
    _write_csv(relocated["advertisers"], advertisers_api.COLUMNS,
               [{"advertiser_id": "Acme", "default_premium": "1.0"}])
    version_id = vs.snapshot("manual_snapshot", "t", ["settings", "advertisers"], force=True)
    _set_setting("revenue_weight", 91 if core._load_settings().revenue_weight != 91 else 92)

    client = _app(vs.router)
    body = client.get(f"/api/versions/{version_id}/diff").json()
    assert body["version_id"] == version_id
    assert set(body["diff"]) == {"settings", "advertisers"}
    assert "changed" in body["diff"]["settings"]
    assert set(body["diff"]["advertisers"]) == {"added", "removed", "changed"}


# --- the safety keystone: an assistant agreement-apply is versioned and undoable ---
def test_assistant_advertiser_apply_is_versioned_and_restores(relocated, monkeypatch) -> None:
    """An advertiser_change applied through the assistant must snapshot the
    advertiser store before the write, so an assistant-driven agreement update is
    undoable from the version timeline. Guards the _LOGICAL_FOR_KIND mapping."""
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(relocated["settings"].parent / "assistant"))
    import kairos_api.assistant_tools as tools

    # Seed one advertiser, capture the original store bytes.
    _write_csv(relocated["advertisers"], advertisers_api.COLUMNS,
               [{"advertiser_id": "Acme", "default_premium": "1.0"}])
    original = relocated["advertisers"].read_bytes()

    item = tools.build_proposal_item(
        "propose_advertiser_change",
        {"advertiser_name": "Acme", "create": False,
         "changes": {"default_premium": 1.5}, "reason": "agreement raises the premium"})
    batch = actions.create_batch("update Acme agreement", [item], "auth-disabled", "m")
    actions.apply_proposals(batch["batch_id"], actions.ItemIdsRequest(item_ids=[item["id"]]), None)

    # The apply recorded a version that covers the advertisers store.
    apply_versions = [m for m in vs._all_manifests() if m["source"] == "assistant_apply"]
    assert len(apply_versions) == 1
    assert "advertisers" in [f["logical"] for f in apply_versions[0]["files"]]

    # The store really changed, and restoring the pre-apply version reverts it byte for byte.
    assert relocated["advertisers"].read_bytes() != original
    vs._restore_logical(apply_versions[0]["version_id"], "advertisers")
    assert relocated["advertisers"].read_bytes() == original
