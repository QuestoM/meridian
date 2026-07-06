"""Contract tests for the assistant action plane (tool loop, apply, restore).

The Claude call is mocked at the module seam (assistant._client_factory) with
scripted tool-use turns; a live key is never required. Everything downstream
is real: proposal validation runs the owning routers' validators, the apply
engine writes through the real settings/constraints/overrides seams
(redirected to tmp copies via the journey suites' monkeypatch points), the
action-plane state lives in a tmp KAIROS_ASSISTANT_DATA_DIR, and role
enforcement runs the real auth store."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_tools as tools
from kairos_api import core

ROOT = Path(__file__).resolve().parents[1]


# --- scripted Anthropic mock ---------------------------------------------------
def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use(name: str, args: dict[str, Any], block_id: str = "") -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=args, id=block_id or f"tu_{name}")


def model_turn(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


def scripted_factory(recorder: dict[str, Any], turns: list[SimpleNamespace]):
    """A fake client factory that replays scripted turns and records calls."""
    remaining = list(turns)

    def factory(api_key: str) -> Any:
        recorder["api_key"] = api_key

        def create(**kwargs: Any) -> Any:
            # Snapshot the messages list: the loop mutates it in place.
            snapshot = dict(kwargs)
            snapshot["messages"] = list(kwargs.get("messages") or [])
            recorder.setdefault("calls", []).append(snapshot)
            return remaining.pop(0)

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    return factory


# --- fixtures -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def actions_env(tmp_path, monkeypatch):
    """Tmp action-plane state, a fake key, a fast context, a fresh rate window."""
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_ACTIONS", raising=False)
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    assistant._reset_rate_limit()
    yield
    assistant._reset_rate_limit()


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


@pytest.fixture()
def tmp_settings(tmp_path, monkeypatch) -> Path:
    """Redirect the settings store to a tmp copy so applies never touch the repo."""
    target = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", target)
    monkeypatch.setattr(core, "SETTINGS_PATH", target)
    return target


def make_batch(*specs: tuple[str, dict[str, Any]], user: str = "auth-disabled") -> dict[str, Any]:
    """Create a real stored batch from (tool_name, args) specs via the real
    validation seam, exactly as the ask loop would."""
    items = [tools.build_proposal_item(name, args) for name, args in specs]
    return actions.create_batch("test question", items, user, "claude-sonnet-4-6")


def audit_events() -> list[dict[str, Any]]:
    return [entry["event"] for entry in actions.read_audit(500)["entries"]]


def apply_items(batch: dict[str, Any], ids: list[str]) -> dict[str, Any]:
    """Call the apply route function directly (auth disabled via conftest)."""
    return actions.apply_proposals(batch["batch_id"], actions.ItemIdsRequest(item_ids=ids), None)


def read_manifest(restore_id: str) -> dict[str, Any]:
    path = actions._restore_root() / restore_id / "manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


# --- the tool loop: reads execute, proposals are captured, nothing mutates ------
def test_loop_executes_reads_and_captures_proposals_without_mutating(
    client: TestClient, tmp_settings: Path, monkeypatch
) -> None:
    saved_revenue_weight = core._load_settings().revenue_weight
    new_weight = 85 if saved_revenue_weight != 85 else 90
    recorder: dict[str, Any] = {}
    turns = [
        model_turn(text_block("checking"), tool_use("get_settings", {}, "tu_1"),
                   tool_use("get_day_detail", {"date": "2024-11-05"}, "tu_2")),
        model_turn(tool_use("propose_settings_change",
                            {"changes": {"revenue_weight": new_weight}, "reason": "מאזן הכנסה"}, "tu_3"),
                   tool_use("propose_recompute", {"scope": "full", "reason": "להחיל את השינוי"}, "tu_4")),
        model_turn(text_block("הצעתי שני שינויים לאישורך"), stop="end_turn"),
    ]
    monkeypatch.setattr(assistant, "_client_factory", scripted_factory(recorder, turns))
    before = tmp_settings.read_bytes()

    body = client.post("/api/assistant/ask", json={"question": "העלה את משקל ההכנסה"}).json()

    # Existing fields unchanged, new fields additive.
    assert set(body) == {"available", "answer", "grounding", "error", "proposals", "tool_trace"}
    assert body["available"] is True and body["error"] is None
    assert body["answer"] == "הצעתי שני שינויים לאישורך"
    # Same tools in order, all ok; read steps now additively carry a provenance
    # source, propose steps do not (a proposal captures no figure to attribute).
    assert [(step["tool"], step["ok"]) for step in body["tool_trace"]] == [
        (name, True) for name in (
            "get_settings", "get_day_detail", "propose_settings_change", "propose_recompute")]
    assert body["tool_trace"][0]["source"] == "saved settings"
    assert body["tool_trace"][1]["source"] == "saved weekly plan, owned channel"
    assert "source" not in body["tool_trace"][2] and "source" not in body["tool_trace"][3]

    # Read-tool results were really executed and fed back to the model.
    second_call = recorder["calls"][1]["messages"]
    results = {block["tool_use_id"]: json.loads(block["content"]) for block in second_call[-1]["content"]}
    assert results["tu_1"]["revenue_weight"] == saved_revenue_weight
    assert results["tu_1"]["operator_channel"]
    assert results["tu_2"]["date"] == "2024-11-05" and results["tu_2"]["segments"]
    # Propose-tool results reported capture, not execution.
    third_call = recorder["calls"][2]["messages"]
    proposal_acks = {block["tool_use_id"]: json.loads(block["content"]) for block in third_call[-1]["content"]}
    assert proposal_acks["tu_3"] == {
        "captured": True, "item_id": body["proposals"]["items"][0]["id"], "status": "pending",
        "summary": f"settings: revenue_weight {saved_revenue_weight} -> {new_weight}"}

    # Captured as pending, persisted, and NOTHING mutated.
    items = body["proposals"]["items"]
    assert [item["kind"] for item in items] == ["settings", "recompute"]
    assert all(item["status"] == "pending" for item in items)
    assert all(item["reason"] for item in items)
    assert tmp_settings.read_bytes() == before, "a proposal must not mutate settings"
    stored = client.get("/api/assistant/proposals").json()["batches"]
    assert stored[0]["batch_id"] == body["proposals"]["batch_id"]
    assert audit_events() == ["ask", "proposal"]


# --- apply: the real seam, a restore point, the audit trail ----------------------
def test_apply_settings_item_via_real_seam_with_restore_and_audit(tmp_settings: Path) -> None:
    before = tmp_settings.read_bytes()
    old_weight = core._load_settings().revenue_weight
    new_weight = 85 if old_weight != 85 else 90
    batch = make_batch(
        ("propose_settings_change", {"changes": {"revenue_weight": new_weight}, "reason": "r"})
    )
    item_id = batch["items"][0]["id"]

    response = apply_items(batch, [item_id])
    assert response["results"] == [{"id": item_id, "status": "applied",
                                    "result": {"changed": {"revenue_weight": new_weight}}}]
    assert response["status"] == "resolved"
    assert core._load_settings().revenue_weight == new_weight, "the real settings seam applied"
    assert tmp_settings.read_bytes() != before

    # Restore point: created before the mutation, manifest names the file.
    restore_id = response["restore_id"]
    assert restore_id
    manifest = read_manifest(restore_id)
    assert manifest["batch_id"] == batch["batch_id"]
    assert manifest["item_ids"] == [item_id]
    assert manifest["files"][0]["path"] == str(tmp_settings)
    assert (actions._restore_root() / restore_id / tmp_settings.name).read_bytes() == before
    listed = actions.list_restore_points()["restore_points"]
    assert [entry["restore_id"] for entry in listed] == [restore_id]

    # Audit: proposal then apply, with the real ids and results.
    entries = actions.read_audit(50)["entries"]
    assert [entry["event"] for entry in entries] == ["apply", "proposal"]
    assert entries[0]["batch_id"] == batch["batch_id"]
    assert entries[0]["restore_id"] == restore_id
    assert entries[0]["results"][0]["status"] == "applied"
    assert entries[0]["user"] == "auth-disabled"

    # Restore: the previous bytes come back, and the audit records it.
    restored = actions.restore_state(restore_id, None)
    assert restored["restored"] == [str(tmp_settings)]
    assert tmp_settings.read_bytes() == before, "restore must be byte-for-byte"
    assert core._load_settings().revenue_weight == old_weight
    assert audit_events()[0] == "restore"


def test_apply_constraint_override_and_pricing_via_real_stores(tmp_path, monkeypatch, tmp_settings) -> None:
    import kairos_api.constraints as constraints_api
    import kairos_api.overrides as overrides_api

    constraint_store = tmp_path / "kairos_constraints.csv"
    override_store = tmp_path / "manual_overrides.csv"
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", constraint_store)
    monkeypatch.setattr(constraints_api, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(overrides_api, "OVERRIDES_PATH", override_store)
    monkeypatch.setattr(overrides_api, "BACKUP_DIR", tmp_path / "_backups")

    batch = make_batch(
        ("propose_constraint", {"constraint": {"scope_type": "weekday", "scope_value": "Tuesday",
                                               "effect": "forbid"}, "reason": "r"}),
        ("propose_override", {"override": {"scope": "segment", "target_id": "2024-11-05|Keshet 12|1",
                                           "kind": "pin", "value": "2"}, "reason": "r"}),
        ("propose_pricing_change", {"changes": {"premiums": {"day_of_week": {"2": 1.11}}},
                                    "reason": "r"}),
    )
    overrides_before = dict(core._load_settings().pricing_overrides)
    ids = [item["id"] for item in batch["items"]]
    response = apply_items(batch, ids)
    assert {entry["id"]: entry["status"] for entry in response["results"]} == dict.fromkeys(ids, "applied")

    # All three stores now hold what the real validated paths wrote.
    assert constraints_api.list_constraints()["constraints"][0]["effect"] == "forbid"
    stored_override = overrides_api.list_overrides()["overrides"]["segment"][0]
    assert stored_override["kind"] == "pin" and stored_override["value"] == "2"
    assert core._load_settings().pricing_overrides["premiums"]["day_of_week"]["2"] == 1.11
    # The snapshot covered exactly the touched state files (settings via pricing).
    manifest = read_manifest(response["restore_id"])
    assert {entry["path"] for entry in manifest["files"]} == {
        str(constraint_store), str(override_store), str(tmp_settings)}
    # The stores absent pre-apply are removed again; settings bytes come back.
    restored = actions.restore_state(response["restore_id"], None)
    assert set(restored["removed"]) == {str(constraint_store), str(override_store)}
    assert restored["restored"] == [str(tmp_settings)]
    assert not constraint_store.exists() and not override_store.exists()
    assert core._load_settings().pricing_overrides == overrides_before


def test_apply_recompute_runs_via_job_registry(monkeypatch) -> None:
    import kairos_api.jobs as jobs
    import kairos_api.recompute_api as recompute_api

    monkeypatch.setattr(recompute_api, "_run_recompute", lambda **kwargs: {"ok": True})
    batch = make_batch(("propose_recompute", {"scope": "full", "reason": "r"}))
    response = apply_items(batch, [batch["items"][0]["id"]])
    entry = response["results"][0]
    assert entry["status"] == "applied"
    job_id = entry["result"]["job_id"]
    assert jobs.get(job_id) is not None, "the job must exist in the real registry"
    assert response["restore_id"] is None, "a recompute touches no snapshot-able state file"


# --- per-item isolation -----------------------------------------------------------
def test_one_invalid_item_fails_alone_and_the_valid_one_applies(tmp_settings: Path) -> None:
    old_weight = core._load_settings().revenue_weight
    new_weight = 85 if old_weight != 85 else 90
    good = tools.build_proposal_item(
        "propose_settings_change", {"changes": {"revenue_weight": new_weight}, "reason": "r"}
    )
    # A pending item whose payload later turns invalid (state-drift simulation):
    # it validated at proposal time, and the apply-time store rejects it.
    broken = tools.build_proposal_item(
        "propose_constraint",
        {"constraint": {"scope_type": "weekday", "scope_value": "Tuesday", "effect": "forbid"},
         "reason": "r"},
    )
    broken["payload"]["constraint"]["scope_type"] = "not-a-scope"
    batch = actions.create_batch("q", [broken, good], "auth-disabled", "claude-sonnet-4-6")

    response = apply_items(batch, [broken["id"], good["id"], "missing-id"])
    by_id = {entry["id"]: entry for entry in response["results"]}
    assert by_id[broken["id"]]["status"] == "failed"
    assert "scope_type" in by_id[broken["id"]]["error"]
    assert by_id[good["id"]]["status"] == "applied"
    assert by_id["missing-id"] == {"id": "missing-id", "status": "failed", "error": "no such item in this batch"}
    assert core._load_settings().revenue_weight == new_weight, "the valid item still applied"

    stored = actions.list_proposals()["batches"][0]
    stored_status = {item["id"]: item["status"] for item in stored["items"]}
    assert stored_status == {broken["id"]: "failed", good["id"]: "applied"}
    assert stored["status"] == "resolved"


# --- allowlist rejection ------------------------------------------------------------
def test_forbidden_settings_field_is_rejected_not_crashed(client: TestClient, monkeypatch) -> None:
    recorder: dict[str, Any] = {}
    turns = [
        model_turn(tool_use("propose_settings_change",
                            {"changes": {"operator_channel": "Reshet 13"}, "reason": "r"}, "tu_1")),
        model_turn(text_block("ההצעה נדחתה"), stop="end_turn"),
    ]
    monkeypatch.setattr(assistant, "_client_factory", scripted_factory(recorder, turns))
    body = client.post("/api/assistant/ask", json={"question": "החלף ערוץ"}).json()

    assert body["error"] is None, "a rejected proposal must not crash the ask"
    assert body["tool_trace"] == [{"tool": "propose_settings_change", "ok": False}]
    item = body["proposals"]["items"][0]
    assert item["status"] == "rejected"
    assert "operator_channel" in item["error"] and "not allowed" in item["error"]
    assert body["proposals"]["status"] == "resolved"
    # The model was told the honest reason.
    ack = json.loads(recorder["calls"][1]["messages"][-1]["content"][0]["content"])
    assert ack["captured"] is False and "operator_channel" in ack["reason"]

    # A rejected item cannot be applied later.
    response = apply_items(body["proposals"], [item["id"]])
    assert response["results"][0]["status"] == "failed"
    assert "only pending items" in response["results"][0]["error"]


def test_invalid_settings_value_and_bad_recompute_scope_are_rejected() -> None:
    cases = [
        ("propose_settings_change", {"changes": {"revenue_weight": 250}, "reason": "r"}, "invalid settings values"),
        ("propose_settings_change", {"changes": {"revenue_weight": 70}}, "reason is required"),
        ("propose_recompute", {"scope": "tomorrow", "reason": "r"}, "'full'"),
    ]
    for name, args, expected in cases:
        item = tools.build_proposal_item(name, args)
        assert item["status"] == "rejected" and expected in item["error"], (name, item)


# --- role enforcement ---------------------------------------------------------------
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
    """Mini-app client with a real session cookie: no server middleware, so a
    403 here proves the ROUTE-LEVEL auth seam, not the global guard."""
    app = FastAPI()
    app.include_router(assistant.router)
    session_client = TestClient(app)
    session_client.cookies.set(auth_store.COOKIE_NAME, auth_store.create_session(username, role))
    return session_client


def test_viewer_403_on_apply_reject_restore_operator_allowed(auth_env, tmp_settings) -> None:
    new_weight = 70 if core._load_settings().revenue_weight != 70 else 75
    batch = make_batch(
        ("propose_settings_change", {"changes": {"revenue_weight": new_weight}, "reason": "r"}),
        ("propose_settings_change", {"changes": {"risk_lambda": 0.5}, "reason": "r"}),
        user="admin",
    )
    ids = [item["id"] for item in batch["items"]]
    payload = {"item_ids": ids[:1]}

    viewer = _session_client(auth_env, "viewer1", "viewer")
    assert viewer.post(f"/api/assistant/proposals/{batch['batch_id']}/apply", json=payload).status_code == 403
    assert viewer.post(f"/api/assistant/proposals/{batch['batch_id']}/reject", json=payload).status_code == 403
    assert viewer.post("/api/assistant/restore/abcdef123456").status_code == 403
    assert viewer.get("/api/assistant/proposals").status_code == 200, "viewer read stays open"
    assert core._load_settings().revenue_weight != new_weight, "the viewer attempt mutated nothing"

    bare_app = FastAPI()
    bare_app.include_router(assistant.router)
    anonymous = TestClient(bare_app)
    assert anonymous.post(
        f"/api/assistant/proposals/{batch['batch_id']}/apply", json=payload
    ).status_code == 401

    operator = _session_client(auth_env, "operator1", "operator")
    applied = operator.post(f"/api/assistant/proposals/{batch['batch_id']}/apply", json=payload)
    assert applied.status_code == 200
    assert applied.json()["results"][0]["status"] == "applied"
    assert core._load_settings().revenue_weight == new_weight
    rejected = operator.post(
        f"/api/assistant/proposals/{batch['batch_id']}/reject", json={"item_ids": ids[1:]}
    )
    assert rejected.status_code == 200
    assert rejected.json()["results"][0]["status"] == "rejected"
    restore_id = applied.json()["restore_id"]
    assert operator.post(f"/api/assistant/restore/{restore_id}").status_code == 200

    # The audit trail names the real acting account on every transition.
    by_event = {entry["event"]: entry for entry in actions.read_audit(50)["entries"]}
    assert by_event["apply"]["user"] == "operator1"
    assert by_event["reject"]["user"] == "operator1"
    assert by_event["restore"]["user"] == "operator1"


def test_full_server_walls_viewer_and_admits_operator(auth_env, tmp_settings) -> None:
    from kairos_api.server import app as server_app

    batch = make_batch(
        ("propose_settings_change", {"changes": {"max_breaks_per_hour": 5}, "reason": "r"}),
        user="admin",
    )
    payload = {"item_ids": [batch["items"][0]["id"]]}

    viewer = TestClient(server_app)
    assert viewer.post("/api/auth/login", json={"username": "viewer1", "password": "viewerpass-123"}).status_code == 200
    assert viewer.post(f"/api/assistant/proposals/{batch['batch_id']}/apply", json=payload).status_code == 403

    operator = TestClient(server_app)
    assert operator.post("/api/auth/login", json={"username": "operator1", "password": "operatorpass-123"}).status_code == 200
    response = operator.post(f"/api/assistant/proposals/{batch['batch_id']}/apply", json=payload)
    assert response.status_code == 200
    assert response.json()["results"][0]["status"] == "applied"


# --- restore-point pruning -----------------------------------------------------------
def test_prune_keeps_the_newest_twenty(tmp_settings: Path) -> None:
    created = [actions._snapshot([tmp_settings], f"batch{index}", [f"item{index}"])
               for index in range(22)]
    assert set(actions._prune_restore_points()) == set(created[:2]), "the two oldest go"
    survivors = {entry["restore_id"] for entry in actions._manifests()}
    assert survivors == set(created[2:])
    assert len(survivors) == actions.MAX_RESTORE_POINTS

    # The apply path prunes automatically: one more apply still leaves 20.
    batch = make_batch(("propose_settings_change", {"changes": {"revenue_weight": 61}, "reason": "r"}))
    response = apply_items(batch, [batch["items"][0]["id"]])
    remaining = {entry["restore_id"] for entry in actions._manifests()}
    assert len(remaining) == actions.MAX_RESTORE_POINTS
    assert response["restore_id"] in remaining
    assert created[2] not in remaining, "the oldest survivor was pruned to make room"


# --- endpoint plumbing ----------------------------------------------------------------
def test_audit_endpoint_orders_newest_first_and_honors_limit(client: TestClient) -> None:
    for index in range(5):
        actions.audit_append("ask", "auth-disabled", question=f"q{index}")
    body = client.get("/api/assistant/audit", params={"limit": 3}).json()
    assert [entry["question"] for entry in body["entries"]] == ["q4", "q3", "q2"]
    assert client.post("/api/assistant/proposals/nope/apply", json={"item_ids": ["x"]}).status_code == 404
    assert client.post("/api/assistant/restore/deadbeef9999").status_code == 404
    assert client.post("/api/assistant/proposals/nope/apply", json={"item_ids": []}).status_code == 422
