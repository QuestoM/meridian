"""Contract tests for the knowledge-and-agreements lane.

Covers the agreement-file uploads (validation, per-user isolation, cap
disclosure), the agreement read tools (list_uploads, get_upload,
find_advertiser), the propose_advertiser_change proposal with its field-level
diff and the generalized settings diff, the advertiser_change apply executor
writing through the real store while riding an undoable snapshot, and the
operator-handbook system wiring.

The Claude call is mocked at the module seam (assistant._client_factory) only
for the end-to-end loop test; everything else exercises the real store code,
the real validators and the real upload parser. State (uploads, proposals,
restore points) lives under a tmp KAIROS_ASSISTANT_DATA_DIR; the advertiser
store is redirected to a tmp CSV so the repository file is never touched.
"""

from __future__ import annotations

import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from openpyxl import Workbook

import kairos_api.assistant as assistant
import kairos_api.assistant_actions as actions
import kairos_api.assistant_tools as tools
import kairos_api.assistant_uploads as uploads
from kairos_api import advertisers as advertisers_api

ROOT = Path(__file__).resolve().parents[1]


# --- fixtures -------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _state(tmp_path, monkeypatch):
    """Tmp action-plane state (uploads, proposals, restore points), no ambient key."""
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_API_KEY", raising=False)
    monkeypatch.delenv("KAIROS_ASSISTANT_MODEL", raising=False)
    assistant._reset_rate_limit()
    yield
    assistant._reset_rate_limit()


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    return TestClient(app)


@pytest.fixture()
def tmp_advertisers(tmp_path, monkeypatch) -> Path:
    """A crafted advertiser store redirected to tmp so writes never touch the repo."""
    target = tmp_path / "advertiser_rules.csv"
    frame = pd.DataFrame(
        [
            {"advertiser_id": "Acme Beverages", "default_premium": "1.20",
             "allow_positions": "ANY", "allow_genres": "ANY", "prime_time_only": "False",
             "urgency_k": "", "ahead_k": "", "notes": "existing"},
            {"advertiser_id": "Globex Telecom", "default_premium": "1.00",
             "allow_positions": "ANY", "allow_genres": "ANY", "prime_time_only": "False",
             "urgency_k": "", "ahead_k": "", "notes": ""},
        ]
    )
    frame.to_csv(target, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(advertisers_api, "RULES_PATH", target)
    monkeypatch.setattr(advertisers_api, "BACKUP_DIR", tmp_path / "_backups")
    return target


def _xlsx_bytes(rows: list[list[Any]], sheet: str = "Deal") -> bytes:
    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = sheet
    for row in rows:
        worksheet.append(row)
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def _upload(client: TestClient, name: str, data: bytes) -> Any:
    return client.post("/api/assistant/upload", files={"file": (name, data, "application/octet-stream")})


# --- upload validation ----------------------------------------------------------
def test_upload_rejects_bad_extension(client: TestClient) -> None:
    response = _upload(client, "notes.txt", b"hello")
    assert response.status_code == 400
    assert ".xlsx" in response.json()["detail"]


def test_upload_rejects_junk_content_that_does_not_parse(client: TestClient) -> None:
    # A .xlsx extension over bytes that are not a spreadsheet is an honest 400.
    response = _upload(client, "deal.xlsx", b"this is not a workbook at all")
    assert response.status_code == 400
    assert "parsed" in response.json()["detail"]


def test_upload_rejects_oversize_file(client: TestClient) -> None:
    oversize = b"header\n" + b"row\n" * 1_700_000  # about 6.8 MB, over the 5 MB cap
    assert len(oversize) > uploads.MAX_BYTES
    response = _upload(client, "big.csv", oversize)
    assert response.status_code == 400
    assert "limit" in response.json()["detail"]


def test_upload_and_read_roundtrip_through_the_tool(client: TestClient) -> None:
    data = _xlsx_bytes([["advertiser_id", "premium"], ["Acme Beverages", "1.2"]])
    created = _upload(client, "acme.xlsx", data).json()
    assert created["filename"] == "acme.xlsx"
    assert created["sheets"][0]["name"] == "Deal"
    upload_id = created["upload_id"]

    listed = client.get("/api/assistant/uploads").json()
    assert [item["upload_id"] for item in listed["uploads"]] == [upload_id]

    # The read tool returns the rows and stamps the filename as its source.
    payload = tools.execute_read_tool("get_upload", {"upload_id": upload_id}, "auth-disabled")
    assert payload["source"] == "uploaded file acme.xlsx"
    assert payload["sheets"][0]["rows"] == [["Acme Beverages", "1.2"]]
    assert payload["sheets"][0]["columns"] == ["advertiser_id", "premium"]

    deleted = client.delete(f"/api/assistant/uploads/{upload_id}")
    assert deleted.status_code == 200
    assert client.get("/api/assistant/uploads").json()["uploads"] == []


def test_get_upload_discloses_the_row_cap(client: TestClient) -> None:
    rows = [["col"]] + [[f"r{index}"] for index in range(uploads.MAX_ROWS + 25)]
    body = "\n".join(row[0] for row in rows).encode("utf-8")
    upload_id = _upload(client, "long.csv", body).json()["upload_id"]

    payload = tools.execute_read_tool("get_upload", {"upload_id": upload_id}, "auth-disabled")
    sheet = payload["sheets"][0]
    assert sheet["total_rows"] == uploads.MAX_ROWS + 25
    assert sheet["rows_shown"] == uploads.MAX_ROWS
    assert sheet["rows_capped"] is True
    assert sheet["rows_omitted"] == 25
    assert "cap_note" in payload


def test_get_upload_missing_id_is_honest_not_error(client: TestClient) -> None:
    # A missing id is a not_found status (no "error" key), so leak scans stay green.
    payload = tools.execute_read_tool("get_upload", {"upload_id": "deadbeef"}, "auth-disabled")
    assert "error" not in payload
    assert payload["status"] == "not_found"


# --- per-user isolation (real sessions) -----------------------------------------
@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    from kairos_api import auth_store

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    auth_store.seed_initial_admin(password="rootpass-1234")
    auth_store.add_user("usera", "userapass-123", "operator", "A", must_change_password=False)
    auth_store.add_user("userb", "userbpass-123", "operator", "B", must_change_password=False)
    yield auth_store
    auth_store.reset_runtime_state()


def _session_client(auth_store, username: str, role: str) -> TestClient:
    app = FastAPI()
    app.include_router(assistant.router)
    session = TestClient(app)
    session.cookies.set(auth_store.COOKIE_NAME, auth_store.create_session(username, role))
    return session


def test_uploads_are_isolated_per_session_user(auth_env) -> None:
    user_a = _session_client(auth_env, "usera", "operator")
    user_b = _session_client(auth_env, "userb", "operator")

    data = _xlsx_bytes([["advertiser_id"], ["Acme Beverages"]])
    upload_id = _upload(user_a, "a.xlsx", data).json()["upload_id"]

    # B never sees A's upload, over the wire or through the read tool.
    assert user_b.get("/api/assistant/uploads").json()["uploads"] == []
    assert uploads.get_summary("userb", upload_id) is None
    assert user_b.delete(f"/api/assistant/uploads/{upload_id}").status_code == 404
    # A still has it after B's failed probes.
    assert [item["upload_id"] for item in user_a.get("/api/assistant/uploads").json()["uploads"]] == [upload_id]


# --- find_advertiser ------------------------------------------------------------
def test_find_advertiser_matches_the_crafted_store(tmp_advertisers: Path) -> None:
    payload = tools.execute_read_tool("find_advertiser", {"name": "acme"}, "auth-disabled")
    assert payload["source"] == "advertiser rules store"
    assert payload["candidates"], "a fuzzy match on 'acme' must find Acme Beverages"
    top = payload["candidates"][0]
    assert top["advertiser_id"] == "Acme Beverages"
    # The full current record rides along, not just the name.
    assert top["default_premium"] == 1.2
    assert {"allow_positions", "prime_time_only", "notes"} <= set(top)

    empty = tools.execute_read_tool("find_advertiser", {"name": ""}, "auth-disabled")
    assert "error" not in empty and empty["candidates"] == []


# --- propose_advertiser_change: field-level diff --------------------------------
def test_propose_advertiser_update_carries_a_field_diff(tmp_advertisers: Path) -> None:
    item = tools.build_proposal_item(
        "propose_advertiser_change",
        {"advertiser_name": "Acme Beverages", "changes": {"default_premium": 1.5},
         "reason": "the signed rate sheet, cell B2"},
    )
    assert item["status"] == "pending"
    assert item["kind"] == "advertiser_change"
    assert item["diff"] == [{"field": "default_premium", "before": 1.2, "after": 1.5}]


def test_propose_advertiser_create_has_null_before(tmp_advertisers: Path) -> None:
    item = tools.build_proposal_item(
        "propose_advertiser_change",
        {"advertiser_name": "Initech", "create": True,
         "changes": {"default_premium": 1.1, "notes": "new agreement"}, "reason": "onboarding"},
    )
    assert item["status"] == "pending"
    assert item["diff"] == [
        {"field": "default_premium", "before": None, "after": 1.1},
        {"field": "notes", "before": None, "after": "new agreement"},
    ]


def test_propose_advertiser_rejects_unknown_edit_and_duplicate_create(tmp_advertisers: Path) -> None:
    missing = tools.build_proposal_item(
        "propose_advertiser_change",
        {"advertiser_name": "Nobody Inc", "changes": {"default_premium": 1.0}, "reason": "r"},
    )
    assert missing["status"] == "rejected" and "create true" in missing["error"]

    dup = tools.build_proposal_item(
        "propose_advertiser_change",
        {"advertiser_name": "Acme Beverages", "create": True,
         "changes": {"default_premium": 1.0}, "reason": "r"},
    )
    assert dup["status"] == "rejected" and "already exists" in dup["error"]


# --- settings_change now carries a diff too -------------------------------------
def test_settings_change_item_carries_a_field_diff(tmp_path, monkeypatch) -> None:
    from kairos_api import core

    settings = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", settings)
    monkeypatch.setattr(core, "SETTINGS_PATH", settings)
    old_weight = core._load_settings().revenue_weight
    new_weight = 85 if old_weight != 85 else 90

    item = tools.build_proposal_item(
        "propose_settings_change", {"changes": {"revenue_weight": new_weight}, "reason": "r"}
    )
    assert item["status"] == "pending"
    assert item["diff"] == [{"field": "revenue_weight", "before": old_weight, "after": new_weight}]


# --- apply executor: real store write, undoable snapshot ------------------------
def test_apply_advertiser_change_writes_store_and_rides_snapshot(tmp_advertisers: Path) -> None:
    before = tmp_advertisers.read_bytes()
    item = tools.build_proposal_item(
        "propose_advertiser_change",
        {"advertiser_name": "Acme Beverages", "changes": {"default_premium": 1.75}, "reason": "r"},
    )
    batch = actions.create_batch("edit acme", [item], "auth-disabled", "claude-sonnet-4-6")

    response = actions.apply_proposals(
        batch["batch_id"], actions.ItemIdsRequest(item_ids=[item["id"]]), None
    )
    assert response["results"][0]["status"] == "applied"
    # The real advertisers store applied the edit.
    updated = advertisers_api.list_advertisers()["advertisers"]
    acme = next(row for row in updated if row["advertiser_id"] == "Acme Beverages")
    assert acme["default_premium"] == 1.75
    assert tmp_advertisers.read_bytes() != before

    # A restore point was taken before the write and covers the advertiser file.
    restore_id = response["restore_id"]
    assert restore_id
    manifest = json.loads((actions._restore_root() / restore_id / "manifest.json").read_text())
    assert [entry["path"] for entry in manifest["files"]] == [str(tmp_advertisers)]

    # The apply is undoable: restoring puts the original bytes back.
    restored = actions.restore_state(restore_id, None)
    assert restored["restored"] == [str(tmp_advertisers)]
    assert tmp_advertisers.read_bytes() == before


# --- operator handbook system wiring --------------------------------------------
def test_handbook_loads_as_a_second_cached_system_block() -> None:
    blocks = assistant._system_blocks()
    assert len(blocks) == 2
    # The breakpoint sits on the LAST block only, so the whole prefix caches as one.
    assert "cache_control" not in blocks[0]
    assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
    # The first block is the grounding contract; the second is the handbook.
    assert "CONTEXT block" in blocks[0]["text"]
    assert "operator handbook" in blocks[1]["text"].lower()
    assert "competitor boundary" in blocks[1]["text"].lower()


def test_handbook_missing_file_is_an_honest_single_block(monkeypatch) -> None:
    monkeypatch.setattr(assistant, "HANDBOOK_PATH", ROOT / "docs" / "no-such-handbook.md")
    assistant._HANDBOOK_CACHE["mtime"] = None
    blocks = assistant._system_blocks()
    assert len(blocks) == 1
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}


# --- the mocked loop reads an upload and proposes an advertiser change ----------
def _turn(*blocks: SimpleNamespace, stop: str = "tool_use") -> SimpleNamespace:
    return SimpleNamespace(content=list(blocks), stop_reason=stop)


def _text(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _use(name: str, args: dict[str, Any], block_id: str) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=args, id=block_id)


def test_mocked_loop_reads_upload_and_proposes_advertiser_change(
    client: TestClient, tmp_advertisers: Path, monkeypatch
) -> None:
    monkeypatch.setenv("KAIROS_ASSISTANT_API_KEY", "test-key")
    monkeypatch.setattr(assistant, "_compose_context", lambda question: ({}, ["settings"]))
    data = _xlsx_bytes([["advertiser", "premium"], ["Acme Beverages", "1.9"]])
    upload_id = _upload(client, "acme_deal.xlsx", data).json()["upload_id"]

    turns = [
        _turn(_use("get_upload", {"upload_id": upload_id}, "t1"),
              _use("find_advertiser", {"name": "Acme Beverages"}, "t2")),
        _turn(_use("propose_advertiser_change",
                   {"advertiser_name": "Acme Beverages", "changes": {"default_premium": 1.9},
                    "reason": "the uploaded rate sheet, premium column"}, "t3")),
        _turn(_text("proposed the premium edit from the agreement for your approval"), stop="end_turn"),
    ]
    remaining = list(turns)

    def factory(api_key: str) -> Any:
        def create(**kwargs: Any) -> Any:
            return remaining.pop(0)

        return SimpleNamespace(messages=SimpleNamespace(create=create))

    monkeypatch.setattr(assistant, "_client_factory", factory)
    body = client.post("/api/assistant/ask", json={"question": "apply the Acme agreement I uploaded"}).json()

    assert body["error"] is None
    assert [(step["tool"], step["ok"]) for step in body["tool_trace"]] == [
        ("get_upload", True), ("find_advertiser", True), ("propose_advertiser_change", True)]
    item = body["proposals"]["items"][0]
    assert item["kind"] == "advertiser_change" and item["status"] == "pending"
    assert item["diff"] == [{"field": "default_premium", "before": 1.2, "after": 1.9}]
