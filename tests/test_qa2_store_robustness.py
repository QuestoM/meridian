"""Store robustness: locked load-mutate-write cycles, atomic writes, input caps.

Covers the QA2 hardening of the operator-state stores and their seams:

- Every CSV store (overrides, constraints, advertiser rules, advertiser
  conditions) serializes its load-mutate-write cycle under a module lock, so
  concurrent creates cannot lose each other's rows, and writes go through a
  temp file plus os.replace so no reader ever sees a torn file.
- write_weekly_schedule and the assistant proposal store / restore manifests
  write atomically the same way.
- The predicate validator caps nesting depth, total node count and regex
  patterns with an honest 400 instead of a stack-blown 500.
- The synchronous recompute refuses to race a running background recompute job
  (409), mirroring the async path's dedup.

Every test relocates its store into tmp_path; nothing touches real data/.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import kairos_api._constraint_options as co
import kairos_api.advertiser_conditions as conditions_api
import kairos_api.advertisers as advertisers_api
import kairos_api.assistant_actions as actions
import kairos_api.constraints as constraints_api
import kairos_api.overrides as overrides_api
import kairos_api.recompute_api as recompute_api

THREADS = 8


def _no_tmp_leftovers(directory: Path) -> None:
    leftovers = list(directory.glob("*.tmp"))
    assert leftovers == [], f"atomic write left temp files behind: {leftovers}"


# --- concurrent creates never lose rows (module lock across load-mutate-write) ---
def test_concurrent_override_creates_lose_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(overrides_api, "OVERRIDES_PATH", tmp_path / "manual_overrides.csv")
    monkeypatch.setattr(overrides_api, "BACKUP_DIR", tmp_path / "_backups")

    def create(index: int) -> None:
        overrides_api.create_override(
            overrides_api.OverrideCreate(
                scope="segment", target_id=f"2024-11-01|Chan|{index}", kind="gold"),
            request=None,
        )

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(create, range(THREADS)))

    frame = overrides_api._load_frame()
    assert len(frame) == THREADS, "a concurrent create dropped another create's row"
    assert frame["target_id"].nunique() == THREADS
    _no_tmp_leftovers(tmp_path)


def test_concurrent_constraint_creates_lose_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", tmp_path / "kairos_constraints.csv")
    monkeypatch.setattr(constraints_api, "BACKUP_DIR", tmp_path / "_backups")

    def create(index: int) -> None:
        constraints_api.create_constraint(
            constraints_api.ConstraintCreate(
                scope_type="weekday", effect="forbid", scope_value=str(index % 7 + 1)),
            request=None,
        )

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(create, range(THREADS)))

    frame = constraints_api._load_frame()
    assert len(frame) == THREADS
    assert frame["constraint_id"].nunique() == THREADS
    _no_tmp_leftovers(tmp_path)


def test_concurrent_advertiser_creates_lose_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(advertisers_api, "RULES_PATH", tmp_path / "advertiser_rules.csv")
    monkeypatch.setattr(advertisers_api, "BACKUP_DIR", tmp_path / "_backups")

    def create(index: int) -> None:
        advertisers_api.create_advertiser(
            advertisers_api.AdvertiserCreate(advertiser_id=f"ADV-{index}"), request=None)

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(create, range(THREADS)))

    frame = advertisers_api._load_frame()
    assert len(frame) == THREADS
    assert frame["advertiser_id"].nunique() == THREADS
    _no_tmp_leftovers(tmp_path)


def test_concurrent_condition_creates_lose_nothing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(conditions_api, "CONDITIONS_PATH", tmp_path / "advertiser_conditions.csv")
    monkeypatch.setattr(conditions_api, "BACKUP_DIR", tmp_path / "_backups")

    def create(index: int) -> None:
        conditions_api.create_condition(
            "ACME",
            conditions_api.ConditionCreate(rule_id=f"r{index}", effect="premium", value=1.1),
            request=None,
        )

    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        list(pool.map(create, range(THREADS)))

    assert len(conditions_api.conditions_for("ACME")) == THREADS
    _no_tmp_leftovers(tmp_path)


# --- weekly schedule CSV writes atomically ---------------------------------------
def test_write_weekly_schedule_swaps_atomically(tmp_path) -> None:
    from kairos.export.schedule import COLUMNS, write_weekly_schedule

    target = tmp_path / "weekly_break_schedule.csv"
    first = pd.DataFrame([{c: "" for c in COLUMNS} | {"channel": "A", "num_breaks": 1}],
                         columns=COLUMNS)
    second = pd.DataFrame([{c: "" for c in COLUMNS} | {"channel": "B", "num_breaks": 2}],
                          columns=COLUMNS)

    assert write_weekly_schedule(path=target, frame=first) == target
    assert write_weekly_schedule(path=target, frame=second) == target

    read_back = pd.read_csv(target)
    assert list(read_back["channel"]) == ["B"], "the second write must fully replace the first"
    _no_tmp_leftovers(tmp_path)


# --- assistant proposal store and restore manifests write atomically -------------
def test_assistant_store_and_manifest_write_atomically(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv(actions.DATA_DIR_ENV, str(tmp_path / "assistant"))

    actions._save_store({"batches": [{"batch_id": "abc", "items": []}]})
    store_path = tmp_path / "assistant" / "proposals.json"
    assert json.loads(store_path.read_text(encoding="utf-8"))["batches"][0]["batch_id"] == "abc"

    state_file = tmp_path / "state.csv"
    state_file.write_text("a,b\n1,2\n", encoding="utf-8")
    restore_id = actions._snapshot([state_file], "batch-1", ["item-1"])
    manifest_path = tmp_path / "assistant" / "restore" / str(restore_id) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["restore_id"] == restore_id
    assert manifest["files"][0]["existed"] is True

    _no_tmp_leftovers(store_path.parent)
    _no_tmp_leftovers(manifest_path.parent)


# --- predicate guards: depth, node count, regex -----------------------------------
def _nested_group(levels: int) -> dict:
    node: dict = {"field": "genre", "operator": "is", "value": "News"}
    for _ in range(levels):
        node = {"combinator": "and", "conditions": [node]}
    return node


def test_predicate_depth_cap_is_a_400_not_a_500() -> None:
    with pytest.raises(HTTPException) as excinfo:
        co.validate_where(_nested_group(co.MAX_PREDICATE_DEPTH + 5))
    assert excinfo.value.status_code == 400
    assert "nested too deeply" in excinfo.value.detail
    # A tree inside the cap still validates.
    assert co.validate_where(_nested_group(3)) is not None


def test_predicate_node_count_cap_is_a_400() -> None:
    wide = {"combinator": "or", "conditions": [
        {"field": "genre", "operator": "is", "value": "News"}
        for _ in range(co.MAX_PREDICATE_NODES + 10)
    ]}
    with pytest.raises(HTTPException) as excinfo:
        co.validate_where(wide)
    assert excinfo.value.status_code == 400
    assert "too many nodes" in excinfo.value.detail


@pytest.mark.parametrize("pattern, fragment", [
    ("x" * (co.MAX_REGEX_LENGTH + 1), "too long"),
    ("(unclosed", "does not compile"),
    (None, "string pattern"),
])
def test_regex_conditions_are_rejected_at_create_time(pattern, fragment) -> None:
    where = {"combinator": "and", "conditions": [
        {"field": "programme", "operator": "regex", "value": pattern},
    ]}
    with pytest.raises(HTTPException) as excinfo:
        co.validate_where(where)
    assert excinfo.value.status_code == 400
    assert fragment in excinfo.value.detail


def test_valid_regex_condition_still_passes() -> None:
    where = {"combinator": "and", "conditions": [
        {"field": "programme", "operator": "regex", "value": "^חדשות.*ערב$"},
    ]}
    assert co.validate_where(where) == where


def test_regex_guard_applies_through_the_constraint_create_route(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", tmp_path / "kairos_constraints.csv")
    monkeypatch.setattr(constraints_api, "BACKUP_DIR", tmp_path / "_backups")
    app = FastAPI()
    app.include_router(constraints_api.router)
    client = TestClient(app)
    response = client.post("/api/constraints", json={
        "scope_type": "weekday", "effect": "forbid", "scope_value": "1",
        "where": {"combinator": "and", "conditions": [
            {"field": "programme", "operator": "regex", "value": "(broken"},
        ]},
    })
    assert response.status_code == 400
    assert "does not compile" in response.json()["detail"]
    assert not (tmp_path / "kairos_constraints.csv").exists(), "a rejected create must write nothing"


# --- synchronous recompute refuses to race a running background job --------------
def test_sync_recompute_returns_409_while_a_job_runs(monkeypatch) -> None:
    import kairos_api.jobs as jobs

    monkeypatch.setattr(recompute_api, "_ENGINE_AVAILABLE", True)
    monkeypatch.setattr(jobs, "running_job",
                        lambda name: "job-busy" if name == "recompute" else None)
    app = FastAPI()
    app.include_router(recompute_api.router)
    client = TestClient(app)

    response = client.post("/api/recompute-schedule")
    assert response.status_code == 409
    assert "job-busy" in response.json()["detail"]


def test_sync_recompute_proceeds_when_no_job_runs(monkeypatch) -> None:
    import kairos_api.jobs as jobs

    monkeypatch.setattr(recompute_api, "_ENGINE_AVAILABLE", True)
    monkeypatch.setattr(jobs, "running_job", lambda name: None)
    monkeypatch.setattr(recompute_api, "_run_recompute", lambda **kw: {"ok": True, "rows": 1})
    app = FastAPI()
    app.include_router(recompute_api.router)
    client = TestClient(app)

    response = client.post("/api/recompute-schedule")
    assert response.status_code == 200
    assert response.json()["ok"] is True
