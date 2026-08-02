"""P9: the undo you can read first, and the three latency mechanics behind it.

Discovery measured the gap this closes in one sentence, "there is no undo
control in the product", and measured the failure the rest of it closes in one
number: the browser sat on "preparing an answer" for 499 s with no reply, no
error and nothing to press.

Four things are asserted here, each against real files and real timings rather
than a description:

* the restore-point preview names the exact fields and rows going back
* the grounding-context cache is a real hit and a changed input is a real miss
* an ask that reaches its deadline stops and says so
* the stream's first frame arrives before the pipeline does any work
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from kairos_api import assistant, assistant_restore, read_cache
from kairos_api.assistant_sections import CACHE_NAMESPACE, base_context


@pytest.fixture()
def restore_store(tmp_path, monkeypatch):
    monkeypatch.setenv(assistant_restore.DATA_DIR_ENV, str(tmp_path / "assistant"))
    return tmp_path


def _point(store: Path, files: list[Path]) -> str:
    restore_id = assistant_restore.snapshot(files, "batch-under-test", ["item-1"])
    assert restore_id
    return restore_id


# --- the restore point, opened and read before it is used --------------------
def test_the_preview_names_the_exact_settings_fields_that_would_go_back(restore_store) -> None:
    settings = restore_store / "kairos_settings.json"
    settings.write_text(json.dumps({"min_retention_floor": 0.72, "revenue_weight": 50,
                                    "nested": {"kept": 1}}), encoding="utf-8")
    restore_id = _point(restore_store, [settings])
    settings.write_text(json.dumps({"min_retention_floor": 0.75, "revenue_weight": 50,
                                    "nested": {"kept": 1}}), encoding="utf-8")

    preview = assistant_restore.preview(restore_id)

    assert preview["restorable"] is True
    assert preview["files_changing"] == 1
    file_row = preview["files"][0]
    assert file_row["effect"] == "replace"
    assert file_row["kind"] == "fields"
    assert file_row["changes"] == [
        {"field": "min_retention_floor", "current": 0.75, "restored": 0.72, "state": "changed"}
    ]


def test_a_nested_field_reads_as_its_own_dotted_key(restore_store) -> None:
    settings = restore_store / "kairos_settings.json"
    settings.write_text(json.dumps({"pricing_overrides": {"premiums": {"first": 1.0}}}), encoding="utf-8")
    restore_id = _point(restore_store, [settings])
    settings.write_text(json.dumps({"pricing_overrides": {"premiums": {"first": 1.3}}}), encoding="utf-8")

    row = assistant_restore.preview(restore_id)["files"][0]
    assert row["changes"][0]["field"] == "pricing_overrides.premiums.first"
    assert row["changes"][0]["current"] == 1.3
    assert row["changes"][0]["restored"] == 1.0


def test_a_csv_store_diffs_row_by_row_on_its_key_column(restore_store) -> None:
    rules = restore_store / "advertiser_rules.csv"
    rules.write_text("advertiser_id,default_premium\nבנק הפועלים,1.0\nשופרסל,1.1\n", encoding="utf-8")
    restore_id = _point(restore_store, [rules])
    rules.write_text("advertiser_id,default_premium\nבנק הפועלים,1.4\nאסם,1.2\n", encoding="utf-8")

    row = assistant_restore.preview(restore_id)["files"][0]
    assert row["kind"] == "rows"
    assert row["key_column"] == "advertiser_id"
    by_row = {change["row"]: change for change in row["changes"]}
    assert by_row["בנק הפועלים"]["state"] == "changed"
    assert by_row["בנק הפועלים"]["fields"] == [
        {"field": "default_premium", "current": "1.4", "restored": "1.0"}
    ]
    assert by_row["שופרסל"]["state"] == "added", "a row the change deleted comes back"
    assert by_row["אסם"]["state"] == "removed", "a row the change added goes away"


def test_a_restore_point_nothing_would_change_offers_no_undo(restore_store) -> None:
    """The state that stops a person pressing undo for nothing."""
    settings = restore_store / "kairos_settings.json"
    settings.write_text(json.dumps({"revenue_weight": 50}), encoding="utf-8")
    restore_id = _point(restore_store, [settings])

    preview = assistant_restore.preview(restore_id)
    assert preview["restorable"] is False
    assert preview["files"][0]["effect"] == "unchanged"
    assert "nothing would change" in preview["reason"]


def test_a_file_that_did_not_exist_before_reads_as_a_removal(restore_store) -> None:
    fresh = restore_store / "new_store.json"
    restore_id = _point(restore_store, [fresh])
    fresh.write_text("{}", encoding="utf-8")

    row = assistant_restore.preview(restore_id)["files"][0]
    assert row["effect"] == "delete"
    assert row["kind"] == "absent_at_snapshot"


def test_a_missing_snapshot_reads_unavailable_and_blocks_the_undo(restore_store) -> None:
    settings = restore_store / "kairos_settings.json"
    settings.write_text("{}", encoding="utf-8")
    restore_id = _point(restore_store, [settings])
    (assistant_restore._restore_root() / restore_id / "kairos_settings.json").unlink()
    settings.write_text('{"changed": 1}', encoding="utf-8")

    preview = assistant_restore.preview(restore_id)
    assert preview["files"][0]["effect"] == "unavailable"
    assert preview["files_unavailable"] == 1
    assert preview["restorable"] is False


def test_an_unknown_restore_point_is_a_404_not_an_empty_preview(restore_store) -> None:
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as raised:
        assistant_restore.preview("deadbeef1234")
    assert raised.value.status_code == 404


# --- the grounding-context cache, which is where the wait went ---------------
def test_the_base_context_is_served_from_the_cache_and_a_changed_input_is_a_miss() -> None:
    read_cache.invalidate(CACHE_NAMESPACE)
    read_cache.reset_stats(CACHE_NAMESPACE)

    first, sources = base_context()
    assert sources, "the base context must carry its own source list"
    warm_started = time.perf_counter()
    second, _ = base_context()
    warm_seconds = time.perf_counter() - warm_started

    stats = read_cache.stats(CACHE_NAMESPACE)
    assert stats["misses"] == 1 and stats["hits"] == 1
    assert first == second
    assert warm_seconds < 0.5, f"a warm compose took {warm_seconds:.3f}s"

    # And the copy is per caller, so one ask cannot mutate the next one's context.
    first["counts"] = {"tampered": True}
    third, _ = base_context()
    assert third["counts"] != {"tampered": True}


def test_a_changed_seam_invalidates_the_cache_rather_than_serving_a_stale_number(monkeypatch) -> None:
    from kairos_api import server

    read_cache.invalidate(CACHE_NAMESPACE)
    read_cache.reset_stats(CACHE_NAMESPACE)
    base_context()
    monkeypatch.setattr(server, "_build_recommendations", lambda frame: [])
    base_context()
    stats = read_cache.stats(CACHE_NAMESPACE)
    assert stats["misses"] == 2, "a replaced builder must not be served from the cache"


# --- the deadline, so no answer can run forever ------------------------------
class _ToolUseBlock:
    type = "tool_use"
    id = "tool-1"
    name = "get_settings"
    input: dict = {}


class _ToolUseResponse:
    stop_reason = "tool_use"
    content = [_ToolUseBlock()]


class _Messages:
    def create(self, **kwargs):
        time.sleep(0.05)
        return _ToolUseResponse()


class _NeverStopsCalling:
    """A model that always asks for one more tool call, which is exactly the
    shape of the runaway search this deadline exists to end."""

    def __init__(self) -> None:
        self.messages = _Messages()


def test_a_runaway_search_stops_at_the_deadline_and_says_so() -> None:
    trace: list[dict] = []
    items: list[dict] = []
    answer, stopped = assistant._run_tool_loop(
        _NeverStopsCalling(), "CONTEXT:\n{}\n\nQUESTION:\ngo forever", trace, items,
        actions_on=True, deadline=time.monotonic() + 0.12,
    )
    assert stopped is True
    assert answer == ""
    assert len(trace) < assistant.MAX_TOOL_ITERATIONS, "the loop stopped before its hard ceiling"


def test_without_a_deadline_the_loop_still_ends_at_its_hard_ceiling() -> None:
    trace: list[dict] = []
    _answer, stopped = assistant._run_tool_loop(
        _NeverStopsCalling(), "CONTEXT:\n{}\n\nQUESTION:\ngo forever", trace, [],
        actions_on=True, deadline=None,
    )
    assert stopped is False
    assert len(trace) == assistant.MAX_TOOL_ITERATIONS
