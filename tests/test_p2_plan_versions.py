"""P2: freezing a plan version, diffing it and rolling it back.

The weekly plan was the one operational artifact nothing versioned, so JS-2's
done condition, "a named, dated plan version is published with an author and a
timestamp", could not be met at all: the word publish appeared zero times in
``kairos_api`` and the plan is not one of the nine logical files the operation
state store captures.

Every test here runs against a relocated store and a relocated plan, so nothing
in the repository is written and the assertions are about the store's own
behaviour rather than about whatever plan happens to be on disk.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from kairos_api import plan_version_store


COLUMNS = ["channel", "date", "day", "num_breaks", "total_break_time", "predicted_revenue"]


def _plan(rows) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=COLUMNS)


BASE_ROWS = [
    ["רשת 13", "2024-11-01", "Fri", 3, 360.0, 1000.0],
    ["רשת 13", "2024-11-02", "Sat", 2, 240.0, 800.0],
    ["קשת 12", "2024-11-01", "Fri", 4, 480.0, 5000.0],
    ["כאן 11", "2024-11-02", "Sat", 1, 120.0, 400.0],
]


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """A relocated version store, a relocated plan, and a fixed operator channel."""
    monkeypatch.setenv(plan_version_store.PLAN_VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    plan = tmp_path / "weekly_break_schedule.csv"
    monkeypatch.setattr(plan_version_store, "plan_path", lambda: plan)
    monkeypatch.setattr(plan_version_store, "meta_path", lambda: plan.with_name(plan.name + ".meta.json"))
    monkeypatch.setattr(plan_version_store, "_settings_basis", lambda: {
        "revenue_weight": 60, "min_retention_floor": 0.72, "max_breaks_per_hour": 4,
        "risk_lambda": 0.0, "objective_mode": "blend", "operator_channel": "רשת 13",
    })
    monkeypatch.setattr(plan_version_store.channel_scope, "operator_channel", lambda settings=None: "רשת 13")
    _plan(BASE_ROWS).to_csv(plan, index=False, encoding="utf-8")
    plan.with_name(plan.name + ".meta.json").write_text(
        json.dumps({"computed_at": "2026-07-28T08:38:38.170135+00:00", "fingerprints": {"data": "abc"}}),
        encoding="utf-8",
    )
    return plan


def test_a_freeze_records_the_bytes_the_author_and_the_provenance(store):
    manifest = plan_version_store.freeze(name="week one", actor="dana", note="the baseline")
    assert manifest["name"] == "week one"
    assert manifest["actor"] == "dana"
    assert manifest["note"] == "the baseline"
    assert manifest["created_at"]
    assert manifest["plan_sha256"] == hashlib.sha256(store.read_bytes()).hexdigest()
    # The engine's own stamp for the frozen file, not a second clock invented here.
    assert manifest["computed_at"] == "2026-07-28T08:38:38.170135+00:00"
    assert manifest["input_fingerprints"] == {"data": "abc"}
    assert manifest["settings_basis"]["operator_channel"] == "רשת 13"


def test_the_headline_money_is_the_operators_and_says_so(store):
    manifest = plan_version_store.freeze(name="week one", actor="dana")
    owned = manifest["summary"]["owned"]
    every = manifest["summary"]["all_channels"]
    # Two rows of רשת 13, five breaks, 1,800 ILS. The competitors' 5,400 is in
    # all_channels and never in the headline.
    assert owned == {
        "rows": 2, "breaks": 5, "ad_seconds": 600, "revenue": 1800.0,
        "channels": 1, "days": 2, "date_from": "2024-11-01", "date_to": "2024-11-02",
    }
    assert every["revenue"] == 7200.0
    assert every["channels"] == 3
    note = manifest["summary"]["scope"]
    assert note["scope_channel"] == "רשת 13"
    assert note["scoped"] is True
    assert note["competitor_rows_excluded"] == 2
    assert note["competitor_channels_excluded"] == 2


def test_a_version_without_a_name_is_refused(store):
    with pytest.raises(ValueError):
        plan_version_store.freeze(name="   ", actor="dana")
    assert plan_version_store.all_manifests() == []


def test_freezing_with_no_saved_plan_is_refused_rather_than_faked(tmp_path, monkeypatch):
    monkeypatch.setenv(plan_version_store.PLAN_VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.setattr(plan_version_store, "plan_path", lambda: tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError):
        plan_version_store.freeze(name="week one", actor="dana")


def test_live_state_reports_whether_the_plan_on_disk_is_already_frozen(store):
    before = plan_version_store.live_state()
    assert before["exists"] is True
    assert before["frozen_as"] is None
    manifest = plan_version_store.freeze(name="week one", actor="dana")
    after = plan_version_store.live_state()
    # A sha256 comparison, so a re-run that produced the same plan reads as
    # already frozen rather than as a change.
    assert after["frozen_as"] == manifest["version_id"]


def test_the_first_version_says_there_is_nothing_before_it(store):
    manifest = plan_version_store.freeze(name="week one", actor="dana")
    answer = plan_version_store.diff(manifest["version_id"])
    assert answer["available"] is False
    assert answer["reason_code"] == "first_version"


def test_a_diff_names_the_days_that_moved_and_the_money_that_moved_with_them(store):
    first = plan_version_store.freeze(name="week one", actor="dana")
    moved = [
        ["רשת 13", "2024-11-01", "Fri", 2, 240.0, 700.0],
        ["רשת 13", "2024-11-02", "Sat", 2, 240.0, 800.0],
        ["קשת 12", "2024-11-01", "Fri", 9, 999.0, 9999.0],
        ["כאן 11", "2024-11-02", "Sat", 1, 120.0, 400.0],
    ]
    _plan(moved).to_csv(store, index=False, encoding="utf-8")
    second = plan_version_store.freeze(name="week one, trimmed", actor="dana")

    answer = plan_version_store.diff(second["version_id"])
    assert answer["available"] is True
    assert answer["against"] == first["version_id"]
    assert answer["identical"] is False
    # One break and 300 ILS came off the operator's own Friday. The competitor's
    # 4,999 ILS swing is not in any figure here.
    assert answer["delta"] == {"rows": 0, "breaks": -1, "ad_seconds": -120, "revenue": -300.0}
    assert [day["date"] for day in answer["changed_days"]] == ["2024-11-01"]
    assert answer["changed_days"][0]["breaks_delta"] == -1
    assert answer["changed_days"][0]["revenue_delta"] == -300.0
    assert answer["scope"]["scope_channel"] == "רשת 13"


def test_two_identical_versions_diff_to_nothing_moved(store):
    plan_version_store.freeze(name="one", actor="dana")
    second = plan_version_store.freeze(name="two", actor="dana")
    answer = plan_version_store.diff(second["version_id"])
    assert answer["identical"] is True
    assert answer["changed_days"] == []


def test_a_diff_against_live_is_what_a_planner_reads_before_rolling_back(store):
    first = plan_version_store.freeze(name="week one", actor="dana")
    _plan(BASE_ROWS[:1]).to_csv(store, index=False, encoding="utf-8")
    answer = plan_version_store.diff(first["version_id"], against="live")
    assert answer["available"] is True
    assert answer["against"] == "live"
    # The frozen version has the Saturday the live plan lost.
    assert answer["delta"]["breaks"] == 2
    assert answer["delta"]["revenue"] == 800.0


def test_a_restore_is_byte_identical_and_freezes_what_it_replaces(store):
    first = plan_version_store.freeze(name="week one", actor="dana")
    original = store.read_bytes()
    _plan(BASE_ROWS[:1]).to_csv(store, index=False, encoding="utf-8")
    assert store.read_bytes() != original

    result = plan_version_store.restore(first["version_id"], actor="dana")
    assert result["ok"] is True
    assert store.read_bytes() == original
    assert result["plan_sha256"] == hashlib.sha256(original).hexdigest()
    # The plan it replaced was frozen first, so the rollback is itself reversible.
    safety = plan_version_store.get(result["safety_version_id"])
    assert safety is not None
    assert safety["source"] == "pre_restore"
    assert safety["summary"]["owned"]["rows"] == 1
    # The freshness sidecar travelled with the bytes.
    meta = json.loads(plan_version_store.meta_path().read_text(encoding="utf-8"))
    assert meta["computed_at"] == "2026-07-28T08:38:38.170135+00:00"


def test_restoring_an_unknown_version_raises_rather_than_writing_anything(store):
    original = store.read_bytes()
    with pytest.raises(KeyError):
        plan_version_store.restore("nope", actor="dana")
    assert store.read_bytes() == original


def test_the_store_prunes_to_its_ceiling_and_keeps_the_newest(store, monkeypatch):
    monkeypatch.setattr(plan_version_store, "MAX_PLAN_VERSIONS", 3)
    for index in range(5):
        plan_version_store.freeze(name=f"version {index}", actor="dana")
    kept = plan_version_store.all_manifests()
    assert len(kept) == 3
    assert kept[0]["name"] == "version 4"
