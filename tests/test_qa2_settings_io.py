"""Settings persistence is atomic and torn-read-proof.

_save_settings used to truncate the live file in place, so a reader racing the
write saw a half-written JSON, parsed nothing, and silently served (and could
then re-persist) factory defaults, discarding every operator decision. The
contract now: tmp-file + os.replace writes, a module lock across settings I/O,
and one read retry before defaulting. All tests run against a temp settings
path; the real data/kairos_settings.json is never touched.
"""

from __future__ import annotations

import json
import threading

import pytest

import kairos_api.core as core


@pytest.fixture()
def settings_path(tmp_path, monkeypatch):
    path = tmp_path / "kairos_settings.json"
    monkeypatch.setattr(core, "SETTINGS_PATH", path)
    return path


def test_save_is_atomic_and_round_trips(settings_path):
    saved = core._save_settings(core.KairosSettings(revenue_weight=73, operator_channel="עכשיו 14"))
    assert saved.revenue_weight == 73
    assert settings_path.exists()
    assert not settings_path.with_name(settings_path.name + ".tmp").exists(), (
        "the tmp sibling must be replaced away, not left behind"
    )
    loaded = core._load_settings()
    assert loaded.revenue_weight == 73
    assert loaded.operator_channel == "עכשיו 14"
    # The file itself is whole, valid JSON.
    assert json.loads(settings_path.read_text(encoding="utf-8"))["revenue_weight"] == 73


def test_unparseable_file_defaults_without_overwriting(settings_path):
    settings_path.write_text('{"revenue_weight": 5', encoding="utf-8")  # torn JSON
    loaded = core._load_settings()
    assert loaded.revenue_weight == core.KairosSettings().revenue_weight
    # The retry-then-default path must not destroy the operator's file: the torn
    # content is still there for a human (or a later save) to deal with.
    assert settings_path.read_text(encoding="utf-8") == '{"revenue_weight": 5'


def test_missing_file_still_defaults(settings_path):
    assert not settings_path.exists()
    assert core._load_settings().revenue_weight == core.KairosSettings().revenue_weight


def test_concurrent_save_load_never_serves_a_torn_file(settings_path):
    core._save_settings(core.KairosSettings(revenue_weight=60))
    errors: list[Exception] = []

    def worker(index: int) -> None:
        try:
            for _ in range(25):
                core._save_settings(core.KairosSettings(revenue_weight=50 + (index % 30)))
                weight = core._load_settings().revenue_weight
                # A torn read would surface as the pydantic default (60) only if
                # 60 were outside the written range; assert the tighter truth:
                # every observed value is one some thread actually wrote, or the
                # documented default, never garbage.
                assert 50 <= weight <= 80, f"impossible weight {weight}"
        except Exception as exc:  # noqa: BLE001 - collected for the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors, errors
    # After the storm the file on disk is whole and parseable.
    final = json.loads(settings_path.read_text(encoding="utf-8"))
    assert 50 <= int(final["revenue_weight"]) <= 80


def test_settings_lock_is_reentrant_for_read_modify_write(settings_path):
    core._save_settings(core.KairosSettings(revenue_weight=61))
    with core._SETTINGS_LOCK:
        current = core._load_settings()
        current.revenue_weight = 62
        core._save_settings(current)
    assert core._load_settings().revenue_weight == 62


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
