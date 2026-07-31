"""The read cache serves a stale value never, and an unbounded one never.

A cache on a money path is a correctness component, not a performance one: the
only way it can hurt is by answering with a value whose inputs have changed.
These tests hold :mod:`kairos_api.read_cache` to that, plus the bound that keeps
a per-day key space from growing forever.
"""

from __future__ import annotations

import threading

import pytest

from kairos_api import read_cache

NAMESPACE = "test_w0_5"


@pytest.fixture(autouse=True)
def clean_namespace():
    read_cache.invalidate(NAMESPACE)
    read_cache.reset_stats(NAMESPACE)
    read_cache.configure(NAMESPACE, capacity=read_cache.DEFAULT_CAPACITY)
    yield
    read_cache.invalidate(NAMESPACE)
    read_cache.reset_stats(NAMESPACE)


def counting_build(values: list):
    def build():
        values.append(len(values) + 1)
        return values[-1]
    return build


def test_a_matching_fingerprint_is_a_hit_and_does_not_rebuild() -> None:
    built: list = []
    first = read_cache.cached(NAMESPACE, "k", ("fp", 1), counting_build(built))
    second = read_cache.cached(NAMESPACE, "k", ("fp", 1), counting_build(built))
    assert first == second == 1
    assert len(built) == 1
    assert read_cache.stats(NAMESPACE)["hits"] == 1


def test_a_changed_fingerprint_rebuilds_rather_than_serving_the_old_value() -> None:
    built: list = []
    read_cache.cached(NAMESPACE, "k", ("fp", 1), counting_build(built))
    again = read_cache.cached(NAMESPACE, "k", ("fp", 2), counting_build(built))
    assert again == 2
    assert len(built) == 2
    assert read_cache.stats(NAMESPACE)["misses"] == 2


def test_two_keys_do_not_answer_for_each_other() -> None:
    built: list = []
    a = read_cache.cached(NAMESPACE, ("channel", "day-a"), "fp", counting_build(built))
    b = read_cache.cached(NAMESPACE, ("channel", "day-b"), "fp", counting_build(built))
    assert a != b
    assert read_cache.stats(NAMESPACE)["entries"] == 2


def test_capacity_evicts_least_recently_used() -> None:
    read_cache.configure(NAMESPACE, capacity=2)
    built: list = []
    for key in ("a", "b"):
        read_cache.cached(NAMESPACE, key, "fp", counting_build(built))
    read_cache.cached(NAMESPACE, "a", "fp", counting_build(built))  # a becomes recent
    read_cache.cached(NAMESPACE, "c", "fp", counting_build(built))  # evicts b
    stats = read_cache.stats(NAMESPACE)
    assert stats["entries"] == 2
    assert stats["evictions"] == 1
    hit, _ = read_cache.lookup(NAMESPACE, "a", "fp")
    assert hit
    miss, _ = read_cache.lookup(NAMESPACE, "b", "fp")
    assert not miss


def test_invalidate_drops_a_key_a_namespace_or_everything() -> None:
    built: list = []
    read_cache.cached(NAMESPACE, "a", "fp", counting_build(built))
    read_cache.cached(NAMESPACE, "b", "fp", counting_build(built))
    assert read_cache.invalidate(NAMESPACE, "a") == 1
    assert read_cache.stats(NAMESPACE)["entries"] == 1
    assert read_cache.invalidate(NAMESPACE) == 1
    assert read_cache.stats(NAMESPACE)["entries"] == 0


def test_file_signature_moves_when_the_file_does(tmp_path) -> None:
    path = tmp_path / "input.csv"
    absent = read_cache.file_signature(path)
    assert absent[1:] == (0, 0)
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    present = read_cache.file_signature(path)
    assert present != absent
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    assert read_cache.file_signature(path) != present


def test_directory_signatures_notice_a_file_appearing(tmp_path) -> None:
    before = read_cache.directory_signatures(tmp_path, "*.yaml")
    (tmp_path / "extra.yaml").write_text("key: value\n", encoding="utf-8")
    after = read_cache.directory_signatures(tmp_path, "*.yaml")
    assert before == ()
    assert len(after) == 1
    assert after != before


def test_a_missing_directory_is_an_empty_signature_not_an_error(tmp_path) -> None:
    assert read_cache.directory_signatures(tmp_path / "nope", "*.yaml") == ()


def test_concurrent_readers_all_get_a_correct_value() -> None:
    """The build runs outside the lock, so a race may build twice; it may never
    hand back a value that does not belong to the fingerprint asked for."""
    results: list = []
    errors: list = []

    def worker(index: int) -> None:
        try:
            value = read_cache.cached(NAMESPACE, "shared", ("fp", 7), lambda: ("built", 7))
            results.append(value)
        except Exception as exc:  # pragma: no cover - a failure is the assertion
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert results == [("built", 7)] * 8


def test_stats_report_every_namespace_without_naming_one() -> None:
    read_cache.cached(NAMESPACE, "k", "fp", lambda: 1)
    everything = read_cache.stats()
    assert NAMESPACE in everything
    assert everything[NAMESPACE]["entries"] == 1


def test_capacity_must_be_positive() -> None:
    with pytest.raises(ValueError):
        read_cache.configure(NAMESPACE, capacity=0)
