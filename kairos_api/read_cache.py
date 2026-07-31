"""A fingerprinted read cache for expensive, placement-independent reads.

The pattern this generalises already exists in the codebase in three hand-rolled
copies: :mod:`kairos.data.loaders` memoizes its parses on ``(path, mtime_ns,
size)``, :mod:`kairos_api.core` pairs an ``lru_cache`` with a ``_signature``
helper, and the frontier keeps its own warm map. Each copy re-derives the same
two rules, and a fourth copy is where a stale answer eventually ships. So the
rules live here once.

The two rules:

1. **A cached value is served only when its fingerprint still matches.** The
   fingerprint is supplied by the caller and is compared with ``==``, so it can
   be any hashable summary of the inputs: file signatures, a settings digest,
   the current date, or the seam functions themselves. A cache keyed on inputs
   alone (and not on their fingerprint) is a stale-answer generator, which on a
   money surface is not a performance question but a correctness one.
2. **Every namespace is bounded.** Entries evict least-recently-used at
   ``DEFAULT_CAPACITY`` so a per-day key space cannot grow without limit.

What this cache is NOT for. It answers a read that is a pure function of files
and settings. It must never hold anything derived from a request body, an
operator identity, or a placement decision, and it must never be the store of
record for a number a person reads: it is a copy of one, valid only while the
fingerprint holds.

Honest limits, stated rather than discovered later:

- The build runs OUTSIDE the lock, so two threads that miss the same key at the
  same time both build it and the last one to finish stores its value. Both
  callers get a correct value, and the only cost is the duplicated build.
- A file signature is ``(path, mtime_ns, size)``. A write that preserves both
  the modification time and the size is invisible to it, which is why a caller
  whose input can be rewritten in place (the settings file) should fold a
  content digest into its fingerprint as well.
- Values are shared, not copied. A caller that hands out a mutable value must
  copy it, or cache only immutable ones.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Optional, TypeVar

T = TypeVar("T")

# Entries kept per namespace before the least recently used one is evicted.
DEFAULT_CAPACITY = 8

_LOCK = threading.Lock()
_ENTRIES: dict[str, "OrderedDict[Hashable, tuple[Hashable, Any]]"] = {}
_CAPACITY: dict[str, int] = {}
_COUNTS: dict[str, dict[str, int]] = {}


def _counts_for(namespace: str) -> dict[str, int]:
    counts = _COUNTS.get(namespace)
    if counts is None:
        counts = {"hits": 0, "misses": 0, "evictions": 0, "stores": 0}
        _COUNTS[namespace] = counts
    return counts


def configure(namespace: str, *, capacity: int) -> None:
    """Set how many entries ``namespace`` keeps before evicting the oldest."""
    if capacity < 1:
        raise ValueError("capacity must be at least 1")
    with _LOCK:
        _CAPACITY[namespace] = capacity
        entries = _ENTRIES.get(namespace)
        if entries is not None:
            _evict_locked(namespace, entries)


def file_signature(path: "str | Path") -> tuple[str, int, int]:
    """``(path, mtime_ns, size)`` for one file, or zeros when it is absent.

    An absent file is a signature in its own right rather than an error, so a
    fingerprint changes the moment an optional input appears or disappears.
    """
    resolved = Path(path)
    try:
        stat = resolved.stat()
    except OSError:
        return (str(resolved), 0, 0)
    return (str(resolved), stat.st_mtime_ns, stat.st_size)


def file_signatures(paths: Iterable["str | Path"]) -> tuple[tuple[str, int, int], ...]:
    """Signatures for several files, in the order given."""
    return tuple(file_signature(path) for path in paths)


def directory_signatures(directory: "str | Path", pattern: str = "*") -> tuple[tuple[str, int, int], ...]:
    """Signatures for every file matching ``pattern`` in ``directory``, sorted.

    A glob rather than a list, so a config file that is added or removed changes
    the fingerprint without anyone remembering to extend an enumeration.
    """
    root = Path(directory)
    try:
        matches = sorted(entry for entry in root.glob(pattern) if entry.is_file())
    except OSError:
        return ()
    return tuple(file_signature(entry) for entry in matches)


def lookup(namespace: str, key: Hashable, fingerprint: Hashable) -> tuple[bool, Any]:
    """``(hit, value)`` for this key, counting the hit or miss."""
    with _LOCK:
        counts = _counts_for(namespace)
        entries = _ENTRIES.get(namespace)
        if entries is not None and key in entries:
            stored_fingerprint, value = entries[key]
            if stored_fingerprint == fingerprint:
                entries.move_to_end(key)
                counts["hits"] += 1
                return True, value
        counts["misses"] += 1
        return False, None


def store(namespace: str, key: Hashable, fingerprint: Hashable, value: Any) -> None:
    """Record ``value`` under this key and fingerprint, evicting if needed."""
    with _LOCK:
        entries = _ENTRIES.setdefault(namespace, OrderedDict())
        entries[key] = (fingerprint, value)
        entries.move_to_end(key)
        _counts_for(namespace)["stores"] += 1
        _evict_locked(namespace, entries)


def _evict_locked(namespace: str, entries: "OrderedDict[Hashable, tuple[Hashable, Any]]") -> None:
    """Drop least-recently-used entries down to the namespace capacity."""
    capacity = _CAPACITY.get(namespace, DEFAULT_CAPACITY)
    while len(entries) > capacity:
        entries.popitem(last=False)
        _counts_for(namespace)["evictions"] += 1


def cached(namespace: str, key: Hashable, fingerprint: Hashable, build: Callable[[], T]) -> T:
    """The value for ``key``, built by ``build`` unless a matching one is held.

    ``build`` is called with the lock released, so a slow build never blocks a
    reader of another key.
    """
    hit, value = lookup(namespace, key, fingerprint)
    if hit:
        return value
    built = build()
    store(namespace, key, fingerprint, built)
    return built


def invalidate(namespace: Optional[str] = None, key: Optional[Hashable] = None) -> int:
    """Drop one key, one namespace, or everything. Returns entries removed."""
    with _LOCK:
        if namespace is None:
            removed = sum(len(entries) for entries in _ENTRIES.values())
            _ENTRIES.clear()
            return removed
        entries = _ENTRIES.get(namespace)
        if entries is None:
            return 0
        if key is None:
            removed = len(entries)
            entries.clear()
            return removed
        return 1 if entries.pop(key, None) is not None else 0


def stats(namespace: Optional[str] = None) -> dict[str, Any]:
    """Hit, miss, store and eviction counts, plus the entries held now.

    Observability for a cache is not optional: a cache nobody can measure is a
    cache nobody can prove is helping.
    """
    with _LOCK:
        if namespace is not None:
            counts = dict(_counts_for(namespace))
            counts["entries"] = len(_ENTRIES.get(namespace, ()))
            counts["capacity"] = _CAPACITY.get(namespace, DEFAULT_CAPACITY)
            return counts
        return {
            name: {
                **_counts_for(name),
                "entries": len(_ENTRIES.get(name, ())),
                "capacity": _CAPACITY.get(name, DEFAULT_CAPACITY),
            }
            for name in sorted(set(_ENTRIES) | set(_COUNTS))
        }


def reset_stats(namespace: Optional[str] = None) -> None:
    """Zero the counters without dropping any cached value."""
    with _LOCK:
        if namespace is None:
            _COUNTS.clear()
            return
        _COUNTS.pop(namespace, None)
