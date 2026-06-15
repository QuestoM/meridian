"""Tests for the AI fallback classifier seam.

The genuinely unclassified set, the cache-once resolver, the offline-honest
unavailable path, and the wrapper that folds AI verdicts back into classification
are all checked without any network, using a deterministic fake AI.
"""

from __future__ import annotations

import json

from kairos.data import ProgramClassifier
from kairos.data.ai_classifier import (
    AIClassification,
    CachedClassifier,
    load_ai_overrides,
    resolve_unclassified,
    unclassified_titles,
)


class _FakeAI:
    """A deterministic stand-in for the grounded LLM, with a call counter."""

    def __init__(self, verdicts: dict[str, str]) -> None:
        self._verdicts = verdicts
        self.calls = 0

    def classify(self, title, *, allowed):
        self.calls += 1
        category = self._verdicts.get(title, "Other")
        if category not in allowed:
            category = "Other"
        return AIClassification(title, category, 0.8, True, "fake", "ai")


def _classifier() -> ProgramClassifier:
    return ProgramClassifier.from_yaml()


def test_unclassified_excludes_known_genres() -> None:
    clf = _classifier()
    titles = ["חדשות הערב", "זzqx nonsense title 999", "חדשות הערב"]
    pending = unclassified_titles(titles, clf)
    # The news title is known; only the nonsense title is genuinely unclassified,
    # and it appears once despite the duplicate news entry.
    assert "זzqx nonsense title 999" in pending
    assert "חדשות הערב" not in pending
    assert len(pending) == len(set(pending))


def test_resolve_caches_and_calls_once(tmp_path) -> None:
    clf = _classifier()
    cache = tmp_path / "ai.json"
    target = clf.categories[0]
    ai = _FakeAI({"zzz unknown program": target})
    titles = ["zzz unknown program", "zzz unknown program"]

    first = resolve_unclassified(titles, clf, ai, cache_path=cache)
    assert first["zzz unknown program"].category == target
    assert ai.calls == 1  # the duplicate is resolved from the in-run map

    # A second run reads the cache and does not call the AI again.
    second = resolve_unclassified(titles, clf, ai, cache_path=cache)
    assert ai.calls == 1
    assert second["zzz unknown program"].category == target
    assert cache.exists()


def test_resolve_offline_records_unavailable(tmp_path) -> None:
    clf = _classifier()
    cache = tmp_path / "ai.json"
    resolved = resolve_unclassified(["zzz unknown program"], clf, None, cache_path=cache)
    verdict = resolved["zzz unknown program"]
    assert verdict.source == "unavailable"
    assert verdict.category == "Other"
    # Nothing is invented: there are no trusted overrides to fold back in.
    assert load_ai_overrides(cache) == {}


def test_cached_classifier_folds_ai_genre_back(tmp_path) -> None:
    clf = _classifier()
    cache = tmp_path / "ai.json"
    target = clf.categories[0]
    ai = _FakeAI({"zzz unknown program": target})
    resolve_unclassified(["zzz unknown program"], clf, ai, cache_path=cache)

    overrides = load_ai_overrides(cache)
    assert overrides == {"zzz unknown program": target}

    wrapped = CachedClassifier(clf, overrides)
    # The base classifier leaves it Other; the wrapper returns the AI genre.
    assert clf.classify("zzz unknown program").category == "Other"
    folded = wrapped.classify("zzz unknown program")
    assert folded.category == target
    assert folded.rule == "ai"
    # A known title is untouched by the wrapper.
    assert wrapped.classify("חדשות הערב").category == clf.classify("חדשות הערב").category


def test_load_overrides_ignores_untrusted(tmp_path) -> None:
    cache = tmp_path / "ai.json"
    payload = {
        "classifications": {
            "trusted": {"category": "News", "confidence": 0.9, "source": "ai"},
            "abstained": {"category": "Other", "confidence": 0.0, "source": "ai"},
            "offline": {"category": "News", "confidence": 0.0, "source": "unavailable"},
        }
    }
    cache.write_text(json.dumps(payload), encoding="utf-8")
    assert load_ai_overrides(cache) == {"trusted": "News"}
