"""Advanced program-type classifier for Israeli TV titles.

This is a config-driven, weighted-scoring classifier that replaces the
first-match keyword loop in tv_break_data_transformer._categorize_programs.

Why it is better than a plain keyword loop:
  - Taxonomy lives in kairos/config/program_categories.yaml (editable, no code change).
  - Weighted scoring: a title that matches several keywords picks the category
    with the strongest evidence (longer keywords weigh more), not the first hit.
  - Hebrew normalisation: unifies quote variants (gershayim), strips [..] and " * "
    separators so titles from the Spots log match programme keywords.
  - Rerun aware: detects the "ש.ח" (שידור חוזר) marker and reports it as metadata
    without distorting the genre.
  - Honest unknowns: titles with no evidence return "Other" with confidence 0.0,
    so uncovered titles can be surfaced and the taxonomy improved (no fabrication).

The rich genre is canonical. A legacy_mapping collapses it to the 8-value
PROGRAM_TYPES set still used by tv_break_model.py.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "program_categories.yaml"

# Quote / apostrophe variants unified to a single double-quote before matching.
_QUOTE_VARIANTS = ("“", "”", "″", "״", "’", "׳", "'", "`")
_OTHER = "Other"


@dataclass(frozen=True)
class Classification:
    """Result of classifying one programme title."""

    category: str       # rich genre, or "Other"
    confidence: float   # 0.0 .. 1.0
    rule: str           # "specific" | "keyword" | "fallback" | "empty"
    is_rerun: bool      # title carried a שידור חוזר marker
    legacy_type: str    # collapsed value in the 8-type PROGRAM_TYPES set


def _normalise_token(text: Any) -> str:
    """Lowercase, unify quote variants, and collapse whitespace."""
    value = str(text)
    for variant in _QUOTE_VARIANTS:
        value = value.replace(variant, '"')
    return re.sub(r"\s+", " ", value).strip().lower()


class ProgramClassifier:
    """Classify Israeli TV programme titles into genres using a YAML taxonomy."""

    def __init__(self, config: dict[str, Any]):
        if not isinstance(config, dict):
            raise TypeError("ProgramClassifier config must be a dict")
        self._raw_rerun_markers: tuple[str, ...] = tuple(config.get("rerun_markers", []))

        categories = config.get("categories") or {}
        # Preserve declaration order: earlier category wins a score tie.
        self._priority: dict[str, int] = {name: idx for idx, name in enumerate(categories)}
        self._category_keywords: dict[str, list[str]] = {}
        for name, body in categories.items():
            keywords = (body or {}).get("keywords", []) if isinstance(body, dict) else []
            normalised = [_normalise_token(k) for k in keywords if str(k).strip()]
            self._category_keywords[name] = [k for k in normalised if k]

        specific = config.get("specific_programs") or {}
        # Sort by length desc so the most specific override wins.
        self._specific: list[tuple[str, str]] = sorted(
            ((_normalise_token(key), str(cat)) for key, cat in specific.items()),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )

        self._legacy_map: dict[str, str] = {
            str(k): str(v) for k, v in (config.get("legacy_mapping") or {}).items()
        }

    @classmethod
    def from_yaml(cls, path: str | Path | None = None) -> "ProgramClassifier":
        """Build a classifier from a YAML taxonomy file."""
        config_path = Path(path) if path else DEFAULT_CONFIG_PATH
        with config_path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        return cls(config)

    @property
    def categories(self) -> list[str]:
        return list(self._priority)

    def _legacy(self, category: str) -> str:
        return self._legacy_map.get(category, _OTHER)

    def _prepare_title(self, raw_title: Any) -> tuple[str, bool]:
        """Return (normalised title for matching, is_rerun)."""
        if raw_title is None:
            return "", False
        text = str(raw_title)
        if not text.strip() or text.strip().lower() == "nan":
            return "", False
        is_rerun = any(marker in text for marker in self._raw_rerun_markers)
        for marker in self._raw_rerun_markers:
            text = text.replace(marker, " ")
        text = text.replace("[", " ").replace("]", " ").replace(" * ", " ")
        return _normalise_token(text), is_rerun

    def classify(self, raw_title: Any) -> Classification:
        """Classify a single title into a rich genre with a confidence score."""
        norm, is_rerun = self._prepare_title(raw_title)
        if not norm:
            return Classification(_OTHER, 0.0, "empty", is_rerun, self._legacy(_OTHER))

        for key, category in self._specific:
            if key and key in norm:
                return Classification(category, 1.0, "specific", is_rerun, self._legacy(category))

        scores: dict[str, int] = {}
        for category, keywords in self._category_keywords.items():
            total = sum(len(keyword) for keyword in keywords if keyword in norm)
            if total > 0:
                scores[category] = total

        if not scores:
            return Classification(_OTHER, 0.0, "fallback", is_rerun, self._legacy(_OTHER))

        ranked = sorted(
            scores.items(),
            key=lambda item: (item[1], -self._priority[item[0]]),
            reverse=True,
        )
        best, top = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0
        confidence = round(top / (top + second), 3) if (top + second) else 0.5
        return Classification(best, confidence, "keyword", is_rerun, self._legacy(best))

    def classify_series(self, titles: Iterable[Any]):
        """Classify many titles. Returns a DataFrame with one row per title."""
        import pandas as pd

        records = [self.classify(title) for title in titles]
        return pd.DataFrame(
            {
                "title": list(titles),
                "category": [r.category for r in records],
                "confidence": [r.confidence for r in records],
                "rule": [r.rule for r in records],
                "is_rerun": [r.is_rerun for r in records],
                "legacy_type": [r.legacy_type for r in records],
            }
        )

    def coverage_report(self, titles: Iterable[Any], low_confidence: float = 0.6) -> dict[str, Any]:
        """Summarise classification over a set of titles for taxonomy tuning."""
        titles = list(titles)
        results = [(title, self.classify(title)) for title in titles]
        by_category: dict[str, int] = {}
        by_rule: dict[str, int] = {}
        uncovered: list[str] = []
        low_conf: list[tuple[str, float]] = []
        rerun_count = 0
        for title, result in results:
            by_category[result.category] = by_category.get(result.category, 0) + 1
            by_rule[result.rule] = by_rule.get(result.rule, 0) + 1
            if result.is_rerun:
                rerun_count += 1
            if result.category == _OTHER:
                uncovered.append(str(title))
            elif result.confidence < low_confidence:
                low_conf.append((str(title), result.confidence))
        total = len(titles)
        covered = total - len(uncovered)
        return {
            "total": total,
            "covered": covered,
            "coverage_pct": round(100 * covered / total, 1) if total else 0.0,
            "rerun_count": rerun_count,
            "by_category": dict(sorted(by_category.items(), key=lambda kv: kv[1], reverse=True)),
            "by_rule": by_rule,
            "uncovered": sorted(set(uncovered)),
            "low_confidence": sorted(set(low_conf), key=lambda kv: kv[1]),
        }
