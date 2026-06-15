"""AI fallback for programme titles the rule-based classifier cannot place.

The rule-based :class:`~kairos.data.classifier.ProgramClassifier` is honest about
its unknowns: a title with no keyword evidence returns ``category == "Other"``
(rule ``fallback`` or ``empty``, confidence 0.0). That is different from a title
whose genre is known but whose positional pricing class happens to be "Other";
the first is genuinely unclassified, the second is not.

This module routes only the genuinely unclassified titles to an LLM with Google
Search grounding and a structured result, caches each verdict so a title is
resolved once, and exposes a wrapper classifier that folds the cached verdicts
back into classification so the placement engine prices the real genre.

No fabrication: when no LLM is configured (no key or client library), the AI
classifier is absent and unclassified titles stay "Other" with the run recording
that the AI was unavailable. A cached verdict is only trusted when it came from
the AI and names a category in the taxonomy.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Protocol

from kairos.data.classifier import Classification, ProgramClassifier

logger = logging.getLogger(__name__)

_OTHER = "Other"
_DEFAULT_MODEL = "gemini-2.5-flash"


def unclassified_titles(titles: Iterable[Any], classifier: ProgramClassifier) -> list[str]:
    """Return the distinct titles the rule-based classifier could not place.

    Only ``category == "Other"`` (no keyword evidence) counts as unclassified; a
    title with a real genre is classified even when its legacy pricing class is
    "Other". Order is preserved and blanks/``nan`` are skipped.
    """
    seen: set[str] = set()
    out: list[str] = []
    for title in titles:
        text = str(title).strip()
        if not text or text.lower() == "nan" or text in seen:
            continue
        if classifier.classify(title).category == _OTHER:
            seen.add(text)
            out.append(text)
    return out


@dataclass(frozen=True)
class AIClassification:
    """One AI verdict on a previously unclassified title."""

    title: str
    category: str       # a taxonomy category, or "Other" when the AI also abstains
    confidence: float   # 0.0 .. 1.0
    grounded: bool      # Google Search grounding returned at least one source
    rationale: str
    source: str         # "ai" | "unavailable"


class GroundedClassifier(Protocol):
    """A title-to-genre classifier backed by an LLM with search grounding."""

    def classify(self, title: str, *, allowed: list[str]) -> Optional[AIClassification]:
        ...


class GeminiGroundedClassifier:
    """Classify an unknown title into the taxonomy via Gemini.

    Uses Google Search grounding (the model must look the programme up) and asks
    for a single strict JSON object as the result. Grounding and a forced
    response schema cannot always be combined on the same call, so the schema is
    enforced by parsing the JSON the prompt requires, and ``grounded`` records
    whether search actually returned sources. Built only when a key and the
    client library are present; :meth:`from_env` returns ``None`` otherwise.
    """

    def __init__(self, client: Any, model: str = _DEFAULT_MODEL) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls) -> Optional["GeminiGroundedClassifier"]:
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            return None
        try:
            from google import genai  # lazy: optional dependency
        except ImportError:
            logger.info("google-genai not installed; AI classifier unavailable")
            return None
        model = os.environ.get("KAIROS_AI_MODEL", _DEFAULT_MODEL)
        return cls(genai.Client(api_key=key), model)

    def classify(self, title: str, *, allowed: list[str]) -> Optional[AIClassification]:
        from google.genai import types  # lazy

        options = ", ".join(allowed)
        prompt = (
            "You classify Israeli TV programme titles into a fixed taxonomy. "
            "Search the web to identify the programme, then choose the single best "
            f"category from this list: {options}. If none fits, answer \"Other\". "
            "Reply with one JSON object and nothing else, shaped exactly as: "
            '{\"category\": <one of the list or \"Other\">, \"confidence\": <0..1>, '
            '\"rationale\": <short reason>}.\n'
            f"Programme title: {title}"
        )
        config = types.GenerateContentConfig(tools=[types.Tool(google_search=types.GoogleSearch())])
        try:
            response = self._client.models.generate_content(
                model=self._model, contents=prompt, config=config,
            )
        except Exception as error:  # network/quota/parse: do not invent a verdict
            logger.warning("AI classify failed for %r: %s", title, error)
            return AIClassification(title, _OTHER, 0.0, False, f"ai error: {error}", "unavailable")
        return self._parse(title, response, allowed)

    @staticmethod
    def _parse(title: str, response: Any, allowed: list[str]) -> AIClassification:
        text = getattr(response, "text", "") or ""
        grounded = _has_grounding(response)
        try:
            start, end = text.index("{"), text.rindex("}") + 1
            data = json.loads(text[start:end])
        except (ValueError, json.JSONDecodeError):
            return AIClassification(title, _OTHER, 0.0, grounded, "unparseable ai reply", "ai")
        category = str(data.get("category", _OTHER))
        if category not in allowed:
            category = _OTHER
        try:
            confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        return AIClassification(
            title, category, confidence, grounded, str(data.get("rationale", "")), "ai",
        )


def _has_grounding(response: Any) -> bool:
    """True when the response carries Google Search grounding metadata."""
    try:
        candidate = response.candidates[0]
        meta = getattr(candidate, "grounding_metadata", None)
        chunks = getattr(meta, "grounding_chunks", None) if meta else None
        return bool(chunks)
    except (AttributeError, IndexError, TypeError):
        return False


def resolve_unclassified(
    titles: Iterable[Any],
    classifier: ProgramClassifier,
    ai: Optional[GroundedClassifier],
    *,
    cache_path: str | Path | None = None,
) -> dict[str, AIClassification]:
    """Resolve every unclassified title once, using and updating a JSON cache.

    Cached titles are reused; uncached ones go to ``ai`` when present, else are
    recorded as ``unavailable``. The cache is written back when a path is given.
    Returns the full title-to-verdict map for the unclassified titles found.
    """
    pending = unclassified_titles(titles, classifier)
    allowed = list(classifier.categories)
    cache = _read_cache(cache_path) if cache_path else {}
    resolved: dict[str, AIClassification] = {}
    changed = False
    for title in pending:
        if title in cache:
            resolved[title] = cache[title]
            continue
        if ai is None:
            verdict = AIClassification(title, _OTHER, 0.0, False, "no ai configured", "unavailable")
        else:
            verdict = ai.classify(title, allowed=allowed) or AIClassification(
                title, _OTHER, 0.0, False, "ai returned nothing", "unavailable",
            )
        resolved[title] = verdict
        cache[title] = verdict
        changed = True
    if cache_path and changed:
        _write_cache(cache_path, cache)
    return resolved


def load_ai_overrides(cache_path: str | Path | None) -> dict[str, str]:
    """Read the cache and return title -> category for trusted AI verdicts only.

    A verdict is trusted when it came from the AI (not ``unavailable``) and names
    a real category (not ``Other``); these are the titles whose genre the engine
    can now use. Missing or unreadable cache yields an empty map.
    """
    cache = _read_cache(cache_path) if cache_path else {}
    return {
        title: v.category
        for title, v in cache.items()
        if v.source == "ai" and v.category != _OTHER
    }


class CachedClassifier:
    """Wrap a rule-based classifier, folding trusted AI verdicts into its output.

    For a title the base classifier leaves as ``Other``, if a trusted AI genre
    exists for it the wrapper returns that genre (rule ``ai``); otherwise it
    defers to the base result unchanged. This is the seam through which AI
    classifications reach pricing and the placement engine.
    """

    def __init__(self, base: ProgramClassifier, overrides: dict[str, str]) -> None:
        self._base = base
        self._overrides = {str(k).strip(): v for k, v in overrides.items()}

    def __getattr__(self, name: str) -> Any:
        # Delegate any method the wrapper does not override (e.g. classify_series,
        # coverage_report) to the base classifier so this is a drop-in.
        return getattr(self._base, name)

    @property
    def categories(self) -> list[str]:
        return self._base.categories

    def classify(self, raw_title: Any) -> Classification:
        result = self._base.classify(raw_title)
        if result.category != _OTHER:
            return result
        category = self._overrides.get(str(raw_title).strip())
        if not category:
            return result
        legacy = self._base._legacy(category)  # reuse the base legacy mapping
        return Classification(category, 0.9, "ai", result.is_rerun, legacy)


def _read_cache(cache_path: str | Path) -> dict[str, AIClassification]:
    path = Path(cache_path)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("unreadable AI cache at %s; ignoring", path)
        return {}
    out: dict[str, AIClassification] = {}
    for title, body in (payload.get("classifications") or {}).items():
        try:
            out[title] = AIClassification(
                title=title,
                category=str(body["category"]),
                confidence=float(body.get("confidence", 0.0)),
                grounded=bool(body.get("grounded", False)),
                rationale=str(body.get("rationale", "")),
                source=str(body.get("source", "ai")),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _write_cache(cache_path: str | Path, cache: dict[str, AIClassification]) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "method": "ai_grounded_classification",
        "classifications": {title: asdict(v) for title, v in cache.items()},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
