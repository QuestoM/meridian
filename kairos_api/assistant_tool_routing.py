"""Small, fail-safe routes for questions with one unambiguous read tool.

The assistant normally chooses among the full read-tool surface. That remains
the fallback. A named-advertiser airing question is different: the broad query
has exactly one complete read, while the neighbouring ranking and pod tools are
both narrower and can only reconstruct a partial answer. When the question
contains an advertiser already present in the observed-name store and language
about airings, this module prefers that one read on the first model turn.

It never guesses an entity. Matching uses the canonical advertiser identity
fold and an observed name or alias; an unknown or ambiguous free-text name falls
back to ordinary model tool selection.
"""

from __future__ import annotations

import re

from kairos.optimize.advertiser_rules_identity import (
    load_advertiser_names,
    normalize_name,
)

ADVERTISER_AIRINGS_TOOL = "get_advertiser_airings"

_AIRING_INTENT = re.compile(
    r"(?:"
    r"פרסמ|שידר|שודר|על(?:ה|ו|תה)?\s+לאוויר|ספוט|תשדיר|קמפיי?ן|קריאייטיב|"
    r"עד\s+היום|כמה\s+פעמים|באילו\s+ברייקים|מתי|איפה|"
    r"advertis|airing|aired|spot|campaign|creative|commercial|broadcast|"
    r"how\s+often|to\s+date|when|where"
    r")",
    flags=re.IGNORECASE,
)


def _observed_tokens() -> list[str]:
    """Canonical observed names and aliases, longest first."""
    try:
        records = load_advertiser_names().values()
    except OSError:
        return []
    tokens: set[str] = set()
    for record in records:
        for raw in (record.name, record.display_name, *record.aliases):
            token = normalize_name(raw)
            if len(token) >= 3:
                tokens.add(token)
    return sorted(tokens, key=lambda token: (-len(token), token))


def named_advertiser(question: str) -> str | None:
    """The longest observed advertiser token literally present in a question."""
    folded = normalize_name(question)
    if not folded:
        return None
    return next((token for token in _observed_tokens() if token in folded), None)


def preferred_read_tool(question: str) -> str | None:
    """Prefer the complete advertiser-airings read, or leave routing untouched."""
    if not _AIRING_INTENT.search(str(question or "")):
        return None
    return ADVERTISER_AIRINGS_TOOL if named_advertiser(question) else None
