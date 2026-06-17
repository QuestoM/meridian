"""Deterministic title features for the series-aware retention layer.

The retention model keys its coefficients on coarse cells (genre x position x
length). The programme TITLE carries extra signal the cell throws away: episodes
of one series ("האח הגדול עונה 2 פרק 2" vs "...פרק 7" vs "עונה 3 פרק 14") share a
true effect and should pool together, and shows that share a host or theme are
closer to each other than to an unrelated programme. This module turns a raw
title into stable, deterministic features that work for ANY title, including ones
never seen in training (the cold-start case the inference side must handle):

  * :func:`canonicalize_series` strips Hebrew season/episode/part markers,
    bracketed tags, repeat markers and trailing digits, returning a stable series
    key so every episode of a series collapses to ONE key.
  * :func:`extract_people` pulls host/people names after "עם " and similar, so two
    shows sharing a host can be recognised as related.
  * :func:`title_tokens` returns normalized content tokens (stopwords and numbers
    dropped) for a lightweight lexical similarity fallback.
  * :func:`embed_title` is an OFFLINE/local multilingual embedding hook. It uses a
    locally installed sentence-embedding model if one is present and returns None
    otherwise. It NEVER calls a remote API and never fabricates a vector.
  * :func:`title_similarity` scores two titles with the embedding when available,
    else token Jaccard, so related themes score above unrelated ones.

Pure standard library plus an optional local embedding import, so it imports and
unit-tests without Meridian, pandas or any network access.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from typing import Optional

# Hebrew letters that have a distinct final (sofit) form. Normalizing finals to
# their medial form keeps a token stable whether it appears mid-title or at the
# end (where Hebrew uses the final glyph), so the series key does not split on it.
_FINAL_TO_MEDIAL = {
    "ך": "כ",  # final kaf  -> kaf
    "ם": "מ",  # final mem  -> mem
    "ן": "נ",  # final nun  -> nun
    "ף": "פ",  # final pe   -> pe
    "ץ": "צ",  # final tsadi-> tsadi
}


def _fold_finals(text: str) -> str:
    """Map Hebrew final (sofit) glyphs to their medial form."""
    return "".join(_FINAL_TO_MEDIAL.get(ch, ch) for ch in text)

# Hebrew numerals 1..30ish, so "עונה ב" / "פרק ג" read like "season 2" / "ep 3".
_HEBREW_ORDINAL_LETTERS = "אבגדהוזחטיכלמנסעפצקרשת"

# Markers that introduce a season / episode / part number and everything after.
# Each pattern removes the marker and any following Hebrew-ordinal or digit run.
_SEASON_WORD = "עונה"
_EPISODE_WORD = "פרק"
_PART_WORD = "חלק"
_NUMBER_TAIL = rf"(?:\s*(?:\d+|[{_HEBREW_ORDINAL_LETTERS}]\b|מס['׳]?\s*\d+))?"
_SEASON_EPISODE_PATTERNS = [
    re.compile(rf"\b{_SEASON_WORD}{_NUMBER_TAIL}"),
    re.compile(rf"\b{_EPISODE_WORD}{_NUMBER_TAIL}"),
    re.compile(rf"\b{_PART_WORD}{_NUMBER_TAIL}"),
]

# Repeat / recording markers and short channel-suffix tags ("ש.ח", "ע.14").
_REPEAT_MARKERS = [
    re.compile(r"\(\s*הקלטה\s*\)"),
    re.compile(r"\bהקלטה\b"),
    re.compile(r"\bשידור\s+חוזר\b"),
    re.compile(r"\bש\.?ח\b"),
    re.compile(r"\bע\.?\s*\d+\b"),
]

# Anything inside brackets or parentheses is a tag, not part of the series name.
_BRACKETED = re.compile(r"[\(\[\{].*?[\)\]\}]")

# Tokens introducing a person / host. We keep the words AFTER the marker. These
# are matched against finals-folded, lower-cased tokens, so they are folded too.
_PEOPLE_MARKERS = tuple(
    _fold_finals(w) for w in ("עם", "בהגשת", "בהנחיית", "מגיש", "מגישה")
)

# Lexical stopwords dropped from content tokens (frequent Hebrew connectives and
# generic programme words that carry no series signal). Folded to match tokens.
_STOPWORDS = {
    _fold_finals(w)
    for w in (
        "עם", "של", "על", "את", "אל", "מן", "כל", "גם", "או", "אך", "כי", "זה",
        "הוא", "היא", "אני", "אנחנו", "הם", "הן", "ה", "ו", "ב", "ל", "מ", "ש",
        "עונה", "פרק", "חלק", "הקלטה", "מהדורה", "מבזק", "מיוחד", "תוכנית",
    )
}

# Folded forms of the season/episode/part words, for the people-extraction break.
_SEASON_WORD_F = _fold_finals(_SEASON_WORD)
_EPISODE_WORD_F = _fold_finals(_EPISODE_WORD)
_PART_WORD_F = _fold_finals(_PART_WORD)

_PUNCT = re.compile(r"[\"'׳״.,;:!?/\\|()\[\]{}<>~`@#$%^&*=+–—_-]")
_WHITESPACE = re.compile(r"\s+")


def _normalize_text(title: object) -> str:
    """NFC-normalize, map Hebrew finals to medial, lower-case, collapse spaces."""
    if title is None:
        return ""
    text = unicodedata.normalize("NFC", str(title))
    text = _fold_finals(text)
    text = text.lower()
    text = _WHITESPACE.sub(" ", text).strip()
    return text


@lru_cache(maxsize=4096)
def canonicalize_series(title: object) -> str:
    """Collapse a programme title to a stable series key.

    Strips Hebrew season/episode/part markers and their numbers, bracketed and
    parenthetical tags, repeat / recording markers, channel-suffix tags and any
    trailing standalone digits, then normalizes whitespace and Hebrew finals.
    Deterministic and total: it returns a key for ANY title, including unseen
    ones, so episodes of one series map to a single key while unrelated titles
    keep distinct keys. Returns the normalized whole title when nothing strips
    (the honest fallback), never an empty string for a non-empty input.
    """
    text = _normalize_text(title)
    if not text:
        return ""

    stripped = _BRACKETED.sub(" ", text)
    for pattern in _REPEAT_MARKERS:
        stripped = pattern.sub(" ", stripped)
    for pattern in _SEASON_EPISODE_PATTERNS:
        stripped = pattern.sub(" ", stripped)
    # Trailing standalone numbers ("מאסטר שף 7") and leftover punctuation.
    stripped = re.sub(r"\b\d+\b", " ", stripped)
    stripped = _PUNCT.sub(" ", stripped)
    stripped = _WHITESPACE.sub(" ", stripped).strip()

    if not stripped:
        # Everything stripped away (a title that was only markers); fall back to
        # the punctuation-cleaned full title so the key is never empty.
        fallback = _PUNCT.sub(" ", text)
        return _WHITESPACE.sub(" ", fallback).strip()
    return stripped


@lru_cache(maxsize=4096)
def extract_people(title: object) -> tuple[str, ...]:
    """Pull host / people names that follow a person marker ("עם ...").

    Returns the normalized run of words after the first person marker, up to the
    next clause break, as an ordered de-duplicated tuple. Connective stopwords
    are dropped so "עם יובל מלחי" yields ("יובל", "מלחי"). Returns an empty tuple
    when no marker is present.
    """
    text = _normalize_text(title)
    if not text:
        return ()
    text = _BRACKETED.sub(" ", text)
    tokens = [_PUNCT.sub("", tok) for tok in text.split()]
    tokens = [tok for tok in tokens if tok]

    people: list[str] = []
    capturing = False
    for tok in tokens:
        if tok in _PEOPLE_MARKERS:
            capturing = True
            continue
        if not capturing:
            continue
        # Stop the run at a season/episode marker or a generic stopword break.
        if tok in (_SEASON_WORD_F, _EPISODE_WORD_F, _PART_WORD_F):
            break
        if tok in _STOPWORDS:
            # A connective ("ו") inside a name list is skipped, not a hard stop.
            continue
        if any(ch.isdigit() for ch in tok):
            break
        people.append(tok)

    seen: dict[str, None] = {}
    for name in people:
        seen.setdefault(name, None)
    return tuple(seen.keys())


@lru_cache(maxsize=4096)
def title_tokens(title: object) -> tuple[str, ...]:
    """Normalized content tokens for lexical similarity (stopwords/numbers dropped).

    Lower-cased, finals-normalized, punctuation-stripped tokens with stopwords and
    any token containing a digit removed, de-duplicated in first-seen order. This
    is the lightweight bag used by :func:`title_similarity` when no embedding is
    available.
    """
    text = _normalize_text(title)
    if not text:
        return ()
    text = _BRACKETED.sub(" ", text)
    raw = [_PUNCT.sub("", tok) for tok in text.split()]
    out: dict[str, None] = {}
    for tok in raw:
        if not tok or tok in _STOPWORDS:
            continue
        if any(ch.isdigit() for ch in tok):
            continue
        out.setdefault(tok, None)
    return tuple(out.keys())


# Lazily-resolved local embedding model. None means "not yet probed"; False means
# "probed, none available". A real model is cached after the first successful
# load. No remote call is ever made, so a missing model degrades silently.
_EMBED_MODEL: object = None


def _load_local_embedder() -> object:
    """Try to load a locally-installed multilingual sentence embedder, else False.

    Probes sentence-transformers only. If the package or a local model is not
    present, returns False so the caller degrades to the token fallback. Never
    downloads or calls a network API: a model is used only if it is already on
    disk, so an offline desktop simply gets None from :func:`embed_title`.
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:  # pragma: no cover - exercised only where a local model is installed
        import os

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer

        _EMBED_MODEL = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
    except Exception:  # noqa: BLE001 - any failure means "no local embedder"
        _EMBED_MODEL = False
    return _EMBED_MODEL


@lru_cache(maxsize=4096)
def embed_title(title: object) -> Optional[list[float]]:
    """Return a local multilingual embedding of the title, or None.

    Uses a locally-installed sentence-embedding model when one is available, and
    returns None (degrading gracefully, no API call, no fabricated vector) when
    none is installed. Cached by title. Callers must treat None as "no embedding"
    and fall back to the lexical path.
    """
    text = _normalize_text(title)
    if not text:
        return None
    model = _load_local_embedder()
    if not model:
        return None
    try:  # pragma: no cover - exercised only where a local model is installed
        vector = model.encode(text, normalize_embeddings=True)
        return [float(x) for x in vector]
    except Exception:  # noqa: BLE001 - a runtime encode failure degrades to None
        return None


def _jaccard(a: tuple[str, ...], b: tuple[str, ...]) -> float:
    """Jaccard overlap of two token tuples (0 when both empty)."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def _cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity of two equal-length vectors (0 on degenerate input)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def title_similarity(a: object, b: object) -> float:
    """Similarity in [0, 1] between two titles, embedding-first then lexical.

    Uses the local embedding cosine when both titles embed; otherwise scores the
    Jaccard overlap of their content tokens. People shared between the two titles
    lift the lexical score so two shows with the same host score above two shows
    that merely share a generic word. The ordering this guarantees (a shared theme
    beats an unrelated title) is what the series layer relies on for cold-start
    neighbours.
    """
    vec_a = embed_title(a)
    vec_b = embed_title(b)
    if vec_a is not None and vec_b is not None:
        return max(0.0, min(1.0, _cosine(vec_a, vec_b)))

    token_score = _jaccard(title_tokens(a), title_tokens(b))
    people_score = _jaccard(extract_people(a), extract_people(b))
    # A shared host is strong evidence; blend it in but keep the score in [0, 1].
    return max(0.0, min(1.0, token_score + (1.0 - token_score) * people_score))
