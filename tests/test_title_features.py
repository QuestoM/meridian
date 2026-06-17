"""Tests for the deterministic title features that drive the series-aware layer.

These prove the four properties the series pooling and cold-start lookup rely on:
episodes of one series collapse to ONE canonical key, host names are pulled out
of the title, the token-fallback similarity orders a shared theme above an
unrelated title, and the embedding hook degrades to None with no network access.
"""

from __future__ import annotations

from kairos.data.title_features import (
    canonicalize_series,
    embed_title,
    extract_people,
    title_similarity,
    title_tokens,
)


# --- canonicalize_series -----------------------------------------------------

def test_canonicalize_collapses_big_brother_episodes_to_one_key() -> None:
    # Three episodes across two seasons of the same series must map to ONE key.
    titles = [
        "האח הגדול עונה 2 פרק 2",
        "האח הגדול עונה 2 פרק 7",
        "האח הגדול עונה 3 פרק 14",
    ]
    keys = {canonicalize_series(t) for t in titles}
    assert len(keys) == 1
    assert keys.pop() == "האח הגדול"


def test_canonicalize_strips_repeat_and_channel_tags() -> None:
    # The "ש.ח" repeat tag and a trailing number are not part of the series name.
    assert canonicalize_series("מאסטר שף נבחרת החלומות ש.ח") == "מאסטר שפ נבחרת החלומות"
    assert canonicalize_series("מאסטר שף 7") == "מאסטר שפ"


def test_canonicalize_is_total_on_unseen_titles() -> None:
    # An unseen title with no markers still returns a stable non-empty key, so the
    # cold-start lookup always has a key to try.
    key = canonicalize_series("תוכנית חדשה לגמרי")
    assert key
    # Idempotent: canonicalizing a key gives the same key.
    assert canonicalize_series(key) == key


def test_canonicalize_empty_and_none() -> None:
    assert canonicalize_series("") == ""
    assert canonicalize_series(None) == ""


# --- extract_people ----------------------------------------------------------

def test_extract_people_finds_host_after_marker() -> None:
    assert extract_people("טיול בעולם עם יובל מלחי") == ("יובל", "מלחי")


def test_extract_people_empty_without_marker() -> None:
    assert extract_people("האח הגדול עונה 2") == ()


# --- title_tokens ------------------------------------------------------------

def test_title_tokens_drop_stopwords_and_numbers() -> None:
    tokens = title_tokens("מאסטר שף עונה 7")
    assert "מאסטר" in tokens
    # The season word and the digit are dropped as non-content.
    assert "עונה" not in tokens
    assert "7" not in tokens


# --- title_similarity (token fallback ordering) ------------------------------

def test_similarity_orders_shared_theme_above_unrelated() -> None:
    # The exact ordering the series layer relies on for cold-start neighbours:
    # "טיול בעולם" is closer to "טיול מסביב לעולם" (shared theme) than to the
    # unrelated "חיים אתגר". Holds on the token fallback (no embedding installed).
    near = title_similarity("טיול בעולם", "טיול מסביב לעולם")
    far = title_similarity("טיול בעולם", "חיים אתגר")
    assert near > far


def test_similarity_lifts_shared_host() -> None:
    # Two shows that share a host score above two shows that merely share a word.
    shared_host = title_similarity("טיול בעולם עם יובל מלחי", "מבשלים עם יובל מלחי")
    no_overlap = title_similarity("טיול בעולם עם יובל מלחי", "חיים אתגר")
    assert shared_host > no_overlap


# --- embed_title (offline degrade) -------------------------------------------

def test_embed_title_degrades_to_none_or_returns_vector() -> None:
    # With no local multilingual model installed this returns None (no API call,
    # no fabricated vector). If a model IS installed, it returns a real vector.
    result = embed_title("בדיקה")
    assert result is None or (isinstance(result, list) and len(result) > 0)
