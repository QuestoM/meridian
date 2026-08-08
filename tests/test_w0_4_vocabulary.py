"""The bilingual vocabulary and the session module, checked against the source.

These two files are the frontend half of this piece and they freeze at the end
of wave zero, so what is pinned here is their contract: one word per concept
per language, the retired words absent, the value sets measured against the
files that actually produce them, and every string that also exists in Python
equal to the Python one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from kairos_api import auth_store, events_access, guardrail_store, model_activation
from kairos_api.affiliation_wall import COMPANY_SURFACE_DETAIL, READ_ONLY_ROLE_DETAIL

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
VOCABULARY = SRC / "vocabulary.js"
SESSION = SRC / "session.js"

# The four words the rebuild retires, in both languages. The critic's grep runs
# over the whole tree; this test runs over the two files this piece owns, which
# are the two that must never carry one.
RETIRED_WORDS = ("recompute", "rebuild", "חישוב מחדש", "בנייה מחדש")

# The terms frozen in the brief. Present as a word somewhere in the vocabulary.
FROZEN_TERMS = (
    "ברייק",
    "ברייקים",
    "נעיצה",
    "ברייקי זהב",
    "רצועת שידור",
    "הכנסה צפויה",
    "עלות שימור",
    "מפעיל",
)


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _block(text: str, name: str) -> str:
    """The source of one exported object literal, by its name."""
    start = text.index(f"export const {name} = ")
    end = text.index("\n};", start)
    return text[start:end]


def _keys(block: str) -> list[str]:
    return re.findall(r"^  '?([A-Za-z0-9_. ]+)'?: \{", block, flags=re.MULTILINE)


def _quoted(line: str) -> str:
    """The value part of a ``key: 'value',`` line, quotes included."""
    return line.split(":", 1)[1].strip().rstrip(",").strip()


# ---------------------------------------------------------------------------
# The file's own laws
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", [VOCABULARY, SESSION])
def test_the_file_obeys_the_source_laws(path):
    """Size, punctuation and one display string per source line.

    The punctuation rules are checked on prose (comments and quoted strings),
    not on operators: a JavaScript negation is not an exclamation mark.
    """
    text = _text(path)
    lines = text.splitlines()
    assert len(lines) < 450, f"{path.name} is over the file-size cap"
    assert chr(0x2014) not in text, "no em dashes"
    for index, line in enumerate(lines, start=1):
        stripped = line.strip()
        prose = [stripped[2:]] if stripped.startswith("//") else []
        prose += re.findall(r"'([^']*)'", line) + re.findall(r'"([^"]*)"', line)
        for value in prose:
            assert chr(33) not in value, f"{path.name}:{index} carries an exclamation mark"
        if stripped.startswith(("en:", "he:", "detail:", "label:")):
            display = [value for value in prose if any(ch.isalpha() for ch in value)]
            assert len(display) <= 1, f"{path.name}:{index} carries more than one display string"


@pytest.mark.parametrize("path", [VOCABULARY, SESSION])
def test_no_retired_word_survives_in_the_owned_files(path):
    text = _text(path).lower()
    for word in RETIRED_WORDS:
        assert word.lower() not in text, f"{word} is retired and may not appear in {path.name}"


def test_the_vocabulary_holds_the_frozen_terms_and_not_the_forbidden_ones():
    text = _text(VOCABULARY)
    for term in FROZEN_TERMS:
        assert term in text, f"the frozen term {term} is missing from the vocabulary"
    assert "משתמש" not in text, "the product says מפעיל, never משתמש"
    # The trade says ברייק זהב. It does NOT say ברייק זהוב, which is the
    # adjective and reads as a description of the colour rather than the name of
    # a thing the trade buys and sells. The owner corrected this on 2026-08-09.
    # Every string in the product already had it right; the wrong form only ever
    # appeared in writing ABOUT the product, which is how a term drifts.
    assert "זהוב" not in text, "the trade says ברייק זהב, never ברייק זהוב"


def test_the_wrong_word_for_a_gold_break_is_nowhere_in_the_product():
    """Not only the vocabulary file: the whole tree, because a term drifts by use.

    זהב is the noun the trade uses, and it is already in FROZEN_TERMS. This is
    the other half of that: the adjective form must not appear anywhere, so a
    string written far from the vocabulary cannot quietly introduce it.
    """
    carriers = [
        path.relative_to(SRC).as_posix()
        for path in sorted(SRC.rglob("*"))
        if path.suffix in {".js", ".jsx", ".css"} and "זהוב" in path.read_text(encoding="utf-8")
    ]
    assert carriers == [], f"{carriers} say ברייק זהוב, and the trade says ברייק זהב"


def test_every_concept_has_exactly_one_word_per_language():
    block = _block(_text(VOCABULARY), "WORDS")
    keys = _keys(block)
    assert len(keys) == len(set(keys)), "a key is declared twice"
    assert len(keys) > 50
    entries = re.findall(r"^  '([^']+)': \{\n(.*?)\n  \},$", block, flags=re.MULTILINE | re.DOTALL)
    assert len(entries) == len(keys), "every entry is one key with one body"
    for key, body in entries:
        lines = [line.strip() for line in body.splitlines()]
        english = [line for line in lines if line.startswith("en:")]
        hebrew = [line for line in lines if line.startswith("he:")]
        assert len(english) == 1 and len(hebrew) == 1, f"{key} does not carry exactly one word per language"
        assert len(_quoted(english[0])) > 2, f"{key} has an empty English word"
        assert len(_quoted(hebrew[0])) > 2, f"{key} has an empty Hebrew word"
    # The concepts that wore several names each now wear one.
    assert "'concept.revenue_balance'" in block
    assert "'concept.programme_genre'" in block
    assert "'concept.pricing_class'" in block


# ---------------------------------------------------------------------------
# The two value sets that used to share a name
# ---------------------------------------------------------------------------

def test_the_genre_set_is_the_plan_files_own_values():
    plan = ROOT / "output" / "weekly_break_schedule.csv"
    if not plan.is_file():
        pytest.skip("no saved weekly plan on disk to measure the genre set against")
    import pandas as pd

    measured = set(
        pd.read_csv(plan, encoding="utf-8-sig")["program_type"].dropna().astype(str).unique()
    )
    declared = set(_keys(_block(_text(VOCABULARY), "PROGRAMME_GENRES")))
    assert declared == measured, "the genre list and the plan file disagree"


def test_the_pricing_class_set_is_the_rate_cards_own_values():
    config = yaml.safe_load((ROOT / "config" / "optimization_weights.yaml").read_text(encoding="utf-8"))
    measured = set(config["premiums"]["program_type"])
    declared = set(_keys(_block(_text(VOCABULARY), "PRICING_CLASSES")))
    assert declared == measured
    # The two sets share only two members, which is why they carry two names.
    genres = set(_keys(_block(_text(VOCABULARY), "PROGRAMME_GENRES")))
    assert declared & genres == {"News", "Other"}


# ---------------------------------------------------------------------------
# The session module against the server it describes
# ---------------------------------------------------------------------------

def test_the_job_list_matches_the_account_store():
    text = _text(SESSION)
    ids = re.findall(r"^    id: '([a-z_]+)',$", text, flags=re.MULTILINE)
    assert ids == list(auth_store.JOBS), "the frontend job list and the account store disagree"
    assert len(ids) == 13
    assert "UNSET_JOB = 'unset'" in text
    assert auth_store.UNSET_JOB == "unset"


def test_every_job_opens_a_door_that_exists():
    text = _text(SESSION)
    doors = set(_keys(_block(text, "DOORS")))
    used = set(re.findall(r"^    door: '([a-z_.]+)',$", text, flags=re.MULTILINE))
    assert used <= doors, "a job names a door the map does not hold"
    assert len(doors) == 13, "thirteen roles, thirteen doors"
    assert len(used) == 13, "no two roles share a first screen"


def test_the_refusal_strings_are_the_ones_the_server_sends():
    details = set(re.findall(r"^    detail: '([^']+)',$", _text(SESSION), flags=re.MULTILINE))
    expected = {
        events_access.COMPANY_ONLY_DETAIL,
        events_access.EVENT_PRICING_COMPANY_ONLY_DETAIL,
        model_activation.AUDIENCE_MODEL_COMPANY_ONLY_DETAIL,
        guardrail_store.GUARDRAIL_ADMIN_ONLY_DETAIL,
        COMPANY_SURFACE_DETAIL,
        READ_ONLY_ROLE_DETAIL,
    }
    assert details == expected, "a refusal a person reads must be the one the server sends"
