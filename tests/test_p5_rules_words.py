"""P5: no engine key and no untranslated refusal reaches a reader.

Two measured defects, both one line of markup each.

The restrictions list printed ``<code>{row.effect}</code>``, so a pre-authoring
row rendered the store's own key, ``fix_offset``, on the surface whose whole
claim is that a rule reads in plain language, while the condition builder three
centimetres below translated the same value to ``היסט קבוע`` off its own private
copy of the table. Two tables, one value, two readings. There is one table now
and both surfaces read it.

The permission refusal was passed through verbatim. The server authors it in
Hebrew, because that is the language the product's operators work in and the
wall holds exactly one string per rule, so the English licence and channel pages
printed ``עריכת מגבלות הרגולציה שמורה למנהל המערכת`` inside otherwise-English
copy. The English sentence is keyed off the frozen wall's own detail, so the two
cannot drift, and a wall the table does not know still gets the server's words
rather than a guess.

The wall details come from the Python constants rather than from a copy of them,
so a change on either side is a failure here.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROBE = Path(__file__).with_name("test_p5_words_probe.mjs")

# The stored effect keys, from the engine's own frozen vocabulary rather than
# from a list written here, plus one the table cannot know.
UNKNOWN_EFFECT = "brand_new_effect"


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped module cannot be run here")
    if not PROBE.exists():
        pytest.skip("the words probe is missing")
    return found


@pytest.fixture(scope="module")
def walls() -> dict:
    """Every wall detail this workspace can print, from the Python constants."""
    from kairos_api.affiliation_wall import COMPANY_SURFACE_DETAIL, READ_ONLY_ROLE_DETAIL
    from kairos_api.compliance_api_licence import OPERATOR_CHANNEL_ADMIN_ONLY_DETAIL
    from kairos_api.events_access import COMPANY_ONLY_DETAIL, EVENT_PRICING_COMPANY_ONLY_DETAIL
    from kairos_api.guardrail_store import GUARDRAIL_ADMIN_ONLY_DETAIL
    from kairos_api.model_activation import AUDIENCE_MODEL_COMPANY_ONLY_DETAIL

    return {
        "walls": {
            "guardrails": {"detail": GUARDRAIL_ADMIN_ONLY_DETAIL},
            "audienceActivation": {"detail": AUDIENCE_MODEL_COMPANY_ONLY_DETAIL},
            "events": {"detail": COMPANY_ONLY_DETAIL},
            "eventPricing": {"detail": EVENT_PRICING_COMPANY_ONLY_DETAIL},
            "companySurface": {"detail": COMPANY_SURFACE_DETAIL},
            "readOnlyRole": {"detail": READ_ONLY_ROLE_DETAIL},
        },
        "refusals": {
            "guardrails": GUARDRAIL_ADMIN_ONLY_DETAIL,
            "activation": AUDIENCE_MODEL_COMPANY_ONLY_DETAIL,
            "channel": OPERATOR_CHANNEL_ADMIN_ONLY_DETAIL,
            "events": COMPANY_ONLY_DETAIL,
            "event_pricing": EVENT_PRICING_COMPANY_ONLY_DETAIL,
            "company_surface": COMPANY_SURFACE_DETAIL,
            "read_only": READ_ONLY_ROLE_DETAIL,
            "unknown_wall": "כלל שאיש עוד לא כתב לו תרגום",
        },
    }


@pytest.fixture(scope="module")
def effect_keys() -> list[str]:
    from kairos.optimize.constraints_store import _EFFECTS

    return sorted(_EFFECTS) + [UNKNOWN_EFFECT]


@pytest.fixture(scope="module")
def read(tmp_path_factory, walls, effect_keys) -> dict:
    node = _node()
    work = tmp_path_factory.mktemp("words")
    payload = work / "walls.json"
    payload.write_text(
        json.dumps({**walls, "effect_keys": effect_keys}, ensure_ascii=False), encoding="utf-8",
    )
    result = subprocess.run(
        [node, str(PROBE), str(payload)], capture_output=True, text=True, check=False, cwd=str(work),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(result.stdout)


def test_every_effect_the_engine_can_store_reads_as_words(read, effect_keys):
    for key in effect_keys:
        if key == UNKNOWN_EFFECT:
            continue
        for locale in ("he", "en"):
            label = read["effects"][key][locale]
            assert label, f"{key} has no {locale} word"
            assert key not in label, f"{key} still renders its own key in {locale}"
            assert "_" not in label, f"{key} renders an engine-shaped token in {locale}"


def test_the_hebrew_and_the_english_word_are_actually_different(read):
    """A table that returned the key twice would pass the test above."""
    hebrew = {read["effects"][key]["he"] for key in read["effects"]}
    english = {read["effects"][key]["en"] for key in read["effects"]}
    assert not (hebrew & english) - {read["effects"][UNKNOWN_EFFECT]["he"]}, (
        "each effect has its own word in each language"
    )


def test_an_effect_the_table_does_not_know_is_spaced_rather_than_raw(read):
    """An effect added to the engine later still never renders as a raw key."""
    for locale in ("he", "en"):
        label = read["effects"][UNKNOWN_EFFECT][locale]
        assert label == "Brand new effect"
        assert "_" not in label


def test_every_refusal_this_workspace_throws_has_an_english_sentence(read, walls):
    for name, detail in walls["refusals"].items():
        answer = read["refusals"][name]
        assert answer["he"] == detail, "a Hebrew reader gets the server's own words"
        if name == "unknown_wall":
            continue
        assert answer["en"] != detail, f"{name} still prints Hebrew to an English reader"
        assert answer["en"].isascii(), f"{name} has no English sentence"
        assert answer["en"].endswith("."), f"{name} is not a sentence"


def test_a_wall_the_table_does_not_know_falls_back_to_the_servers_words(read):
    """Honest rather than guessed: a new wall prints what the server said."""
    answer = read["refusals"]["unknown_wall"]
    assert answer["en"] == answer["he"] == "כלל שאיש עוד לא כתב לו תרגום"
