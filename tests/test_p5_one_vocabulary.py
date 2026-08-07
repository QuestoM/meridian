"""P5: no display string this destination serves is frozen in one language.

THE CLASS, not one instance of it. One display string authored in ONE language
at the place it is PRODUCED, then printed verbatim to a reader of the other. It
survived four rounds because each round closed the site a critic had pointed at
and a suite named after that one sentence went green again. The sites measured
on the shipped product, all of them the same defect:

1. The licence section printed ``שינוי מגבלות הרגולציה שמור לצוות החברה`` above
   four English fields. The wall holds one string per rule and it is Hebrew,
   because the 403 detail and the reason a control renders before the click are
   one string by contract, and ``can_edit_reason`` carried it verbatim.
2. The mirror of the same defect under the same save button: every
   ``GuardrailError`` was authored in English alone and reaches a Hebrew screen
   through the 400 the route raises, naming the engine's own key while it did it.
3. The rate card labelled every category with the raw config key, so each
   language got the half that is not its own: three Hebrew words down the English
   card's ad-type column, four Latin ones down the Hebrew card's programme
   column, and the zero-multiplier warning put a Hebrew category inside an
   English sentence.
4. The same card took a layer's description half from a table and half from the
   API, so a layer the table did not name printed English prose to a Hebrew
   reader, and the two languages could say different things about one layer.
5. The condition builder read every saved row back in English to a Hebrew
   reader, because ``create_restriction`` wrote the English rendering into the
   ``notes`` column, a free-text field it also offers a person.
6. Every other refusal these routes raise was English-only prose in ``detail``.
7. The two figure panels printed an empty-basis ``reason`` verbatim, and the
   endpoints that produce it author it in English alone.

THE LAW THIS ASSERTS, which is what makes it a class guard rather than a seventh
sentence-shaped test: for every display string this destination serves, either
the two readers get DIFFERENT strings, each in its own reader's script, or they
get the SAME string, which is the producer's own words unaltered. What may never
happen is two different strings with one of them in the wrong script for its
reader. Every message every module here can produce is driven, in both locales.

The fix shape is the one the uploads destination already proved: both halves
come from ONE call that returns them together or returns neither, so a sentence
cannot exist in one language only. ``guardrail_store.say`` and
``constraints_sentence.say`` are that call, and the surface reads the pair off
the payload rather than translating a sentence back.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import kairos_api.constraints_language as constraints_language
import kairos_api.constraints_sentence as constraints_sentence
import kairos_api.guardrail_store as guardrail_store

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
PROBE = Path(__file__).with_name("test_p5_one_vocabulary_probe.mjs")

HEBREW = re.compile(r"[֐-׿]")
# A first-strong isolate is punctuation, not script, and every sentence that
# wraps a foreign run in one carries it in both halves.
ISOLATES = str.maketrans("", "", "⁦⁧⁨⁩")


def _has_hebrew(text: str) -> bool:
    return bool(HEBREW.search(str(text).translate(ISOLATES)))


def assert_reader_gets_own_language(where: str, english: str, hebrew: str, authored: bool = True) -> None:
    """The one law, applied to one pair.

    ``authored`` is a string this destination is responsible for saying, and it
    has to be two different strings, each in its own reader's script. Identical
    halves are NOT a pass for one of those: the shipped rate card returned the
    raw config key to both readers, which is one string and is still the English
    reader being served Hebrew. ``authored=False`` is the honest passthrough, a
    proper noun or a producer's own words nobody here may re-author, and then the
    two readers must get exactly the same string rather than one language each.
    """
    english, hebrew = str(english or ""), str(hebrew or "")
    if not authored:
        assert english == hebrew, f"{where}: a passthrough must be the same words for both readers, got {english!r} and {hebrew!r}"
        return
    assert english, f"{where}: the English half is empty while the Hebrew half is not"
    assert hebrew, f"{where}: the Hebrew half is empty while the English half is not"
    assert english != hebrew, f"{where}: one string is served to both readers, so one of them is reading the other's language: {english!r}"
    assert not _has_hebrew(english), f"{where}: the English reader is served Hebrew: {english!r}"
    assert _has_hebrew(hebrew), f"{where}: the Hebrew reader is served no Hebrew: {hebrew!r}"


# --- the tables, walked whole ------------------------------------------------

TABLES = [
    ("guardrail_store.WORDS", guardrail_store.WORDS, "dict"),
    ("constraints_sentence.REFUSALS", constraints_sentence.REFUSALS, "dict"),
    ("guardrail_store.LIMIT_NAMES", guardrail_store.LIMIT_NAMES, "pair"),
    ("constraints_sentence._FIELD_NAMES", constraints_sentence._FIELD_NAMES, "pair"),
    ("constraints_language._PARAM_NAMES", constraints_language._PARAM_NAMES, "pair"),
]


@pytest.mark.parametrize("name,table,shape", TABLES, ids=[row[0] for row in TABLES])
def test_every_entry_in_every_table_carries_both_halves(name, table, shape):
    """Not one sentence: every entry of every table this destination authors."""
    assert table, f"{name} is empty, so this guard would pass vacuously"
    for key, value in table.items():
        english, hebrew = (value["en"], value["he"]) if shape == "dict" else (value[0], value[1])
        assert english and hebrew, f"{name}[{key!r}] is missing a half"
        assert_reader_gets_own_language(f"{name}[{key!r}]", english, hebrew)


def test_every_code_the_licence_store_can_say_comes_back_as_a_pair():
    """Every code, with the fields it takes, driven through the one call."""
    fields = {
        "bad_date": {"value": "2026-13-01"},
        "not_a_guardrail": {"name": guardrail_store.LIMIT_NAMES["max_breaks_per_hour"]},
        "not_a_number": {"name": guardrail_store.LIMIT_NAMES["max_breaks_per_hour"]},
        "out_of_bounds": {"name": guardrail_store.LIMIT_NAMES["max_ad_minutes_per_hour"], "low": 0.0, "high": 60.0},
    }
    for code in guardrail_store.WORDS:
        english, hebrew = guardrail_store.say(code, **fields.get(code, {}))
        assert english and hebrew, f"{code} rendered a half"
        assert "{" not in english and "{" not in hebrew, f"{code} left a hole in a sentence"
        assert_reader_gets_own_language(f"say({code!r})", english, hebrew)
    assert guardrail_store.say("no_such_code") == ("", ""), "an unknown code must render neither half, never one"


def test_every_refusal_the_constraints_routes_can_raise_comes_back_as_a_pair():
    fields = {
        "bad_iso_date": {"field": constraints_sentence.field_name("expires_on"), "value": "2026-13-01"},
        "will_not_compile": {"problem": ("it touches too many airings", "היא נוגעת ביותר מדי שידורים")},
    }
    for code in constraints_sentence.REFUSALS:
        english, hebrew = constraints_sentence.say(code, **fields.get(code, {}))
        assert english and hebrew, f"{code} rendered a half"
        assert "{" not in english and "{" not in hebrew, f"{code} left a hole in a sentence"
        assert_reader_gets_own_language(f"say({code!r})", english, hebrew)
        raised = constraints_sentence.refuse(code, **fields.get(code, {}))
        assert raised.detail["en"] == english and raised.detail["he"] == hebrew
        assert raised.detail["code"] == code, "the surface has to be able to key off the code, not the prose"
    assert constraints_sentence.say("no_such_code") == ("", "")


def test_every_value_the_licence_can_refuse_is_refused_in_both_languages():
    """The mirror defect: a rejected save, authored in English alone."""
    cases = [
        ({}, "no_limits"),
        ({"not_a_limit": 1}, "not_a_guardrail"),
        ({"max_breaks_per_hour": "x"}, "not_a_number"),
        ({"max_breaks_per_hour": 99}, "out_of_bounds"),
    ]
    for values, code in cases:
        with pytest.raises(guardrail_store.GuardrailError) as caught:
            guardrail_store._clean_values(values)
        error = caught.value
        assert error.code == code
        assert str(error) == error.english, "str() is the English half, which is what the frozen route forwards"
        assert_reader_gets_own_language(f"GuardrailError({code})", error.english, error.hebrew)
        assert not _has_hebrew(error.english) and _has_hebrew(error.hebrew)
        for key in guardrail_store.GUARDRAIL_KEYS:
            assert key not in error.english and key not in error.hebrew, "a refusal must not name the engine's own key"
    with pytest.raises(guardrail_store.GuardrailError) as caught:
        guardrail_store._as_day("2026-13-01")
    assert_reader_gets_own_language("GuardrailError(bad_date)", caught.value.english, caught.value.hebrew)


def test_every_reason_the_restriction_compiler_can_raise_carries_both_halves():
    from kairos_api.constraints_language import RestrictionError, compile_restriction

    with pytest.raises(RestrictionError) as caught:
        compile_restriction("not_a_kind", {}, None, [])
    assert_reader_gets_own_language("RestrictionError(kind)", str(caught.value), caught.value.hebrew)
    with pytest.raises(RestrictionError) as caught:
        constraints_language._int_param({"protected_minutes": "x"}, "protected_minutes", low=0, high=9)
    assert_reader_gets_own_language("RestrictionError(minutes)", str(caught.value), caught.value.hebrew)
    for half in (str(caught.value), caught.value.hebrew):
        assert "protected_minutes" not in half, "a refusal must not name the store's own parameter key"


def test_the_licence_wall_reports_the_same_gate_in_both_languages():
    """``refusal`` and ``reason`` walk the same gates, so the pair and the 403
    body cannot drift apart. With no request both read open, which is what an
    in-process call and a deployment without login get."""
    wall = guardrail_store.GUARDRAIL_WALL
    assert wall.refusal(None) == ("", ""), "an open gate owes no refusal in either language"
    assert wall.reason(None) is None
    for code, expected in (("company_only", wall.detail), ("admin_only", wall.role_detail)):
        english, hebrew = guardrail_store.say(code)
        assert hebrew == expected, "the wall's own detail is the Hebrew half of its own entry, not a restatement"
        assert_reader_gets_own_language(f"wall({code})", english, hebrew)


def test_a_saved_restriction_writes_no_rendered_sentence_into_the_free_text_column():
    """``notes`` is the field the builder offers a person. The English rendering
    used to be written into it at save time, which is why every saved row read
    back in English to a Hebrew reader."""
    source = (ROOT / "kairos_api" / "constraints_restrictions.py").read_text(encoding="utf-8")
    assert '"notes": words["sentence_en"]' not in source
    assert '"notes": words["sentence_he"]' not in source, "the mirror of it would be the same defect"
    assert '"notes": ""' in source


# --- the same law on the surface, driven through the shipped modules ---------

# The layers the surface authors a description for. Every other layer, named or
# not, takes the producer's own words and gives them to both readers unaltered.
LAYER_TEXT_WITH_A_DESCRIPTION = ("program", "day", "show", "position", "ad_type", "events")


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped module cannot be run here")
    if not (APP / "node_modules" / "rolldown").is_dir():
        pytest.skip("the bundler the product builds with is not installed")
    if not PROBE.exists():
        pytest.skip("the vocabulary probe is missing")
    return found


@pytest.fixture(scope="module")
def surface() -> dict:
    node = _node()
    proc = subprocess.run([node, str(PROBE)], capture_output=True, text=True, check=False, cwd=str(APP))
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout)


def test_every_rate_card_label_reaches_its_reader_in_its_own_language(surface):
    """Every key the shipped rate card can carry, on every layer, both locales."""
    labels = surface["labels"]
    assert len(labels) >= 22, "the probe stopped covering the shipped vocabulary"
    for where, pair in labels.items():
        assert_reader_gets_own_language(f"keyLabel({where})", pair["en"], pair["he"])
    assert not _has_hebrew(labels["ad_type::פרומו"]["en"]), "the English card printed the Hebrew config key"
    assert _has_hebrew(labels["program::News"]["he"]), "the Hebrew card printed the Latin config key"


def test_a_programme_title_is_a_proper_noun_and_passes_through_untranslated(surface):
    pair = surface["passthrough"]
    assert_reader_gets_own_language("keyLabel(show)", pair["en"], pair["he"], authored=False)
    assert pair["en"] == "האח הגדול"


def test_every_layer_says_the_same_thing_in_both_languages_or_the_wire_words(surface):
    for name, layer in surface["layers"].items():
        assert layer["has_both_halves"], f"{name} has one description half authored and takes the other off the wire"
        # A layer no table names is a passthrough on both its title and its
        # description: the producer's own words, the same for both readers.
        named = name in LAYER_TEXT_WITH_A_DESCRIPTION
        assert_reader_gets_own_language(
            f"layerLabel({name})", layer["title"]["en"], layer["title"]["he"], authored=name != "__unknown__",
        )
        assert_reader_gets_own_language(
            f"layerDescription({name})", layer["description"]["en"], layer["description"]["he"], authored=named,
        )
    unknown = surface["layers"]["__unknown__"]["description"]
    assert_reader_gets_own_language("layerDescription(unknown)", unknown["en"], unknown["he"], authored=False)
    assert unknown["en"] == "FROM THE WIRE", "a layer nobody named gets the producer's own words, both readers"


def test_the_zero_multiplier_warning_names_its_categories_in_the_readers_language(surface):
    assert_reader_gets_own_language("categoryList", surface["warning"]["en"], surface["warning"]["he"])
    assert not _has_hebrew(surface["warning"]["en"]), "a Hebrew category inside an English sentence"


def test_the_licence_refusal_reaches_its_reader_in_its_own_language(surface):
    refusals = surface["refusals"]
    assert_reader_gets_own_language("refusalWords(pair)", refusals["pair"]["en"], refusals["pair"]["he"])
    assert not _has_hebrew(refusals["pair"]["en"]), "the measured defect: a Hebrew sentence above four English fields"
    assert refusals["pair"]["he"] == "שינוי מגבלות הרגולציה שמור לצוות החברה", "the Hebrew reader still gets the wall's own words"
    for name, pair in refusals["every_wall"].items():
        assert_reader_gets_own_language(f"wall {name}", pair["en"], pair["he"])
    # An older payload with only the Hebrew still resolves, off the table keyed
    # on the wall's own detail, so an endpoint nobody has upgraded is not a leak.
    assert not _has_hebrew(refusals["hebrew_only_payload"]["en"])
    assert not _has_hebrew(refusals["session_gate"]["en"])


def test_a_rejected_save_reaches_its_reader_in_its_own_language(surface):
    details = surface["details"]
    assert_reader_gets_own_language("detailWords(pair)", details["pair"]["en"], details["pair"]["he"])
    assert not _has_hebrew(details["pair"]["en"]) and _has_hebrew(details["pair"]["he"])
    assert_reader_gets_own_language(
        "detailWords(single)", details["single_string"]["en"], details["single_string"]["he"], authored=False,
    )


def test_the_surface_refuses_an_impossible_limit_in_the_readers_own_language(surface):
    limits = surface["limits"]
    for case in ("refused", "not_a_number"):
        assert_reader_gets_own_language(f"limitBoundsRefusal({case})", limits[case]["en"], limits[case]["he"])
        assert limits[case]["en"] and limits[case]["he"]
    assert limits["accepted"]["en"] == limits["accepted"]["he"] == "", "a value the licence can hold is refused in neither"


def test_every_empty_basis_reason_reaches_a_hebrew_reader_in_hebrew(surface):
    basis = surface["basis"]
    assert len(basis) >= 10, "the probe stopped covering what the two producers can emit"
    for reason, pair in basis.items():
        assert pair["en"] == reason, "the producer's own English is never re-authored"
        assert _has_hebrew(pair["he"]), f"a Hebrew reader is served English prose: {reason!r}"
        assert_reader_gets_own_language(f"basisReason({reason!r})", pair["en"], pair["he"])
