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


# tv-break-dashboard/src/shell/bidi.jsx is a primitive added after this probe
# was written: rules-bidi.js now imports `isolate` from it, and the probe only
# ever copies rules-words.js and rules-bidi.js into its scratch tree, so node
# resolves `../shell/bidi` against a directory that was never given the file.
# The probe script itself is frozen, so the fix cannot live in the copy it
# makes; it has to arrive before that script even starts resolving imports.
# Node's `--import` preloads a module in the same process ahead of the entry
# point and lets that module call `registerHooks`, which is exactly what the
# P4 fix (tests/test_p4_rollup_tristate.py) does for its own harness's own
# entry module. This does the equivalent from outside a file this suite may
# not edit: a resolver hook, installed by a small module written to the same
# scratch tree the payload already lives in, standing in for the one import
# the copied files actually reach for.
#
# The stub mirrors only what a plain-node ESM probe can honestly mirror: no
# React, because nothing here compiles JSX or supplies a React runtime, and no
# invisible isolation marks, because a test about words must not depend on
# control codes the eye cannot see. The real characters have their own guard
# in npm run test:direction.
BIDI_STUB = """
export function isolate(value) {
  return String(value ?? '').trim();
}
export function documentDirection(locale) {
  return locale === 'he' ? 'rtl' : 'ltr';
}
export function Figure() { return null; }
export function Code() { return null; }
export function Name() { return null; }
export function DirectionRoot() { return null; }
export function Prose() { return null; }
"""

RESOLVE_BIDI_HOOK = """
import { registerHooks } from 'node:module';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));
const BIDI = pathToFileURL(join(here, 'bidi-stub.mjs')).href;

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('shell/bidi')) return { url: BIDI, shortCircuit: true };
    return nextResolve(specifier, context);
  },
});
"""


@pytest.fixture(scope="module")
def read(tmp_path_factory, walls, effect_keys) -> dict:
    node = _node()
    work = tmp_path_factory.mktemp("words")
    payload = work / "walls.json"
    payload.write_text(
        json.dumps({**walls, "effect_keys": effect_keys}, ensure_ascii=False), encoding="utf-8",
    )
    (work / "bidi-stub.mjs").write_text(BIDI_STUB, encoding="utf-8")
    hook = work / "resolve-bidi.mjs"
    hook.write_text(RESOLVE_BIDI_HOOK, encoding="utf-8")
    result = subprocess.run(
        [node, "--import", hook.as_uri(), str(PROBE), str(payload)],
        capture_output=True, text=True, check=False, cwd=str(work),
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


def test_the_scheduled_changes_count_agrees_with_its_own_verb(read):
    """The licence page's own count, not templated with a verb that ignores it.

    Measured: the sentence used to read "2 change is recorded" whenever more
    than one licence revision was filed ahead of its effective date, which is
    the normal state right after a revision, not the rare one.
    """
    one = read["scheduled"]["1"]
    assert one["en"] == "1 change is recorded for a future date and is not in force yet."
    assert one["he"] == "שינוי אחד תועד לתאריך עתידי ואינו בתוקף עדיין."
    two = read["scheduled"]["2"]
    assert two["en"] == "2 changes are recorded for a future date and are not in force yet."
    assert two["he"] == "2 שינויים תועדו לתאריך עתידי ואינם בתוקף עדיין."
