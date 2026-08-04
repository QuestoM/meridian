"""P5: the rate card reads the operator's own week, and a count of one is not plural.

Three defects a blind critic measured on the running product, each closed here
against the shipped source rather than against a copy of it.

The first is the one that mattered. ``GET /api/pricing`` keys the day-of-week
premium by ISO weekday, where 1 is Monday and 7 is Sunday, and the card rendered
those keys in the order a plain JavaScript object hands them back, which for
integer-like keys is ascending numeric order. So the yield owner's own door read
the Israeli week Monday first and put Sunday last, in both languages, while the
same workspace already got it right twice over in the predicate builder and the
calendar. The store is ISO and stays ISO; only the reading order moves.

The same page carries a second weekday control, the price-slot tester's, and it
held a literal week of its own. The case below fixes that reader in place here;
what the control actually renders is measured in ``test_p5_tester_week.py``.

The second is that one restriction wrote "1 כללים" and "1 rules", on a list where
a restriction binding a single airing is the ordinary case.

The third is a law rather than a defect: a display string is one source line, so
this file also sweeps every module this piece owns for a string hard-wrapped
across two of them.

The first two are measured through the shipped modules themselves, bundled with
the bundler the product builds with and called in node, and the day-order test
is run a second time against a mutant of the shipped file with the ordering
removed, so a pass here can never be vacuous.
"""

from __future__ import annotations

import ast
import json
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
RULES = APP / "src" / "rules"
LIB = RULES / "pricing-layers-lib.js"
MANAGER = RULES / "PricingManager.jsx"
TESTER = RULES / "PricingSlotTester.jsx"

# The Israeli week, Sunday first, as the reader sees it. ISO keys, because that
# is what the store holds and what a save has to send back.
SUNDAY_FIRST_KEYS = ["7", "1", "2", "3", "4", "5", "6"]
SUNDAY_FIRST_HE = ["ראשון", "שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת"]
SUNDAY_FIRST_EN = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]

# The guard the mutant removes. Asserted present before it is cut, so the mutant
# can never be a silent no-op.
ORDER_GUARD = "if (!layer || layer.name !== 'day') return entries;"

# Every module this piece owns, including the helpers declared under the
# `<parent stem>_<role>.py` rule.
OWNED_MODULES = (
    "compliance_api.py",
    "compliance_api_licence.py",
    "yield_api.py",
    "constraints.py",
    "constraints_airings.py",
    "constraints_restrictions.py",
    "constraints_cost.py",
    "constraints_language.py",
    "constraints_sentence.py",
    "_constraint_options.py",
    "pricing_api.py",
    "pricing_api_effect.py",
    "events_api.py",
    "events_holidays.py",
    "model_activation.py",
    "guardrail_store.py",
)

ENTRY = """
export * from '{lib}';
export * as rules from '{rules}';
"""

# One node run: bundle the shipped modules where they live, so their own imports
# resolve exactly as they do in the browser build, then ask them for the order
# and the words a person reads.
PROBE = """
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';

const [entry, outDir, payloadFile, outFile, libSource] = process.argv.slice(2);
const require_ = createRequire('APP_PACKAGE');
const MAP = { rolldown: pathToFileURL(require_.resolve('rolldown')).href };
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) return { url: MAP[specifier], shortCircuit: true };
    return nextResolve(specifier, context);
  },
});

const { build } = await import('rolldown');
await build({
  input: entry,
  output: { dir: outDir, format: 'esm', entryFileNames: 'surface.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  logLevel: 'silent',
  plugins: [{
    name: 'pricing-layers-under-test',
    load(id) {
      return id === 'LIB_PATH' ? fs.readFileSync(libSource, 'utf8') : null;
    },
  }],
});

// node has no Vite import.meta.env. Only that accessor is rewritten, only in
// the emitted bundle, and never in the shipped source.
const built = `${outDir}/surface.mjs`;
fs.writeFileSync(built, fs.readFileSync(built, 'utf8').replaceAll('import.meta.env', '({})'), 'utf8');
const surface = await import(pathToFileURL(built).href);
const payload = JSON.parse(fs.readFileSync(payloadFile, 'utf8'));

const layers = {};
for (const layer of payload.layers || []) {
  const entries = surface.layerEntries(layer);
  layers[layer.name] = {
    keys: entries.map(([key]) => String(key)),
    values: entries.map(([, value]) => value),
    labels_he: entries.map(([key]) => surface.keyLabel(layer.name, key, 'he')),
    labels_en: entries.map(([key]) => surface.keyLabel(layer.name, key, 'en')),
    server_keys: Object.keys(layer.values || {}).map(String),
    server_values: Object.values(layer.values || {}),
  };
}

const say = (fn, count, locale) => surface.rules[fn](count, locale);
fs.writeFileSync(outFile, JSON.stringify({
  day_order: surface.DAY_ORDER,
  layers,
  sentences: {
    written_he_1: say('rulesWrittenSentence', 1, 'he'),
    written_he_2: say('rulesWrittenSentence', 2, 'he'),
    written_en_1: say('rulesWrittenSentence', 1, 'en'),
    written_en_2: say('rulesWrittenSentence', 2, 'en'),
    unauthored_he_1: say('unauthoredSentence', 1, 'he'),
    unauthored_he_3: say('unauthoredSentence', 3, 'he'),
    unauthored_en_1: say('unauthoredSentence', 1, 'en'),
    unauthored_en_3: say('unauthoredSentence', 3, 'en'),
  },
}), 'utf8');
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped modules cannot be run here")
    if not (APP / "node_modules" / "rolldown").is_dir():
        pytest.skip("the bundler the product builds with is not installed")
    return found


@pytest.fixture(scope="module")
def pricing() -> dict:
    """The rate card the product serves, not a fixture that imitates one."""
    from kairos_api.pricing_api import router

    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get("/api/pricing")
    assert response.status_code == 200, response.text
    return response.json()


def _probe(tmp_path: Path, payload: dict, lib_source: str) -> dict:
    node = _node()
    work = tmp_path / "probe"
    work.mkdir(parents=True, exist_ok=True)
    under_test = work / "lib-under-test.js"
    under_test.write_text(lib_source, encoding="utf-8")
    entry = work / "entry.mjs"
    entry.write_text(
        ENTRY.format(lib=LIB.as_posix(), rules=(RULES / "rules-lib.js").as_posix()),
        encoding="utf-8",
    )
    script = work / "probe.mjs"
    script.write_text(
        PROBE.replace("APP_PACKAGE", (APP / "package.json").as_posix()).replace("LIB_PATH", LIB.as_posix()),
        encoding="utf-8",
    )
    payload_file = work / "pricing.json"
    payload_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run(
        [node, str(script), str(entry), str(work / "bundle"), str(payload_file), str(out), str(under_test)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(work),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shipped(tmp_path_factory, pricing) -> dict:
    source = LIB.read_text(encoding="utf-8")
    assert ORDER_GUARD in source, "the ordering under test is not in the shipped module any more"
    return _probe(tmp_path_factory.mktemp("shipped"), pricing, source)


@pytest.fixture(scope="module")
def without_the_order(tmp_path_factory, pricing) -> dict:
    """The same modules with the day ordering cut out, which is what shipped before."""
    mutant = LIB.read_text(encoding="utf-8").replace(ORDER_GUARD, "if (true) return entries;")
    return _probe(tmp_path_factory.mktemp("mutant"), pricing, mutant)


def test_the_card_serves_the_week_iso_keyed_which_is_why_the_reading_order_is_a_choice(pricing):
    """The state under test exists, or everything below is about nothing."""
    day = next(layer for layer in pricing["layers"] if layer["name"] == "day")
    assert sorted(day["values"]) == ["1", "2", "3", "4", "5", "6", "7"]
    assert set(SUNDAY_FIRST_KEYS) == set(day["values"]), "the reader may not invent or drop a day"


def test_the_day_layer_reads_the_week_sunday_first_in_both_locales(shipped):
    day = shipped["layers"]["day"]
    assert day["keys"][0] == "7", "the Israeli week starts on Sunday, which is ISO weekday 7"
    assert day["keys"] == SUNDAY_FIRST_KEYS
    assert day["labels_he"] == SUNDAY_FIRST_HE
    assert day["labels_en"] == SUNDAY_FIRST_EN
    assert day["labels_he"][-2:] == ["שישי", "שבת"], "the weekend is Friday and Saturday, and it is last"


def test_reordering_the_reading_moved_no_key_and_no_multiplier(shipped):
    """The store is ISO and stays ISO: a save still sends the key it was sent."""
    day = shipped["layers"]["day"]
    assert sorted(day["keys"]) == sorted(day["server_keys"]), "no day gained or lost"
    assert dict(zip(day["keys"], day["values"])) == dict(zip(day["server_keys"], day["server_values"]))


def test_without_the_order_the_same_card_reads_monday_first(without_the_order):
    """The mutant, which is exactly what a critic measured on the running product."""
    day = without_the_order["layers"]["day"]
    assert day["keys"] == ["1", "2", "3", "4", "5", "6", "7"]
    assert day["labels_he"] == ["שני", "שלישי", "רביעי", "חמישי", "שישי", "שבת", "ראשון"]
    assert day["labels_en"] == ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def test_every_other_layer_keeps_the_order_the_server_sent(shipped, without_the_order):
    """Only the week and the positions have a reading order of their own.

    The position layer is the second exception and it is the server's own doing:
    it ships a ``vocabulary`` (the trade's positions 1 to 5 then L then the
    middle default) and the reader follows it, so an ordinal nobody has priced
    still appears and is settable. Every other layer keeps the order it was sent.
    """
    for name, layer in shipped["layers"].items():
        if name in ("day", "position"):
            continue
        assert layer["keys"] == layer["server_keys"], f"{name} was reordered and should not be"
        assert layer["keys"] == without_the_order["layers"][name]["keys"]


def test_the_position_layer_reads_its_declared_vocabulary_and_keeps_every_key(shipped):
    """1 to 5, then L, then the middle default, and nothing the server sent is lost."""
    position = shipped["layers"]["position"]
    assert position["keys"] == ["1", "2", "3", "4", "5", "L", "default_middle"]
    assert set(position["server_keys"]) <= set(position["keys"]), "no priced key was dropped"
    # 4 and 5 are unset on the shipped card, so they read as absent, not as 1.0.
    by_key = dict(zip(position["keys"], position["values"]))
    assert by_key["4"] is None and by_key["5"] is None
    assert by_key["L"] == 1.2


def test_a_day_the_order_does_not_name_is_kept_rather_than_dropped(tmp_path, pricing):
    """A rate card carrying something new still shows it, at the end."""
    payload = {"layers": [{"name": "day", "kind": "premium", "values": {"7": 1.0, "9": 1.4}}]}
    day = _probe(tmp_path, payload, LIB.read_text(encoding="utf-8"))["layers"]["day"]
    assert day["keys"] == ["7", "9"]
    assert day["values"] == [1.0, 1.4]


def test_the_rate_card_renders_through_the_ordered_reader(shipped):
    """The component under measurement is the one wired to the ordering."""
    source = MANAGER.read_text(encoding="utf-8")
    assert "const entries = layerEntries(layer);" in source
    assert "Object.entries(layer.values" not in source, "the unordered read is what put Sunday last"
    assert "layerEntries" in source.split("from './pricing-layers-lib'")[0], "it has to be imported"
    assert "saveMultiplier(layer.name, key, event.target.value)" in source, (
        "the save still sends the ISO key the entry carries"
    )


def test_the_price_tester_reads_the_week_through_the_ordered_reader_too():
    """The other weekday control on the same page, so it cannot drift off again."""
    source = TESTER.read_text(encoding="utf-8")
    assert "[1, 2, 3, 4, 5, 6, 7]" not in source, "the literal week is what put Sunday last"
    assert "const WEEKDAY_OPTIONS = DAY_ORDER.map(Number);" in source
    assert "DAY_ORDER" in source.split("from './pricing-layers-lib'")[0], "it has to be imported"
    assert "weekday_iso: FIRST_WEEKDAY," in source, "and it opens on the first day of that week"


def test_one_rule_is_written_as_one_rule_in_both_languages(shipped):
    said = shipped["sentences"]
    assert said["written_he_1"] == "כלל אחד נכתב לתוכנית"
    assert said["written_he_2"] == "2 כללים נכתבו לתוכנית"
    assert said["written_en_1"] == "1 rule written to the plan"
    assert said["written_en_2"] == "2 rules written to the plan"
    assert said["unauthored_he_1"].startswith("כלל אחד מחייב את התוכנית ואין לו מחבר")
    assert said["unauthored_he_3"].startswith("3 כללים מחייבים את התוכנית ואין להם מחבר")
    assert said["unauthored_en_1"].startswith("1 rule binds the plan and carries no author")
    assert said["unauthored_en_3"].startswith("3 rules bind the plan and carry no author")
    for text in said.values():
        assert "1 כללים" not in text and "1 rules" not in text


def _wrapped_strings(path: Path) -> list[tuple[int, str]]:
    """Every string literal in one module that is hard-wrapped across source lines.

    A docstring is not a display string and a triple-quoted literal is one line
    of text however it is laid out, so both are left alone. What is flagged is
    implicit concatenation: two adjacent literals on two lines, which is the one
    form that splits a sentence a person reads.
    """
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        literal = isinstance(node, ast.JoinedStr) or (
            isinstance(node, ast.Constant) and isinstance(node.value, str)
        )
        if not literal:
            continue
        segment = ast.get_source_segment(source, node) or ""
        if "\n" not in segment:
            continue
        head = segment.lstrip("fFrRbBuU")
        if head.startswith('"""') or head.startswith("'''"):
            continue
        found.append((node.lineno, segment.splitlines()[0].strip()[:70]))
    return found


def test_no_display_string_this_piece_owns_is_split_across_two_source_lines():
    offenders: list[str] = []
    for name in OWNED_MODULES:
        path = ROOT / "kairos_api" / name
        assert path.exists(), f"{name} is on this piece's row and is not on disk"
        offenders += [f"{name}:{line} {text}" for line, text in _wrapped_strings(path)]
    assert offenders == [], "a display string was hard-wrapped: " + "; ".join(offenders)


def test_no_display_string_on_this_surface_is_concatenated_across_lines():
    """The same law on the frontend, where the wrap reads `'text ' +` at line end."""
    offenders: list[str] = []
    for path in sorted(list(RULES.glob("*.jsx")) + list(RULES.glob("*.js"))):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.rstrip()
            if stripped.endswith("' +") or stripped.endswith('" +') or stripped.endswith("` +"):
                offenders.append(f"{path.name}:{number} {stripped[-60:]}")
    assert offenders == [], "a display string was hard-wrapped: " + "; ".join(offenders)
