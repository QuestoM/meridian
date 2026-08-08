"""P11, Bar 3: the shipped JavaScript, executed rather than read.

The third file of this guard, split from ``test_p11_surface_regression.py`` on
the boundary that matters: everything here runs the real module in node, and
everything there reads a component as text.

The distinction earns its own file. A guard that greps for a string passes on
the day somebody renames the thing it greps for, and it cannot see a defect that
is about syntax or about arithmetic at all. The first test below exists because
one of those was shipping: nothing in the repo imports this tree, so the build
never parsed it, and a mismatched JSX tag sat in ``PacingWorkspace.jsx`` where no
text guard could reach it.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import makegood_store, pacing_alerts_api
from kairos_api import pacing_alerts_api_board, pacing_alerts_api_read

# The two bidi controls a figure is allowed to carry: the first-strong isolate
# and the pop that closes it, the same pair ``src/shell/bidi.jsx`` uses.
FSI = "\u2068"
PDI = "\u2069"

SURFACE = Path("tv-break-dashboard/src/clients/pacing")
ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / SURFACE / "pacing-helpers.js"

# The bundler driver, held here rather than in a script beside the surface.
#
# It lived in ``src/clients/pacing/verify-parses.mjs`` and this file shelled out
# to it by path. Measured: that script was never committed while this test was,
# so at HEAD the one guard that can see a syntax error on this surface could not
# run at all, and the defect it was written for was sitting in the committed
# tree. A guard that lives in one file and runs from another is only as tracked
# as its weakest half, so both halves are now the same file.
#
# It is executed with ``node --input-type=module --eval`` from the frontend
# package, which is what makes the bare ``rolldown`` specifier resolve: node
# resolves a specifier in an evaluated module against the working directory.
#
# Everything outside this directory is external and every stylesheet is stubbed,
# because the question is whether these files are valid, not whether the app
# links.
PARSE_DRIVER = """
import { build } from 'rolldown';

const CSS_STUB = '\\0pacing-css-stub';

const stubStylesheets = {
  name: 'stub-stylesheets',
  resolveId(id) {
    return id.endsWith('.css') ? { id: CSS_STUB, external: false } : null;
  },
  load(id) {
    return id === CSS_STUB ? 'export default {}' : null;
  },
};

try {
  await build({
    input: %s,
    // A bare specifier is somebody else's module and is not this tree's to parse.
    external: (id) => !id.startsWith('.') && !id.startsWith('/') && !id.startsWith('\\0'),
    resolve: { extensions: ['.js', '.jsx'] },
    plugins: [stubStylesheets],
    output: { dir: %s },
    logLevel: 'silent',
  });
  process.stdout.write('PARSE_OK');
} catch (error) {
  process.stdout.write(`PARSE_FAIL\\n${String(error.message || error)}`);
  process.exitCode = 1;
}
"""


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(pacing_alerts_api.router)
    return TestClient(app)


def test_every_component_on_this_surface_actually_compiles(tmp_path) -> None:
    """The repo build never parses this tree, so a syntax error here is silent.

    Nothing imports ``src/clients/pacing/**``, so ``npm run build`` transforms
    3,510 modules and none of them is one of these. Measured: PacingWorkspace.jsx
    opened a ``<Figure>`` and closed a ``</span>``, and the whole surface failed
    to compile the instant the published mount was applied to a scratch copy. No
    guard in this file could see it, because every one of them reads a component
    as text.

    So the real bundler parses the tree, with everything outside it external and
    the stylesheets stubbed. Proven to fail against that exact defect and pass
    against the fix.

    An empty entry list is a mis-invocation and not a parse failure, so it is
    asserted here rather than handed to a bundler that would answer PARSE_OK
    about nothing at all.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    package = ROOT / "tv-break-dashboard"
    if not (package / "node_modules" / "rolldown").exists():
        pytest.skip("the frontend dependencies are not installed here")
    entries = [str(path.relative_to(ROOT / "tv-break-dashboard"))
               for path in sorted((ROOT / SURFACE).glob("*.jsx"))]
    assert entries, "no component was found to parse, which is a mis-invocation and not a pass"
    script = PARSE_DRIVER % (json.dumps(entries), json.dumps(str(tmp_path / "parse-out")))
    done = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        cwd=package, capture_output=True, text=True, timeout=180,
    )
    assert done.stdout.startswith("PARSE_OK"), done.stdout or done.stderr


def test_a_figure_that_carries_a_unit_isolates_its_numeral_and_never_its_words() -> None:
    """A left-to-right isolate lays its contents out left to right.

    Wrapped around a phrase that is already Hebrew it puts the words in the wrong
    order. Measured in a browser on the shipped board before the fix, the headline
    figure of every one of 56 rows read ``4.4 מתוך נקודות רייטינג 35``, the unit
    ahead of its own number, because ``pair`` isolated the whole of
    ``amount(35, rating_points, he)``. The isolate now sits inside ``amount``
    around the numeral alone.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    script = f"""
import {{ amount, pair, bare }} from {json.dumps(str(HELPERS))};
process.stdout.write(JSON.stringify({{
  points: amount(35, 'rating_points', 'he'),
  money: amount(70000, 'ils', 'he'),
  pair: pair(4.4, 35, 'rating_points', 'he'),
  english: amount(35, 'rating_points', 'en'),
  bare: bare(4.4, 'rating_points', 'he'),
}}));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    read = json.loads(done.stdout)
    for key in ("points", "money", "pair"):
        value = read[key]
        assert FSI in value and PDI in value, (key, value)
        for run in re.findall(f"{FSI}(.*?){PDI}", value):
            assert not re.search(r"[֐-׿]", run), (key, run)
    # The English forms are already left to right and take no isolate at all.
    assert FSI not in read["english"]
    assert FSI not in read["bare"]


def test_a_unit_on_a_form_label_is_a_word_and_never_the_stored_key() -> None:
    """The ledger read carries no unit vocabulary, so reaching for one prints the key.

    Measured in a browser after the first attempt at this fix: the offer field
    read "Offer, in rating_points".
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    script = f"""
import {{ unitWord }} from {json.dumps(str(HELPERS))};
process.stdout.write(JSON.stringify({{
  pointsEn: unitWord('rating_points', 'en'),
  pointsHe: unitWord('rating_points', 'he'),
  moneyEn: unitWord('ils', 'en'),
  moneyHe: unitWord('ils', 'he'),
}}));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    read = json.loads(done.stdout)
    for value in read.values():
        assert "_" not in value, value
        assert value not in {"rating_points", "ils"}, value
    assert re.search(r"[֐-׿]", read["pointsHe"])
    assert re.search(r"[֐-׿]", read["moneyHe"])
    ledger = (ROOT / SURFACE / "MakeGoodLedger.jsx").read_text(encoding="utf-8")
    assert "vocabulary.units" not in ledger


def test_the_api_and_the_surface_hold_one_rule_for_when_a_make_good_may_be_raised() -> None:
    """Two rules for one act let any client put a debt in the ledger the product denies.

    Measured before this: ``POST /api/make-goods`` on CMP_D040 answered 201 with
    ``deficit_kind: to_date``, while the surface's own ``remedyFor`` offered that
    raise on 0 of 56 rows. 13 of the 56 reached the ``to_date`` rung. The rule
    kept is the surface's, and the trade says why: a make-good compensates a spot
    that did not air or aired wrong, and a flight with unbooked days ahead has
    had no spot fail yet.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    body = _client().get("/api/pacing").json()
    assert body["raise_rule"]["raisable_deficit_kinds"] == list(pacing_alerts_api_read.RAISABLE_KINDS)
    assert makegood_store.TO_DATE not in body["raise_rule"]["raisable_deficit_kinds"]

    # The surface's answer for every shipped row, out of the shipped module.
    script = f"""
import {{ remedyFor }} from {json.dumps(str(HELPERS))};
const rows = {json.dumps(body["rows"])};
process.stdout.write(JSON.stringify(rows.map((row) => remedyFor(row, {{}}).kind)));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    offered = json.loads(done.stdout)
    assert len(offered) == len(body["rows"])

    # The server's answer for the same rows, through the same reader the write
    # path uses. The two must agree row for row.
    from kairos_api import pacing_alerts_api_wire as wire

    full = wire.expand(pacing_alerts_api_read.board_payload())
    as_of = pacing_alerts_api_board.parse_date(full["as_of"]["instant"])
    for row, kind in zip(full["rows"], offered):
        deficit, why = pacing_alerts_api_read.raisable_deficit(row, as_of)
        assert (deficit is not None) == (kind == "raise"), (row["campaign_id"], kind, why)


def test_the_javascript_that_expands_the_board_is_the_inverse_of_the_python_that_collapses_it() -> None:
    """The expansion ships in the browser, so it is executed rather than read."""
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    collapsed = pacing_alerts_api_read.board_payload()
    script = f"""
import {{ readFileSync }} from 'node:fs';
const source = readFileSync({json.dumps(str(ROOT / SURFACE / 'pacing-api.js'))}, 'utf8');
const body = source.slice(source.indexOf('const PROSE'), source.indexOf('export function loadBoard'))
  .replace('export function expandBoard', 'function expandBoard');
const run = new Function(`${{body}}; return expandBoard;`);
process.stdout.write(JSON.stringify(run()({json.dumps(collapsed)})));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    rebuilt = json.loads(done.stdout)
    for key in ("reasons", "forward_reasons", "reference_rule", "wire"):
        rebuilt.pop(key, None)
    assert _shape(rebuilt) == _shape(_uncollapsed())


def _uncollapsed() -> dict:
    """The board with nothing lifted off it, for the round-trip comparison.

    The same helper the Python side of this round trip uses, kept in both files
    rather than imported across them, because a test module that imports another
    test module makes the collection order load-bearing.
    """
    from kairos_api import pacing_alerts_api_wire as wire

    keep = wire.collapse
    wire.collapse = lambda payload: payload
    try:
        return pacing_alerts_api_read.board_payload()
    finally:
        wire.collapse = keep


def _shape(value):
    """The payload with whole floats and ints read as one number.

    node writes 5280.0 as 5280 on the way back through JSON, which is the same
    figure and a different string. The comparison is about the shape and the
    values, not about how a serialiser spells a round number.
    """
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_shape(item) for item in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value)
    return value


def test_the_two_counting_sentences_count_the_board_they_are_about() -> None:
    """The headline above the list is arithmetic, so it is executed and not greped.

    Both sentences were inside ``PacingWorkspace.jsx`` and both were guarded by a
    text search for a field name, which passes on the day somebody changes what
    the field is divided by. They moved to ``pacing-summary.js`` when the panel
    reached the size law, whole and with their prose unchanged, and the move is
    what makes this possible: node runs the shipped module against the shipped
    board and the counts are checked against the payload's own.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    body = _client().get("/api/pacing").json()
    module = ROOT / SURFACE / "pacing-summary.js"
    script = f"""
import {{ headlineSentence, seededSentence, decidedCount }} from {json.dumps(str(module))};
const board = {{ status: 'ready', payload: {json.dumps(body)} }};
process.stdout.write(JSON.stringify({{
  en: headlineSentence(board, 'en'),
  he: headlineSentence(board, 'he'),
  seeded: seededSentence(board, 'en'),
  decided: decidedCount(board),
  loading: headlineSentence({{ status: 'loading', payload: null }}, 'en'),
  failed: headlineSentence({{ status: 'failed', payload: null }}, 'en'),
}}));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    read = json.loads(done.stdout)
    counts = body["counts"]
    asking = counts["behind"] + counts["at_risk"]
    # Every figure in the sentence is one the server counted, and the one figure
    # it derives is the subtraction the reader would otherwise have to do.
    assert f"{asking - read['decided']} of {counts['total']} campaigns still need a decision" in read["en"]
    assert f"{counts['unknown']} cannot be paced yet" in read["en"]
    assert f"{counts['demo']} of the {counts['total']} are demo rows" in read["seeded"]
    assert f"{counts['demo_needing_a_decision']} of the {asking} rows that need a decision" in read["seeded"]
    # A read in flight and a read that failed are two facts and neither is a count.
    assert not re.search(r"\d", read["loading"])
    assert not re.search(r"\d", read["failed"])
    # The Hebrew joins its figures through the first-strong isolate, as the rest
    # of this surface does, and never through a left-to-right one.
    assert FSI in read["he"] and "\u2066" not in read["he"]


def test_a_refusal_that_is_a_plain_string_still_reaches_the_person_who_was_refused() -> None:
    """The auth middleware answers with detail as a string, not as the bilingual shape.

    Reading only the bilingual shape returned an empty string for it, so a
    refused write said nothing at all.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    script = f"""
import {{ readFileSync }} from 'node:fs';
const source = readFileSync({json.dumps(str(ROOT / SURFACE / 'pacing-api.js'))}, 'utf8');
const start = source.indexOf('export function refusalText');
const body = source.slice(start, source.indexOf('export function refusalOpens'))
  .replace('export function refusalText', 'function refusalText');
const run = new Function(`${{body}}; return refusalText;`);
const refusalText = run();
process.stdout.write(JSON.stringify({{
  plain: refusalText({{ detail: 'Your account may read this and not change it.' }}, 'he'),
  bilingual: refusalText({{ detail: {{ message_en: 'en', message_he: 'he' }} }}, 'he'),
  nothing: refusalText({{ detail: null }}, 'he'),
}}));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    read = json.loads(done.stdout)
    assert read["plain"] == "Your account may read this and not change it."
    assert read["bilingual"] == "he"
    assert read["nothing"] == ""


def test_a_percentage_never_rounds_a_short_figure_up_to_the_one_it_missed() -> None:
    """A campaign short of its reference and one level with it were one figure.

    Measured on the shipped board by the round-three critic, over the payload's
    own 99 pace ratios: 38 of them sit between 0.995 and 1 and every one printed
    ``100%``, beside the 7 that are exactly 1 and printed the same string. The
    percentage is the one number on this row a reader takes at face value, and
    on that data it said a campaign had reached a reference it had not.

    Whole percent is kept everywhere else on purpose. The verdict is decided at
    0.95 and 0.85, so a column reading 90.9 and 93.7 invites a comparison the
    rule does not make.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    script = f"""
import {{ percent }} from {json.dumps(str(HELPERS))};
process.stdout.write(JSON.stringify({{
  short: percent(0.9989, 'en'),
  shortHe: percent(0.9989, 'he'),
  nearly: percent(0.9996, 'en'),
  half: percent(0.995, 'en'),
  level: percent(1, 'en'),
  over: percent(1.04, 'en'),
  ordinary: percent(0.88, 'en'),
  ninetyOne: percent(0.9091, 'en'),
  missing: percent(null, 'en'),
}}));
"""
    done = subprocess.run(["node", "--input-type=module", "--eval", script],
                          capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    read = json.loads(done.stdout)
    # Nothing below the reference may print as having reached it.
    for key in ("short", "shortHe", "nearly", "half"):
        assert read[key] != "100%", (key, read[key])
        assert float(read[key].rstrip("%").replace(",", "")) < 100, (key, read[key])
    # And the residual rounding only ever goes down, so the figure is never a
    # claim of more than was counted.
    assert read["short"] == "99.8%"
    assert read["nearly"] == "99.9%"
    # Level is level, over is over, and every other figure keeps whole percent.
    assert read["level"] == "100%"
    assert read["over"] == "104%"
    assert read["ordinary"] == "88%"
    assert read["ninetyOne"] == "91%"
    assert read["missing"] is None

    # And the board really does carry the figures this is about, so the guard
    # cannot quietly stop testing anything.
    body = _client().get("/api/pacing").json()
    ratios = [line["pace"]["ratio"]
              for row in body["rows"] for key in ("rating", "money")
              for line in [row.get(key) or {}]
              if (line.get("pace") or {}).get("ratio") is not None]
    near = [value for value in ratios if 0.995 <= value < 1]
    assert near, "no ratio on the shipped board reaches this case, so re-measure before trusting the guard"


def test_the_ledger_says_so_when_it_does_not_know_who_acted() -> None:
    """The ledger's own sentence promises who acted and the trail dropped it.

    Measured by the round-three critic on a server with an uninitialised account
    store, which is how this product runs before an operator creates the first
    account: ``/api/auth/me`` answers ``auth_disabled`` true, a write lands with
    ``raised_by`` empty, and the record's trail read ``Recorded 07/08/2026, 13:50
    UTC. Counted as of 27/04/2025, 22:59.`` with no actor and no statement that
    there was none, while the sign-off block two lines above said this product
    records who acted. Tri-state: an unknown actor is stated, never omitted.
    """
    ledger = (ROOT / SURFACE / "MakeGoodLedger.jsx").read_text(encoding="utf-8")
    assert "function actorClause" in ledger
    # One clause, so the trail and the closure cannot come to say it two ways.
    assert ledger.count("actorClause(") >= 3
    # The conditional that dropped the actor is gone from both sites.
    assert "record.raised_by ?" not in ledger
    assert "record.closed_by ?" not in ledger
