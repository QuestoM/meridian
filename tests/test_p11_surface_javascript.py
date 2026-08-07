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


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(pacing_alerts_api.router)
    return TestClient(app)


def test_every_component_on_this_surface_actually_compiles() -> None:
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
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed here")
    package = ROOT / "tv-break-dashboard"
    if not (package / "node_modules" / "rolldown").exists():
        pytest.skip("the frontend dependencies are not installed here")
    entries = [str(path.relative_to("tv-break-dashboard")) for path in sorted(SURFACE.glob("*.jsx"))]
    done = subprocess.run(
        ["node", str(ROOT / SURFACE / "verify-parses.mjs"), *entries],
        cwd=package, capture_output=True, text=True, timeout=180,
    )
    assert done.stdout.startswith("PARSE_OK"), done.stdout


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
