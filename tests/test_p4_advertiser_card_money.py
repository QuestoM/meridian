"""P4: a pricing row carries the money of the advertiser it actually prices.

The measured defect. ``GET /api/advertisers/identity`` returns 41 resolved
advertisers, each carrying its shown name, what binds it and its gross, net and
spots from the priced daily ledger, and no surface read it: measured, zero hits
for the route across the whole dashboard source. So the pricing rows rendered
with revenue "-" not because the money was missing but because nobody joined it,
and the reader met 45 rows that could not name anybody.

This file runs the shipped join. ``advertiser-stats-helpers.js`` and the two
modules it imports are copied verbatim into a temporary directory, node parses
them as ES modules with their extensionless imports resolved, and the payload
fed in is the real shape of that route, keys and all.

Four states are asserted, because a row and an advertiser are not the same thing
and the join has to keep them apart: a row bound to an advertiser with priced
spots carries that advertiser's real figure, a row bound to nobody carries a
dash and the sentence that says where it gets bound, a bound advertiser with no
priced spot keeps its dash and the ledger's own reason rather than a zero, and
the total across rows never invents money for a row that prices nobody.

The last test cuts the join out of the shipped source and asserts the figure
disappears, so a pass here can never be vacuous.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ROOT / "tv-break-dashboard" / "src" / "clients"
STATS = CLIENTS / "advertiser-stats-helpers.js"
NAMES = CLIENTS / "advertiser-name-helpers.js"
SHARED = CLIENTS / "advertisers-helpers.js"

# The client the critic measured, and its figures in the shipped priced ledger.
CLIENT = "פריסבי"
CLIENT_GROSS = 56034.0
CLIENT_NET = 53792.64
CLIENT_SPOTS = 6
BASIS = "Wally_Prime_Reshet_Example_2025-04-27.csv"

# The join, as shipped, and the cut that removes it.
JOIN = "    const boundTo = record && record.rules ? record.rules.advertiser_id : null;"
NO_JOIN = "    const boundTo = null;"

HARNESS = """
import fs from 'node:fs';
import { registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';

const [statsPath, namesPath, sharedPath, payloadPath, outPath] = process.argv.slice(2);
const MAP = {
  './advertiser-name-helpers': pathToFileURL(namesPath).href,
  './advertisers-helpers': pathToFileURL(sharedPath).href,
};
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) {
      return { url: MAP[specifier], shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

const stats = await import(pathToFileURL(statsPath).href);
const { identity, rows } = JSON.parse(fs.readFileSync(payloadPath, 'utf8'));
const index = stats.indexIdentityByRow(identity);
const merged = rows.map((row) => stats.mergeRowWithIdentity(row, index));
fs.writeFileSync(outPath, JSON.stringify({
  indexedRows: [...index.keys()],
  merged,
  provenance: merged.map((row) => stats.revenueProvenance(row, 'he')),
}), 'utf8');
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped join cannot be executed here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True, text=True, check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so the imports cannot be resolved")
    for path in (STATS, NAMES, SHARED):
        if not path.is_file():
            pytest.skip(f"{path.name} is not in this tree")
    return found


def _payload() -> dict:
    """The route's real record shape: one bound, one bound with nothing priced."""
    return {
        "identity": {
            "count": 3,
            "advertisers": [
                {
                    "advertiser": CLIENT, "shown_name": CLIENT, "aliases": [],
                    "source": "observed", "resolved": True,
                    "rules": {"bound": True, "advertiser_id": "ADV_03",
                              "baseline_premium": 1.15, "effective_premium": 1.15,
                              "rule_count": 0, "reason": ""},
                    "money": {"advertiser": CLIENT, "gross": CLIENT_GROSS, "net": CLIENT_NET,
                              "spots": CLIENT_SPOTS, "basis": BASIS, "reason": ""},
                },
                {
                    "advertiser": "בית בכפר", "shown_name": "בית בכפר", "aliases": [],
                    "source": "observed", "resolved": True,
                    "rules": {"bound": True, "advertiser_id": "ADV_04",
                              "baseline_premium": 1.0, "effective_premium": 1.0,
                              "rule_count": 0, "reason": ""},
                    "money": {"advertiser": "בית בכפר", "gross": None, "net": None,
                              "spots": 0, "basis": BASIS,
                              "reason": "למפרסם הזה אין תשדיר מתומחר בקובץ היומי הנקרא."},
                },
                {
                    "advertiser": "סלקום", "shown_name": "סלקום", "aliases": [],
                    "source": "observed", "resolved": True,
                    "rules": {"bound": False, "advertiser_id": None,
                              "baseline_premium": None, "effective_premium": 1.0,
                              "rule_count": 0, "reason": "No rules row is bound to this advertiser."},
                    "money": {"advertiser": "סלקום", "gross": 12000.0, "net": 11520.0,
                              "spots": 2, "basis": BASIS, "reason": ""},
                },
            ],
        },
        "rows": [
            {"advertiser_id": "ADV_03", "name": CLIENT, "display_name": "", "aliases": "",
             "default_premium": 1.15, "conditions": []},
            {"advertiser_id": "ADV_04", "name": "בית בכפר", "display_name": "", "aliases": "",
             "default_premium": 1.0, "conditions": []},
            {"advertiser_id": "ADV_01", "name": "", "display_name": "", "aliases": "",
             "default_premium": 1.0, "conditions": []},
        ],
    }


def _run(tmp_path: Path, stats_source: str) -> dict:
    node = _node()
    work = tmp_path / "join"
    work.mkdir(parents=True, exist_ok=True)
    stats = work / "stats.mjs"
    stats.write_text(stats_source, encoding="utf-8")
    names = work / "names.mjs"
    names.write_text(NAMES.read_text(encoding="utf-8"), encoding="utf-8")
    shared = work / "shared.mjs"
    shared.write_text(SHARED.read_text(encoding="utf-8"), encoding="utf-8")
    harness = work / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    payload = work / "payload.json"
    payload.write_text(json.dumps(_payload(), ensure_ascii=False), encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run(
        # shell/bidi is a real shell primitive advertiser-stats-helpers.js and
        # advertisers-helpers.js import; the copies land outside the dashboard
        # tree so plain node resolution cannot find it, and this loader hook
        # resolves it to the real compiled module instead.
        [node, "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
         str(harness), str(stats), str(names), str(shared), str(payload), str(out)],
        capture_output=True, text=True, check=False, cwd=str(work),
    )
    if result.returncode != 0:
        pytest.fail(f"the shipped join did not run: {result.stderr.strip()[:600]}")
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture()
def joined(tmp_path) -> dict:
    return _run(tmp_path, STATS.read_text(encoding="utf-8"))


def test_the_index_holds_only_the_rows_that_price_somebody(joined) -> None:
    """An advertiser bound to no row cannot put its money on any card."""
    assert joined["indexedRows"] == ["ADV_03", "ADV_04"]


def test_the_bound_row_carries_the_real_figure_of_the_advertiser_it_prices(joined) -> None:
    """The join the surface was missing, on the client the critic opened."""
    row = joined["merged"][0]
    assert row["bound_advertiser"] == CLIENT
    assert row["revenue"] == pytest.approx(CLIENT_GROSS)
    assert row["revenue_net"] == pytest.approx(CLIENT_NET)
    assert row["bound_spots"] == CLIENT_SPOTS
    assert row["revenue_basis"] == BASIS
    assert BASIS in joined["provenance"][0]


def test_a_bound_advertiser_with_no_priced_spot_keeps_its_dash(joined) -> None:
    """Nothing rounds an absent figure to zero, and the reason is the ledger's."""
    row = joined["merged"][1]
    assert row["bound_advertiser"] == "בית בכפר"
    assert row["revenue"] is None
    assert row["bound_spots"] == 0
    assert joined["provenance"][1] == "למפרסם הזה אין תשדיר מתומחר בקובץ היומי הנקרא."


def test_a_row_that_prices_nobody_shows_no_money_and_says_where_to_bind_it(joined) -> None:
    """The state all forty five shipped rows are in, stated rather than blank."""
    row = joined["merged"][2]
    assert row["bound_advertiser"] == ""
    assert row["revenue"] is None
    assert row["revenue_net"] is None
    assert "בכרטיס הלקוח" in joined["provenance"][2]


def test_no_money_reaches_a_card_through_an_unbound_advertiser(joined) -> None:
    """סלקום has money and no rules row, so it appears on no card at all."""
    assert all(row["bound_advertiser"] != "סלקום" for row in joined["merged"])
    assert sum(row["revenue"] or 0 for row in joined["merged"]) == pytest.approx(CLIENT_GROSS)


def test_without_the_join_the_figure_disappears_from_every_card(tmp_path) -> None:
    """The state that shipped: the payload is read and no card can use it."""
    source = STATS.read_text(encoding="utf-8")
    assert JOIN in source, "the shipped join moved, so this mutation is stale"
    cut = _run(tmp_path, source.replace(JOIN, NO_JOIN))
    assert cut["indexedRows"] == []
    assert all(row["revenue"] is None for row in cut["merged"])
