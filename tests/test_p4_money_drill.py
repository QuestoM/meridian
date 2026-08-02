"""P4: the record's money control opens that client's rows, or opens nothing.

The measured defect. A client record prints three figures under a control that
says it opens every row behind them. Pressing it called back into the workspace,
which set the view to money and nothing else, because which row the board had
open was the board's own state and no caller could reach it. On פריסבי, a client
with real money, the press produced no drill at all and left the reader to find
the row again in a 41-row ranking. On a client booked with no priced spot it was
worse: the board cannot hold a row that is not in the ledger, so the screen
showed the leader's money beside a record whose own figures were dashes, which is
one client's money presented as the answer to a question about another.

This file measures the fix on the real priced ledger and the real client tree.
The shipped helper module is copied verbatim into a temporary ``.mjs`` file so
node parses it as an ES module, its one import is resolved to a stub, and the
harness performs the two steps the product performs: resolve what the record's
control opens, then resolve that against the rows the board renders. The last
test mutates the guard away and asserts the booked client lands on a board that
cannot hold it, so a pass here can never be vacuous.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ROOT / "tv-break-dashboard" / "src" / "clients"
HELPERS = CLIENTS / "clients-money-helpers.js"
BOARD = CLIENTS / "MoneyBoard.jsx"
WORKSPACE = CLIENTS / "ClientsWorkspace.jsx"
RECORD = CLIENTS / "ClientRecord.jsx"

# The name booked for this measurement. It exists only inside a temporary store,
# so the operator's own campaign file is never written by the suite.
BOOKED_CLIENT = "חברת בדיקה חדשה"

# The guard the fix turns on, and the mutation that removes it.
GUARD = "  return Boolean(client) && client.gross !== null && client.gross !== undefined;"
WITHOUT_GUARD = "  return Boolean(client);"

STUB = """
export function pageText(locale, en, he) {
  return locale === 'he' ? he : en;
}
"""

HARNESS = """
import fs from 'node:fs';
import { registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';

const HELPERS = pathToFileURL(process.argv[2]).href;
const STUB = pathToFileURL(process.argv[3]).href;

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.endsWith('shell/format')) {
      return { url: STUB, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

const helpers = await import(HELPERS);
const tree = JSON.parse(fs.readFileSync(process.argv[4], 'utf8'));
const money = JSON.parse(fs.readFileSync(process.argv[5], 'utf8'));
const probes = JSON.parse(process.argv[6]);

// Which column each grouping is keyed by, exactly as MoneyBoard's GROUPS holds it.
const FIELDS = { advertisers: 'advertiser', agencies: 'agency', campaigns: 'campaign', breaks: 'break_id' };

const rows = helpers.flattenClients(
  tree.agencies || [],
  tree.unlinked || [],
  tree.clients_booked_without_spots || [],
);

// The workspace's own openMoneyFor, as it now reads.
function openMoneyFor(state, name) {
  const target = helpers.moneyTarget(rows.find((row) => row.advertiser === name));
  if (!target) {
    return state;
  }
  return { view: 'money', drill: target, client: name };
}

// What MoneyBoard puts on screen for a drill, counted the way the DOM counts it:
// a key renders one .clients-drill and no ranking, no key renders the ranking.
function render(drill) {
  const group = drill.group || helpers.NO_DRILL.group;
  const key = drill.key || '';
  const field = FIELDS[group];
  const groupRows = money[group] || [];
  const ranked = [...groupRows].sort((left, right) => right.gross - left.gross);
  const open = key ? groupRows.find((row) => String(row[field]) === key) || null : null;
  return {
    group,
    key,
    drills: key ? 1 : 0,
    opened: open ? { name: String(open[field]), gross: open.gross, net: open.net, spots: open.spots } : null,
    rankedRows: key ? 0 : ranked.length,
    leader: ranked.length ? String(ranked[0][field]) : '',
  };
}

const out = { pressed: {}, walk: [] };
probes.forEach((name) => {
  const record = rows.find((row) => row.advertiser === name) || null;
  const before = { view: 'clients', drill: helpers.NO_DRILL, client: name };
  const after = openMoneyFor(before, name);
  out.pressed[name] = {
    record: record ? {
      gross: record.gross,
      net: record.net,
      spots: record.spots,
      reason_en: record.money_reason_en,
      reason_he: record.money_reason_he,
    } : null,
    target: helpers.moneyTarget(record),
    view: after.view,
    client: after.client,
    board: render(after.drill),
  };
});

// The board's own drilling, which is the capability that must not get worse.
// Every one of these was a setState inside the board and is now the same value
// arriving as a prop.
let drill = helpers.NO_DRILL;
const step = (next, label) => {
  drill = next;
  out.walk.push([label, render(drill)]);
};
step({ group: 'campaigns', key: '' }, 'group tab campaigns');
const campaign = money.campaigns[0];
step({ group: 'campaigns', key: String(campaign.campaign) }, 'row click');
step({ group: 'breaks', key: String(campaign.breaks[0]) }, 'break chip');
step({ group: 'breaks', key: '' }, 'back');

process.stdout.write(JSON.stringify(out));
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped helper cannot be executed here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so the helper's import cannot be stubbed")
    return found


@pytest.fixture(scope="module")
def payload(tmp_path_factory):
    """The real client tree and the real ledger, with one client booked.

    The shipped campaign file is empty, so the state measured on screen (a client
    that exists because someone booked it and has no priced spot) is created here
    through the store's own writer into a temporary file. ``data/campaigns.csv``
    is never touched by this suite.
    """
    from kairos_api import campaigns_api_store as store
    from kairos_api.campaigns_read_clients import client_tree
    from kairos_api.campaigns_read_money import board

    tmp = tmp_path_factory.mktemp("campaign-store")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store, "CAMPAIGNS_PATH", tmp / "campaigns.csv")
        patch.setattr(store, "BACKUP_DIR", tmp / "_backups")
        row = store.blank_row()
        row.update({
            "record_type": store.CAMPAIGN,
            "campaign_id": "CMP_0001",
            "name": "קמפיין ראשון",
            "advertiser": BOOKED_CLIENT,
            "status": "active",
            "starts_on": "2026-09-01",
            "ends_on": "2026-09-30",
        })
        store.write_frame(store.append(store.load_frame(), row))
        return {"tree": client_tree(), "money": board()}


@pytest.fixture(scope="module")
def probes(payload):
    """The leader, a client far down the ranking, and the booked client."""
    ranked = payload["money"]["advertisers"]
    assert len(ranked) >= 20, "the ledger is smaller than the ranking this file measures"
    return [ranked[0]["advertiser"], ranked[19]["advertiser"], BOOKED_CLIENT]


def _run(tmp_path: Path, payload: dict, probes: list[str], source: str) -> dict:
    """Press the record's money control on each probe, against one helper version."""
    module = tmp_path / "helpers.mjs"
    module.write_text(source, encoding="utf-8")
    stub = tmp_path / "format.mjs"
    stub.write_text(STUB, encoding="utf-8")
    harness = tmp_path / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    tree_file = tmp_path / "tree.json"
    tree_file.write_text(json.dumps(payload["tree"], ensure_ascii=False), encoding="utf-8")
    money_file = tmp_path / "money.json"
    money_file.write_text(json.dumps(payload["money"], ensure_ascii=False), encoding="utf-8")
    result = subprocess.run(
        [
            _node(),
            str(harness),
            str(module),
            str(stub),
            str(tree_file),
            str(money_file),
            json.dumps(probes, ensure_ascii=False),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.fixture(scope="module")
def shipped() -> str:
    source = HELPERS.read_text(encoding="utf-8")
    assert GUARD in source, "the guard under test is not in the shipped helper any more"
    return source


def test_the_control_opens_the_rows_behind_this_clients_own_figures(tmp_path, payload, probes, shipped):
    """The press lands on a drill, and the drill is this client's money."""
    seen = _run(tmp_path, payload, probes, shipped)["pressed"]
    for name in probes[:2]:
        pressed = seen[name]
        assert pressed["view"] == "money"
        assert pressed["client"] == name, "the record stays open beside the money it opened"
        assert pressed["board"]["drills"] == 1, "the press must land on a drill, not on the ranking"
        assert pressed["board"]["rankedRows"] == 0
        assert pressed["board"]["opened"]["name"] == name
        assert pressed["board"]["opened"]["gross"] == pressed["record"]["gross"]
        assert pressed["board"]["opened"]["net"] == pressed["record"]["net"]
        assert pressed["board"]["opened"]["spots"] == pressed["record"]["spots"]


def test_a_client_down_the_ranking_gets_its_own_money_not_the_leaders(tmp_path, payload, probes, shipped):
    """The sharpest form of the defect: rank 20 must not answer with rank 1."""
    seen = _run(tmp_path, payload, probes, shipped)["pressed"]
    leader, twentieth = probes[0], probes[1]
    assert leader != twentieth
    pressed = seen[twentieth]
    assert pressed["board"]["leader"] == leader, "rank 1 is the row the old behaviour showed"
    assert pressed["board"]["opened"]["name"] == twentieth
    assert pressed["board"]["opened"]["gross"] != seen[leader]["record"]["gross"]


def test_a_client_with_no_priced_spot_opens_nothing_and_keeps_its_reason(tmp_path, payload, probes, shipped):
    """No row in the ledger means no navigation, so the record's reason stays."""
    result = _run(tmp_path, payload, probes, shipped)
    pressed = result["pressed"][BOOKED_CLIENT]
    assert pressed["record"] is not None, "a booked client must be in the tree even with no money"
    assert pressed["record"]["gross"] is None
    assert pressed["record"]["spots"] is None
    assert pressed["record"]["reason_en"], "the record must say why there is no money"
    assert pressed["record"]["reason_he"], "and it must say it in Hebrew"
    assert pressed["target"] is None
    assert pressed["view"] == "clients", "the reader must not be moved to a board that cannot hold the row"
    assert pressed["client"] == BOOKED_CLIENT
    assert pressed["board"]["drills"] == 0
    ledger = {row["advertiser"] for row in payload["money"]["advertisers"]}
    assert BOOKED_CLIENT not in ledger


def test_the_board_still_drills_every_way_it_did(tmp_path, payload, probes, shipped):
    """The regression bar: the key moved out of the board, the drilling did not."""
    walk = dict(_run(tmp_path, payload, probes, shipped)["walk"])
    assert walk["group tab campaigns"]["drills"] == 0
    assert walk["group tab campaigns"]["rankedRows"] == len(payload["money"]["campaigns"])
    assert walk["row click"]["drills"] == 1
    assert walk["row click"]["opened"]["name"] == payload["money"]["campaigns"][0]["campaign"]
    assert walk["break chip"]["group"] == "breaks"
    assert walk["break chip"]["opened"]["name"] == str(payload["money"]["campaigns"][0]["breaks"][0])
    assert walk["back"]["drills"] == 0
    assert walk["back"]["rankedRows"] == len(payload["money"]["breaks"])


def test_without_the_guard_the_booked_client_lands_on_a_board_that_cannot_hold_it(tmp_path, payload, probes, shipped):
    """Proof the tests above bite. Remove the guard and the press navigates again."""
    mutant = shipped.replace(GUARD, WITHOUT_GUARD)
    assert mutant != shipped
    pressed = _run(tmp_path, payload, probes, mutant)["pressed"][BOOKED_CLIENT]
    assert pressed["target"] is not None
    assert pressed["view"] == "money"
    assert pressed["board"]["opened"] is None, "the row is not in the ledger, so nothing can open"
    assert pressed["board"]["leader"] != BOOKED_CLIENT


def test_the_board_takes_the_open_row_as_a_prop_and_keeps_no_key_of_its_own():
    """The cause of the defect, held shut in the shipped component."""
    source = BOARD.read_text(encoding="utf-8")
    assert "useState" not in source, "a key the board owns is a key no caller can set"
    assert "drill = NO_DRILL, onDrill" in source
    assert "const openKey = drill.key || '';" in source
    assert "onDrill({ group, key: String(row[definition.field]) })" in source


def test_the_workspace_owns_the_drill_and_refuses_a_row_that_is_not_there():
    """The other half, which needs React and so is pinned in the source."""
    source = WORKSPACE.read_text(encoding="utf-8")
    assert "const [drill, setDrill] = useState(NO_DRILL);" in source
    assert "const target = moneyTarget(rows.find((row) => row.advertiser === advertiser));" in source
    assert "if (!target) {\n      return;\n    }" in source
    assert "drill={drill}" in source
    assert "onDrill={setDrill}" in source


def test_the_record_offers_the_control_only_where_there_are_rows_behind_it():
    """One call site for the control, inside the branch that has a ledger row."""
    source = RECORD.read_text(encoding="utf-8")
    assert "const opensRows = hasLedgerRow(client);" in source
    assert source.count("onOpenMoney(client.advertiser)") == 1
    assert "Open every row behind these figures" in source
    assert "No priced spot for this client, so there is no row to open" in source
    assert "אין תשדיר מתומחר ללקוח הזה, ולכן אין שורה לפתוח" in source
    opens = source.index("{opensRows ? (")
    assert opens < source.index("onOpenMoney(client.advertiser)") < source.index("clients-money-open empty")
