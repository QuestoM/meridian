"""P4: the sentence over the money answers the question the board is asked.

The measured defect. The money board's headline was one hard-coded question,
"which client delivered the most, gross and net of agency rebates", and the line
under it printed whatever the active grouping ranks. Measured on screen in both
languages, three of the four groupings put a false sentence over a true number:
by agency it answered with יוניברסל, an agency; by campaign with a campaign; by
break with 22:03:06, a clock. An analyst reading it would report an agency as
the top client, on the one surface JS-9 exists to answer.

Two more dead ends on the client record are measured in the same run, because
they are the same class of defect: a fact printed rather than a control. The
agency line named the agency whose full record is one tab away, and the observed
campaign names named campaigns that are rows with real money on the board's own
campaign grouping, and neither one opened anything.

Everything below renders the shipped components against the real priced ledger
and the real client tree. The last test mutates the fix away and asserts the
false sentence comes back, so a pass here can never be vacuous.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
CLIENTS = APP / "src" / "clients"
BOARD = CLIENTS / "MoneyBoard.jsx"

# The shipped line, and the mutation that puts the old defect back.
QUESTION_LINE = "          {pageText(locale, definition.questionEn, definition.questionHe)}"
ONE_QUESTION = "          {pageText(locale, 'Which client delivered the most, gross and net of agency rebates', 'איזה לקוח סיפק הכי הרבה, ברוטו ונטו אחרי רבייט הסוכנות')}"

GROUPS = ("advertisers", "agencies", "campaigns", "breaks")
QUESTIONS_HE = {
    "advertisers": "איזה לקוח סיפק הכי הרבה",
    "agencies": "איזו סוכנות סיפקה הכי הרבה",
    "campaigns": "איזה קמפיין סיפק הכי הרבה",
    "breaks": "איזה ברייק סיפק הכי הרבה",
}
QUESTIONS_EN = {
    "advertisers": "Which client delivered the most",
    "agencies": "Which agency delivered the most",
    "campaigns": "Which campaign delivered the most",
    "breaks": "Which break delivered the most",
}

ENTRY = """
export {{ default as MoneyBoard }} from '{board}';
export {{ default as ClientRecord }} from '{record}';
export * as helpers from '{helpers}';
"""

RENDER = """
import { createRequire, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';

const [entry, outDir, moneyFile, treeFile, clientName, outFile, boardSource] = process.argv.slice(2);
const require_ = createRequire('APP_PACKAGE');
const MAP = {};
for (const bare of ['react', 'react/jsx-runtime', 'react-dom/server', 'lucide-react', 'rolldown']) {
  MAP[bare] = pathToFileURL(require_.resolve(bare)).href;
}
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (MAP[specifier]) {
      return { url: MAP[specifier], shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

const { build } = await import('rolldown');
await build({
  input: entry,
  external: ['react', 'react-dom', 'react/jsx-runtime', 'lucide-react'],
  output: { dir: outDir, format: 'esm', entryFileNames: 'surface.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  logLevel: 'silent',
  plugins: [{
    name: 'money-board-under-test',
    load(id) {
      return id === 'BOARD_PATH' ? fs.readFileSync(boardSource, 'utf8') : null;
    },
  }],
});

const React = (await import('react')).default;
const { renderToStaticMarkup } = await import('react-dom/server');
const surface = await import(pathToFileURL(`${outDir}/surface.mjs`).href);
const money = JSON.parse(fs.readFileSync(moneyFile, 'utf8'));
const tree = JSON.parse(fs.readFileSync(treeFile, 'utf8'));

const board = (locale, group, key) => renderToStaticMarkup(React.createElement(surface.MoneyBoard, {
  money,
  locale,
  drill: { group, key: key || '' },
  onDrill: () => {},
  onOpenClient: () => {},
}));

const boards = {};
['he', 'en'].forEach((locale) => {
  ['advertisers', 'agencies', 'campaigns', 'breaks'].forEach((group) => {
    boards[`${locale}:${group}`] = board(locale, group, '');
  });
});

// One client with removed spots, opened, so the rows behind the figure and the
// rows a rule removed are both on screen.
const removed = money.advertisers.find((row) => row.dropped_by_frequency > 0);
const detail = board('he', 'advertisers', removed ? String(removed.advertiser) : '');

const rows = surface.helpers.flattenClients(
  tree.agencies || [],
  tree.unlinked || [],
  tree.clients_booked_without_spots || [],
);
const client = rows.find((row) => row.advertiser === clientName) || null;
const record = client ? renderToStaticMarkup(React.createElement(surface.ClientRecord, {
  client,
  rows,
  locale: 'he',
  basis: tree.basis,
  delivery: null,
  statuses: [],
  goalWords: [],
  ruleRows: [],
  ledgerCampaigns: surface.helpers.ledgerCampaignKeys(money),
  onClose: () => {},
  onStep: () => {},
  onOpenMoney: () => {},
  onCreateRule: () => {},
  onAddSpelling: () => {},
  onOpenRuleCard: () => {},
  onBookCampaign: () => {},
  onOpenAgency: () => {},
  onOpenCampaignMoney: () => {},
})) : '';

fs.writeFileSync(outFile, JSON.stringify({
  boards,
  detail,
  record,
  leaders: Object.fromEntries(['advertisers', 'agencies', 'campaigns', 'breaks'].map((group) => {
    const field = { advertisers: 'advertiser', agencies: 'agency', campaigns: 'campaign', breaks: 'break_id' }[group];
    const ranked = [...(money[group] || [])].sort((left, right) => right.gross - left.gross);
    return [group, ranked.length ? String(ranked[0][field]) : ''];
  })),
  removed: removed ? String(removed.advertiser) : '',
}), 'utf8');
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped component cannot be rendered here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so react cannot be resolved from the app")
    for package in ("react-dom", "rolldown"):
        if not (APP / "node_modules" / package).is_dir():
            pytest.skip(f"the dashboard's {package} is not installed, so nothing can be rendered")
    return found


@pytest.fixture(scope="module")
def payload() -> dict:
    """The real priced ledger and the real client tree, read only."""
    from kairos_api.campaigns_read_clients import client_tree
    from kairos_api.campaigns_read_money import board

    money = board()
    if not money["available"]:
        pytest.skip("no priced day on disk, so the money board cannot be measured")
    return {"money": money, "tree": client_tree()}


@pytest.fixture(scope="module")
def probe(payload) -> str:
    """A client that buys through an agency and has campaigns on air."""
    for agency in payload["tree"]["agencies"]:
        for client in agency["clients"]:
            if client["observed_campaigns"]:
                return client["advertiser"]
    pytest.skip("no client on the tree carries an observed campaign")
    return ""


def _render(tmp_path: Path, payload: dict, probe: str, board_source: str) -> dict:
    node = _node()
    work = tmp_path / "surface"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "board-under-test.jsx"
    source.write_text(board_source, encoding="utf-8")
    entry = work / "entry.mjs"
    entry.write_text(
        ENTRY.format(
            board=BOARD.as_posix(),
            record=(CLIENTS / "ClientRecord.jsx").as_posix(),
            helpers=(CLIENTS / "clients-money-helpers.js").as_posix(),
        ),
        encoding="utf-8",
    )
    script = work / "render.mjs"
    script.write_text(
        RENDER.replace("APP_PACKAGE", (APP / "package.json").as_posix()).replace("BOARD_PATH", BOARD.as_posix()),
        encoding="utf-8",
    )
    money_file = work / "money.json"
    money_file.write_text(json.dumps(payload["money"], ensure_ascii=False), encoding="utf-8")
    tree_file = work / "tree.json"
    tree_file.write_text(json.dumps(payload["tree"], ensure_ascii=False), encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run(
        [node, str(script), str(entry), str(work / "bundle"), str(money_file),
         str(tree_file), probe, str(out), str(source)],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(work),
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def shipped() -> str:
    source = BOARD.read_text(encoding="utf-8")
    assert QUESTION_LINE in source, "the headline under test is not in the shipped board any more"
    return source


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, payload, probe, shipped) -> dict:
    return _render(tmp_path_factory.mktemp("shipped"), payload, probe, shipped)


def test_each_grouping_asks_its_own_question_in_hebrew(rendered):
    """The defect, in the language the product is read in."""
    for group in GROUPS:
        html = rendered["boards"][f"he:{group}"]
        assert QUESTIONS_HE[group] in html, group
        for other, wording in QUESTIONS_HE.items():
            if other != group:
                assert wording not in html, f"{group} still asks {other}'s question"


def test_each_grouping_asks_its_own_question_in_english(rendered):
    for group in GROUPS:
        html = rendered["boards"][f"en:{group}"]
        assert QUESTIONS_EN[group] in html, group
        for other, wording in QUESTIONS_EN.items():
            if other != group:
                assert wording not in html, f"{group} still asks {other}'s question"


def test_the_answer_under_each_question_is_that_grouping_leader(rendered):
    """The number was always right. This is the sentence and the number together."""
    for group in GROUPS:
        html = rendered["boards"][f"he:{group}"]
        leader = rendered["leaders"][group]
        assert leader, group
        question = html.index(QUESTIONS_HE[group])
        answer = html.index(leader, question)
        assert answer > question, f"{group} names its leader after the question it answers"


def test_the_agency_leader_is_never_presented_as_a_client(rendered, payload):
    """The sharpest form: יוניברסל is an agency and is listed as one two tabs away."""
    agency = rendered["leaders"]["agencies"]
    assert agency in {row["agency"] for row in payload["money"]["agencies"]}
    html = rendered["boards"]["he:agencies"]
    head = html[: html.index(agency)]
    assert QUESTIONS_HE["agencies"] in head
    assert QUESTIONS_HE["advertisers"] not in head


def test_a_removed_spot_says_why_in_words_and_not_in_a_log_line(rendered, payload):
    """DEFAULT_ONE_PER_BREAK and max_per_break=1 are a log. This is the sentence."""
    assert rendered["removed"], "no client on this day has a spot a rule removed"
    html = rendered["detail"]
    assert "כלל מתיר לכל היותר תשדיר אחד ללקוח בכל ברייק" in html
    assert "max_per_break=1" not in html, "the machine reason must not be the explanation"
    assert "כלל ⁦DEFAULT_ONE_PER_BREAK⁩" in html, "the rule id stays, labelled as an id"
    dropped = next(
        row for row in payload["money"]["dropped"]
        if row["kind"] == "frequency" and row["advertiser"] == rendered["removed"]
    )
    assert dropped["explanation_he"] in html
    assert dropped["reason"] not in html


def test_the_agency_named_on_a_client_record_opens_the_agency(rendered, payload, probe):
    """A fact printed is a dead end. The agency record is one tab away."""
    record = rendered["record"]
    client = next(
        row for agency in payload["tree"]["agencies"] for row in agency["clients"]
        if row["advertiser"] == probe
    )
    agency = next(
        agency for agency in payload["tree"]["agencies"]
        if client["advertiser"] in {row["advertiser"] for row in agency["clients"]}
    )
    assert f'<button type="button" class="clients-link">קונה דרך {agency["name"]}</button>' in record


def test_each_campaign_seen_on_air_opens_its_own_money(rendered, payload, probe):
    """Each name is a row with real money under the board's campaign grouping."""
    record = rendered["record"]
    client = next(
        row for agency in payload["tree"]["agencies"] for row in agency["clients"]
        if row["advertiser"] == probe
    )
    keys = {str(row["campaign"]) for row in payload["money"]["campaigns"]}
    assert client["observed_campaigns"], probe
    for name in client["observed_campaigns"]:
        assert str(name) in keys, "a name that opens nothing must not be offered as a control"
        assert f'<button type="button" class="clients-link">{name}</button>' in record


def test_the_client_source_is_a_word_and_not_a_column_value(rendered):
    """observed is how the store spells it. נצפה בנתונים is how a person reads it."""
    record = rendered["record"]
    assert "נצפה בנתונים" in record
    assert ">observed<" not in record


def test_one_question_over_four_rankings_fails_this_file(tmp_path, payload, probe, shipped):
    """The mutant: put the single hard-coded question back and the sentence lies."""
    mutant = shipped.replace(QUESTION_LINE, ONE_QUESTION)
    assert mutant != shipped
    seen = _render(tmp_path, payload, probe, mutant)
    html = seen["boards"]["he:agencies"]
    assert QUESTIONS_HE["advertisers"] in html, "the mutation must reproduce the defect"
    assert QUESTIONS_HE["agencies"] not in html
    agency = seen["leaders"]["agencies"]
    assert html.index(QUESTIONS_HE["advertisers"]) < html.index(agency)
