"""P4: the delivery ledger this destination computes reaches the screen, whole.

The measured defect. ``GET /api/clients/campaigns`` returns a real per-campaign
delivery ledger derived from the traffic log: for CMP_D001, one sourced day of a
seven day flight, two aired spots over fifty seconds, 8.7 planned rating points,
6,600 shekels, one spot removed by a rule, the six unsourced dates named, a floor
sentence, the counted-as-of instant with its basis, and a rating and budget
progress each carrying a percent and the word ``floor``. Not one figure of it was
on any operator surface. The one column that named it, the Delivered column of
every flight row, rendered a hard-coded literal, the word unknown, with no
dependence on the payload at all, and the reason and path that were supposed to
stand under it were gated on ``!delivery.available`` and so resolved to two empty
paragraphs the moment the ledger became available.

Four rounds of a fix loop failed on this piece because each round fixed the one
site it was pointed at and the same class survived at the adjacent one. So this
file asserts the class, not the site.

The **structural** half holds the seam shut: no surface may format a delivery
figure by hand. Every raw ledger field is readable in exactly two files, the
helper and the component, and every surface that renders a figure renders the
basis beside it. A fifth surface added tomorrow that prints ``aired.spots``
itself fails here before it can ship.

The **rendered** half runs the shipped components against the real payload and
reads the HTML: the figure, the state word, the floor word, the counted instant,
the named source file and the named unsourced dates all have to be in it, and a
percent whose state is ``floor`` has to read as a floor. The last test puts the
hard-coded literal back and asserts the pass disappears, so no pass here can be
vacuous.
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
FLIGHTS = CLIENTS / "CampaignFlights.jsx"

# The two files allowed to know what a raw ledger field is called. Everything
# else on the destination reads delivery through them, which is what makes "a
# figure without its state" unwritable rather than merely absent today.
DELIVERY_OWNERS = {"delivery-helpers.js", "DeliveryState.jsx"}

# Raw ledger fields. A surface that names one of these is formatting a delivery
# figure by hand, which is exactly how the state and the basis get dropped.
RAW_FIELDS = (
    "rating_points_planned",
    "spend_ils",
    "spots_dropped_by_rule",
    "sourced_days",
    "flight_days",
    "rating_progress",
    "budget_progress",
    "broadcast_date",
    "dropped_rule_id",
    "floor_note",
    "rating_basis",
    "spend_basis",
    "path_forward",
    "delivery.aired",
    "delivery.scheduled",
    "delivery.unknown",
    "delivery.available",
)

# Sentences the shipped code asserted about this repository that stopped being
# true the day the ledger landed. They are named so they cannot drift back in.
FALSE_CLAIMS = (
    "nothing in this repository observes delivery",
    "what a flight delivered is unknown",
    "refuses to render the third column",
    "delivered: unknown",
    "nothing aired yet",
)

# The cell as it shipped: a literal with no dependence on the payload.
HARD_CODED_CELL = """                  <td>
                    <DeliveryCell
                      delivery={delivery}
                      window={{ starts_on: flight.starts_on, ends_on: flight.ends_on }}
                      vocabulary={airStates}
                      locale={locale}
                    />
                  </td>"""
MUTANT_CELL = """                  <td><span className="clients-unknown">{pageText(locale, 'unknown', 'לא ידוע')}</span></td>"""

ENTRY = """
export {{ default as CampaignBoard }} from '{board}';
export {{ default as CampaignDetail }} from '{detail}';
export {{ default as ClientRecord }} from '{record}';
export * as helpers from '{helpers}';
export * as delivery from '{delivery}';
"""

RENDER = """
import { createRequire, isBuiltin, registerHooks } from 'node:module';
import { pathToFileURL } from 'node:url';
import fs from 'node:fs';

const [entry, outDir, boardFile, treeFile, outFile, flightsSource] = process.argv.slice(2);
const require_ = createRequire('APP_PACKAGE');
const MAP = new Map();
function fromApp(specifier) {
  if (!MAP.has(specifier)) {
    try {
      const found = require_.resolve(specifier);
      MAP.set(specifier, found.startsWith('/') ? pathToFileURL(found).href : '');
    } catch {
      MAP.set(specifier, '');
    }
  }
  return MAP.get(specifier);
}
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (!isBuiltin(specifier) && !/^[./]|^node:|^file:/.test(specifier)) {
      const found = fromApp(specifier);
      if (found) {
        return { url: found, shortCircuit: true };
      }
    }
    return nextResolve(specifier, context);
  },
});

const { build } = await import('rolldown');
await build({
  input: entry,
  external: (id) => !/^[./]/.test(id),
  output: { dir: outDir, format: 'esm', entryFileNames: 'surface.mjs' },
  resolve: { extensions: ['.js', '.jsx'] },
  logLevel: 'silent',
  plugins: [{
    name: 'flights-under-test',
    // A stylesheet is not markup and this bundler refuses to carry one. Every
    // style import is redirected to one empty JS module, so a render measures
    // what the component says rather than dying on how it looks.
    resolveId(source) {
      return /\\.css$/.test(source) ? { id: '\\0no-styles.js', moduleSideEffects: false } : null;
    },
    // Vite's import.meta.env does not exist in node, and the API module on this
    // destination reads it the moment it is imported. It is pointed at a plain
    // object so a render measures the surface rather than the build tool.
    transform(code) {
      return code.includes('import.meta.env')
        ? { code: code.split('import.meta.env').join('globalThis.__viteEnv') }
        : null;
    },
    load(id) {
      if (id === '\\0no-styles.js') {
        return 'export default {};';
      }
      return id === 'FLIGHTS_PATH' ? fs.readFileSync(flightsSource, 'utf8') : null;
    },
  }],
});

globalThis.__viteEnv = {};
const React = (await import('react')).default;
const { renderToStaticMarkup: markup } = await import('react-dom/server');
const { CacheProvider } = await import('@emotion/react');
const cacheModule = await import('@emotion/cache');
const createCache = cacheModule.default.default || cacheModule.default;
const cache = createCache({ key: 'kairos-test' });
const renderToStaticMarkup = (element) => markup(React.createElement(CacheProvider, { value: cache }, element));
const surface = await import(pathToFileURL(`${outDir}/surface.mjs`).href);
const board = JSON.parse(fs.readFileSync(boardFile, 'utf8'));
const tree = JSON.parse(fs.readFileSync(treeFile, 'utf8'));

// The campaign the diagnosis was measured on: delivery available, real aired
// spots, and unknown days still in its flight, which is the tri-state in one row.
const probe = board.campaigns.find((row) => row.delivery
  && row.delivery.available
  && row.delivery.aired.spots > 0
  && row.delivery.unknown.days > 0) || null;

const gate = { canEdit: true, reason: '' };
const boards = {};
['he', 'en'].forEach((locale) => {
  boards[locale] = renderToStaticMarkup(React.createElement(surface.CampaignBoard, {
    board,
    locale,
    notify: () => {},
    gate,
    agencies: {},
    openCampaignId: '',
    onOpened: () => {},
    onOnboard: () => {},
    onOpenClient: () => {},
    onOpenAgency: () => {},
    onReload: () => {},
  }));
});

const details = {};
if (probe) {
  ['he', 'en'].forEach((locale) => {
    details[locale] = renderToStaticMarkup(React.createElement(surface.CampaignDetail, {
      campaign: probe,
      board,
      options: null,
      optionsError: '',
      locale,
      canEdit: false,
      notify: () => {},
      onChanged: () => {},
    }));
  });
}

// The same ledger on the client record, reached the way the workspace reaches
// it: one index off the board's own read, never a second read of its own.
const index = surface.delivery.campaignDeliveryIndex(board);
const rows = surface.helpers.flattenClients(
  tree.agencies || [],
  tree.unlinked || [],
  tree.clients_booked_without_spots || [],
);
const owner = probe ? rows.find((row) => row.advertiser === probe.advertiser) : null;
const record = owner ? renderToStaticMarkup(React.createElement(surface.ClientRecord, {
  client: owner,
  rows,
  locale: 'he',
  basis: tree.basis,
  delivery: board.delivery,
  deliveryByCampaign: index,
  airStates: board.delivery.air_state_vocabulary,
  statuses: board.status_vocabulary,
  goalWords: board.goal_kind_vocabulary,
  ruleRows: [],
  ledgerCampaigns: [],
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
  details,
  record,
  probe: probe ? probe.campaign_id : '',
  indexed: Object.keys(index).length,
}), 'utf8');
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped components cannot be rendered here")
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


def sources() -> list[Path]:
    return sorted(path for path in CLIENTS.iterdir() if path.suffix in {".js", ".jsx"})


# --------------------------------------------------------------------------
# The structural half: the seam that makes a bare figure unwritable
# --------------------------------------------------------------------------

def test_no_surface_reads_a_raw_delivery_field_for_itself():
    """The class, not the site. Two files know the ledger's field names."""
    offenders = {}
    for path in sources():
        if path.name in DELIVERY_OWNERS:
            continue
        text = path.read_text(encoding="utf-8")
        hits = [field for field in RAW_FIELDS if field in text]
        if hits:
            offenders[path.name] = hits
    assert not offenders, (
        "these surfaces format a delivery figure by hand, which is how the state "
        f"and the basis get dropped: {offenders}"
    )


def test_every_surface_that_prints_a_figure_prints_the_basis():
    """A figure and the basis it was counted on ship together or not at all."""
    printing = [path for path in sources() if "<DeliveryCell" in path.read_text(encoding="utf-8")]
    assert {"CampaignBoard.jsx", "CampaignFlights.jsx", "ClientRecord.jsx"} <= {path.name for path in printing}
    for path in printing:
        text = path.read_text(encoding="utf-8")
        assert "<DeliveryBasis" in text, f"{path.name} prints a delivery figure and no basis for it"


def test_the_reason_and_the_path_are_no_longer_gated_on_the_ledger_being_absent():
    """The exact gate that turned an honest empty state into two blank lines."""
    for path in sources():
        text = path.read_text(encoding="utf-8")
        assert "!delivery.available ?" not in text, (
            f"{path.name} still gates the delivery sentences on the ledger being unavailable, "
            "which renders nothing at all once it is available"
        )


def test_no_shipped_comment_asserts_something_false_about_this_repository():
    """Two comments claimed this product observes no delivery. It does."""
    for path in sources():
        text = path.read_text(encoding="utf-8").lower()
        for claim in FALSE_CLAIMS:
            assert claim.lower() not in text, f"{path.name} still says: {claim}"


def test_the_weekday_line_claims_a_refusal_only_when_one_would_happen():
    """check_weekday_scope returns early on a zero percent, so the line must too.

    Both screens that carry this pair held their own copy of the sentence, both
    copies were wrong the same way, and the correction had to be made twice. The
    sentence lives in one module now and neither screen may hold a copy.
    """
    shared = (CLIENTS / "weekday-scope-helpers.js").read_text(encoding="utf-8")
    assert "export function wouldRefuse(selected, percent)" in shared
    assert "Number.isFinite(amount) && amount !== 0" in shared
    no_refusal = "Nothing is refused, because there is no discount percent to give a day to."
    assert no_refusal in shared
    assert "דבר אינו נדחה" in shared
    for name in ("CampaignTerms.jsx", "OnboardClientFlow.jsx"):
        text = (CLIENTS / name).read_text(encoding="utf-8")
        assert "from './weekday-scope-helpers'" in text, f"{name} does not use the shared sentence"
        assert "function weekdayCoverage" not in text, f"{name} still holds its own copy"
        assert no_refusal not in text


def test_no_source_file_on_this_destination_is_over_the_cap():
    for path in sorted(CLIENTS.iterdir()):
        if path.suffix in {".js", ".jsx", ".css"}:
            lines = len(path.read_text(encoding="utf-8").splitlines())
            assert lines <= 450, f"{path.name} is {lines} lines"


# --------------------------------------------------------------------------
# The rendered half: what an operator actually reads
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def payload() -> dict:
    """The real board payload and the real client tree, read only."""
    from kairos_api.campaigns_api import list_campaigns
    from kairos_api.campaigns_read_clients import client_tree

    board = list_campaigns()
    if not board["delivery"]["available"]:
        pytest.skip("no delivery ledger on disk, so nothing about it can be measured on screen")
    sourced = [
        row for row in board["campaigns"]
        if row["delivery"]["available"] and row["delivery"]["aired"]["spots"] > 0
        and row["delivery"]["unknown"]["days"] > 0
    ]
    if not sourced:
        pytest.skip("no campaign on disk carries both aired spots and unknown days")
    return {"board": board, "tree": client_tree(), "probe": sourced[0]}


def _plain(value):
    """Numpy scalars cross the wire as numbers, never as their repr.

    ``default=str`` turned a spot count into the string "2", which still
    rendered and still passed a substring assertion while every numeric
    comparison inside the component silently changed meaning.
    """
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def _render(tmp_path: Path, payload: dict, flights_source: str) -> dict:
    node = _node()
    work = tmp_path / "surface"
    work.mkdir(parents=True, exist_ok=True)
    source = work / "flights-under-test.jsx"
    source.write_text(flights_source, encoding="utf-8")
    entry = work / "entry.mjs"
    entry.write_text(
        ENTRY.format(
            board=(CLIENTS / "CampaignBoard.jsx").as_posix(),
            detail=(CLIENTS / "CampaignDetail.jsx").as_posix(),
            record=(CLIENTS / "ClientRecord.jsx").as_posix(),
            helpers=(CLIENTS / "clients-money-helpers.js").as_posix(),
            delivery=(CLIENTS / "delivery-helpers.js").as_posix(),
        ),
        encoding="utf-8",
    )
    board_file = work / "board.json"
    board_file.write_text(json.dumps(payload["board"], ensure_ascii=False, default=_plain), encoding="utf-8")
    tree_file = work / "tree.json"
    tree_file.write_text(json.dumps(payload["tree"], ensure_ascii=False, default=_plain), encoding="utf-8")
    out_file = work / "rendered.json"
    script = work / "render.mjs"
    script.write_text(
        RENDER
        .replace("APP_PACKAGE", (APP / "package.json").as_posix())
        .replace("FLIGHTS_PATH", FLIGHTS.as_posix()),
        encoding="utf-8",
    )
    done = subprocess.run(
        [
            node,
            # the shell moved bidi.jsx and dates.js under src/shell; this hook
            # resolves both to the real modules so the bundle under test can import them.
            "--import", str(ROOT / "tests" / "js" / "shell-resolver.mjs"),
            str(script), str(entry), str(work / "out"), str(board_file), str(tree_file),
            str(out_file), str(source),
        ],
        capture_output=True,
        text=True,
        cwd=str(APP),
        check=False,
    )
    if done.returncode != 0:
        pytest.fail(f"the shipped surface could not be rendered: {done.stderr[-3000:]}")
    return json.loads(out_file.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rendered(tmp_path_factory, payload) -> dict:
    return _render(tmp_path_factory.mktemp("delivery"), payload, FLIGHTS.read_text(encoding="utf-8"))


def _spots(count, locale):
    """The exact phrase the shipped cell prints for a counted floor."""
    unit = "spot" if count == 1 else "spots"
    if locale == "en":
        return f"at least {count} {unit}"
    return f"\u05dc\u05e4\u05d7\u05d5\u05ea \u2066{count}\u2069 \u05ea\u05e9\u05d3\u05d9\u05e8\u05d9\u05dd"


def test_the_board_prints_what_was_counted_instead_of_the_word_unknown(payload, rendered):
    """The headline defect, measured on the surface an operator opens.

    Not "the digit 2 is somewhere in the page": the exact sentence the cell is
    supposed to print, in both languages, for the campaign the ledger really
    holds two aired spots for.
    """
    counted = payload["probe"]["delivery"]["aired"]["spots"]
    for locale, html in rendered["boards"].items():
        assert "clients-delivered" in html, f"{locale}: the board has no delivered column at all"
        assert "clients-air-state" in html, f"{locale}: a figure is printed with no state beside it"
        assert _spots(counted, locale) in html, (
            f"{locale}: the board does not print what the ledger counted, as a floor"
        )


def test_the_board_names_the_instant_the_split_was_taken_at(payload, rendered):
    """A number without its basis is the defect class this whole piece exists to kill."""
    as_of = payload["board"]["delivery"]["as_of"]
    assert as_of["instant"], "the ledger under test carries no instant, so this cannot be measured"
    assert as_of["instant"] in rendered["boards"]["en"]
    assert as_of["instant"] in rendered["boards"]["he"]
    assert "Counted as of" in rendered["boards"]["en"]
    assert "\u05e0\u05e1\u05e4\u05e8 \u05e0\u05db\u05d5\u05df \u05dc" in rendered["boards"]["he"]
    if as_of["basis"]:
        assert as_of["basis"] in rendered["boards"]["he"], (
            "the recorded basis is dropped in Hebrew, which is the caveat going missing"
        )


def test_the_flight_row_carries_the_state_the_figure_and_the_named_missing_days(payload, rendered):
    """Tri-state on one row: what was counted, and which days nobody counted."""
    ledger = payload["probe"]["delivery"]
    assert ledger["unknown"]["dates"], "no unknown day on this campaign, so the guard needs re-aiming"
    for locale, html in rendered["details"].items():
        assert "clients-delivered" in html, f"{locale}: the flights table prints no delivery state"
        for date in ledger["unknown"]["dates"]:
            assert date in html, f"{locale}: the unsourced day {date} is not named on the surface"
        assert str(ledger["unknown"]["days"]) in html


def test_the_flight_row_names_the_file_the_count_was_read_from(payload, rendered):
    """The path back to the evidence, printed beside the figure it produced."""
    files = {
        str(day["source_file"]) for day in payload["probe"]["delivery"]["days"]
        if day.get("source_file")
    }
    assert files, "the ledger under test names no source file, so this cannot be measured"
    for locale, html in rendered["details"].items():
        for name in files:
            assert name in html, f"{locale}: the source file {name} is not named on the surface"


def test_a_floor_percent_is_labelled_a_floor_and_never_a_finished_figure(payload, rendered):
    """The one figure most easily read as complete, held to what it really is."""
    ledger = payload["probe"]["delivery"]
    measured = 0
    for key in ("rating_progress", "budget_progress"):
        block = ledger[key]
        if block["percent"] is None:
            continue
        measured += 1
        assert block["state"] == "floor", "this payload no longer reports a floor, so the guard needs re-aiming"
        shown = f"{block['percent']:.2f}".rstrip("0").rstrip(".")
        assert shown in rendered["details"]["en"], f"{key} is on the payload and its percent is not on the screen"
        assert shown in rendered["details"]["he"]
    assert measured, "no progress percent on this campaign, so the guard needs re-aiming"
    assert "at least" in rendered["details"]["en"] and "floor, not a total" in rendered["details"]["en"]
    assert "\u05dc\u05e4\u05d7\u05d5\u05ea" in rendered["details"]["he"]
    assert "\u05e8\u05e3 \u05ea\u05d7\u05ea\u05d5\u05df, \u05dc\u05d0 \u05e1\u05db\u05d5\u05dd" in rendered["details"]["he"]


def test_the_client_record_reads_the_same_ledger_the_board_reads(payload, rendered):
    """One read, two surfaces. Two reads of the same ledger could drift; one cannot."""
    assert rendered["indexed"] == len(payload["board"]["campaigns"])
    record = rendered["record"]
    assert record, "no client on the tree owns the campaign under test"
    assert "\u05e1\u05d5\u05e4\u05e7: \u05dc\u05d0 \u05d9\u05d3\u05d5\u05e2" not in record, (
        "the hard-coded literal is still on the client record"
    )
    assert "clients-delivered" in record and "clients-air-state" in record
    assert _spots(payload["probe"]["delivery"]["aired"]["spots"], "he") in record


def test_putting_the_hard_coded_literal_back_brings_the_defect_back(tmp_path, payload):
    """Proof this file bites: the measured defect, restored, fails the tests above."""
    shipped = FLIGHTS.read_text(encoding="utf-8")
    assert HARD_CODED_CELL in shipped, "the shipped cell moved, so this mutation needs re-aiming"
    mutant = shipped.replace(HARD_CODED_CELL, MUTANT_CELL)
    assert mutant != shipped
    html = _render(tmp_path, payload, mutant)["details"]["he"]
    assert "\u05dc\u05d0 \u05d9\u05d3\u05d5\u05e2</span></td>" in html, (
        "this is exactly what was measured on the shipped bundle"
    )
    assert _spots(payload["probe"]["delivery"]["aired"]["spots"], "he") not in html.split("clients-flight-table")[-1], (
        "with the literal back, the flight row states nothing the ledger holds"
    )
    assert payload["probe"]["delivery"]["unknown"]["dates"][0] in html, (
        "the basis block survives the mutation, so the cell is what this proves"
    )
