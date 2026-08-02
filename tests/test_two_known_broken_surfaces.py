"""Two surfaces the owner could already see on main, closed here.

1. THE DEMO MARKING. ``data/campaigns.csv`` carries ``is_demo``, ``demo_note``
   and ``data_source`` on every row, and none of it reached the screen: a grep
   for ``is_demo``/``isDemo`` across the whole dashboard source returned zero
   files before this change. Fifty one seeded demo campaigns presented as
   booked campaigns with nothing on screen saying otherwise.

2. THE ADVERTISER NAMES. ``GET /api/advertisers/identity`` already joins the
   rules store to the real observed names, and ``AdvertiserRecordsPanel.jsx``
   already fetches it and merges it onto each row as ``bound_advertiser``. But
   ``advertiser-name-helpers.js``, the module every card and the detail drawer
   actually render through, only ever looked at ``row.name`` and
   ``row.display_name``, which are empty on every seeded rules row. So the
   money joined and the name never did: a card still showed ``ADV_01``.

Both are frontend-only defects; the backend payloads already carried what was
needed. This file is source-level (grep-and-run-the-shipped-module), matching
the convention the rest of the P4 wave already uses, so a pass here can never
be vacuous: every assertion runs against the exact file that ships.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ROOT / "tv-break-dashboard" / "src" / "clients"


def read(name: str) -> str:
    return (CLIENTS / name).read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# The demo marking
# --------------------------------------------------------------------------

DEMO_BADGE_SURFACES = [
    "CampaignBoard.jsx",
    "CampaignDetail.jsx",
    "CampaignFlights.jsx",
    "ClientRecord.jsx",
]
DEMO_SURFACES = DEMO_BADGE_SURFACES + ["ClientTree.jsx"]


def test_demo_badge_exists_and_renders_only_when_the_backend_says_demo():
    """The one component every surface below renders demo through."""
    badge = read("DemoBadge.jsx")
    assert "demo.is_demo" in badge
    assert "return null" in badge, "a real campaign must render nothing, not an empty chip"


@pytest.mark.parametrize("name", DEMO_BADGE_SURFACES)
def test_every_campaign_surface_imports_and_uses_the_demo_badge(name: str):
    source = read(name)
    assert "DemoBadge" in source, f"{name} does not mark demo rows"


def test_the_client_tree_marks_demo_campaigns_without_the_shared_badge():
    """The tree shows a per-client tally rather than one campaign's own block
    (a row here is a client, not a campaign), so it uses the same amber
    ``clients-flag`` class directly instead of the single-campaign DemoBadge."""
    tree = read("ClientTree.jsx")
    assert "clients-flag" in tree
    assert "demoCampaigns" in tree


def test_the_campaign_board_header_states_the_demo_split_not_a_bare_count():
    board = read("CampaignBoard.jsx")
    assert "demo_count" in board
    assert "booked_count" in board
    assert "demo seed data" in board
    assert "נתוני זרע הדגמה" in board


def test_the_client_tree_header_counts_demo_campaigns_from_its_own_payload():
    """The tree read has no demo tally of its own; each campaign record inside
    it does, so the count is arithmetic on data already on the wire, not a
    number invented by this screen."""
    tree = read("ClientTree.jsx")
    assert "campaign.is_demo" in tree
    assert "demoTally" in tree


def test_the_client_record_marks_its_campaigns_and_flights_as_demo():
    record = read("ClientRecord.jsx")
    assert "demo={campaign.demo}" in record
    assert "demo={flight.demo}" in record


def test_no_demo_marking_file_is_over_the_line_cap():
    for name in DEMO_SURFACES + ["DemoBadge.jsx"]:
        path = CLIENTS / name
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= 450, f"{name} is {lines} lines"


# --------------------------------------------------------------------------
# The advertiser names
# --------------------------------------------------------------------------

NAMES = CLIENTS / "advertiser-name-helpers.js"

HARNESS = """
import fs from 'node:fs';
const mod = await import(process.argv[2]);
const [, , , rowsPath, outPath] = process.argv;
const rows = JSON.parse(fs.readFileSync(rowsPath, 'utf8'));
const out = rows.map((row) => ({
  advertiser_id: row.advertiser_id,
  displayName: mod.displayNameOf(row, 'he'),
  isUnnamed: mod.isUnnamed(row),
  showsRawIdLine: mod.showsRawIdLine(row, 'he'),
}));
fs.writeFileSync(outPath, JSON.stringify(out), 'utf8');
"""


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH")
    if not NAMES.is_file():
        pytest.skip("advertiser-name-helpers.js is not in this tree")
    return found


def _run(tmp_path: Path, rows: list[dict]) -> list[dict]:
    node = _node()
    work = tmp_path / "names"
    work.mkdir(parents=True, exist_ok=True)
    module = work / "names.mjs"
    module.write_text(NAMES.read_text(encoding="utf-8"), encoding="utf-8")
    harness = work / "harness.mjs"
    harness.write_text(HARNESS, encoding="utf-8")
    rows_path = work / "rows.json"
    rows_path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run(
        [node, str(harness), str(module), str(rows_path), str(out)],
        capture_output=True, text=True, check=False, cwd=str(work),
    )
    if result.returncode != 0:
        pytest.fail(f"the shipped module did not run: {result.stderr.strip()[:600]}")
    return json.loads(out.read_text(encoding="utf-8"))


def test_a_row_bound_only_through_the_identity_join_shows_its_real_name(tmp_path):
    """The measured defect: a rules row (ADV_01, empty name/display_name) that
    ``mergeRowWithIdentity`` attached ``bound_advertiser`` to must show that
    name, not the raw seed id, and must no longer count as unnamed."""
    rows = [{
        "advertiser_id": "ADV_01",
        "name": "",
        "display_name": "",
        "name_source": "",
        "bound_advertiser": "פריסבי",
    }]
    result = _run(tmp_path, rows)[0]
    assert result["displayName"] == "פריסבי"
    assert result["isUnnamed"] is False
    assert result["showsRawIdLine"] is True, "the raw id stays visible underneath, it just is not the headline"


def test_a_row_with_no_identity_join_and_no_stored_name_stays_honestly_unnamed(tmp_path):
    """A pricing row bound to nobody must never be dressed up as a name."""
    rows = [{
        "advertiser_id": "ADV_02",
        "name": "",
        "display_name": "",
        "name_source": "",
        "bound_advertiser": "",
    }]
    result = _run(tmp_path, rows)[0]
    assert result["displayName"] == "ADV_02"
    assert result["isUnnamed"] is True


def test_the_operators_own_stored_name_still_wins_over_the_identity_join(tmp_path):
    rows = [{
        "advertiser_id": "ADV_03",
        "name": "",
        "display_name": "השם שהמפעיל בחר",
        "name_source": "",
        "bound_advertiser": "השם מהזיהוי",
    }]
    result = _run(tmp_path, rows)[0]
    assert result["displayName"] == "השם שהמפעיל בחר"


def test_the_detail_drawer_resolves_its_headline_the_same_way_the_card_does():
    """The card and the drawer it opens must never disagree about the name."""
    drawer = read("AdvertiserDetailDrawer.jsx")
    assert "identityName" in drawer
    assert "operatorName(row) || identityName(row) || boundName(row)" in drawer
