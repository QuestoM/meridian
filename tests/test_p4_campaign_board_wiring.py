"""P4: the campaign board can amend what it books, and the row leads somewhere.

The endpoints for amending a booked campaign existed and were proven by
``test_p4_campaigns_api.py`` before this file, and none of them had a caller.
``clients-api.js`` exported ``createCampaign``, ``updateCampaign``, ``addFlight``
and ``removeFlight`` and a grep over the whole frontend tree found zero
importers of all four, so a campaign could be created and never changed. Two
smaller dead ends rode along on the same component: the client name was plain
text on a board where the identical name is a button everywhere else, and the
agency column printed ``AGY_10``, which is a database key and not something a
person calls anybody.

Two halves are asserted here, because either alone would be a false pass.

The **wiring** half pins the source: every write this destination exports has a
caller, the client cell routes through the workspace's own ``onOpenClient``, and
the agency cell resolves a name. A capability nobody can reach is not a
capability, and a grep is exactly what caught it.

The **payload** half runs the requests the controls actually send. The form
sends only the fields that changed, which is what keeps an amend from
disturbing a term nobody touched, and it refuses to empty a percent rather than
sending a request the endpoint would silently ignore. Both are asserted against
the real routes.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
CLIENTS = ROOT / "tv-break-dashboard" / "src" / "clients"
SRC = ROOT / "tv-break-dashboard" / "src"


def read(name: str) -> str:
    return (CLIENTS / name).read_text(encoding="utf-8")


def frontend_sources() -> list[Path]:
    return [path for path in SRC.rglob("*.js") if path.is_file()] + [
        path for path in SRC.rglob("*.jsx") if path.is_file()
    ]


# --------------------------------------------------------------------------
# The wiring half
# --------------------------------------------------------------------------

def test_every_write_this_destination_exports_has_a_caller():
    """The grep that caught the defect, kept as the assertion that holds it shut."""
    api = read("clients-api.js")
    exported = sorted(
        line.split("export async function ")[1].split("(")[0]
        for line in api.splitlines()
        if line.startswith("export async function ")
    )
    assert {"createCampaign", "updateCampaign", "addFlight", "removeFlight", "updateFlight"} <= set(exported)

    bodies = {
        path: path.read_text(encoding="utf-8")
        for path in frontend_sources()
        if path.name != "clients-api.js"
    }
    for name in exported:
        callers = [path.name for path, text in bodies.items() if f"{name}(" in text]
        assert callers, f"{name} is exported and nothing calls it, so no screen can reach it"


def test_the_four_row_controls_call_the_four_endpoints():
    """Add a flight, edit a flight, remove a flight, amend the campaign."""
    flights = read("CampaignFlights.jsx")
    assert "import { addFlight, removeFlight, updateFlight } from './clients-api'" in flights
    assert "await addFlight(campaign.campaign_id" in flights
    assert "await updateFlight(campaign.campaign_id, flight.flight_id" in flights
    assert "await removeFlight(campaign.campaign_id, flightId)" in flights

    terms = read("CampaignTerms.jsx")
    assert "import { createCampaign, updateCampaign } from './clients-api'" in terms
    assert "await updateCampaign(campaign.campaign_id" in terms
    assert "await createCampaign(" in terms


def test_removing_a_flight_is_confirmed_and_says_the_campaign_stays():
    """Deactivate beats delete for a campaign; a flight is a line and says so."""
    flights = read("CampaignFlights.jsx")
    assert "pendingRemove" in flights
    assert "The flight is deleted. The campaign stays." in flights
    assert "טיסת השידור נמחקת. הקמפיין נשאר." in flights


def test_the_client_name_on_the_campaign_board_opens_the_client():
    """The same name is a button on the tree and the money board, so it is here.

    The wiring is now two hops: the workspace binds its own setter into the
    `on` bundle, and ClientsPanels (the markup half of the file-size split)
    hands that bundle's opener to the board.
    """
    board = read("CampaignBoard.jsx")
    assert "onClick={() => onOpenClient(campaign.advertiser)}" in board
    workspace = read("ClientsWorkspace.jsx")
    assert "openClient: setOpenClient" in workspace
    panels = read("ClientsPanels.jsx")
    campaign_block = panels.split("<CampaignBoard")[1].split("/>")[0]
    assert "onOpenClient={on.openClient}" in campaign_block, (
        "the board is handed the workspace's own opener"
    )


def test_the_agency_column_reads_as_a_name_with_the_id_underneath():
    """AGY_10 is a storage key. The name leads and the key stays findable."""
    board = read("CampaignBoard.jsx")
    assert "const name = agencies[campaign.agency_id];" in board
    assert "<strong>{name}</strong>" in board
    assert "<small className=\"clients-campaign-id\">{campaign.agency_id}</small>" in board
    assert "name not on file" in board, "an id with no agency behind it is a stated state"
    workspace = read("ClientsWorkspace.jsx")
    assert "agencyIndex" in workspace, "the workspace still owns the name index"
    panels = read("ClientsPanels.jsx")
    assert "agencies={data.agencies}" in panels, (
        "the board receives the index through the panels' data bundle"
    )


def test_the_weekday_chips_keep_the_order_the_endpoint_sends():
    """ISO keys, Sunday first, and the surface never re-sorts them itself."""
    terms = read("CampaignTerms.jsx")
    assert "weekdays.map((day)" in terms
    assert ".sort()" in terms, "the scope string is sorted, which is what the store stores"
    assert "weekdays.sort" not in terms, "the presentation order is the endpoint's, not this file's"


def test_no_source_file_on_this_destination_is_over_the_cap():
    for path in sorted(CLIENTS.iterdir()):
        if path.suffix in {".js", ".jsx", ".css"}:
            lines = len(path.read_text(encoding="utf-8").splitlines())
            assert lines <= 450, f"{path.name} is {lines} lines"


# --------------------------------------------------------------------------
# The payload half
# --------------------------------------------------------------------------

@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client over P4's routers, with every store pointed at tmp_path."""
    from kairos_api import agencies, agency_conditions, campaigns_api, campaigns_api_store, version_store

    monkeypatch.setattr(agencies, "AGENCIES_PATH", tmp_path / "agencies.csv")
    monkeypatch.setattr(agencies, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(agency_conditions, "LINKS_PATH", tmp_path / "agency_advertisers.csv")
    monkeypatch.setattr(agency_conditions, "CONDITIONS_PATH", tmp_path / "agency_conditions.csv")
    monkeypatch.setattr(agency_conditions, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(campaigns_api_store, "CAMPAIGNS_PATH", tmp_path / "campaigns.csv")
    monkeypatch.setattr(campaigns_api_store, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(version_store, "snapshot_manual_edit", lambda request, logical: None)
    monkeypatch.setattr(agency_conditions, "_latest_daily_pairs", lambda: ([], None))

    app = FastAPI()
    app.include_router(agencies.router)
    app.include_router(agency_conditions.router)
    app.include_router(campaigns_api.router)
    return TestClient(app)


ORDER = {
    "advertiser": "טורנדו מוצרי צריכה",
    "campaign_name": "קיץ 2026",
    "campaign_starts_on": "2026-08-02",
    "campaign_ends_on": "2026-08-29",
    "rebate_percent": 4.0,
    "surcharge_discount_percent": 20.0,
    "surcharge_weekdays": "6",
    "flights": [
        {"starts_on": "2026-08-02", "ends_on": "2026-08-15", "goal_kind": "spots", "goal_value": 40},
    ],
}


def _booked(client) -> str:
    response = client.post("/api/agencies", json={"agency_id": "AGY_02", "name": "יוניברסל", "rebate_percent": 4.0})
    assert response.status_code == 201, response.text
    created = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_02"}, **ORDER})
    assert created.status_code == 201, created.text
    return created.json()["campaign"]["campaign_id"]


def test_an_amend_carries_only_what_changed_and_leaves_the_rest_alone(client):
    """The window and the weekday scope move; the rebate nobody touched does not."""
    campaign_id = _booked(client)
    amended = client.put(f"/api/clients/campaigns/{campaign_id}", json={
        "ends_on": "2026-09-30",
        "surcharge_weekdays": "5,6",
        "notes": "הוסף שבוע אחרון לפי בקשת הלקוח",
    })
    assert amended.status_code == 200, amended.text
    record = amended.json()
    assert record["ends_on"] == "2026-09-30"
    assert record["surcharge_weekdays"] == "5,6"
    assert record["notes"] == "הוסף שבוע אחרון לפי בקשת הלקוח"
    assert record["rebate_percent"] == 4.0, "a field the form did not send must not move"
    assert record["starts_on"] == "2026-08-02"
    assert record["advertiser"] == ORDER["advertiser"]
    assert len(client.get("/api/clients/campaigns").json()["campaigns"][0]["flights"]) == 1


def test_a_percent_already_on_file_cannot_be_emptied_through_this_route(client):
    """Why the form refuses instead of sending a request that changes nothing."""
    campaign_id = _booked(client)
    unchanged = client.put(f"/api/clients/campaigns/{campaign_id}", json={"rebate_percent": None})
    assert unchanged.status_code == 200
    assert unchanged.json()["rebate_percent"] == 4.0, "null is a no-op, so the surface must not offer it as a clear"
    zeroed = client.put(f"/api/clients/campaigns/{campaign_id}", json={"rebate_percent": 0})
    assert zeroed.json()["rebate_percent"] == 0.0, "zero is the honest way to record no rebate"


def test_a_flight_goal_and_its_unit_change_in_place(client):
    """The row edit sends the two fields it changed and the flight keeps its id."""
    campaign_id = _booked(client)
    flight_id = client.get("/api/clients/campaigns").json()["campaigns"][0]["flights"][0]["flight_id"]
    changed = client.put(f"/api/clients/campaigns/{campaign_id}/flights/{flight_id}", json={
        "goal_value": 55, "goal_kind": "grp",
    })
    assert changed.status_code == 200, changed.text
    assert changed.json()["goal_value"] == 55.0
    assert changed.json()["goal_kind"] == "grp"
    assert changed.json()["flight_id"] == flight_id
    stored = client.get("/api/clients/campaigns").json()["campaigns"][0]
    assert stored["starts_on"] == "2026-08-02", "amending a flight leaves the campaign alone"
    assert "delivered" not in stored["flights"][0], "no delivery figure appears from an edit"


def test_a_second_campaign_for_a_client_on_file_needs_no_second_onboarding(client):
    """The dead end the board's compact form exists to close, stated as a test."""
    _booked(client)
    repeat = client.post("/api/clients/onboarding", json={"agency": {"agency_id": "AGY_02"}, **ORDER})
    assert repeat.status_code == 409, "re-running the whole flow to reach a client is refused"

    direct = client.post("/api/clients/campaigns", json={
        "name": "סתיו 2026",
        "advertiser": ORDER["advertiser"],
        "agency_id": "AGY_02",
        "starts_on": "2026-10-04",
        "ends_on": "2026-10-31",
        "rebate_percent": 3.5,
    })
    assert direct.status_code == 201, direct.text
    assert direct.json()["campaign_id"] == "CMP_002"
    assert len(client.get("/api/agencies").json()["agencies"]) == 1, "no second agency was invented"
    links = client.get("/api/agencies/AGY_02/advertisers").json()
    assert links["manual"].count(ORDER["advertiser"]) == 1, "no duplicate client link"
    assert len(client.get("/api/clients/campaigns").json()["campaigns"]) == 2
