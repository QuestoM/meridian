"""P6 Sources, the report shelf: the row count is a promise about a file.

Split from ``tests/test_p6_sources.py`` under the 450-line law and carried on
this piece's reserved prefix. Every test here asks one question: does the number
beside a report equal the number of rows the download it names actually carries,
and can a reader open those rows and check.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.downloads_api as downloads_api
import kairos_api.downloads_api_preview as downloads_api_preview
import kairos_api.uploads as uploads
import kairos_api.uploads_preview as uploads_preview


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(uploads.router)
    app.include_router(downloads_api.router)
    return TestClient(app)


@pytest.fixture()
def owned(monkeypatch) -> str:
    """A configured operator channel, pinned rather than read off disk."""
    from kairos.data.loaders import CHANNELS
    from kairos_api import channel_scope, read_cache

    channel = CHANNELS[-1]
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: channel)
    read_cache.invalidate(uploads_preview.PREVIEW_NAMESPACE)
    return channel


# --- the report shelf, unchanged --------------------------------------------
def test_every_row_count_equals_the_rows_its_own_download_carries() -> None:
    """The promise the shelf prints, measured against the real files.

    The surface tells the reader that the number beside a report is the exact
    number of rows that download will carry, so this counts them: the two
    server-streamed CSVs by fetching them, and the three browser-built ones by
    fetching the payload their builder maps, row for row. Measured before this
    piece the revenue card printed 2,391 against a file of 30 rows.
    """
    from kairos_api.server import app

    live = TestClient(app)
    reports = {report["id"]: report for report in live.get("/api/reports").json()["reports"]}

    def csv_rows(path: str) -> int:
        text = live.get(path).text
        return len([line for line in text.splitlines() if line.strip()]) - 1

    assert reports["weekly-plan"]["rows"] == csv_rows("/api/export/schedule.csv")
    assert reports["daily-spots"]["rows"] == csv_rows("/api/export/spots.csv")
    assert reports["compliance"]["rows"] == len(live.get("/api/compliance").json()["checks"])
    assert reports["revenue"]["rows"] == len(live.get("/api/forecasts").json()["by_day"])
    assert reports["data-quality"]["rows"] == len(live.get("/api/files").json()["files"])


def test_the_five_reports_survive_with_their_owners(client: TestClient) -> None:
    reports = client.get("/api/reports").json()["reports"]
    assert {report["id"] for report in reports} == {
        "weekly-plan",
        "compliance",
        "revenue",
        "daily-spots",
        "data-quality",
    }
    assert {report["owner"] for report in reports} == {"Traffic", "Legal / Ops", "Revenue", "Data"}


def test_the_source_audit_report_counts_the_rows_its_download_carries(client: TestClient) -> None:
    """The audit CSV writes one row per audited file, present or missing, so
    that is what the count beside it has to be. It counted only the present
    ones before, which is the same number today and a different one the first
    day a file goes missing."""
    reports = {report["id"]: report for report in client.get("/api/reports").json()["reports"]}
    files = client.get("/api/files").json()["files"]
    assert reports["data-quality"]["rows"] == len(files)


def test_every_report_declares_what_one_row_is_and_the_basis_it_was_built_on(client: TestClient) -> None:
    """Stripe attaches the declared facts to the report itself. A row count with
    no unit is a number two readers count two different things with."""
    for report in client.get("/api/reports").json()["reports"]:
        assert report["unit"]["en"] and report["unit"]["he"], f"{report['id']} has no unit"
        assert report["basis"], f"{report['id']} declares no basis"
        for fact in report["basis"]:
            assert fact["label_en"] and fact["label_he"], f"{report['id']} basis fact has no label"
            assert fact["value_en"] and fact["value_he"], f"{report['id']} basis fact has no value"
        codes = [fact["code"] for fact in report["basis"]]
        assert "scope" in codes, f"{report['id']} prints a figure with no scope"
        assert len(codes) == len(set(codes)), f"{report['id']} declares a fact twice"


def test_no_report_basis_names_a_channel_the_operator_does_not_own(client: TestClient) -> None:
    from kairos.data.loaders import CHANNELS
    from kairos_api import channel_scope

    owned = channel_scope.operator_channel()
    payload = json.dumps(client.get("/api/reports").json(), ensure_ascii=False)
    for rival in [channel for channel in CHANNELS if channel != owned]:
        assert rival not in payload, "a report basis named a channel the operator does not own"


def test_the_two_server_streamed_reports_open_the_rows_their_file_carries(client: TestClient, owned: str) -> None:
    """The Stripe drill: the number opens the rows behind it, from the same
    source the download streams."""
    for report_id in downloads_api_preview.SERVED:
        body = client.get(f"/api/reports/{report_id}/preview?limit=4").json()
        assert body["available"] is True, f"{report_id} cannot open its own rows"
        assert body["columns"], f"{report_id} previewed no columns"
        assert 0 < body["shown_rows"] <= 4
        assert all(len(row) == len(body["columns"]) for row in body["rows"])


def test_the_weekly_plan_preview_matches_the_file_total_and_hides_the_rest(client: TestClient, owned: str) -> None:
    """The download carries the whole plan file and Bar 3 freezes that. The
    preview is a screen, so it states the file's own total and shows only the
    operator's rows, with the count it withheld disclosed and unnamed."""
    reports = {report["id"]: report for report in client.get("/api/reports").json()["reports"]}
    body = client.get("/api/reports/weekly-plan/preview?limit=5").json()
    if not body["available"]:
        pytest.skip("no saved plan on disk to preview")
    assert body["total_rows"] == reports["weekly-plan"]["rows"]
    assert body["scoped_rows"] <= body["total_rows"]
    from kairos.data.loaders import CHANNELS

    payload = json.dumps(body, ensure_ascii=False)
    for rival in [channel for channel in CHANNELS if channel != owned]:
        assert rival not in payload, "the plan preview named a channel the operator does not own"


def test_the_plan_preview_shows_nothing_until_a_channel_is_configured(client: TestClient, monkeypatch) -> None:
    from kairos_api import channel_scope

    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: "")
    body = client.get("/api/reports/weekly-plan/preview?limit=5").json()
    assert body["available"] is False
    assert body["notes"][0]["code"] == "no_channel"
    assert body["notes"][0]["he"]


def test_a_client_built_report_names_the_endpoint_its_rows_come_from(client: TestClient) -> None:
    """Three of the five are built in the browser from a live endpoint. This
    route refuses to re-derive them and names that endpoint instead, because a
    second derivation is how two surfaces come to disagree."""
    for report_id, source in downloads_api_preview.CLIENT_SOURCES.items():
        response = client.get(f"/api/reports/{report_id}/preview")
        assert response.status_code == 404
        assert source in response.json()["detail"]
    assert client.get("/api/reports/nonsense/preview").status_code == 404


def test_every_input_publishes_what_the_door_checks_and_what_it_cannot(client: TestClient) -> None:
    """Frame.io publishes the proxy ladder exactly, including what will not play
    at all. This is the same promise: the real code that runs, and the limits."""
    from kairos_api import uploads_checks, uploads_validate

    for entry in client.get("/api/uploads/status").json()["inputs"]:
        checks = entry["checks"]
        assert checks["required_columns"] == uploads.REQUIRED_COLUMNS[entry["kind"]]
        assert checks["checked_en"] and checks["checked_he"]
        assert len(checks["cannot_verify"]) == len(uploads_checks.CANNOT_VERIFY)
        for limit in checks["cannot_verify"]:
            assert limit["en"] and limit["he"], "a stated limit must be readable in both languages"
        has_contract = entry["kind"] in uploads_validate.CONTRACT_VALIDATORS
        assert bool(checks["contract"]) == has_contract, f"{entry['kind']} misreports its contract"
        if checks["loader"]:
            module, _, name = checks["loader"].rpartition(".")
            assert getattr(__import__(module, fromlist=[name]), name), "the named loader does not exist"
