"""P6 Sources: the two files this destination streams, and what is inside them.

A second boundary file, and the reason is measured rather than editorial.
``tests/test_p6_boundary.py`` sweeps every route on this piece's row by reading
``response.json()``, and the two routes that carry more channel names than any
payload in the product do not answer JSON at all. So for ten rounds a test
named for the boundary passed while the largest breach on this row sat one
click from the operator's screen, unasked. The sweep and the file it needed are
here, plus the assertion that the two sweeps together are the whole row, so the
next route added under this destination cannot land under neither.

The disclosure is the other half. The download keeps every row it has today,
because section 8.5 freezes its row count, and the card that offers it now
prints the same measured split the preview computes. That is a screen, so it is
asserted here too: the split has to be the file's own arithmetic, and it may
carry a count of other channels and never a name.
"""

from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.downloads_api as downloads_api
import kairos_api.downloads_api_reports as downloads_api_reports
import kairos_api.exporters as exporters
import kairos_api.uploads as uploads
import kairos_api.uploads_preview as uploads_preview
from kairos.data.loaders import CHANNELS

# Every GET route on this piece's row of the section 8.2 table, as the template
# the router registers, split by what it answers with. The payload sweep lives
# in the sibling file and reads JSON; the download sweep is here and reads a
# streamed file. The third test asserts these two sets together are the whole
# row, so a route added later is red until somebody sweeps it.
JSON_ROUTES = (
    "/api/uploads/status",
    "/api/uploads/{kind}/preview",
    "/api/files",
    "/api/reports",
    "/api/reports/{report_id}/preview",
)
CSV_ROUTES = ("/api/export/schedule.csv", "/api/export/spots.csv")


@pytest.fixture()
def owned(monkeypatch) -> str:
    """A configured operator channel, pinned rather than read off disk.

    The settings file is shared writable state, and a boundary assertion that
    another process can turn into a skip is not an assertion. The channel is one
    the shipped settings do NOT name, so a filter that happens to hide three
    particular names cannot pass this.
    """
    from kairos_api import channel_scope, read_cache

    channel = CHANNELS[-1]
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: channel)
    read_cache.invalidate(uploads_preview.PREVIEW_NAMESPACE)
    downloads_api._reports_cached.cache_clear()
    yield channel
    downloads_api._reports_cached.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(uploads.router)
    app.include_router(downloads_api.router)
    app.include_router(exporters.router)
    return TestClient(app)


def _rivals(owned: str) -> list[str]:
    return [channel for channel in CHANNELS if channel != owned]


def _weekly_plan(client: TestClient) -> dict:
    return next(
        report for report in client.get("/api/reports").json()["reports"] if report["id"] == "weekly-plan"
    )


def test_the_downloads_of_this_destination_name_no_channel_the_operator_does_not_own(
    client: TestClient, owned: str
) -> None:
    """The two files this piece streams, swept the way its payloads are swept.

    **This test is red on the tree, and it is red for the reason it exists.**
    ``GET /api/export/schedule.csv`` streams the saved plan file whole and that
    file is the whole market: measured with the shipped settings, 8,704 data
    rows of which 6,164 are on three channels this operator does not own, every
    one of them naming that channel in its own column and again inside
    ``segment_id``, and carrying its ``predicted_revenue``, ``base_rate`` and
    ``baseline_tvr``. The button is on the weekly plan card and again under
    "download every ready report". ``spots.csv`` carries no channel name at all
    and passes here, which is what makes this a measurement of one route rather
    than of a category.

    Scoping the file is not a change this piece may take by itself: section 8.5
    freezes this download's row count at 8,704 on P2's row and freezes "all five
    report CSVs, all with the same row counts" on P6's, so the boundary at 8.3
    and the regression rows at 8.5 cannot both stand. That conflict is written
    up with these numbers at section 14 of this piece's contract and C2 rules on
    it. Until the ruling lands the file keeps every row it has today, the card
    discloses the split at the click, and this stays red rather than green over
    a boundary nobody is holding.
    """
    for path in CSV_ROUTES:
        response = client.get(path)
        assert response.status_code == 200, f"{path} answered {response.status_code}"
        body = response.text
        named = {rival: body.count(rival) for rival in _rivals(owned) if rival in body}
        assert named == {}, f"{path} streams channels this operator does not own: {named}"


def test_every_route_on_this_pieces_row_is_under_one_of_the_two_sweeps() -> None:
    """A route this piece owns that no sweep asks is the defect above, exactly.

    The route set is read off the routers rather than from a list written by
    hand, so a route added later is red here on the day it is added instead of
    on the day somebody thinks to look.
    """
    routers = (uploads.router, downloads_api.router, exporters.router)
    live = {
        route.path
        for router in routers
        for route in router.routes
        if "GET" in (getattr(route, "methods", None) or set())
    }
    assert live == set(JSON_ROUTES) | set(CSV_ROUTES), "a GET route on this row is under no boundary sweep"


def test_the_card_that_offers_the_file_says_what_is_inside_it(client: TestClient, owned: str) -> None:
    """The disclosure at the click, not only in a panel nobody opens.

    The person who clicks download never opens the preview, so the split the
    preview computes travels on the report itself. It has to be the file's own
    arithmetic: the two row counts sum to the row count the button promises.
    """
    report = _weekly_plan(client)
    scope = report.get("download_scope")
    assert scope, "the weekly plan card offers a file and says nothing about what is in it"
    if scope["state"] != "real":
        pytest.skip("no channel column on the saved plan, so there is no split to state")
    # Ruling 009 split these apart. rows_total is what the FILE holds and the
    # button's count is what the DOWNLOAD serves, the operator's own channel, so
    # a card that made them equal would be advertising rows it does not deliver.
    # Both are asserted, and the relationship between them is the disclosure.
    assert scope["rows_served"] == report["rows"], "the button offers a count the disclosure does not"
    assert scope["rows_served"] == scope["rows_owned"], "the download serves something other than your own rows"
    assert scope["rows_total"] > scope["rows_served"], "the file holds no more than the download, so there is nothing to disclose"
    assert scope["rows_owned"] + scope["rows_other"] == scope["rows_total"], "the split does not sum to the file"
    assert scope["rows_other"] > 0, "the shipped plan file carries rows this operator does not own"
    assert scope["channels_other"] > 0, "rows on other channels came from no other channel"
    assert scope["en"] and scope["he"], "the disclosure is not readable in both languages"


def test_the_disclosure_counts_the_other_channels_and_never_names_one(client: TestClient, owned: str) -> None:
    """A count is the unnamed aggregate the boundary allows. A name is not."""
    scope = _weekly_plan(client)["download_scope"]
    body = json.dumps(scope, ensure_ascii=False)
    for rival in _rivals(owned):
        assert rival not in body, "the disclosure named a channel this operator does not own"
    assert owned not in body, "the disclosure names the owned channel where 'your own channel' reads in both"


def test_the_disclosure_is_the_same_measurement_the_preview_opens(client: TestClient, owned: str) -> None:
    """One measurement, two surfaces. A card and a drawer that derive the same
    split separately are how two screens come to disagree about one file."""
    scope = _weekly_plan(client)["download_scope"]
    preview = client.get("/api/reports/weekly-plan/preview?limit=1").json()
    if not preview["available"] or scope["state"] != "real":
        pytest.skip("no saved plan with a channel column to compare against")
    assert scope["rows_total"] == preview["total_rows"]
    assert scope["rows_owned"] == preview["scoped_rows"]
    assert scope["rows_other"] == preview["scope"]["competitor_rows_excluded"]
    assert scope["channels_other"] == preview["scope"]["competitor_channels_excluded"]


def test_with_no_operator_channel_the_split_is_unknown_and_says_where_to_set_it(
    client: TestClient, monkeypatch
) -> None:
    """The honest unknown. Nothing here knows which rows are the operator's, so
    the card says so and names the screen that answers it, and the two row
    figures are absent rather than zero."""
    from kairos_api import channel_scope

    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: "")
    downloads_api._reports_cached.cache_clear()
    try:
        scope = _weekly_plan(client)["download_scope"]
    finally:
        downloads_api._reports_cached.cache_clear()
    assert scope["state"] == "unknown"
    assert "rows_owned" not in scope and "rows_other" not in scope, "an unknown split printed a figure"
    assert scope["rows_total"] > 0, "the file's own row count is known even when the split is not"
    assert "settings" in scope["en"] and "הגדרות" in scope["he"], "the unknown does not say where to set it"


def test_a_plan_of_one_channel_is_stated_as_such_rather_than_as_a_zero() -> None:
    """The wording is a fact about the file, so a file with nothing to disclose
    says that in words instead of printing 'and 0 rows on 0 other channels'."""
    import pandas as pd

    frame = pd.DataFrame({"channel": ["A", "A"], "predicted_revenue": [1.0, 2.0]})
    scope = downloads_api_reports.download_scope(frame, "A")
    assert scope["state"] == "real"
    assert scope["rows_owned"] == 2 and scope["rows_other"] == 0 and scope["channels_other"] == 0
    assert "0" not in scope["en"].replace("2", ""), "the one-channel sentence prints a zero"
    assert downloads_api_reports.download_scope(pd.DataFrame(), "A") is None
