"""P6 Sources: the state is derived from content and from siblings.

A blind critic measured the status card reporting a healthy state for two
situations that were not healthy, and both are here. One: a valid daily file
uploaded through the surface, accepted, committed, on disk, and named on no
screen, because the engine reads the newest daily file by the airing date in
its name and that one named an earlier day. Two: a live input the engine really
does read that carries a header and zero rows, reported as in use with nothing
to do.

A third measurement, two rounds later, is the pair of sentences a state prints.
The same shadowed daily card carried a remedy saying an upload here changes
nothing and, one line below it, a consequence saying it replaces what the plan
is computed from, with nothing on the card naming the airing date that decides
between them. So the two are swept here for a card that carries both claims at
once, in both languages, over a fixture and over the shipped tree.

The door's own answers moved to ``test_p6_prospect.py`` when this file reached
the size law: what a candidate WILL do is a different question from the state
its kind is in, and that one is graded against what committing really does.

Same fixtures as ``test_p6_sources.py`` and deliberately its own file: that one
is at the size law, and the shape of a state is its own subject.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.downloads_api as downloads_api
import kairos_api.uploads as uploads
import kairos_api.uploads_status as uploads_status
from kairos.data.loaders import DAILY_COLUMN_MAP

ROOT = Path(__file__).resolve().parents[1]
LIVE_DAILY = ROOT / "data" / "daily_input" / "Wally_Prime_Reshet_Example_2025-04-27.csv"


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(uploads.router)
    app.include_router(downloads_api.router)
    return TestClient(app)


@pytest.fixture()
def isolated(tmp_path, monkeypatch) -> TestClient:
    """Every writable path relocated, so no repository input is touched."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(uploads, "DATA_DIR", data_dir)
    monkeypatch.setattr(uploads, "DAILY_DIR", data_dir / "daily_input")
    monkeypatch.setattr(uploads, "BACKUP_DIR", data_dir / "_backups")
    monkeypatch.setattr(
        uploads, "VALIDATION_REPORTS_PATH", tmp_path / "output" / "upload_validation_reports.json"
    )
    app = FastAPI()
    app.include_router(uploads.router)
    return TestClient(app)


def _daily_bytes(*, date_value: str = "4/27/2025", duration: str = "30", rows: int = 1) -> bytes:
    row = {name: "" for name in DAILY_COLUMN_MAP}
    row["תאריך"] = date_value
    row["שעה"] = "18:01:00"
    row["שעת התחלת ברייק"] = "18:00:00"
    row["מפרסם"] = "Acme"
    row["קמפיין"] = "Acme Summer"
    row["תוכנית מוזמנת"] = "Evening Show"
    row["שעת התחלת תוכנית"] = "18:00"
    row["אורך תשדיר"] = duration
    row["מיקום בברייק"] = "1"
    row["רייטינג ברייקים מתוכנן"] = "5.5"
    frame = pd.DataFrame([row] * rows, columns=list(DAILY_COLUMN_MAP.keys()))
    return frame.to_csv(index=False).encode("utf-8")


def _flights_header() -> bytes:
    """A campaign flight file with the real header and not one data row."""
    return (",".join(uploads.REQUIRED_COLUMNS["campaign_flights"]) + "\n").encode("utf-8")


def test_a_daily_file_with_an_older_airing_date_is_named_as_stored_and_unread(isolated: TestClient) -> None:
    """The measured hole: a valid daily file, accepted, committed, landed on disk,
    and invisible on every view of the destination.

    ``_newest_daily`` orders by the airing date in the name, so the engine kept
    reading the earlier upload and the status answered ``in_use`` with the
    remedy "nothing to do" for a kind whose most recent file nothing reads. The
    honest sentence existed and was reachable only from the POST response, which
    is gone on the next page load.
    """
    live = isolated.post(
        "/api/uploads/daily", files={"file": ("Wally_2025-04-27.csv", _daily_bytes(), "text/csv")}
    )
    assert live.status_code == 200, live.text
    live_path = uploads.DAILY_DIR / "Wally_2025-04-27.csv"
    # The file the engine reads landed FIRST, pinned rather than raced.
    os.utime(live_path, (1_700_000_000, 1_700_000_000))

    older = isolated.post(
        "/api/uploads/daily",
        files={"file": ("Wally_2020-01-05.csv", _daily_bytes(date_value="1/5/2020"), "text/csv")},
    )
    assert older.status_code == 200, older.text
    assert older.json()["in_use"] is False, "the door already knew this file would not be read"

    entry = next(
        item for item in isolated.get("/api/uploads/status").json()["inputs"] if item["kind"] == "daily"
    )
    assert entry["filename"] == "Wally_2025-04-27.csv", "the engine still reads the later airing date"
    assert entry["state"] == "shadowed", "a kind whose newest file nothing reads is not in use"
    # The remedy for THIS shadow is not the remedy for a kind another file is
    # read instead of. That one ends "keep uploading here and the plan will not
    # change", which is false here: the next upload is read or not read by the
    # day its name carries. See the pair test below.
    assert entry["remedy"]["code"] == "shadowed_by_a_later_day" and entry["remedy"]["he"]
    assert entry["stored_unread_total"] == 1
    stored = entry["stored_unread"][0]
    assert stored["filename"] == "Wally_2020-01-05.csv", "the stored file is named on the status"
    assert stored["path"].endswith("Wally_2020-01-05.csv")
    assert stored["arrived_after_live"] is True, "it arrived after the file the engine reads"
    assert stored["rows"] == 1, "a named file carries its own row count, not the live file's"
    assert "Wally_2025-04-27.csv" in stored["reason"]["en"], "the reason names the file read instead"
    assert "Wally_2025-04-27.csv" in stored["reason"]["he"], "and it names it in Hebrew too"
    assert any("Wally_2025-04-27.csv" in warning for warning in entry["warnings"]), (
        "a reader that renders warnings and nothing else still sees it"
    )


def test_an_archived_daily_file_is_named_without_turning_the_kind_amber(isolated: TestClient) -> None:
    """Yesterday's file is an archive, not a problem, and the two are not one word.

    The distinction is which one arrived last: a file that landed BEFORE the one
    the engine reads is the day before, and calling that shadowed would cry wolf
    every morning until nobody read the badge at all.
    """
    first = isolated.post(
        "/api/uploads/daily",
        files={"file": ("Wally_2025-04-26.csv", _daily_bytes(date_value="4/26/2025"), "text/csv")},
    )
    assert first.status_code == 200, first.text
    os.utime(uploads.DAILY_DIR / "Wally_2025-04-26.csv", (1_700_000_000, 1_700_000_000))
    second = isolated.post(
        "/api/uploads/daily", files={"file": ("Wally_2025-04-27.csv", _daily_bytes(), "text/csv")}
    )
    assert second.status_code == 200, second.text

    entry = next(
        item for item in isolated.get("/api/uploads/status").json()["inputs"] if item["kind"] == "daily"
    )
    assert entry["state"] == "in_use", "the file that arrived last is the file that is read"
    assert entry["stored_unread_total"] == 1, "and the day before is still named"
    stored = entry["stored_unread"][0]
    assert stored["filename"] == "Wally_2025-04-26.csv"
    assert stored["arrived_after_live"] is False
    assert stored["reason"]["code"] == "another_day_is_read"
    assert stored["reason"]["he"], "an archived file says why it is not read, in Hebrew"


def test_a_live_file_the_engine_reads_with_no_rows_is_not_nothing_to_do(isolated: TestClient) -> None:
    """The second half of the same defect: content, not just the read path.

    ``data/campaign_flights.csv`` is read by the pacing loader and carries a
    header and zero rows, and the card reported it in use with nothing to do.
    There is no populated flight file anywhere in this repository, so the state
    that names the missing input is the only honest one.
    """
    response = isolated.post(
        "/api/uploads/campaign_flights",
        files={"file": ("campaign_flights.csv", _flights_header(), "text/csv")},
    )
    assert response.status_code == 200, response.text
    entry = next(
        item
        for item in isolated.get("/api/uploads/status").json()["inputs"]
        if item["kind"] == "campaign_flights"
    )
    assert entry["rows"] == 0
    assert entry["in_use"] is True, "pacing does read this path, and that is not what is wrong with it"
    assert entry["state"] == "empty", "a file the engine reads with no rows in it is not in use"
    assert entry["remedy"]["code"] == "empty"
    assert entry["remedy"]["he"] and entry["remedy"]["en"]
    assert uploads_status.NO_ROWS_WARNING in entry["warnings"]
    assert entry["consequence"]["code"] == "replaces_live_input", "uploading here is the way to supply it"


def test_the_shipped_flight_file_reports_what_it_really_is(client: TestClient) -> None:
    """The same measurement on the repository's own tree rather than a fixture."""
    entry = next(
        item
        for item in client.get("/api/uploads/status").json()["inputs"]
        if item["kind"] == "campaign_flights"
    )
    if not entry["exists"]:
        assert entry["state"] == "missing"
        return
    if entry["rows"]:
        pytest.skip("the shipped flight file now carries rows, so there is no empty state to measure")
    assert entry["state"] == "empty", "a header with no rows under it is not a live input"
    assert entry["remedy"]["he"], "and the empty state names what to do about it, in Hebrew"


# What a sentence claims an upload here does to the plan, read off the words the
# card really prints rather than off the code that chose them. One card may
# carry at most one of these two claims. Both at once is what shipped: a
# shadowed daily card printed "keep uploading here and the plan will not change"
# one line above "this is the live input, uploading replaces what the plan is
# computed from", and nothing on it said the difference is the airing date.
CHANGES_THE_PLAN = {
    "en": ("uploading replaces what the plan is computed from",),
    "he": ("העלאה מחליפה את מה שהתוכנית מחושבת ממנו",),
}
CHANGES_NOTHING = {
    "en": ("the plan will not change", "uploading changes no number"),
    "he": ("והתוכנית לא תשתנה", "העלאה לא משנה אף מספר"),
}


def _claims(record: dict, markers: dict, locale: str) -> bool:
    """Whether this sentence makes this claim, in the language it is read in."""
    return any(marker.lower() in str((record or {}).get(locale) or "").lower() for marker in markers[locale])


def _disagrees(entry: dict, locale: str) -> bool:
    pair = (entry.get("remedy"), entry.get("consequence"))
    return any(_claims(one, CHANGES_THE_PLAN, locale) for one in pair) and any(
        _claims(one, CHANGES_NOTHING, locale) for one in pair
    )


def test_no_card_renders_a_remedy_and_a_consequence_that_disagree(isolated: TestClient) -> None:
    """Three states in one payload, and no card carrying both claims at once.

    The daily kind is the one that can hold the contradiction: the engine reads
    it and the operator's own last upload lost the resolver, so neither "this is
    the live input" nor "nothing reads this" is true of the next upload. The
    flight file is uploaded too, because a sweep whose markers match nothing
    passes over everything, and it is a live input that claims so.
    """
    assert isolated.post(
        "/api/uploads/daily", files={"file": ("Wally_2025-04-27.csv", _daily_bytes(), "text/csv")}
    ).status_code == 200
    os.utime(uploads.DAILY_DIR / "Wally_2025-04-27.csv", (1_700_000_000, 1_700_000_000))
    assert isolated.post(
        "/api/uploads/daily",
        files={"file": ("Wally_2020-01-05.csv", _daily_bytes(date_value="1/5/2020"), "text/csv")},
    ).status_code == 200
    assert isolated.post(
        "/api/uploads/campaign_flights",
        files={"file": ("campaign_flights.csv", _flights_header(), "text/csv")},
    ).status_code == 200

    inputs = isolated.get("/api/uploads/status").json()["inputs"]
    for locale in ("en", "he"):
        # A marker set that matches nothing would make the sweep vacuous, so
        # both are checked against the two sentences they were read from: the
        # remedy and the consequence that were on one card together.
        shadowed = uploads_status.labelled(uploads_status._REMEDIES, "shadowed")
        live = uploads_status.labelled(uploads_status._CONSEQUENCES, "replaces_live_input")
        assert _claims(shadowed, CHANGES_NOTHING, locale), "the no-change marker matches nothing"
        assert _claims(live, CHANGES_THE_PLAN, locale), "the plan-change marker matches nothing"
        assert any(_claims(entry["consequence"], CHANGES_THE_PLAN, locale) for entry in inputs)
        for entry in inputs:
            assert not _disagrees(entry, locale), (
                f"the {entry['kind']} card says both things in {locale}: "
                f"{entry['remedy'][locale]} / {entry['consequence'][locale]}"
            )

    daily = next(entry for entry in inputs if entry["kind"] == "daily")
    assert daily["state"] == "shadowed" and daily["in_use"] is True
    assert daily["consequence"]["code"] == "replaces_only_a_later_day"
    for locale, rule in (("en", "airing date"), ("he", "תאריך השידור")):
        for sentence in (daily["consequence"][locale], daily["remedy"][locale]):
            assert "Wally_2025-04-27.csv" in sentence, f"the file that won is unnamed in {locale}"
        assert rule in daily["consequence"][locale], f"the rule that decides is unnamed in {locale}"


def test_no_shipped_card_pairs_two_claims_that_disagree(client: TestClient) -> None:
    """The same sweep over the repository's own tree rather than a fixture."""
    for entry in client.get("/api/uploads/status").json()["inputs"]:
        for locale in ("en", "he"):
            assert not _disagrees(entry, locale), f"{entry['kind']} says both things in {locale}"



def test_a_stored_file_the_engine_does_not_read_is_a_row_on_the_file_audit(
    client: TestClient, monkeypatch
) -> None:
    """No dead end: a filename a card prints resolves to a row on the file list.

    The path is a real one in the repository, so the record and its size and
    time are real. What is stubbed is only which kind claims it, because the
    alternative is writing a second daily file into a shared working tree.
    """
    if not LIVE_DAILY.exists():
        pytest.skip("no daily file in the repository to claim as a stored one")
    record = {
        "filename": LIVE_DAILY.name,
        "path": f"data/daily_input/{LIVE_DAILY.name}",
        "rows": 175,
        "size_bytes": LIVE_DAILY.stat().st_size,
        "last_modified": None,
        "arrived_after_live": True,
        "reason": uploads_status.stored_reason("arrived_after_the_file_that_is_read", "Wally_other.csv"),
    }
    monkeypatch.setattr(uploads, "stored_unread_files", lambda kind: [record] if kind == "daily" else [])
    body = client.get("/api/files").json()
    stored = body["stored"]
    assert [row["path"].replace("\\", "/") for row in stored] == [record["path"]]
    assert stored[0]["in_use"] is False, "a file nothing reads is never counted as one that is"
    assert stored[0]["role"] == "input"
    assert stored[0]["note"]["he"] and stored[0]["note"]["en"]
    assert len(body["files"]) == 8, "the audited set the source-file report counts is unchanged"
