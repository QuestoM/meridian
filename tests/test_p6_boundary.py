"""P6 Sources: no channel this operator does not own reaches this destination.

Two leaks were measured through the shipped surface before this file existed,
with an operator channel configured in settings.

**The refusal panel printed three of them.** ``POST /api/uploads/dayparts/check``
on a plausibly re-exported dayparts file, the shipped ``data/Dayparts.csv`` with
its channel headers renamed, answered one finding whose message listed all four
channels the loader knows. ``SourceCard`` renders a finding's message verbatim
inside its red panel, so ``document.body.innerText`` then carried three channel
names this account does not own.

**The status carried them too**, which the sweep below found and no reader had.
``GET /api/uploads/status`` returned the dayparts input's real header, which is
one column per channel, in the payload the browser and the assistant's own read
tool both receive.

**And a third: the boundary was applied when a refusal was WRITTEN and never
when it was read back.** Measured on the shipped card with the operator channel
``עכשיו 14``: a dayparts refusal stored earlier, under ``רשת 13``, replayed that
name three times through ``GET /api/uploads/status`` and the card printed it as
this account's own channel. The sweep below was green at the time only because
the store on disk happened to be empty, which makes a boundary assertion whose
verdict depends on whether anybody has uploaded a file. It no longer can be:
the last three tests here plant a stored report and read it back under another
channel, so the store is never the reason this file passes.

**And a fourth, which this file was certifying without ever asking for it.**
The sweep below reads ``response.json()``, and the two routes on this piece's
row that carry more channel names than any payload in the product do not answer
JSON: ``GET /api/export/schedule.csv`` streams the saved plan file whole.
Measured with the shipped settings, that file is 8,704 data rows of which 6,164
are on three channels this operator does not own, each row naming its channel
twice and carrying that channel's own predicted revenue. It is one click from
the report card. A test named for a boundary that skips the routes most likely
to breach it is worse than no test, so those two routes are swept in
``tests/test_p6_downloads.py``, which is a second file because this one reached
the 450-line law and a law is not dodged by writing shorter reasons. A test
there requires the two sweeps together to be every GET route on this piece's
row, so a route under neither of them is red on the day it is added.

The sweep is the test. Every route this piece owns is asked for what it answers
and every one is required to name no channel but the configured one, so a fifth
place to leak fails here rather than on a screen. The operator channel is
pinned to a channel the shipped settings do NOT name, because a filter that
happens to hide three particular names would pass a test that pinned the same
channel the product is configured with.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.downloads_api as downloads_api
import kairos_api.uploads as uploads
import kairos_api.uploads_preview as uploads_preview
from kairos.data.loaders import CHANNELS

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "tv-break-dashboard" / "src" / "sources"

KINDS = ("programmes", "spots", "dayparts", "daily", "advertiser_rules", "rate_card", "campaign_flights")
REPORTS = ("weekly-plan", "compliance", "revenue", "daily-spots", "data-quality")


@pytest.fixture()
def owned(monkeypatch):
    """A configured operator channel, pinned rather than read off disk.

    The settings file is shared writable state, and a boundary assertion that
    another process can turn into a skip is not an assertion.
    """
    from kairos_api import channel_scope, read_cache

    channel = CHANNELS[-1]
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: channel)
    read_cache.invalidate(uploads_preview.PREVIEW_NAMESPACE)
    # The report shelf is memoised on the settings file's own signature, which
    # a patched accessor does not move, so the memo is dropped either side of
    # this test rather than served from before the channel was pinned.
    downloads_api._reports_cached.cache_clear()
    yield channel
    downloads_api._reports_cached.cache_clear()


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(uploads.router)
    app.include_router(downloads_api.router)
    return TestClient(app)


@pytest.fixture()
def store(tmp_path, monkeypatch) -> Path:
    """The report store, relocated to the same path ``isolated`` relocates it to.

    Both fixtures point at one file, so a test may take both in either order and
    still read back what the door it just knocked on wrote.
    """
    path = tmp_path / "output" / "upload_validation_reports.json"
    monkeypatch.setattr(uploads, "VALIDATION_REPORTS_PATH", path)
    return path


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


def _renamed_dayparts() -> bytes:
    """A dayparts export whose channel headers were renamed on the way out.

    The realistic morning file: every value is real and the header is a plain
    English re-export, so the loader melts nothing and the file yields zero
    audience rows while passing the Dates+Timebands header gate.
    """
    frame = pd.DataFrame(
        {
            "Dates": ["01/05/2025", "02/05/2025"],
            "Timebands": ["20:00", "20:01"],
            "Channel 14": ["1.0", "2.0"],
            "Channel 12": ["3.0", "4.0"],
            "Channel 13": ["5.0", "6.0"],
            "Channel 11": ["7.0", "8.0"],
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def _real_dayparts() -> bytes:
    """A dayparts export with the four headers the loader really recognizes."""
    frame = pd.DataFrame(
        {
            "Dates": ["01/05/2025", "02/05/2025"],
            "Timebands": ["20:00", "20:01"],
            **{channel: ["1.0", "2.0"] for channel in CHANNELS},
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def _rivals(owned: str) -> list[str]:
    return [channel for channel in CHANNELS if channel != owned]


def _named(payload, owned: str) -> list[str]:
    """Every channel in a payload that this operator does not own."""
    body = json.dumps(payload, ensure_ascii=False)
    return [rival for rival in _rivals(owned) if rival in body]


def test_no_route_of_this_destination_names_a_channel_the_operator_does_not_own(
    client: TestClient, owned: str
) -> None:
    """The sweep, over every route on this piece's row that answers JSON.

    The two that answer a streamed file are swept in ``tests/test_p6_downloads.py``,
    which also asserts that the two lists together are every GET route on this
    row, so the split between them is a difference in how a body is read and
    never a hole.
    """
    paths = ["/api/uploads/status", "/api/files", "/api/reports"]
    paths += [f"/api/uploads/{kind}/preview?limit=100" for kind in KINDS]
    paths += [f"/api/reports/{report}/preview" for report in REPORTS]
    for path in paths:
        response = client.get(path)
        if response.status_code == 404:
            continue
        assert response.status_code == 200, f"{path} answered {response.status_code}"
        assert _named(response.json(), owned) == [], f"{path} named a channel this operator does not own"

    for name, payload in (("renamed", _renamed_dayparts()), ("recognized", _real_dayparts())):
        body = client.post(
            "/api/uploads/dayparts/check", files={"file": ("Dayparts.csv", payload, "text/csv")}
        ).json()
        assert _named(body, owned) == [], f"the door named a rival channel on the {name} file"


def test_the_dayparts_refusal_names_the_operators_own_channel_and_no_other(
    client: TestClient, owned: str
) -> None:
    """The measured leak, at the exact place it was measured.

    The refusal must stay actionable: it still says why the file yields nothing
    and still lists the headers it did not recognise, so the export can be
    fixed. What it may not do is answer the question "which names does the
    loader know" with a list of this operator's competitors.
    """
    body = client.post(
        "/api/uploads/dayparts/check", files={"file": ("Dayparts.csv", _renamed_dayparts(), "text/csv")}
    ).json()
    assert body["accepted"] is False, "a file that yields zero audience rows is refused at the door"
    finding = next(f for f in body["findings"] if f["code"] == "no_recognized_channel_columns")
    assert owned in finding["message"], "the refusal does not name the one channel it may name"
    for rival in _rivals(owned):
        assert rival not in finding["message"], "the refusal named a channel this operator does not own"
    assert "Channel 12" in finding["message"], "the refusal stopped naming the headers it did not recognise"
    assert finding["severity"] == "error"
    # The Hebrew half of the same refusal is a second string on the same screen,
    # so the boundary is checked on it directly rather than only by the sweep.
    assert owned in finding["message_he"], "the Hebrew refusal does not name the one channel it may name"
    for rival in _rivals(owned):
        assert rival not in finding["message_he"], "the Hebrew refusal named a channel this operator does not own"


def test_the_refusal_says_where_to_set_the_channel_when_there_is_none(client: TestClient, monkeypatch) -> None:
    """With no channel configured there is no name to check a header against,
    and every channel is then a name this account may not own."""
    from kairos_api import channel_scope

    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: "")
    body = client.post(
        "/api/uploads/dayparts/check", files={"file": ("Dayparts.csv", _renamed_dayparts(), "text/csv")}
    ).json()
    finding = next(f for f in body["findings"] if f["code"] == "no_recognized_channel_columns")
    for channel in CHANNELS:
        assert channel not in finding["message"], "a refusal named a channel with no owner configured"
    assert "settings" in finding["message"], "the refusal does not say where to set the operator channel"


def test_withholding_a_name_never_moves_the_number_of_columns(client: TestClient, owned: str) -> None:
    """The count is the disclosure. A file audit whose column count fell by
    three because three names were withheld would be a false figure, which is
    the one thing the boundary may not buy itself."""
    entry = next(
        item for item in client.get("/api/uploads/status").json()["inputs"] if item["kind"] == "dayparts"
    )
    if not entry["exists"]:
        pytest.skip("no dayparts file on disk to count the columns of")
    real = pd.read_csv(ROOT / entry["path"], encoding="utf-8-sig", nrows=1).columns
    assert entry["columns_withheld"] == len([c for c in real if str(c) in _rivals(owned)])
    assert entry["columns_withheld"] > 0, "the shipped dayparts header names channels this account does not own"
    assert len(entry["columns"]) + entry["columns_withheld"] == len(real), "the column count moved"


def test_a_commit_echoes_no_channel_the_operator_does_not_own(isolated: TestClient, owned: str) -> None:
    """The header a commit answers with is the header of the file just handed
    over, and a dayparts file names a channel in every one of its columns."""
    body = isolated.post(
        "/api/uploads/dayparts", files={"file": ("Dayparts.csv", _real_dayparts(), "text/csv")}
    ).json()
    assert _named(body, owned) == [], "the commit echoed a channel this operator does not own"
    assert body["columns_withheld"] == len(_rivals(owned))
    assert len(body["columns"]) + body["columns_withheld"] == 2 + len(CHANNELS), "the column count moved"
    assert owned in body["columns"], "the operator's own channel column is not withheld from them"


def _validation(client: TestClient, kind: str) -> dict:
    """The stored report the status hands back for one input."""
    entry = next(item for item in client.get("/api/uploads/status").json()["inputs"] if item["kind"] == kind)
    assert entry["last_validation"], f"the stored {kind} report is not on the status at all"
    return entry["last_validation"]


def test_a_report_stored_under_one_channel_is_read_under_the_channel_owned_now(
    isolated: TestClient, store: Path, monkeypatch
) -> None:
    """The measured leak: a refusal written under one channel, read under another.

    The refusal is stored by the commit route BEFORE it answers 400, so the
    sentence, with the operator channel resolved at that instant, used to be
    frozen on disk and replayed verbatim on every later read. This is the same
    two moments, one test: written under the first channel, read under the last.
    """
    from kairos_api import channel_scope

    current = {"channel": CHANNELS[0]}
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: current["channel"])
    refused = isolated.post(
        "/api/uploads/dayparts", files={"file": ("Dayparts.csv", _renamed_dayparts(), "text/csv")}
    )
    assert refused.status_code == 400, "a dayparts export with no channel column is refused at the door"
    assert current["channel"] in json.dumps(refused.json(), ensure_ascii=False), (
        "the refusal did not name the channel configured when it was written"
    )

    current["channel"] = CHANNELS[-1]
    body = isolated.get("/api/uploads/status").json()
    assert _named(body, current["channel"]) == [], "a stored refusal printed a channel this account does not own"
    validation = _validation(isolated, "dayparts")
    finding = next(f for f in validation["findings"] if f["code"] == "no_recognized_channel_columns")
    assert current["channel"] in finding["message"], "the reason stopped naming the channel this account owns"
    assert current["channel"] in finding["message_he"], "the Hebrew reason stopped naming it too"
    assert "Channel 12" in finding["message"], "the headers it did not recognise are no longer listed"
    assert validation["errors"], "the flat list the assistant's read tool parses lost the reason"


def test_the_stored_report_freezes_no_channel_name_and_no_sentence(
    isolated: TestClient, store: Path, owned: str
) -> None:
    """What is on disk is what makes the test above true, so it is asserted too.

    A rendered sentence on disk is a channel name frozen at the moment it was
    written. The store holds the finding's code and the fields it was rendered
    from instead, and the one field that depends on the boundary is held as the
    names this account may read plus the count of the ones it may not.
    """
    isolated.post("/api/uploads/dayparts", files={"file": ("Dayparts.csv", _renamed_dayparts(), "text/csv")})
    raw = store.read_text(encoding="utf-8")
    for channel in CHANNELS:
        assert channel not in raw, f"the stored report froze the channel name {channel}"
    record = json.loads(raw)["dayparts"]
    assert record["renders_at_read"] is True, "the store does not say it holds codes"
    finding = next(f for f in record["findings"] if f["code"] == "no_recognized_channel_columns")
    assert "message" not in finding and "message_he" not in finding, "the store kept the sentence"
    assert finding["boundary"]["names"], "the store kept nothing to render the sentence from"
    assert finding["severity"] == "error"


def test_a_reason_stored_before_this_format_is_withheld_rather_than_replayed(
    isolated: TestClient, store: Path, monkeypatch
) -> None:
    """The two reports already on disk carry sentences and no fields.

    Nothing can re-author them, so the rule is the boundary's own: a sentence
    naming a channel this account may not read is withheld, in both languages,
    with what to do about it, and the code and the severity a reader acts on
    stay. The same record, read by the account that owns that channel, is
    printed exactly as it was stored.
    """
    from kairos_api import channel_scope

    stranger = CHANNELS[0]
    sentence = f"the only one this account may be shown is your own channel {stranger}"
    store.parent.mkdir(parents=True, exist_ok=True)
    store.write_text(
        json.dumps(
            {
                "dayparts": {
                    "dataset": "dayparts",
                    "filename": "Dayparts.csv",
                    "checked_at": "2026-07-31T09:00:00+00:00",
                    "accepted": False,
                    "is_valid": False,
                    "rows_loaded": 0,
                    "errors": [f"[error] channels: no_recognized_channel_columns - {sentence}"],
                    "warnings": [],
                    "findings": [
                        {
                            "column": "channels",
                            "code": "no_recognized_channel_columns",
                            "message": sentence,
                            "message_he": sentence,
                            "severity": "error",
                        }
                    ],
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: stranger)
    kept = _validation(isolated, "dayparts")
    assert sentence in kept["findings"][0]["message"], "the account that owns that channel lost its own reason"

    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: CHANNELS[-1])
    withheld = _validation(isolated, "dayparts")
    assert _named(withheld, CHANNELS[-1]) == [], "a report stored before this format replayed a rival channel"
    finding = withheld["findings"][0]
    assert finding["code"] == "no_recognized_channel_columns", "the code a reader acts on went with the sentence"
    assert finding["severity"] == "error", "the severity went with it too"
    assert finding["message_he"], "the notice that replaced the reason is not readable in Hebrew"
    assert withheld["errors"] and _named(withheld["errors"], CHANNELS[-1]) == [], "the flat list replayed it"


def test_the_surface_carries_no_channel_name_of_its_own(owned: str) -> None:
    """The server decides which channel a screen may name, so the destination
    hardcodes none of them: not the operator's, and not anybody else's."""
    for path in sorted(SOURCES.iterdir()):
        if path.suffix not in (".jsx", ".js", ".css"):
            continue
        body = path.read_text(encoding="utf-8")
        for channel in CHANNELS:
            assert channel not in body, f"{path.name} hardcodes the channel name {channel}"
