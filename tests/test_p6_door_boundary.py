"""P6 Sources, the door: a live check may print no rival channel name either.

Split out of ``test_p6_boundary.py`` under the file-size cap, itself already at
the cap once these three tests were written. The gap they close: the boundary
sweep in :mod:`kairos_api.uploads_replay` ran on a stored report read back
through :func:`rendered` and nowhere else, so a frozen contract's own English
detail, quoting a rival channel name verbatim, reached ``POST /{kind}/check``
and, from there, whatever :func:`kairos_api.uploads_validate.store_report`
went on to persist, unswept, live, in both languages, before it ever touched
disk. ``kairos_api.uploads_replay.at_the_door`` now runs the same two sweep
functions on that live payload, so message, message_he, errors and warnings
pass the same boundary leaving the door that they pass leaving the store.

The withheld notice is also two sentences now, not one claiming a cause it
never checked: :data:`kairos_api.uploads_replay.WITHHELD_STORED` for a
sentence recorded earlier under a different channel, and
:data:`kairos_api.uploads_replay.WITHHELD_LIVE` for a rival name found inside
the file just checked, which is certain the moment nothing has been written
yet. Neither may send a reader back to the same leak.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.uploads as uploads
from kairos.data.loaders import CHANNELS


@pytest.fixture()
def owned(monkeypatch) -> str:
    """A configured operator channel, pinned rather than read off disk."""
    from kairos_api import channel_scope

    channel = CHANNELS[-1]
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: channel)
    return channel


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(uploads.router)
    return TestClient(app)


@pytest.fixture()
def store(tmp_path, monkeypatch) -> Path:
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


def _rivals(owned: str) -> list[str]:
    return [channel for channel in CHANNELS if channel != owned]


def _named(payload, owned: str) -> list[str]:
    """Every channel in a payload that this operator does not own."""
    body = json.dumps(payload, ensure_ascii=False)
    return [rival for rival in _rivals(owned) if rival in body]


def _validation(client: TestClient, kind: str) -> dict:
    """The stored report the status hands back for one input."""
    entry = next(item for item in client.get("/api/uploads/status").json()["inputs"] if item["kind"] == kind)
    assert entry["last_validation"], f"the stored {kind} report is not on the status at all"
    return entry["last_validation"]


def _rival_spots(rival: str) -> bytes:
    """A spots export whose Channel column reads a rival name with a stray
    trailing space, the exact case ``uploads_channels.withhold``'s own
    docstring says the substring rule exists for. The loader's exact match
    fails it into ``unknown_channel`` rather than recognising it."""
    frame = pd.DataFrame(
        {
            "Date": ["01/05/2025", "01/05/2025", "01/05/2025"],
            "Start time": ["20:00:00", "20:01:00", "20:02:00"],
            "Campaign": ["Acme", "Acme", "Acme"],
            "Channel": [f"{rival} ", f"{rival} ", f"{rival} "],
            "Duration": [30, 30, 30],
            "TVR": [5.5, 5.5, 5.5],
        }
    )
    return frame.to_csv(index=False).encode("utf-8")


def test_a_live_check_carrying_a_rival_name_returns_neither_the_name_nor_a_false_cause(
    client: TestClient, owned: str
) -> None:
    """The measured leak: a spots file whose Channel column reads a rival name
    with a stray trailing space, so the frozen contract's own English detail
    quotes it verbatim. Before this fix, that quoted sentence, and the Hebrew
    count beside it, left ``POST /{kind}/check`` unswept in both languages,
    live, never written to disk at all. The notice that replaces it may not
    claim a cause this live check never measured either: nothing was recorded
    earlier under a different channel here.
    """
    rival = _rivals(owned)[0]
    body = client.post(
        "/api/uploads/spots/check",
        files={"file": ("Spots.csv", _rival_spots(rival), "text/csv")},
    ).json()
    finding = next(f for f in body["findings"] if f["code"] == "unknown_channel")
    assert _named(body, owned) == [], "the live door named a channel this operator does not own"
    assert finding.get("message_he"), "the withheld notice is not readable in Hebrew"
    assert "configured" not in finding["message"], "the door claimed a channel-history cause it never measured"
    assert "אחר" not in finding["message_he"], "the Hebrew door claimed a channel-history cause it never measured"


def test_a_commit_carrying_a_rival_name_stores_the_same_notice_the_door_gave(
    isolated: TestClient, store: Path, owned: str
) -> None:
    """The doubled leak, closed: one finding, shown one way, not two.

    Before this fix, a live check and the stored report it went on to become
    disagreed about the same finding: the live response printed the rival
    name and the later read replaced it with a notice about a channel that was
    never in force. Sweeping once, before either exit, leaves only one
    sentence to disagree with itself.
    """
    rival = _rivals(owned)[0]
    raw = _rival_spots(rival)
    live = isolated.post("/api/uploads/spots/check", files={"file": ("Spots.csv", raw, "text/csv")}).json()
    committed = isolated.post("/api/uploads/spots", files={"file": ("Spots.csv", raw, "text/csv")})
    assert committed.status_code == 200, "an unknown channel name is a warning, not a refusal"
    later = _validation(isolated, "spots")
    live_finding = next(f for f in live["findings"] if f["code"] == "unknown_channel")
    later_finding = next(f for f in later["findings"] if f["code"] == "unknown_channel")
    assert live_finding["message"] == later_finding["message"], "the live check and the stored read disagree"
    for half in ("message_en", "message_he"):
        assert live_finding.get(half) == later_finding.get(half), f"the {half} halves disagree"
    assert _named(later, owned) == [], "the stored read named a channel this operator does not own"


def test_the_live_and_stored_notices_are_two_sentences_and_neither_invents_a_cause() -> None:
    """Split, per the measured defect: a stored read cannot know whether a
    withheld sentence came from a different channel or from a live content
    collision, so it may not claim the first, and a live check may not send a
    reader who just read the freshest answer back to read it again."""
    from kairos_api import uploads_replay

    assert uploads_replay.WITHHELD_LIVE != uploads_replay.WITHHELD_STORED
    for notice in (uploads_replay.WITHHELD_LIVE, uploads_replay.WITHHELD_STORED):
        assert "configured" not in notice["en"], "a withheld notice claimed an unmeasured channel-history cause"
        assert notice["he"], "a withheld notice has no Hebrew half"
    assert "check the file again" not in uploads_replay.WITHHELD_LIVE["en"], (
        "the live notice sent a reader of the freshest answer back to read it again"
    )
