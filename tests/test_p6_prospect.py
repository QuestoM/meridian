"""P6 Sources: what the door predicts about one file, against what really happens.

Split out of ``test_p6_state.py`` under the 450-line law, and by subject: every
test here posts a candidate to ``/check`` and then grades that prediction
against the outcome of committing the same file, which is a different question
from the state an input is in once its files are on disk.

The measured lie these exist for: the door derived its answer from the read path
of the KIND and never from the candidate's own name, so two daily files with
opposite outcomes got byte-identical answers, and committing the one the door
called the live input replaced nothing. Every prediction below is compared with
what the product then reports, because a door that is merely self-consistent is
how a round passes its own tests while telling a steward the opposite of what
will happen.

Same fixtures as ``test_p6_state.py``: every writable path is relocated, so
nothing here can touch a repository input.
"""

from __future__ import annotations

import os

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.uploads as uploads
from kairos.data.loaders import DAILY_COLUMN_MAP


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


def _check_daily(client: TestClient, name: str, date_value: str) -> dict:
    response = client.post(
        "/api/uploads/daily/check", files={"file": (name, _daily_bytes(date_value=date_value), "text/csv")}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["accepted"] is True, f"{name} is a valid morning file: {body.get('errors')}"
    return body


def test_the_door_answers_for_the_candidate_and_not_for_its_kind(isolated: TestClient) -> None:
    """The measured lie, and the only moment at which the answer still matters.

    Two daily files, identical but for the airing date in the name, opposite
    outcomes, and the door gave both the same answer: this is the live input,
    uploading replaces what the plan is computed from, replacing the same named
    file. It came from ``_in_use(kind)`` with no candidate in it, so the read
    path of the kind stood in for the fate of the file. Committing the older one
    replaced nothing, which the commit response said one line later and the
    moment of decision did not.
    """
    live = isolated.post(
        "/api/uploads/daily", files={"file": ("Wally_2025-04-27.csv", _daily_bytes(), "text/csv")}
    )
    assert live.status_code == 200, live.text
    live_path = uploads.DAILY_DIR / "Wally_2025-04-27.csv"
    os.utime(live_path, (1_700_000_000, 1_700_000_000))

    older = _check_daily(isolated, "Wally_2019-03-04.csv", "3/4/2019")
    newer = _check_daily(isolated, "Wally_2099-12-31.csv", "12/31/2099")

    assert older["consequence"] != newer["consequence"], "two opposite outcomes, one answer"
    assert older["consequence"]["code"] == "stored_without_replacing"
    assert older["will_be_read"] is False
    assert older["replaces"] is None, "a file that will not be read replaces nothing"
    assert older["saves_to"].endswith("Wally_2019-03-04.csv"), "and it names where it would land"
    assert older["engine_reads_after"] == str(live_path), "the file that will still be read is named"
    for locale in ("en", "he"):
        assert "Wally_2025-04-27.csv" in older["consequence"][locale], f"unnamed in {locale}"
    assert older["will_be_read_reason"], "the honest reason travels with the false verdict"

    assert newer["consequence"]["code"] == "replaces_live_input"
    assert newer["will_be_read"] is True
    assert newer["replaces"] == str(live_path), "the file it takes the place of is named"
    assert newer["engine_reads_after"] == newer["saves_to"], "the engine will read the new file"


def test_what_the_door_predicted_is_what_committing_really_does(isolated: TestClient) -> None:
    """The door is graded against the outcome, never against a second opinion.

    Both branches are committed through the surface and every prediction is
    compared with what the product then reports, because a door that is merely
    self-consistent is how the previous round passed its own tests while
    telling a steward the opposite of what would happen.
    """
    assert isolated.post(
        "/api/uploads/daily", files={"file": ("Wally_2025-04-27.csv", _daily_bytes(), "text/csv")}
    ).status_code == 200
    os.utime(uploads.DAILY_DIR / "Wally_2025-04-27.csv", (1_700_000_000, 1_700_000_000))

    predicted = _check_daily(isolated, "Wally_2019-03-04.csv", "3/4/2019")
    committed = isolated.post(
        "/api/uploads/daily",
        files={"file": ("Wally_2019-03-04.csv", _daily_bytes(date_value="3/4/2019"), "text/csv")},
    ).json()
    assert committed["in_use"] is predicted["will_be_read"], "the door and the outcome disagree"
    assert committed["engine_reads"] == predicted["engine_reads_after"]
    assert committed["saved_path"] == predicted["saves_to"]
    assert committed["in_use_reason"] == predicted["will_be_read_reason"]
    # The measured half that moved rather than closed: the door said amber, the
    # confirmation said "this is the live input, uploading replaces what the
    # plan is computed from", in a body whose own in_use was false, because the
    # commit derived its consequence from the KIND. Same file, same sentence.
    assert committed["consequence"] == predicted["consequence"], "the toast contradicts the door"
    assert committed["consequence"]["code"] == "stored_without_replacing"
    assert "Wally_2025-04-27.csv" in committed["consequence"]["he"], "unnamed in the language it is read in"

    entry = next(
        item for item in isolated.get("/api/uploads/status").json()["inputs"] if item["kind"] == "daily"
    )
    assert entry["state"] == "shadowed", "the card flipped, exactly as the door said it would"

    predicted = _check_daily(isolated, "Wally_2099-12-31.csv", "12/31/2099")
    committed = isolated.post(
        "/api/uploads/daily",
        files={"file": ("Wally_2099-12-31.csv", _daily_bytes(date_value="12/31/2099"), "text/csv")},
    ).json()
    assert committed["in_use"] is predicted["will_be_read"] is True
    assert committed["engine_reads"] == predicted["engine_reads_after"]
    assert committed["consequence"] == predicted["consequence"], "the toast contradicts the door"
    assert committed["consequence"]["code"] == "replaces_live_input"
    entry = next(
        item for item in isolated.get("/api/uploads/status").json()["inputs"] if item["kind"] == "daily"
    )
    assert entry["filename"] == "Wally_2099-12-31.csv", "the engine reads the file the door said it would"
    assert entry["state"] == "in_use"


def test_a_file_with_no_date_in_its_name_is_ranked_as_the_day_it_would_land(isolated: TestClient) -> None:
    """The realistic morning file is called whatever the inbox called it.

    The resolver falls back to the day a dateless file landed, so a candidate
    with no date in its name must be ranked as today and not as the day zero a
    missing date would otherwise sort to. The name it would be stored under is
    also not the name that was chosen, and the door says so before the click.
    """
    assert isolated.post(
        "/api/uploads/daily", files={"file": ("Wally_2025-04-27.csv", _daily_bytes(), "text/csv")}
    ).status_code == 200
    os.utime(uploads.DAILY_DIR / "Wally_2025-04-27.csv", (1_700_000_000, 1_700_000_000))

    body = _check_daily(isolated, "morning.csv", "4/27/2025")
    assert body["will_be_read"] is True, "a file landing today outranks a file airing last year"
    assert body["consequence"]["code"] == "replaces_live_input"
    assert body["saves_to"].endswith("Wally_morning.csv"), "the prefix the engine globs is named at the door"
    assert body["engine_reads_after"] == body["saves_to"]

    same = _check_daily(isolated, "Wally_2025-04-27.csv", "4/27/2025")
    assert same["will_be_read"] is True, "overwriting the live file leaves it live"
    assert same["replaces"] == same["saves_to"] == str(uploads.DAILY_DIR / "Wally_2025-04-27.csv")


def test_the_door_names_the_workbook_that_shadows_a_channel_source(isolated: TestClient) -> None:
    """The same question on the kind whose upload is shadowed by a workbook.

    ``data/reference/Programmes.xlsx`` is read before the uploaded CSV, so the
    consequence for a programmes file is that it changes nothing, and the door
    names the file that will be read instead of it rather than saying only that
    nothing reads this one.
    """
    reference = uploads.SHADOWING_REFERENCE["programmes"]
    if not reference.exists():
        pytest.skip("no reference workbook on disk to shadow the upload with")
    frame = pd.DataFrame(
        [
            {
                "Title": "Evening Show",
                "Channel": "רשת 13",
                "Date": "27/04/2025",
                "Start time": "18:00",
                "End time": "19:00",
                "Duration": "3600",
            }
        ]
    )
    body = isolated.post(
        "/api/uploads/programmes",
        files={"file": ("Programmes.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv")},
    )
    assert body.status_code == 200, body.text
    checked = isolated.post(
        "/api/uploads/programmes/check",
        files={"file": ("Programmes.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv")},
    ).json()
    assert checked["accepted"] is True, checked.get("errors")
    assert checked["will_be_read"] is False
    assert checked["replaces"] is None
    assert checked["consequence"]["code"] == "stored_without_replacing"
    for locale in ("en", "he"):
        assert "Programmes.xlsx" in checked["consequence"][locale], f"the workbook is unnamed in {locale}"
