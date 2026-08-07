"""P6 Sources, the door: the count inside a contract's Hebrew sentence.

Split out of ``test_p6_door.py`` under the file-size cap, itself already at the
cap. Two things measured wrong on the shipped card and closed here, one seam:
the count :func:`kairos_api.uploads_validate.finding_records` puts into a
frozen contract's Hebrew is that code's own quantity, recomputed on the frame
the contract validated, never a row count assumed to fit every code and never
withheld just because that frame is a melted one no row number may print off.
"""

from __future__ import annotations

import re

import pandas as pd
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

import kairos_api.uploads as uploads

HEBREW = re.compile(r"[֐-׿]")


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


def test_a_melted_kind_still_gets_its_hebrew(isolated: TestClient) -> None:
    """No row number may print off a melted frame, but a Hebrew count still can.

    The measured gap: this exact request answered with the English
    ``"1 value(s) below zero"`` and a null ``message_he``, because the count
    the Hebrew needed was read off ``rows_total``, which a melted kind never
    carries. The count is now recomputed on the loaded frame directly,
    independent of whether ``rows`` may ever be printed for this kind.
    """
    frame = pd.DataFrame(
        {"Dates": ["01/05/2025", "02/05/2025"], "Timebands": ["20:00", "20:01"], "רשת 13": ["1.0", "-2.0"]}
    )
    body = isolated.post(
        "/api/uploads/dayparts/check",
        files={"file": ("Dayparts.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv")},
    ).json()
    finding = next(f for f in body["findings"] if f["code"] == "negative_values")
    assert "rows" not in finding, "a melted frame must still invent no row number"
    assert HEBREW.search(finding.get("message_he") or ""), "a melted kind's refusal still reads English"
    assert "1" in finding["message_he"], "the one value below zero is not the count in the Hebrew sentence"


def test_unknown_channel_hebrew_counts_names_not_rows(isolated: TestClient) -> None:
    """``{count}`` in ``unknown_channel``'s Hebrew names channels, not rows.

    The measured gap: forty rows all carrying one unknown channel name
    printed Hebrew that opened with forty, while the English beside it, on
    the same finding, listed one name. Three rows here carry two distinct
    unknown names between them, so the honest count is two.
    """
    spots = pd.DataFrame(
        {
            "Date": ["01/05/2025", "01/05/2025", "01/05/2025"],
            "Start time": ["20:00:00", "20:01:00", "20:02:00"],
            "Campaign": ["Acme", "Acme", "Acme"],
            "Channel": ["זרזיר 99", "זרזיר 99", "ערוץ פלוני"],
            "Duration": [30, 30, 30],
            "TVR": [5.5, 5.5, 5.5],
        }
    )
    body = isolated.post(
        "/api/uploads/spots/check",
        files={"file": ("Spots.csv", spots.to_csv(index=False).encode("utf-8"), "text/csv")},
    ).json()
    finding = next(f for f in body["findings"] if f["code"] == "unknown_channel")
    assert finding["rows_total"] == 3, "three rows carry an unknown channel name"
    assert HEBREW.search(finding.get("message_he") or ""), "the unknown-channel warning still reads English"
    assert "2" in finding["message_he"], "the Hebrew count must be the two NAMES, not the three rows"
    assert "3" not in finding["message_he"], "the row count leaked into a sentence about channel names"
