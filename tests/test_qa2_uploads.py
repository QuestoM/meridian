"""Upload honesty: size caps, daily-name truth, contract gates, atomic writes.

Every writable path (data dir, daily dir, backups, the validation-report store)
is relocated to tmp, so no repository input file is touched. The reference
xlsx workbooks are only ever read (shadow checks), never written.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
from datetime import datetime

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile as StarletteUploadFile
from starlette.requests import Request as StarletteRequest

import kairos_api.uploads as uploads
from kairos.data.loaders import DAILY_COLUMN_MAP, count_ambiguous_daily_dates

ROOT = uploads.ROOT


@pytest.fixture()
def isolated(tmp_path, monkeypatch) -> TestClient:
    """Relocate every path the router writes so no repo input is touched."""
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


def _post(client: TestClient, kind: str, name: str, data: bytes):
    return client.post(f"/api/uploads/{kind}", files={"file": (name, data, "text/csv")})


def _daily_bytes(*, date_value: str = "4/27/2025", duration: str = "30", planned: str = "5.5") -> bytes:
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
    row["רייטינג ברייקים מתוכנן"] = planned
    frame = pd.DataFrame([row], columns=list(DAILY_COLUMN_MAP.keys()))
    return frame.to_csv(index=False).encode("utf-8")


FLIGHTS_HEADER = ",".join(uploads.REQUIRED_COLUMNS["campaign_flights"])


def _flights_bytes(rows: list[str]) -> bytes:
    return ("\n".join([FLIGHTS_HEADER, *rows]) + "\n").encode("utf-8")


# --- size caps (streamed, never whole-body-in-memory) ----------------------------
def test_content_length_precheck_rejects_before_reading_the_body() -> None:
    """An honestly-declared oversize body is refused before ANY read."""

    class _Boom:
        async def read(self, size: int = -1) -> bytes:
            raise AssertionError("the body must not be read after the Content-Length pre-check")

    request = StarletteRequest(
        {
            "type": "http",
            "method": "POST",
            "path": "/api/uploads/daily",
            "headers": [(b"content-length", str(uploads.MAX_UPLOAD_BYTES * 2).encode())],
        }
    )
    response = asyncio.run(uploads.upload_file("daily", request, _Boom()))
    assert response.status_code == 400
    assert b"limit" in response.body


def test_streamed_body_over_the_cap_is_rejected(monkeypatch) -> None:
    """A body that undershoots the declared pre-check but crosses the cap while
    streaming is rejected at the crossing (cap lowered so the test stays small:
    5 KB body, 1 KB cap, 66 KB pre-check threshold, so only the stream can trip)."""
    monkeypatch.setattr(uploads, "MAX_UPLOAD_BYTES", 1024)
    request = StarletteRequest(
        {"type": "http", "method": "POST", "path": "/api/uploads/daily", "headers": []}
    )
    upload = StarletteUploadFile(file=io.BytesIO(b"x" * 5000), filename="Wally_big.csv")
    response = asyncio.run(uploads.upload_file("daily", request, upload))
    assert response.status_code == 400
    assert b"limit" in response.body


def test_cap_is_generously_above_the_largest_real_export() -> None:
    largest = max(
        path.stat().st_size for path in (ROOT / "data" / "reference").glob("*.xlsx")
    )
    assert uploads.MAX_UPLOAD_BYTES >= 10 * largest


# --- parse errors: generic to the client, detailed in the server log -------------
def test_parse_error_is_generic_and_logged(isolated, caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="kairos_api.uploads"):
        response = _post(isolated, "daily", "junk.csv", b"\xff\xfe\x00\x01\x02 not a csv")
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert detail == uploads._GENERIC_PARSE_ERROR
    assert "pandas" not in detail.lower() and "codec" not in detail.lower()
    assert "Upload parse failed" in caplog.text, "the parser detail must be logged server-side"


# --- daily: every stored name matches the pattern the engine resolver reads ------
def test_daily_destination_normalizes_to_wally_pattern() -> None:
    assert uploads._destination("daily", "plan_today.csv").name == "Wally_plan_today.csv"
    assert uploads._destination("daily", "wally_case.csv").name == "Wally_case.csv"
    assert uploads._destination("daily", "Wally_ok.csv").name == "Wally_ok.csv"
    local_today = datetime.now().astimezone().strftime("%Y-%m-%d")
    assert uploads._destination("daily", None).name == f"Wally_{local_today}.csv"


def test_daily_nonwally_upload_is_saved_where_the_engine_reads(isolated) -> None:
    response = _post(isolated, "daily", "plan_today.csv", _daily_bytes())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["saved_path"].endswith("Wally_plan_today.csv")
    assert payload["in_use"] is True
    live = uploads._newest_daily()
    assert live is not None and live.name == "Wally_plan_today.csv"
    body = isolated.get("/api/uploads/status").json()
    daily = next(item for item in body["inputs"] if item["kind"] == "daily")
    assert daily["exists"] is True and daily["valid"] is True
    assert daily["filename"] == "Wally_plan_today.csv"


def test_daily_upload_shadowed_by_newer_airing_date_reports_amber(isolated, tmp_path) -> None:
    daily_dir = uploads.DAILY_DIR
    daily_dir.mkdir(parents=True, exist_ok=True)
    newer = daily_dir / "Wally_future_2031-01-01.csv"
    newer.write_bytes(_daily_bytes())
    os.utime(newer, (1_000_000_000, 1_000_000_000))  # ancient mtime, newest airing date

    response = _post(isolated, "daily", "Wally_old_2025-01-01.csv", _daily_bytes())
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["in_use"] is False, "a save the resolver will not pick must not claim in_use"
    assert "Wally_future_2031-01-01.csv" in payload["in_use_reason"]
    assert payload["engine_reads"].endswith("Wally_future_2031-01-01.csv")


def test_newest_daily_prefers_airing_date_over_mtime(isolated) -> None:
    daily_dir = uploads.DAILY_DIR
    daily_dir.mkdir(parents=True, exist_ok=True)
    older_day = daily_dir / "Wally_a_2025-05-01.csv"
    newer_day = daily_dir / "Wally_b_2025-06-01.csv"
    older_day.write_bytes(_daily_bytes())
    newer_day.write_bytes(_daily_bytes())
    os.utime(newer_day, (1_000_000_000, 1_000_000_000))  # re-uploaded LATER day, OLDER mtime
    live = uploads._newest_daily()
    assert live is not None and live.name == "Wally_b_2025-06-01.csv"


# --- dayparts: renamed channel columns can no longer validate green --------------
def test_dayparts_with_unrecognized_channel_columns_is_rejected(isolated) -> None:
    data = "Dates,Timebands,Channel_A,Channel_B\n01/11/2024,20:00 - 20:01,5.1,4.2\n".encode("utf-8")
    response = _post(isolated, "dayparts", "Dayparts.csv", data)
    assert response.status_code == 400, response.text
    detail = response.json()["detail"]
    assert "zero audience rows" in detail
    assert "Channel_A" in detail, "the unrecognized headers must be named"
    assert "קשת 12" in detail, "the recognized channel set must be named"


def test_dayparts_with_a_real_channel_column_is_accepted(isolated) -> None:
    data = "Dates,Timebands,קשת 12\n01/11/2024,20:00 - 20:01,5.1\n".encode("utf-8")
    response = _post(isolated, "dayparts", "Dayparts.csv", data)
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["valid"] is True
    assert payload["validation"]["rows_loaded"] == 1


# --- contracts are wired: error severity refuses, warnings ride along ------------
def test_daily_contract_error_refuses_and_the_report_is_surfaced(isolated) -> None:
    bad = _post(isolated, "daily", "Wally_bad.csv", _daily_bytes(duration="-5"))
    assert bad.status_code == 400, bad.text
    assert any("duration_sec" in err for err in bad.json()["errors"])
    assert not (uploads.DAILY_DIR.exists() and list(uploads.DAILY_DIR.glob("*.csv"))), (
        "a refused upload must never land on disk"
    )
    body = isolated.get("/api/uploads/status").json()
    daily = next(item for item in body["inputs"] if item["kind"] == "daily")
    assert daily["last_validation"] is not None
    assert daily["last_validation"]["accepted"] is False

    good = _post(isolated, "daily", "Wally_good.csv", _daily_bytes())
    assert good.status_code == 200, good.text
    body = isolated.get("/api/uploads/status").json()
    daily = next(item for item in body["inputs"] if item["kind"] == "daily")
    assert daily["last_validation"]["accepted"] is True
    assert daily["last_validation"]["dataset"] == "daily_input"


def test_daily_contract_warning_rides_along_without_refusing(isolated) -> None:
    response = _post(isolated, "daily", "Wally_warn.csv", _daily_bytes(planned=""))
    assert response.status_code == 200, response.text
    payload = response.json()
    assert any("planned_tvr" in warning for warning in payload["warnings"])
    assert payload["validation"]["is_valid"] is True
    assert payload["validation"]["warnings"]


def test_programmes_with_no_parseable_dates_is_refused(isolated) -> None:
    data = (
        "Title,Channel,Date,Start time,End time,Duration\n"
        "Show,קשת 12,not-a-date,20:00:00,21:00:00,3600\n"
    ).encode("utf-8")
    response = _post(isolated, "programmes", "Programmes.csv", data)
    assert response.status_code == 400, response.text
    assert any("parseable date" in err for err in response.json()["errors"])


# --- ambiguous day/month dates are counted into the upload warnings --------------
def test_count_ambiguous_daily_dates_is_exact() -> None:
    series = pd.Series(["4/5/2025", "4/27/2025", "5/5/2025", "2025-04-05", None, "12/1/25"])
    assert count_ambiguous_daily_dates(series) == 2


def test_ambiguous_daily_dates_warn_at_upload(isolated) -> None:
    response = _post(isolated, "daily", "Wally_amb.csv", _daily_bytes(date_value="4/5/2025"))
    assert response.status_code == 200, response.text
    warnings = response.json()["warnings"]
    assert any("ambiguous" in warning for warning in warnings), warnings


# --- campaign flights: a real upload kind wired to the pacing loader -------------
def test_campaign_flights_kind_is_reported_and_uploads(isolated) -> None:
    body = isolated.get("/api/uploads/status").json()
    entry = next(item for item in body["inputs"] if item["kind"] == "campaign_flights")
    assert entry["cadence"] == "config"
    assert entry["label_he"], "the dashboard renders the Hebrew label from the API"
    assert entry["in_use"] is True, "pacing reads data/campaign_flights.csv directly"

    header_only = _post(isolated, "campaign_flights", "campaign_flights.csv", _flights_bytes([]))
    assert header_only.status_code == 200, header_only.text
    assert header_only.json()["validation"]["rows_loaded"] == 0

    good = _post(
        isolated,
        "campaign_flights",
        "campaign_flights.csv",
        _flights_bytes(["CAMP1,2026-07-01,2026-07-31,120000,,10000,עכשיו 14,,,,summer"]),
    )
    assert good.status_code == 200, good.text
    payload = good.json()
    assert payload["in_use"] is True
    assert payload["validation"]["rows_loaded"] == 1
    saved = uploads.DATA_DIR / "campaign_flights.csv"
    assert saved.exists()


def test_campaign_flights_with_no_loadable_row_is_refused(isolated) -> None:
    response = _post(
        isolated,
        "campaign_flights",
        "campaign_flights.csv",
        _flights_bytes(["CAMP1,,,120000,,0,,,,,missing flight dates"]),
    )
    assert response.status_code == 400, response.text
    assert any("zero campaigns" in err for err in response.json()["errors"])


def test_campaign_flights_partially_skipped_rows_warn(isolated) -> None:
    response = _post(
        isolated,
        "campaign_flights",
        "campaign_flights.csv",
        _flights_bytes(
            [
                "CAMP1,2026-07-01,2026-07-31,120000,,0,,,,,ok",
                "CAMP2,,,50000,,0,,,,,broken",
            ]
        ),
    )
    assert response.status_code == 200, response.text
    assert any("skipped" in warning for warning in response.json()["warnings"])


def test_campaign_flights_header_matches_the_shipped_seed() -> None:
    seed_header = (
        (ROOT / "data" / "campaign_flights.csv").read_text(encoding="utf-8-sig").splitlines()[0]
    )
    assert seed_header.split(",") == uploads.REQUIRED_COLUMNS["campaign_flights"]


# --- atomic writes, timestamps, engine_reads honesty ------------------------------
def test_upload_write_is_atomic_replace_with_no_tmp_leftover(isolated) -> None:
    data = _daily_bytes()
    response = _post(isolated, "daily", "Wally_atomic.csv", data)
    assert response.status_code == 200, response.text
    destination = uploads.DAILY_DIR / "Wally_atomic.csv"
    assert destination.read_bytes() == data, "bytes must land verbatim"
    assert list(uploads.DAILY_DIR.glob("*.tmp")) == [], "the temp file must be renamed away"


def test_last_modified_carries_a_utc_offset(isolated) -> None:
    _post(isolated, "daily", "Wally_tz.csv", _daily_bytes())
    body = isolated.get("/api/uploads/status").json()
    daily = next(item for item in body["inputs"] if item["kind"] == "daily")
    stamp = datetime.fromisoformat(daily["last_modified"])
    assert stamp.tzinfo is not None, "last_modified must be ISO with an explicit offset"


def test_status_names_the_file_the_engine_actually_reads(isolated) -> None:
    body = isolated.get("/api/uploads/status").json()
    entries = {item["kind"]: item for item in body["inputs"]}
    if (ROOT / "data" / "reference" / "Programmes.xlsx").exists():
        assert entries["programmes"]["engine_reads"] == "data/reference/Programmes.xlsx"
    assert entries["rate_card"]["engine_reads"] == "config/optimization_weights.yaml"
    assert entries["daily"]["engine_reads"] is None, "no daily file exists in the isolated dir"
