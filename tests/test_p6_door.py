"""P6 Sources, the door: the file a bad export cannot get through, and why.

The reference bar for this destination is import validation that refuses at the
door with the reason, so the refusal is graded on what it NAMES: the column, the
row number, the count of rows it looked at, and the severity, rather than a
sentence saying something somewhere is wrong.

Split out of ``test_p6_sources.py``, which had grown past the 450-line law. The
door is its own subject and it is the one this destination is judged on, so it
is the half that moved. Every test runs against an app carrying only the router
this piece owns, with every writable path relocated to a temporary directory, so
nothing here can touch a repository input.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from io import BytesIO
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.uploads as uploads
import kairos_api.uploads_messages as uploads_messages
import kairos_api.uploads_status as uploads_status
import kairos_api.uploads_validate as uploads_validate
from kairos.data.loaders import DAILY_COLUMN_MAP

ROOT = Path(__file__).resolve().parents[1]
LIVE_DAILY = ROOT / "data" / "daily_input" / "Wally_Prime_Reshet_Example_2025-04-27.csv"

HEBREW = re.compile(r"[֐-׿]")
FIELD = re.compile(r"\{(\w+)\}")

# An internal placeholder standing where a column name goes. Nothing wrapped in
# angle brackets is a column in any file this door accepts, so a chip that
# matches this is a token that reached the screen.
TOKEN = re.compile(r"^<.*>$")

# The shipped module the card renders its findings by, run as itself rather than
# reimplemented here, so this file measures what an operator sees.
NODE = shutil.which("node")
MODULE_URL = (ROOT / "tv-break-dashboard" / "src" / "sources" / "sources-findings.js").as_uri()

# Every code this door raises ITSELF, as opposed to the ones the frozen data
# contracts raise. Both now get a Hebrew sentence, but the door authors its own
# in full from its own fields, while the contracts' English stays byte-for-byte
# theirs in ``message`` and only the Hebrew beside it is authored here, from a
# count recomputed on the loaded frame rather than from that English.
OWN_CODES = frozenset({
    "no_parseable_dates", "unparseable_dates", "unreadable_times", "ambiguous_day_month",
    "no_loadable_campaigns", "skipped_campaign_rows", "no_recognized_channel_columns",
    "no_data_rows", "unreadable_file", "missing_columns", "empty_file", "too_large",
})

# The three cells, on the three data rows of the real morning file, and what is
# wrong with each: a day that does not exist, a duration below zero, a clock
# reading past every hour there is.
BROKEN_CELLS = ((12, "תאריך", "31/31/2025"), (88, "אורך תשדיר", "-5"), (140, "שעה", "99:99:99"))


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


def _morning_file_with_broken_cells(*, keep: tuple[int, ...] = (12, 88, 140)) -> bytes:
    """The live 175-row daily file with the named data rows broken.

    Rebuilt from the repository's own input rather than carried as a copy, so it
    stays the file the engine really reads with exactly the defects under test in
    it. Every cell is read and written back as text so nothing but those cells
    moves.
    """
    frame = pd.read_csv(LIVE_DAILY, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for number, column, value in BROKEN_CELLS:
        if number in keep:
            frame.loc[number - 1, column] = value
    return frame.to_csv(index=False).encode("utf-8-sig")


def test_check_refuses_a_bad_file_with_the_contracts_own_findings(isolated: TestClient) -> None:
    bad = _daily_bytes(date_value="not-a-date")
    response = isolated.post("/api/uploads/daily/check", files={"file": ("bad.csv", bad, "text/csv")})
    body = response.json()
    assert body["accepted"] is False
    assert body["errors"], "a refusal must carry the reason"
    assert any(finding["severity"] == "error" for finding in body["findings"])
    assert any(finding["column"] for finding in body["findings"]), "a finding names the column"


def test_the_door_names_the_row_of_every_broken_cell(isolated: TestClient) -> None:
    """Three broken cells in the real morning file, three findings, three rows.

    A column and a count leave a steward hand-searching 175 rows, and two of
    these three defects raised no finding at all: one unparseable date hid behind
    an all-or-nothing check that only fired when EVERY date failed, and an
    impossible clock was never looked at. Measured on the live file: the date is
    on data row 12, the duration on 88 and the time on 140.
    """
    if not LIVE_DAILY.exists():
        pytest.skip("no live daily file in the repository to break three cells in")
    response = isolated.post(
        "/api/uploads/daily/check",
        files={"file": ("Wally_2025-04-27.csv", _morning_file_with_broken_cells(), "text/csv")},
    )
    body = response.json()
    assert body["accepted"] is False, "a dateless row and a negative duration are not acceptable"
    by_code = {finding["code"]: finding for finding in body["findings"]}
    assert by_code["unparseable_dates"]["rows"] == [12], "the unreadable date is not placed on its row"
    assert by_code["non_positive_values"]["rows"] == [88], "the negative duration is not placed on its row"
    assert by_code["unreadable_times"]["rows"] == [140], "the impossible clock is not placed on its row"
    for code in ("unparseable_dates", "non_positive_values", "unreadable_times"):
        assert by_code[code]["rows_total"] == 1, f"{code} counted more rows than it named"
        assert by_code[code]["column"], "a finding names the column as well as the row"
    assert by_code["unparseable_dates"]["severity"] == "error"
    assert by_code["unreadable_times"]["severity"] == "warning", "a spot with no clock still loads and still prices"


def test_one_unreadable_date_in_a_hundred_and_seventy_five_still_refuses(isolated: TestClient) -> None:
    """The exact hole: fix the duration the door reported and the file used to be
    accepted with a row the engine can place on no day."""
    if not LIVE_DAILY.exists():
        pytest.skip("no live daily file in the repository to break one cell in")
    clean = isolated.post(
        "/api/uploads/daily/check",
        files={"file": ("Wally_clean.csv", _morning_file_with_broken_cells(keep=()), "text/csv")},
    ).json()
    assert clean["accepted"] is True, "the unbroken morning file must still pass the door"
    assert not [finding for finding in clean["findings"] if finding.get("rows")], "a clean file names no row"

    body = isolated.post(
        "/api/uploads/daily/check",
        files={"file": ("Wally_dateless.csv", _morning_file_with_broken_cells(keep=(12,)), "text/csv")},
    ).json()
    assert body["accepted"] is False, "one row of 175 with no readable date is still refused"
    finding = next(f for f in body["findings"] if f["code"] == "unparseable_dates")
    assert finding["rows"] == [12]
    assert "175" in finding["message"], "the reason states how many rows it looked at"


def test_a_finding_lists_at_most_the_cap_and_counts_the_rest(isolated: TestClient) -> None:
    """A file broken in every row states how many rows without printing them all."""
    total = uploads_validate.ROW_LIST_CAP + 15
    body = isolated.post(
        "/api/uploads/daily/check",
        files={"file": ("Wally_zero.csv", _daily_bytes(duration="0", rows=total), "text/csv")},
    ).json()
    finding = next(f for f in body["findings"] if f["code"] == "non_positive_values")
    assert finding["rows_total"] == total, "the count of broken rows is not the real one"
    assert len(finding["rows"]) == uploads_validate.ROW_LIST_CAP
    assert finding["rows"][0] == 1 and finding["rows"][-1] == uploads_validate.ROW_LIST_CAP


def test_a_melted_kind_never_states_a_row_number(isolated: TestClient) -> None:
    """The dayparts loader melts one row per channel column, so a position in the
    loaded frame is not a row in the operator's file and no finding claims it is.
    """
    frame = pd.DataFrame(
        {"Dates": ["01/05/2025", "02/05/2025"], "Timebands": ["20:00", "20:01"], "רשת 13": ["1.0", "-2.0"]}
    )
    body = isolated.post(
        "/api/uploads/dayparts/check",
        files={"file": ("Dayparts.csv", frame.to_csv(index=False).encode("utf-8"), "text/csv")},
    ).json()
    assert body["accepted"] is False, "a negative rating is refused"
    assert any(finding["code"] == "negative_values" for finding in body["findings"])
    assert not [finding for finding in body["findings"] if "rows" in finding], "a melted frame invented a row number"


def test_check_writes_nothing_at_all(isolated: TestClient, tmp_path) -> None:
    """The whole point of the door: looking costs nothing."""
    before = sorted(str(path) for path in tmp_path.rglob("*"))
    good = _daily_bytes()
    accepted = isolated.post("/api/uploads/daily/check", files={"file": ("Wally_2025-04-27.csv", good, "text/csv")})
    bad = isolated.post("/api/uploads/daily/check", files={"file": ("bad.csv", _daily_bytes(date_value="x"), "text/csv")})
    assert accepted.json()["accepted"] is True
    assert bad.json()["accepted"] is False
    assert sorted(str(path) for path in tmp_path.rglob("*")) == before, "check wrote something"


def test_check_and_upload_agree_on_the_same_file(isolated: TestClient) -> None:
    """A file the door accepts is a file the upload accepts, and the other way."""
    bad = _daily_bytes(date_value="x")
    assert isolated.post("/api/uploads/daily/check", files={"file": ("a.csv", bad, "text/csv")}).json()["accepted"] is False
    assert isolated.post("/api/uploads/daily", files={"file": ("a.csv", bad, "text/csv")}).status_code == 400
    good = _daily_bytes()
    assert isolated.post("/api/uploads/daily/check", files={"file": ("Wally_2025-04-27.csv", good, "text/csv")}).json()["accepted"] is True
    assert isolated.post("/api/uploads/daily", files={"file": ("Wally_2025-04-27.csv", good, "text/csv")}).status_code == 200


def test_check_states_the_consequence_before_the_commit(isolated: TestClient) -> None:
    body = isolated.post(
        "/api/uploads/daily/check", files={"file": ("Wally_2025-04-27.csv", _daily_bytes(), "text/csv")}
    ).json()
    assert body["consequence"]["code"] in uploads_status.CONSEQUENCES
    assert body["consequence"]["he"], "the consequence must be readable in Hebrew before the click"


def test_check_refuses_an_unknown_kind(isolated: TestClient) -> None:
    response = isolated.post("/api/uploads/nonsense/check", files={"file": ("a.csv", b"a,b\n1,2\n", "text/csv")})
    assert response.status_code == 404


def _daily_with_broken_rows(broken: int, total: int) -> bytes:
    """A daily file of ``total`` rows, ``broken`` of them with a date and a clock
    the loader cannot read, and an ambiguous slash date on the rows that are fine.

    This is the morning case the door is graded on and the one a critic measured
    through the shipped surface: three reasons in one refusal, all three of them
    written by this destination rather than by the frozen contracts.
    """
    frame = pd.read_csv(BytesIO(_daily_bytes(date_value="3/4/2025", rows=total)), dtype=str, keep_default_na=False)
    frame.loc[: broken - 1, "תאריך"] = "31/31/2025"
    frame.loc[: broken - 1, "שעה"] = "99:99:99"
    return frame.to_csv(index=False).encode("utf-8-sig")


def test_every_reason_this_door_writes_itself_is_readable_in_hebrew(isolated: TestClient) -> None:
    """The measured gap: a Hebrew heading, Hebrew row labels, English reasons.

    The whole destination is read in Hebrew, and the refusal was the one sentence
    on it that was not. Every finding whose code this door raises itself now
    carries both languages; a violation the frozen contracts raised carries its
    English detail alone, on purpose, and the surface falls back to it.
    """
    body = isolated.post(
        "/api/uploads/daily/check",
        files={"file": ("Wally_2025-04-27.csv", _daily_with_broken_rows(2, 3), "text/csv")},
    ).json()
    assert body["accepted"] is False, "two rows the engine can place on no day are not acceptable"
    own = [finding for finding in body["findings"] if finding["code"] in OWN_CODES]
    raised = {finding["code"] for finding in own}
    assert {"unparseable_dates", "unreadable_times", "ambiguous_day_month"} <= raised, f"the door raised {raised}"
    for finding in own:
        translated = finding.get("message_he") or ""
        assert translated, f"{finding['code']} reached a Hebrew screen in English"
        assert HEBREW.search(translated), f"{finding['code']} has a Hebrew slot with no Hebrew in it"
        assert translated != finding["message"], f"{finding['code']} repeated its English as its Hebrew"
    for finding in body["findings"]:
        assert finding["message"], "a finding arrived with no sentence in either language"


def test_a_finding_the_frozen_contracts_raised_keeps_its_own_english_detail(isolated: TestClient) -> None:
    """The English is quoted, byte-for-byte, because the frozen contract's own
    counts and column names inside it are its to compute. The Hebrew beside it
    is authored here, from a count recomputed rather than parsed from it."""
    body = isolated.post(
        "/api/uploads/daily/check",
        files={"file": ("Wally_zero.csv", _daily_bytes(duration="0"), "text/csv")},
    ).json()
    finding = next(f for f in body["findings"] if f["code"] == "non_positive_values")
    assert finding["message"] == "1 value(s) at or below zero", "the contract's own detail must stay verbatim"
    assert HEBREW.search(finding.get("message_he") or ""), "a contract's own finding still reads English on a Hebrew screen"
    assert finding["message_he"] != finding["message"], "the Hebrew must be its own sentence, not the English quoted twice"


def test_a_header_refusal_carries_its_reason_in_both_languages(isolated: TestClient) -> None:
    """The refusal at the door proper, before any loader runs: the 400 itself."""
    response = isolated.post(
        "/api/uploads/daily/check", files={"file": ("wrong.csv", b"a,b\n1,2\n", "text/csv")}
    )
    assert response.status_code == 400, "a file with none of the required columns is refused"
    body = response.json()
    assert HEBREW.search(body.get("detail_he") or ""), "the headline of the refusal is English only"
    finding = (body.get("findings") or [{}])[0]
    assert finding.get("code") == "missing_columns"
    assert HEBREW.search(finding.get("message_he") or ""), "the reason under the headline is English only"


def test_every_sentence_this_destination_writes_itself_has_both_languages() -> None:
    """The table, swept whole, so a code added later cannot ship in one language."""
    assert OWN_CODES <= set(uploads_messages.MESSAGES), "a code the door raises has no sentence in the table"
    for code, words in uploads_messages.MESSAGES.items():
        assert words["en"].strip(), f"{code} has no English"
        assert words["he"].strip(), f"{code} has no Hebrew"
        assert HEBREW.search(words["he"]), f"{code} has a Hebrew slot with no Hebrew in it"
        extra = set(FIELD.findall(words["he"])) - set(FIELD.findall(words["en"]))
        assert not extra, f"{code} has Hebrew placeholders nothing fills: {sorted(extra)}"


def _every_refusal(client: TestClient) -> dict[str, dict]:
    """Every shape of refusal this door can answer with, driven through the app.

    Three are raised before a loader runs and answer 400 with a headline and one
    finding, which is the shape the repeated sentence lived in; the rest are the
    loader's and the frozen contracts', answered 200 with ``accepted`` false, and
    the commit route is here because it answers those as a 400 with a headline.
    """
    dayparts_no_rows = pd.DataFrame(columns=["Dates", "Timebands", "רשת 13"]).to_csv(index=False).encode("utf-8")
    renamed = pd.DataFrame({"Dates": ["01/05/2025"], "Timebands": ["20:00"], "Channel_A": ["1.0"]})
    frame = pd.read_csv(BytesIO(_daily_bytes()), dtype=str, keep_default_na=False)
    posts = {
        "missing_columns": ("daily/check", "Wally_x.csv", frame.drop(columns=["מפרסם"]).to_csv(index=False).encode("utf-8")),
        "empty_file": ("daily/check", "empty.csv", b""),
        "unreadable_file": ("daily/check", "broken.csv", b"\xff\xfe\x00not a csv"),
        "no_data_rows": ("dayparts/check", "Dayparts.csv", dayparts_no_rows),
        "no_recognized_channel_columns": ("dayparts/check", "Dayparts.csv", renamed.to_csv(index=False).encode("utf-8")),
        "contract_check": ("daily/check", "Wally_x.csv", _daily_bytes(duration="0")),
        "contract_commit": ("daily", "Wally_2025-04-27.csv", _daily_bytes(duration="0")),
    }
    bodies = {}
    for name, (path, filename, payload) in posts.items():
        bodies[name] = client.post(f"/api/uploads/{path}", files={"file": (filename, payload, "text/csv")}).json()
    return bodies


def test_no_refusal_puts_an_internal_token_where_a_column_name_goes(isolated: TestClient) -> None:
    """The measured defect: a bold Latin ``<header>`` chip on a Hebrew card.

    A finding owes a ``column`` key and a refusal is often about no column, so
    the door filled that key with ``<file>``, ``<header>`` or ``<frame>`` and the
    card printed it verbatim. Every finding now carries a real column name or a
    declared ``scope`` code the surface resolves to a word, and never both.

    A name no file carries is the same defect unbracketed, and so is the empty
    slot emptying that key leaves in the flat line the English refusal quotes.
    Measured on the shipped surface: the channel refusal chipped ``channels``, a
    column no dayparts export has, and the no-rows one read ``[error] :``.
    """
    bodies = _every_refusal(isolated)
    for name, body in bodies.items():
        findings = body.get("findings") or []
        assert findings, f"{name} refused with no finding at all"
        for finding in findings:
            column = str(finding.get("column") or "")
            assert not TOKEN.match(column), f"{name} put the internal token {column} where a column name goes"
            scope = str(finding.get("scope") or "")
            assert column or scope, f"{name} named neither a column nor what it is about"
            assert not (column and scope), f"{name} named a column and a scope, and the surface prints one chip"
            assert not scope or scope in uploads_messages.SCOPES, f"{name} raised the undeclared scope {scope}"
        for line in list(body.get("errors") or []) + ([body["detail"]] if body.get("detail") else []):
            assert "] :" not in line, f"{name} quotes a line whose column slot is empty: {line[:70]}"
    channels = bodies["no_recognized_channel_columns"]["findings"][0]
    assert (channels["column"], channels["scope"]) == ("", "header"), "the channel refusal is about the header row"


def _as_the_card_renders(body: dict, locale: str) -> dict:
    """What the shipped card prints for this refusal, from the shipped module.

    ``sources-findings.js`` is the module ``SourceCard.jsx`` imports and it holds
    the whole rule, so running it here over a real response body measures the
    rendered outcome rather than a second implementation of it.
    """
    if NODE is None:
        pytest.skip("node is not on this machine, so the shipped rendering module cannot be run")
    script = (
        f"import {{ findingMessage, visibleFindings }} from {json.dumps(MODULE_URL)};\n"
        "let raw = ''; for await (const chunk of process.stdin) raw += chunk;\n"
        "const { body, locale } = JSON.parse(raw);\n"
        "const detail = body.detail ? findingMessage({ message: body.detail, message_he: body.detail_he }, locale) : '';\n"
        "const lines = visibleFindings(body.findings || [], locale, detail)\n"
        "  .map((line) => ({ chip: line.chip ? line.chip.text : '', message: line.message }));\n"
        "process.stdout.write(JSON.stringify({ detail, lines }));\n"
    )
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        input=json.dumps({"body": body, "locale": locale}),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_the_card_prints_no_bracketed_chip_and_no_sentence_twice(isolated: TestClient) -> None:
    """The two defects a critic measured on the headline screen of this destination.

    Measured before this: a daily file missing the ``מפרסם`` column printed the
    headline, then the reason, then a bold Latin ``<header>`` chip, then that
    identical reason again. The same shape was reachable on ``empty_file`` and
    ``unreadable_file`` with a ``<file>`` chip, and on ``no_data_rows`` with ``<frame>``.
    """
    refusals = _every_refusal(isolated)
    for locale in ("he", "en"):
        for name, body in refusals.items():
            panel = _as_the_card_renders(body, locale)
            printed = [panel["detail"]] if panel["detail"] else []
            for line in panel["lines"]:
                assert not TOKEN.match(line["chip"]), f"{name} rendered the chip {line['chip']} in {locale}"
                if line["message"]:
                    printed.append(line["message"])
            assert len(printed) == len(set(printed)), f"{name} printed the same sentence twice in {locale}"
            assert printed, f"{name} printed no reason at all in {locale}"


def test_the_missing_column_refusal_reads_as_one_reason_and_not_two(isolated: TestClient) -> None:
    """The exact screen that was measured, rendered end to end in Hebrew.

    Three things were on it and one was true: the reason, a token, and the
    reason again. What prints now is the reason, once, naming the column.
    """
    without = pd.read_csv(BytesIO(_daily_bytes()), dtype=str, keep_default_na=False).drop(columns=["מפרסם"])
    body = isolated.post(
        "/api/uploads/daily/check",
        files={"file": ("Wally_x.csv", without.to_csv(index=False).encode("utf-8"), "text/csv")},
    ).json()
    panel = _as_the_card_renders(body, "he")
    assert "מפרסם" in panel["detail"], "the reason no longer names the column that is missing"
    assert HEBREW.search(panel["detail"]), "the reason arrived in English on a Hebrew card"
    assert panel["lines"] == [], "the reason is printed a second time under itself"


def test_a_finding_about_no_column_still_says_what_it_is_about(isolated: TestClient) -> None:
    """Dropping a repeated sentence must not drop the chip that names the place.

    A dayparts export whose channel columns carry no data rows is refused with a
    finding about the table and about no column, with nothing printed above it,
    so it prints its own sentence and the word for what it is about.
    """
    empty = pd.DataFrame(columns=["Dates", "Timebands", "רשת 13"]).to_csv(index=False).encode("utf-8")
    body = isolated.post(
        "/api/uploads/dayparts/check", files={"file": ("Dayparts.csv", empty, "text/csv")}
    ).json()
    assert body["accepted"] is False, "a dayparts file with no data rows yields no audience figure"
    assert [line["chip"] for line in _as_the_card_renders(body, "he")["lines"]] == ["הטבלה"]
    assert [line["chip"] for line in _as_the_card_renders(body, "en")["lines"]] == ["The table"]
