"""P6 Sources, the door: the accepted file a green tick would read as good news.

Two shapes of it, and the same lesson twice. The first is the file that carries
its header and no rows at all. The second is the file that carries rows the
engine cannot fully read: measured on the shipped bundle, a daily log whose 20
of 20 rows carry a clock the loader cannot parse was answered ``accepted: true``,
``will_be_read: true`` and ``replaces_live_input`` with one warn finding, and the
card printed the teal tone, the heading "the file passed every check" and an
enabled commit button over it. Committing it makes the live daily input a file
from which no spot has a daypart.

The reference bar for this destination is import validation that refuses at the
door with the reason, and the first row of that catalogue everywhere is the
empty file. Measured on this door before these tests: a CSV with the real header
and zero data rows was ACCEPTED for six of the seven kinds, with ``findings: []``
and ``rows: 0``, and for three of those six the same body answered
``will_be_read: true`` with the consequence ``replaces_live_input``. The card
therefore printed a green tick, ``0 שורות`` and "this is the live input" over a
commit button, and committing it really did empty the live daily input: 175 rows
in use before, zero rows and ``state: empty`` after. The one screen that named
the outcome named it after the click.

Split from ``test_p6_door.py`` by the 450-line law and by subject, which is the
same law and the same reason that made ``test_p6_door.py`` and
``test_p6_prospect.py`` their own files. That file is at 449 lines of a 450-line
cap, so the sweep of all seven kinds could not land in it without pushing a
passing file over a law this row asserts on itself.

Every test runs against an app carrying only the router this piece owns, with
every writable path relocated to a temporary directory, so nothing here can
touch a repository input.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.uploads as uploads
import kairos_api.uploads_empty as uploads_empty
import kairos_api.uploads_inputs as uploads_inputs
import kairos_api.uploads_status as uploads_status
from kairos.data.loaders import DAILY_COLUMN_MAP

HEBREW = re.compile(r"[֐-׿]")

ROOT = Path(__file__).resolve().parents[1]

# The two shipped modules the card decides its verdict by, run as themselves
# rather than reimplemented here: the rule, and the words it resolves to.
NODE = shutil.which("node")
SOURCES = ROOT / "tv-break-dashboard" / "src" / "sources"
FINDINGS_URL = (SOURCES / "sources-findings.js").as_uri()
COPY_URL = (SOURCES / "sources-copy.js").as_uri()

# The name a file of each kind arrives under, as an operator's export really is
# named, so the daily one carries a date the resolver can rank.
FILENAMES = {
    "programmes": "Programmes.csv",
    "spots": "Spots.csv",
    "dayparts": "Dayparts.csv",
    "advertiser_rules": "advertiser_rules.csv",
    "rate_card": "rate_card_premiums.csv",
    "campaign_flights": "campaign_flights.csv",
    "daily": "Wally_2026-08-09.csv",
}


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


def header_only(kind: str) -> bytes:
    """A real export of this kind with every required column and no data row.

    The dayparts header carries the operator's own channel column as well,
    because without one that file is refused for a different reason and the
    thing under test here would never be reached.
    """
    columns = list(uploads_inputs.REQUIRED_COLUMNS[kind])
    if kind == "dayparts":
        columns = columns + ["רשת 13"]
    return pd.DataFrame(columns=columns).to_csv(index=False).encode("utf-8")


def check(client: TestClient, kind: str) -> dict:
    response = client.post(
        f"/api/uploads/{kind}/check", files={"file": (FILENAMES[kind], header_only(kind), "text/csv")}
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_every_kind_this_door_accepts_declares_what_no_rows_means() -> None:
    """The rule is declared for all seven kinds, so an eighth cannot inherit the hole.

    The gap was not that the rule was wrong. It was that the rule existed for
    exactly one kind, the one somebody had happened to write it for, and the
    other six were never asked the question at all.
    """
    declared = {meta["kind"] for meta in uploads_inputs.INPUTS}
    assert set(uploads_empty.SEVERITY) == declared, "a kind the door accepts has no answer for an empty file"
    assert set(uploads_empty.SEVERITY.values()) <= {"error", "warning"}
    for kind in declared - {"dayparts"}:
        assert kind in uploads_empty.ROW_NOUN, f"{kind} has no word for what one of its rows is"


def test_no_kind_takes_a_file_with_a_header_and_no_rows_without_saying_so(isolated: TestClient) -> None:
    """All seven kinds, one file each: the header, and nothing under it.

    Every kind now answers with the ``no_data_rows`` finding at the severity its
    own table declares, in both languages, about the table and about no column.
    Four of the seven were silent before this and answered ``accepted: true``
    with an empty findings list.
    """
    for kind in FILENAMES:
        body = check(isolated, kind)
        assert body["rows"] == 0
        finding = next((f for f in body["findings"] if f["code"] == "no_data_rows"), None)
        assert finding is not None, f"{kind} took a file with no rows and said nothing about it"
        assert finding["severity"] == uploads_empty.SEVERITY[kind], f"{kind} answered the wrong severity"
        assert finding["column"] == "" and finding["scope"] == "frame", f"{kind} named a column it cannot name"
        assert HEBREW.search(finding.get("message_he") or ""), f"{kind} reached a Hebrew screen in English"
        assert finding["message"] and finding["message_he"] != finding["message"]
        assert body["accepted"] is (uploads_empty.SEVERITY[kind] == "warning"), f"{kind} took the wrong verdict"


def test_the_four_kinds_an_empty_file_can_never_be_right_for_are_refused(isolated: TestClient) -> None:
    """A lineup, a spot history, a rules table and a rate card with no rows.

    There is no world in which an operator meant to publish one of these empty,
    so the file does not reach a commit button: the door refuses it with the
    reason, which is the mechanic this destination is graded on.
    """
    for kind in ("programmes", "spots", "advertiser_rules", "rate_card"):
        body = check(isolated, kind)
        assert body["accepted"] is False, f"{kind} accepted a file that can carry no meaning"
        assert body["errors"], f"{kind} refused with no reason"
        assert any("no_data_rows" in line for line in body["errors"]), f"{kind} refused for another reason only"


def test_the_two_kinds_an_empty_file_is_real_for_are_accepted_and_named(isolated: TestClient) -> None:
    """A broadcast day with nothing booked, and a flight list nobody has filled in.

    Both really occur, so both are accepted, and the consequence says what
    committing one does instead of the sentence that reads as good news over it.
    """
    for kind in ("daily", "campaign_flights"):
        body = check(isolated, kind)
        assert body["accepted"] is True, f"{kind} refused a state that really occurs"
        assert body["will_be_read"] is True, f"{kind} is not the live input in this fixture"
        assert body["consequence"]["code"] == "replaces_live_input_with_no_rows", f"{kind} promised the wrong outcome"
        assert body["consequence"]["code"] in uploads_status.CONSEQUENCES
        assert HEBREW.search(body["consequence"]["he"]), f"{kind} names the outcome in English only"
        assert body["warnings"], f"{kind} carried the finding and no warning line"


def test_no_line_this_refusal_quotes_leaves_the_place_slot_empty(isolated: TestClient) -> None:
    """The flat lines a reader parses, on a finding that is about no column.

    ``errors`` was rebuilt from the scope in an earlier round and ``warnings``
    was not, which nothing had reached because no warning about the whole table
    existed. One now does, and the two lists are written by the same function.
    """
    seen = 0
    for kind in FILENAMES:
        body = check(isolated, kind)
        for line in list(body["errors"]) + list(body["warnings"]):
            assert "] :" not in line, f"{kind} quotes a line whose place slot is empty: {line[:60]}"
        for line in body["warnings"]:
            if "no_data_rows" in line:
                assert "] frame:" in line, f"{kind} does not say what the warning is about"
                seen += 1
    # A sweep that matched nothing would pass without looking at anything, which
    # is exactly how the empty slot survived on this list for ten rounds.
    assert seen == 2, f"the sweep read {seen} of the two warning lines it exists for"


# --- the second shape: rows the engine reads and cannot fully read -------------


def daily_bytes(*, clock: str = "18:01:00", date_value: str = "4/27/2025", rows: int = 20) -> bytes:
    """A daily export with real rows, and whatever clock or date is under test.

    Every other cell is a legitimate morning value, so the only finding a file of
    this shape can raise is the one the caller asked for.
    """
    row = {name: "" for name in DAILY_COLUMN_MAP}
    row["תאריך"] = date_value
    row["שעה"] = clock
    row["שעת התחלת ברייק"] = "18:00:00"
    row["מפרסם"] = "Acme"
    row["קמפיין"] = "Acme Summer"
    row["תוכנית מוזמנת"] = "Evening Show"
    row["שעת התחלת תוכנית"] = "18:00"
    row["אורך תשדיר"] = "30"
    row["מיקום בברייק"] = "1"
    row["רייטינג ברייקים מתוכנן"] = "5.5"
    frame = pd.DataFrame([row] * rows, columns=list(DAILY_COLUMN_MAP.keys()))
    return frame.to_csv(index=False).encode("utf-8")


def check_daily(client: TestClient, **kwargs) -> dict:
    """The door's answer about one daily candidate that has not been written."""
    response = client.post(
        "/api/uploads/daily/check",
        files={"file": ("Wally_2026-08-09.csv", daily_bytes(**kwargs), "text/csv")},
    )
    assert response.status_code == 200, response.text
    return response.json()


def as_the_card_renders(body: dict, locale: str) -> dict:
    """The verdict panel the shipped card prints over this candidate.

    ``sources-findings.js`` holds the rule and ``SourceCard.jsx`` renders exactly
    what it returns, and ``sources-copy.js`` holds the words, so both modules are
    run here over a real response body: what this returns is the heading an
    operator reads and the tone the panel carries, not a second implementation.
    """
    if NODE is None:
        pytest.skip("node is not on this machine, so the shipped rendering modules cannot be run")
    script = (
        f"import {{ acceptedVerdict }} from {json.dumps(FINDINGS_URL)};\n"
        f"import {{ text }} from {json.dumps(COPY_URL)};\n"
        "let raw = ''; for await (const chunk of process.stdin) raw += chunk;\n"
        "const { body, locale } = JSON.parse(raw);\n"
        "const verdict = acceptedVerdict(body);\n"
        "process.stdout.write(JSON.stringify({ ...verdict, words: text(verdict.heading, locale) }));\n"
    )
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        input=json.dumps({"body": body, "locale": locale}),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_the_door_names_a_warning_on_the_file_the_engine_will_read(isolated: TestClient) -> None:
    """The measured screen: 20 of 20 rows carrying a clock the loader cannot read.

    The file is not refused, and that is correct: a spot with no clock still
    loads and still prices, so refusing it would lose a real morning file over a
    field the engine can do without on some rows. What was wrong is what the
    door then said about it, which was that this is the live input and nothing
    else, in the sentence that reads as good news. The consequence now names how
    many warnings the candidate carries and the field they are about, before the
    click, in the language the card is read in.
    """
    body = check_daily(isolated, clock="99:99:99")
    assert body["accepted"] is True, "a spot with no clock still loads, so the file is not refused"
    assert body["will_be_read"] is True, "the candidate is the live input in this fixture"
    finding = next(f for f in body["findings"] if f["code"] == "unreadable_times")
    assert (finding["severity"], finding["rows_total"]) == ("warning", 20), "the file under test is not the measured one"
    assert body["consequence"]["code"] == "replaces_live_input_with_warnings", "the file passed for a clean replacement"
    assert body["consequence"]["code"] in uploads_status.CONSEQUENCES
    for locale in ("en", "he"):
        sentence = body["consequence"][locale]
        # The field is named as the file names it. ``spot_time`` is the loader's
        # name for it and no export carries that word, so the sentence said this
        # operator's file has a column it does not have, in both languages.
        assert "שעה" in sentence, f"the field the warning is about is unnamed in {locale}"
        assert "spot_time" not in sentence, f"a name no daily export carries is in the {locale} sentence"
        assert "1" in sentence, f"how many warnings ride on this file is unstated in {locale}"
    assert HEBREW.search(body["consequence"]["he"]), "the outcome is named in English only"


def test_no_candidate_carrying_a_warning_renders_the_plain_pass(isolated: TestClient) -> None:
    """The rendered outcome, through the shipped module, in both languages.

    The control is the first row and it is the point of the sweep: a clean file
    still reads as a clean pass in the ok tone, so this cannot be satisfied by
    turning every verdict amber. Every other row carries a warn-severity finding
    on a file the engine will read, and none of them may print the plain heading
    or the ok tone. Two warnings on one file are asserted together, because the
    consequence names a count and a list and a count of one proves neither.
    """
    clean = check_daily(isolated)
    assert not [f for f in clean["findings"] if f["severity"] == "warning"], "the control file is not clean"
    for locale in ("en", "he"):
        panel = as_the_card_renders(clean, locale)
        assert (panel["heading"], panel["tone"]) == ("accepted", "ok"), "a clean file lost its clean pass"
        assert panel["words"], f"the heading {panel['heading']} has no word in {locale}"

    # The heading each one prints is named, because two warnings are not one
    # piece of news: the clock is a field the engine never got, and the slash
    # date is a value it did get and read one of the two ways it could be read.
    # Reading the same heading over both is how a morning file that is perfect
    # except for the eleven days of a month whose day number is twelve or under
    # came to look like a broken export.
    warned = {
        "an unreadable clock on every row": (check_daily(isolated, clock="99:99:99"), "acceptedWarned"),
        "an ambiguous date on every row": (check_daily(isolated, date_value="3/4/2025"), "acceptedRead"),
        "both at once": (check_daily(isolated, clock="99:99:99", date_value="3/4/2025"), "acceptedWarned"),
    }
    for name, (body, expected) in warned.items():
        assert body["accepted"] is True and body["will_be_read"] is True, f"{name} is not the case under test"
        assert [f for f in body["findings"] if f["severity"] == "warning"], f"{name} raised no warning at all"
        for locale in ("en", "he"):
            panel = as_the_card_renders(body, locale)
            assert panel["heading"] != "accepted", f"{name} rendered the plain pass in {locale}"
            assert panel["tone"] != "ok", f"{name} rendered the ok tone in {locale}"
            assert panel["heading"] == expected, f"{name} read as {panel['heading']} in {locale}"
            assert panel["words"], f"{name} rendered the heading {panel['heading']} with no word in {locale}"
    both = warned["both at once"][0]["consequence"]
    assert both["code"] == "replaces_live_input_with_warnings"
    for locale in ("en", "he"):
        assert "2" in both[locale], f"two warnings were counted as something else in {locale}"
        # Named as the operator's own export names them, and never as the loader
        # renamed them: a file whose header row says otherwise cannot be fixed.
        for column in ("שעה", "תאריך"):
            assert column in both[locale], f"{column} is unnamed in {locale}"
        for internal in ("spot_time", "date"):
            assert internal not in both[locale], f"the loader's own {internal} is in the {locale} sentence"


def test_what_the_door_says_about_a_warned_file_is_what_the_commit_says(isolated: TestClient) -> None:
    """The toast may not contradict the door, on this outcome as on the others.

    The commit derives its consequence from the same function and the same
    findings, so the sentence a person reads before the click and the sentence
    they read after it are one sentence.
    """
    predicted = check_daily(isolated, clock="99:99:99")
    committed = isolated.post(
        "/api/uploads/daily",
        files={"file": ("Wally_2026-08-09.csv", daily_bytes(clock="99:99:99"), "text/csv")},
    )
    assert committed.status_code == 200, committed.text
    body = committed.json()
    assert body["consequence"] == predicted["consequence"], "the toast contradicts the door"
    assert body["consequence"]["code"] == "replaces_live_input_with_warnings"
    entry = next(
        item for item in isolated.get("/api/uploads/status").json()["inputs"] if item["kind"] == "daily"
    )
    # What the door predicted really happened: the file is the live input now,
    # and the warning it carried is what every figure from it now rests on.
    assert (entry["state"], entry["rows"]) == ("in_use", 20), "the file the door described is not the live input"
