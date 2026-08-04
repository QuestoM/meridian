"""P6 Sources, the door: the column a refusal names is the operator's own.

The reference bar for this destination is import validation that refuses at the
door with the reason, and the reason is only actionable if the column it names
is one the person can find in the file they exported. Measured on the shipped
bundle before this file existed: a spots export whose dates will not parse was
refused with a bold Latin chip ``air_dt`` beside the sentence, on a file whose
header row is ``Campaign,Channel,Date,Start time,Duration,TVR``. ``air_dt`` is
in no export of any kind; it is a column the loader computes after the read. The
same defect stood on the amber side, where a file that was accepted was answered
"about ``spot_time, date``" over headers that read ``שעה`` and ``תאריך``.

Every finding is raised on the frame the LOADER built, whose names are renames
(``duration_sec`` for ``אורך תשדיר``), melts (``tvr``, for whichever channel
column a value sat under) or computations (``air_dt``, from ``Date`` and
``Start time``). One resolver, :mod:`kairos_api.uploads_columns`, turns each of
them back into the header row of the file in hand, at the one place a finding
record is made, so the chip, the flat line and the consequence sentence cannot
speak different vocabularies about the same column.

The exception is declared and measured rather than left implicit: a finding that
the file has NOT got a column the engine needs names that column, because the
name to add is the only actionable word such a sentence has.

Split from ``test_p6_door.py``, which is at the 450-line law. Every test runs
against an app carrying only the router this piece owns, with every writable
path relocated to a temporary directory, and the rendering half runs the shipped
module the card imports rather than a second copy of its rule.
"""

from __future__ import annotations

import io
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.uploads as uploads
import kairos_api.uploads_columns as uploads_columns
import kairos_api.uploads_inputs as uploads_inputs
import kairos_api.uploads_messages as uploads_messages
import kairos_api.uploads_validate as uploads_validate
from kairos.data.loaders import DAILY_COLUMN_MAP

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
FINDINGS_URL = (ROOT / "tv-break-dashboard" / "src" / "sources" / "sources-findings.js").as_uri()
COPY_URL = (ROOT / "tv-break-dashboard" / "src" / "sources" / "sources-copy.js").as_uri()

# Every name the loader invents or renames to. Not one is a header of any export
# this door accepts, and each is compound enough that it cannot turn up inside
# an ordinary sentence, so a sweep for them over what a person reads has no
# false positive to explain away.
LOADED_NAMES = (
    "air_dt", "start_dt", "end_dt", "spot_time", "duration_sec", "planned_tvr",
    "position_in_break", "break_start", "program_start", "house_number",
    "spot_type", "pricing_type", "break_type",
)

# The words the surface resolves a scope to, both languages, from
# ``sources-copy.js``. A chip is one of these or a name from the header row.
SCOPE_WORDS = ("The whole file", "The header row", "The table", "הקובץ כולו", "שורת הכותרת", "הטבלה")


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


def _header_only(kind: str) -> bytes:
    """This kind's real header row and no data rows at all."""
    return pd.DataFrame(columns=uploads_inputs.REQUIRED_COLUMNS[kind]).to_csv(index=False).encode("utf-8")


def _daily(duration: str = "-5", clock: str = "99:99:99", date_value: str = "3/4/2025", rows: int = 3) -> bytes:
    """A daily export broken four ways at once, every one of them about a column."""
    row = {name: "" for name in DAILY_COLUMN_MAP}
    row["תאריך"] = date_value
    row["שעה"] = clock
    row["שעת התחלת ברייק"] = "18:00:00"
    row["מפרסם"] = "Acme"
    row["קמפיין"] = "Acme Summer"
    row["תוכנית מוזמנת"] = "Evening Show"
    row["שעת התחלת תוכנית"] = "18:00"
    row["אורך תשדיר"] = duration
    row["מיקום בברייק"] = "1"
    return pd.DataFrame([row] * rows, columns=list(DAILY_COLUMN_MAP)).to_csv(index=False).encode("utf-8")


def _flights() -> bytes:
    """A flight file whose one row the pacing loader will skip."""
    columns = uploads_inputs.REQUIRED_COLUMNS["campaign_flights"]
    row = {name: "" for name in columns}
    row["target_impressions"] = "1000"
    return pd.DataFrame([row], columns=columns).to_csv(index=False).encode("utf-8")


# One candidate per kind, each broken so the door has something to say, and each
# carrying a complete header row, so every finding it raises is about a column
# the file really has. The one candidate whose column is genuinely absent has
# its own test below, because it is the one case with the opposite answer.
CANDIDATES: dict[str, tuple[str, bytes]] = {
    "programmes": (
        "Programmes.csv",
        "Title,Channel,Date,Start time,End time,Duration,TVR\nShow,רשת 13,zzz,20:00:00,21:00:00,3600,1.0\n".encode("utf-8"),
    ),
    "spots": (
        "Spots.csv",
        "Campaign,Channel,Date,Start time,Duration,TVR\nA,רשת 13,zzz,20:00:00,30,1.0\n".encode("utf-8"),
    ),
    "dayparts": (
        "Dayparts.csv",
        "Dates,Timebands,רשת 13\n01/05/2025,20:00 - 20:01,-2.0\n".encode("utf-8"),
    ),
    "daily": ("Wally_2026-08-09.csv", _daily()),
    "campaign_flights": ("campaign_flights.csv", _flights()),
    "advertiser_rules": ("advertiser_rules.csv", _header_only("advertiser_rules")),
    "rate_card": ("rate_card_premiums.csv", _header_only("rate_card")),
}


def _check(client: TestClient, kind: str) -> dict:
    """The door's whole answer about this kind's candidate, writing nothing."""
    filename, payload = CANDIDATES[kind]
    return client.post(
        f"/api/uploads/{kind}/check", files={"file": (filename, payload, "text/csv")}
    ).json()


def _headers(kind: str) -> list[str]:
    """The candidate's header row, read off the same bytes that were posted."""
    _, payload = CANDIDATES[kind]
    return [str(column) for column in pd.read_csv(io.BytesIO(payload), nrows=0).columns]


def _named_columns(finding: dict) -> list[str]:
    """The column names one finding carries, which is two when a datetime is
    built from two headers and none when it is about no column at all."""
    column = str((finding or {}).get("column") or "")
    return [part for part in (piece.strip() for piece in column.split(",")) if part]


def test_no_finding_names_a_column_the_candidate_does_not_carry(isolated: TestClient) -> None:
    """The bar, over every kind: the name is a word from that file's header row.

    A finding about no column at all carries a declared scope instead, which the
    surface resolves to a word, and never both and never neither.
    """
    for kind in CANDIDATES:
        body = _check(isolated, kind)
        findings = body.get("findings") or []
        assert findings, f"the {kind} candidate raised no finding to measure"
        headers = _headers(kind)
        for finding in findings:
            named = _named_columns(finding)
            scope = str(finding.get("scope") or "")
            assert named or scope, f"{kind}/{finding['code']} named neither a column nor a place"
            assert not (named and scope), f"{kind}/{finding['code']} named a column and a place"
            assert not scope or scope in uploads_messages.SCOPES, f"{kind} raised the undeclared scope {scope}"
            for column in named:
                assert column in headers, (
                    f"{kind}/{finding['code']} names {column}, absent from that file's header row {headers}"
                )


def test_no_operator_facing_string_carries_a_name_the_loader_invented(isolated: TestClient) -> None:
    """The same sweep over every string in the answer, not only over the chip.

    The flat ``errors`` line, the headline a refusal quotes, the Hebrew half and
    the consequence sentence over an accepted file are all read by a person or
    by the assistant, and each used to be built somewhere else. All of them are
    built from the finding record now, so one resolution covers every one.
    """
    for kind in CANDIDATES:
        body = _check(isolated, kind)
        blob = json.dumps(
            {key: body.get(key) for key in ("detail", "detail_he", "errors", "warnings", "findings", "consequence", "validation")},
            ensure_ascii=False,
        )
        for invented in LOADED_NAMES:
            assert invented not in blob, f"the {kind} answer carries the loader's own {invented}"


def test_a_column_the_export_has_not_got_is_named_so_it_can_be_added(isolated: TestClient) -> None:
    """The declared exception, measured: the file is missing a column the engine
    needs, and the name of that column is the one actionable word there is.

    Resolving this one against the header row would delete it, and the sentence
    that is left, "required column is absent from the frame", names nothing at
    all. So the rule is not "never print a name the header row has not got"; it
    is "never print a name the operator's file has under another name".
    """
    missing = "Campaign,Channel,Date,Start time,Duration\nA,רשת 13,01/05/2025,20:00:00,30\n".encode("utf-8")
    body = isolated.post(
        "/api/uploads/spots/check", files={"file": ("Spots.csv", missing, "text/csv")}
    ).json()
    finding = next(f for f in body["findings"] if f["code"] == "missing_column")
    assert finding["column"] == "TVR", "the column that has to be added is unnamed"
    assert "TVR" not in [str(c) for c in pd.read_csv(io.BytesIO(missing), nrows=0).columns]
    assert body["errors"][0].startswith("[error] TVR:"), body["errors"][0]


def test_every_name_the_door_raises_itself_resolves_to_a_header(isolated: TestClient) -> None:
    """The static half: no file has to be posted for this one to be checkable.

    Each kind's date column and the daily clock column are the four names this
    door raises findings about itself. Against the header row every file of that
    kind must carry to get past the door at all, each of them resolves to a real
    header and never to the fallback, so the scope is a safety net rather than
    the usual answer.
    """
    named = list(uploads_validate.LOADED_DATE_COLUMN.items())
    named.append(("daily", uploads_validate.DAILY_CLOCK_COLUMN))
    for kind, column in named:
        headers = uploads_inputs.REQUIRED_COLUMNS[kind]
        resolved, scope = uploads_columns.place(column, headers, kind)
        assert resolved and not scope, f"{kind}/{column} fell back to the scope on a file that passed the gate"
        for part in resolved.split(", "):
            assert part in headers, f"{kind}/{column} resolved to {part}, which the gate does not require"


def _chips(body: dict, locale: str) -> list[str]:
    """The chips the shipped card prints for this answer, from its own module."""
    if NODE is None:
        pytest.skip("node is not on this machine, so the shipped rendering module cannot be run")
    script = (
        f"import {{ findingMessage, visibleFindings }} from {json.dumps(FINDINGS_URL)};\n"
        "let raw = ''; for await (const chunk of process.stdin) raw += chunk;\n"
        "const { body, locale } = JSON.parse(raw);\n"
        "const detail = body.detail ? findingMessage({ message: body.detail, message_he: body.detail_he }, locale) : '';\n"
        "const lines = visibleFindings(body.findings || [], locale, detail).map((line) => (line.chip ? line.chip.text : ''));\n"
        "process.stdout.write(JSON.stringify(lines));\n"
    )
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        input=json.dumps({"body": body, "locale": locale}),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_the_chip_the_card_prints_is_a_header_or_the_word_for_the_place(isolated: TestClient) -> None:
    """The rendered end of it, in both languages, through the shipped module.

    What the payload carries and what the card prints are two claims, and this
    destination has been caught on the gap between them before: the chip is
    built from ``finding.column`` in ``sources-findings.js`` and printed by
    ``SourceCard.jsx``, so the rule is measured where a person would see it.
    """
    for kind in CANDIDATES:
        body = _check(isolated, kind)
        headers = _headers(kind)
        for locale in ("he", "en"):
            for chip in _chips(body, locale):
                if not chip or chip in SCOPE_WORDS:
                    continue
                for part in _named_columns({"column": chip}):
                    assert part in headers, (
                        f"the {kind} card prints the chip {chip} in {locale}, and {part} is neither a header nor a place"
                    )
