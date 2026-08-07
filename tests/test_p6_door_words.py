"""P6 Sources, the door: no Python value may reach a card where a word goes.

Written as a CLASS guard and not a site guard, because this destination has now
lost four rounds to the same law wearing a different name: an internal
serialisation reaching an operator surface. It cost three rounds each on
``air_dt``, on ``<header>`` and on ``channels``, and it came back a fourth time
in one language only.

**Measured on the shipped card, locale en, before the fix these tests guard:** a
spots export whose Channel column carried two unknown names rendered
``channel(s) not in known set: ['זרזיר 99', 'ערוץ פלוני']``, brackets, quotes and
comma exactly as ``repr`` wrote them, and the bidi run then broke them across the
wrap so a stray ``]`` sat alone on a line. The identical finding one click away
in Hebrew read a plain comma-joined list with a recomputed count. The English
half of the refusal headline quoted the flat ``[error] column: code - detail``
lines on top of that, and ``NaN value(s) present; not imputed``, ``expected a
datetime64 dtype`` and ``DataFrame is None`` are the engine talking to itself on
a card a media steward reads. A test named after one of those sentences is how
the class survived four rounds, so nothing here is named after a sentence.

Two things are guarded and both are swept whole. The copy table, every entry and
every language slot in it, so a code somebody adds next year cannot ship with a
collection in it. And every finding a live check really returns, rendered
through the module the card imports, in BOTH locales, so the wiring that carries
the authored half to the screen is measured and not assumed.

What is deliberately NOT swept: the flat ``errors`` and ``warnings`` lines. They
are the machine record every existing reader parses and their ``[error]``
scaffolding is theirs by contract. They are not a card.
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
import kairos_api.uploads_messages as uploads_messages

ROOT = Path(__file__).resolve().parents[1]
HEBREW = re.compile(r"[֐-׿]")
FIELD = re.compile(r"\{(\w+)\}")

# The shipped module the card renders its findings by, run as itself rather than
# reimplemented here, so this file measures what an operator sees.
NODE = shutil.which("node")
MODULE_URL = (ROOT / "tv-break-dashboard" / "src" / "sources" / "sources-findings.js").as_uri()

# What a Python value looks like when it lands on a card instead of a sentence.
# The brackets of a list and the braces of a dict or of a placeholder nothing
# filled; the quote-comma-quote a repr writes between two items, straight or
# curly; and the engine's own vocabulary for its own internals. Not one of these
# is a word in either language this destination is read in.
REPR_ON_A_CARD = re.compile(
    r"[\[\]{}]"
    r"|['‘’\"“”]\s*,\s*['‘’\"“”]"
    r"|\bNone\b|\bnan\b|\bNaN\b|\bDataFrame\b|\bdtype\b|\bdatetime64\b"
)

# The eleven codes the frozen kairos.data.contracts raise, whose English detail
# is the contract's own and whose readable pair this destination authors.
FROZEN_CODES = frozenset({
    "missing_frame", "not_a_frame", "missing_column", "not_datetime",
    "non_numeric_values", "nan_values", "negative_values", "non_positive_values",
    "nan_channel", "unknown_channel", "end_before_start",
})


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


def _spots(**columns: list) -> bytes:
    """A spots export, valid except for whatever the caller breaks in it."""
    frame = pd.DataFrame({
        "Date": ["01/05/2025", "01/05/2025", "01/05/2025"],
        "Start time": ["20:00:00", "20:01:00", "20:02:00"],
        "Campaign": ["Acme", "Acme", "Acme"],
        "Channel": ["רשת 13", "רשת 13", "רשת 13"],
        "Duration": [30, 30, 30],
        "TVR": [5.5, 5.5, 5.5],
        **columns,
    })
    return frame.to_csv(index=False).encode("utf-8")


def _dayparts(**columns: list) -> bytes:
    frame = pd.DataFrame({
        "Dates": ["01/05/2025", "02/05/2025"],
        "Timebands": ["20:00", "20:01"],
        "רשת 13": ["1.0", "2.0"],
        **columns,
    })
    return frame.to_csv(index=False).encode("utf-8")


# Every check this door can be driven through here, chosen so that between them
# they raise the frozen codes a real export can actually reach, each alongside
# the door's own. The names say what is wrong with the file, never which
# sentence is expected back, because the point is to sweep whatever comes.
BROKEN_FILES: dict[str, tuple[str, bytes]] = {
    "two unknown channel names": ("spots", _spots(Channel=["זרזיר 99", "זרזיר 99", "ערוץ פלוני"])),
    "one unknown channel name": ("spots", _spots(Channel=["רשת 13", "רשת 13", "ערוץ פלוני"])),
    "one blank channel": ("spots", _spots(Channel=["רשת 13", "רשת 13", ""])),
    "one unreadable duration": ("spots", _spots(Duration=[30, 30, "not a number"])),
    "one blank rating": ("spots", _spots(TVR=[5.5, 5.5, ""])),
    "one duration at zero": ("spots", _spots(Duration=[30, 30, 0])),
    "every date unreadable": ("spots", _spots(Date=["31/31/2025"] * 3)),
    "one rating below zero": ("dayparts", _dayparts(**{"רשת 13": ["1.0", "-2.0"]})),
    "every rating below zero": ("dayparts", _dayparts(**{"רשת 13": ["-1.0", "-2.0"]})),
    "no data rows at all": ("dayparts", pd.DataFrame(columns=["Dates", "Timebands", "רשת 13"]).to_csv(index=False).encode("utf-8")),
    "no channel column recognised": ("dayparts", _dayparts(**{"רשת 13": None, "Channel_A": ["1.0", "2.0"]})),
    "none of the required columns": ("daily", b"a,b\n1,2\n"),
    "not a csv at all": ("daily", b"\xff\xfe\x00not a csv"),
    "an empty file": ("daily", b""),
}


def _checked(client: TestClient) -> dict[str, dict]:
    """Every response body this door can be driven to, keyed by what was wrong."""
    return {
        name: client.post(f"/api/uploads/{kind}/check", files={"file": (f"{kind}.csv", raw, "text/csv")}).json()
        for name, (kind, raw) in BROKEN_FILES.items()
    }


def _as_the_card_renders(body: dict, locale: str) -> list[str]:
    """Every sentence the shipped card prints for this response, in one locale.

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
        "const lines = visibleFindings(body.findings || [], locale, detail).map((line) => line.message);\n"
        "process.stdout.write(JSON.stringify([detail, ...lines].filter(Boolean)));\n"
    )
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        input=json.dumps({"body": body, "locale": locale}),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_no_entry_in_the_copy_table_carries_a_python_value_in_any_language() -> None:
    """The table itself, every entry and every language slot, swept whole.

    A placeholder is removed before the sweep because a placeholder is filled;
    a brace left after that removal is a hole nothing filled, which prints on a
    card exactly as badly as a bracket does.
    """
    for code, words in uploads_messages.MESSAGES.items():
        for slot, sentence in words.items():
            found = REPR_ON_A_CARD.search(FIELD.sub("", sentence))
            assert not found, f"{code}.{slot} prints a Python value on a card: {found.group(0)!r}"


def test_every_language_slot_this_table_offers_is_offered_in_both_languages() -> None:
    """The asymmetry that caused this round: a half authored for one reader only.

    An entry may have two slots or four, never three, and the placeholders in
    each half must be the same set, so no sentence can carry a number the other
    language has no room for.
    """
    for code, words in uploads_messages.MESSAGES.items():
        assert ("en" in words) and ("he" in words), f"{code} is missing one of its two languages"
        assert ("en_one" in words) == ("he_one" in words), f"{code} has a singular in one language only"
        assert HEBREW.search(words["he"]), f"{code} has a Hebrew slot with no Hebrew in it"
        for one, many in (("en_one", "en"), ("he_one", "he")):
            if one in words:
                extra = set(FIELD.findall(words[one])) - set(FIELD.findall(words[many]))
                assert not extra, f"{code}.{one} has a placeholder its plural has no room for: {sorted(extra)}"


def test_both_halves_of_a_frozen_code_are_built_by_one_call_and_arrive_together() -> None:
    """The seam itself: one entry, one pass, so the two cannot drift again.

    A code that cannot be rendered renders nothing in BOTH languages rather
    than leaving one reader with the contract's own sentence while the other
    gets a written one, which is exactly the state this round found.
    """
    for code in FROZEN_CODES:
        for count in (None, 0, 1, 2, 40):
            english, hebrew = uploads_messages.contract_say(code, count, "ערוץ פלוני")
            assert bool(english) == bool(hebrew), f"{code} at count {count} rendered one language only"
            if english:
                assert HEBREW.search(hebrew), f"{code} at count {count} has no Hebrew in its Hebrew"
                for half in (english, hebrew):
                    found = REPR_ON_A_CARD.search(half)
                    assert not found, f"{code} at count {count} prints {found.group(0)!r}"


def test_a_count_of_one_reads_as_one_in_both_languages() -> None:
    """The defect round six ruled on, back in the half this round authored.

    Measured live before this: one missing value read ``1 ערכים בעמודה הזאת
    חסרים`` and one missing channel would have read ``ב־1 שורות``, both
    ungrammatical for a count of one, and the English beside them hedged with
    ``value(s)``. A sentence about one thing carries no digit and no hedge.
    """
    for code in FROZEN_CODES:
        many = uploads_messages.contract_say(code, 2, "ערוץ פלוני")
        if not many[0] or "2" not in many[0]:
            continue
        for half in uploads_messages.contract_say(code, 1, "ערוץ פלוני"):
            assert "1" not in half, f"{code} said one as a numeral: {half}"
            assert "(s)" not in half, f"{code} hedged the plural instead of saying one: {half}"


def test_no_finding_a_live_check_returns_prints_a_python_value_on_the_card(isolated: TestClient) -> None:
    """The class, measured where it lands: the shipped card, both locales.

    Every response this door can be driven to, rendered by the module the card
    imports, every sentence it would print swept for the punctuation a Python
    value carries. This is the assertion that would have failed for four rounds.
    """
    bodies = _checked(isolated)
    for locale in ("he", "en"):
        for name, body in bodies.items():
            printed = _as_the_card_renders(body, locale)
            assert printed, f"the {name} check printed no reason at all in {locale}"
            for sentence in printed:
                found = REPR_ON_A_CARD.search(sentence)
                assert not found, f"{name} printed {found.group(0)!r} in {locale}: {sentence}"


def test_every_frozen_finding_carries_a_written_sentence_for_each_reader(isolated: TestClient) -> None:
    """The wiring, not the words: the authored halves must reach the payload.

    A frozen violation keeps the contract's own detail in ``message``, where the
    machine record needs it, and carries ``message_en`` and ``message_he``
    beside it. Either half missing puts that reader back on the contract's own
    sentence, which is the state this test exists to make loud.
    """
    seen: set[str] = set()
    for name, body in _checked(isolated).items():
        for finding in body.get("findings") or []:
            if finding["code"] not in FROZEN_CODES:
                continue
            seen.add(finding["code"])
            for half in ("message_en", "message_he"):
                assert finding.get(half), f"{name} left {finding['code']} with no {half}"
                assert finding[half] != finding["message"], f"{name} quoted the contract as its {half}"
    assert len(seen) >= 5, f"this sweep reached too few frozen codes to mean anything: {sorted(seen)}"


def test_the_english_card_lists_the_channel_names_rather_than_a_python_list(isolated: TestClient) -> None:
    """The critic's exact reproduction, on the exact card, in the failing locale.

    Two unknown names in a spots export's Channel column. What the English card
    printed was the contract's ``repr``; what it prints is the names, joined the
    way the Hebrew beside it has joined them since round two, and both halves
    count the two NAMES rather than the three rows carrying them.
    """
    body = isolated.post(
        "/api/uploads/spots/check",
        files={"file": ("Spots.csv", _spots(Channel=["זרזיר 99", "זרזיר 99", "ערוץ פלוני"]), "text/csv")},
    ).json()
    finding = next(f for f in body["findings"] if f["code"] == "unknown_channel")
    assert finding["rows_total"] == 3, "three rows carry an unknown channel name"
    printed = " ".join(_as_the_card_renders(body, "en"))
    assert "זרזיר 99" in printed and "ערוץ פלוני" in printed, "the English card lost the names it is about"
    assert "'" not in printed and "[" not in printed, f"the English card still prints a Python list: {printed}"
    assert "2" in finding["message_en"], "the English count is not the two names"
    assert "3" not in finding["message_en"], "the row count leaked into a sentence about channel names"


def test_the_refusal_headline_reads_the_same_reasons_in_both_languages(isolated: TestClient) -> None:
    """The second site of the same class, one level up from the finding.

    The English half of a contract refusal quoted the flat ``[error] column:
    code - detail`` lines, so one locale read internal scaffolding and a bracket
    where the other read three sentences. Both halves now read the findings.
    ``errors`` still carries the flat lines for the readers that parse them.
    """
    response = isolated.post(
        "/api/uploads/spots", files={"file": ("Spots.csv", _spots(Date=["31/31/2025"] * 3), "text/csv")}
    )
    assert response.status_code == 400, "a file the engine can place on no day may not replace the live input"
    body = response.json()
    for half in ("detail", "detail_he"):
        found = REPR_ON_A_CARD.search(body[half])
        assert not found, f"the refusal headline prints {found.group(0)!r} in {half}: {body[half]}"
        assert "[error]" not in body[half], f"{half} quotes the machine record at a reader"
    assert any(line.startswith("[error] ") for line in body["errors"]), "the flat machine lines were dropped"
