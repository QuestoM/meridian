"""The words this destination puts in front of a reader, and where they come from.

Three defects of one kind, closed together and guarded here. A run surface may
print an engine's number; it may not print the engine's own vocabulary, and it
may not print a sentence that was true when it was typed rather than when it was
read.

  * The money card's band line was a hand-written operator sentence in Hebrew
    and the backend's precise engine sentence in English, so the English reader
    got ``coefficient``, ``risk_lambda``, ``ci_low``, ``ci_high`` and two
    payload field names under the net figure. Both languages are written for the
    reader now, and the engine's sentence stays in the payload for whoever reads
    the API. The net comparison card, one component below in the same file, had
    the identical split and the identical hit, and is closed the same way.
  * The drill's delivered block asserted a fact about the disk. It is read off
    the disk now, on every call.
  * The job picker told a model steward their console was being built while it
    was reachable, and it sat above the three answers on the one visit where
    nobody has explained the screen yet.

Everything here is checked against the source and against the real data, never
against a copy of the sentence, so a rewrite that keeps the meaning passes and a
rewrite that loses it fails.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
TODAY = SRC / "today"
MONEY = TODAY / "MoneyWaterfall.jsx"
PICKER = TODAY / "JobPicker.jsx"
PAGE = TODAY / "OverviewPage.jsx"
BRIDGE = SRC / "model" / "console-bridge.jsx"
SHIPPED_DAILY = ROOT / "data" / "daily_input"

# Section 4.2's list, plus the payload and engine names the backend's own band
# sentence carries. None of them belongs in a sentence a general manager reads.
TRAINING_LEXICON = (
    "gate",
    "held_out",
    "tau",
    "drift",
    "coefficient",
    "pooling",
    "p_value",
    "training_window",
    "wartime",
)
INTERNAL_NAMES = ("ci_low", "ci_high", "risk_lambda", "retention_cost_high", "retention_cost_low")


def _band_note() -> str:
    """The body of bandBasisNote, which is where the two sentences are written."""
    text = MONEY.read_text(encoding="utf-8")
    assert "function bandBasisNote(" in text, "the money card no longer has a band note to guard"
    return text.split("function bandBasisNote(", 1)[1].split("\n}", 1)[0]


def test_the_money_card_writes_both_sentences_rather_than_printing_the_engines():
    body = _band_note()
    sentences = re.findall(r"return '([^']+)';", body)
    assert len(sentences) == 2, "one sentence per language, both written for the reader"
    assert any(re.search(r"[֐-׿]", sentence) for sentence in sentences), "the Hebrew branch is gone"
    assert any(re.search(r"[A-Za-z]", sentence) for sentence in sentences), "the English branch is gone"
    for sentence in sentences:
        for word in TRAINING_LEXICON + INTERNAL_NAMES:
            assert not re.search(word, sentence, re.I), f"{word!r} reaches the money card"


def test_the_comparison_card_writes_its_basis_line_the_same_way():
    """The same defect lived one component below, in the same file, unnamed."""
    text = MONEY.read_text(encoding="utf-8")
    assert "payload.basis" not in text and "payload?.basis" not in text
    block = text.split("net-compare-basis", 1)[1].split("</p>", 1)[0]
    sentences = re.findall(r"'([^']{40,})'", block)
    assert len(sentences) == 2, "one basis sentence per language, both written for the reader"
    for sentence in sentences:
        for word in TRAINING_LEXICON + INTERNAL_NAMES:
            assert not re.search(word, sentence, re.I), f"{word!r} reaches the comparison card"
    assert sum("forecast" in sentence for sentence in sentences) == 1
    assert sum("תחזית" in sentence for sentence in sentences) == 1


def test_the_engines_own_band_sentence_is_read_by_nothing_on_this_destination():
    """It stays in the payload. This is the test that would catch it coming back."""
    from kairos_api.yield_api import _RETENTION_BAND_BASIS

    assert re.search("coefficient", _RETENTION_BAND_BASIS), "the backend sentence this guards has changed"
    for path in sorted(TODAY.iterdir()):
        if path.suffix not in {".js", ".jsx"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert ".retention_cost_basis" not in text, f"{path.name} still reads the engine's sentence"
        assert _RETENTION_BAND_BASIS not in text, f"{path.name} carries a copy of the engine's sentence"


def test_the_model_stewards_row_names_the_console_instead_of_calling_it_unbuilt():
    text = PICKER.read_text(encoding="utf-8")
    notes = text.split("const DOOR_NOTES = {", 1)[1].split("\n};", 1)[0]
    row = [line for line in notes.splitlines() if "model.console" in line]
    assert len(row) == 1, "the model steward's row lost its note"
    assert "being built" not in row[0] and "בבנייה" not in row[0]
    assert "console" in row[0].lower() and "קונסולת המודל" in row[0]


def test_the_picker_opens_the_console_at_the_address_the_console_publishes():
    """A name that reaches nothing is a dead end, and a wrong address is worse."""
    published = re.search(r"const CONSOLE_HASH = '([^']+)'", BRIDGE.read_text(encoding="utf-8"))
    assert published, "the console no longer publishes a hash, so this door needs rewiring"
    text = PICKER.read_text(encoding="utf-8")
    used = re.search(r"const CONSOLE_HASH = '([^']+)'", text)
    assert used, "the picker no longer holds the console address"
    assert used.group(1) == published.group(1)
    assert "window.location.hash = CONSOLE_HASH" in text
    assert "row.door === 'model.console'" in text, "the address must be reachable from that row only"


def test_the_picker_asks_its_question_below_the_three_answers():
    """The unaided visit is the one that can least afford a scroll to the answers."""
    page = PAGE.read_text(encoding="utf-8")
    picker = page.index("<JobPicker")
    for answer in ("<TodayMoney", "<TodayHealth", "<TodayDecisions"):
        assert page.index(answer) < picker, f"{answer} renders after the picker"


# ---------------------------------------------------------------------------
# The delivered absence, derived rather than asserted
# ---------------------------------------------------------------------------


def _delivered(iso_date: str) -> dict:
    from kairos_api.overview_api_drill import delivered_state

    return delivered_state(iso_date)


def test_the_absence_names_the_coverage_the_ledger_on_disk_actually_has():
    from kairos.export.spots_coverage import daily_input_days

    covered = sorted(daily_input_days())
    assert covered, "the shipped ledger has coverage, so this test can bite"
    body = _delivered("2024-11-03")
    assert body["available"] is False
    assert body["state"] == "unavailable"
    assert body["covers"] == covered
    for date in covered:
        assert date in body["reason_en"] and date in body["reason_he"]


def test_a_day_the_ledger_does_reach_gets_a_different_absence_and_a_different_path():
    from kairos.export.spots_coverage import daily_input_days

    covered = sorted(daily_input_days())
    body = _delivered(covered[0])
    assert body["available"] is False
    assert body["opens"] == "plan", "the missing thing there is the link, not the feed"
    assert covered[0] not in body["reason_en"], "it says this day, not a date it has just been handed"


def test_a_delivery_file_for_the_planned_week_changes_the_sentence(tmp_path, monkeypatch):
    """The defect a constant sentence has: it cannot notice the file that ends it.

    A real daily file, re-dated to a planned day and read by the same reader the
    product uses, has to move the answer. Nothing is stubbed but the folder.
    """
    from kairos.export import spots_coverage

    shipped = sorted(SHIPPED_DAILY.glob("*.csv"))
    assert shipped, "there is no daily file to re-date, so this test would prove nothing"
    frame = pd.read_csv(shipped[0], encoding="utf-8")
    assert "תאריך" in frame.columns
    frame["תאריך"] = "11/3/2024"
    frame.to_csv(tmp_path / "delivery_for_the_planned_week.csv", index=False, encoding="utf-8")
    monkeypatch.setattr(spots_coverage, "DAILY_INPUT_DIR", tmp_path)

    body = _delivered("2024-11-03")
    assert body["covers"] == ["2024-11-03"]
    assert body["opens"] == "plan"
    assert "covers this day" in body["reason_en"]
    assert _delivered("2024-11-04")["covers"] == ["2024-11-03"]
    assert "2024-11-03" in _delivered("2024-11-04")["reason_en"]


def test_an_empty_ledger_folder_reads_as_no_ledger_and_never_as_a_zero(tmp_path, monkeypatch):
    from kairos.export import spots_coverage

    monkeypatch.setattr(spots_coverage, "DAILY_INPUT_DIR", tmp_path)
    body = _delivered("2024-11-03")
    assert body["state"] == "unavailable"
    assert body["covers"] == []
    assert body["opens"] == "sources"
    assert "no priced spot ledger on disk" in body["reason_en"]


def test_a_ledger_that_cannot_be_read_is_unknown_and_not_absent(monkeypatch):
    """The third state. Not there and cannot tell are different answers."""
    import kairos.export.spots_coverage as spots_coverage

    def raise_it(*_args, **_kwargs):
        raise OSError("the folder is not readable")

    monkeypatch.setattr(spots_coverage, "daily_input_days", raise_it)
    body = _delivered("2024-11-03")
    assert body["state"] == "unknown"
    assert body["covers"] is None
    assert "unknown" in body["reason_en"]


@pytest.mark.parametrize(
    "covered,expected",
    [
        (["2025-04-27"], "a single day, 2025-04-27"),
        (["2025-04-27", "2025-05-30"], "2 days between 2025-04-27 and 2025-05-30"),
    ],
)
def test_the_coverage_phrase_is_built_from_the_dates_and_not_from_a_guess(covered, expected):
    from kairos_api.overview_api_drill import _span

    english, hebrew = _span(covered)
    assert english == expected
    for date in (covered[0], covered[-1]):
        assert date in hebrew


def test_no_figure_ever_appears_inside_the_delivered_block():
    """An absence with a number in it is the thing this whole block exists to avoid."""
    for iso_date in ("2024-11-03", "2030-01-01"):
        body = _delivered(iso_date)
        assert not any(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in body.values()
        )
