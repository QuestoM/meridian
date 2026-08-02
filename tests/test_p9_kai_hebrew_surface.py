"""P9: no server-built machine sentence reaches the operator surface, in either
language.

Two blind readers measured this defect on screen, once per language. First the
Hebrew surface: the undo object printed "nothing would change: every file
already matches the restore point" under a Hebrew heading, and every proposal
item printed a server-built English summary such as "recompute: full week"
verbatim. Then the English surface, after the Hebrew reading landed and the
English branch was left reading the record: the approval card printed the
approved label "Run the plan" and directly under it "recompute: full week",
which puts a word section 8.3 retires in BOTH languages back on the card.

The English strings are records and they stay: the summary goes to the audit
trail and back to the model, and the restore reason is what an API reader gets.
What changed is that each one travels with a stable code, and the surface says
the sentence in the reader's own language from that code, with neither language
falling back on the other.

So the property under test is a pairing, and it is checked in both directions
against the real files rather than described: every code the server can emit has
a reading in the component that prints it, in both languages; no component
prints the raw server string where a reading exists; and no reading in either
language carries a retired word. The frontend is read as text because these are
React modules with no test runner in this repository; a grep that names the
exact line it requires is a weaker test than a rendered assertion and a much
stronger one than nothing.

The last section covers the other half of the same law, found by measuring the
built surface rather than the source. A label is not the only thing a person
reads on the approval card: under it sits prose the model composed, and three
stored proposal reasons and twelve saved answers carry חישוב מחדש. No label
table can reach those, and the prompt rule that forbids the word did not stop
them arriving, so the rename runs at render time. That module is plain
JavaScript and it is executed here through node against the real strings, not
grepped.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from kairos_api import assistant_pipeline, assistant_restore, assistant_summary_terms
from kairos_api import assistant_read_tools, assistant_sections, assistant_tools

ROOT = Path(__file__).resolve().parents[1]
KAI = ROOT / "tv-break-dashboard" / "src" / "kai"
SUMMARY_VIEW = (KAI / "AssistantProposalSummary.jsx").read_text(encoding="utf-8")
UNDO_VIEW = (KAI / "AssistantUndo.jsx").read_text(encoding="utf-8")
TRACE_VIEW = (KAI / "AssistantRunTrace.jsx").read_text(encoding="utf-8")
CARD_VIEW = (KAI / "AssistantProposalCard.jsx").read_text(encoding="utf-8")
CHANGES_VIEW = (KAI / "AssistantConversationsChanges.jsx").read_text(encoding="utf-8")
PANEL_STATE = (KAI / "assistant-panel-state.js").read_text(encoding="utf-8")
THREAD_VIEW = (KAI / "AssistantThread.jsx").read_text(encoding="utf-8")
PANEL_VIEW = (KAI / "AssistantPanel.jsx").read_text(encoding="utf-8")
KAI_VOCABULARY = (KAI / "kai-vocabulary.js").read_text(encoding="utf-8")
DASHBOARD = ROOT / "tv-break-dashboard"
ASSISTANT_DATA = ROOT / "data" / "assistant"

HEBREW = re.compile(r"[֐-׿]")


def _an_advertiser() -> str:
    """A real key from the advertiser store, so the edit path is exercised
    against a record that exists rather than an invented one."""
    from kairos_api.advertisers import _load_frame

    frame = _load_frame()
    assert not frame.empty, "the advertiser store is empty, so no edit can be proposed"
    return str(frame["advertiser_id"].iloc[0])


@pytest.fixture()
def restore_store(tmp_path, monkeypatch):
    monkeypatch.setenv(assistant_restore.DATA_DIR_ENV, str(tmp_path / "assistant"))
    return tmp_path


# --- the proposal summary ----------------------------------------------------
def test_every_english_summary_carries_the_terms_behind_it() -> None:
    """One item per validator whose summary is English, each with its terms."""
    calls = [
        ("propose_settings_change", {"changes": {"min_retention_floor": 0.75}}, "settings"),
        ("propose_recompute", {"scope": "full"}, "recompute"),
        ("propose_recompute", {"scope": {"days": ["2024-11-04"]}}, "recompute"),
        ("propose_constraint", {"constraint": {"scope_type": "always", "effect": "gold"}}, "constraint"),
        ("propose_override", {"override": {"scope": "segment", "kind": "pin",
                                           "target_id": "SEG-1", "value": "2"}}, "override"),
        ("propose_pricing_change", {"changes": {"base_cpp": 120.0}}, "pricing"),
        ("propose_advertiser_change", {"advertiser_name": _an_advertiser(), "create": False,
                                       "changes": {"default_premium": 1.2}}, "advertiser"),
    ]
    for tool, args, code in calls:
        item = assistant_tools.build_proposal_item(tool, {**args, "reason": "test"})
        assert item["status"] == "pending", f"{tool} rejected: {item.get('error')}"
        assert not HEBREW.search(item["summary"]) or code == "pricing", (
            f"{tool} writes a Hebrew summary, so it needs no terms")
        terms = item.get("summary_terms")
        assert terms is not None, f"{tool} carries no terms behind its English summary"
        assert terms["code"] == code
        assert terms["code"] in assistant_summary_terms.CODES


def test_the_terms_carry_the_same_values_the_english_sentence_names() -> None:
    """The two readings come from one payload, so they cannot disagree."""
    item = assistant_tools.build_proposal_item(
        "propose_recompute", {"scope": {"days": ["2024-11-05", "2024-11-04"]}, "reason": "test"})
    assert item["summary"] == "recompute: days 2024-11-04, 2024-11-05"
    assert item["summary_terms"] == {"code": "recompute", "scope": "days",
                                     "days": ["2024-11-04", "2024-11-05"]}

    override = assistant_tools.build_proposal_item(
        "propose_override",
        {"override": {"scope": "segment", "kind": "pin", "target_id": "SEG-9", "value": "3"},
         "reason": "test"})
    assert override["summary"] == "override: pin=3 on segment SEG-9"
    assert override["summary_terms"] == {"code": "override", "kind": "pin", "scope": "segment",
                                         "target_id": "SEG-9", "value": "3"}


def test_a_stored_item_written_before_the_terms_existed_still_gets_them() -> None:
    """Measured on the real batch store: all fifteen items already on disk carry
    their validated payload and none carries terms, because they were written
    before this module existed. Every one of them therefore fell back to the
    English record on both surfaces, which is the half of the leak that no
    change to the component could reach. The terms are derived at read time
    from the item's own payload, so a stored item says what a fresh one says."""
    fresh = assistant_tools.build_proposal_item(
        "propose_recompute", {"scope": "full", "reason": "test"})
    historical = {key: value for key, value in fresh.items() if key != "summary_terms"}
    assert "summary_terms" not in historical
    assert historical["payload"] == {"scope": "full"}, "the payload is what makes this honest"
    assert assistant_summary_terms.terms_for_item(historical) == fresh["summary_terms"]
    assert assistant_summary_terms.terms_for_item(fresh) == fresh["summary_terms"]
    assert assistant_summary_terms.terms_for_item({"kind": "agency_change", "payload": {}}) is None


def test_both_stored_batch_readers_derive_the_terms() -> None:
    """A reading the surface cannot reach is not a reading, so both endpoints
    that hand a stored batch to the surface are checked, not just one."""
    from kairos_api import assistant_actions, assistant_conversations_api

    for module in (assistant_actions, assistant_conversations_api):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "terms_for_item" in source, f"{module.__name__} hands back bare stored items"


def test_an_already_hebrew_summary_carries_no_terms_and_is_printed_as_it_is() -> None:
    """The calendar-event and agency validators write Hebrew, so the surface
    prints their summary itself; inventing terms for them would be a second
    vocabulary that could drift from the first."""
    item = assistant_tools.build_proposal_item(
        "propose_agency_change",
        {"agency_id": "no-such-agency", "action": "deactivate", "reason": "test"})
    # Whether it validates or not, it never gains terms: its kind is not in the map.
    assert item.get("summary_terms") is None


def test_the_surface_has_a_reading_for_every_summary_code() -> None:
    for code in assistant_summary_terms.CODES:
        assert f"terms.code === '{code}'" in SUMMARY_VIEW, f"no reading for {code}"


# The two readings a bilingual pair puts on one source line: t('English', 'עברית')
# for a phrase, and { he: 'עברית', en: 'English' } for a vocabulary token.
_PHRASE_PAIRS = re.compile(r"\bt\('([^']*)', '([^']*)'\)")
_TOKEN_PAIRS = re.compile(r"\{ he: '([^']*)', en: '([^']*)' \}")
RETIRED = re.compile(r"recompute|rebuild|חישוב מחדש|בנייה מחדש", re.IGNORECASE)


def _readings() -> list[tuple[str, str]]:
    """Every (english, hebrew) reading the summary component can print."""
    pairs = [(en, he) for en, he in _PHRASE_PAIRS.findall(SUMMARY_VIEW)]
    pairs += [(en, he) for he, en in _TOKEN_PAIRS.findall(SUMMARY_VIEW)]
    return pairs


def test_the_english_branch_reads_the_terms_instead_of_the_record() -> None:
    """The measured defect: the reading was gated on the Hebrew locale, so an
    operator with the header set to English got the server's record verbatim."""
    assert "locale === 'he' && terms" not in SUMMARY_VIEW, (
        "the reading is gated on Hebrew, so English prints the raw summary again")
    assert "const said = terms ? say(terms, locale) : null;" in SUMMARY_VIEW, (
        "the reading is not taken for every locale")
    assert "export function hasReading(item, locale)" in SUMMARY_VIEW


def test_every_reading_carries_both_languages_and_neither_is_the_record() -> None:
    pairs = _readings()
    assert len(pairs) >= len(assistant_summary_terms.CODES), (
        f"only {len(pairs)} bilingual readings for {len(assistant_summary_terms.CODES)} codes")
    for english, hebrew in pairs:
        assert english.strip(), f"a reading has no English side: {hebrew!r}"
        assert hebrew.strip(), f"a reading has no Hebrew side: {english!r}"
        assert HEBREW.search(hebrew), f"the Hebrew side is not Hebrew: {hebrew!r}"
        assert not HEBREW.search(english), f"the English side is not English: {english!r}"


def test_no_reading_in_either_language_carries_a_retired_word() -> None:
    """Section 8.3 retires recompute, rebuild, חישוב מחדש and בנייה מחדש in BOTH
    languages, and the summary component is the one surface that used to print
    the server's own word for the act."""
    for english, hebrew in _readings():
        assert not RETIRED.search(english), f"a retired word on the English surface: {english!r}"
        assert not RETIRED.search(hebrew), f"a retired word on the Hebrew surface: {hebrew!r}"


def test_the_run_act_is_said_with_the_approved_label_in_both_languages() -> None:
    """The server's record for this act is 'recompute: full week' and it stays a
    record. What the card prints under the 'Run the plan' heading is this."""
    item = assistant_tools.build_proposal_item("propose_recompute", {"scope": "full", "reason": "t"})
    assert item["summary"] == "recompute: full week", "the record changed, not the reading"
    assert item["summary_terms"] == {"code": "recompute", "scope": "full"}
    readings = dict(_readings())
    for english, hebrew in readings.items():
        if english.startswith("Run the plan"):
            assert "הרצת התוכנית" in hebrew
            break
    else:
        raise AssertionError("no reading says the run act with the approved English label")
    assert "Run the plan for the whole week" in readings
    assert "Run the plan for these days: " in readings


def test_the_card_and_the_changes_view_both_print_through_the_reading() -> None:
    for name, source in (("card", CARD_VIEW), ("changes view", CHANGES_VIEW)):
        assert "<ProposalSummary item={" in source, f"the {name} does not use the reading"
        assert "{String(item.summary)}" not in source, f"the {name} still prints the raw summary"
        assert "{item.summary}<" not in source, f"the {name} still prints the raw summary"
    assert "summary_terms" in PANEL_STATE, "the batch whitelist would drop the terms"
    # The changes endpoint sends the terms beside a frozen item key set, so the
    # view has to merge them back or its reading silently falls back to the
    # record. Both halves are asserted, the payload here and the merge below.
    assert "item_terms" in CHANGES_VIEW, "the changes view drops the terms it is sent"
    assert "withTerms(item, batch)" in CHANGES_VIEW


# --- the restore preview -----------------------------------------------------
def test_every_reason_the_preview_emits_carries_its_code(restore_store) -> None:
    settings = restore_store / "kairos_settings.json"
    settings.write_text(json.dumps({"revenue_weight": 50}), encoding="utf-8")
    restore_id = assistant_restore.snapshot([settings], "batch", ["item"])
    preview = assistant_restore.preview(restore_id)
    assert preview["reason_code"] == "nothing_would_change"
    assert preview["reason"], "the English record stays for an API reader"

    (assistant_restore._restore_root() / restore_id / "kairos_settings.json").unlink()
    settings.write_text('{"revenue_weight": 60}', encoding="utf-8")
    row = assistant_restore.preview(restore_id)["files"][0]
    assert row["effect"] == "unavailable"
    assert row["reason_code"] == "snapshot_missing"


def test_a_file_absent_at_snapshot_carries_its_note_code(restore_store) -> None:
    fresh = restore_store / "new_store.json"
    restore_id = assistant_restore.snapshot([fresh], "batch", ["item"])
    fresh.write_text("{}", encoding="utf-8")
    row = assistant_restore.preview(restore_id)["files"][0]
    assert row["note_code"] == "absent_at_snapshot"


def test_the_undo_panel_has_a_hebrew_reading_for_every_preview_code() -> None:
    for code in assistant_restore.PREVIEW_CODES:
        assert f"{code}: [" in UNDO_VIEW, f"no reading for {code}"
        block = UNDO_VIEW.split(f"{code}: [", 1)[1].split("],", 1)[0]
        assert HEBREW.search(block), f"the reading for {code} carries no Hebrew"
    assert "{String(preview.reason)}" in UNDO_VIEW, "the honest fallback is gone"
    assert "reasonText(preview.reason_code, locale) ||" in UNDO_VIEW, (
        "the fallback must be second, not first")


# --- the run trace -----------------------------------------------------------
def test_every_provenance_source_the_server_stamps_has_a_hebrew_reading() -> None:
    """A source chip prints on every step of every run, so an unread one is the
    most visible English on the surface."""
    missing = [source for source in sorted(set(assistant_read_tools.SOURCE_BY_TOOL.values()))
               if f"'{source}':" not in TRACE_VIEW]
    assert missing == [], f"sources with no Hebrew reading: {missing}"


def test_the_model_words_are_not_carried_into_the_operator_reading() -> None:
    """Section 4.2's lexicon test: a run surface shows no gate, no coefficient,
    no drift. Two English sources name them; their readings do not."""
    readings = re.findall(r"':\s*'([^']+)'", TRACE_VIEW.split("const SOURCE_HE", 1)[1].split("};", 1)[0])
    assert readings, "the source table did not parse"
    for reading in readings:
        assert "שער האימון" not in reading
        assert "מקדמים" not in reading
        assert "סטייה" not in reading


def test_the_scope_reason_travels_with_a_code_the_trace_reads() -> None:
    facts = assistant_pipeline.grounding_facts(
        {"counts": {"segments": 0, "breaks": 0,
                    "reason": assistant_sections.EMPTY_PLAN_REASON}})
    assert facts["scope_reason_code"] == "empty_plan"
    for code in assistant_sections.SCOPE_REASON_CODES.values():
        assert f"{code}: [" in TRACE_VIEW, f"the trace has no reading for {code}"


# --- prose the model wrote ---------------------------------------------------
# The card prints two things under one another: a label the product wrote, and a
# sentence the model wrote. The label half is above. This half is the sentence,
# where there is no table to fix and the word arrives at render time.
def _renamed(strings: list[str]) -> list[str]:
    """The surface's own rename, executed by node exactly as the bundler
    imports it, so this is the behaviour and not a description of it."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH, so the rename cannot be executed")
    script = (
        "const m = await import('./src/kai/kai-vocabulary.js');"
        "const input = JSON.parse(process.env.KAI_TEST_INPUT);"
        "process.stdout.write(JSON.stringify(input.map((text) => m.inApprovedWords(text))));"
    )
    done = subprocess.run(
        [node, "--input-type=module", "-e", script],
        cwd=DASHBOARD, capture_output=True, text=True, timeout=120,
        env={**os.environ, "KAI_TEST_INPUT": json.dumps(strings)},
    )
    assert done.returncode == 0, f"the rename module did not load: {done.stderr[-800:]}"
    return json.loads(done.stdout)


def _stored_model_prose() -> list[tuple[str, str]]:
    """Every string on disk that the surface prints as prose the model wrote:
    a proposal's reason and a saved answer. The question is not here, because it
    is the operator's own text and is shown back unchanged."""
    found: list[tuple[str, str]] = []
    proposals = ASSISTANT_DATA / "proposals.json"
    if proposals.exists():
        for batch in json.loads(proposals.read_text(encoding="utf-8")).get("batches", []):
            for item in batch.get("items", []):
                if item.get("reason"):
                    found.append((f"reason {str(item.get('id'))[:8]}", str(item["reason"])))
    for path in sorted((ASSISTANT_DATA / "threads").glob("*.json")) if (ASSISTANT_DATA / "threads").exists() else []:
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        for entry in saved.get("entries", []) if isinstance(saved, dict) else []:
            if entry.get("answer"):
                found.append((f"answer {path.stem}", str(entry["answer"])))
    return found


def test_the_rename_says_the_activity_in_the_products_own_word() -> None:
    """Both languages, both retired stems, and the tense the sentence was in."""
    given = [
        "חישוב מחדש כדי ששינוי משקל ההכנסות ל-62 ייכנס לתוכנית השבועית.",
        "ללא חישוב מחדש התוכנית תישאר ישנה.",
        "בנייה מחדש של הלוח.",
        "Recompute to refresh the schedule.",
        "This requires a recompute and a rebuild.",
        "Rebuilding the plan is recomputing it.",
        "the schedule was recomputed and rebuilt",
    ]
    out = _renamed(given)
    for original, said in zip(given, out):
        assert not RETIRED.search(said), f"a retired word survived: {original!r} -> {said!r}"
    assert out[0].startswith("הרצה כדי")
    assert out[3] == "Run to refresh the schedule."
    assert out[5] == "Running the plan is running it."


def test_the_rename_leaves_an_identifier_and_a_route_exactly_as_written() -> None:
    """A retired word inside a name is a name. The tool really is called
    propose_recompute and the route really is /api/recompute, and a sentence
    that quotes either must still be true after the rename."""
    given = ["The tool is propose_recompute and the route is /api/recompute.",
             "recompute_api.py holds it."]
    assert _renamed(given) == given


def test_the_operators_own_question_is_never_renamed() -> None:
    """Two render sites take the operator's own words. Both use the plain
    renderer, and only the model's prose goes through ModelText."""
    for name, source in (("thread", THREAD_VIEW), ("panel", PANEL_VIEW)):
        for line in source.splitlines():
            if 'className="asst-q"' in line:
                assert "<RichText" in line, f"the {name} renames the operator's own question"
    assert '<ModelText className="asst-a" text={entry.answer} />' in THREAD_VIEW
    assert '<ModelText className="asst-a" text={live.text} />' in PANEL_VIEW
    assert "{inApprovedWords(item.reason)}" in CARD_VIEW, "the card prints the raw reason"


def test_the_approved_word_is_read_from_the_shared_vocabulary() -> None:
    """Section 8.3 gives vocabulary.js the words. This module holds the retired
    forms and nothing else, so every approved reading it emits has to start with
    the activity noun that file defines."""
    assert "from '../vocabulary.js'" in KAI_VOCABULARY, "the approved word is not imported"
    vocabulary = (ROOT / "tv-break-dashboard" / "src" / "vocabulary.js").read_text(encoding="utf-8")
    assert re.search(r"'activity\.run':\s*\{\s*\n\s*en: 'run'", vocabulary), (
        "the vocabulary's word for the activity moved, so the table below must move with it")
    for approved in re.findall(r"approved: '([^']+)'", KAI_VOCABULARY):
        assert approved.startswith("run"), f"an approved reading not built on the vocabulary noun: {approved!r}"
    assert "approved: RUN_HE" in KAI_VOCABULARY, "the Hebrew reading is spelled here instead of imported"


def test_no_retired_word_survives_on_any_stored_reason_or_answer() -> None:
    """The measurement that found this, run as the test. Every reason and every
    saved answer on disk goes through the surface's rename, and none of them may
    still carry one of the four."""
    prose = _stored_model_prose()
    if not prose:
        pytest.skip("no stored assistant prose on this machine")
    leaking = [(where, text) for where, text in prose if RETIRED.search(text)]
    said = _renamed([text for _, text in prose])
    still = [(where, out) for (where, _), out in zip(prose, said) if RETIRED.search(out)]
    assert still == [], f"a retired word reaches the surface: {still[:2]}"
    assert leaking, (
        "no stored string carries a retired word, so this test proved nothing. "
        "It is kept because the defect it guards was measured on this data")
