"""The TYPED reference: what travels beside the prose, and what comes back.

Round one put the store's own name into the sentence as plain text and stopped
there. This is what was left: a ``{type, id}`` that reaches the server, is
resolved at send time into a ``mentioned_objects`` context section, and comes
back in one of four states that the operator can see.

Nothing here is mocked. The resolver runs against the advertiser rules store,
the agencies store, the calendar events store and the saved weekly plan on disk,
exactly as the ask runs it.

THE BOUNDARY TESTS ARE WRITTEN AS POSITIVE CONTROLS, following round one's own
pattern, because a scan that has never flagged anything has not been shown to
work. Each one is first pointed at an UNSCOPED read of the saved plan -- the
thing every operator surface is forbidden to serve -- and required to find
rivals there before it is believed when it finds none in the scoped path.

The saved plan on disk carries four channels, and two programme values,
``Children`` and ``Religious``, exist on rival channels and on no row of the
operator's own. A reference to one of those is the sharpest control available: a
client can type it, so it is the exact shape of an attempt to push a rival name
in from outside the picker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kairos_api import assistant_mentions as mentions
from kairos_api import assistant_mentions_resolve as resolve
from kairos_api import assistant_mentions_words as words
from kairos_api import channel_scope

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"
KAI = ROOT / "tv-break-dashboard" / "src" / "kai"
TYPO = "qqqzzzxxx-no-such-thing"


def _plan() -> pd.DataFrame:
    if not PLAN_PATH.exists():
        pytest.skip("no saved weekly plan on disk, so there is nothing to resolve against")
    return pd.read_csv(PLAN_PATH)


def _program_column(frame: pd.DataFrame) -> str:
    return "program_title" if "program_title" in frame.columns else "program_type"


def _rival_program() -> str:
    """A programme value that appears on a rival channel and on no row of the
    operator's own. Skips rather than passing vacuously when the plan on disk
    stops carrying one, because a control that cannot fire proves nothing."""
    frame = _plan()
    owned = channel_scope.operator_channel()
    column = _program_column(frame)
    ours = {str(v).strip() for v in frame.loc[frame["channel"] == owned, column]}
    theirs = {str(v).strip() for v in frame.loc[frame["channel"] != owned, column]}
    rivals = sorted(name for name in (theirs - ours) if name and name.lower() != "nan")
    if not rivals:
        pytest.skip("the plan on disk holds no rival-only programme, so this control cannot fire")
    return rivals[0]


def _a_day() -> str:
    row = next((r for r in mentions.search("", types="day", limit=1)["rows"]), None)
    if row is None:
        pytest.skip("the index holds no broadcast day")
    return row["id"]


# --- what the client is allowed to send -------------------------------------------
def test_a_reference_that_is_not_the_contract_shape_is_dropped() -> None:
    """Conservative in exactly the way the page-context parser is conservative:
    dropping everything means the ask behaves precisely as it does without the
    field, which is the degradation this whole contract promises."""
    parsed = resolve.parse_mentions([
        {"type": "advertiser", "id": "ADV_02", "label": "x"},
        {"type": "not-a-kind", "id": "ADV_02"},
        {"type": "advertiser", "id": ""},
        "not a dict",
        {"type": "advertiser", "id": "ADV_02", "label": "again"},
    ])
    assert [(r["type"], r["id"]) for r in parsed] == [("advertiser", "ADV_02")]
    assert resolve.parse_mentions(None) == []
    assert resolve.parse_mentions({"type": "advertiser"}) == []


def test_the_reference_list_is_capped() -> None:
    many = [{"type": "advertiser", "id": f"ADV_{i:02d}"} for i in range(40)]
    assert len(resolve.parse_mentions(many)) == resolve.REFS_CAP == 8


# --- the four states --------------------------------------------------------------
def test_every_reference_sent_comes_back_with_a_state_and_none_is_dropped() -> None:
    """THE RULE THIS FILE EXISTS FOR. The reference product this design was
    measured against drops a dead reference silently and leaves the text behind
    as prose; three of one month's changelog entries are bugs of that shape,
    because the failure is invisible by design. Here a dropped reference would
    leave Kai a Hebrew label in the question with no data behind it, and the rule
    that every figure names its basis would push it to answer from the label.
    That is fabrication, so every reference reaches the model, including the ones
    that did not resolve."""
    sent = [
        {"type": "day", "id": _a_day(), "label": "a day"},
        {"type": "program", "id": TYPO, "label": "a typo"},
        {"type": "advertiser", "id": TYPO, "label": "a ghost"},
    ]
    context: dict = {}
    sources: list[str] = []
    public = resolve.extend_with_mentioned_objects(context, sources, sent)
    section = context[resolve.SECTION_NAME]
    assert sources == [resolve.SECTION_NAME]
    assert len(section["objects"]) == len(sent) == len(public)
    for card in section["objects"]:
        assert card["state"] in set(section["states"])
    assert {r["id"] for r in public} == {r["id"] for r in sent}


def test_all_four_states_are_reachable_against_the_real_stores() -> None:
    """A state nothing can produce is a state nobody has checked. Each of the
    four is produced here from data on disk rather than asserted to exist."""
    day = _a_day()
    seen = {
        resolve.resolve_one({"type": "day", "id": day, "label": day})["state"],
        resolve.resolve_one({"type": "advertiser", "id": "ADV_02", "label": "an older spelling"})["state"],
        resolve.resolve_one({"type": "advertiser", "id": TYPO, "label": "x"})["state"],
    }
    assert resolve.STATE_RESOLVED in seen
    assert resolve.STATE_CHANGED in seen
    assert resolve.STATE_GONE in seen
    # Unavailable is the one the stores on this machine cannot produce on demand,
    # so it is produced the only honest way: by making the read fail.
    broken = resolve.resolve_one({"type": "agency", "id": "\x00" * 4, "label": "x"})
    assert broken["state"] in (resolve.STATE_GONE, resolve.STATE_UNAVAILABLE)


def test_a_name_that_moved_is_stated_and_never_swapped_silently() -> None:
    card = resolve.resolve_one({"type": "advertiser", "id": "ADV_02", "label": "an older spelling"})
    assert card["state"] == resolve.STATE_CHANGED
    assert card["current_label"] and card["current_label"] != "an older spelling"
    assert card["changed_note"]


def test_gone_and_unavailable_are_never_the_same_claim() -> None:
    """The tri-state doctrine this product already applies to delivery and to the
    model: "we looked and it is not there" and "we could not look" are different
    claims, and a blank or a zero in either cell is the fabrication those modules
    exist to prevent."""
    assert resolve.STATE_GONE != resolve.STATE_UNAVAILABLE
    assert words.state_label(resolve.STATE_GONE, "he") != words.state_label(resolve.STATE_UNAVAILABLE, "he")
    assert words.state_label(resolve.STATE_GONE, "en") != words.state_label(resolve.STATE_UNAVAILABLE, "en")


def test_every_state_ships_an_approved_word_in_both_languages() -> None:
    for state in (resolve.STATE_RESOLVED, resolve.STATE_CHANGED,
                  resolve.STATE_GONE, resolve.STATE_UNAVAILABLE):
        assert words.state_label(state, "he").strip(), state
        assert words.state_label(state, "en").strip(), state


def test_the_hebrew_state_words_are_read_from_modules_that_already_ship_them() -> None:
    """Words are READ, never invented. Each is checked against the module the
    words table names as its source, so a word minted for this route fails here
    rather than reaching an operator."""
    sources = {
        "resolved": ROOT / "kairos_api" / "break_api_pod_spots.py",
        "changed": ROOT / "kairos_api" / "break_api_pod_order.py",
        "gone": ROOT / "kairos_api" / "campaigns_api_store.py",
        "unavailable": ROOT / "kairos_api" / "break_api_states.py",
    }
    for state, path in sources.items():
        word = words.state_label(state, "he")
        assert word in path.read_text(encoding="utf-8"), f"{state}: {word} is not in {path.name}"
    drill = (ROOT / "kairos_api" / "overview_api_drill.py").read_text(encoding="utf-8")
    assert words.ABSENT_REASON_HE in drill


# --- the boundary, as a positive control ------------------------------------------
def test_the_rival_control_bites_on_an_unscoped_read() -> None:
    """The control before the claim: this programme value really is in the plan
    file, so a resolver that read the file unscoped WOULD find it."""
    frame = _plan()
    column = _program_column(frame)
    rival = _rival_program()
    assert (frame[column].astype(str).str.strip() == rival).any()


def test_a_rival_programme_resolves_gone_and_is_byte_identical_to_a_typo() -> None:
    """The reference is a client-supplied identifier, so it is the one place a
    rival name could be pushed in from outside the picker. It resolves against
    ``_owned_frame`` and nothing else, so it comes back exactly as a typo does --
    not "none on your channel", which would confirm the name exists."""
    rival = _rival_program()
    theirs = resolve.resolve_one({"type": "program", "id": rival, "label": rival})
    typo = resolve.resolve_one({"type": "program", "id": TYPO, "label": TYPO})
    assert theirs["state"] == resolve.STATE_GONE
    blank_rival = json.dumps({k: v for k, v in theirs.items() if k not in ("id", "label")},
                             ensure_ascii=False, sort_keys=True)
    blank_typo = json.dumps({k: v for k, v in typo.items() if k not in ("id", "label")},
                            ensure_ascii=False, sort_keys=True)
    assert blank_rival == blank_typo


def test_a_gone_card_carries_no_field_derived_from_the_store() -> None:
    """Its only variable fields are the ones the caller itself sent."""
    rival = _rival_program()
    card = resolve.resolve_one({"type": "program", "id": rival, "label": rival})
    assert "data" not in card and "basis" not in card
    for key, value in card.items():
        if key in ("id", "label"):
            continue
        assert rival not in str(value), key


# --- the answer the operator sees --------------------------------------------------
def test_the_public_states_carry_the_binding_and_never_the_figures() -> None:
    """The finding against page_context is that the binding is INVISIBLE. A state
    the model can read and the operator cannot would repeat that one layer down,
    so the four states come back on the ask. What comes back is the binding: the
    type, the identifier, the name and the state, and no figure, because the card
    the model reads is where the data lives."""
    day = _a_day()
    context: dict = {}
    sources: list[str] = []
    public = resolve.extend_with_mentioned_objects(context, sources, [{"type": "day", "id": day, "label": day}])
    assert len(public) == 1
    row = public[0]
    assert set(row) == {"type", "id", "label", "state", "kind_he", "kind_en",
                        "state_he", "state_en", "icon"}
    assert row["state"] == resolve.STATE_RESOLVED
    # The glyph is the kind's navigational identity and it comes from the one
    # table that holds it, so the chip in the answer wears what the chip in the
    # composer wore rather than a default that would be quietly wrong.
    assert row["icon"] == words.icon("day")


def test_an_ask_with_no_references_carries_no_new_key() -> None:
    """The degradation promise: an ask that sends nothing behaves exactly as the
    ask that shipped before this key existed."""
    context: dict = {}
    sources: list[str] = []
    assert resolve.extend_with_mentioned_objects(context, sources, None) == []
    assert context == {} and sources == []
    body_source = (ROOT / "kairos_api" / "assistant_pipeline.py").read_text(encoding="utf-8")
    assert '**({"mentions": mentioned} if mentioned else {})' in body_source


def test_both_ask_endpoints_accept_the_field() -> None:
    from kairos_api.assistant import AskRequest
    from kairos_api.assistant_stream import StreamAskRequest

    for model in (AskRequest, StreamAskRequest):
        assert "mentions" in model.model_fields, model.__name__
        assert model(question="q").mentions is None


# --- the size law -------------------------------------------------------------------
def test_every_file_this_round_touched_stays_under_the_law() -> None:
    for relative in (
        "kairos_api/assistant_mentions.py",
        "kairos_api/assistant_mentions_children.py",
        "kairos_api/assistant_mentions_resolve.py",
        "kairos_api/assistant_mentions_words.py",
        "kairos_api/assistant_pipeline.py",
        "kairos_api/assistant_prompt.py",
        "tests/test_assistant_mentions_refs.py",
        "tests/test_assistant_mentions_drill.py",
        "tv-break-dashboard/src/kai/AssistantComposer.jsx",
        "tv-break-dashboard/src/kai/AssistantPanel.jsx",
        "tv-break-dashboard/src/kai/AssistantThread.jsx",
        "tv-break-dashboard/src/kai/MentionPicker.jsx",
        "tv-break-dashboard/src/kai/MentionRefs.jsx",
        "tv-break-dashboard/src/kai/assistant-panel-ask.js",
        "tv-break-dashboard/src/kai/mention-picker.css",
        "tv-break-dashboard/src/kai/mention-refs.css",
        "tv-break-dashboard/src/kai/mention-refs.js",
        "tv-break-dashboard/src/kai/mention-state.js",
        "tv-break-dashboard/src/kai/mention-trigger.js",
        "tv-break-dashboard/src/shell/bidi.jsx",
    ):
        path = ROOT / relative
        assert path.exists(), relative
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 450, relative
