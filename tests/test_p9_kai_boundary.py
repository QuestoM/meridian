"""P9: the two lines Kai may never cross, asserted rather than described.

The competitor boundary and the model-disclosure wall are both Kai's duty under
sections 7.2 and 8.3 of the rebuild specification, and neither had a test that
ran against a real account on the wrong side of a line. These do.

Nothing here is mocked away: the context is composed from the real saved plan,
the read tools run their real executors, and the channel account is a real
record in a real auth store, so a wall that only works with authentication
disabled fails here.
"""

from __future__ import annotations

import json

import pytest

from kairos_api import (
    assistant_keywords,
    assistant_model_disclosure as disclosure,
    assistant_sections,
    assistant_tools,
    auth_store,
)
from kairos_api.assistant_read_tools import execute_read_tool

OPERATOR_CHANNEL = "רשת 13"
RIVAL_CHANNELS = ("קשת 12", "כאן 11", "עכשיו 14")


@pytest.fixture()
def channel_account(tmp_path, monkeypatch):
    """A real, stored, channel-affiliated account, with auth enforced."""
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    auth_store.seed_initial_admin(password="rootpass-1234")
    auth_store.add_user("channel_person", "channelpass-123", role="operator",
                        affiliation="channel")
    auth_store.add_user("company_person", "companypass-123", role="operator",
                        affiliation="company")
    assert auth_store.is_company_user("channel_person") is False
    assert auth_store.is_company_user("company_person") is True
    yield "channel_person"
    auth_store.reset_runtime_state()


# --- the propose-only contract, the Bar 3 floor for this piece ---------------
def test_the_tool_surface_is_still_40_read_9_propose_and_0_write() -> None:
    # 38 when the read half of the coverage gap closed: the pod the traffic
    # operator assembles (2 tools), the break the scheduler places (2), and
    # pacing against goal with the remedy the campaign manager is told to name
    # (3). Then 40 and 9, closing the half that was left. Kai could READ a
    # problem it could not ACT on, so propose_pacing_decision records the two
    # endings a board row has and the ledger moves after them. The two reads are
    # estimate_restriction_cost, which prices a rule about somebody else's
    # revenue before anyone writes it, and get_accounts, the
    # account-administrator persona, which also states the four licence limits
    # the propose path already refused to move and could not name.
    #
    # There is deliberately no tool for an account change: creating one,
    # resetting a password and moving an affiliation are credential acts, and a
    # review-first assistant that could stage them would be staging a way in.
    assert len(assistant_tools.READ_TOOL_NAMES) == 40
    assert len(assistant_tools.PROPOSE_TOOL_NAMES) == 9
    every_name = assistant_tools.READ_TOOL_NAMES | assistant_tools.PROPOSE_TOOL_NAMES
    assert len(assistant_tools.anthropic_tools()) == len(every_name)
    # No tool applies anything. The apply engine is the only writer, and it is
    # reached by an approval, never by the model.
    assert not any(name.startswith("apply_") for name in every_name)


def test_no_tool_can_reach_a_training_artifact() -> None:
    """The training test of section 4.1: an act is training if and only if its
    output lands under models/. Not one tool writes there, and the read tools
    that read there are the walled three."""
    reading_models = {"get_audience_model", "get_audience_stability", "get_event_pipeline"}
    assert reading_models <= assistant_tools.READ_TOOL_NAMES
    assert disclosure.WALLED_READ_TOOLS == reading_models
    assert not (assistant_tools.PROPOSE_TOOL_NAMES & reading_models)


# --- the competitor boundary -------------------------------------------------
@pytest.fixture()
def owned_channel(monkeypatch):
    """The operator's channel, declared by this test rather than borrowed.

    ``operator_channel`` lives in a settings document the running product
    rewrites, so a test that reads whatever is on disk asserts the boundary only
    for as long as nobody edits the channel picker. This fixture pins it to the
    channel the reference plan is built around, so the assertion below is about
    the scoping seam and never about the state of a shared file.
    """
    from kairos_api import core

    settings = core._load_settings()
    monkeypatch.setattr(core, "_load_settings",
                        lambda: settings.model_copy(update={"operator_channel": OPERATOR_CHANNEL}))
    return OPERATOR_CHANNEL


def test_the_counts_section_is_scoped_to_the_operators_own_channel(owned_channel) -> None:
    """Measured before this fix: counts reported 8,704 segments and 9,026 breaks
    across all four channels while the overview beside it reported the operator's
    own 2,391. The two figures now come from the same channel."""
    from kairos_api import channel_scope, server

    counts = assistant_sections._section_counts()
    assert counts["scope_channel"] == owned_channel, (
        "the counts section must disclose the scope it counted")
    overview = assistant_sections._section_overview_summary()
    assert counts["breaks"] == overview["total_breaks"], (
        "the context must not carry two different break counts on two different scopes")

    # And the rows really are dropped at the shared boundary seam, which is
    # where the count of what went belongs. It is deliberately not in the
    # context: how many rival rows exist is itself a fact about rivals.
    _scoped, note = channel_scope.scope_frame(server._load_break_schedule())
    assert note["scoped"] is True
    assert note["competitor_rows_excluded"] > 0, "the reference plan carries rival rows to exclude"
    assert note["competitor_channels_excluded"] == 3


def test_no_rival_channel_reaches_the_composed_context(owned_channel) -> None:
    context, _sources = assistant_sections.compose_context("מה מצב התוכנית השבוע")
    blob = json.dumps(context, ensure_ascii=False, default=str)
    for rival in RIVAL_CHANNELS:
        assert rival not in blob, f"competitor channel {rival} leaked into Kai's context"


# --- the model-disclosure wall ----------------------------------------------
@pytest.mark.parametrize("tool", sorted(disclosure.WALLED_READ_TOOLS))
def test_a_channel_account_gets_no_model_internals_from_a_walled_tool(channel_account, tool) -> None:
    payload = execute_read_tool(tool, {}, channel_account)
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    assert disclosure.internals_hits(blob) == [], (
        f"{tool} carried model internals to a channel account: {disclosure.internals_hits(blob)}")


@pytest.mark.parametrize("tool", ["get_audience_model", "get_audience_stability"])
def test_a_wholly_replaced_tool_carries_not_one_word_of_the_critics_lexicon(channel_account, tool) -> None:
    """The two tools whose entire payload is training content are replaced, not
    edited, so they pass even the blunt grep of section 4.2 including the word
    coefficient. The event pipeline is walled surgically and keeps its run-side
    stages, which is why it is measured against the internals list instead."""
    payload = execute_read_tool(tool, {}, channel_account)
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    assert disclosure.lexicon_hits(blob) == [], disclosure.lexicon_hits(blob)


def test_the_word_coefficients_survives_only_where_the_operator_already_reads_it(
        channel_account, monkeypatch) -> None:
    """The one word the wall deliberately does not strip, and exactly where.

    The plan-freshness verdict names coefficients as a changed input group, and
    the operator's own staleness banner already prints that group on every page
    (shell/ScheduleStalenessBanner.jsx:39). Stripping it here would make Kai
    disagree with the banner beside it and would regress P1's Bar 3 floor.

    The verdict is pinned rather than read off the deployed plan, because which
    inputs are currently stale is state any run changes, and the contract under
    test is what the wall does with the word rather than whether today's plan
    happens to carry it.
    """
    import kairos.export.schedule_freshness as freshness

    monkeypatch.setattr(freshness, "schedule_freshness", lambda root: {
        "status": "stale", "computed_at": "2026-07-28T08:38:38.170135+00:00",
        "changed": ["settings", "coefficients"]})

    payload = execute_read_tool("get_event_pipeline", {}, channel_account)
    carriers = [key for key, value in payload.items()
                if "coefficient" in json.dumps(value, ensure_ascii=False, default=str).lower()]
    assert carriers == ["pricing_layer", "freshness"], carriers
    assert "coefficients" in payload["freshness"]["changed_groups"]
    # And a negative statement, which discloses nothing about the model.
    assert "untouched" in payload["pricing_layer"]["basis"]


def test_a_company_account_still_sees_the_measured_gates(channel_account) -> None:
    """The wall closes one side only. A company account keeps the full payload,
    which is what makes the refusal a boundary rather than a feature removal."""
    payload = execute_read_tool("get_audience_model", {}, "company_person")
    assert payload.get("gates"), "the company side lost the gate verdicts"
    assert disclosure.lexicon_hits(json.dumps(payload, ensure_ascii=False)), (
        "the company payload should still carry the measured vocabulary")


def test_a_walled_tool_names_what_it_withheld_and_offers_the_version(channel_account) -> None:
    payload = execute_read_tool("get_audience_model", {}, channel_account)
    assert payload["available"] is False
    assert payload["reason"] == disclosure.WITHHELD_REASON
    assert payload["model_version"]["state"] in {"real", "unknown"}
    assert payload["release_note"]["state"] in {"real", "unavailable", "unknown"}
    if payload["release_note"]["state"] != "real":
        assert payload["release_note"]["supplied_by"] == disclosure.RELEASE_NOTE_PATH


def test_the_event_pipeline_keeps_every_run_side_stage_for_a_channel_account(channel_account) -> None:
    """The wall is surgical: the events store, the asserted pricing layer and
    plan freshness are run-side facts a channel account owns, and only the
    measured verdict is replaced."""
    payload = execute_read_tool("get_event_pipeline", {}, channel_account)
    assert "events_store" in payload and "pricing_layer" in payload and "freshness" in payload
    assert "training_gate" not in payload
    assert len(payload["operational_order"]) == 4
    assert payload["operational_order"][3] == disclosure.EVENT_STAGE_FOUR_OPERATOR_HE


# --- the grounding section discovery found missing ---------------------------
@pytest.mark.parametrize("question", ["למה השתנו המקדמים", "did the coefficients change",
                                      "מה מצב המודל", "has the model drifted"])
def test_a_question_about_the_model_now_grounds_on_the_model_state(question) -> None:
    context: dict = {}
    sources: list[str] = []
    assistant_keywords.extend_with_keyword_sections(context, sources, question)
    assert "model_state" in context, f"{question!r} still grounds on nothing"
    assert context["model_state"]["model_version"]["state"] in {"real", "unknown"}


def test_the_model_state_section_is_walled_for_a_channel_account(channel_account) -> None:
    company: dict = {}
    channel: dict = {}
    assistant_keywords.extend_with_keyword_sections(company, [], "למה השתנו המקדמים", "company_person")
    assistant_keywords.extend_with_keyword_sections(channel, [], "למה השתנו המקדמים", channel_account)
    assert company["model_state"].get("coefficients"), "the company side lost the measured state"
    assert "coefficients" not in channel["model_state"]
    assert disclosure.internals_hits(json.dumps(channel["model_state"], ensure_ascii=False)) == []


def test_an_unmatched_question_still_adds_no_section() -> None:
    context: dict = {}
    assistant_keywords.extend_with_keyword_sections(context, [], "כמה ברייקים יש ביום ראשון")
    assert "model_state" not in context
