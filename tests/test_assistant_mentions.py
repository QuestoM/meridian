"""The composer's mention index, against the real stores on disk.

Nothing is mocked. The index is built from the advertiser rules store, the
agencies store, the calendar events store and the saved weekly plan exactly as
the route builds it, and the boundary assertions are made about that payload
rather than about a fixture.

THE BOUNDARY TEST IS WRITTEN AS A POSITIVE CONTROL, which is the only kind
worth having. A scan that has never flagged anything has not been shown to work,
so the same scan is first pointed at an UNSCOPED read of the saved plan -- the
thing every operator surface is forbidden to serve -- and is required to find
rivals there. Only then is it pointed at the route, where it must find none.

The saved plan on disk carries four channels, and two programme values,
``Children`` and ``Religious``, exist on rival channels and on no row of the
operator's own. They are the sharpest available control: a value that could only
have come from the market read.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kairos_api import assistant_mentions as mentions
from kairos_api import assistant_mentions_words as words
from kairos_api import channel_scope

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"
TYPO = "qqqzzzxxx-no-such-thing"


def _plan() -> pd.DataFrame:
    if not PLAN_PATH.exists():
        pytest.skip("no saved weekly plan on disk, so there is no index to scope")
    return pd.read_csv(PLAN_PATH)


def _program_column(frame: pd.DataFrame) -> str:
    return "program_title" if "program_title" in frame.columns else "program_type"


def _rival_names() -> set[str]:
    """Every name in the saved plan that belongs to somebody else: the rival
    channels themselves, and every programme value that appears on a rival row
    and on no row of the operator's own."""
    frame = _plan()
    owned = channel_scope.operator_channel()
    channels = {str(value).strip() for value in frame["channel"].dropna().unique()}
    rivals = {name for name in channels if name and name != owned}
    column = _program_column(frame)
    ours = {str(value).strip() for value in frame.loc[frame["channel"] == owned, column]}
    theirs = {str(value).strip() for value in frame.loc[frame["channel"] != owned, column]}
    return rivals | {name for name in (theirs - ours) if name and name.lower() != "nan"}


def _names_found(payload: object, names: set[str]) -> set[str]:
    """Which of these names appear anywhere in a payload, serialized as the
    browser would receive it. Substring, not equality: a rival hidden inside a
    label or a parent path is still a rival that reached the operator."""
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    return {name for name in names if name in blob}


def _unscoped_program_rows() -> list[dict[str, object]]:
    """The index this route would have had without the scope: programme rows
    over the WHOLE saved plan, every channel included. This exists to be caught."""
    frame = _plan()
    column = _program_column(frame)
    rows = []
    for (channel, title), group in frame.groupby(["channel", column], sort=False):
        rows.append(
            {
                "type": "program",
                "id": str(title),
                "label": str(title),
                "parent": [{"kind": "name", "text": str(channel)}, {"kind": "figure",
                                                                    "text": str(len(group))}],
            }
        )
    return rows


# --- the shape ---------------------------------------------------------------------
def test_the_route_is_mounted_under_the_assistant() -> None:
    from kairos_api import assistant

    paths = {route.path for route in assistant.router.routes}
    assert "/api/assistant/mentions" in paths


def test_every_kind_ships_an_approved_word_in_both_languages_and_an_icon() -> None:
    for kind in words.KIND_NAMES:
        assert words.label(kind, "he").strip(), kind
        assert words.label(kind, "en").strip(), kind
        assert words.icon(kind).strip(), kind
        assert words.rank(kind) < 999, kind


def test_the_gold_break_spelling_rule_is_not_violated_by_any_word() -> None:
    # The campaign's own example of a word that must be read and never invented.
    blob = json.dumps(words.KINDS, ensure_ascii=False)
    assert "זהוב" not in blob


def test_a_row_carries_the_kind_the_icon_and_a_parent_path() -> None:
    payload = mentions.search("", limit=mentions.SHOW_CAP)
    assert payload["rows"], "the index is empty, so nothing below can be shown to work"
    for row in payload["rows"]:
        assert row["type"] in words.KINDS
        assert row["label"].strip()
        assert row["icon"] == words.icon(row["type"])
        assert isinstance(row["parent"], list)
        # No internal fold key ever reaches the browser.
        assert not [key for key in row if key.startswith("_")]


# --- the caps ----------------------------------------------------------------------
def test_the_default_cap_is_the_eight_rows_a_popup_shows() -> None:
    payload = mentions.search("")
    assert payload["limit"] == mentions.SHOW_CAP == 8
    assert len(payload["rows"]) <= mentions.SHOW_CAP


def test_the_omitted_count_is_taken_after_scoping_and_after_matching() -> None:
    """Raising the cap must move rows from omitted into rows, one for one.

    That identity can only hold if omitted counts matches that survived the
    scope. A count taken over the unscoped index would not move in step with the
    cap, because the rows it counted were never eligible to be shown.
    """
    small = mentions.search("", limit=1)
    large = mentions.search("", limit=mentions.FETCH_CAP)
    assert small["count"] + small["omitted"] == large["count"] + large["omitted"]
    assert large["count"] >= small["count"]


def test_a_query_that_matches_nothing_omits_nothing() -> None:
    payload = mentions.search(TYPO)
    assert payload["rows"] == []
    assert payload["omitted"] == 0


# --- the boundary, with its control ------------------------------------------------
def test_the_boundary_scan_bites_on_an_unscoped_index() -> None:
    """The positive control. Same scan, same names, pointed at the index this
    route would have had if the scope were removed. It must find rivals there or
    the assertion below proves nothing at all."""
    rivals = _rival_names()
    assert rivals, "the saved plan has no rival rows, so there is nothing to control for"
    found = _names_found(_unscoped_program_rows(), rivals)
    assert found, "the scan failed to flag an unscoped plan read"


def test_no_rival_name_reaches_the_mention_index() -> None:
    rivals = _rival_names()
    assert _names_found(mentions.build_index(), rivals) == set()


def test_no_rival_name_reaches_the_route_for_any_query() -> None:
    rivals = _rival_names()
    for name in sorted(rivals):
        payload = mentions.search(name, limit=mentions.FETCH_CAP)
        assert _names_found(payload, rivals) == set(), name
        assert payload["rows"] == [], name


def test_a_rival_name_and_a_typo_are_byte_identical() -> None:
    """Boundary rule two, in its strongest form. 'None on your channel' would
    confirm the name exists, so the two answers may not differ at all -- and
    they do not, because the route does not echo the query back."""
    typo = mentions.search(TYPO)
    for name in sorted(_rival_names()):
        assert mentions.search(name) == typo, name


def test_the_scope_block_names_the_operators_own_channel_only() -> None:
    payload = mentions.search("")
    assert payload["scope"]["scope_channel"] == (channel_scope.operator_channel() or None)
    # A count of what the scope removed is a fact about rivals and is not served.
    assert set(payload["scope"]) == {"scope_channel", "scoped"}


# --- Hebrew ------------------------------------------------------------------------
def test_the_hebrew_stripper_is_the_existing_one_and_there_is_no_second_copy() -> None:
    from kairos_api import assistant_context

    source = Path(mentions.__file__).read_text(encoding="utf-8")
    assert "_strip_hebrew_prefixes" in source
    # The prefix alphabet appears in exactly one module, and it is not this one.
    assert "ובלמהשכ" in Path(assistant_context.__file__).read_text(encoding="utf-8")
    assert "ובלמהשכ" not in source


def test_a_one_letter_prefix_on_either_side_still_matches() -> None:
    """בחדשות must find חדשות, which is the requirement neither reference
    product has, and it must do it through the EXISTING stripper.

    The candidate is chosen with a first word longer than three characters
    because that is the shared stripper's own rule, and the rule is deliberate:
    it will not strip a prefix off a three-letter word, since בעד and בית are
    real words whose first letter is not a prefix. Measured on the stores here,
    that limit is real: the event ``עד 120 מרכזי מגורים`` is NOT reachable as
    ``בעד 120``. Widening it belongs to assistant_context, not to a second copy
    of the stripper made here to dodge the case.
    """
    index = mentions.build_index()
    hebrew = [row["label"] for row in index
              if row["label"][:1] in "אבגדהוזחטיכלמנסעפצקרשת" and len(row["label"].split()[0]) > 3]
    if not hebrew:
        pytest.skip("no Hebrew label long enough for the shared stripper's own rule")
    label = max(hebrew, key=len)
    plain = {row["label"] for row in mentions.search(label, limit=mentions.FETCH_CAP)["rows"]}
    prefixed = {row["label"] for row in mentions.search("ב" + label, limit=mentions.FETCH_CAP)["rows"]}
    assert label in plain
    assert label in prefixed


# --- the accelerator is not a gate --------------------------------------------------
def test_the_free_text_paths_this_piece_must_not_touch_still_work() -> None:
    """The measurement this whole piece turns on: a mention system that is the
    only way to name a thing goes unused. These are the ways of asking that
    existed before it, and every one of them must still answer."""
    from kairos_api.assistant_context import _question_dates, _strip_hebrew_prefixes
    from kairos_api.assistant_read_tools import execute_read_tool

    assert _question_dates("what happened on 01/11/2024", ["2024-11-01"]) == ["2024-11-01"]
    assert _strip_hebrew_prefixes("בחדשות") == "חדשות"
    advertisers = [row for row in mentions.build_index() if row["type"] == "advertiser"]
    if advertisers:
        found = execute_read_tool("find_advertiser", {"name": advertisers[0]["label"]}, None)
        assert found["candidates"], "find_advertiser stopped resolving a name it used to resolve"


def test_the_label_the_picker_inserts_is_the_store_s_own_identifier_or_name() -> None:
    """R1's whole contract with the model: what lands in the question is a
    string the existing read tools already resolve, so no new resolution path is
    needed for a mention to be useful."""
    from kairos_api.advertisers import _load_frame, _row_to_record

    names = set()
    for _, raw in _load_frame().iterrows():
        record = _row_to_record(raw)
        names.add(str(record.get("display_name") or record.get("name") or "").strip())
        names.add(str(record.get("advertiser_id") or "").strip())
    for row in mentions.build_index():
        if row["type"] == "advertiser":
            assert row["label"] in names, row["label"]


# --- the size law -------------------------------------------------------------------
def test_every_file_this_piece_added_is_under_the_size_law() -> None:
    for relative in (
        "kairos_api/assistant_mentions.py",
        "kairos_api/assistant_mentions_words.py",
        "tests/test_assistant_mentions.py",
        "tv-break-dashboard/src/kai/MentionPicker.jsx",
        "tv-break-dashboard/src/kai/mention-picker.css",
        "tv-break-dashboard/src/kai/AssistantComposer.jsx",
        "tv-break-dashboard/src/kai/assistant-console.css",
        "tv-break-dashboard/src/kai/AssistantPanel.jsx",
    ):
        path = ROOT / relative
        assert path.exists(), relative
        assert len(path.read_text(encoding="utf-8").splitlines()) <= 450, relative
