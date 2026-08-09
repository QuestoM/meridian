"""Descending a container, and the offsets that keep a chip attached.

The part both reference products verifiably declined to build. A flat fuzzy
search substitutes for navigation exactly when every leaf has a unique typeable
path; one product kills chaining with a trailing space and the other collapses
the directory kind into the file kind, and both are right for what they hold.
This product's leaves are not like that: a spot has no name, a break is a clock
reading that recurs, and a programme value repeats every weekday. So something
has to be enterable.

Nothing here is mocked. The descent runs against the saved weekly plan on disk,
and the offset algebra is DRIVEN in node against the shipped module rather than
described, which is the pattern mention-trigger and kai-keep-warm already set.

THE BOUNDARY IS SHARPER ONE LEVEL DOWN THAN IT IS IN THE SEARCH. Descending into
an object that is not ours must answer exactly what descending into an object
that does not exist answers, so the route echoes nothing at all and every empty
descent returns the same bytes. That is asserted here as byte equality rather
than as an absence of the obvious fields.

WHAT THIS DRILL ACTUALLY SHOWS TODAY, said plainly rather than papered over:
there are no programme TITLES in this data. The saved plan has no
``program_title`` column and the kind is a genre. The ladder is right and the
rungs read as genres until titles land.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from kairos_api import assistant_mentions as mentions
from kairos_api import assistant_mentions_children as children
from kairos_api import assistant_mentions_words as words
from kairos_api import channel_scope

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"
KAI = ROOT / "tv-break-dashboard" / "src" / "kai"
TYPO = "qqqzzzxxx-no-such-thing"


def _plan() -> pd.DataFrame:
    if not PLAN_PATH.exists():
        pytest.skip("no saved weekly plan on disk, so there is nothing to descend into")
    return pd.read_csv(PLAN_PATH)


def _program_column(frame: pd.DataFrame) -> str:
    return "program_title" if "program_title" in frame.columns else "program_type"


def _rival_program() -> str:
    """A programme value that appears on a rival channel and on no row of the
    operator's own. Skips rather than passing vacuously when the plan stops
    carrying one, because a control that cannot fire proves nothing."""
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


def test_the_rival_control_bites_on_an_unscoped_read() -> None:
    """The control before the claim: this programme value really is in the plan
    file, so a descent that read the file unscoped WOULD surface it."""
    frame = _plan()
    assert (frame[_program_column(frame)].astype(str).str.strip() == _rival_program()).any()


def test_descending_a_day_returns_its_own_programmes_only() -> None:
    payload = children.children(type="day", id=_a_day())
    assert payload["rows"], "the day has no programmes, so nothing below is shown to work"
    frame = _plan()
    owned = channel_scope.operator_channel()
    column = _program_column(frame)
    ours = {str(v).strip() for v in frame.loc[frame["channel"] == owned, column]}
    for row in payload["rows"]:
        assert row["type"] == "program"
        assert row["label"] in ours


def test_no_rival_name_reaches_any_descent() -> None:
    frame = _plan()
    owned = channel_scope.operator_channel()
    column = _program_column(frame)
    ours = {str(v).strip() for v in frame.loc[frame["channel"] == owned, column]}
    theirs = {str(v).strip() for v in frame.loc[frame["channel"] != owned, column]}
    rivals = {str(v).strip() for v in frame["channel"].dropna().unique() if str(v).strip() != owned}
    rivals |= {name for name in (theirs - ours) if name and name.lower() != "nan"}
    assert rivals, "no rivals in the plan, so this scan cannot be shown to work"
    for day in [row["id"] for row in mentions.search("", types="day", limit=8)["rows"]]:
        blob = json.dumps(children.children(type="day", id=day), ensure_ascii=False)
        for name in rivals:
            assert name not in blob, f"{name} reached the descent of {day}"


def test_the_omitted_count_of_a_descent_is_taken_after_scoping() -> None:
    """An omitted count computed before scoping IS a rival count. This one is a
    count of the operator's own rows that lost to the cap and nothing else."""
    day = _a_day()
    frame = _plan()
    owned = channel_scope.operator_channel()
    column = _program_column(frame)
    ours = frame[(frame["channel"] == owned) & (frame["date"].astype(str) == day)]
    total = len({str(v).strip() for v in ours[column] if str(v).strip() and str(v).strip().lower() != "nan"})
    payload = children.children(type="day", id=day)
    assert payload["count"] + payload["omitted"] == total


def test_every_empty_descent_returns_the_same_bytes() -> None:
    """Descending into a rival's object, into one deleted this morning and into
    an outright typo must be indistinguishable, so the stated absence names the
    EDGE that was asked for and never the container that was asked about."""
    rival = _rival_program()
    answers = {
        json.dumps(children.children(type="program", id=rival, edge="day"), ensure_ascii=False),
        json.dumps(children.children(type="program", id=TYPO, edge="day"), ensure_ascii=False),
        json.dumps(children.children(type="program", id="", edge="day"), ensure_ascii=False),
    }
    assert len(answers) == 1, answers


def test_an_empty_descent_states_the_absence_in_both_languages() -> None:
    """Never a bare empty list: an empty list in a picker reads as "zero of
    them", and "there are none" and "we hold nothing of this kind here" are
    different claims."""
    payload = children.children(type="program", id=TYPO, edge="day")
    assert payload["rows"] == []
    assert payload["absent"]["reason"].strip()
    assert payload["absent"]["reason_he"].strip()


def test_no_edge_names_a_kind_whose_store_cannot_be_scoped() -> None:
    """Boundary rule three, one level down. Every kind on either end of an edge
    is a kind the search itself already serves, and the search's own test proves
    each of those is scopable."""
    for parent, kinds in words.EDGES.items():
        assert parent in words.KINDS, parent
        for child in kinds:
            assert child in words.KINDS, child
            assert (parent, child) in children._EDGE_BUILDERS, (parent, child)


def test_a_container_row_says_so_and_a_leaf_does_not() -> None:
    rows = mentions.search("", limit=mentions.SHOW_CAP)["rows"]
    assert rows
    for row in rows:
        assert row["container"] is words.is_container(row["type"])


# --- the offsets, driven in node against the shipped module -------------------------
REFS = KAI / "mention-refs.js"
TRIGGER = KAI / "mention-trigger.js"

HARNESS = """
import { writeFileSync } from 'node:fs';
import { addRef, chipRuns, edgeKeys, liveRefs, shiftRefs } from './mention-refs.js';
import { insertMention, readMentionQuery } from './mention-trigger.js';

const out = {};

// Which arrow goes IN. In a right-to-left document the leading edge is the
// right, so descending is ArrowLeft; in a left-to-right one it is ArrowRight.
out.rtlKeys = edgeKeys('rtl');
out.ltrKeys = edgeKeys('ltr');

// Insertion reports the span it created, which is what a typed {type, id} binds
// to. The mid-line case is the one that catches an off-by-one.
const run = readMentionQuery('what about @co', 14);
out.inserted = insertMention('what about @co', run, 'Coca-Cola');
const mid = readMentionQuery('ask @co about it', 6);
out.insertedMid = insertMention('ask @co about it', mid, 'אסם');

// Carrying a span across an edit.
const text = 'about Coca-Cola today';
const refs = [{ start: 6, len: 9, type: 'advertiser', id: 'ADV_02', label: 'Coca-Cola' }];
// Typing BEFORE the span moves it.
out.shiftedAfter = shiftRefs(text, 'well about Coca-Cola today', refs);
// Typing AFTER the span leaves it alone.
out.shiftedBefore = shiftRefs(text, 'about Coca-Cola today please', refs);
// Editing INSIDE the span drops it: the sentence now names something else.
out.droppedInside = shiftRefs(text, 'about Coca-Kola today', refs);
// Deleting the whole span drops it too.
out.droppedWhole = shiftRefs(text, 'about  today', refs);

// The runs the highlight paints, and the gate in front of them.
out.runs = chipRuns(text, refs);
out.staleRuns = chipRuns('about Pepsi today', refs);
out.live = liveRefs(text, refs).length;
out.staleLive = liveRefs('about Pepsi today', refs).length;

writeFileSync(process.argv[2], JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def driven(tmp_path_factory) -> dict:
    """Run the SHIPPED modules in node, unmodified. Both have no imports for
    exactly this reason: nothing is rewritten on the way into the temp directory,
    so what is proved is their own behaviour rather than a description of it."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on this machine")
    for path in (REFS, TRIGGER):
        if not path.exists():
            pytest.skip(f"{path.name} is not in this tree")
    work = tmp_path_factory.mktemp("mention-refs")
    for path in (REFS, TRIGGER):
        (work / path.name).write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    (work / "harness.mjs").write_text(HARNESS, encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run([node, str(work / "harness.mjs"), str(out)],
                            capture_output=True, text=True, check=False, cwd=str(work))
    if result.returncode != 0:
        pytest.fail(f"the shipped modules did not run: {result.stderr.strip()[:600]}")
    return json.loads(out.read_text(encoding="utf-8"))


def test_the_leading_edge_descends_and_hebrew_is_the_left_arrow(driven: dict) -> None:
    """Never hardcoded, and this is the assertion that says why it matters: get
    it backwards and the drill is on the key that ascends, which makes the whole
    ladder unreachable in the language the product is written in."""
    assert driven["rtlKeys"] == {"descend": "ArrowLeft", "ascend": "ArrowRight"}
    assert driven["ltrKeys"] == {"descend": "ArrowRight", "ascend": "ArrowLeft"}
    source = (KAI / "mention-state.js").read_text(encoding="utf-8")
    assert "edgeKeys(documentDirection(locale))" in source


def test_insertion_reports_the_span_the_reference_binds_to(driven: dict) -> None:
    assert driven["inserted"]["start"] == len("what about ")
    assert driven["inserted"]["len"] == len("Coca-Cola")
    assert driven["insertedMid"]["start"] == len("ask ")
    assert driven["insertedMid"]["len"] == len("אסם")


def test_an_edit_around_a_reference_carries_it_and_an_edit_inside_drops_it(driven: dict) -> None:
    """The whole judgement of this module. An operator who edits the letters of
    an inserted name is changing what the sentence says, and a binding kept
    across that would send one object's identifier while the prose named
    another."""
    assert driven["shiftedAfter"][0]["start"] == 11
    assert driven["shiftedBefore"][0]["start"] == 6
    assert driven["droppedInside"] == []
    assert driven["droppedWhole"] == []


def test_the_highlight_never_paints_over_the_wrong_words(driven: dict) -> None:
    marked = [run for run in driven["runs"] if run["chip"]]
    assert [run["text"] for run in marked] == ["Coca-Cola"]
    assert "".join(run["text"] for run in driven["runs"]) == "about Coca-Cola today"
    # A span whose characters are no longer its label paints nothing.
    assert not [run for run in driven["staleRuns"] if run["chip"]]
    assert driven["live"] == 1 and driven["staleLive"] == 0
