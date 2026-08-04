"""P5: the composer offers every night, so JS-4's own trigger can be said.

The measured defect. The night picker rendered ``airings.slice(0, 12)`` and
nothing else: no search, no date field, no disclosure and no count of what was
hidden. On the reference plan for ``משחקי השף עונה 7 ש.ח`` the endpoint returns
43 airings on 19 distinct nights, the head line said 43, and the twelve chips
under it covered six of those nights because two airings of one programme on one
night consume two slots. Thirteen nights were unreachable, the two Sunday airings
sat at positions 24 and 25, and the last night of the run, 2024-11-30, could not
be named at all. Across the 40 titles the operator's channel carries, 26 run more
than twelve airings and 1,896 airings sat behind the cap. The story this surface
exists for is "the season finale airs Sunday and the last eight minutes must stay
clean", and a finale is the last airing.

The unit was wrong as well as the count. A restriction scoped to a date compiles
to ``date is <day>``, so two airings on one night were always one choice rendered
twice. Deduplicating by night is what makes rendering all of them small: a
broadcast month bounds the list whatever the programme, and the busiest title on
the channel is 1,551 airings on 5 nights.

So this file bundles the shipped picker with the bundler the product builds
with, renders it against the real payload, and counts the nights a person can
actually click. The last test puts the cap back into the source and asserts the
truncation returns, so a pass here can never be vacuous.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "tv-break-dashboard"
NIGHTS = APP / "src" / "rules" / "AiringNights.jsx"
COMPOSER = APP / "src" / "rules" / "RestrictionComposer.jsx"
PROBE = Path(__file__).with_name("test_p5_composer_probe.mjs")

CHANNEL = "רשת 13"
# The programme the critic measured. Chosen because it runs far past twelve
# airings and its nights include a Sunday, which is the story's own night.
PROGRAMME = "משחקי השף עונה 7 ש.ח"

# The line the mutant replaces, asserted present before it is cut so the
# mutation can never be a silent no-op.
WHOLE_LIST = "{list.map((night) => ("
CAPPED_LIST = "{list.slice(0, 12).map((night) => ("

CHIP = re.compile(r'<button(?P<attrs>[^>]*class="rules-airing-chip[^"]*"[^>]*)>(?P<body>.*?)</button>', re.S)
ATTR = re.compile(r'(?P<name>[a-z-]+)="(?P<value>[^"]*)"')
TEXT = re.compile(r"<[^>]+>")


def _node() -> str:
    found = shutil.which("node")
    if not found:
        pytest.skip("node is not on PATH, so the shipped component cannot be rendered here")
    probe = subprocess.run(
        [found, "-e", "const m = require('node:module'); process.stdout.write(typeof m.registerHooks)"],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.stdout.strip() != "function":
        pytest.skip("this node has no module.registerHooks, so react cannot be resolved from the app")
    if not (APP / "node_modules" / "react-dom").is_dir():
        pytest.skip("the dashboard's node_modules is not installed, so nothing can be rendered")
    if not (APP / "node_modules" / "rolldown").is_dir():
        pytest.skip("the bundler the product builds with is not installed")
    if not PROBE.exists():
        pytest.skip("the render probe is missing")
    return found


@pytest.fixture(scope="module")
def payload() -> dict:
    """The airings payload the product serves for the programme under test."""
    from kairos_api.constraints import router
    from kairos_api.core import _load_settings

    if str(getattr(_load_settings(), "operator_channel", "")).strip() != CHANNEL:
        pytest.skip(f"the declared operator channel is not {CHANNEL}, so this programme is out of scope")
    app = FastAPI()
    app.include_router(router)
    response = TestClient(app).get(
        "/api/constraints/restrictions/airings", params={"title": PROGRAMME}
    )
    assert response.status_code == 200, response.text
    body = response.json()
    if not body["count"]:
        pytest.skip(f"'{PROGRAMME}' is not in the plan window as it stands, so there is nothing to render")
    return body


def _render(tmp_path: Path, body: dict, source: str, dead_end: dict | None = None) -> dict:
    node = _node()
    work = tmp_path / "surface"
    work.mkdir(parents=True, exist_ok=True)
    under_test = work / "nights-under-test.jsx"
    under_test.write_text(source, encoding="utf-8")
    airings_file = work / "airings.json"
    airings_file.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    out = work / "out.json"
    command = [node, str(PROBE), str(work / "bundle"), str(out), str(under_test), str(airings_file)]
    if dead_end is not None:
        dead_end_file = work / "dead-end.json"
        dead_end_file.write_text(json.dumps(dead_end, ensure_ascii=False), encoding="utf-8")
        command.append(str(dead_end_file))
    result = subprocess.run(command, capture_output=True, text=True, check=False, cwd=str(work))
    assert result.returncode == 0, result.stderr[-2000:]
    return json.loads(out.read_text(encoding="utf-8"))


def _chips(rendered: dict, locale: str, key: str = "nights") -> list[dict]:
    """Every chip the picker rendered, in the order a person meets them."""
    html = rendered["picked"] if key == "picked" else rendered[key][locale]
    out = []
    for match in CHIP.finditer(html):
        row = {name: value for name, value in ATTR.findall(match.group("attrs"))}
        row["label"] = row.get("aria-label", "")
        row["html"] = match.group("body")
        row["text"] = TEXT.sub(" ", match.group("body")).strip()
        out.append(row)
    return out


def _nights(rendered: dict, locale: str) -> list[dict]:
    """The night chips, which is every chip after the leading all-airings one."""
    return _chips(rendered, locale)[1:]


@pytest.fixture(scope="module")
def dead_end(payload) -> dict:
    """A night the plan already keeps clean, and the same rule over the whole run.

    Derived rather than fixed. A window rule compiles one store row per airing
    that breaches it, so a night the plan already keeps clean compiles nothing
    and the save has nothing to write. Which nights those are is a fact about
    the plan of the day, so the composer's own draft is priced night by night
    until one of them lands in that state.
    """
    from kairos_api.constraints import router

    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    def price(day: str) -> dict:
        conditions = [{"field": "programme", "operator": "is", "value": PROGRAMME}]
        if day:
            conditions.append({"field": "date", "operator": "is", "value": day})
        response = client.post(
            "/api/constraints/restrictions/preview",
            json={
                "kind": "clean_tail",
                "params": {"protected_minutes": 8},
                "where": {"combinator": "and", "conditions": conditions},
            },
        )
        assert response.status_code == 200, response.text
        return response.json()

    wider = price("")
    if not wider["compiled_rows"]:
        pytest.skip("no airing of this programme breaches an eight minute tail as the plan stands")
    for night in payload["nights"]:
        priced = price(night["day"])
        if not priced["compiled_rows"]:
            return {"night": night["day"], "wider": wider, "night_preview": priced}
    pytest.skip("every night of this programme breaches this rule as the plan stands")
    return {}


@pytest.fixture(scope="module")
def shipped(tmp_path_factory, payload, dead_end) -> dict:
    source = NIGHTS.read_text(encoding="utf-8")
    assert WHOLE_LIST in source, "the render under test is not in the shipped component any more"
    return _render(tmp_path_factory.mktemp("shipped"), payload, source, dead_end=dead_end)


def test_the_payload_this_renders_is_the_case_that_could_not_be_said(payload):
    """The shape of the defect, taken from the running product rather than fixed."""
    days = sorted({row["day"] for row in payload["airings"]})
    assert payload["count"] > 12, "this programme has to run past the old cap or it proves nothing"
    assert len(payload["nights"]) == payload["night_count"]
    assert [row["day"] for row in payload["nights"]] == days, (
        "the night list has to be exactly the distinct dates of the airings, in order"
    )
    assert len(days) < payload["count"], "and some night has to carry more than one airing"
    assert sum(row["airings"] for row in payload["nights"]) == payload["count"], (
        "every airing has to be accounted for by exactly one night"
    )


def test_every_night_the_programme_runs_is_selectable(shipped, payload):
    """The bar: the number in the head line is the number of choices below it."""
    days = [row["day"] for row in payload["nights"]]
    for locale in ("he", "en"):
        chips = _nights(shipped, locale)
        assert [chip["label"].split(",")[1].strip() for chip in chips] == days, (
            "every night, in plan order, and nothing invented or dropped"
        )
        assert len(chips) == payload["night_count"]


def test_the_last_night_of_the_run_can_be_named(shipped, payload):
    """A season finale is the last airing, which the twelve-chip cap could not reach."""
    last = payload["nights"][-1]["day"]
    for locale in ("he", "en"):
        labels = [chip["label"] for chip in _nights(shipped, locale)]
        assert any(last in label for label in labels), f"the last night {last} has to be selectable"
    assert shipped["picked_day"] == last
    picked = [row for row in _chips(shipped, "he", "picked") if "active" in row["class"]]
    assert len(picked) == 1, "choosing a night selects exactly one chip"
    assert last in picked[0]["label"], "and it is the night that was chosen"


def test_the_head_line_states_both_counts_and_they_are_the_real_ones(shipped, payload):
    head = re.search(r'<span class="rules-airings-head">(.*?)</span>', shipped["nights"]["en"], re.S)
    assert head, "the picker has to say what it is offering"
    words = TEXT.sub(" ", head.group(1))
    assert f"{payload['count']} airings" in words
    assert f"{payload['night_count']} nights" in words
    hebrew = TEXT.sub(" ", re.search(
        r'<span class="rules-airings-head">(.*?)</span>', shipped["nights"]["he"], re.S,
    ).group(1))
    assert f"{payload['count']} שידורים" in hebrew and f"{payload['night_count']} לילות" in hebrew


def test_a_night_with_two_airings_is_one_choice_and_says_so(shipped, payload):
    """Two airings on one night compile to one predicate, so they are one chip."""
    doubled = [row for row in payload["nights"] if row["airings"] > 1]
    if not doubled:
        pytest.skip("no night of this programme carries more than one airing in the plan as it stands")
    night = doubled[0]
    chips = {chip["label"].split(",")[1].strip(): chip for chip in _nights(shipped, "en")}
    assert night["day"] in chips
    assert f"{night['airings']} airings" in chips[night["day"]]["text"], (
        "a night that holds more than one airing has to say how many"
    )


def test_the_week_reads_sunday_first_in_both_locales(shipped, payload):
    """The Israeli week, on the story's own night. ISO on the record, read back local."""
    from datetime import date

    sundays = [row["day"] for row in payload["nights"] if date.fromisoformat(row["day"]).weekday() == 6]
    if not sundays:
        pytest.skip("this programme runs on no Sunday in the plan as it stands")
    for locale, word in (("he", "ראשון"), ("en", "Sunday")):
        chips = {chip["label"].split(",")[1].strip(): chip["label"] for chip in _nights(shipped, locale)}
        for day in sundays:
            assert chips[day].startswith(word), f"{day} is a Sunday and has to read as one in {locale}"


def test_no_rival_channel_name_reaches_the_picker(shipped, payload):
    rivals = ["כאן 11", "קשת 12", "עכשיו 14"]
    assert payload["channel"] == CHANNEL
    for locale in ("he", "en"):
        for name in rivals:
            assert name not in shipped["nights"][locale]
            assert name not in shipped["composer"][locale]


def test_no_engine_word_reaches_the_path_a_representative_walks(shipped):
    """JS-4's own bar: zero engine words anywhere on the path.

    Measured on the visible text rather than on the markup, deliberately, and
    the distinction is the point. The kind control's option values are still
    ``clean_tail`` and ``clean_open``, because that is what the store holds and
    what a save has to post; what a person reads is the sentence frame in their
    own language. A raw key rendered as text is the defect, and this catches it.
    """
    words = [
        "fix_offset", "pin_count", "offset_seconds", "clean_tail", "clean_open",
        "segment_id", "where_json", "predicate", "combinator", "planned_breaks",
    ]
    for locale in ("he", "en"):
        for surface in ("nights", "composer"):
            visible = TEXT.sub(" ", shipped[surface][locale])
            found = [word for word in words if word in visible]
            assert not found, f"{surface} in {locale} reads {found}"
    assert 'value="clean_tail"' in shipped["composer"]["he"], (
        "and the control still posts the key the store holds, or this passed by losing the value"
    )


def test_the_picker_reads_in_the_direction_the_page_is_written_in(shipped):
    """The chip's detail is a Latin-digit run inside Hebrew, so it is isolated."""
    chips = _nights(shipped, "he")
    assert chips, "there is nothing to check the direction of"
    for chip in chips:
        assert '<small dir="ltr">' in chip["html"], chip["html"]
    # And the state a screen reader is given is the pressed one, not a colour.
    assert all("aria-pressed" in chip for chip in _chips(shipped, "he"))


def test_the_composer_hands_the_whole_night_list_over(shipped):
    """The picker cannot render what the composer never gives it."""
    source = COMPOSER.read_text(encoding="utf-8")
    assert "nights={airings.nights}" in source, "the composer has to pass the whole list"
    assert ".slice(0, 12)" not in source, "the cap is gone from the composer"
    for locale in ("he", "en"):
        assert "rules-composer" in shipped["composer"][locale], "and the composer still renders"


def test_a_night_the_plan_keeps_clean_says_so_and_offers_the_rule_that_does_bind(shipped, dead_end):
    """Naming a night is only half the story if naming it shuts the save.

    A window rule derives one store row per airing that breaches it, so a night
    the plan already keeps clean compiles nothing and there is nothing to store.
    Measured on the plan as it stands: 7 of the 19 nights of this programme
    breach an eight minute tail, so the other 12 end on a shut button. The note
    states what that night does, states what the whole run does, and offers the
    rule that can be written, all from the preview route's own figures.
    """
    night = dead_end["night"]
    assert dead_end["night_preview"]["compiled_rows"] == 0, "the night under test has to be the dead end"
    breaching = dead_end["wider"]["compiled_rows"]
    matched = dead_end["wider"]["matched_airings"]
    assert 0 < breaching < matched, "and the whole run has to breach it, or there is no way out to offer"
    for locale in ("he", "en"):
        html = shipped["widen"][locale]
        words = TEXT.sub(" ", html)
        assert night in words, f"the note has to name the night that was chosen, {night}"
        assert str(breaching) in words and str(matched) in words, words
        assert "rules-widen-action" in html, "and it has to offer the wider rule as an act"
    assert "שום דבר בחלון התוכנית" not in shipped["widen"]["he"], (
        "with a night named, the note is about that night and not about the window"
    )


def test_the_composer_routes_its_empty_state_through_that_note(shipped):
    """The note cannot help anybody if the composer still prints the old line."""
    source = COMPOSER.read_text(encoding="utf-8")
    assert "<WiderScopeNote" in source
    assert "Nothing in the plan window breaks this rule" not in source, (
        "the whole-window sentence moved into the note, which says it only when no night is named"
    )
    for locale in ("he", "en"):
        assert shipped["widen"][locale], "and the note renders in both languages"


def test_with_the_cap_back_the_same_picker_hides_the_last_night(tmp_path, payload):
    """The mutant, which is exactly what a critic measured on the shipped bundle."""
    source = NIGHTS.read_text(encoding="utf-8")
    mutant = source.replace(WHOLE_LIST, CAPPED_LIST)
    assert mutant != source
    rendered = _render(tmp_path, payload, mutant)
    chips = _nights(rendered, "he")
    assert len(chips) == 12, "the mutant is the twelve-chip cap"
    last = payload["nights"][-1]["day"]
    assert not any(last in chip["label"] for chip in chips), (
        f"under the cap the last night {last} is unreachable, which is the defect"
    )
    head = TEXT.sub(" ", re.search(
        r'<span class="rules-airings-head">(.*?)</span>', rendered["nights"]["he"], re.S,
    ).group(1))
    assert f"{payload['night_count']} לילות" in head, (
        "while the head line three centimetres above went on promising every night, which is the defect"
    )
