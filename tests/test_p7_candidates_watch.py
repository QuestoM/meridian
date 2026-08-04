"""The candidate shelf follows the measurement it is showing, in a real browser.

**The defect this file exists to keep dead.** A blind measurement of the shipped
tree: the steward opened the shelf with two money measurements in flight, the
page issued one read of ``/api/model/candidates`` plus one per candidate at open
and never asked again. The store recorded a finished measurement 75.2 s later,
and 60.1 s after that, with zero presses, the screen still carried two blocks
reading "the plan is being computed twice". Leaving the section and returning
read the truth, which proved the store was right and only the screen was wrong.
This is the money story's own screen, so a false state on it is a false state
about shekels.

**Why this is a browser test.** A measurement ends on the server with no press
behind it, so the whole defect lives in the gap between two reads that nobody
triggers. Nothing about the source of a component can show whether the screen
learns about an event that happens while it is idle. So the real bridge is built
with the product's own bundler, run in a real headless Chrome, handed the bodies
the route really serves, and asked what is on screen after doing nothing at all.

**What is real here and what is a stand-in, stated plainly.** The four bodies are
real and they come from one real money measurement: the candidate is measured
through ``model_console_api.measure_candidate``, the product's own write, into a
temporary releases store seeded with the four other real measurement records,
and ``candidate_list()`` is taken before it, while it runs and after it ends. The
measurement computes the whole weekly plan twice on the real data, which took
179.0 s on this tree while the rest of the wave was building and 85.4 to 87.2 s
on the quiet tree that produced the four seeded records. So the browser is handed
the product's own figures for a measurement that really happened, including the
shekels it really moved on the operator's own channel.

The one stand-in is the per-candidate read: every card is answered with the
subject's own ``/api/model/candidates/{id}`` body. That read carries only the
verdict and the model version, both of which are empty in a fresh store, so every
card renders the same honest "no verdict recorded" line it would render anyway.
The shell around the console is the same stand-in the other three browser
measurements use, for the same reason and with the same premise asserted
separately from source.

Two companion modules carry what would put this file over the 450-line law and
neither holds a test: ``test_p7_candidates_watch_page.py`` is the page the
browser runs, and ``test_p7_candidates_watch_read.py`` is what this file reads
out of the product and off the screen.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from test_p7_candidates_watch_page import (
    DETAIL_PREFIX,
    FINISH_BUDGET_MS,
    LIST_ROUTE,
    SUBJECT,
    page_script,
)
from test_p7_candidates_watch_read import (
    SHIPPED_COEFFICIENTS,
    SHIPPED_MEASUREMENTS,
    SUBJECT_ARTIFACT,
    card as _card,
    digits as _digits,
    js_number as _js_number,
    row as _row,
    sha256 as _sha256,
    shekels as _shekels,
    thread_failures as _thread_failures,
    word as _word,
)
from test_p7_console_bridge_harness import (
    BRIDGE,
    build_harness,
    run_scenario,
    skip_unless_a_real_browser_is_available,
)

# How long a real measurement gets before this test gives up on it. Measured:
# 179.0 s on this tree, 85.4 to 87.2 s on the quiet one.
MEASUREMENT_BUDGET_S = 600


@pytest.fixture(scope="module")
def bodies(tmp_path_factory) -> "dict":
    """One real money measurement, and the shelf payloads around it."""
    from kairos_api import model_console_api as console

    store_dir = tmp_path_factory.mktemp("releases")
    seeded = json.loads(SHIPPED_MEASUREMENTS.read_text(encoding="utf-8"))
    seeded.pop(SUBJECT, None)
    (store_dir / "candidate_measurements.json").write_text(
        json.dumps(seeded, ensure_ascii=False), encoding="utf-8")

    with pytest.MonkeyPatch.context() as patch, _thread_failures() as failures:
        patch.setenv("KAIROS_MODEL_RELEASES_DIR", str(store_dir))
        untouched_before = {path: _sha256(path) for path in
                            (SHIPPED_MEASUREMENTS, SHIPPED_COEFFICIENTS, SUBJECT_ARTIFACT)}
        before = console.candidate_list()
        started = console.measure_candidate(SUBJECT)
        measuring = console.candidate_list()
        deadline = time.time() + MEASUREMENT_BUDGET_S
        finished = measuring
        while time.time() < deadline:
            finished = console.candidate_list()
            if (_row(finished).get("money") or {}).get("state") != "measuring":
                break
            time.sleep(1.0)
        yield {
            "before": before,
            "measuring": measuring,
            "finished": finished,
            "started": started,
            "detail": console.candidate_detail(SUBJECT),
            "seeded": seeded,
            "failures": list(failures),
            "untouched_before": untouched_before,
            "untouched_after": {path: _sha256(path) for path in untouched_before},
        }


@pytest.fixture(scope="module")
def scenario(tmp_path_factory, bodies) -> "dict":
    """The browser, handed those bodies in the order the route would serve them.

    The measuring body is served twice on purpose. One repeat proves the shelf
    keeps asking rather than getting lucky with a single retry, and it makes the
    measurement independent of exactly which read follows the write.
    """
    skip_unless_a_real_browser_is_available()
    state = (_row(bodies["finished"]).get("money") or {}).get("state")
    if state == "measuring":
        pytest.fail(f"the measurement never ended within {MEASUREMENT_BUDGET_S}s")
    work = tmp_path_factory.mktemp("p7-candidates")
    dist = build_harness(work, page_script(os.path.relpath(BRIDGE, work.resolve() / "src")))
    sequence = [bodies["before"], bodies["measuring"], bodies["measuring"], bodies["finished"]]
    return run_scenario(dist, work, {DETAIL_PREFIX: bodies["detail"], LIST_ROUTE: sequence},
                        {DETAIL_PREFIX: bodies["started"]})


@pytest.fixture(scope="module")
def unpressed(tmp_path_factory, bodies) -> "dict":
    """The same shelf, opened on a measurement this browser never started.

    This is the entry the defect was measured on. A measurement can be started
    from another window, or from the route, or by a steward who then leaves and
    comes back, so the shelf is often opened onto one already in flight with no
    press behind it at all. The same page runs with its press turned off and the
    same real bodies behind it, so the only thing that differs is the entry.
    """
    skip_unless_a_real_browser_is_available()
    if (_row(bodies["finished"]).get("money") or {}).get("state") == "measuring":
        pytest.fail(f"the measurement never ended within {MEASUREMENT_BUDGET_S}s")
    work = tmp_path_factory.mktemp("p7-candidates-open")
    dist = build_harness(work, page_script(
        os.path.relpath(BRIDGE, work.resolve() / "src"), press=False))
    sequence = [bodies["measuring"], bodies["measuring"], bodies["finished"]]
    return run_scenario(dist, work, {DETAIL_PREFIX: bodies["detail"], LIST_ROUTE: sequence})


# ---------------------------------------------------------------------------
# The measurement the browser was handed, before anything about the screen
# ---------------------------------------------------------------------------


def test_the_measurement_the_browser_was_handed_really_ran_and_really_ended(bodies) -> None:
    """The premise. Without it the measurement below would be about a fixture."""
    assert (_row(bodies["before"]).get("money") or {})["state"] == "not_measured", (
        "the subject was already measured, so its measurement was never watched starting"
    )
    running = _row(bodies["measuring"])["money"]
    assert running["state"] == "measuring", (
        f"the shelf never reported the measurement in flight: {running['state']}"
    )
    assert bodies["started"]["state"] == "measuring", bodies["started"]
    assert running["past_durations_seconds"] == sorted(
        record["duration_seconds"] for record in bodies["seeded"].values()
    ), "the measuring block quotes durations that are not the store's own"

    money = _row(bodies["finished"])["money"]
    assert money["state"] in {"measured", "stale"}, (
        f"the measurement ended in {money['state']} and the thread that ran it"
        f" reported: {bodies['failures'] or 'nothing at all'}"
    )
    assert money["duration_seconds"] > 0, "the record carries no duration, so nothing was run"
    delta = (money.get("operator_channel_delta") or {}).get("revenue_delta")
    assert isinstance(delta, (int, float)), f"the measurement produced no shekel figure: {delta}"
    assert money["scope"]["operator_channel"]["rows"] > 0, "the figure was measured on no rows"


def test_the_measurement_wrote_nothing_it_reads(bodies) -> None:
    """The measurement's own safety claim, re-measured on a real run.

    ``build_plan_totals`` says it writes no output, no artifact and no version.
    The three files it reads are hashed either way, because a run that failed
    could still have written something on its way there.
    """
    for path, digest in bodies["untouched_before"].items():
        assert digest != "absent", f"{path} is missing, so this test would prove nothing"
        assert bodies["untouched_after"][path] == digest, (
            f"{path} moved while a candidate money measurement ran"
        )


# ---------------------------------------------------------------------------
# The screen, in a real browser
# ---------------------------------------------------------------------------


def test_the_candidates_section_opened_on_its_own_route_with_nothing_measuring(scenario) -> None:
    """The premise of every assertion below: the right panel, in its rest state."""
    assert "failed" not in scenario, scenario.get("failed")
    phase = scenario["phases"]["opened"]
    assert phase["route"] == LIST_ROUTE, (
        f"the rail position opened {phase['route']} instead of the candidate shelf"
    )
    assert len(phase["cards"]) == 5, f"the shelf rendered {len(phase['cards'])} cards"
    assert [card["measuring"] for card in phase["cards"]] == [False] * 5
    assert phase["watch"] == "", "the shelf says it is following a measurement before one started"
    assert _card(phase)["controls"] == [_word("candidates.measure")], (
        f"the unmeasured candidate offers {_card(phase)['controls']}"
    )


def test_pressing_measure_puts_that_candidate_on_the_screen_as_measuring(scenario, bodies) -> None:
    """One card, in the measuring state, in the product's own words for it."""
    phase = scenario["phases"]["started"]
    assert phase["appeared"] is True, "the started measurement never appeared on the screen"
    measuring = [card["id"] for card in phase["cards"] if card["measuring"]]
    assert measuring == [SUBJECT], f"{measuring} read as measuring for one measurement"
    card = _card(phase)
    assert _word("candidates.measuring") in card["measuringWords"], (
        f"the measuring card reads {card['measuringWords']}"
    )
    past = _row(bodies["measuring"])["money"]["past_durations_seconds"]
    assert _digits(card["pastRuns"]) == _digits(_js_number(past[len(past) // 2])), (
        f"the past-measurement line reads {card['pastRuns']} against {past}"
    )
    assert phase["watch"] == _word("candidates.watching"), (
        f"the shelf does not say it is following the measurement: {phase['watch']}"
    )


def test_the_shelf_learns_the_measurement_ended_with_no_press_at_all(scenario, bodies) -> None:
    """The defect, dead.

    Between the press that started the measurement and this snapshot the
    scenario presses nothing, so every change on the screen came from the shelf
    asking again. The figure asserted is the one the real measurement produced,
    to the shekel, not a figure chosen here.
    """
    started = scenario["phases"]["started"]
    finished = scenario["phases"]["finished"]
    assert finished["reached"] is True, (
        f"the screen still read {_card(started)['measuringWords']} after {finished['waitedMs']} ms"
    )
    assert finished["clicks"] == started["clicks"], (
        f"{finished['clicks'] - started['clicks']} presses happened during the measurement"
    )
    assert finished["waitedMs"] <= FINISH_BUDGET_MS
    assert finished["reads"] > started["reads"], (
        "the screen changed without the route being read, which cannot be true"
    )
    money = _row(bodies["finished"])["money"]
    card = _card(finished)
    assert card["measuring"] is False
    assert card["stale"] is (money["state"] == "stale"), (
        f"the card reads stale={card['stale']} for a measurement the store calls {money['state']}"
    )
    assert card["block"] is True, "the ended measurement shows no money block at all"
    assert "₪" in card["owned"], f"the money figure carries no currency: {card['owned']}"
    assert _digits(card["owned"]) == _shekels(money["operator_channel_delta"]["revenue_delta"]), (
        f"the screen reads {card['owned']} for a measured"
        f" {money['operator_channel_delta']['revenue_delta']}"
    )
    assert finished["watch"] == "", (
        f"the shelf still says it is following a measurement that ended: {finished['watch']}"
    )


def test_the_ended_card_carries_the_scope_its_figure_was_measured_on(scenario, bodies) -> None:
    """Stripe's rule on the screen the money lands on: the figure or its basis, never one alone."""
    money = _row(bodies["finished"])["money"]
    if money["state"] != "measured":
        pytest.skip(f"an input moved while the measurement ran, so it settled {money['state']}")
    card = _card(scenario["phases"]["finished"])
    assert card["moneyRows"] == 2, (
        f"the measured card shows {card['moneyRows']} scopes rather than the owned channel and the plan"
    )
    assert _digits(card["scope"]) == str(money["scope"]["operator_channel"]["rows"]), (
        f"the first scope line reads {card['scope']} against"
        f" {money['scope']['operator_channel']['rows']} rows"
    )


def test_the_shelf_stops_reading_the_route_once_no_measurement_is_open(scenario) -> None:
    """A watch that never ends is a second defect wearing the first one's fix."""
    finished = scenario["phases"]["finished"]
    settled = scenario["phases"]["settled"]
    assert settled["reads"] == finished["reads"], (
        f"the shelf read the route {settled['reads'] - finished['reads']} more times"
        f" in {settled['afterMs']} ms with no measurement open"
    )
    assert settled["clicks"] == finished["clicks"]
    assert settled["cards"] == finished["cards"], "the screen moved after the measurement had ended"


def test_a_shelf_opened_onto_a_measurement_it_never_started_says_it_is_following_it(
        unpressed) -> None:
    """The entry with no press in it, which is the one the defect was found on."""
    assert "failed" not in unpressed, unpressed.get("failed")
    opened = unpressed["phases"]["opened"]
    measuring = [card["id"] for card in opened["cards"] if card["measuring"]]
    assert measuring == [SUBJECT], f"{measuring} read as measuring at open"
    assert opened["watch"] == _word("candidates.watching"), (
        f"the shelf does not say it is following the measurement: {opened['watch']}"
    )


def test_that_shelf_learns_the_measurement_ended_without_being_touched_at_all(
        unpressed, bodies) -> None:
    """Nothing on this screen was ever pressed, and it still tells the truth.

    The only presses in the whole run open the console and the section. From
    there the browser is left alone, so the money that appears is money the
    shelf went and asked for.
    """
    opened = unpressed["phases"]["opened"]
    finished = unpressed["phases"]["finished"]
    settled = unpressed["phases"]["settled"]
    assert finished["reached"] is True, (
        f"the screen still read measuring after {finished['waitedMs']} ms with nothing pressed"
    )
    assert finished["clicks"] == opened["clicks"], (
        f"{finished['clicks'] - opened['clicks']} presses happened while the shelf was watched"
    )
    money = _row(bodies["finished"])["money"]
    card = _card(finished)
    assert card["measuring"] is False
    assert _digits(card["owned"]) == _shekels(money["operator_channel_delta"]["revenue_delta"]), (
        f"the screen reads {card['owned']} for a measured"
        f" {money['operator_channel_delta']['revenue_delta']}"
    )
    assert finished["watch"] == "", f"the shelf still says it is following: {finished['watch']}"
    assert settled["reads"] == finished["reads"], (
        f"the shelf read the route {settled['reads'] - finished['reads']} more times"
        f" in {settled['afterMs']} ms with no measurement open"
    )


def test_the_watch_moved_only_the_card_whose_measurement_ended(scenario) -> None:
    """The other four carry real money of their own and none of it may be disturbed."""
    opened = {card["id"]: card for card in scenario["phases"]["opened"]["cards"]}
    finished = {card["id"]: card for card in scenario["phases"]["finished"]["cards"]}
    assert set(opened) == set(finished), "the shelf gained or lost a candidate while watching"
    moved = sorted(name for name in opened if opened[name] != finished[name])
    assert moved == [SUBJECT], f"the watch also moved {sorted(set(moved) - {SUBJECT})}"


def test_the_browser_and_the_server_agree_on_how_many_calls_there_were(scenario) -> None:
    """Two independent counts of the same thing, because one of them is the page's."""
    settled = scenario["phases"]["settled"]
    served = scenario["server_reads"]
    assert served[f"POST {DETAIL_PREFIX}"] == 1, (
        f"the scenario started {served[f'POST {DETAIL_PREFIX}']} measurements"
    )
    assert served[LIST_ROUTE] == settled["reads"], (
        f"the server answered {served[LIST_ROUTE]} shelf reads and the browser"
        f" recorded {settled['reads']}"
    )
    assert served[LIST_ROUTE] >= 4, (
        f"the shelf was read {served[LIST_ROUTE]} times, which is fewer than opening the section,"
        " the read after the write and two watch reads"
    )


def test_no_rival_channel_name_reaches_the_shelf_a_person_reads(scenario) -> None:
    """The boundary, asserted on the rendered text rather than on the payload.

    The payload is pinned separately. This asks the other half of the question:
    the shelf prints a whole-plan figure, and a whole-plan figure is exactly the
    place a channel name could arrive on a screen the payload test never sees.
    """
    from kairos_api import model_console_artifacts as artifacts

    base = (artifacts.read_artifact(artifacts.AUDIENCE_ARTIFACT) or {}).get("base") or {}
    owned = str(base.get("owned_channel") or "")
    rivals = sorted(name for name in (base.get("hist_channel") or {}) if name != owned)
    assert owned and rivals, "the artifact holds no rival channels, so this would prove nothing"
    rendered = scenario["phases"]["settled"]["consoleText"]
    assert rendered, "the console rendered no text at all, so this would prove nothing"
    for rival in rivals:
        assert rival not in rendered, f"the shelf shows the rival channel {rival}"
