"""P9: the first token, and the ask that does not have to be asked twice.

Two measured defects from round five, both named by a blind critic and both
about seconds the operator spends for nothing.

**The first token.** "First token on an action ask was 2,855 ms against the 2 s
budget in job-stories.md." Round five wrote the model's cached prefix on the
warm call the dock already makes at mount (kairos_api/assistant_warm.py), which
covers the dock that is opened and used. It does not cover the dock that is
opened and left open, and that is the ordinary case, because Kai is docked
beside the work: the server holds its prefix record for 240 s and the question
comes minutes later. Measured here on 2026-08-05 as a controlled pair, the same
request sent twice, cold then warm, four times: cold 2.478, 1.634, 4.472 and
2.718 s to the first token against warm 1.624, 2.907, 1.938 and 2.288 s, with
the API's own usage record showing about 16,740 tokens written on the first of
each pair and read on the second. So the composer now says when a question is
being written and the prefix is asked for then.

**The second ask.** Measured on this machine before rule 30 existed, six asks
through the dock's own streaming endpoint: four were direct instructions naming
a field and a value, and only three of the six imperatives measured that day
ended with a proposal recorded. The others simulated, reported the numbers and
asked whether to record a proposal, which makes the person approve the same
change twice and costs a whole second round trip, measured by the critic at
25.8 s. JS-10 budgets 45 s from question to applied change. After rule 30 the
same six asks recorded 4 of 4 imperatives, and the two that correctly recorded
nothing were a question and a value the saved state already held.

The node harness drives the shipped keep-warm module itself, with a clock and a
warm call this file owns, so what is proved is the module's own behaviour rather
than a description of it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from kairos_api import assistant_prompt

ROOT = Path(__file__).resolve().parents[1]
KAI = ROOT / "tv-break-dashboard" / "src" / "kai"
KEEP_WARM = KAI / "kai-keep-warm.js"
COMPOSER = KAI / "AssistantComposer.jsx"
PANEL = KAI / "AssistantPanel.jsx"

# What node needs and the bundler does not: the shipped module imports
# './assistant-stream' with no extension, which vite resolves and node does not.
# The one specifier is rewritten on the way into the temp directory and nothing
# else about the file is touched, so the code under test is the shipped code.
STUB = """
export const calls = [];
export function warmContext(signal) {
  const entry = { signal, settled: false };
  calls.push(entry);
  return new Promise((resolve) => {
    entry.finish = () => { entry.settled = true; resolve({ state: 'warming' }); };
  });
}
"""

HARNESS = """
import { writeFileSync } from 'node:fs';
import { calls } from './assistant-stream.js';
import { keepPrefixWarm, resetPrefixWarm, MIN_INTERVAL_MS } from './kai-keep-warm.js';

let clock = 1000000;
Date.now = () => clock;

const out = {};

// One request, and a second call while it is open joins it rather than opening
// another: a keystroke is not a reason to write the same prefix twice.
const first = keepPrefixWarm();
const joined = keepPrefixWarm();
out.afterTwoCalls = calls.length;
out.joinedTheSamePromise = first === joined;
calls[0].finish();
await first;

// Settled, but inside the interval, so nothing is asked for and the caller is
// told so rather than handed a promise that resolves to nothing.
out.throttledReturn = keepPrefixWarm();
out.afterThrottled = calls.length;

// Past the interval, a question being typed asks again.
clock += MIN_INTERVAL_MS + 1;
const third = keepPrefixWarm();
out.afterInterval = calls.length;
calls[calls.length - 1].finish();
await third;

// An aborted request never reached the server, so it must not spend the
// interval. Unmounting the dock aborts on purpose.
resetPrefixWarm();
const controller = new AbortController();
const aborted = keepPrefixWarm(controller.signal);
controller.abort();
calls[calls.length - 1].finish();
await aborted;
const before = calls.length;
const retried = keepPrefixWarm();
out.retriedAfterAbort = calls.length === before + 1;
calls[calls.length - 1].finish();
await retried;

out.interval = MIN_INTERVAL_MS;
out.total = calls.length;
writeFileSync(process.argv[2], JSON.stringify(out));
"""


def _node() -> str:
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not on this machine")
    return node


@pytest.fixture(scope="module")
def driven(tmp_path_factory) -> dict:
    if not KEEP_WARM.exists():
        pytest.skip("kai-keep-warm.js is not in this tree")
    work = tmp_path_factory.mktemp("keep-warm")
    source = KEEP_WARM.read_text(encoding="utf-8")
    assert "from './assistant-stream'" in source, "the import this harness rewrites moved"
    (work / "kai-keep-warm.js").write_text(
        source.replace("from './assistant-stream'", "from './assistant-stream.js'"),
        encoding="utf-8")
    (work / "assistant-stream.js").write_text(STUB, encoding="utf-8")
    (work / "harness.mjs").write_text(HARNESS, encoding="utf-8")
    out = work / "out.json"
    result = subprocess.run([_node(), str(work / "harness.mjs"), str(out)],
                            capture_output=True, text=True, check=False, cwd=str(work))
    if result.returncode != 0:
        pytest.fail(f"the shipped keep-warm module did not run: {result.stderr.strip()[:600]}")
    return json.loads(out.read_text(encoding="utf-8"))


# --- the module, driven ------------------------------------------------------

def test_a_burst_of_keystrokes_opens_one_request_and_not_one_each(driven: dict) -> None:
    assert driven["afterTwoCalls"] == 1
    assert driven["joinedTheSamePromise"] is True


def test_inside_the_interval_nothing_is_asked_for_and_the_caller_is_told(driven: dict) -> None:
    """Null rather than a promise, so a caller can tell a skipped call from a
    real one instead of awaiting something that never happened."""
    assert driven["throttledReturn"] is None
    assert driven["afterThrottled"] == 1


def test_past_the_interval_a_question_being_typed_asks_again(driven: dict) -> None:
    assert driven["afterInterval"] == 2


def test_the_interval_sits_inside_the_hold_the_server_keeps(driven: dict) -> None:
    """240 s on the server (kairos_api/assistant_warm.py PREFIX_TTL_SECONDS). A
    client interval at or above it would let the prefix lapse between two
    keystrokes, which is the whole failure this module exists to prevent."""
    from kairos_api import assistant_warm

    assert driven["interval"] < assistant_warm.PREFIX_TTL_SECONDS * 1000


def test_an_aborted_request_does_not_spend_the_interval(driven: dict) -> None:
    """It never reached the server, so it warmed nothing. A dock opened, closed
    and opened again would otherwise go unwarmed for two minutes."""
    assert driven["retriedAfterAbort"] is True


# --- the surface that calls it ------------------------------------------------

def test_the_composer_reports_a_question_being_written(driven: dict) -> None:
    """The one thing only the composer can know: the panel cannot see a cursor
    land in the box. Focus and every keystroke, because the module throttles."""
    text = COMPOSER.read_text(encoding="utf-8")
    assert "onFocus={() => activity()}" in text
    assert "onChange={(event) => { activity(); onQuestionChange(event.target.value); }}" in text
    assert "const activity = onActivity || (() => {});" in text, "a panel without it must not throw"


def test_the_panel_warms_on_mount_and_passes_the_same_function_down() -> None:
    text = PANEL.read_text(encoding="utf-8")
    assert "import { keepPrefixWarm } from './kai-keep-warm';" in text
    assert "keepPrefixWarm(controller.signal);" in text
    assert "onActivity={keepPrefixWarm}" in text


def test_every_file_this_round_touched_stays_under_the_cap() -> None:
    for path in (KEEP_WARM, COMPOSER, PANEL):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        assert lines <= 450, f"{path.name} is {lines} lines"


# --- the second ask ----------------------------------------------------------

def test_a_direct_instruction_is_recorded_rather_than_asked_about() -> None:
    """Rule 30. Measured before it existed: 3 of 6 imperatives ended with a
    proposal recorded, and the rest asked whether to record one, which is the
    same change approved twice. After it: 4 of 4 in the same batch."""
    text = assistant_prompt.SYSTEM_PROMPT
    assert "A direct instruction is a request to record the change" in text
    assert "call the propose_* tool for it in the same run" in text
    assert "Never answer a direct instruction by asking whether to record a proposal" in text
    assert "simulate, then propose, in the same run" in text


def test_the_rule_names_the_two_cases_that_are_not_it() -> None:
    """A rule with no exception written into it gets applied where it does not
    belong. Neither of these may end in a proposal: a guess, or a no-op."""
    text = assistant_prompt.SYSTEM_PROMPT
    assert "whose field or value you would have to guess" in text
    assert "a value the saved state already holds" in text


def test_the_new_rule_does_not_contradict_the_one_it_sits_beside() -> None:
    """Rule 4 governs what may be SAID: nothing is described as recorded unless
    a propose call stands behind it. Rule 30 governs what should be DONE. The
    order matters and the prompt states it, or the two read as a licence to
    claim first and call afterwards."""
    text = assistant_prompt.SYSTEM_PROMPT
    assert "unless a propose_* tool actually returned a result in this turn" in text
    assert text.index("4. Proposals") < text.index("30. A direct instruction")
    assert "and then say what that call returned" in text


def test_the_new_prose_keeps_the_house_style() -> None:
    """No em-dash, no exclamation mark, no retired word, in everything a person
    or a model reads that this round added."""
    comments = "\n".join(line for line in KEEP_WARM.read_text(encoding="utf-8").splitlines()
                         if line.strip().startswith("//"))
    rule = assistant_prompt.SYSTEM_PROMPT[assistant_prompt.SYSTEM_PROMPT.index("30. A direct"):]
    for text in (comments, rule, __doc__ or ""):
        assert "—" not in text and "–" not in text and "!" not in text
        assert "משתמש" not in text
        assert "חישוב מחדש" not in text and "הפסקות" not in text
