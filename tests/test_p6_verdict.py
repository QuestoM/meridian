"""P6 Sources: the verdict over an accepted file, swept over the whole vocabulary.

Three times this piece has printed a green tick over a file that was not good
news, and three times it has closed the one instance somebody measured. Round
eight: a candidate the engine would never read. Round twelve: a candidate the
engine would read that carried no rows. Round four of this wave: a daily log
whose 20 of 20 rows carried a clock the loader cannot read, answered
``accepted: true``, ``will_be_read: true``, ``replaces_live_input`` with one warn
finding, and rendered ``source-verdict ok`` under the heading "the file passed
every check" over an enabled commit button.

Each of those was closed as its own case. This file closes the class instead, on
the same pattern as :func:`tests.test_p6_empty.test_every_kind_this_door_accepts_declares_what_no_rows_means`:
the rule is declared for every consequence the server can send, so an eighth code
cannot inherit the hole by being added at one end and forgotten at the other. The
gap in round four was exactly that shape. The outcome existed, the server had no
code for it, and the card had no heading for it, so the file fell through to the
sentence that reads as good news.

Two rules are asserted here, both through the shipped modules run as themselves.

**No candidate carrying a warn-severity finding renders the plain pass.** Not for
any consequence the server can send, and not for either kind of warning: one that
costs the engine a field, and one it read a value two ways to keep.

**What a clean candidate renders is declared for every code.** The table below has
to name every one of the server's own consequences and no others, so adding a
code without deciding what the card prints over it fails this test rather than
shipping as a green tick. Three of the seven really are good news and really do
print the plain pass, which is what stops the rule above being satisfied by
turning every verdict amber.

Nothing here talks to the door. :mod:`tests.test_p6_empty` already drives real
files through it and asserts that the codes this sweep enumerates are the ones it
really sends; what is under test here is the rendering rule over the whole
vocabulary, including the combinations one morning's file cannot reach.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import kairos_api.uploads_status as uploads_status

HEBREW = re.compile(r"[֐-׿]")

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")
SOURCES = ROOT / "tv-break-dashboard" / "src" / "sources"
FINDINGS_URL = (SOURCES / "sources-findings.js").as_uri()
COPY_URL = (SOURCES / "sources-copy.js").as_uri()

# What the card prints over a candidate the door accepted and that carries no
# warning at all, for every consequence the server can send. Three of them are
# the plain pass and the other four are not, and each of the four says something
# a green tick would hide: nothing will read this file, it will be read and it
# carries no rows, or the server counted a warning the payload did not carry.
CLEAN_VERDICT = {
    "replaces_live_input": ("accepted", "ok"),
    "changes_model_basis": ("accepted", "ok"),
    "replaces_only_a_later_day": ("accepted", "ok"),
    "stored_not_read": ("acceptedNotRead", "warn"),
    "stored_without_replacing": ("acceptedNotRead", "warn"),
    "replaces_live_input_with_no_rows": ("acceptedNoRows", "warn"),
    "replaces_live_input_with_warnings": ("acceptedWarned", "warn"),
}

# The two shapes a warning on an accepted file comes in, as the door really
# sends them. The first is a field the engine never got, measured on the daily
# log whose every row lost its clock. The second is a value it did get and read
# one of the two ways it could be read, measured on the slash dates whose day
# number is twelve or under. They are different news and they do not share a
# heading, and neither of them is a pass.
WARNINGS = {
    "a field the engine never got": [
        {"severity": "warning", "code": "unreadable_times", "column": "שעה", "effect": "field_lost"},
    ],
    "a value read one of two ways": [
        {"severity": "warning", "code": "ambiguous_day_month", "column": "תאריך", "effect": "value_interpreted"},
    ],
    "one of each on the same file": [
        {"severity": "warning", "code": "unreadable_times", "column": "שעה", "effect": "field_lost"},
        {"severity": "warning", "code": "ambiguous_day_month", "column": "תאריך", "effect": "value_interpreted"},
    ],
}

# A finding with no effect key at all, from a door too old to send one and from
# the frozen data contracts, whose warnings this destination does not author.
WARNINGS["a warning that names no effect"] = [{"severity": "warning", "code": "contract_violation", "column": ""}]


def candidate(code: str, findings: list[dict]) -> dict:
    """One accepted response body, as the door really answers about a file.

    ``will_be_read`` is derived from the code rather than set by hand, because
    the server derives it the same way: the two stored codes are exactly the
    answers it gives when nothing will read the file.
    """
    return {
        "accepted": True,
        "checked": True,
        "will_be_read": not code.startswith("stored_"),
        "rows": 20,
        "consequence": {"code": code, "en": "", "he": ""},
        "findings": findings,
    }


def as_the_card_renders(cases: dict[str, dict]) -> dict[str, dict]:
    """The verdict panel the shipped card prints over each candidate.

    ``sources-findings.js`` holds the rule and ``SourceCard.jsx`` renders exactly
    what it returns, and ``sources-copy.js`` holds the words, so both are run
    here as themselves over the whole matrix in one process: what comes back is
    the heading an operator reads and the tone the panel carries, in both
    languages, and not a second implementation of either.
    """
    if NODE is None:
        pytest.skip("node is not on this machine, so the shipped rendering modules cannot be run")
    script = (
        f"import {{ acceptedVerdict }} from {json.dumps(FINDINGS_URL)};\n"
        f"import {{ text }} from {json.dumps(COPY_URL)};\n"
        "let raw = ''; for await (const chunk of process.stdin) raw += chunk;\n"
        "const out = {};\n"
        "for (const [name, body] of Object.entries(JSON.parse(raw))) {\n"
        "  const verdict = acceptedVerdict(body);\n"
        "  out[name] = { ...verdict, en: text(verdict.heading, 'en'), he: text(verdict.heading, 'he') };\n"
        "}\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    result = subprocess.run(
        [NODE, "--input-type=module", "--eval", script],
        input=json.dumps(cases),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_every_consequence_the_server_sends_declares_what_a_clean_file_renders() -> None:
    """The table above names every code, so an eighth cannot be added in silence.

    This is the assertion that makes the sweep below total. The gap in round four
    was a real outcome with no code at one end and no heading at the other, and a
    sweep over a hand-written list would have missed it for the same reason the
    card did.
    """
    assert set(CLEAN_VERDICT) == set(uploads_status.CONSEQUENCES), "a consequence the server sends has no declared verdict"
    plain = {code for code, (heading, _) in CLEAN_VERDICT.items() if heading == "accepted"}
    # Without this the rule below could be met by printing amber over everything,
    # which would say nothing and would train a steward to ignore the colour.
    assert plain, "no consequence renders the plain pass, so the amber means nothing"


def test_no_candidate_carrying_a_warning_renders_the_plain_pass_for_any_consequence() -> None:
    """The class, over every consequence and every shape of warning.

    A file the engine will read whose rows carry something it cannot read is not
    a pass, whatever the consequence code says, and the heading over it has to be
    a real sentence in both languages rather than a key with no word behind it.
    """
    cases = {
        f"{code} with {shape}": candidate(code, findings)
        for code in uploads_status.CONSEQUENCES
        for shape, findings in WARNINGS.items()
    }
    rendered = as_the_card_renders(cases)
    for name, panel in rendered.items():
        assert panel["tone"] != "ok", f"{name} rendered the ok tone over a warning"
        assert panel["heading"] != "accepted", f"{name} rendered the plain pass over a warning"
        assert panel["en"], f"{name} rendered the heading {panel['heading']} with no English word"
        assert HEBREW.search(panel["he"] or ""), f"{name} reaches a Hebrew screen in English"


def test_a_clean_candidate_renders_exactly_what_this_destination_declared() -> None:
    """The control, and the other half of the rule: a clean file still passes.

    Three of the seven codes are good news and print the plain pass in the ok
    tone. The other four are not good news even with an empty findings list, and
    each of them names what is wrong instead: nothing will read the file, it will
    be read and carries no rows, or the server counted a warning this payload did
    not carry, which is the fail-safe way round.
    """
    cases = {code: candidate(code, []) for code in uploads_status.CONSEQUENCES}
    rendered = as_the_card_renders(cases)
    for code, panel in rendered.items():
        assert (panel["heading"], panel["tone"]) == CLEAN_VERDICT[code], f"{code} rendered {panel['heading']} in the {panel['tone']} tone"
        assert panel["en"] and HEBREW.search(panel["he"] or ""), f"{code} has no heading in one of the two languages"
