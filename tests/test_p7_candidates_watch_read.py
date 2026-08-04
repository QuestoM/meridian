"""What the candidate-shelf measurement reads, out of the product and out of the screen.

Split out of ``test_p7_candidates_watch.py`` when that file passed the 450-line
law, and nothing changed on the way out: the same readers, the same rounding, the
same failure collector. It defines no test of its own. Every fixture and every
assertion stayed in the file that owns them, beside the file that holds the page.

Three kinds of reader live here, and they are here together because each one
exists to keep a measurement honest rather than convenient:

- **The product's own words and figures.** A word the screen must show is read
  out of the module that defines it, never typed here, so a re-worded panel
  fails on the assertion that cares rather than on a stale copy of a string.
- **The screen's own arithmetic.** A shekel figure is compared digit for digit
  against what the browser would print, rounded the way the browser rounds, so
  the two agree on every half rather than agreeing most of the time.
- **What went wrong inside a thread nobody here owns.** A measurement that fails
  inside its own thread reaches nothing that can be asserted on, so both channels
  are collected while it runs and quoted in the failure message.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import re
import threading
import traceback
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from test_p7_candidates_watch_page import SUBJECT

ROOT = Path(__file__).resolve().parents[1]
WORDS = ROOT / "tv-break-dashboard" / "src" / "model" / "console" / "console-words.js"

# The store the product ships, read to seed the temporary one and hashed to
# prove the measurement never wrote it.
SHIPPED_MEASUREMENTS = ROOT / "models" / "releases" / "candidate_measurements.json"
SHIPPED_COEFFICIENTS = ROOT / "models" / "tv_break_coefficients.json"

# The candidate artifact the measurement reads and must not touch.
SUBJECT_ARTIFACT = ROOT / "models" / "candidates" / f"tv_break_coefficients_{SUBJECT}.json"


def word(key: str, locale: str = "he") -> str:
    """The product's own word for a key, read from the module that defines it."""
    pattern = rf"'{re.escape(key)}':\s*\{{[^}}]*\b{locale}: '([^']+)'"
    found = re.search(pattern, WORDS.read_text(encoding="utf-8"))
    assert found, f"the console defines no {locale} word for {key}"
    return found.group(1)


def row(payload: "dict", identifier: str = SUBJECT) -> "dict":
    """The subject's row out of a shelf payload."""
    for candidate in payload.get("candidates") or []:
        if candidate.get("id") == identifier:
            return candidate
    raise AssertionError(f"the shelf payload carries no candidate called {identifier}")


def card(phase: "dict", identifier: str = SUBJECT) -> "dict":
    """The subject's card out of a browser phase."""
    for rendered in phase["cards"]:
        if rendered["id"] == identifier:
            return rendered
    raise AssertionError(f"the screen carries no card for {identifier}: {phase['cards']}")


def digits(value: str) -> str:
    return "".join(character for character in value if character.isdigit())


def js_number(value: float) -> str:
    """A number spelled the way a template literal spells it, so 87.0 reads 87."""
    return str(int(value)) if float(value).is_integer() else str(value)


def shekels(value: float) -> str:
    """The digits the browser prints for a shekel figure, rounded as it rounds.

    ``toLocaleString`` with no fraction digits rounds half away from zero, which
    is what ``ROUND_HALF_UP`` means here, so the two agree on every half.
    """
    rounded = Decimal(str(value)).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return digits(str(rounded))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "absent"


class _Collected(logging.Handler):
    """Whatever the measurement thread says about itself, kept for the message."""

    def __init__(self, lines: "list[str]") -> None:
        super().__init__(level=logging.ERROR)
        self.lines = lines

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))


@contextlib.contextmanager
def thread_failures():
    """Both ways the measurement thread can end badly, collected while it runs.

    The measurement runs in a thread the test file does not own, so a failure
    inside it reaches nothing that can be asserted on: the register clears, the
    store stays empty, and the shelf honestly reads not measured. Observed once
    on 2026-08-04, on a tree with eight other builds running, and the failure
    named no cause at all. Both channels are taken because the route logs what it
    catches and the interpreter reports what it does not.
    """
    lines: "list[str]" = []
    handler = _Collected(lines)
    root = logging.getLogger()
    root.addHandler(handler)
    previous = threading.excepthook

    def hook(args) -> None:
        lines.append("".join(traceback.format_exception(
            args.exc_type, args.exc_value, args.exc_traceback)))
        previous(args)

    threading.excepthook = hook
    try:
        yield lines
    finally:
        threading.excepthook = previous
        root.removeHandler(handler)
