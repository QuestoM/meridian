"""The job picker's doors have to open something that exists.

The picker is the answer to a new starter whose account has no job: thirteen
rows, each naming a job and the door it opens. Section 8.3 makes P1 the piece
that renders it, over the contract W0-4 froze in ``session.js``.

The failure this file exists to catch is the quiet one: a door that names a
surface the product does not have, so a person on their first morning clicks
their own job and lands nowhere. It is checked statically, against the frozen
door map and the shell's own navigation list, because both are source of truth
files and neither needs a browser to read.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
SESSION = SRC / "session.js"
NAV = SRC / "shell" / "nav.js"
PICKER = SRC / "today" / "JobPicker.jsx"


def _doors() -> set[str]:
    """The door ids W0-4 froze, read out of the DOORS map itself."""
    text = SESSION.read_text(encoding="utf-8")
    block = text.split("export const DOORS = {", 1)[1].split("\n};", 1)[0]
    return set(re.findall(r"^\s{2}'?([a-z_.]+)'?:\s*\{", block, re.M))


def _nav_labels() -> set[str]:
    text = NAV.read_text(encoding="utf-8")
    block = text.split("export const navItems = [", 1)[1].split("];", 1)[0]
    return set(re.findall(r"\[\s*'([^']+)'", block))


def _door_views() -> dict[str, str | None]:
    text = PICKER.read_text(encoding="utf-8")
    block = text.split("export const DOOR_VIEWS = {", 1)[1].split("\n};", 1)[0]
    views: dict[str, str | None] = {}
    for door, value in re.findall(r"^\s+'?([a-z_.]+)'?:\s*(null|'[^']*'),", block, re.M):
        views[door] = None if value == "null" else value.strip("'")
    return views


def test_every_frozen_door_has_a_destination_decided_for_it():
    """Thirteen doors, thirteen answers, and no row that silently does nothing."""
    doors = _doors()
    views = _door_views()
    assert len(doors) == 13
    assert set(views) == doors


def test_every_named_destination_is_a_surface_the_product_actually_has():
    labels = _nav_labels()
    for door, view in _door_views().items():
        if view is not None:
            assert view in labels, f"{door} points at {view!r}, which is not a navigation entry"


def test_the_two_doors_that_are_not_views_are_answered_rather_than_mis_routed():
    """Neither is a page of this shell, so neither may claim a navigation entry.

    One is a menu and the other mounts its own root over the page. What each row
    says about that is guarded in ``test_p1_reader_words.py``, which also holds
    the model steward's row to the address the console publishes.
    """
    views = _door_views()
    assert views["account.accounts"] is None
    assert views["model.console"] is None
    text = PICKER.read_text(encoding="utf-8")
    block = text.split("const DOOR_NOTES = {", 1)[1].split("\n};", 1)[0]
    assert "account.accounts" in block
    assert "model.console" in block


def test_the_picker_reads_its_rows_from_the_frozen_contract_and_writes_through_it():
    """It never rebuilds the job list, so the company-only row cannot leak."""
    text = PICKER.read_text(encoding="utf-8")
    assert "jobPickerRows" in text
    assert "saveJob" in text
    assert "JOBS" not in text.replace("jobPickerRows", "")


def test_the_picker_is_rendered_only_when_the_account_has_no_job():
    page = (SRC / "today" / "OverviewPage.jsx").read_text(encoding="utf-8")
    assert "needsJobPicker(session)" in page
