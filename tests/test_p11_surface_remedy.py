"""P11: the remedy a pacing row offers, executed out of the shipped module.

This runs `tv-break-dashboard/src/clients/pacing/pacing-helpers.js` in node, so the
assertions are about the file the browser loads and not about a copy of its logic.
The module imports nothing, which is why it can be executed directly and why the
ordering decision was put there rather than inside a component.

The bar is the product judgement this piece is most exposed on. A campaign that is
behind on day one of seven owes nobody anything: the flight can still make it up,
and a make-good raised against a day-one gap would put a debt in the ledger that
the week itself is about to settle. A make-good is offered when the shortfall is
owed, which is when everything on the traffic log is counted and it still falls
short. The last test is the guard: it holds the same behind-pace campaign in both
shapes and asserts the offered act changes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPERS = ROOT / "tv-break-dashboard" / "src" / "clients" / "pacing" / "pacing-helpers.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed here")


def _row(*, verdict, forward_state, gap=None, remaining=None, unsourced=()):
    return {
        "campaign_id": "CMP_T1",
        "headline": {"verdict": verdict, "ratio": None, "code": "", "reason_en": "", "reason_he": ""},
        "rating": {
            "unit": "rating_points",
            "goal": 70.0,
            "counted": {"through_counted_day": 30.0, "booked_total": 40.0},
            "reference": {"expected_through_counted_day": 40.0},
            "pace": {"verdict": verdict, "ratio": 0.75, "gap_to_reference": gap,
                     "code": "gap_in_elapsed" if verdict == "unknown" else "",
                     "reason_en": "named", "reason_he": "named"},
            "forward": {"state": forward_state, "remaining_to_goal": remaining,
                        "unsourced_remaining_days": list(unsourced)},
        },
        "money": None,
    }


def _remedy(row, open_ids=None):
    script = f"""
import {{ remedyFor }} from {json.dumps(str(HELPERS))};
const row = {json.dumps(row)};
const open = {json.dumps(open_ids or {})};
process.stdout.write(JSON.stringify(remedyFor(row, open)));
"""
    completed = subprocess.run(
        ["node", "--input-type=module", "--eval", script],
        capture_output=True, text=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


def test_a_fully_booked_flight_that_is_still_short_offers_the_make_good() -> None:
    row = _row(verdict="behind", forward_state="short_certain", gap=10.0, remaining=30.0)
    remedy = _remedy(row)
    assert remedy["kind"] == "raise"
    assert remedy["value"] == 30.0
    assert remedy["unit"] == "rating_points"


def test_a_row_whose_remaining_days_carry_no_source_offers_the_booking_not_a_debt() -> None:
    row = _row(verdict="behind", forward_state="not_booked_yet", gap=10.0, remaining=30.0,
               unsourced=("2025-05-01", "2025-05-02"))
    remedy = _remedy(row)
    assert remedy["kind"] == "book"
    assert remedy["days"] == ["2025-05-01", "2025-05-02"]
    assert remedy["remaining"] == 30.0


def test_an_unknown_pace_offers_the_thing_that_is_missing() -> None:
    row = _row(verdict="unknown", forward_state="not_booked_yet", gap=None, remaining=None)
    remedy = _remedy(row)
    assert remedy["kind"] == "supply"
    assert remedy["block"]["reason_he"] == "named"


def test_a_campaign_that_already_carries_a_make_good_opens_it_rather_than_raising_a_second() -> None:
    row = _row(verdict="behind", forward_state="short_certain", gap=10.0, remaining=30.0)
    remedy = _remedy(row, {"CMP_T1": ["MG_0007"]})
    assert remedy == {"kind": "open", "makeGoodId": "MG_0007"}


def test_a_covered_flight_on_pace_offers_nothing_and_says_nothing() -> None:
    row = _row(verdict="on_pace", forward_state="covered", gap=0.0, remaining=0.0)
    assert _remedy(row)["kind"] == "none"


def test_the_same_behind_campaign_offers_a_debt_only_when_the_debt_is_owed() -> None:
    """The guard. One difference between the two rows, one difference in the act.

    Both are behind on pace by the same amount. The first has every remaining
    broadcast day on the traffic log and still falls short, so the shortfall is
    owed. The second has remaining days nobody has a source for, so the flight can
    still deliver and there is nothing to compensate yet. A ladder that read the
    pace verdict before the forward state would offer a make-good for both, which
    is a debt raised against a week that has not happened.
    """
    owed = _remedy(_row(verdict="behind", forward_state="short_certain", gap=10.0, remaining=30.0))
    open_flight = _remedy(_row(verdict="behind", forward_state="not_booked_yet", gap=10.0,
                               remaining=30.0, unsourced=("2025-05-01",)))
    assert owed["kind"] == "raise"
    assert open_flight["kind"] == "book"
    assert owed["kind"] != open_flight["kind"]
    # And neither ever offers the pace gap as the amount owed.
    assert owed["value"] == 30.0
    assert open_flight.get("value") is None
