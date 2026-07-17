"""Break-operations board money is the committed plan's own money.

The board used to re-derive each break's revenue from the per-second base_rate
fed into 30-second CPP units, with the programme premium applied a second time
and EPG ratings fillna'd to 1.0, understating the board roughly 56x against the
committed plan. The contract now: every displayed break carries a plan-derived
split of its programme's predicted_revenue, the programme's breaks sum back to
the plan figure to the cent, and nothing (ratings included) is fabricated.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"

pytestmark = pytest.mark.skipif(
    not CSV_PATH.exists(), reason="no committed weekly plan on disk"
)


@pytest.fixture(scope="module")
def board():
    from kairos_api.core import _load_break_schedule, _load_programmes
    from kairos_api.dashboard_api import _build_break_operations

    plan = _load_break_schedule()
    programmes = _load_programmes()
    if plan.empty or programmes.empty:
        pytest.skip("plan or EPG unavailable")
    return _build_break_operations(programmes, plan), plan


def test_program_breaks_sum_to_the_matched_plan_revenue_to_the_cent(board):
    """For every board programme with displayed breaks, the sum of its breaks'
    revenue_calculated equals the matched plan rows' predicted_revenue exactly
    (2 decimal places), on the real committed CSV."""
    operations, plan = board
    from kairos_api.dashboard_api import _plan_by_program_key

    plan_index = _plan_by_program_key(plan)
    per_program = defaultdict(float)
    for item in operations["breaks"]:
        per_program[item["program_key"]] += float(item["revenue_calculated"])

    checked = 0
    for program in operations["programs"]:
        if program["break_markers"] <= 0 or program["revenue"] is None:
            continue
        plan_row = plan_index.get(
            (program["channel"], program["date"], program["start_time"])
        )
        assert plan_row is not None, f"board programme {program['key']} lost its plan row"
        expected = round(float(plan_row["predicted_revenue"]), 2)
        got = round(per_program[program["key"]], 2)
        assert got == pytest.approx(expected, abs=0.005), (
            f"{program['key']}: board breaks sum {got} != plan predicted_revenue {expected}"
        )
        assert round(float(program["revenue"]), 2) == pytest.approx(expected, abs=0.005)
        checked += 1
    assert checked > 0, "no board programme carried breaks; the reconciliation checked nothing"


def test_board_summary_revenue_equals_the_sum_of_break_money(board):
    operations, _plan = board
    total = round(sum(float(item["revenue_calculated"]) for item in operations["breaks"]), 2)
    assert operations["summary"]["revenue"] == pytest.approx(total, abs=0.01)
    # The old CPP re-derivation left the board ~56x under the plan splits; the
    # plan-derived board must carry real money whenever breaks are displayed.
    if operations["breaks"]:
        assert total > 0


def test_premium_is_display_only_and_never_remultiplied(board):
    """revenue_premium is provenance for the label; the plan's predicted_revenue
    already prices it, so the split must not scale with it. The direct proof is
    the sum-to-plan equality above, which cannot hold if a premium were
    multiplied in; this test pins that non-1.0 premiums genuinely appear on the
    board (so the equality is not vacuously multiplying by one)."""
    operations, _plan = board
    premiums = {float(item["revenue_premium"]) for item in operations["breaks"]}
    assert premiums, "no breaks on the board"
    assert any(abs(value - 1.0) > 1e-9 for value in premiums), (
        "every board premium is exactly 1.0, so the no-remultiplication proof is vacuous"
    )


def test_ratings_are_never_fabricated_to_one_point_zero():
    """A programme with no measured TVR and a plan row without baseline_tvr must
    report a null rating, not the old fillna(1.0)."""
    from kairos_api.dashboard_api import _build_break_operations

    programmes = pd.DataFrame(
        [
            {
                "Channel": "עכשיו 14",
                "Title": "Probe show",
                "Start_datetime": "2024-11-01 20:00:00",
                "End_datetime": "2024-11-01 21:30:00",
                "Duration": 5400,
                # No TVR column at all: the honest rating is unknown.
            }
        ]
    )
    schedule = pd.DataFrame(
        [
            {
                "channel": "עכשיו 14",
                "date": "2024-11-01",
                "day": "Fri",
                "program_type": "Movie",
                "start_time": "20:00",
                "num_breaks": 2,
                "break_length": 120.0,
                "total_break_time": 240.0,
                "predicted_revenue": 1000.01,
                "predicted_retention": 0.9,
                # No baseline_tvr column: nothing to source a rating from.
            }
        ]
    )
    operations = _build_break_operations(programmes, schedule)
    assert len(operations["breaks"]) == 2
    for item in operations["breaks"]:
        assert item["rating_predicted"] is None, (
            "rating was fabricated for a spot with no measured TVR anywhere"
        )
    # The remainder-on-last split reconciles an odd cent exactly.
    total = round(sum(item["revenue_calculated"] for item in operations["breaks"]), 2)
    assert total == pytest.approx(1000.01, abs=0.001)


def test_rating_prefers_the_plan_baseline_tvr(board):
    """When the matched plan row carries baseline_tvr, the displayed rating is
    that value (the basis predicted_revenue was priced on), not the raw EPG TVR."""
    operations, plan = board
    from kairos_api.dashboard_api import _plan_by_program_key

    plan_index = _plan_by_program_key(plan)
    checked = 0
    for program in operations["programs"]:
        plan_row = plan_index.get((program["channel"], program["date"], program["start_time"]))
        if plan_row is None or program["break_markers"] <= 0:
            continue
        baseline = plan_row.get("baseline_tvr")
        if baseline is None or pd.isna(baseline):
            continue
        breaks = [b for b in operations["breaks"] if b["program_key"] == program["key"]]
        for item in breaks:
            assert item["rating_predicted"] == pytest.approx(round(float(baseline), 2))
            checked += 1
    assert checked > 0, "no plan-matched break carried a baseline_tvr to verify against"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
