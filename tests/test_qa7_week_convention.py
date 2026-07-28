"""The frozen Israeli week convention, pinned at the algorithm level.

The week starts on Sunday and ends on Saturday; the weekend is Friday and
Saturday (ISO 5 and 6); Sunday is a regular workday, never weekend. Data-layer
weekday numbers stay ISO (Monday=1 .. Sunday=7); only presentation order, week
windows and weekend semantics follow the Israeli convention. These tests pin:

  * the overview summary's planning-week window is Sunday-to-Saturday around
    the reference date, for every weekday the reference can fall on;
  * the Meridian training controls mark exactly Friday and Saturday as
    is_weekend (Sunday is a workday);
  * the weekday option lists both condition/constraint builders render are in
    Israeli order (Sunday first, Saturday last) while the keys stay ISO.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos.model.prepare import _control_frame  # noqa: E402
from kairos_api import core  # noqa: E402
from kairos_api.core import KairosSettings, _summarize_schedule  # noqa: E402

# 2026-07-19 is a Sunday (isoweekday 7) and 2026-07-25 a Saturday (isoweekday 6).
WEEK_SUNDAY = date(2026, 7, 19)
WEEK_SATURDAY = date(2026, 7, 25)


def _plan_frame() -> pd.DataFrame:
    """A synthetic one-channel July 2026 plan, one row per calendar day."""
    days = pd.date_range("2026-07-01", "2026-07-31", freq="D")
    return pd.DataFrame(
        {
            "date": [d.strftime("%Y-%m-%d") for d in days],
            "channel": ["TestChannel"] * len(days),
            "num_breaks": [3] * len(days),
            "total_break_time": [420] * len(days),
            "predicted_revenue": [1000.0] * len(days),
            "predicted_retention": [0.9] * len(days),
            "baseline_tvr": [5.0] * len(days),
        }
    )


def _summary_with_reference(monkeypatch: pytest.MonkeyPatch, reference: date) -> dict:
    monkeypatch.setattr(core, "_load_settings", lambda: KairosSettings())
    monkeypatch.setattr(core, "_reference_today", lambda settings: reference)
    return _summarize_schedule(_plan_frame())


# --- the summary week window is Sunday..Saturday ------------------------------

@pytest.mark.parametrize(
    "reference",
    [WEEK_SUNDAY + pd.Timedelta(days=offset).to_pytimedelta() for offset in range(7)],
    ids=["sunday", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday"],
)
def test_week_window_is_the_sunday_to_saturday_week_of_the_reference_date(
    monkeypatch: pytest.MonkeyPatch, reference: date
) -> None:
    week = _summary_with_reference(monkeypatch, reference)["week"]
    assert week is not None
    assert week["basis"] == "reference_date"
    # Whichever weekday "today" is, the window is the SAME Sunday..Saturday week.
    assert week["date_from"] == WEEK_SUNDAY.isoformat()
    assert week["date_to"] == WEEK_SATURDAY.isoformat()
    assert week["n_dates"] == 7


def test_week_window_starts_on_a_sunday_and_ends_on_a_saturday(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    week = _summary_with_reference(monkeypatch, date(2026, 7, 21))["week"]
    assert date.fromisoformat(week["date_from"]).isoweekday() == 7  # Sunday
    assert date.fromisoformat(week["date_to"]).isoweekday() == 6  # Saturday


def test_sunday_reference_opens_a_new_week_not_the_previous_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Sunday is the FIRST day of the Israeli week: a Sunday reference must not
    # fall into the week that ended the day before.
    week = _summary_with_reference(monkeypatch, WEEK_SUNDAY)["week"]
    assert week["date_from"] == WEEK_SUNDAY.isoformat()


def test_reference_outside_the_plan_falls_back_to_the_first_seven_dates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    week = _summary_with_reference(monkeypatch, date(2030, 1, 1))["week"]
    assert week["basis"] == "plan_first_week"
    assert week["date_from"] == "2026-07-01"
    assert week["n_dates"] == 7


def test_week_slice_sums_exactly_the_seven_window_days(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A wrong window would be a genuine money bug: 7 days at 1000 ILS each.
    week = _summary_with_reference(monkeypatch, date(2026, 7, 22))["week"]
    assert week["projected_revenue"] == pytest.approx(7000.0)
    assert week["total_breaks"] == 21


# --- weekend semantics: Friday and Saturday, never Sunday ---------------------

def test_is_weekend_control_marks_friday_and_saturday_only() -> None:
    days = [(WEEK_SUNDAY + pd.Timedelta(days=i).to_pytimedelta()).isoformat() for i in range(7)]
    controls, names = _control_frame(days)
    assert names[-1] == "is_weekend"
    weekend_flags = controls[0, :, -1].tolist()
    # Sunday..Thursday are workdays; Friday and Saturday are the weekend.
    assert weekend_flags == [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0]


def test_sunday_is_never_weekend_in_the_training_controls() -> None:
    controls, names = _control_frame([WEEK_SUNDAY.isoformat()])
    assert controls[0, 0, names.index("is_weekend")] == 0.0
    assert controls[0, 0, names.index("day_of_week_Sunday")] == 1.0


def test_day_one_hots_stay_iso_keyed_in_the_data_layer() -> None:
    # The data layer keeps ISO day identities; only presentation reorders.
    friday = date(2026, 7, 24).isoformat()
    controls, names = _control_frame([friday])
    assert controls[0, 0, names.index("day_of_week_Friday")] == 1.0
    assert float(np.sum(controls[0, 0, : names.index("is_weekend")])) == 1.0


# --- weekday option lists render in Israeli order, keys stay ISO --------------

ISRAELI_KEY_ORDER = ["7", "1", "2", "3", "4", "5", "6"]


def test_constraint_builder_weekday_options_are_sunday_first() -> None:
    from kairos_api._constraint_options import weekday_options

    options = weekday_options()
    assert [option["key"] for option in options] == ISRAELI_KEY_ORDER
    assert options[0]["en"] == "Sunday"
    assert options[0]["he"] == "ראשון"
    assert options[-1]["en"] == "Saturday"
    assert options[-1]["he"] == "שבת"


def test_condition_builder_weekday_options_are_sunday_first() -> None:
    from kairos_api.condition_validation import weekday_options

    options = weekday_options()
    assert [option["key"] for option in options] == ISRAELI_KEY_ORDER
    assert options[0]["en"] == "Sunday"
    assert options[-1]["en"] == "Saturday"


def test_weekday_option_keys_cover_the_full_iso_vocabulary() -> None:
    from kairos_api._constraint_options import weekday_options as constraint_days
    from kairos_api.condition_validation import weekday_options as condition_days

    for options in (constraint_days(), condition_days()):
        assert {option["key"] for option in options} == {str(n) for n in range(1, 8)}
