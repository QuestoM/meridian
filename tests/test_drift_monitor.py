"""Weekly level-drift monitor: hand-computed math and honest absent states.

The drift statistic (docs/model-validation/uncertainty-calibration.md finding
4) is the mean log effect of the last 7 measured days minus the mean of the
preceding base, flagged binding when it exceeds twice the pooled coefficient's
95 percent half-width. Every expected number below is computed by hand from
the fixture values, so a regression in the module's arithmetic cannot hide
behind the module itself.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.model.drift_monitor import (
    BINDING_HALF_WIDTH_MULTIPLE,
    MIN_WINDOW_DAYS,
    level_drift,
)

_REQUIRED_KEYS = {
    "status", "reason", "n_breaks", "n_weeks", "window_days", "weekly_levels",
    "drift_per_week", "drift_se", "slope_per_week", "slope_se",
    "pooled_half_width_95", "binding_threshold", "binding", "criterion",
}


def _frame(rows: list[tuple[str, float]], *, cell: str = "News_first_short") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "channel_name": [cell] * len(rows),
            "break_start": [pd.Timestamp(day) for day, _ in rows],
            "log_effect": [effect for _, effect in rows],
        }
    )


# Two-week fixture. Week 1 (days 0..6): mean -0.04. Week 2 (days 7..13):
# mean -0.02. The trailing 7-day window (Nov 8..14) is exactly week 2.
_TWO_WEEKS = [
    ("2024-11-01", -0.02), ("2024-11-02", -0.04), ("2024-11-03", -0.06),
    ("2024-11-09", -0.01), ("2024-11-11", -0.02), ("2024-11-14", -0.03),
]


def test_two_week_drift_matches_hand_computation() -> None:
    result = level_drift(_frame(_TWO_WEEKS))

    assert set(result) == _REQUIRED_KEYS
    assert result["status"] == "measured"
    assert result["n_breaks"] == 6
    assert result["window_days"] == 14
    assert result["n_weeks"] == 2

    weekly = result["weekly_levels"]
    assert [w["week"] for w in weekly] == [1, 2]
    assert [w["n"] for w in weekly] == [3, 3]
    assert weekly[0]["mean_log_effect"] == pytest.approx(-0.04)
    assert weekly[1]["mean_log_effect"] == pytest.approx(-0.02)

    # drift = mean(week 2) - mean(week 1) = -0.02 - (-0.04) = +0.02.
    assert result["drift_per_week"] == pytest.approx(0.02)
    # var(wk1) = ((0.02)^2 + 0 + (0.02)^2) / 2 = 4e-4; var(wk2) = 1e-4;
    # se = sqrt(4e-4/3 + 1e-4/3) = sqrt(1.6667e-4) = 0.0129099.
    assert result["drift_se"] == pytest.approx(0.0129099445, abs=1e-9)
    # With exactly two week indices the OLS slope equals the mean difference.
    assert result["slope_per_week"] == pytest.approx(0.02, abs=1e-12)
    assert result["slope_se"] == pytest.approx(0.0129099445, abs=1e-9)


def test_binding_flag_false_when_drift_within_band() -> None:
    result = level_drift(_frame(_TWO_WEEKS))

    # Single cell: pooled within-variance = rss / (n - 1) with cell mean -0.03:
    # rss = 1e-4 + 1e-4 + 9e-4 + 4e-4 + 1e-4 + 0 = 1.6e-3, s2 = 3.2e-4;
    # half-width = 1.96 * sqrt(3.2e-4 / 6) = 0.01431382.
    assert result["pooled_half_width_95"] == pytest.approx(0.01431382, abs=1e-7)
    expected_threshold = BINDING_HALF_WIDTH_MULTIPLE * 0.014313818
    assert result["binding_threshold"] == pytest.approx(expected_threshold, abs=1e-7)
    # |drift| = 0.02 < 0.0286: the level moved less than the band tolerates.
    assert result["binding"] is False
    assert "half-width" in result["criterion"]


def test_binding_flag_true_when_drift_exceeds_twice_the_band() -> None:
    rows = _TWO_WEEKS[:3] + [
        ("2024-11-09", -0.11), ("2024-11-11", -0.12), ("2024-11-14", -0.13),
    ]
    result = level_drift(_frame(rows))

    # drift = -0.12 - (-0.04) = -0.08. Cell mean -0.08, rss = 0.0106,
    # s2 = 0.00212, half-width = 1.96 * sqrt(0.00212 / 6) = 0.03684243,
    # threshold = 0.07368486 < |drift|.
    assert result["drift_per_week"] == pytest.approx(-0.08)
    assert result["binding_threshold"] == pytest.approx(0.07368486, abs=1e-7)
    assert result["binding"] is True


def test_under_two_weeks_reports_honest_absent_state() -> None:
    result = level_drift(_frame(_TWO_WEEKS[:3]))  # 3 days of data

    assert result["status"] == "insufficient_data"
    assert str(MIN_WINDOW_DAYS) in result["reason"]
    assert result["window_days"] == 3
    assert result["n_weeks"] == 1
    assert len(result["weekly_levels"]) == 1
    assert result["drift_per_week"] is None
    assert result["drift_se"] is None
    assert result["binding"] is None
    assert result["binding_threshold"] is None
    assert set(result) == _REQUIRED_KEYS


def test_thirteen_day_span_is_still_insufficient() -> None:
    rows = [("2024-11-01", -0.02), ("2024-11-13", -0.06)]  # span 13 days
    result = level_drift(_frame(rows))
    assert result["status"] == "insufficient_data"
    assert result["binding"] is None


def test_exactly_two_weeks_is_measured_with_honest_missing_se() -> None:
    rows = [("2024-11-01", -0.02), ("2024-11-14", -0.06)]  # span 14 days
    result = level_drift(_frame(rows))
    assert result["status"] == "measured"
    assert result["drift_per_week"] == pytest.approx(-0.04)
    # One break per side: a standard error cannot be estimated, so it is
    # None rather than a fabricated zero.
    assert result["drift_se"] is None
    # Fewer than three breaks: no slope either.
    assert result["slope_per_week"] is None
    assert result["binding"] is False


def test_empty_and_missing_columns_are_absent_not_invented() -> None:
    empty = level_drift(pd.DataFrame(columns=["channel_name", "break_start", "log_effect"]))
    assert empty["status"] == "insufficient_data"
    assert empty["n_breaks"] == 0
    assert empty["binding"] is None

    no_time = level_drift(pd.DataFrame({"channel_name": ["a"], "log_effect": [-0.1]}))
    assert no_time["status"] == "insufficient_data"
    assert no_time["binding"] is None


def test_works_without_cell_column_using_plain_variance() -> None:
    frame = _frame(_TWO_WEEKS).drop(columns=["channel_name"])
    result = level_drift(frame)
    assert result["status"] == "measured"
    assert result["drift_per_week"] == pytest.approx(0.02)
    # Overall sample variance of the six log effects: mean -0.03,
    # ss = 1.6e-3, var = 1.6e-3 / 5 = 3.2e-4: identical to the single-cell
    # pooled variance here, so the half-width matches the cell-aware path.
    assert result["pooled_half_width_95"] == pytest.approx(0.01431382, abs=1e-7)
