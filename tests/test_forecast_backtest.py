"""Walk-forward accuracy: the measurement, and the refusals it makes honestly.

The backtest is the surface that keeps the forecast's confidence band from being
a decoration, so the properties tested here are about the MEASUREMENT rather than
the model: that no fold ever sees its own test rows, that coverage is reported
against the level it was scored at, that a fold the data cannot support says so
instead of reporting a figure, that MAPE names the rows it could not score, and
that the comparison against the pre-model historical mean is reported on BOTH
objectives rather than only the one the model was tuned on.

One test runs the full real measurement. It costs a few seconds and it is the
only thing that proves the walk actually walks.
"""

import numpy as np
import pandas as pd
import pytest

from kairos.model.audience_frame import build_training_frame
from kairos.model.forecast_backtest import (
    MAPE_TVR_FLOOR,
    MIN_TEST_OBSERVATIONS,
    date_blocks,
    verdict_for,
    walk_forward,
)


@pytest.fixture(scope="module")
def frame():
    return build_training_frame()


@pytest.fixture(scope="module")
def report(frame):
    """The real walk-forward measurement, computed once for the whole module."""
    return walk_forward(frame=frame)


# ------------------------------------------------------------------- the blocks

def test_blocks_are_cut_on_whole_dates_and_never_inside_a_day(frame):
    """A day's observations share a calendar context and a competitor lineup, so
    splitting inside one would leak both across the fold boundary."""
    blocks = date_blocks(frame, 6)
    assert len(blocks) == 6
    seen = [day for block in blocks for day in block]
    assert len(seen) == len(set(seen)), "no date may appear in two blocks"
    assert seen == sorted(seen), "blocks must be contiguous and in time order"
    # Every observation date is covered exactly once.
    assert set(seen) == set(pd.to_datetime(frame["date"]).unique())


def test_a_frame_with_too_few_dates_forms_no_fold_and_says_so(frame):
    """A single day cannot be forecast from its own past, and the refusal must
    reach the TOP of the payload rather than hide in the per-fold detail."""
    one_day = frame[frame["date"] == frame["date"].min()]
    report = walk_forward(frame=one_day, spots=None, owned_channel="")
    assert report["available"] is False
    assert report["reason"], "an unavailable measurement must state its reason"
    assert "no fold could be scored" in report["reason"]
    assert not [f for f in report["folds"] if f.get("available")]
    assert report["overall"]["available"] is False


# ---------------------------------------------------------------- the walk shape

def test_the_first_fold_refuses_because_it_has_no_past(report):
    """Absence of prior data is not a failure; it is the training seed."""
    first = report["folds"][0]
    assert first["fold"] == 1
    assert first["available"] is False
    assert first["n_train"] == 0
    assert "no prior observations" in first["reason"]
    assert "mae" not in first, "an unavailable fold must not report a figure"
    assert any(gap["fold"] == 1 for gap in report["gaps"] if gap["kind"] == "fold_unavailable")


def test_every_scored_fold_trained_strictly_on_its_own_past(report):
    """The property the whole design rests on: no fold saw its own test rows."""
    scored = [f for f in report["folds"] if f.get("available")]
    assert len(scored) >= 3, report["folds"]
    for fold in scored:
        assert fold["train_to"] < fold["test_from"], fold
        assert fold["n_train"] > 0 and fold["n_test"] >= MIN_TEST_OBSERVATIONS
        # The training set grows as the walk advances; the test block does not
        # move backwards.
        assert fold["train_from"] <= fold["train_to"]
    for earlier, later in zip(scored, scored[1:]):
        assert later["n_train"] > earlier["n_train"]
        assert later["test_from"] > earlier["test_from"]


def test_a_fold_the_data_cannot_support_reports_unavailable_with_the_reason(frame):
    """Cut the window so finely that blocks fall under the scoring floor. Each
    such fold must name the reason rather than report a noisy figure."""
    narrow = frame[frame["date"] <= frame["date"].min() + pd.Timedelta(days=1)]
    report = walk_forward(frame=narrow, spots=None, owned_channel="", n_blocks=2)
    unavailable = [f for f in report["folds"] if not f.get("available")]
    assert unavailable, report["folds"]
    for fold in unavailable:
        assert fold["reason"]
        assert "mae" not in fold


# ------------------------------------------------------------------ the metrics

def test_coverage_is_reported_against_the_level_it_was_scored_at(report):
    """The honesty check on the band. Coverage far below the level would mean the
    published range is too narrow no matter how good the point forecast is."""
    overall = report["overall"]
    assert overall["available"] is True
    assert overall["interval_level"] == pytest.approx(0.80)
    coverage = overall["interval_coverage"]
    assert coverage is not None
    assert 0.0 <= coverage <= 1.0
    assert overall["interval_n"] > 0
    assert overall["interval_n"] + overall["interval_missing_n"] == overall["n"]
    assert overall["interval_mean_width"] > 0
    # Measured on the real window the band is CONSERVATIVE rather than narrow:
    # it covers more often than the nominal level. That is the honest direction
    # to err, and the verdict block says which way it went.
    assert report["verdict"]["interval_is_conservative"] is True


def test_mape_names_the_rows_it_could_not_score(report):
    """A percentage error against a measured rating of zero is unbounded, so
    those rows are excluded, counted, and explained."""
    overall = report["overall"]
    assert overall["mape"] is not None and overall["mape"] > 0
    assert overall["mape_n"] + overall["mape_excluded_n"] == overall["n"]
    assert overall["mape_excluded_n"] > 0, (
        "the real history carries measured zero ratings, which MAPE cannot score"
    )
    assert str(MAPE_TVR_FLOOR) in overall["mape_excluded_reason"]


def test_bias_is_reported_separately_from_absolute_error(report):
    """A forecast that is consistently low is a different failure from one that
    is off in both directions, and only the first is fixable by a constant."""
    overall = report["overall"]
    assert "bias" in overall and "mae" in overall
    assert abs(overall["bias"]) <= overall["mae"] + 1e-9, (
        "mean signed error can never exceed mean absolute error"
    )


def test_the_breakdowns_always_report_n_per_cell(report):
    """A per-genre figure without its observation count invites reading a
    three-observation cell as a finding."""
    for name in ("by_genre", "by_slot"):
        breakdown = report[name]
        assert breakdown, name
        for cell, metrics in breakdown.items():
            assert "n" in metrics, (name, cell)
            if metrics.get("available"):
                assert metrics["n"] > 0
                assert metrics["mae"] >= 0
        assert sum(m["n"] for m in breakdown.values()) == report["overall"]["n"]


def test_the_competitor_family_exclusion_is_declared_not_buried(report):
    """Its feature is a rival title's measured mean, which only prior data may
    supply, so a test date's lineup is unknown here. That costs the measurement
    something and the payload says so on its face."""
    excluded = [g for g in report["gaps"] if g["kind"] == "family_excluded"]
    assert len(excluded) == 1
    assert excluded[0]["family"] == "competitor_lineup"
    assert "only prior data" in excluded[0]["reason"]
    assert "competitor_note" in report["method"]


# ------------------------------------------------------------------ the verdict

def test_the_verdict_scores_the_model_against_the_pre_model_baseline(report):
    """History versus forecast, on the same held-out rows, in both objectives.

    On the real one-month window the two disagree, and that disagreement is the
    finding: the model wins in log space, which is the objective its gates were
    measured on, and loses in arithmetic rating points, which is the unit the
    plan prices in. Reporting only the first would be picking the scoreboard
    after the game."""
    overall = report["overall"]
    for key in ("historical_mae", "historical_rmse", "log_rmse", "historical_log_rmse"):
        assert key in overall, key
    verdict = report["verdict"]
    assert verdict["available"] is True
    assert verdict["headline_en"] and verdict["headline_he"]
    assert isinstance(verdict["beats_historical_in_log_space"], bool)
    assert isinstance(verdict["beats_historical_in_points"], bool)
    # The verdict's booleans must agree with the numbers behind them.
    assert verdict["beats_historical_in_log_space"] == (
        overall["log_rmse"] < overall["historical_log_rmse"])
    assert verdict["beats_historical_in_points"] == (
        overall["mae"] < overall["historical_mae"])
    if verdict["beats_historical_in_log_space"] and not verdict["beats_historical_in_points"]:
        assert overall["bias"] < 0
        assert verdict["mechanism_note_en"], (
            "a log-space win beside a points-space loss must name the "
            "retransformation shortfall rather than leave it for the reader"
        )


def test_the_verdict_refuses_when_nothing_was_scored():
    assert verdict_for({"available": False, "reason": "no scored observations"}) == {
        "available": False, "reason": "no scored observations",
    }


def test_the_measurement_is_reproducible_on_the_same_frame(frame):
    """Deterministic given the data: only the artifact's stamp reads a clock, and
    the fold models are stamped from their own training windows."""
    first = walk_forward(frame=frame, n_blocks=3)
    second = walk_forward(frame=frame, n_blocks=3)
    assert first["overall"] == second["overall"]
    assert first["by_genre"] == second["by_genre"]


def test_the_scored_totals_add_up(report):
    scored = [f for f in report["folds"] if f.get("available")]
    assert report["n_folds_scored"] == len(scored)
    assert report["n_observations_scored"] == sum(f["n"] for f in scored)
    assert report["n_observations_scored"] == report["overall"]["n"]
    assert report["window"]["n_observations"] >= report["n_observations_scored"]
