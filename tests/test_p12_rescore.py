"""The held-out re-score: the metric, the two bars and the staleness.

Every test here builds its own tiny world rather than reading the repository's
artifacts, so a failure names a rule that broke and not a number that moved.
Two of them exist to prove a guard bites: the fold-dispersion bar is restored to
the naive form it replaces and the wrong verdict is asserted to come back.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scripts import adopt_candidate_rescore as rescore


def _frame(cells, values, start="2024-11-01"):
    stamps = pd.date_range(start, periods=len(values), freq="h")
    return pd.DataFrame({"channel_name": list(cells), "log_effect": list(values),
                         "break_start": stamps})


def _tree(tmp_path, shipped, candidates):
    (tmp_path / "models" / "candidates").mkdir(parents=True)
    (tmp_path / "models" / "releases").mkdir(parents=True)
    (tmp_path / "models" / "tv_break_coefficients.json").write_text(
        json.dumps({"coefficients": shipped, "metadata": {}, "detail": {}}), encoding="utf-8")
    for name, coefficients in candidates.items():
        (tmp_path / "models" / "candidates" / f"tv_break_coefficients_{name}.json").write_text(
            json.dumps({"coefficients": coefficients, "metadata": {}, "detail": {}}), encoding="utf-8")
    return rescore.Paths(root=tmp_path)


def test_identical_predictions_report_identical_and_not_a_tiny_difference(tmp_path):
    paths = _tree(tmp_path, {"a": 0.1, "b": -0.2}, {"twin": {"a": 0.1, "b": -0.2}})
    frame = _frame(["a", "b"] * 8, [0.1, -0.3] * 8)
    payload = rescore.rescore(paths, frame)
    row = payload["candidates"][0]
    assert row["verdict"]["state"] == "identical"
    assert row["paired"]["rmse_delta"] == 0.0


def test_a_large_move_clears_both_bars_and_is_called_worse(tmp_path):
    paths = _tree(tmp_path, {"a": 0.0}, {"wild": {"a": 5.0}})
    frame = _frame(["a"] * 40, list(np.linspace(-0.2, 0.2, 40)))
    row = rescore.rescore(paths, frame)["candidates"][0]
    assert row["verdict"]["state"] == "worse"
    assert row["verdict"]["clears_paired_bar"] and row["verdict"]["clears_fold_dispersion"]


def _fold_swinging_frame():
    """Five temporal folds that disagree sharply, with little noise inside one.

    Constructed so the paired statistic over all 200 breaks clears its bar while
    the movement in RMSE is a fraction of the fold-to-fold dispersion. That is
    the exact shape a one-bar rule gets wrong, and it is not a contrived corner:
    it is what a month whose weeks differ looks like.
    """
    generator = np.random.default_rng(3)
    values = np.concatenate([mean + generator.normal(0, 0.02, 40)
                             for mean in (0.5, -0.5, 0.5, -0.5, -0.15)])
    return _frame(["a"] * 200, values)


def test_a_move_smaller_than_the_fold_dispersion_is_not_distinguishable(tmp_path):
    paths = _tree(tmp_path, {"a": 0.0}, {"nudge": {"a": 0.1}})
    row = rescore.rescore(paths, _fold_swinging_frame())["candidates"][0]
    paired = row["paired"]
    assert abs(paired["paired_statistic"]) >= rescore.PAIRED_T_BAR
    assert abs(paired["rmse_delta"]) < paired["fold_dispersion"]
    assert row["verdict"]["state"] == "not_distinguishable"


def test_the_fold_bar_is_what_holds_that_case_and_removing_it_brings_the_defect_back(tmp_path):
    """Restore the naive one-bar rule and assert the wrong verdict returns.

    A guard that has never been shown to fail has never been shown to work.
    """
    paths = _tree(tmp_path, {"a": 0.0}, {"nudge": {"a": 0.1}})
    paired = rescore.rescore(paths, _fold_swinging_frame())["candidates"][0]["paired"]
    naive = "better" if paired["rmse_delta"] < 0 else "worse"
    assert abs(paired["paired_statistic"]) >= rescore.PAIRED_T_BAR
    assert naive == "worse"
    assert rescore.verdict(paired, identical=False)["state"] == "not_distinguishable"


def test_the_baselines_predict_each_break_from_the_others_and_never_from_itself(tmp_path):
    paths = _tree(tmp_path, {"a": 0.0}, {})
    frame = _frame(["a"] * 6, [1.0, 1.0, 1.0, 1.0, 1.0, 0.0])
    payload = rescore.rescore(paths, frame)
    baselines = {row["id"]: row for row in payload["baselines"]}
    assert baselines["global_mean_loo"]["out_of_sample"] is True
    assert baselines["cell_mean_loo"]["out_of_sample"] is True
    assert baselines["global_mean_loo"]["rmse"] > 0


def test_the_cell_split_earns_its_place_only_when_the_cells_genuinely_differ(tmp_path):
    paths = _tree(tmp_path, {"a": 0.0, "b": 0.0}, {})
    separated = _frame(["a"] * 30 + ["b"] * 30, [1.0] * 30 + [-1.0] * 30)
    assert rescore.rescore(paths, separated)["cell_structure"]["earns_its_place"] is True
    noise = np.random.default_rng(7).normal(0, 1, 60)
    mixed = _frame(["a", "b"] * 30, noise)
    assert rescore.rescore(paths, mixed)["cell_structure"]["earns_its_place"] is False


def test_a_cell_the_artifact_does_not_carry_is_counted_and_not_predicted_at_zero(tmp_path):
    paths = _tree(tmp_path, {"a": 0.5}, {"partial": {"a": 0.5}})
    frame = _frame(["a", "unseen"] * 5, [0.5, 0.5] * 5)
    payload = rescore.rescore(paths, frame)
    assert payload["shipped"]["cells_not_carried"] == 5
    assert payload["candidates"][0]["cells_not_carried"] == 5


def test_two_candidates_that_predict_the_same_thing_are_named_as_duplicates(tmp_path):
    paths = _tree(tmp_path, {"a": 0.0},
                  {"one": {"a": 0.3}, "two": {"a": 0.3}, "three": {"a": 0.9}})
    payload = rescore.rescore(paths, _frame(["a"] * 12, [0.1] * 12))
    assert payload["duplicate_groups"] == [["one", "two"]]
    by_id = {row["id"]: row for row in payload["candidates"]}
    assert by_id["one"]["duplicate_of"] == ["two"]
    assert by_id["three"]["duplicate_of"] == []


def test_a_stored_score_goes_stale_when_a_candidate_moves_and_names_which_one(tmp_path):
    paths = _tree(tmp_path, {"a": 0.0}, {"one": {"a": 0.3}})
    rescore.save_rescore(rescore.rescore(paths, _frame(["a"] * 8, [0.1] * 8)), paths)
    assert rescore.rescore_state(paths)["state"] == "current"
    target = paths.candidates_dir / "tv_break_coefficients_one.json"
    target.write_text(json.dumps({"coefficients": {"a": 0.4}, "metadata": {}, "detail": {}}),
                      encoding="utf-8")
    state = rescore.rescore_state(paths)
    assert state["state"] == "stale"
    assert state["changed"] == ["candidate:one"]


def test_a_score_that_was_never_run_says_so_rather_than_reading_as_zero(tmp_path):
    paths = _tree(tmp_path, {"a": 0.0}, {"one": {"a": 0.3}})
    assert rescore.rescore_state(paths)["state"] == "not_measured"
    assert rescore.load_rescore(paths) is None


def test_the_in_sample_limit_rides_on_every_payload_with_what_would_lift_it(tmp_path):
    paths = _tree(tmp_path, {"a": 0.0}, {"one": {"a": 0.3}})
    limit = rescore.rescore(paths, _frame(["a"] * 8, [0.1] * 8))["limit"]
    assert limit["state"] == "in_sample"
    assert "optimistic" in limit["en"]
    assert limit["unblocked_by_en"]
    assert limit["unblocked_by_he"]


@pytest.mark.parametrize("state", ["identical", "better", "worse", "not_distinguishable"])
def test_every_verdict_carries_both_languages(state):
    assert rescore.VERDICTS[state]["en"] and rescore.VERDICTS[state]["he"]
