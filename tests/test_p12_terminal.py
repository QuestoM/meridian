"""What the terminal reads as: its legends, its words and its one flag.

Three findings from a blind sweep of the shipped surface, each pinned here so it
cannot come back. All three are about the same failure mode: a figure or a state
that the payload knew and the screen did not say.

- The artifacts table printed a ``stat`` column and a verdict word with no bar
  and no rule anywhere on screen, so a reader had to open the stored payload to
  learn what -0.52 was being compared against.
- The verdict screen printed ``not_shipped`` and ``not_distinguishable``, which
  are the store's own keys, on the one screen where the steward is choosing
  between exactly those two things.
- ``--json`` was accepted only before the subcommand, so the natural
  ``show --json`` was an argparse error and the payload this piece publishes was
  reachable only by guessing the flag order.
"""

from __future__ import annotations

from scripts import adopt_candidate as entry
from scripts import adopt_candidate_decide as decide
from scripts import adopt_candidate_render as render
from scripts import adopt_candidate_words as words


def _payload(bar=2.0):
    return {
        "live_version": {"name": "2026-08-07", "short": "abc12345", "artifacts": {}},
        "rescore_state": {"state": "current", "measured_at": "2026-08-07T00:00:00+00:00"},
        "evaluation": {"breaks": 12, "cells": 2, "window": "2024-11-01 to 2024-11-30",
                       "folds": 5, "metric_en": "rmse", "target_sd": 0.24,
                       "target_sd_en": "the spread of the target"},
        "limit": {"en": "in sample", "unblocked_by_en": "a second month"},
        "baselines": [], "shipped": {"rmse": 0.24}, "cell_structure": {},
        "duplicate_groups": [], "adoptions": [],
        "candidates": [{
            "id": "twin", "file": "models/candidates/tv_break_coefficients_twin.json",
            "bytes": 10, "rmse": 0.24, "rmse_delta": -0.0001, "paired_statistic": -0.52,
            "paired_bar": bar, "fold_dispersion": 0.0004,
            "verdict": "not_distinguishable", "duplicate_of": [], "cell_delta": None,
            "money": {"state": "not_measured", "revenue_delta": None},
            "decisions": 0, "latest_decision": None, "decision_on_rescore": False,
            "owner_approval": False, "adopted": False, "adoption_id": None,
            "next_act": {"en": "Measure the money.", "command": "measure twin"},
        }],
    }


def test_the_statistic_is_rendered_with_the_bar_it_is_read_against():
    """A statistic with no bar beside it is a number nobody can act on."""
    text = "\n".join(render.render(_payload(bar=2.0)))
    assert "-0.52" in text
    assert "paired statistic" in text
    assert "2.0" in text
    assert "fold dispersion" in text


def test_the_bar_is_read_from_the_measurement_and_never_typed_into_the_legend():
    """Move the measured bar and the sentence moves with it, or it is a claim."""
    text = "\n".join(render.render(_payload(bar=3.5)))
    assert "3.5" in text
    assert "reaches 2.0" not in text


def test_a_row_with_no_measured_bar_prints_no_legend_rather_than_a_default():
    payload = _payload()
    payload["candidates"][0]["paired_bar"] = None
    text = "\n".join(render.render(payload))
    assert "paired statistic" not in text


def test_the_paired_legend_carries_both_halves():
    assert words.PAIRED_LEGEND["en"].strip() and words.PAIRED_LEGEND["he"].strip()
    assert "{bar}" in words.PAIRED_LEGEND["en"] and "{bar}" in words.PAIRED_LEGEND["he"]


def test_the_verdict_screen_prints_no_raw_store_key():
    """``not_shipped`` and ``not_distinguishable`` are keys, not words."""
    text = "\n".join(decide.render({
        "candidate_id": "twin", "decision": "not_shipped", "outcome": "ready",
        "checks": [], "money": {"state": "not_measured"},
        "rescore_verdict": "not_distinguishable", "reason_is_hebrew": True,
        "evidence": {}, "money_direction": "unknown",
    }))
    assert "not_shipped" not in text
    assert "not_distinguishable" not in text
    assert "no ship" in text
    assert "no difference" in text


def test_the_two_verdict_words_carry_both_halves_and_neither_is_a_store_key():
    for key, entry in words.DECISION_WORDS.items():
        assert entry["en"].strip() and entry["he"].strip(), key
        assert key not in entry["en"] and key not in entry["he"], key


def test_the_json_flag_is_accepted_on_either_side_of_the_subcommand():
    parser = entry.build_parser()
    assert parser.parse_args(["--json", "show"]).json is True
    assert parser.parse_args(["show", "--json"]).json is True
    assert parser.parse_args(["show"]).json is False
    assert parser.parse_args(["diff", "twin", "--json"]).json is True
    assert parser.parse_args(["--json", "checks", "twin"]).json is True


def test_every_subcommand_takes_the_flag_so_none_of_them_is_a_dead_end():
    parser = entry.build_parser()
    argv = {"show": ["show"], "rescore": ["rescore"], "measure": ["measure", "twin"],
            "checks": ["checks", "twin"], "diff": ["diff", "twin"],
            "decide": ["decide", "twin", "--decision", "not_shipped"],
            "adopt": ["adopt", "twin"], "revert": ["revert", "ad-1"], "report": ["report"],
            "publish": ["publish"]}
    assert sorted(argv) == sorted(entry.COMMANDS)
    for name, arguments in argv.items():
        assert parser.parse_args([*arguments, "--json"]).json is True, name


def test_the_usage_block_names_the_flag_it_publishes():
    assert "--json" in entry.__doc__
