"""Tests for the first-break retention adjustment.

Three layers are exercised: the optimizer math (the multiplier only charges the
first break, never moves revenue when off), the measurement gate (it ships a
value only when the contrast is large and significant), and the transform/schedule
plumbing (off without a model, on when assumptions carry a measured multiplier).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.model.measure import first_break_gate
from kairos.optimize.optimizer import ProgramSegment, _segment_retention
from kairos.optimize.pricing import OptimizerAssumptions


def _segment(multiplier: float = 1.0, coef: float = -0.03) -> ProgramSegment:
    return ProgramSegment(
        segment_id="t", channel="c", day="2024-11-01", start_seconds=0.0,
        duration_seconds=3600.0, program_type="News", baseline_tvr=10.0, cpp=100.0,
        impact_coefficient=coef, retention_baseline=1.0, first_break_multiplier=multiplier,
    )


def test_multiplier_one_is_exactly_the_base_model():
    """With multiplier 1.0 the retention is the unchanged linear model at every k."""
    base = _segment(1.0)
    for k in range(0, 5):
        assert _segment_retention(base, k) == pytest.approx(1.0 + (-0.03) * k)


def test_first_break_charged_extra_only_once():
    """The extra cost (coef * (mult - 1)) is applied once, when k >= 1."""
    seg = _segment(1.6)
    # k=0: no break, no extra.
    assert _segment_retention(seg, 0) == pytest.approx(1.0)
    # k=1: base + extra. extra = -0.03 * 0.6 = -0.018.
    assert _segment_retention(seg, 1) == pytest.approx(1.0 - 0.03 - 0.018)
    # k=2: two base breaks + the single first-break extra (not doubled).
    assert _segment_retention(seg, 2) == pytest.approx(1.0 - 0.06 - 0.018)


def test_first_break_lowers_retention_so_first_break_costs_more():
    """A multiplier > 1 must lower retention vs off (more cost), never raise it."""
    off = _segment(1.0)
    on = _segment(1.6)
    for k in range(1, 5):
        assert _segment_retention(on, k) < _segment_retention(off, k)


def test_multiplier_below_one_rejected():
    with pytest.raises(ValueError):
        _segment(0.9).validate()
    with pytest.raises(ValueError):
        OptimizerAssumptions(first_break_multiplier=0.5)


def _effects_frame(first_logs, later_logs, prog_key=("c", "2024-11-01", 0)):
    """Build a minimal effects frame with first/later breaks in one programme."""
    rows = []
    for i, lg in enumerate(first_logs):
        rows.append({"log_effect": lg, "ordinal": 1.0, "prog_key": (prog_key[0], prog_key[1], i)})
    # later breaks belong to the SAME programmes as the firsts (need >=2 per prog).
    for i, lg in enumerate(later_logs):
        key_idx = i % max(1, len(first_logs))
        rows.append({"log_effect": lg, "ordinal": 2.0, "prog_key": (prog_key[0], prog_key[1], key_idx)})
    return pd.DataFrame(rows)


def test_gate_off_when_effect_is_noise():
    """No real difference between first and later -> multiplier stays 1.0 (off)."""
    rng = np.random.default_rng(0)
    first = rng.normal(-0.02, 0.05, 300).tolist()
    later = rng.normal(-0.02, 0.05, 300).tolist()
    out = first_break_gate(_effects_frame(first, later))
    assert out["first_break_multiplier"] == 1.0
    assert out["first_break_active"] is False


def test_gate_ships_value_when_first_break_sheds_more():
    """A real, large, significant first-break effect ships a multiplier > 1.0."""
    rng = np.random.default_rng(1)
    # First breaks shed clearly more (more negative) than later breaks.
    first = rng.normal(-0.05, 0.04, 400).tolist()
    later = rng.normal(-0.015, 0.04, 400).tolist()
    out = first_break_gate(_effects_frame(first, later))
    assert out["first_break_active"] is True
    assert out["first_break_multiplier"] > 1.0
    assert out["first_break_p_value"] < 0.01


def test_gate_empty_or_missing_columns_is_off():
    assert first_break_gate(pd.DataFrame())["first_break_multiplier"] == 1.0
    assert first_break_gate(pd.DataFrame({"log_effect": [-0.1]}))["first_break_multiplier"] == 1.0


def test_transform_applies_multiplier_only_with_impact_model():
    """The segment carries 1.0 without a model and the measured value with one."""
    from kairos.data.transform import _first_break_multiplier

    a = OptimizerAssumptions(first_break_multiplier=1.6)
    assert _first_break_multiplier(None, a) == 1.0          # no model -> off
    # A truthy "model" object is enough; the function only checks for None.
    assert _first_break_multiplier(object(), a) == 1.6
