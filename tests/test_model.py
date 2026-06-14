"""Tests for the env-gated model layer.

These run and pass on the Kairos desktop, where neither Meridian nor TensorFlow
is installed. They prove the pure-Python spec, the honest assumption fallback,
and the clear training error, without ever needing the trained stack.
"""

from __future__ import annotations

import pytest

from kairos.model import (
    AssumptionImpactModel,
    ImpactModel,
    build_channel_spec,
    can_train,
    load_impact_model,
    meridian_available,
    train_tv_break_model,
)
from kairos.model.spec import (
    DEFAULT_BREAK_LENGTHS,
    DEFAULT_BREAK_POSITIONS,
    DEFAULT_PROGRAM_TYPES,
)
from kairos.optimize.pricing import OptimizerAssumptions


def test_meridian_available_is_a_bool() -> None:
    assert isinstance(meridian_available(), bool)


def test_build_channel_spec_produces_full_cross_product() -> None:
    spec = build_channel_spec()
    expected = (
        len(DEFAULT_PROGRAM_TYPES)
        * len(DEFAULT_BREAK_POSITIONS)
        * len(DEFAULT_BREAK_LENGTHS)
    )
    assert spec.num_channels == expected
    assert spec.kpi_name == "retention_tvr"
    # Channel names join the three attributes with underscores.
    assert "News_first_short" in spec.channel_names
    assert "Other_last_long" in spec.channel_names


def test_channel_descriptors_keep_attributes_addressable() -> None:
    spec = build_channel_spec()
    first = spec.channels[0]
    assert first.program_type == DEFAULT_PROGRAM_TYPES[0]
    assert first.break_position == DEFAULT_BREAK_POSITIONS[0]
    assert first.break_length == DEFAULT_BREAK_LENGTHS[0]
    assert first.name == "_".join(
        (first.program_type, first.break_position, first.break_length)
    )


def test_build_channel_spec_rejects_empty_vocabulary() -> None:
    with pytest.raises(ValueError):
        build_channel_spec(program_types=())


def test_assumption_model_returns_declared_default_and_is_negative() -> None:
    model = AssumptionImpactModel()
    default = OptimizerAssumptions().retention_impact_per_break
    coefficient = model.coefficient_for("News", "first", "short")
    assert coefficient == default
    assert coefficient < 0
    assert model.source == "assumption"
    assert model.is_trained is False


def test_assumption_model_is_flat_across_attributes() -> None:
    model = AssumptionImpactModel()
    a = model.coefficient_for("News", "first", "short")
    b = model.coefficient_for("Other", "last", "long")
    assert a == b


def test_load_impact_model_missing_path_returns_assumption_fallback() -> None:
    model = load_impact_model("does/not/exist/posterior.pkl")
    assert isinstance(model, AssumptionImpactModel)
    assert isinstance(model, ImpactModel)
    assert model.coefficient_for("News", "first", "short") == (
        OptimizerAssumptions().retention_impact_per_break
    )


def test_can_train_is_a_bool() -> None:
    assert isinstance(can_train(), bool)


def test_train_raises_clear_error_when_deps_absent() -> None:
    if can_train():
        pytest.skip("training stack is present; the deps-absent error does not apply")
    with pytest.raises(RuntimeError) as excinfo:
        train_tv_break_model()
    message = str(excinfo.value)
    assert "3.11" in message and "3.12" in message
    assert "tensorflow" in message
    assert "google-meridian" in message
