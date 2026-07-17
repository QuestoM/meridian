"""Guard-path unit tests for the env-gated training entrypoint.

:mod:`kairos.model.train` refuses to train obscurely: it names the Python versions
and packages the Meridian stack needs and raises a clear error when they are
absent, and it never fabricates a coefficient. These tests exercise the pure
capability checks (``can_train``, ``_missing_packages``) and the input guards of
``train_tv_break_model`` without ever entering the heavy training body: on the
default desktop (Meridian and TensorFlow absent) the environment guard fires first
and raises before any data is touched, and where the stack is present the
None-data guard raises before the Meridian import.

Mirrors the deps-absent skip idiom of ``tests/test_model.py`` (skip the branch that
is unreachable under the current environment) so the file passes on every machine.
"""

from __future__ import annotations

import sys

import pandas as pd
import pytest

from kairos.model.train import (
    REQUIRED_PACKAGES,
    SUPPORTED_PYTHON,
    _missing_packages,
    _python_supported,
    can_train,
    train_tv_break_model,
)


def _tiny_synthetic_frame() -> pd.DataFrame:
    """A minimal placeholder frame; the guard paths never actually consume it."""
    return pd.DataFrame(
        {
            "channel": ["News__first__short", "News__first__short"],
            "time": [0, 1],
            "retention_tvr": [1.0, 0.98],
        }
    )


# 1. Pure capability checks ---------------------------------------------------
def test_can_train_is_bool_and_matches_its_definition() -> None:
    result = can_train()
    assert isinstance(result, bool)
    # can_train is exactly (supported Python AND no missing packages).
    assert result == (_python_supported() and not _missing_packages())


def test_missing_packages_is_a_subset_of_required_and_import_consistent() -> None:
    missing = _missing_packages()
    assert isinstance(missing, tuple)
    assert set(missing) <= set(REQUIRED_PACKAGES)
    # Each reported-missing package genuinely fails to import; each not-reported one
    # genuinely imports. This proves _missing_packages reflects real import state.
    import importlib.util

    for package in REQUIRED_PACKAGES:
        importable = importlib.util.find_spec(package) is not None
        assert (package in missing) == (not importable)


def test_python_supported_matches_the_declared_matrix() -> None:
    assert _python_supported() == (sys.version_info[:2] in SUPPORTED_PYTHON)


# 2. Environment guard: deps absent -> clear RuntimeError before any data ------
def test_train_env_guard_raises_before_touching_data_when_deps_absent() -> None:
    if can_train():
        pytest.skip("training stack is present; the environment guard does not apply")
    # The environment guard runs before the training-data guard, so passing a real
    # frame still raises the environment RuntimeError (never a ValueError, never a
    # silent partial run). This is the reachable guard path on the desktop.
    with pytest.raises(RuntimeError) as excinfo:
        train_tv_break_model(_tiny_synthetic_frame())
    message = str(excinfo.value)
    assert "3.11" in message and "3.12" in message
    assert "tensorflow" in message
    assert "google-meridian" in message
    # No model file is produced when the stack is missing (nothing fabricated).


def test_train_env_guard_fires_with_no_data_too() -> None:
    if can_train():
        pytest.skip("training stack is present; the environment guard does not apply")
    # With deps absent the environment guard wins even when data is omitted, so the
    # None-data ValueError is shadowed by the honest RuntimeError.
    with pytest.raises(RuntimeError):
        train_tv_break_model()


# 3. Input guard: deps present -> None data raises ValueError before Meridian ---
def test_train_input_guard_rejects_missing_data_when_stack_present() -> None:
    if not can_train():
        pytest.skip("training stack absent; the None-data guard is shadowed by the env guard")
    # When training is possible, omitting the data must raise a ValueError from the
    # input guard, which sits before the Meridian import, so this asserts the guard
    # without running (or needing) the real training flow.
    with pytest.raises(ValueError) as excinfo:
        train_tv_break_model(None)
    assert "training_data is required" in str(excinfo.value)
