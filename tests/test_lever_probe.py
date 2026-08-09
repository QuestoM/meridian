"""The inert-lever guard must itself fail loudly, or it is decoration.

:mod:`tests.lever_probe` exists to catch a defect whose whole signature is
silence, so every one of its own refusals is tested here with a case that
triggers it. The most important is the VACUITY refusal: an unconstrained fixture
makes both verdicts meaningless, and the first draft of the demand-weight test
was written that way and passed while proving nothing.
"""

from __future__ import annotations

import pytest

from tests.lever_probe import (
    assert_lever_bites,
    assert_lever_is_inert,
    probe_lever,
)

# A fixture "binds" when its output is below the unconstrained maximum of 10.
BINDS = lambda output: output < 10          # noqa: E731
NEVER_BINDS = lambda output: False          # noqa: E731


def _probe(outputs, *, binds=BINDS):
    """Probe a fake lever whose settings map straight onto ``outputs``."""
    table = dict(enumerate(outputs))
    return probe_lever(
        name="fake lever",
        run=lambda setting: table[setting],
        settings=list(table),
        binds=binds,
    )


# ---------------------------------------------------------------------------
# The vacuity refusal: no verdict at all on an unconstrained fixture.
# ---------------------------------------------------------------------------


def test_bites_refuses_an_unconstrained_fixture() -> None:
    """A lever that moved is still no proof when nothing was scarce."""
    with pytest.raises(AssertionError, match="UNCONSTRAINED"):
        assert_lever_bites(_probe([3, 7], binds=NEVER_BINDS))


def test_inert_refuses_an_unconstrained_fixture() -> None:
    """The inert pin is the direction that passes vacuously, so guard it hardest."""
    with pytest.raises(AssertionError, match="UNCONSTRAINED"):
        assert_lever_is_inert(_probe([5, 5], binds=NEVER_BINDS), because="measured")


def test_unconstrained_message_names_the_offending_settings() -> None:
    """The refusal must say WHICH runs were unconstrained, or it cannot be fixed."""
    with pytest.raises(AssertionError, match=r"settings \[0, 1\]"):
        assert_lever_bites(_probe([3, 7], binds=NEVER_BINDS))


# ---------------------------------------------------------------------------
# The two verdicts.
# ---------------------------------------------------------------------------


def test_bites_fails_on_an_inert_lever() -> None:
    """Identical output across a binding range is the defect this class names."""
    with pytest.raises(AssertionError, match="INERT LEVER"):
        assert_lever_bites(_probe([5, 5, 5]))


def test_bites_passes_on_a_live_lever() -> None:
    assert_lever_bites(_probe([5, 7]))


def test_inert_pin_fails_when_the_lever_starts_working() -> None:
    """The alarm: a pinned-inert lever that moves is an unmeasured money change."""
    with pytest.raises(AssertionError, match="pinned inert"):
        assert_lever_is_inert(_probe([5, 7]), because="measured reason")


def test_inert_pin_passes_while_the_lever_stays_inert() -> None:
    assert_lever_is_inert(_probe([5, 5]), because="measured reason")


# ---------------------------------------------------------------------------
# Input contracts.
# ---------------------------------------------------------------------------


def test_a_single_setting_is_not_a_range() -> None:
    with pytest.raises(ValueError, match="at least two"):
        probe_lever(name="fake", run=lambda s: s, settings=[1], binds=BINDS)


def test_the_inert_pin_demands_a_reason() -> None:
    """'Because' is the measured reason; an empty one is a guess in disguise."""
    with pytest.raises(ValueError, match="measured reason"):
        assert_lever_is_inert(_probe([5, 5]), because="   ")


def test_distinct_outputs_counts_unhashable_results() -> None:
    """Plans are dicts; the probe must compare them by value, not by hash."""
    probe = _probe([{"a": 1}, {"a": 1}, {"a": 2}], binds=lambda output: True)
    assert probe.distinct_outputs == 2
    assert probe.moved
