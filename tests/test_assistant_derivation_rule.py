"""Refusing a figure you can derive is a worse answer, not a safer one.

A first-run study asked Kai what an advertiser was worth before and after a
price change. It gave the gross on both sides and then declined to state the
net for the earlier one, saying no tool had returned it - while holding that
gross and the agency's rebate percent, which is one multiplication away. The
operator noticed and called it the right mistake to make. It is still a
mistake: the person asked for net, and honesty about provenance is supposed to
make an answer trustworthy, not absent.

The rule this pins draws the line where it actually falls. Arithmetic on
figures read this turn is allowed and must be labelled as computed here, naming
the operation and its inputs. Anything that needs an assumption the data does
not carry - projecting a day onto a month, picking between two rebate bases an
agreement never settled - is not arithmetic and stays refused, with the missing
assumption named. Both halves are asserted, because a rule that only permitted
would have traded one failure for a worse one.
"""

from __future__ import annotations

import inspect

from kairos_api import assistant_prompt


def _prompt_text() -> str:
    return inspect.getsource(assistant_prompt)


def test_the_rule_permits_deriving_from_figures_read_this_turn():
    text = _prompt_text()
    assert "Deriving a figure no tool returned" in text
    assert "compute it and give it" in text
    assert "Refusing a figure you can derive is not honesty" in text


def test_the_derivation_must_be_labelled_with_its_operation_and_inputs():
    text = _prompt_text()
    assert "computed here rather than read from a tool" in text
    assert "name the operation and the source figures" in text


def test_the_inputs_must_be_read_this_turn_and_not_remembered():
    """A figure carried from an earlier turn may describe a state that has since
    changed, which is the defect rule 18 already exists to prevent."""
    text = _prompt_text()
    assert "never one remembered, assumed or carried from an earlier turn" in text


def test_an_assumption_the_data_does_not_carry_is_still_refused():
    text = _prompt_text()
    for missing in (
        "projecting one broadcast day onto a month",
        "choosing between two rebate bases",
        "treating a partial period as a whole one",
    ):
        assert missing in text, f"the rule must name {missing!r} as NOT arithmetic"
    assert "Name the assumption that is missing" in text


def test_the_competitor_boundary_is_not_weakened_by_it():
    """Deriving is about the operator's own figures. A rival's revenue is not
    derivable, because none of its inputs exist here at all."""
    text = _prompt_text()
    assert "never propose or discuss actions on another channel" in text
    assert "only as aggregate counts, never by name or by figure" in text
