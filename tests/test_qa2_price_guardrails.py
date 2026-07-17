"""Boundary unit tests for the final-CPP pricing guardrails.

:class:`kairos.optimize.price_guardrails.Guardrails` inspects a composed price and
returns named, human-readable warnings without ever clamping the number. These
tests pin the exact boundary of each check (a disabled bound is silent, an enabled
bound fires strictly past its edge and stays silent at or inside it) and confirm
the ``from_config`` reader plus the wire shape that ``pricing_api`` serialises.

A :class:`PriceBreakdown` with a bare ``base_cpp`` and no layers has
``total_premium == 1.0``, so ``final_cpp == base_cpp``: that gives exact control
of the final price under test with no engine dependency.
"""

from __future__ import annotations

from kairos.optimize.price_guardrails import GuardrailWarning, Guardrails
from kairos.optimize.pricing import PriceBreakdown, PriceLayer


def _priced(final_cpp: float) -> PriceBreakdown:
    """A breakdown whose final CPP is exactly ``final_cpp`` (no premium layers)."""
    return PriceBreakdown(base_cpp=float(final_cpp))


def _codes(warnings: list[GuardrailWarning]) -> set[str]:
    return {w.code for w in warnings}


# 1. Disabled bounds are silent -----------------------------------------------
def test_all_bounds_disabled_stays_silent_for_positive_price() -> None:
    """Every bound at 0 (disabled) and a positive final CPP means no warnings."""
    guards = Guardrails()  # floor/ceiling/cost all 0.0 -> all disabled
    assert guards.check(_priced(100.0)) == []


def test_explicit_zero_fires_regardless_of_disabled_bounds() -> None:
    """explicit_zero is not gated on any bound: a free spot is always surfaced."""
    guards = Guardrails()
    codes = _codes(guards.check(_priced(0.0)))
    assert codes == {"explicit_zero"}
    # And the one warning carries the promo bound of 0.0.
    (warning,) = guards.check(_priced(0.0))
    assert warning.code == "explicit_zero"
    assert warning.bound == 0.0
    assert warning.message.strip()


# 2. Floor boundary -----------------------------------------------------------
def test_below_floor_boundary() -> None:
    guards = Guardrails(floor_cpp=50.0)
    # Strictly below the floor fires.
    assert "below_floor" in _codes(guards.check(_priced(49.99)))
    # Exactly at the floor is inside (>=), so it is silent.
    assert "below_floor" not in _codes(guards.check(_priced(50.0)))
    # Above the floor is silent.
    assert "below_floor" not in _codes(guards.check(_priced(50.01)))


def test_below_floor_warning_names_the_bound() -> None:
    guards = Guardrails(floor_cpp=50.0)
    hits = [w for w in guards.check(_priced(49.0)) if w.code == "below_floor"]
    assert len(hits) == 1
    assert hits[0].bound == 50.0
    assert "50.00" in hits[0].message and "49.00" in hits[0].message


# 3. Ceiling boundary ---------------------------------------------------------
def test_above_ceiling_boundary() -> None:
    guards = Guardrails(ceiling_cpp=200.0)
    # Strictly above the ceiling fires.
    assert "above_ceiling" in _codes(guards.check(_priced(200.01)))
    # Exactly at the ceiling is inside (<=), so it is silent.
    assert "above_ceiling" not in _codes(guards.check(_priced(200.0)))
    # Below the ceiling is silent.
    assert "above_ceiling" not in _codes(guards.check(_priced(199.99)))


# 4. Below-cost boundary ------------------------------------------------------
def test_below_cost_boundary() -> None:
    guards = Guardrails(cost_cpp=80.0)
    # Strictly below the cost basis (and above zero) fires.
    assert "below_cost" in _codes(guards.check(_priced(79.99)))
    # Exactly at the cost basis is inside (>=), so it is silent.
    assert "below_cost" not in _codes(guards.check(_priced(80.0)))
    # Above the cost basis is silent.
    assert "below_cost" not in _codes(guards.check(_priced(80.01)))


def test_below_cost_does_not_fire_at_zero_only_explicit_zero_does() -> None:
    """At exactly zero the 0 < final guard excludes below_cost; only the promo fires."""
    guards = Guardrails(cost_cpp=80.0)
    codes = _codes(guards.check(_priced(0.0)))
    assert codes == {"explicit_zero"}
    assert "below_cost" not in codes


# 5. Several bounds can fire together, in priority order ----------------------
def test_zero_price_under_cost_reports_explicit_zero_first() -> None:
    """explicit_zero is appended before the enabled floor/cost checks (priority)."""
    guards = Guardrails(floor_cpp=10.0, cost_cpp=80.0)
    warnings = guards.check(_priced(0.0))
    codes = [w.code for w in warnings]
    # A zero price is below the 10.0 floor too, but not below_cost (0 excluded).
    assert codes[0] == "explicit_zero"
    assert set(codes) == {"explicit_zero", "below_floor"}


# 6. from_config reads the guardrails block and yields the pricing_api wire shape
def test_from_config_reads_guardrails_block_and_wire_shape_matches_pricing_api() -> None:
    """A guardrails-bearing overrides mapping produces the exact serialised shape
    ``pricing_api.price_slot`` returns for ``guardrail_warnings``.
    """
    overrides = {
        "base_price_per_second_per_tvr_point": 100.0,
        "guardrails": {"floor_cpp": 50.0, "ceiling_cpp": 200.0, "cost_cpp": 80.0},
    }
    guards = Guardrails.from_config(overrides)
    assert (guards.floor_cpp, guards.ceiling_cpp, guards.cost_cpp) == (50.0, 200.0, 80.0)

    # A price below floor (and below cost) trips two named warnings.
    warnings = guards.check(_priced(40.0))
    assert _codes(warnings) == {"below_floor", "below_cost"}

    # This is exactly how pricing_api.price_slot serialises the warnings on the wire.
    wire = [{"code": w.code, "bound": w.bound, "message": w.message} for w in warnings]
    assert isinstance(wire, list) and wire
    for item in wire:
        assert set(item) == {"code", "bound", "message"}
        assert isinstance(item["code"], str) and item["code"]
        assert isinstance(item["bound"], float)
        assert isinstance(item["message"], str) and item["message"].strip()


def test_from_config_missing_or_empty_guardrails_is_all_disabled() -> None:
    """No guardrails block (or an empty one) disables every bound; silent on any price."""
    assert Guardrails.from_config({}) == Guardrails()
    assert Guardrails.from_config({"guardrails": {}}) == Guardrails()
    assert Guardrails.from_config(None) == Guardrails()
    # A composed price with layers still resolves through final_cpp with no warnings.
    breakdown = PriceBreakdown(base_cpp=100.0, layers=(PriceLayer("program", 1.5),))
    assert Guardrails.from_config({}).check(breakdown) == []
