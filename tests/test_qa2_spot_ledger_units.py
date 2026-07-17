"""The daily spot ledger prices on the engine's per-second basis.

base_price_per_second_per_tvr_point is quoted PER SECOND (config yaml line one,
and the weekly segments in kairos.data.transform pass unit_seconds=1.0). The
spot ledger used break_revenue's default 30-second unit, charging every CPP
spot 1/30th of the engine's own basis. These tests pin the per-second contract.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.export.spots import price_daily_spots
from kairos.optimize._frequency_rules import FrequencyRuleSet
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.objective import break_revenue
from kairos.optimize.pricing import PricingModel


def _daily_row(**overrides):
    row = {
        "advertiser": "BRAND",
        "campaign": "C",
        "program": "Show",
        "position_in_break": 2,
        "planned_tvr": 5.0,
        "duration_sec": 30.0,
        "pricing_type": "CPP",
        "price": None,
        "spot_time": "21:00:00",
    }
    row.update(overrides)
    return row


def _neutral_engine() -> AdvertiserRuleEngine:
    # No baselines and no conditions: effective premium 1.0, nothing dropped.
    return AdvertiserRuleEngine(baselines={}, conditions={})


def test_known_cpp_spot_prices_on_the_per_second_basis():
    """10 ILS/second/point x 5.0 TVR x 30 seconds x premium 1.0 = 1500 ILS.
    The old 30-second-unit default returned 50 ILS for the same spot."""
    pricing = PricingModel(base_price_per_second_per_tvr_point=10.0)
    daily = pd.DataFrame([_daily_row()])
    result = price_daily_spots(
        daily, engine=_neutral_engine(), pricing=pricing, frequency=FrequencyRuleSet()
    )
    assert len(result.priced) == 1
    spot = result.priced[0]
    assert spot.revenue == pytest.approx(1500.0)
    assert spot.placement_value == pytest.approx(1500.0)


def test_spot_ledger_matches_the_weekly_segment_price_identity():
    """The ledger and the weekly segments must value the same seconds the same
    way: spot revenue equals break_revenue(..., unit_seconds=1.0) exactly."""
    base = 60.0
    tvr = 2.5
    duration = 45.0
    pricing = PricingModel(base_price_per_second_per_tvr_point=base)
    daily = pd.DataFrame([_daily_row(planned_tvr=tvr, duration_sec=duration)])
    result = price_daily_spots(
        daily, engine=_neutral_engine(), pricing=pricing, frequency=FrequencyRuleSet()
    )
    expected = break_revenue(tvr, duration, base, unit_seconds=1.0)
    assert result.priced[0].revenue == pytest.approx(round(expected, 2))
    # And the old basis is measurably different, so this test cannot pass by luck.
    old_basis = break_revenue(tvr, duration, base)
    assert expected == pytest.approx(old_basis * 30.0)


def test_fixed_price_spots_are_unaffected_by_the_unit_change():
    pricing = PricingModel(base_price_per_second_per_tvr_point=10.0)
    daily = pd.DataFrame([_daily_row(pricing_type="FIX", price=777.0)])
    result = price_daily_spots(
        daily, engine=_neutral_engine(), pricing=pricing, frequency=FrequencyRuleSet()
    )
    assert result.priced[0].revenue == pytest.approx(777.0)


def test_module_docstring_no_longer_claims_thirty_second_units():
    import kairos.export.spots as spots_module

    assert "30-second units" not in (spots_module.__doc__ or "")
    assert "per-second" in (spots_module.__doc__ or "")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
