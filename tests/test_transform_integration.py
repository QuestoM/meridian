"""Real-data integration smoke for the full transform-to-optimizer pipeline.

This reads the actual Programmes.xlsx, so it is slower than the unit tests and is
excluded from the fast gate (run it explicitly, like test_loaders.py). It proves
the pipeline runs end to end on the real grid and returns a compliant schedule.
"""

from __future__ import annotations

import pytest

from kairos.data import ProgramClassifier
from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import PricingModel


def _pick_channel_day(programmes):
    row = programmes[programmes["start_dt"].notna()].sort_values("start_dt").iloc[0]
    return str(row["Channel"]), row["start_dt"].strftime("%Y-%m-%d")


def test_real_programmes_optimize_to_compliant_schedule() -> None:
    programmes = load_programmes()
    classifier = ProgramClassifier.from_yaml()
    pricing = PricingModel.from_yaml()
    channel, day = _pick_channel_day(programmes)

    segments = build_segments_from_programmes(
        programmes, classifier, pricing, channel=channel, day=day,
    )
    assert segments, f"no segments built for {channel} on {day}"
    assert all(s.channel == channel for s in segments)
    assert all(s.baseline_tvr >= 0 for s in segments)
    assert all(s.cpp == pricing.base_price for s in segments)

    result = optimize_breaks(segments, Guardrails(), revenue_weight=0.6)
    assert result.is_compliant
    assert result.total_revenue >= 0
    assert result.total_breaks == len(result.placements)
    # The decision trail explains every break it placed.
    assert len(result.decisions) == result.total_breaks


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
