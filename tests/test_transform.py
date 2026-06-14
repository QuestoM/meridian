"""Tests for the programmes-to-segments transform.

These use small synthetic frames with the real classifier and pricing model, so
they stay fast (no xlsx) while proving the bridge end to end, including the
positional pricing-class sequence and the optimizer integration.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.data.transform import build_segments_from_programmes
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.optimizer import optimize_breaks
from kairos.optimize.pricing import OptimizerAssumptions, PricingModel


@pytest.fixture(scope="module")
def pricing() -> PricingModel:
    return PricingModel.from_yaml()


def make_frame() -> pd.DataFrame:
    # 2024-11-04 is a Monday (day premium 1.0), so each segment premium equals
    # its program-type premium, which makes the pricing class easy to assert.
    rows = [
        ("חדשות הערב", "קשת 12", "20:00:00", 3600, 5.0),
        ("התוכנית הראשונה", "קשת 12", "21:00:00", 3600, 5.0),
        ("התוכנית השנייה", "קשת 12", "22:00:00", 3600, 5.0),
        ("התוכנית השלישית", "קשת 12", "23:00:00", 3600, 5.0),
        ("תוכנית ערוץ אחר", "רשת 13", "21:00:00", 3600, 4.0),
    ]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "time", "Duration", "TVR"])
    frame["start_dt"] = pd.to_datetime("2024-11-04 " + frame["time"])
    return frame


def test_pricing_class_sequence_after_news(pricing: PricingModel) -> None:
    from kairos.data import ProgramClassifier  # local import keeps the fixture lean

    classifier = ProgramClassifier.from_yaml()
    segments = build_segments_from_programmes(
        make_frame(), classifier, pricing, channel="קשת 12",
    )
    premiums = [round(s.premium, 4) for s in segments]
    # News, then PrimeShow1, PrimeShow2, then Other (Monday day premium is 1.0).
    assert premiums == [1.15, 1.10, 1.00, 0.80]


def test_segment_fields_come_from_data_and_assumptions(pricing: PricingModel) -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    assumptions = OptimizerAssumptions()
    segments = build_segments_from_programmes(
        make_frame(), classifier, pricing, channel="קשת 12", assumptions=assumptions,
    )
    news = segments[0]
    assert news.program_type == "News"
    assert news.baseline_tvr == 5.0              # real rating from the data
    assert news.cpp == pricing.base_price        # 60, per second
    assert news.unit_seconds == 1.0
    assert news.impact_coefficient == assumptions.retention_impact_per_break
    assert news.retention_baseline == assumptions.retention_baseline
    assert news.max_breaks == assumptions.default_max_breaks
    assert news.start_seconds == 20 * 3600       # 20:00 from midnight


def test_channel_filter_excludes_other_channels(pricing: PricingModel) -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    segments = build_segments_from_programmes(
        make_frame(), classifier, pricing, channel="קשת 12",
    )
    assert {s.channel for s in segments} == {"קשת 12"}
    assert len(segments) == 4


def test_day_filter(pricing: PricingModel) -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    segments = build_segments_from_programmes(
        make_frame(), classifier, pricing, channel="קשת 12", day="2024-11-04",
    )
    assert len(segments) == 4
    assert build_segments_from_programmes(
        make_frame(), classifier, pricing, channel="קשת 12", day="2024-11-05",
    ) == []


def test_nonpositive_duration_and_missing_tvr_are_handled(pricing: PricingModel) -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    frame = make_frame()
    frame.loc[1, "Duration"] = 0          # dropped
    frame.loc[2, "TVR"] = float("nan")    # kept, baseline_tvr falls to 0.0
    segments = build_segments_from_programmes(frame, classifier, pricing, channel="קשת 12")
    assert len(segments) == 3             # one of four dropped
    zero_tvr = [s for s in segments if s.baseline_tvr == 0.0]
    assert len(zero_tvr) == 1


def test_segments_optimize_to_a_compliant_schedule(pricing: PricingModel) -> None:
    from kairos.data import ProgramClassifier

    classifier = ProgramClassifier.from_yaml()
    segments = build_segments_from_programmes(make_frame(), classifier, pricing)
    result = optimize_breaks(segments, Guardrails(), revenue_weight=1.0)
    assert result.is_compliant
    assert result.total_breaks > 0
    assert result.total_revenue > 0
