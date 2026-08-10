"""Round-quarter-hour settlement billing: window math, fixtures, and the OFF gate.

Two things are proven here. First, the billed-points computation itself is
correct on hand-built fixtures (contained break, straddling break, shared
window, coverage fallback): every expected value is computed by hand in the
test body. Second, the activation gate is honest: with
``pricing_activation.qh_settlement`` off (the shipped default) the optimizer
result is the SAME OBJECT, so every shipped path is byte-identical, and with
the flag on only the revenue currency changes (counts, placements and
retention are untouched).
"""

from __future__ import annotations

import pytest

from kairos.optimize.objective import break_revenue, weighted_objective
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks
from kairos.optimize.pricing import PricingModel
from kairos.optimize.qh_billing import (
    MEASURED_DIP_BY_LENGTH,
    QH_WINDOW_SECONDS,
    QHSettlementConfigurationError,
    billed_points,
    break_window_spans,
    maybe_restate,
    measured_dip_fraction,
    qh_settlement_enabled,
    restate_on_billed_points,
    validate_qh_settlement_provenance,
    window_start_of,
)


def _segment(**kwargs) -> ProgramSegment:
    base = dict(
        segment_id="seg-a",
        channel="Test 1",
        day="2024-11-01",
        start_seconds=46800.0,        # 13:00
        duration_seconds=3600.0,
        program_type="Other",
        baseline_tvr=10.0,
        cpp=1000.0,
        impact_coefficient=-0.02,
        max_breaks=4,
        break_length_seconds=120.0,
        rating_audience_basis="jewish_households",
        rating_vintage="overnight_plus_1",
        rating_source="fixture://verified-ratings",
    )
    base.update(kwargs)
    return ProgramSegment(**base)


# --- window math ---------------------------------------------------------


def test_window_start_of() -> None:
    assert window_start_of(46800.0) == 46800.0          # 13:00 is its own window
    assert window_start_of(47399.0) == 46800.0          # 13:09:59 -> 13:00 window
    assert window_start_of(47700.0) == 47700.0          # 13:15
    assert window_start_of(0.0) == 0.0


def test_break_window_spans_contained_and_straddling() -> None:
    # Fully inside the 13:00 window.
    assert break_window_spans(47100.0, 120.0) == [(46800.0, 120.0)]
    # 13:14:00 - 13:18:00 straddles the 13:15 boundary: 60s + 180s.
    assert break_window_spans(47640.0, 240.0) == [(46800.0, 60.0), (47700.0, 180.0)]
    # Zero duration bills nothing.
    assert break_window_spans(47100.0, 0.0) == []
    # A break longer than a window touches three windows.
    spans = break_window_spans(46800.0 + 800.0, 1200.0)
    assert [w for w, _ in spans] == [46800.0, 47700.0, 48600.0]
    assert sum(s for _, s in spans) == pytest.approx(1200.0)


def test_measured_dip_fraction_bins() -> None:
    # Bin edges follow the measured Nov-2024 medians (settlement_results.json).
    assert measured_dip_fraction(30.0) == 0.0377
    assert measured_dip_fraction(90.0) == 0.0434
    assert measured_dip_fraction(150.0) == 0.0548
    assert measured_dip_fraction(200.0) == 0.0677
    assert measured_dip_fraction(300.0) == 0.0620
    assert measured_dip_fraction(400.0) == 0.0908
    assert measured_dip_fraction(700.0) == 0.2040
    assert measured_dip_fraction(0.0) == 0.0
    # The table is total (every positive length maps to a bin).
    assert MEASURED_DIP_BY_LENGTH[-1][0] == float("inf")


# --- billed points on hand-built fixtures --------------------------------


class _FakePlacement:
    """Minimal stand-in for BreakPlacement (only the fields billed_points reads)."""

    def __init__(self, segment_id, channel, day, start, duration, position=1, revenue=0.0):
        self.segment_id = segment_id
        self.channel = channel
        self.day = day
        self.start_seconds = start
        self.duration_seconds = duration
        self.position_in_segment = position
        self.revenue = revenue


def test_contained_break_billed_at_window_average() -> None:
    seg = _segment()
    # One 120s break at 13:05; exactly 2 minutes lands in the left-closed 2-3m bin.
    brk = _FakePlacement("seg-a", "Test 1", "2024-11-01", 47100.0, 120.0)
    report = billed_points([seg], [brk])
    dip = 0.0548  # a 120s break sits in the left-closed 2-3m bin
    expected_avg = (10.0 * 900.0 - dip * 10.0 * 120.0) / 900.0
    assert len(report.windows) == 1
    window = report.windows[0]
    assert window.window_start_seconds == 46800.0
    assert window.covered_seconds == pytest.approx(900.0)
    assert window.break_seconds == pytest.approx(120.0)
    assert window.average_tvr == pytest.approx(expected_avg)
    bill = report.breaks[0]
    assert bill.billed_tvr == pytest.approx(expected_avg)
    assert bill.billed_revenue == pytest.approx(
        break_revenue(expected_avg, 120.0, 1000.0)
    )
    # The window average sits between the diluted in-break level and content.
    assert 10.0 * (1 - dip) < bill.billed_tvr < 10.0


def test_straddling_break_bills_each_second_in_its_own_window() -> None:
    seg = _segment()
    # 240s break at 13:14: 60s in the 13:00 window, 180s in the 13:15 window.
    brk = _FakePlacement("seg-a", "Test 1", "2024-11-01", 47640.0, 240.0)
    report = billed_points([seg], [brk])
    dip = 0.0620  # a 240s break sits in the left-closed 4-6m bin
    avg1 = (10.0 * 900.0 - dip * 10.0 * 60.0) / 900.0
    avg2 = (10.0 * 900.0 - dip * 10.0 * 180.0) / 900.0
    expected = (avg1 * 60.0 + avg2 * 180.0) / 240.0
    assert report.breaks[0].billed_tvr == pytest.approx(expected)
    by_start = {w.window_start_seconds: w for w in report.windows}
    assert by_start[46800.0].average_tvr == pytest.approx(avg1)
    assert by_start[47700.0].average_tvr == pytest.approx(avg2)


def test_straddle_beats_containment_on_billed_points() -> None:
    # The measured directional fact: at equal length, an even straddle bills
    # higher than a contained placement because each window is less diluted.
    seg = _segment()
    contained = _FakePlacement("seg-a", "2", "2024-11-01", 47100.0, 240.0)
    contained.channel = "Test 1"
    straddled = _FakePlacement("seg-a", "Test 1", "2024-11-01", 47580.0, 240.0)  # 13:13-13:17
    r_contained = billed_points([seg], [contained])
    r_straddled = billed_points([seg], [straddled])
    assert r_straddled.breaks[0].billed_tvr > r_contained.breaks[0].billed_tvr


def test_shared_window_couples_co_window_breaks() -> None:
    seg = _segment()
    # Two 120s breaks inside the same 13:00 window: both dips land in ONE average.
    b1 = _FakePlacement("seg-a", "Test 1", "2024-11-01", 46980.0, 120.0, position=1)
    b2 = _FakePlacement("seg-a", "Test 1", "2024-11-01", 47340.0, 120.0, position=2)
    report = billed_points([seg], [b1, b2])
    dip = 0.0548  # 120s breaks: left-closed 2-3m bin
    expected_avg = (10.0 * 900.0 - dip * 10.0 * 240.0) / 900.0
    assert len(report.windows) == 1
    assert report.windows[0].break_seconds == pytest.approx(240.0)
    for bill in report.breaks:
        assert bill.billed_tvr == pytest.approx(expected_avg)
    # Each break bills LOWER than it would alone (the other break's dip is in
    # its window average): the shared-window coupling, expressed.
    alone = billed_points([seg], [b1])
    assert report.breaks[0].billed_tvr < alone.breaks[0].billed_tvr


def test_cross_segment_content_enters_the_shared_window() -> None:
    # Window 13:00 covered half by seg-a (tvr 10) and half by seg-b (tvr 20):
    # the settlement average blends both programmes' content.
    seg_a = _segment(duration_seconds=450.0)
    seg_b = _segment(segment_id="seg-b", start_seconds=47250.0, duration_seconds=450.0,
                     baseline_tvr=20.0)
    brk = _FakePlacement("seg-a", "Test 1", "2024-11-01", 46900.0, 120.0)
    report = billed_points([seg_a, seg_b], [brk])
    dip = 0.0548
    expected_avg = (10.0 * 450.0 + 20.0 * 450.0 - dip * 10.0 * 120.0) / 900.0
    assert report.windows[0].average_tvr == pytest.approx(expected_avg)


def test_uncovered_window_falls_back_to_own_diluted_level() -> None:
    # A break placed outside any segment span: no lineup to average, so it
    # bills at its own diluted level instead of a fabricated window figure.
    seg = _segment()
    brk = _FakePlacement("seg-a", "Test 1", "2024-11-01", 86400.0 + 900.0, 120.0)
    report = billed_points([seg], [brk])
    assert report.breaks[0].billed_tvr == pytest.approx(10.0 * (1 - 0.0548))
    assert report.windows[0].covered_seconds == 0.0


def test_unknown_segment_raises() -> None:
    seg = _segment()
    brk = _FakePlacement("nope", "Test 1", "2024-11-01", 47100.0, 120.0)
    with pytest.raises(ValueError):
        billed_points([seg], [brk])


# --- the owner gate ------------------------------------------------------


def _pricing(qh_on: bool) -> PricingModel:
    return PricingModel.from_weights({
        "base_price_per_second_per_tvr_point": 60.0,
        "pricing_activation": {
            "qh_settlement": qh_on,
            "qh_audience_basis": "jewish_households",
            "qh_rating_vintage": "overnight_plus_1",
            "qh_rating_source": "fixture://verified-ratings",
        },
    })


def _two_segments() -> list[ProgramSegment]:
    return [
        _segment(segment_id="seg-a"),
        _segment(segment_id="seg-b", start_seconds=50400.0, baseline_tvr=6.0),
    ]


def test_flag_defaults_off_from_config() -> None:
    assert PricingModel.from_weights({}).enable_qh_settlement is False
    assert qh_settlement_enabled(None) is False
    assert qh_settlement_enabled(_pricing(False)) is False
    assert qh_settlement_enabled(_pricing(True)) is True


def test_requested_flag_without_currency_provenance_is_not_enabled() -> None:
    pricing = PricingModel.from_weights({
        "pricing_activation": {"qh_settlement": True},
    })
    assert pricing.enable_qh_settlement is True
    assert qh_settlement_enabled(pricing) is False
    segments = _two_segments()
    result = optimize_breaks(segments, revenue_weight=0.6)
    with pytest.raises(QHSettlementConfigurationError, match="not proven"):
        maybe_restate(result, segments, pricing)


def test_config_claim_cannot_substitute_for_segment_provenance() -> None:
    segments = _two_segments()
    segments[0] = _segment(rating_audience_basis="", rating_vintage="", rating_source="")
    result = optimize_breaks(segments, revenue_weight=0.6)
    with pytest.raises(QHSettlementConfigurationError, match="segment 'seg-a'"):
        maybe_restate(result, segments, _pricing(True))


def test_segment_source_must_match_the_configured_provenance() -> None:
    segments = _two_segments()
    segments[0] = _segment(rating_source="file://different-ratings.csv")
    result = optimize_breaks(segments, revenue_weight=0.6)
    with pytest.raises(QHSettlementConfigurationError, match="does not match"):
        maybe_restate(result, segments, _pricing(True))


def test_pricing_state_does_not_call_config_only_effective(monkeypatch) -> None:
    import kairos_api.preview_inputs as preview_module
    from kairos_api.pricing_api import _qh_activation

    unproven = _two_segments()
    unproven[0] = _segment(rating_audience_basis="", rating_vintage="", rating_source="")
    monkeypatch.setattr(preview_module, "preview_inputs", lambda *_args: (unproven, {}))
    blocked = _qh_activation(_pricing(True))
    assert blocked["qh_settlement_requested"] is True
    assert blocked["qh_settlement"] is False
    assert blocked["qh_data_provenance_valid"] is False

    monkeypatch.setattr(preview_module, "preview_inputs", lambda *_args: (_two_segments(), {}))
    enabled = _qh_activation(_pricing(True))
    assert enabled["qh_settlement"] is True
    assert enabled["qh_data_provenance_valid"] is True


def test_requested_qh_refuses_an_empty_rating_dataset() -> None:
    with pytest.raises(QHSettlementConfigurationError, match="no billed segments"):
        validate_qh_settlement_provenance(_pricing(True), [])


def test_pricing_write_refuses_activation_against_unproven_current_data(monkeypatch) -> None:
    from fastapi import HTTPException
    import kairos_api.preview_inputs as preview_module
    import kairos_api.pricing_api as pricing_api

    class _Settings:
        pricing_overrides = {}

    unproven = _two_segments()
    unproven[0] = _segment(rating_audience_basis="", rating_vintage="", rating_source="")
    monkeypatch.setattr(preview_module, "preview_inputs", lambda *_args: (unproven, {}))
    monkeypatch.setattr(
        pricing_api,
        "_settings_io",
        lambda: (lambda: _Settings(), lambda _settings: pytest.fail("invalid activation was saved")),
    )
    update = pricing_api.PricingUpdate(overrides={
        "pricing_activation": {
            "qh_settlement": True,
            "qh_audience_basis": "jewish_households",
            "qh_rating_vintage": "overnight_plus_1",
            "qh_rating_source": "fixture://verified-ratings",
        }
    })
    with pytest.raises(HTTPException) as caught:
        pricing_api.put_pricing(update)
    assert caught.value.status_code == 422
    assert "not proven" in str(caught.value.detail)


def test_flag_off_returns_the_same_object() -> None:
    segments = _two_segments()
    result = optimize_breaks(segments, revenue_weight=0.6)
    assert maybe_restate(result, segments, None) is result
    assert maybe_restate(result, segments, _pricing(False)) is result


def test_flag_on_restates_revenue_only() -> None:
    segments = _two_segments()
    result = optimize_breaks(segments, revenue_weight=0.6)
    restated = maybe_restate(result, segments, _pricing(True))
    assert restated is not result
    # The decision is untouched: same counts, positions, retention, guardrails.
    assert restated.total_breaks == result.total_breaks
    assert [p.num_breaks for p in restated.segments] == [p.num_breaks for p in result.segments]
    assert [p.start_seconds for p in restated.placements] == [p.start_seconds for p in result.placements]
    assert restated.aggregate_retention == result.aggregate_retention
    assert restated.violations == result.violations
    assert restated.revenue_scale == result.revenue_scale
    assert restated.revenue_basis == "round_quarter_hour_rating_points"
    assert restated.rating_audience_basis == "jewish_households"
    assert restated.rating_vintage == "overnight_plus_1"
    assert restated.rating_source == "fixture://verified-ratings"
    # Only the currency changed, and coherently: totals equal the sum of parts,
    # and the objective is the same blend recomputed on the restated revenue.
    assert restated.total_revenue != result.total_revenue
    assert restated.total_revenue == pytest.approx(sum(p.revenue for p in restated.placements))
    assert restated.objective == pytest.approx(weighted_objective(
        restated.total_revenue, restated.aggregate_retention,
        revenue_weight=restated.revenue_weight, revenue_scale=restated.revenue_scale,
    ))


def test_restated_break_matches_billed_points_report() -> None:
    segments = _two_segments()
    result = optimize_breaks(segments, revenue_weight=0.6)
    report = billed_points(segments, result.placements)
    restated = restate_on_billed_points(result, segments)
    by_key = {(b.segment_id, b.position_in_segment): b for b in report.breaks}
    for placement in restated.placements:
        bill = by_key[(placement.segment_id, placement.position_in_segment)]
        assert placement.revenue == pytest.approx(bill.billed_revenue)
    assert report.engine_revenue == pytest.approx(result.total_revenue)
    assert report.billed_revenue == pytest.approx(restated.total_revenue)


def test_window_constant_is_a_round_quarter_hour() -> None:
    assert QH_WINDOW_SECONDS == 900.0
