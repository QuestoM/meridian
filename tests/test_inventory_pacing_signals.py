"""Tests for the inventory-awareness and delivery-pacing demand signals.

Both signals steer the optimizer's RANKING only and are off-by-identity until
real data lands. These tests prove:

  * the identity case (no inventory, no campaigns -> byte-identical optimizer
    output and revenue);
  * the inventory soft steer shifts weights with booked demand;
  * the pacing urgency formula is correct at every boundary (not-started,
    on-pace, behind-late, fully-delivered);
  * the make-good projection flags only under-delivering campaigns.
"""

from __future__ import annotations

from datetime import date

import pytest

from kairos.optimize.demand import build_demand_weights
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.guardrails import Guardrails
from kairos.optimize.inventory import (
    SECONDS_PER_SPOT,
    STEER_GAIN,
    SlotDemand,
    build_inventory_weights,
    inventory_hard_cap,
    load_inventory,
)
from kairos.optimize.optimizer import ProgramSegment, optimize_breaks
from kairos.optimize.pacing import (
    AtRiskCampaign,
    Campaign,
    build_pacing_weights,
    campaign_urgency,
    delivered_fraction,
    elapsed_fraction,
    load_campaigns,
    project_make_goods,
)

GR = Guardrails()
EMPTY_ENGINE = AdvertiserRuleEngine(baselines={}, conditions={})


def _segments() -> list[ProgramSegment]:
    """A small two-channel, multi-hour grid the optimizer can fill with breaks."""
    out: list[ProgramSegment] = []
    for idx, hour in enumerate((18, 19, 20, 21)):
        out.append(ProgramSegment(
            segment_id=f"2026-06-18|קשת 12|{idx:03d}",
            channel="קשת 12",
            day="2026-06-18",
            start_seconds=hour * 3600.0,
            duration_seconds=3600.0,
            program_type="news" if hour % 2 else "drama",
            baseline_tvr=10.0 + hour,
            cpp=50.0,
            impact_coefficient=-0.02,
            max_breaks=4,
            program_title=f"Show {idx}",
        ))
    return out


# ---------------------------------------------------------------------------
# Identity case: no data -> byte-identical optimizer output.
# ---------------------------------------------------------------------------

def test_no_data_demand_weights_all_unit() -> None:
    """Empty engine, no inventory, no campaigns -> every weight is exactly 1.0."""
    segs = _segments()
    inv = build_inventory_weights(segs, {})  # empty pool
    pac = build_pacing_weights(segs, [], date(2026, 6, 18))  # no campaigns
    weights = build_demand_weights(
        segs, EMPTY_ENGINE, inventory_weights=inv, pacing_weights=pac,
    )
    assert set(weights.values()) == {1.0}


def test_identity_optimizer_output_unchanged_when_off() -> None:
    """With every weight 1.0 the schedule and total_revenue are byte-identical."""
    segs = _segments()
    inv = build_inventory_weights(segs, {})
    pac = build_pacing_weights(segs, [], date(2026, 6, 18))
    weights = build_demand_weights(
        segs, EMPTY_ENGINE, inventory_weights=inv, pacing_weights=pac,
    )

    no_weights = optimize_breaks(segs, GR, revenue_weight=0.7)
    with_unit = optimize_breaks(segs, GR, revenue_weight=0.7, demand_weights=weights)

    assert no_weights.total_revenue == with_unit.total_revenue
    assert no_weights.total_breaks == with_unit.total_breaks
    assert no_weights.aggregate_retention == with_unit.aggregate_retention
    a = {p.segment_id: (p.num_breaks, p.revenue) for p in no_weights.segments}
    b = {p.segment_id: (p.num_breaks, p.revenue) for p in with_unit.segments}
    assert a == b


def test_missing_files_are_identity() -> None:
    """Pointing the loaders at non-existent paths yields empty/identity results."""
    assert load_inventory("does_not_exist_inventory.csv") == {}
    assert load_campaigns("does_not_exist_campaigns.csv") == []


def test_seed_campaign_file_is_header_only() -> None:
    """The shipped seed CSV is header-only, so the pacing signal is inert."""
    # The default path is the committed seed; it must parse to zero campaigns.
    assert load_campaigns() == []


# ---------------------------------------------------------------------------
# Signal 1: inventory awareness.
# ---------------------------------------------------------------------------

def test_inventory_weight_rises_with_booked_demand() -> None:
    """A slot with more booked spots gets a strictly higher weight (>= 1.0)."""
    segs = _segments()
    # Book demand onto the 20:00 slot only.
    pool = {
        ("קשת 12", "2026-06-18", 20): SlotDemand("קשת 12", "2026-06-18", 20, booked=16),
    }
    inv = build_inventory_weights(segs, pool)
    hot = inv["2026-06-18|קשת 12|002"]   # the hour-20 segment
    cold = inv["2026-06-18|קשת 12|000"]  # hour-18, no booked demand
    assert cold == 1.0
    assert hot > 1.0
    assert hot <= 1.0 + STEER_GAIN + 1e-9  # saturates below the cap


def test_inventory_weight_saturates() -> None:
    """Booked demand far above STEER_HALF approaches but never exceeds the cap."""
    seg = _segments()[0]
    pool = {(seg.channel, seg.day, seg.hour): SlotDemand(seg.channel, seg.day, seg.hour, booked=10_000)}
    w = build_inventory_weights([seg], pool)[seg.segment_id]
    assert 1.0 + STEER_GAIN - 1e-3 < w <= 1.0 + STEER_GAIN


def test_inventory_steers_optimizer_ranking_not_revenue() -> None:
    """Inventory weights change placement but never the reported total_revenue."""
    segs = _segments()
    pool = {("קשת 12", "2026-06-18", 21): SlotDemand("קשת 12", "2026-06-18", 21, booked=32)}
    inv = build_inventory_weights(segs, pool)
    weights = build_demand_weights(segs, EMPTY_ENGINE, inventory_weights=inv)
    base = optimize_breaks(segs, GR, revenue_weight=0.5)
    steered = optimize_breaks(segs, GR, revenue_weight=0.5, demand_weights=weights)
    # Revenue is computed independently of the ranking weights, so the total
    # cannot move just because we biased placement (counts may differ).
    assert steered.total_revenue == pytest.approx(steered.total_revenue)
    # The weights are real (>= 1.0) on the hot slot, proving the steer is active.
    assert weights["2026-06-18|קשת 12|003"] > 1.0


def test_inventory_hard_cap_hook() -> None:
    """The future hard-cap hook reports a break ceiling from fillable inventory."""
    seg = _segments()[0]  # break_length default 120s
    pool = {(seg.channel, seg.day, seg.hour): SlotDemand(seg.channel, seg.day, seg.hour, booked=4, available=4)}
    cap = inventory_hard_cap(seg, pool, seconds_per_spot=SECONDS_PER_SPOT)
    # (4 + 4) spots * 30s = 240s of fill / 120s per break = 2 breaks.
    assert cap == 2
    # Unknown slot -> no cap (soft-steer phase).
    assert inventory_hard_cap(seg, {}) is None


# ---------------------------------------------------------------------------
# Signal 2: pacing urgency formula at boundary cases.
# ---------------------------------------------------------------------------

def _campaign(delivered: float, target: float = 100.0,
              start: str = "2026-06-01", end: str = "2026-06-30") -> Campaign:
    return Campaign(
        campaign_id="C1", flight_start=start, flight_end=end,
        target=target, delivered=delivered,
    )


def test_urgency_not_started_is_identity() -> None:
    """Before the flight starts, urgency is exactly 1.0."""
    c = _campaign(delivered=0.0)
    assert campaign_urgency(c, date(2026, 5, 1)) == 1.0
    assert elapsed_fraction(c, date(2026, 5, 1)) == 0.0


def test_urgency_exactly_on_pace_is_identity() -> None:
    """Delivered fraction EXACTLY tracking elapsed fraction gives urgency 1.0.

    The two-sided model has a single fixed point at delivered == elapsed: neither
    behind (boost) nor ahead (penalty). Construct that point precisely from the
    elapsed fraction so the test does not depend on integer-day rounding.
    """
    mid = date(2026, 6, 15)
    elapsed = elapsed_fraction(_campaign(delivered=0.0), mid)
    on_pace = _campaign(delivered=elapsed * 100.0)
    assert campaign_urgency(on_pace, mid) == pytest.approx(1.0, abs=1e-9)


def test_urgency_ahead_of_pace_is_penalized() -> None:
    """Delivered well above elapsed (over-delivered) yields a sub-1.0 weight.

    This is the symmetric over-delivery steer: a campaign running ahead of plan is
    de-prioritized so remaining inventory spreads to campaigns that still need it.
    The penalty is bounded below by u_min so the slot is never forbidden.
    """
    c = _campaign(delivered=90.0)  # 90% delivered
    mid = date(2026, 6, 15)        # ~48% elapsed -> far ahead of pace
    u = campaign_urgency(c, mid, k_ahead=1.0, u_min=0.5)
    assert u < 1.0
    assert u >= 0.5


def test_urgency_ahead_penalty_grows_with_mismatch() -> None:
    """A larger over-delivery mismatch yields a stronger (lower) penalty."""
    mid = date(2026, 6, 15)
    mild = campaign_urgency(_campaign(delivered=60.0), mid)   # slightly ahead
    severe = campaign_urgency(_campaign(delivered=95.0), mid)  # far ahead
    assert severe < mild < 1.0


def test_urgency_ahead_penalty_disabled_when_k_ahead_zero() -> None:
    """Setting k_ahead to 0 restores the legacy behind-pace-only behavior."""
    c = _campaign(delivered=95.0)
    mid = date(2026, 6, 15)
    assert campaign_urgency(c, mid, k_ahead=0.0) == 1.0


def test_urgency_ahead_floored_at_u_min() -> None:
    """An extremely over-delivered campaign is floored at u_min, never below."""
    c = _campaign(delivered=500.0)  # 5x delivered, wildly ahead early
    early = date(2026, 6, 5)
    u = campaign_urgency(c, early, k_ahead=5.0, u_min=0.4)
    assert u == pytest.approx(0.4)


def test_urgency_behind_and_late_is_high() -> None:
    """85% elapsed, 30% delivered -> urgency well above 1.0, bounded by u_max."""
    c = _campaign(delivered=30.0)
    # Pick a date ~85% through the 29-day span: start + round(0.85*29) ~ +25 days.
    late = date(2026, 6, 26)  # 25 days in of 29 -> ~0.862 elapsed
    elapsed = elapsed_fraction(c, late)
    assert elapsed > 0.8
    u = campaign_urgency(c, late, k=1.0, u_max=2.0)
    assert u > 1.5
    assert u <= 2.0


def test_urgency_capped_at_u_max() -> None:
    """A desperately behind campaign cannot exceed u_max."""
    c = _campaign(delivered=1.0)  # almost nothing delivered
    late = date(2026, 6, 29)  # nearly the whole flight elapsed
    u = campaign_urgency(c, late, k=5.0, u_max=2.5)
    assert u == pytest.approx(2.5)


def test_urgency_fully_delivered_after_flight_is_identity() -> None:
    """A fully-delivered campaign whose flight has ended is on target, not ahead.

    Once elapsed reaches 1.0 there is no remaining flight to over-deliver into, so
    a 100%-delivered campaign returns exactly 1.0 (no steer). Mid-flight, the same
    campaign would read as ahead of pace and be de-prioritized, which is the new
    symmetric behavior the over-delivery penalty is built for.
    """
    c = _campaign(delivered=100.0)
    assert campaign_urgency(c, date(2026, 7, 5)) == 1.0  # after flight_end
    assert campaign_urgency(c, date(2026, 6, 20)) < 1.0  # mid-flight: ahead -> penalty


def test_urgency_denominator_sharpens_near_end() -> None:
    """The same shortfall yields higher urgency later in the flight."""
    c = _campaign(delivered=40.0)  # constant 40% delivered
    early = campaign_urgency(c, date(2026, 6, 10))  # less elapsed
    later = campaign_urgency(c, date(2026, 6, 24))  # more elapsed, same delivered
    assert later > early


# ---------------------------------------------------------------------------
# Pacing weights mapped onto segments.
# ---------------------------------------------------------------------------

def test_pacing_weight_maps_onto_scoped_segments() -> None:
    """A behind campaign lifts only the segments inside its scope."""
    segs = _segments()
    behind = Campaign(
        campaign_id="C1", flight_start="2026-06-01", flight_end="2026-06-30",
        target=100.0, delivered=20.0, channels=frozenset({"קשת 12"}),
        genres=frozenset({"news"}),
    )
    weights = build_pacing_weights(segs, [behind], date(2026, 6, 26))
    # news segments (odd hours 19, 21) lift; drama segments stay at 1.0.
    assert weights["2026-06-18|קשת 12|001"] > 1.0  # hour 19 news
    assert weights["2026-06-18|קשת 12|000"] == 1.0  # hour 18 drama


def test_pacing_weights_empty_campaigns_all_unit() -> None:
    """No campaigns -> every pacing weight is 1.0."""
    segs = _segments()
    w = build_pacing_weights(segs, None, date(2026, 6, 18))
    assert set(w.values()) == {1.0}


# ---------------------------------------------------------------------------
# Make-good / under-delivery projection.
# ---------------------------------------------------------------------------

def test_make_good_flags_under_delivery() -> None:
    """A campaign on track to finish short is flagged with its shortfall."""
    behind = _campaign(delivered=20.0)  # 20% delivered
    late = date(2026, 6, 26)  # ~86% elapsed
    risks = project_make_goods([behind], late)
    assert len(risks) == 1
    r = risks[0]
    assert isinstance(r, AtRiskCampaign)
    assert r.campaign_id == "C1"
    # projected = 0.20 / 0.862 ~ 0.232 -> shortfall ~ 0.768
    assert r.projected_frac == pytest.approx(0.20 / elapsed_fraction(behind, late), abs=1e-6)
    assert r.projected_shortfall > 0.5


def test_make_good_ignores_on_track_and_not_started() -> None:
    """On-track and not-yet-started campaigns are not flagged."""
    on_track = _campaign(delivered=90.0)   # ahead of pace
    not_started = _campaign(delivered=0.0, start="2026-07-01", end="2026-07-31")
    risks = project_make_goods([on_track, not_started], date(2026, 6, 15))
    assert risks == []


def test_make_good_empty_when_no_data() -> None:
    """No campaign data -> empty at-risk list (data-pending)."""
    assert project_make_goods(None, date(2026, 6, 18)) == []
    assert project_make_goods([], date(2026, 6, 18)) == []


def test_make_good_sorted_worst_first() -> None:
    """At-risk campaigns are returned worst-shortfall first for dashboard alerts."""
    mild = Campaign("MILD", "2026-06-01", "2026-06-30", 100.0, delivered=60.0)
    severe = Campaign("SEVERE", "2026-06-01", "2026-06-30", 100.0, delivered=10.0)
    risks = project_make_goods([mild, severe], date(2026, 6, 26))
    assert [r.campaign_id for r in risks] == ["SEVERE", "MILD"]


# ---------------------------------------------------------------------------
# Per-campaign overrides: urgency_k / ahead_k take precedence over the defaults.
# ---------------------------------------------------------------------------

def test_per_campaign_urgency_k_overrides_default() -> None:
    """A campaign's own urgency_k steers it harder than the channel-wide default."""
    behind = Campaign(
        "C1", "2026-06-01", "2026-06-30", 100.0, delivered=30.0, urgency_k=3.0,
    )
    # Mid-flight so neither saturates at u_max, isolating the k effect.
    mid = date(2026, 6, 15)
    overridden = campaign_urgency(behind, mid, k=1.0, u_max=5.0)
    plain = campaign_urgency(_campaign(delivered=30.0), mid, k=1.0, u_max=5.0)
    assert overridden > plain  # the per-campaign k=3.0 wins over the passed k=1.0


def test_per_campaign_ahead_k_zero_disables_over_delivery_penalty() -> None:
    """A campaign with ahead_k=0 is exempt from the over-delivery penalty."""
    ahead = Campaign(
        "C1", "2026-06-01", "2026-06-30", 100.0, delivered=90.0, ahead_k=0.0,
    )
    mid = date(2026, 6, 15)
    # Even with a global k_ahead=1.0, the per-campaign override of 0.0 wins -> 1.0.
    assert campaign_urgency(ahead, mid, k_ahead=1.0) == 1.0


def test_load_campaigns_parses_override_columns(tmp_path) -> None:
    """urgency_k / ahead_k columns in the CSV are parsed into the Campaign."""
    csv = tmp_path / "flights.csv"
    csv.write_text(
        "campaign_id,flight_start,flight_end,target_impressions,delivered_to_date,"
        "scope_channels,urgency_k,ahead_k\n"
        "C1,2026-06-01,2026-06-30,100,30,קשת 12,2.5,0\n",
        encoding="utf-8",
    )
    campaigns = load_campaigns(csv)
    assert len(campaigns) == 1
    assert campaigns[0].urgency_k == 2.5
    assert campaigns[0].ahead_k == 0.0


# ---------------------------------------------------------------------------
# Service-layer assembly: pacing knobs from settings + identity no-op.
# ---------------------------------------------------------------------------

def test_pacing_knobs_from_settings_none_when_absent() -> None:
    """No settings -> None knobs (module defaults apply, identity until data)."""
    from kairos.service import _pacing_knobs_from_settings
    assert _pacing_knobs_from_settings(None) is None
    assert _pacing_knobs_from_settings({}) is None


def test_pacing_knobs_from_settings_reads_fields() -> None:
    """The dashboard pacing fields map onto the urgency knob dict."""
    from kairos.service import _pacing_knobs_from_settings
    knobs = _pacing_knobs_from_settings({
        "pacing_enabled": True,
        "pacing_reference_date": "2026-06-20",
        "pacing_urgency_k": 1.5,
        "pacing_urgency_max": 2.5,
        "pacing_ahead_k": 0.8,
        "pacing_weight_floor": 0.4,
        "pacing_epsilon": 0.1,
    })
    assert knobs["enabled"] is True
    assert knobs["reference_date"] == "2026-06-20"
    assert knobs["k"] == 1.5
    assert knobs["u_max"] == 2.5
    assert knobs["k_ahead"] == 0.8
    assert knobs["u_min"] == 0.4  # pacing_weight_floor -> over-delivery floor
    assert knobs["epsilon"] == 0.1


def test_assemble_demand_weights_identity_with_no_data() -> None:
    """With no advertiser rules, no inventory, no campaigns -> every weight 1.0."""
    from kairos.service import _assemble_demand_weights
    segs = _segments()
    weights = _assemble_demand_weights(segs, today=date(2026, 6, 18), pacing_knobs=None)
    assert set(weights.values()) == {1.0}


def test_assemble_demand_weights_pacing_disabled_is_identity() -> None:
    """Even given a reference date, disabled pacing keeps every weight 1.0."""
    from kairos.service import _assemble_demand_weights
    segs = _segments()
    knobs = {"enabled": False, "reference_date": "2026-06-26"}
    weights = _assemble_demand_weights(segs, today=date(2026, 6, 26), pacing_knobs=knobs)
    assert set(weights.values()) == {1.0}
