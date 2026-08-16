"""The obligations engine: standings are floors, alarms come from pace, and a
guarantee in a currency the product does not hold reports UNKNOWN instead of
comparing two different things."""

from datetime import date

import pandas as pd

from kairos.trade import obligations
from kairos.trade.obligations import Inputs


def _delivery(rows):
    return pd.DataFrame(rows, columns=[
        "campaign_id", "broadcast_date", "air_state", "channel", "spots",
        "seconds", "rating_points_planned", "spend_ils", "counted_as_of",
    ])


def _campaigns(rows):
    return pd.DataFrame(rows, columns=["campaign_id", "advertiser"])


def _links(rows):
    return pd.DataFrame(rows, columns=["agency_id", "agency_name", "advertiser"])


def _inputs(delivery, campaigns=None, links=None, today=date(2026, 6, 30), preferred=None):
    return Inputs(
        delivery=delivery,
        campaigns=_campaigns(campaigns or []),
        agency_links=_links(links or []),
        today=today,
        preferred_rate=preferred,
    )


def _termset(instances):
    return {"version_id": "v-t", "agreement_id": "agr-t", "instances": instances}


def _head(counterparty=None, window=None):
    return {
        "agreement_id": "agr-t",
        "counterparty": counterparty or {"advertiser": "טכנו-קור"},
        "window": window or {"starts_on": "2026-01-01", "ends_on": "2026-12-31"},
    }


def _budget_instance(amount=1_200_000):
    return {
        "instance_id": "i-b", "term_id": "budget-commitment",
        "params": {"amount": {"amount": amount, "basis": "ratecard"}, "period": "year"},
        "scope": {}, "window": {},
    }


def test_budget_standing_is_a_floor_and_paces_midyear():
    delivery = _delivery([
        ("C1", "2026-03-01", "aired", "רשת 13", 10, 300, 12.5, 400_000, "t"),
        ("C1", "2026-07-10", "scheduled", "רשת 13", 5, 150, 6.0, 150_000, "t"),
        ("C1", "2026-04-01", "unknown", "רשת 13", "", "", "", "", "t"),
        ("C9", "2026-03-01", "aired", "רשת 13", 4, 120, 3.0, 90_000, "t"),  # other advertiser
    ])
    campaigns = [("C1", "טכנו-קור"), ("C9", "אחר")]
    (snap,) = obligations.evaluate_all(
        _termset([_budget_instance()]), _head(), _inputs(delivery, campaigns)
    )
    assert snap["standing"]["counted"] == 400_000.0          # aired floor only
    assert snap["standing"]["scheduled_ahead"] == 150_000.0  # named separately
    assert snap["standing"]["unknown_days"] == 1
    assert snap["target"]["value"] == 1_200_000
    # Mid-year on a 1.2M annual commitment: expected ≈ 600k, counted 400k → at risk.
    assert snap["alarm"] == obligations.AT_RISK
    assert snap["used_default_bands"] is True
    assert snap["projection"] == 550_000.0                   # counted + scheduled


def test_agency_counterparty_resolves_through_links():
    delivery = _delivery([
        ("C1", "2026-03-01", "aired", "רשת 13", 10, 300, 12.5, 250_000, "t"),
        ("C2", "2026-03-02", "aired", "רשת 13", 10, 300, 12.5, 250_000, "t"),
    ])
    campaigns = [("C1", "מפרסם א"), ("C2", "מפרסם ב")]
    links = [("AG_1", "אופק מדיה", "מפרסם א"), ("AG_1", "אופק מדיה", "מפרסם ב")]
    (snap,) = obligations.evaluate_all(
        _termset([_budget_instance()]),
        _head(counterparty={"agency": "אופק מדיה"}),
        _inputs(delivery, campaigns, links),
    )
    assert snap["standing"]["counted"] == 500_000.0
    assert "מפרסם א" in snap["resolution"]["resolved"]


def test_trp_guarantee_counts_planned_points_on_the_all_viewers_base():
    delivery = _delivery([
        ("C1", "2026-02-01", "aired", "רשת 13", 10, 300, 120.0, 100_000, "t"),
        ("C1", "2026-08-01", "scheduled", "רשת 13", 10, 300, 40.0, 40_000, "t"),
    ])
    inst = {
        "instance_id": "i-t", "term_id": "trp-delivery-guarantee",
        "params": {"points": 300, "audience": "כלל הצופים", "window": "year",
                   "tolerance_percent": 5},
        "scope": {}, "window": {},
    }
    (snap,) = obligations.evaluate_all(
        _termset([inst]), _head(), _inputs(delivery, [("C1", "טכנו-קור")])
    )
    assert snap["standing"]["counted"] == 120.0
    assert snap["alarm"] in (obligations.AT_RISK, obligations.WATCH)
    assert "כלל הצופים" in snap["standing"]["basis"]


def test_trp_guarantee_in_a_target_audience_reports_unknown_not_a_number():
    delivery = _delivery([
        ("C1", "2026-02-01", "aired", "רשת 13", 10, 300, 120.0, 100_000, "t"),
    ])
    inst = {
        "instance_id": "i-t", "term_id": "trp-delivery-guarantee",
        "params": {"points": 300, "audience": "נשים 25-54", "window": "year"},
        "scope": {}, "window": {},
    }
    (snap,) = obligations.evaluate_all(
        _termset([inst]), _head(), _inputs(delivery, [("C1", "טכנו-קור")])
    )
    assert snap["alarm"] == obligations.UNKNOWN
    assert snap["standing"]["counted"] is None
    assert "מטבעות" in snap["alarm_reason"]


def test_window_closed_within_tolerance_is_on_track_and_below_is_breached():
    inst = {
        "instance_id": "i-t", "term_id": "trp-delivery-guarantee",
        "params": {"points": 100, "audience": "כלל הצופים", "window": "campaign",
                   "tolerance_percent": 5},
        "scope": {}, "window": {"from": "2026-01-01", "to": "2026-03-31"},
    }
    base = [("C1", "2026-02-01", "aired", "רשת 13", 10, 300, 96.0, 0, "t")]
    (ok,) = obligations.evaluate_all(
        _termset([inst]), _head(),
        _inputs(_delivery(base), [("C1", "טכנו-קור")], today=date(2026, 4, 15)),
    )
    assert ok["alarm"] == obligations.ON_TRACK
    assert ok["window_closed"] is True

    short = [("C1", "2026-02-01", "aired", "רשת 13", 10, 300, 80.0, 0, "t")]
    (bad,) = obligations.evaluate_all(
        _termset([inst]), _head(),
        _inputs(_delivery(short), [("C1", "טכנו-קור")], today=date(2026, 4, 15)),
    )
    assert bad["alarm"] == obligations.BREACHED


def test_effective_cpp_cap_compares_floor_spend_to_floor_points():
    delivery = _delivery([
        ("C1", "2026-02-01", "aired", "רשת 13", 10, 300, 100.0, 130_000, "t"),
    ])
    inst = {
        "instance_id": "i-c", "term_id": "effective-cpp-cap",
        "params": {"cap": 1200, "audience": "בתי אב", "window": "year",
                   "spend_basis": "gross"},
        "scope": {}, "window": {},
    }
    (snap,) = obligations.evaluate_all(
        _termset([inst]), _head(), _inputs(delivery, [("C1", "טכנו-קור")])
    )
    assert snap["standing"]["counted"] == 1300.0
    assert snap["alarm"] == obligations.AT_RISK


def test_preferred_position_guarantee_measures_through_the_injected_seam():
    inst = {
        "instance_id": "i-p", "term_id": "preferred-position-guarantee",
        "params": {"preferred_positions": ["1", "2", "L"], "target_percent": 40,
                   "counting_method": "agency", "window": "year"},
        "scope": {}, "window": {},
    }
    calls = {}

    def seam(**kwargs):
        calls.update(kwargs)
        return {"rate_percent": 44.0, "method": "agency"}

    (snap,) = obligations.evaluate_all(
        _termset([inst]), _head(),
        _inputs(_delivery([]), [("C1", "טכנו-קור")], preferred=seam),
    )
    assert snap["alarm"] == obligations.ON_TRACK
    assert calls["method"] == "agency"
    assert calls["positions"] == ["1", "2", "L"]
    assert snap["standing"]["method"] == "agency"


def test_share_commitment_is_honest_about_its_missing_denominator():
    inst = {
        "instance_id": "i-s", "term_id": "share-commitment",
        "params": {"share_percent": 25, "period": "year",
                   "denominator_source": "הצהרה שנתית"},
        "scope": {}, "window": {},
    }
    (snap,) = obligations.evaluate_all(
        _termset([inst]), _head(), _inputs(_delivery([]))
    )
    assert snap["alarm"] == obligations.UNKNOWN
    assert "מכנה" in snap["alarm_reason"]
