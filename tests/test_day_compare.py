"""Three competing versions of one day compared at once, with the reasoning shown.

The product's claim on this surface is not "here are three columns of numbers".
It is that a decision-maker can read WHY one version is worth more than another,
in one line, and that the explanation is arithmetic rather than narration. So the
central test here is an equality: the money difference decomposed by what
actually changed on each programme must sum EXACTLY to the total difference, with
no residue rounded into a neighbouring bucket.

Beside the money, the two dimensions a revenue figure cannot carry: what the day
leaves sellable against the licence ceilings, and which signed commitments each
version advances or endangers, measured by re-running the real obligations engine
against the day each version would produce.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from kairos_api import channel_scope, day_compare
from kairos_api import day_compare_attribution as attribution
from kairos_api import day_compare_standing as standing
from kairos_api import day_proposal_store as store
from tests.test_day_proposals import (BASE_ROWS, CHANNEL, DAY, SETTINGS_BASIS,
                                      frame, ref_for)

# The licence ceilings the day is measured against. The daily cap is set just
# above the baseline day on purpose: the highest-revenue version breaches it,
# which is the whole reason the inventory dimension exists.
CAPS = {"max_daily_ad_seconds": 1500.0, "max_ad_seconds_per_hour": 720.0,
        "max_breaks_per_hour": 4}

# --- the three competing versions -------------------------------------------
# A: load prime, pay for it in the morning.
LOAD_PRIME = [
    ("08:00", "Morning", 1, 120.0, 48_000.0, "S1"),
    ("13:00", "Talk", 2, 240.0, 120_000.0, "S2"),
    ("18:00", "News", 1, 120.0, 90_000.0, "S3"),
    ("20:00", "Drama", 4, 480.0, 520_000.0, "S4"),
    ("21:00", "Entertainment", 4, 480.0, 630_000.0, "S5"),
    ("23:30", "Late", 1, 120.0, 40_000.0, "S6"),
]
# B: protect the prime audience, make some of it back on longer noon breaks.
PROTECT_VIEWING = [
    ("08:00", "Morning", 2, 240.0, 100_000.0, "S1"),
    ("13:00", "Talk", 2, 300.0, 145_000.0, "S2"),
    ("18:00", "News", 1, 120.0, 90_000.0, "S3"),
    ("20:00", "Drama", 3, 360.0, 400_000.0, "S4"),
    ("21:00", "Entertainment", 2, 240.0, 345_000.0, "S5"),
    ("23:30", "Late", 1, 120.0, 40_000.0, "S6"),
]
# C: drop the late programme, add a break to the news, re-price the morning.
CONSOLIDATE = [
    ("08:00", "Morning", 2, 240.0, 108_000.0, "S1"),
    ("13:00", "Talk", 2, 240.0, 120_000.0, "S2"),
    ("18:00", "News", 2, 240.0, 175_000.0, "S3"),
    ("20:00", "Drama", 3, 360.0, 400_000.0, "S4"),
    ("21:00", "Entertainment", 3, 360.0, 500_000.0, "S5"),
]


@pytest.fixture()
def three_way(tmp_path, monkeypatch):
    """Three proposals on one day, all authored against the same baseline."""
    monkeypatch.setenv(store.PROPOSALS_DIR_ENV, str(tmp_path / "day_proposals"))
    monkeypatch.setattr(store, "_settings_basis", lambda: dict(SETTINGS_BASIS))
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: CHANNEL)
    baseline = frame(BASE_ROWS)
    ref = ref_for(baseline)
    made = {}
    for label, rows, author in (
        ("load_prime", LOAD_PRIME, "dana"),
        ("protect", PROTECT_VIEWING, "yossi"),
        ("consolidate", CONSOLIDATE, "rina"),
    ):
        made[label] = store.create_proposal(
            channel=CHANNEL, date=DAY, name=label, author=author, rows=frame(rows),
            baseline_ref=ref, rows_source="engine-day-plan-with-edits",
            engine={"compliance": {"available": True, "compliant": True,
                                   "checks_run": 6, "violations": []}},
        )
    return {"baseline": baseline, "ref": ref, "ids": {
        key: manifest["proposal_id"] for key, manifest in made.items()}}


def run(three_way, **kwargs):
    ids = three_way["ids"]
    order = kwargs.pop("order", ["load_prime", "protect", "consolidate"])
    kwargs.setdefault("caps", CAPS)
    return day_compare.compare(
        CHANNEL, DAY, [ids[key] for key in order],
        baseline_rows=three_way["baseline"], baseline_ref=three_way["ref"], **kwargs,
    )


def side_by(payload, name):
    return next(side for side in payload["sides"] if side.get("label") == name)


# ----------------------------------------------------------------- three sides

def test_three_versions_compare_at_once_with_distinct_money_outcomes(three_way):
    payload = run(three_way)
    assert payload["available"] is True
    assert payload["side_count"] == 3
    assert payload["scored_sides"] == 3
    assert payload["baseline"]["money"]["revenue"] == 1_250_000.0
    assert payload["baseline"]["money"]["breaks"] == 12
    assert payload["baseline"]["money"]["ad_seconds"] == 1440

    money = {side["label"]: side["money"]["revenue"] for side in payload["sides"]}
    delta = {side["label"]: side["delta"]["revenue"] for side in payload["sides"]}
    assert money == {"load_prime": 1_448_000.0, "protect": 1_120_000.0,
                     "consolidate": 1_303_000.0}
    assert delta == {"load_prime": 198_000.0, "protect": -130_000.0,
                     "consolidate": 53_000.0}
    assert payload["highest_revenue_side"] == three_way["ids"]["load_prime"]
    # Breaks and seconds move independently of the money, and are reported so.
    assert side_by(payload, "load_prime")["delta"]["breaks"] == 1
    assert side_by(payload, "load_prime")["delta"]["ad_seconds"] == 120.0
    assert side_by(payload, "consolidate")["delta"]["breaks"] == 0
    assert side_by(payload, "consolidate")["delta"]["ad_seconds"] == 0.0


def test_more_than_two_sides_is_the_point_and_two_is_the_floor(three_way):
    ids = three_way["ids"]
    one = day_compare.compare(CHANNEL, DAY, [ids["load_prime"]],
                              baseline_rows=three_way["baseline"],
                              baseline_ref=three_way["ref"], caps=CAPS)
    assert one["available"] is False
    assert "at least two sides" in one["reason"]
    assert one["reason_he"]


def test_a_comparison_with_no_baseline_refuses_instead_of_inventing_one(three_way):
    payload = day_compare.compare(CHANNEL, DAY, list(three_way["ids"].values()),
                                  baseline_rows=None, baseline_ref=three_way["ref"])
    assert payload["available"] is False
    assert payload["reason_he"]


# ------------------------------------------------------- the attribution proof

def test_the_attribution_sums_exactly_to_the_total_delta(three_way):
    payload = run(three_way)
    for side in payload["sides"]:
        attributed = side["attribution"]
        assert attributed["available"] is True
        cells = sum(round(cell["revenue_delta"] * 100) for cell in attributed["cells"])
        buckets = sum(round(bucket["revenue_delta"] * 100) for bucket in attributed["buckets"])
        total = round(attributed["revenue_delta"] * 100)
        headline = round(side["delta"]["revenue"] * 100)
        # Four levels, one arithmetic: cells -> buckets -> attribution total ->
        # the scoped headline the decision-maker actually reads.
        assert cells == buckets == total == headline
        assert attributed["reconciliation"]["exact"] is True
        assert attributed["reconciliation"]["difference"] == 0.0
        assert not [cell for cell in attributed["cells"]
                    if cell["bucket"] == attribution.UNATTRIBUTED]


def test_the_attribution_names_the_cause_and_the_daypart_of_every_agora(three_way):
    payload = run(three_way)

    prime = side_by(payload, "load_prime")["attribution"]
    cells = {(cell["bucket"], cell["daypart"]): cell for cell in prime["cells"]}
    # +250,000 from two breaks added in prime, -52,000 from one removed in the morning.
    added = cells[("breaks_added", "prime")]
    assert added["revenue_delta"] == 250_000.0
    assert added["breaks_delta"] == 2
    assert added["segments"] == 2
    assert added["ad_seconds_delta"] == 240.0
    assert "פריים" in added["sentence_he"] and "250,000" in added["sentence_he"]
    removed = cells[("breaks_removed", "morning")]
    assert removed["revenue_delta"] == -52_000.0
    assert removed["breaks_delta"] == -1
    assert removed["segments"] == 1
    assert sum(cell["revenue_delta"] for cell in prime["cells"]) == 198_000.0

    protect = side_by(payload, "protect")["attribution"]
    protect_cells = {(c["bucket"], c["daypart"]): c for c in protect["cells"]}
    assert protect_cells[("breaks_removed", "prime")]["revenue_delta"] == -155_000.0
    # Same break count, longer breaks: a length change, never a re-price.
    length = protect_cells[("length_changed", "noon")]
    assert length["revenue_delta"] == 25_000.0
    assert length["breaks_delta"] == 0
    assert length["ad_seconds_delta"] == 60.0

    consolidate = side_by(payload, "consolidate")["attribution"]
    buckets = {bucket["bucket"]: bucket for bucket in consolidate["buckets"]}
    assert buckets["breaks_added"]["revenue_delta"] == 85_000.0
    # Same breaks, same seconds, different money: the price moved and nothing else.
    assert buckets["repriced"]["revenue_delta"] == 8_000.0
    assert buckets["repriced"]["breaks_delta"] == 0
    assert buckets["repriced"]["ad_seconds_delta"] == 0.0
    assert buckets["segment_removed"]["revenue_delta"] == -40_000.0


def test_every_changed_programme_is_addressable_by_its_segment_id(three_way):
    prime = side_by(run(three_way), "load_prime")["attribution"]
    changed = {row["segment_id"]: row for row in prime["changed_segments"]}
    assert set(changed) == {f"{DAY}|{CHANNEL}|{name}" for name in ("S1", "S4", "S5")}
    drama = changed[f"{DAY}|{CHANNEL}|S4"]
    assert drama["breaks_before"] == 3 and drama["breaks_after"] == 4
    assert drama["revenue_before"] == 400_000.0 and drama["revenue_after"] == 520_000.0
    assert drama["revenue_delta"] == 120_000.0
    assert drama["bucket"] == "breaks_added"
    assert drama["start_time"] == "20:00"


def test_an_unattributable_residue_is_reported_and_never_absorbed():
    """The bucket that must stay empty, exercised by making it non-empty.

    A row whose money moved but that carries no segment id cannot be keyed to a
    programme, so its agorot land in the residue rather than being folded into a
    bucket that would then be wrong.
    """
    baseline = frame(BASE_ROWS)
    moved = frame(BASE_ROWS)
    moved.loc[0, "predicted_revenue"] = 130_000.0
    moved.loc[0, "segment_id"] = ""
    attributed = attribution.attribute(baseline, moved)
    residue = [cell for cell in attributed["cells"]
               if cell["bucket"] == attribution.UNATTRIBUTED]
    assert len(residue) == 1
    assert residue[0]["reason_he"]
    assert attributed["reconciliation"]["exact"] is False
    assert attributed["reconciliation"]["difference"] != 0.0
    # Still an identity: the reported total is the scoped delta, residue included.
    assert attributed["revenue_delta"] == \
        attributed["reconciliation"]["scoped_revenue_delta"]


# ------------------------------------------------------------- money and scope

def test_the_scope_note_travels_with_every_money_figure(three_way):
    payload = run(three_way)
    assert payload["baseline"]["money"]["scope"]["scope_channel"] == CHANNEL
    for side in payload["sides"]:
        note = side["money"]["scope"]
        assert note["scope_channel"] == CHANNEL
        assert note["scoped"] is True
        assert side["money"]["currency"] == "ILS"


def test_a_competitor_row_never_reaches_a_side_headline(three_way, tmp_path):
    rows = pd.concat([frame(LOAD_PRIME), frame(BASE_ROWS[:1], channel="קשת 12")],
                     ignore_index=True)
    manifest = store.create_proposal(
        channel=CHANNEL, date=DAY, name="עם שורת מתחרה", author="dana", rows=rows,
        baseline_ref=three_way["ref"], rows_source="engine-day-plan")
    payload = day_compare.compare(
        CHANNEL, DAY, [manifest["proposal_id"], three_way["ids"]["protect"]],
        baseline_rows=three_way["baseline"], baseline_ref=three_way["ref"], caps=CAPS)
    side = side_by(payload, "עם שורת מתחרה")
    assert side["money"]["revenue"] == 1_448_000.0
    assert side["money"]["scope"]["competitor_rows_excluded"] == 1
    assert side["delta"]["revenue"] == 198_000.0


def test_revenue_net_of_retention_is_reported_when_the_rows_can_value_it(three_way):
    payload = run(three_way)
    baseline = payload["baseline"]["money"]
    assert baseline["net_available"] is True
    assert baseline["retention_cost"] > 0
    assert baseline["revenue_net_of_retention"] < baseline["revenue"]
    for side in payload["sides"]:
        assert side["money"]["net_available"] is True
        assert side["delta"]["revenue_net_of_retention"] is not None


def test_a_side_whose_rows_cannot_value_retention_says_why(three_way):
    thin = frame(LOAD_PRIME).drop(columns=["baseline_tvr"])
    manifest = store.create_proposal(
        channel=CHANNEL, date=DAY, name="ללא רייטינג", author="dana", rows=thin,
        baseline_ref=three_way["ref"], rows_source="engine-day-plan")
    payload = day_compare.compare(
        CHANNEL, DAY, [manifest["proposal_id"], three_way["ids"]["protect"]],
        baseline_rows=three_way["baseline"], baseline_ref=three_way["ref"], caps=CAPS)
    side = side_by(payload, "ללא רייטינג")
    assert side["money"]["net_available"] is False
    assert "baseline_tvr" in side["money"]["net_reason"]
    assert side["money"]["revenue_net_of_retention"] is None
    assert side["delta"]["revenue_net_of_retention"] is None
    # The money itself is still fully comparable; only the net is unavailable.
    assert side["delta"]["revenue"] == 198_000.0


# -------------------------------------------------------------- the inventory

def test_the_inventory_consequence_is_reported_per_side_against_the_ceilings(three_way):
    payload = run(three_way)
    assert payload["baseline"]["inventory"]["daily_ad_seconds_remaining"] == 60.0
    prime = side_by(payload, "load_prime")["inventory"]
    assert prime["ad_seconds_planned"] == 1560.0
    assert prime["daily_ad_seconds_remaining"] == -60.0
    assert prime["over_daily_cap"] is True
    protect = side_by(payload, "protect")["inventory"]
    assert protect["daily_ad_seconds_remaining"] == 120.0
    assert protect["over_daily_cap"] is False
    assert protect["hours_covered"] == 6
    # The highest-revenue side is the one that breaches the cap, and the headline
    # a decision-maker reads says so in the same sentence as the money.
    assert "חריגה" in side_by(payload, "load_prime")["headline"]


def test_an_undefined_cap_is_reported_as_absent_and_never_as_zero(three_way):
    payload = run(three_way, caps={})
    inventory = side_by(payload, "load_prime")["inventory"]
    assert inventory["daily_ad_seconds_cap"] is None
    assert inventory["daily_ad_seconds_remaining"] is None
    assert inventory["over_daily_cap"] is False
    assert inventory["hourly_breaks_remaining"] is None
    assert inventory["cap_note_he"]


# --------------------------------------------------------- contractual standing

def _delivery():
    """A budget commitment part-delivered, with this Tuesday still ahead of it.

    1,200 seconds already aired on 1 November for 700,000 ILS, and 1,500 seconds
    booked on the Tuesday being re-planned for 300,000 more. The day the versions
    argue over is the day that decides whether the commitment closes.
    """
    return pd.DataFrame([
        ("C1", "2024-11-01", "aired", CHANNEL, 40, 1200, 20.0, 700_000),
        ("C1", DAY, "scheduled", CHANNEL, 50, 1500, 25.0, 300_000),
    ], columns=["campaign_id", "broadcast_date", "air_state", "channel", "spots",
                "seconds", "rating_points_planned", "spend_ils"])


def _trade_context():
    return {
        "approved": [({
            "agreement_id": "agr-techno",
            "title": "מסגרת שנתית טכנו-קור",
            "counterparty": {"advertiser": "טכנו-קור"},
            "window": {"starts_on": "2024-11-01", "ends_on": "2024-11-30"},
        }, {
            "version_id": "v1", "agreement_id": "agr-techno",
            "instances": [{
                "instance_id": "i-budget", "term_id": "budget-commitment",
                "params": {"amount": {"amount": 1_000_000, "basis": "ratecard"},
                           "period": "month"},
                "scope": {}, "window": {},
            }],
        })],
        "delivery": _delivery(),
        "campaigns": pd.DataFrame([("C1", "טכנו-קור")],
                                  columns=["campaign_id", "advertiser"]),
        "links": pd.DataFrame(columns=["agency_id", "agency_name", "advertiser"]),
        "today": date(2024, 11, 20),
    }


def test_a_version_that_cuts_the_day_endangers_the_commitment_it_was_carrying(three_way):
    payload = run(three_way, trade_context=_trade_context())

    protect = side_by(payload, "protect")["commitments"]
    assert protect["available"] is True
    assert protect["basis"] == "projection"
    assert protect["method"]["note_he"]
    # 1,500 seconds are booked; this version supplies 1,380 of them.
    assert protect["day_capacity"]["booked_seconds"] == 1500.0
    assert protect["day_capacity"]["side_seconds"] == 1380.0
    assert protect["day_capacity"]["shortfall_seconds"] == 120.0
    (obligation,) = protect["obligations"]
    assert obligation["verdict"] == standing.ENDANGERS
    assert obligation["verdict_he"] == "מסכן"
    assert obligation["agreement_title"] == "מסגרת שנתית טכנו-קור"
    assert obligation["term_id"] == "budget-commitment"
    assert obligation["side"]["target"] == 1_000_000.0
    # The commitment closes at 988,000 as the day stands and at 976,000 under
    # this version; the alarm has not moved yet and the projection has.
    assert obligation["baseline"]["projection"] == 988_000.0
    assert obligation["side"]["projection"] == 976_000.0
    assert obligation["baseline"]["alarm"] == obligation["side"]["alarm"] == "on_track"
    assert obligation["reason_he"]
    assert protect["counts"][standing.ENDANGERS] == 1
    assert "בסיכון" in side_by(payload, "protect")["headline"]


def test_a_version_with_the_seconds_to_carry_the_day_advances_it(three_way):
    payload = run(three_way, trade_context=_trade_context())
    prime = side_by(payload, "load_prime")["commitments"]
    (obligation,) = prime["obligations"]
    # 1,560 seconds supplied against 1,500 booked: the whole day is carried, so
    # the commitment closes at its target instead of 12,000 short.
    assert prime["day_capacity"]["side_factor"] == 1.0
    assert prime["day_capacity"]["shortfall_seconds"] == 0.0
    assert obligation["verdict"] == standing.ADVANCES
    assert obligation["side"]["projection"] == 1_000_000.0
    assert prime["counts"][standing.ADVANCES] == 1
    # A version that changes no ad seconds changes no commitment.
    consolidate = side_by(payload, "consolidate")["commitments"]
    assert consolidate["obligations"][0]["verdict"] == standing.UNCHANGED


def test_a_day_with_no_delivery_row_is_unknown_and_not_nil(three_way):
    context = _trade_context()
    context["delivery"] = context["delivery"].head(1)  # only the other day
    payload = run(three_way, trade_context=context)
    commitments = side_by(payload, "protect")["commitments"]
    assert commitments["available"] is False
    assert "unknown rather than nil" in commitments["reason"]
    assert commitments["reason_he"]
    assert commitments["obligations"] == []


def test_no_approved_agreement_means_nothing_is_measured_and_it_says_so(three_way):
    context = _trade_context()
    context["approved"] = []
    commitments = side_by(run(three_way, trade_context=context), "protect")["commitments"]
    assert commitments["available"] is False
    assert commitments["reason_he"]


def test_without_a_trade_context_the_dimension_is_named_unmeasured(three_way):
    commitments = side_by(run(three_way), "protect")["commitments"]
    assert commitments["available"] is False
    assert commitments["reason_he"]
    assert "לא נבדק" in side_by(run(three_way), "protect")["headline"]


# ------------------------------------------------------------- honest unknowns

def test_a_side_that_does_not_exist_says_so_and_the_others_still_compare(three_way):
    ids = three_way["ids"]
    payload = day_compare.compare(
        CHANNEL, DAY, [ids["load_prime"], "ffffffffffff", ids["protect"]],
        baseline_rows=three_way["baseline"], baseline_ref=three_way["ref"], caps=CAPS)
    assert payload["available"] is True
    assert payload["side_count"] == 3
    assert payload["scored_sides"] == 2
    missing = next(side for side in payload["sides"] if side["side_id"] == "ffffffffffff")
    assert missing["available"] is False
    assert missing["reason_he"]


def test_a_stale_side_carries_its_staleness_into_the_comparison(three_way):
    moved = list(BASE_ROWS)
    moved[3] = ("20:00", "Drama", 4, 480.0, 520_000.0, "S4")
    payload = day_compare.compare(
        CHANNEL, DAY, list(three_way["ids"].values()),
        baseline_rows=frame(moved), baseline_ref=ref_for(frame(moved)), caps=CAPS)
    for side in payload["sides"]:
        assert side["staleness"]["stale"] is True
        assert [item["field"] for item in side["staleness"]["moved"]] == ["day_sha256"]


def test_the_live_committed_day_is_reference_and_refuses_to_be_attributed(three_way):
    payload = run(three_way, include_live=True, live_rows=frame(CONSOLIDATE),
                  order=["load_prime", "protect"])
    live = next(side for side in payload["sides"] if side["side_id"] == "live")
    assert live["available"] is True
    assert live["basis"] == "committed-weekly-plan"
    assert live["basis_matches_baseline"] is False
    # Its money is real and reported; attributing a delta across two bases is not.
    assert live["money"]["revenue"] == 1_303_000.0
    assert live["attribution"]["available"] is False
    assert "committed-weekly-plan" in live["attribution"]["reason"]
    assert live["attribution"]["reason_he"]
    assert live["attribution"]["cells"] == []


def test_a_live_day_the_plan_does_not_carry_says_so(three_way):
    payload = run(three_way, include_live=True, live_rows=None,
                  order=["load_prime", "protect"])
    live = next(side for side in payload["sides"] if side["side_id"] == "live")
    assert live["available"] is False
    assert live["reason_he"]


def test_the_guardrail_verdict_recorded_at_authoring_travels_with_the_side(three_way):
    payload = run(three_way)
    assert side_by(payload, "load_prime")["compliance"]["compliant"] is True
    assert side_by(payload, "load_prime")["compliance"]["checks_run"] == 6


def test_a_side_authored_without_an_engine_verdict_reports_compliance_unknown(three_way):
    manifest = store.create_proposal(
        channel=CHANNEL, date=DAY, name="ללא פסק מנוע", author="dana",
        rows=frame(LOAD_PRIME), baseline_ref=three_way["ref"],
        rows_source="engine-day-plan")
    payload = day_compare.compare(
        CHANNEL, DAY, [manifest["proposal_id"], three_way["ids"]["protect"]],
        baseline_rows=three_way["baseline"], baseline_ref=three_way["ref"], caps=CAPS)
    compliance = side_by(payload, "ללא פסק מנוע")["compliance"]
    assert compliance["available"] is False
    assert compliance["reason_he"]


# ------------------------------------------------------------------- the headline

def test_the_headline_carries_money_cause_commitments_and_inventory_in_one_line(three_way):
    payload = run(three_way, trade_context=_trade_context())
    headline = side_by(payload, "load_prime")["headline"]
    assert "\n" not in headline
    assert "+198,000 ₪" in headline
    assert "+15.8%" in headline
    assert "+1 ברייקים" in headline
    assert "ברייקים נוספו בפריים טיים" in headline
    assert "מתקדמת" in headline
    assert "חריגה של 60 שניות" in headline

    quiet = side_by(payload, "consolidate")["headline"]
    assert "ללא שינוי במספר הברייקים" in quiet
    assert "אין שינוי בעמידה בהתחייבויות" in quiet
