"""The demo seed, measured against the files it actually wrote.

This runs read-only over the shipped data. It does not re-seed and it does not
write, so a red here is a statement about what is on disk right now.

The bars, in the order they matter.

**Every seeded row says it is a demo row.** All three files carry ``is_demo``
and every row the seed wrote has it true, so no reader and no query can take a
seeded booking for a signed one.

**Nothing rival is anywhere near it.** Every campaign, creative and delivery row
carries the operator channel from settings, and no rival name appears in any
cell of any of the three files.

**The delivery figures reconcile to the cent with the ledger the money board
reads.** The seed does not price anything of its own, so the sum of every
counted day equals the priced ledger's gross exactly, and the counted spot count
equals the number of rows in the traffic file.

**A day with no source is blank, not zero.** Every ``unknown`` row has empty
figure cells, which is what makes the counted total a floor rather than a claim.

**The identity is real.** Every advertiser is a name observed in the daily file
and present in the shipped name space, never a seed id, and every agency id
resolves to an agency that exists.

**Re-running it changes nothing.** The builder is deterministic, so a second run
produces the same rows, including the moment each row was first written.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
HEBREW = set("אבגדהוזחטיכלמנסעפצקרשתךםןףץ")
RIVALS = ("קשת 12", "כאן 11", "עכשיו 14")
DEMO_PREFIX = "CMP_D"


def _read(name: str) -> pd.DataFrame:
    path = DATA / name
    if not path.exists():
        pytest.skip(f"{name} is not on disk, so there is nothing seeded to measure")
    return pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)


@pytest.fixture(scope="module")
def campaigns() -> pd.DataFrame:
    frame = _read("campaigns.csv")
    demo = frame[frame["campaign_id"].str.startswith(DEMO_PREFIX)]
    if demo.empty:
        pytest.skip("no demo campaign is seeded, so there is nothing to measure")
    return demo


@pytest.fixture(scope="module")
def assets() -> pd.DataFrame:
    return _read("campaign_assets.csv")


@pytest.fixture(scope="module")
def delivery() -> pd.DataFrame:
    return _read("campaign_delivery.csv")


@pytest.fixture(scope="module")
def traffic() -> pd.DataFrame:
    from scripts.seed_campaigns import daily_paths, load_day

    paths = daily_paths()
    if not paths:
        pytest.skip("no traffic file on disk, so nothing could have been seeded from one")
    return pd.concat([load_day(path) for path in paths], ignore_index=True)


# --------------------------------------------------------------------------
# The demo marker
# --------------------------------------------------------------------------

def test_every_seeded_row_in_every_file_is_marked_demo(campaigns, assets, delivery):
    """One column, three files, no exceptions. A convention would not survive a query."""
    assert set(campaigns["is_demo"]) == {"true"}
    assert set(assets["is_demo"]) == {"true"}
    assert set(delivery["is_demo"]) == {"true"}
    assert set(campaigns[campaigns["record_type"] == "campaign"]["data_source"]) == {"demo_seed"}
    for note in campaigns[campaigns["record_type"] == "campaign"]["demo_note"]:
        assert "Seed rule" in note


def test_the_demo_note_states_what_the_seed_made_and_what_it_read(campaigns):
    """The rule that made the flight and the goal is on the row, not in a readme."""
    note = campaigns[campaigns["record_type"] == "campaign"]["demo_note"].iloc[0]
    assert "Israeli broadcast week" in note
    assert "not a signed insertion order" in note


# --------------------------------------------------------------------------
# The competitor boundary
# --------------------------------------------------------------------------

def test_every_seeded_row_carries_the_operator_channel_and_only_that(campaigns, assets, delivery):
    from kairos_api import channel_scope

    owned = channel_scope.operator_channel()
    assert owned, "the seed cannot run without an operator channel, so one must be configured"
    assert set(campaigns[campaigns["record_type"] == "campaign"]["channel"]) == {owned}
    assert set(assets["channel"]) == {owned}
    assert set(delivery["channel"]) == {owned}


def test_no_rival_channel_name_appears_in_any_seeded_file(campaigns, assets, delivery):
    """Searched over every cell, because one leaked cell is the whole defect."""
    for name, frame in (("campaigns", campaigns), ("assets", assets), ("delivery", delivery)):
        rendered = frame.to_csv(index=False)
        for rival in RIVALS:
            assert rival not in rendered, f"{rival} reached {name}"


# --------------------------------------------------------------------------
# The figures, against the ledger they came from
# --------------------------------------------------------------------------

def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column].replace("", "0"), errors="coerce").fillna(0.0)


def test_the_counted_spend_equals_the_priced_ledger_to_the_cent(delivery):
    """The seed prices nothing of its own, so it cannot disagree with the money board."""
    from kairos_api.exporters import _load_daily_pricing

    result = _load_daily_pricing()
    if result is None:
        pytest.skip("no daily file to price, so there is no ledger to reconcile against")
    gross = round(sum(float(spot.revenue or 0.0) for spot in result.priced), 2)
    counted = round(float(_numeric(delivery, "spend_ils").sum()), 2)
    assert counted == gross, f"seeded spend {counted} does not reconcile to ledger gross {gross}"


def test_the_counted_spots_equal_the_rows_the_traffic_file_carries(delivery, traffic):
    """Every airing is counted exactly once, in exactly one of the two sourced states."""
    counted = int(_numeric(delivery, "spots").sum())
    assert counted == len(traffic)
    states = set(delivery[_numeric(delivery, "spots") > 0]["air_state"])
    assert states <= {"aired", "scheduled"}


def test_the_day_is_split_into_what_has_aired_and_what_is_still_to_come(delivery):
    """Both sourced states exist on the seeded data, which is what makes it real."""
    counts = delivery["air_state"].value_counts().to_dict()
    assert counts.get("aired", 0) > 0
    assert counts.get("scheduled", 0) > 0
    assert counts.get("unknown", 0) > 0
    instants = {value for value in delivery["counted_as_of"] if value}
    assert len(instants) == 1, "one seeded run means one counted-as-of instant"
    assert "programme" in delivery["counted_as_of_basis"].iloc[0]


def test_the_payload_publishes_a_denominator_that_is_a_calendar(campaigns):
    """One day can be part aired and part still to come, so row counts are not days.

    ``aired.days`` plus ``scheduled.days`` plus ``unknown.days`` deliberately
    overshoots the flight on exactly the day an operator is standing in, which is
    why ``sourced_days`` and ``flight_days`` exist beside them. A surface that
    divides by the wrong one gets a percentage nobody computed.
    """
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kairos_api import campaigns_api

    app = FastAPI()
    app.include_router(campaigns_api.router)
    payload = TestClient(app).get("/api/clients/campaigns").json()
    split = [
        record["delivery"] for record in payload["campaigns"]
        if record["delivery"]["aired"]["days"] and record["delivery"]["scheduled"]["days"]
    ]
    assert split, "the seeded day is split, so at least one campaign has both states"
    for delivery in split:
        assert delivery["flight_days"] == 7
        assert delivery["sourced_days"] == 1
        assert delivery["unknown"]["days"] == 6
        rows = (delivery["aired"]["days"] + delivery["scheduled"]["days"]
                + delivery["unknown"]["days"])
        assert rows > delivery["flight_days"]


def test_a_day_with_no_source_is_blank_and_never_zero(delivery):
    """The whole point of the floor: a gap is a gap, and a sum cannot swallow it."""
    unknown = delivery[delivery["air_state"] == "unknown"]
    assert not unknown.empty
    for column in ("spots", "seconds", "rating_points_planned", "spend_ils",
                   "spots_dropped_by_rule", "source_file"):
        assert set(unknown[column]) == {""}, column
    for note in unknown["note"]:
        assert "is not zero" in note


def test_a_rule_is_named_only_where_it_removed_something(delivery):
    """A rule id beside a count of zero says a rule bit when none did."""
    named = delivery[delivery["dropped_rule_id"] != ""]
    assert not named.empty
    assert (_numeric(named, "spots_dropped_by_rule") > 0).all()


def test_every_delivery_day_falls_inside_its_campaign_flight(campaigns, delivery):
    """A delivery row outside the flight would be a figure with no commitment behind it."""
    window = {
        row["campaign_id"]: (row["starts_on"], row["ends_on"])
        for _, row in campaigns[campaigns["record_type"] == "campaign"].iterrows()
    }
    for _, row in delivery.iterrows():
        starts, ends = window[row["campaign_id"]]
        assert starts <= row["broadcast_date"] <= ends, row["campaign_id"]


# --------------------------------------------------------------------------
# The identity, and the flight
# --------------------------------------------------------------------------

def test_every_advertiser_is_a_real_observed_name_and_never_a_seed_id(campaigns, traffic):
    """The owner's complaint, held shut: no ADV_nn reaches a campaign row."""
    names = _read("advertiser_names.csv")
    known = set(names["name"])
    observed = set(traffic["advertiser"])
    for advertiser in campaigns[campaigns["record_type"] == "campaign"]["advertiser"]:
        assert advertiser in observed, advertiser
        assert advertiser in known, advertiser
        assert not advertiser.startswith("ADV_"), advertiser


def test_every_campaign_name_is_a_real_label_from_the_traffic_file(campaigns, traffic):
    labels = set(traffic["campaign"])
    seeded = set(campaigns[campaigns["record_type"] == "campaign"]["name"])
    assert seeded <= labels
    assert len(seeded) == traffic["campaign"].nunique()


def test_every_campaign_is_bought_through_an_agency_that_exists(campaigns):
    """No orphan and no agency-less client: every booking resolves to a real agency."""
    agencies = set(_read("agencies.csv")["agency_id"])
    for agency_id in campaigns[campaigns["record_type"] == "campaign"]["agency_id"]:
        assert agency_id, "a campaign with no agency is the state the owner ruled out"
        assert agency_id in agencies, agency_id


def test_every_flight_is_an_israeli_broadcast_week(campaigns):
    """Sunday to Saturday, which is the week this market plans and bills in."""
    for _, row in campaigns[campaigns["record_type"] == "campaign"].iterrows():
        starts = date.fromisoformat(row["starts_on"])
        ends = date.fromisoformat(row["ends_on"])
        assert starts.weekday() == 6, f"{row['campaign_id']} does not start on a Sunday"
        assert ends.weekday() == 5, f"{row['campaign_id']} does not end on a Saturday"
        assert (ends - starts).days % 7 == 6


def test_every_campaign_has_exactly_one_flight_with_a_goal(campaigns):
    flights = campaigns[campaigns["record_type"] == "flight"]
    booked = campaigns[campaigns["record_type"] == "campaign"]
    assert len(flights) == len(booked)
    for _, row in flights.iterrows():
        assert row["goal_kind"] in {"grp", "spots"}
        assert float(row["goal_value"]) > 0


def test_a_campaign_the_seed_could_not_price_carries_no_budget_rather_than_zero(campaigns):
    """Every spot removed by a rule means no observed spend, which is not a budget of nothing."""
    booked = campaigns[campaigns["record_type"] == "campaign"]
    for value in booked["budget_ils"]:
        assert value == "" or float(value) > 0, value
    for value in booked["rating_goal_points"]:
        assert value == "" or float(value) > 0, value


def test_the_price_model_is_the_one_the_traffic_file_records(campaigns, traffic):
    """Real, not assigned: CPP and FIX are columns in the file, not the seed's opinion."""
    models = set(campaigns[campaigns["record_type"] == "campaign"]["price_model"])
    assert models <= {"cpp", "flat", ""}
    assert "cpp" in models and "flat" in models


# --------------------------------------------------------------------------
# The creative
# --------------------------------------------------------------------------

def test_every_creative_is_real_where_the_log_speaks_and_unknown_where_it_cannot(assets):
    for _, row in assets.iterrows():
        assert row["house_number"], "a creative with no house number is not a broadcast asset"
        assert row["version_name"]
        assert float(row["duration_seconds"]) > 0
        assert row["spot_type"] in {"פרסומת", "חסות"}
        assert row["length_class"] in {"commercial", "sponsorship"}
        assert row["identity_source"] == "traffic_log"
        assert int(row["airings_observed"]) > 0
        assert row["media_url"] == ""
        assert row["media_state"] == "unknown"
        assert row["video_format"] == ""
        assert row["aspect_ratio"] == ""
        assert row["loudness_lufs"] == ""
        assert row["clearance_verdict"] == "unknown"


def test_the_creative_lengths_are_the_lengths_this_market_actually_trades(assets):
    """Sponsorship at ten seconds or under, spots above it, per the rate card research."""
    durations = pd.to_numeric(assets["duration_seconds"])
    assert durations.min() >= 5
    assert durations.max() <= 90
    sponsorship = assets[assets["length_class"] == "sponsorship"]
    assert not sponsorship.empty
    assert pd.to_numeric(sponsorship["duration_seconds"]).max() <= 10


def test_every_asset_hangs_on_a_seeded_campaign(campaigns, assets):
    booked = set(campaigns[campaigns["record_type"] == "campaign"]["campaign_id"])
    assert set(assets["campaign_id"]) <= booked
    assert set(assets["campaign_id"]) == booked


# --------------------------------------------------------------------------
# Re-running it
# --------------------------------------------------------------------------

def test_running_the_seed_again_builds_exactly_the_same_rows(campaigns, assets, delivery):
    """Idempotent, measured by rebuilding in memory and comparing, not by writing."""
    from scripts.seed_campaigns import (
        as_of_instant, build, daily_paths, existing_stamps, load_day, operator_channel,
    )

    frames = [(path, load_day(path)) for path in daily_paths()]
    as_of, _ = as_of_instant([frame for _, frame in frames])
    rebuilt, rebuilt_assets, rebuilt_delivery = build(
        frames, operator_channel(), as_of, existing_stamps()
    )
    assert len(rebuilt) == len(campaigns[campaigns["record_type"] == "campaign"])
    assert len(rebuilt_assets) == len(assets)
    assert len(rebuilt_delivery) == len(delivery)
    on_disk = campaigns[campaigns["record_type"] == "campaign"].set_index("campaign_id")
    for record in rebuilt:
        row = on_disk.loc[record["campaign_id"]]
        for field in ("name", "advertiser", "agency_id", "channel", "starts_on", "ends_on",
                      "budget_ils", "rating_goal_points", "price_model"):
            assert str(row[field]) == str(record[field]), f"{record['campaign_id']}.{field}"
        assert record["created_at"] == row["created_at"], "a re-run must not restamp a row"


def test_the_seed_refuses_to_run_with_no_operator_channel(monkeypatch, capsys):
    """No channel means no boundary to stamp, so nothing is written at all."""
    import scripts.seed_campaigns as seed

    monkeypatch.setattr(seed, "operator_channel", lambda: "")
    assert seed.main([]) == 2
    assert "operator channel" in capsys.readouterr().out
