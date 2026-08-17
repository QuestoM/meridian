"""Adversarial verification of the day-comparison surface, by a second pair of eyes.

The committed suite proves the four-level attribution identity on clean fixtures:
cells sum to buckets, buckets to the attributed total, and the total to the scoped
headline a decision-maker reads. These tests attack that identity with the rows a
fixture would never contain - two rows claiming one programme, a row carrying no
programme id at all, money that runs negative, and revenue carrying fractions
below one agora - because an identity that holds only on round numbers is not an
identity.

Two of the attacks landed when this suite was written, and both were recorded
as strict xfails - a ratchet that fails the moment somebody fixes the
underlying behaviour. Both ratchets have since fired and come off: the
attribution now measures its truth figure on the headline's own rounding
basis (one sum, one round), and an all-unknown commitments standing names its
unknowns instead of reading as "no change". The tests below now lock the
fixed behaviour.

The rest are regression locks on behaviour that is genuinely right and that a
future refactor of the rounding basis could easily break.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from kairos_api import channel_scope, day_compare
from kairos_api import day_compare_attribution as attribution
from kairos_api import day_compare_standing as standing
from kairos_api import day_proposal_store as store
from tests.test_day_proposals import (BASE_ROWS, CHANNEL, DAY, SETTINGS_BASIS,
                                      frame, ref_for)

CAPS = {"max_daily_ad_seconds": 1500.0, "max_ad_seconds_per_hour": 720.0,
        "max_breaks_per_hour": 4}


@pytest.fixture()
def scoped(tmp_path, monkeypatch):
    """A relocated store and a fixed operator channel, as the committed suite uses."""
    monkeypatch.setenv(store.PROPOSALS_DIR_ENV, str(tmp_path / "day_proposals"))
    monkeypatch.setattr(store, "_settings_basis", lambda: dict(SETTINGS_BASIS))
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: CHANNEL)
    return tmp_path


def identity(baseline: pd.DataFrame, side: pd.DataFrame) -> dict:
    """The four levels the surface stakes its credibility on, in integer agorot.

    Computed the way the payload's own reader would: the money block and the
    attribution come from the same two functions the comparison calls, so this
    measures the shipped arithmetic rather than a re-implementation of it.
    """
    money_base, owned_base = day_compare._money(baseline)
    money_side, owned_side = day_compare._money(side)
    delta = day_compare._money_delta(money_side, money_base)
    attributed = attribution.attribute(owned_base, owned_side)
    return {
        "cells": sum(round(cell["revenue_delta"] * 100)
                     for cell in attributed.get("cells", [])),
        "buckets": sum(round(bucket["revenue_delta"] * 100)
                       for bucket in attributed.get("buckets", [])),
        "total": round(attributed["revenue_delta"] * 100),
        "headline": round(delta["revenue"] * 100),
        "exact": attributed["reconciliation"]["exact"],
        "residual": attributed["reconciliation"]["difference"],
        "attributed": attributed,
    }


def rows_with(mapper) -> pd.DataFrame:
    return frame([mapper(row) for row in BASE_ROWS])


# --------------------------------------------------- the attack that lands (1)

def test_sub_agora_fractions_must_not_break_the_identity_or_the_exact_flag(scoped):
    """Revenue below one agora per row: the identity the surface promises.

    The engine's ``predicted_revenue`` is a float product of rate, rating and
    seconds, so fractions finer than an agora are the normal case rather than a
    contrived one. Whichever rounding basis wins, the two figures must be the
    same figure, and ``exact`` must never assert an agreement that does not hold.
    """
    baseline = frame(BASE_ROWS)
    fractional = rows_with(lambda row: (row[0], row[1], row[2], row[3],
                                        row[4] + 0.001, row[5]))
    measured = identity(baseline, fractional)
    assert measured["total"] == measured["headline"], (
        f"attributed total {measured['total']} agorot disagrees with the headline "
        f"{measured['headline']} agorot while exact={measured['exact']}")
    assert measured["cells"] == measured["buckets"] == measured["total"]


def test_the_divergence_is_zero_at_any_row_count(scoped):
    """The historical defect's own measurements, inverted into the lock.

    Before the fix, 6 rows carrying +0.001 ILS diverged by 1 agora and 60 rows
    by 6, with exact:True beside both. One rounding basis (the headline's: sum
    once, round once) makes the divergence structurally zero; these are the
    same two measurements asserting the repaired identity.
    """
    # The identity must CLOSE against the headline, and the sub-agora drift
    # the per-row cells cannot carry must surface as an explicit unattributed
    # remainder with exact:False - declared, not denied. exact:True beside a
    # divergence was the defect; exact:False beside a printed one-agora
    # remainder is the honesty mechanism working.
    six = identity(frame(BASE_ROWS),
                   rows_with(lambda r: (r[0], r[1], r[2], r[3], r[4] + 0.001, r[5])))
    assert six["headline"] - six["total"] == 0
    assert six["exact"] is False

    base_many = [("20:00", "Drama", 3, 360.0, 10_000.0, f"X{index}")
                 for index in range(60)]
    side_many = [("20:00", "Drama", 3, 360.0, 10_000.001, f"X{index}")
                 for index in range(60)]
    many = identity(frame(base_many), frame(side_many))
    assert many["headline"] - many["total"] == 0
    assert many["exact"] is False


# --------------------------------------------------- the attack that lands (2)

def test_an_all_unknown_standing_must_not_read_as_no_change():
    """The headline sentence for a dimension that measured nothing.

    The module's own honesty rule is that no source means unknown and never zero.
    The dimension body obeys it - the obligation carries verdict ``unknown`` with
    its reason. The one line above it does not, and the line is what gets read.
    """
    all_unknown = {
        "available": True,
        "counts": {standing.ADVANCES: 0, standing.ENDANGERS: 0, standing.BREAKS: 0,
                   standing.UNCHANGED: 0, standing.UNKNOWN: 1},
        "obligations": [{"verdict": standing.UNKNOWN}],
    }
    phrase = day_compare._commitment_phrase(all_unknown)
    assert "אין שינוי" not in phrase, (
        f"a dimension that measured only unknowns rendered as {phrase!r}")


def test_a_genuinely_unchanged_standing_may_say_so():
    """The control for the test above: 'no change' is correct when it is measured.

    Without this the xfail could be satisfied by deleting the phrase entirely,
    which would lose a true statement instead of fixing a false one.
    """
    measured_unchanged = {
        "available": True,
        "counts": {standing.ADVANCES: 0, standing.ENDANGERS: 0, standing.BREAKS: 0,
                   standing.UNCHANGED: 3, standing.UNKNOWN: 0},
        "obligations": [{"verdict": standing.UNCHANGED}] * 3,
    }
    assert "אין שינוי" in day_compare._commitment_phrase(measured_unchanged)
    assert day_compare._commitment_phrase({"available": False}) == \
        "עמידה בהתחייבויות: לא נבדק"


# ------------------------------------------- regression locks that already hold

def test_two_rows_claiming_one_programme_surface_as_unattributed(scoped):
    """A duplicate segment_id: the money is real, the diff cannot key it.

    ``_row_index`` keeps one row per segment id, so the second row's money would
    vanish from the explanation. It does not vanish from the frame, and the
    reconciliation is measured against the frame, so the difference lands in the
    ``unattributed`` bucket with its reason and the identity still closes.
    """
    baseline = frame(BASE_ROWS)
    doubled = frame(list(BASE_ROWS)
                    + [("21:00", "Entertainment", 3, 360.0, 250_000.0, "S5")])
    measured = identity(baseline, doubled)
    assert measured["cells"] == measured["buckets"] == measured["total"] == \
        measured["headline"]
    assert measured["exact"] is False, "an unkeyable row must be reported, not hidden"
    assert measured["residual"] == 500_000.0
    unattributed = [cell for cell in measured["attributed"]["cells"]
                    if cell["bucket"] == attribution.UNATTRIBUTED]
    assert len(unattributed) == 1
    assert unattributed[0]["sentence_he"]
    assert unattributed[0]["revenue_delta"] == 500_000.0


def test_a_row_with_no_programme_id_is_reported_and_never_absorbed(scoped):
    """A blank segment_id: the row's money must not be folded into a neighbour."""
    baseline = frame(BASE_ROWS)
    blanked = frame(BASE_ROWS)
    blanked.loc[3, "segment_id"] = ""
    measured = identity(baseline, blanked)
    assert measured["cells"] == measured["buckets"] == measured["total"] == \
        measured["headline"]
    assert measured["exact"] is False
    assert measured["residual"] == 400_000.0
    buckets = {bucket["bucket"] for bucket in measured["attributed"]["buckets"]}
    assert attribution.UNATTRIBUTED in buckets


def test_neither_side_carrying_a_programme_id_refuses_rather_than_returning_zero(scoped):
    """No keyable row at all is a refusal with a reason, not an empty attribution."""
    blank_base = frame(BASE_ROWS)
    blank_side = frame(BASE_ROWS)
    blank_base["segment_id"] = ""
    blank_side["segment_id"] = ""
    attributed = attribution.attribute(blank_base, blank_side)
    assert attributed["available"] is False
    assert attributed["reason_he"]
    assert "cells" not in attributed or not attributed.get("cells")


def test_money_that_runs_negative_keeps_the_identity(scoped):
    """A day whose every row is a loss. Integer agorot must stay symmetric."""
    baseline = frame(BASE_ROWS)
    negative = rows_with(lambda row: (row[0], row[1], row[2], row[3],
                                      -abs(row[4]), row[5]))
    measured = identity(baseline, negative)
    assert measured["cells"] == measured["buckets"] == measured["total"] == \
        measured["headline"]
    assert measured["exact"] is True
    # The baseline day carries 1,250,000 ILS; every row flipping sign moves it to
    # -1,250,000, so the whole day swings by -2,500,000 ILS.
    assert measured["total"] == -250_000_000


def test_an_hour_past_midnight_is_classified_and_never_dropped(scoped):
    """Hour 24 and 25 are real in a broadcast day and must land in a daypart.

    The engine's broadcast day runs past midnight, so ``24:30`` is a legitimate
    clock cell. A row whose daypart could not be resolved would still carry money,
    so silently dropping it would break the identity.
    """
    baseline = frame([("24:30", "Overnight", 1, 120.0, 30_000.0, "S9")])
    side = frame([("24:30", "Overnight", 2, 240.0, 61_000.0, "S9")])
    measured = identity(baseline, side)
    assert measured["total"] == measured["headline"] == 31_000_00
    cell = measured["attributed"]["cells"][0]
    assert cell["daypart"] is not None, "hour 24 fell into no daypart"
    assert cell["bucket"] == attribution.BREAKS_ADDED


# ------------------------------------------------------- the stale-baseline race

def test_a_baseline_that_moves_after_the_comparison_refuses_the_adoption(scoped):
    """The race the decision moment has to lose safely.

    A person compares three versions, walks away, the day is re-planned under
    them, and they come back and adopt. The proposal's own figures are still
    internally consistent, so nothing about the version itself reveals the
    problem. Adoption must refuse against the day as it stands NOW and name what
    moved.
    """
    baseline = frame(BASE_ROWS)
    authored_against = ref_for(baseline)
    proposal = store.create_proposal(
        channel=CHANNEL, date=DAY, name="authored before the day moved",
        author="dana", rows=frame(BASE_ROWS), baseline_ref=authored_against,
        rows_source="engine-day-plan-with-edits")
    proposal_id = proposal["proposal_id"]

    # Adoption against the unmoved day is allowed.
    assert store.check_adoptable(CHANNEL, DAY, proposal_id,
                                 current_ref=authored_against)["proposal_id"] == proposal_id

    # The day is re-planned: one programme now carries a different length.
    moved = frame([(row[0], row[1], row[2], row[3] + 60.0, row[4], row[5])
                   for row in BASE_ROWS])
    moved_ref = ref_for(moved)
    assert moved_ref["day_sha256"] != authored_against["day_sha256"]

    state = store.staleness(proposal, moved_ref)
    assert state["known"] is True and state["stale"] is True
    assert [item["field"] for item in state["moved"]] == ["day_sha256"]

    with pytest.raises(store.ProposalRefused) as refusal:
        store.check_adoptable(CHANNEL, DAY, proposal_id, current_ref=moved_ref)
    assert refusal.value.code == "stale"
    assert "day_sha256" in refusal.value.reason
    assert refusal.value.reason_he

    # And an explicit re-base, on the record, is what unblocks it.
    store.rebase(CHANNEL, DAY, proposal_id, actor="yossi", new_ref=moved_ref,
                 note="הבדיקה חוזרת: הגרסה עומדת גם מול היום המעודכן")
    assert store.check_adoptable(CHANNEL, DAY, proposal_id, current_ref=moved_ref)


def test_a_settings_move_under_a_proposal_is_also_a_baseline_move(scoped):
    """The day's rows can be identical while the decision basis underneath moved."""
    baseline = frame(BASE_ROWS)
    proposal = store.create_proposal(
        channel=CHANNEL, date=DAY, name="authored under other settings",
        author="dana", rows=baseline, baseline_ref=ref_for(baseline),
        rows_source="engine-day-plan-with-edits")
    tightened = dict(SETTINGS_BASIS, min_retention_floor=0.80)
    moved_ref = ref_for(baseline, settings=tightened)
    state = store.staleness(proposal, moved_ref)
    assert state["stale"] is True
    assert "settings.min_retention_floor" in [item["field"] for item in state["moved"]]
    with pytest.raises(store.ProposalRefused) as refusal:
        store.check_adoptable(CHANNEL, DAY, proposal["proposal_id"],
                              current_ref=moved_ref)
    assert refusal.value.code == "stale"


def test_an_adoption_that_beat_the_race_still_refuses_a_second_one(scoped):
    """Two people adopting two versions of one day at the same moment.

    The second adoption must lose by name, and the loser must be able to read
    which version took the day.
    """
    baseline = frame(BASE_ROWS)
    ref = ref_for(baseline)
    first = store.create_proposal(channel=CHANNEL, date=DAY, name="דנה",
                                  author="dana", rows=baseline, baseline_ref=ref,
                                  rows_source="engine-day-plan-with-edits")
    second = store.create_proposal(channel=CHANNEL, date=DAY, name="יוסי",
                                   author="yossi", rows=frame(BASE_ROWS),
                                   baseline_ref=ref,
                                   rows_source="engine-day-plan-with-edits")
    store.update_status(CHANNEL, DAY, first["proposal_id"], store.ADOPTED,
                        actor="dana", note="נבחרה", current_ref=ref)
    with pytest.raises(store.ProposalRefused) as refusal:
        store.check_adoptable(CHANNEL, DAY, second["proposal_id"], current_ref=ref)
    assert refusal.value.code == "already_adopted"
    assert "דנה" in refusal.value.reason_he
    # The loser stays readable, which is the whole point of arguing in public.
    assert store.get(CHANNEL, DAY, second["proposal_id"])["status"] == store.PROPOSED


# ---------------------------------------- the dimension against the REAL store

def _resolvable_world(head: dict, today: datetime.date) -> tuple:
    """A ledger the obligations engine can actually resolve for THIS agreement.

    Deliberately not empty frames. The engine now answers ``unknown`` for a world
    with no delivery rows in the measurement window - correctly, since an absent
    source is not a zero - which means an empty world reports ``unknown`` for a
    budget commitment and for a term nobody measures alike. A tripwire built on
    empty frames therefore cannot tell those two apart, and the one below has to.

    So the campaign is named after the agreement's own counterparty and the
    delivery row is placed inside its window, which is what makes a continuously
    measured term come back measured.
    """
    counterparty = head.get("counterparty") or {}
    advertiser = str(counterparty.get("advertiser") or "").strip()
    agency = str(counterparty.get("agency") or "").strip()
    subject = advertiser or agency or "probe-advertiser"
    window = head.get("window") or {}
    start = str(window.get("starts_on") or "")[:10]
    try:
        aired_on = datetime.date.fromisoformat(start)
    except ValueError:
        aired_on = today - datetime.timedelta(days=30)

    campaigns = pd.DataFrame([("PROBE1", subject)],
                             columns=["campaign_id", "advertiser"])
    links = pd.DataFrame(
        [(agency, agency, subject)] if agency else [],
        columns=["agency_id", "agency_name", "advertiser"])
    delivery = pd.DataFrame([
        ("PROBE1", aired_on.isoformat(), "aired", "רשת 13", 20, 600, 40.0, 400_000),
    ], columns=["campaign_id", "broadcast_date", "air_state", "channel", "spots",
                "seconds", "rating_points_planned", "spend_ils"])
    return campaigns, links, delivery


def _measured_terms(heads: list[dict], today: datetime.date) -> list[str]:
    """Which obliging terms in these agreements come back with a real alarm."""
    from kairos.trade import obligations as obligation_engine
    from kairos_api import trade_store

    measured: list[str] = []
    for head in heads:
        campaigns, links, delivery = _resolvable_world(head, today)
        inputs = obligation_engine.Inputs(
            delivery=delivery, campaigns=campaigns, agency_links=links, today=today)
        termset = trade_store.load_termset(str(head["agreement_id"]),
                                          str(head["current_version_id"]))
        for snapshot in obligation_engine.evaluate_all(termset, head, inputs):
            if snapshot["alarm"] != obligation_engine.UNKNOWN:
                measured.append(str(snapshot["term_id"]))
    return measured


def test_the_tripwire_below_can_actually_trip():
    """The control that keeps the next test from passing for the wrong reason.

    A detector nobody has seen fire is not a detector. This seeds a budget
    commitment into the same evaluation path the next test uses and proves it
    comes back MEASURED, so a green result there means "nothing measurable is on
    disk" rather than "the probe was blind".

    Written after the probe below WAS blind: it used empty frames, and once the
    engine started answering unknown for an unresolvable world - the right
    behaviour - a seeded budget commitment would have slipped past it silently.
    """
    from kairos.trade import obligations as obligation_engine

    head = {
        "agreement_id": "agr-probe", "title": "probe",
        "counterparty": {"advertiser": "probe-advertiser"},
        "window": {"starts_on": "2026-01-01", "ends_on": "2026-12-31"},
    }
    termset = {
        "version_id": "v-probe", "agreement_id": "agr-probe",
        "instances": [{
            "instance_id": "i-budget", "term_id": "budget-commitment",
            "params": {"amount": {"amount": 1_000_000, "basis": "ratecard"}},
            "scope": {}, "window": {},
        }],
    }
    today = datetime.date(2026, 6, 15)
    campaigns, links, delivery = _resolvable_world(head, today)
    inputs = obligation_engine.Inputs(delivery=delivery, campaigns=campaigns,
                                      agency_links=links, today=today)
    (snapshot,) = obligation_engine.evaluate_all(termset, head, inputs)
    assert snapshot["alarm"] != obligation_engine.UNKNOWN, (
        "the probe world cannot measure even a budget commitment, so the tripwire "
        "below would never fire")
    assert snapshot["standing"]["counted"] == 400_000.0

    # And the same world still answers unknown for a term nobody measures, which
    # is the discrimination the tripwire depends on.
    untracked = dict(termset, instances=[{
        "instance_id": "i-av", "term_id": "added-value-media",
        "params": {"percent": 8}, "scope": {}, "window": {}}])
    (unmeasured,) = obligation_engine.evaluate_all(untracked, head, inputs)
    assert unmeasured["alarm"] == obligation_engine.UNKNOWN


@pytest.mark.realdata
def test_the_seed_carries_a_commitment_the_dimension_can_actually_measure():
    """The seed must keep an agreement whose standing a version can move.

    This began life as the opposite assertion. When it was first written the only
    approved agreement carried one obliging term - added-value-media - which the
    engine routes to the untracked path, so the commitments dimension was correct,
    honest and unable to demonstrate itself on real data. A second agreement has
    since been approved carrying a budget commitment and a TRP guarantee, so the
    dimension can now show a real commitment moving, and the useful assertion is
    the one that stops that capability from being deleted by accident.
    """
    from kairos_api import trade_store

    approved = [head for head in trade_store.list_agreements()
                if head.get("current_version_id")]
    if not approved:
        pytest.skip("no approved agreement is seeded on this tree")

    measured = _measured_terms(approved, datetime.date(2026, 6, 15))
    assert measured, (
        "no approved agreement carries a continuously measured obligation any more, "
        "so the day comparison can no longer demonstrate a commitment moving")


@pytest.mark.realdata
def test_a_real_agreement_shows_commitments_advancing_and_endangered(scoped):
    """The demonstration itself, end to end, on a real approved agreement.

    One day, two versions of it, and the signed commitments of a real counterparty
    behind them: the version that starves the day endangers what it was carrying and
    the version that carries it advances the same commitments. This is the claim the
    comparison surface exists to make, so it is measured against the real agreement
    store rather than against a fixture.

    ``scoped`` is not optional decoration. Written without it, this test created
    two proposals under ``data/day_proposals`` on the real tree, one directory away
    from the seeded demo versions of 2024-11-01. The agreements are read from the
    real store on purpose; everything this test WRITES goes to ``tmp_path``.
    """
    from kairos_api import trade_store

    head = next((item for item in trade_store.list_agreements()
                 if item.get("current_version_id")
                 and _measured_terms([item], datetime.date(2026, 6, 15))), None)
    if head is None:
        pytest.skip("no approved agreement carries a measurable obligation")
    termset = trade_store.load_termset(str(head["agreement_id"]),
                                       str(head["current_version_id"]))
    day = "2026-03-10"
    advertiser = (head.get("counterparty") or {}).get("advertiser") or "probe"
    delivery = pd.DataFrame([
        ("CS1", "2026-02-01", "aired", CHANNEL, 40, 1200, 60.0, 900_000),
        ("CS1", day, "scheduled", CHANNEL, 50, 1500, 70.0, 400_000),
    ], columns=["campaign_id", "broadcast_date", "air_state", "channel", "spots",
                "seconds", "rating_points_planned", "spend_ils"])
    context = {
        "approved": [(head, termset)],
        "delivery": delivery,
        "campaigns": pd.DataFrame([("CS1", advertiser)],
                                  columns=["campaign_id", "advertiser"]),
        "links": pd.DataFrame(columns=["agency_id", "agency_name", "advertiser"]),
        "today": datetime.date(2026, 6, 15),
    }
    baseline = frame(BASE_ROWS, day=day)
    ref = ref_for(baseline)
    ids = []
    for name, factor in (("מצמצם את היום", 0.7), ("נושא את היום", 1.4)):
        rows = frame([(row[0], row[1], row[2], row[3] * factor, row[4], row[5])
                      for row in BASE_ROWS], day=day)
        ids.append(store.create_proposal(
            channel=CHANNEL, date=day, name=name, author="probe", rows=rows,
            baseline_ref=ref, rows_source="engine-day-plan-with-edits")["proposal_id"])

    payload = day_compare.compare(
        CHANNEL, day, ids, baseline_rows=baseline, baseline_ref=ref,
        caps={"max_daily_ad_seconds": 3000.0}, trade_context=context)
    starved, carried = payload["sides"][0], payload["sides"][1]

    assert starved["commitments"]["counts"][standing.ENDANGERS] >= 1
    assert "בסיכון" in starved["headline"]
    assert carried["commitments"]["counts"][standing.ADVANCES] >= 1
    assert "מתקדמות" in carried["headline"] or "מתקדמת" in carried["headline"]

    # The same commitment, moved in both directions by the two versions.
    def projection(side, term):
        return next(item for item in side["commitments"]["obligations"]
                    if item["term_id"] == term)

    budget_down = projection(starved, "budget-commitment")
    budget_up = projection(carried, "budget-commitment")
    assert budget_down["side"]["projection"] < budget_down["baseline"]["projection"]
    assert budget_up["side"]["projection"] > budget_up["baseline"]["projection"]
    assert budget_down["verdict"] == standing.ENDANGERS
    assert budget_up["verdict"] == standing.ADVANCES


@pytest.mark.realdata
def test_a_bound_rule_resolves_back_to_its_clause_and_its_quote():
    """Rule attribution: a live rule id names the agreement, clause and wording."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kairos_api import trade_store
    from kairos_api.trade_api import router

    approved = next((head for head in trade_store.list_agreements()
                     if head.get("current_version_id")), None)
    if approved is None:
        pytest.skip("no approved agreement is seeded on this tree")
    termset = trade_store.load_termset(str(approved["agreement_id"]),
                                       str(approved["current_version_id"]))
    instance = next((item for item in termset.get("instances", [])
                     if item.get("citations")), None)
    if instance is None:
        pytest.skip("the approved termset carries no cited instance")
    rule_id = (f"TRD:{approved['agreement_id']}:{approved['current_version_id']}"
               f":{instance['instance_id']}")

    app = FastAPI()
    app.include_router(router)
    payload = TestClient(app).get(f"/api/trade/attribution/{rule_id}").json()
    assert payload["resolved"] is True
    assert payload["agreement_id"] == approved["agreement_id"]
    citation = payload["term"]["citations"][0]
    assert citation["quote"].strip(), "a bound rule must quote the wording it came from"
    assert citation["clause_id"]


@pytest.mark.realdata
def test_an_unbound_rule_id_answers_unresolved_rather_than_pretending():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from kairos_api.trade_api import router

    app = FastAPI()
    app.include_router(router)
    payload = TestClient(app).get(
        "/api/trade/attribution/TRD:agr-does-not-exist:v-nope:i-nope").json()
    assert payload["resolved"] is False
    assert payload["trade_rule"] is True


# --------------------------------------------------------------- latent breakage

def test_scaling_the_delivery_ledger_does_not_rely_on_deprecated_pandas(scoped):
    """The projection writes floats into integer columns of the delivery ledger.

    ``day_compare_standing._projected_delivery`` multiplies ``spots``, ``seconds``
    and ``spend_ils`` by a fractional capacity factor and assigns the result back
    in place. A real ledger carries those three as int64, so a factor that lands
    on a fraction writes a float into an integer column. Pandas currently promotes
    the column and warns; the warning says it will RAISE in a future version, and
    an exception there takes the entire commitments dimension down with it.

    No money is lost today - the values below are preserved exactly - so this is
    pinned as latent breakage on a dependency upgrade rather than as a live wrong
    number. ``simplefilter`` is explicit because the warning is emitted once per
    code location per process and another test may already have consumed it.
    """
    import warnings

    delivery = pd.DataFrame([
        ("C1", DAY, "scheduled", CHANNEL, 48, 1501, 25.0, 300_001),
    ], columns=["campaign_id", "broadcast_date", "air_state", "channel", "spots",
                "seconds", "rating_points_planned", "spend_ils"])
    assert str(delivery["spots"].dtype) == "int64"
    assert str(delivery["spend_ils"].dtype) == "int64"

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        scaled = standing._projected_delivery(delivery, CHANNEL, DAY, 0.8)
    assert FutureWarning in {item.category for item in caught}, (
        "the in-place float assignment into an int64 column no longer warns; if "
        "pandas now raises instead, the commitments dimension is broken")

    # Values survive the promotion, which is why this is latent and not live.
    assert float(scaled.loc[0, "spots"]) == pytest.approx(38.4)
    assert float(scaled.loc[0, "seconds"]) == pytest.approx(1200.8)
    assert float(scaled.loc[0, "spend_ils"]) == pytest.approx(240_000.8)
