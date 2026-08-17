"""Adversarial verification of the decision moment and the commitments behind it.

The money identity is attacked next door in ``test_p2_adversarial_attribution.py``,
whose fixtures this module imports. What is attacked here is everything a money
figure cannot answer, and everything that has to hold at the instant somebody
settles a day:

- the sentence the standing dimension renders when it measured nothing,
- the race where the day moves under a proposal between the comparison and the
  adoption, and the race where two people adopt at once,
- the dimension driven against the REAL approved agreements on disk rather than a
  fixture, including the demonstration that a version can move a signed
  commitment in both directions,
- rule attribution back to the clause and the wording it came from,
- and one latent breakage that is not a wrong number today but would become an
  exception on a dependency upgrade.

One defect recorded here originally landed as a strict xfail (an all-unknown
standing rendering as "no change") and has since been fixed, so it stands as a
lock. Its measurements are kept in the docstrings because the figure a defect used
to produce is the cheapest way to check that a fix moved it.
"""

from __future__ import annotations

import datetime

import pandas as pd
import pytest

from kairos_api import day_compare
from kairos_api import day_compare_standing as standing
from kairos_api import day_proposal_store as store
from tests.test_day_proposals import BASE_ROWS, CHANNEL, DAY, frame, ref_for
from tests.test_p2_adversarial_attribution import SETTINGS_BASIS, scoped  # noqa: F401


# ------------------------------------ the standing sentence (was defect 2, HIGH)

def test_an_all_unknown_standing_must_not_read_as_no_change():
    """The headline sentence for a dimension that measured nothing.

    The module's own honesty rule is that no source means unknown and never zero.
    The dimension body always obeyed it - the obligation carries verdict
    ``unknown`` with its reason - but the one line above it did not, and the line
    is what gets read. It fell through to "אין שינוי בעמידה בהתחייבויות", turning
    an unmeasured commitment into a reassurance, and it did so on the live path:
    the מאפיית שדות agreement's only obliging term is one the engine routes to
    the untracked evaluator, so every obligation came back unknown.
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

    Without this, the fix could have been satisfied by deleting the phrase
    altogether, which would lose a true statement instead of correcting a false
    one. Both the measured-unchanged sentence and the not-measured-at-all
    sentence have to survive.
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

    Deliberately not empty frames. The engine answers ``unknown`` for a world with
    no delivery rows in the measurement window - correctly, since an absent source
    is not a zero - which means an empty world reports ``unknown`` for a budget
    commitment and for a term nobody measures alike. A probe built on empty frames
    therefore cannot tell those two apart, and the tripwire below has to.

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
    commitment into the same evaluation path the next test uses and proves it comes
    back MEASURED, so a result there means "nothing measurable is on disk" rather
    than "the probe was blind". Written after the probe WAS blind: it used empty
    frames, and once the engine began answering unknown for an unresolvable world -
    the right behaviour - a seeded budget commitment slipped past it. Measured then:
    empty frames gave unknown, a resolvable ledger gave watch with counted 400,000.
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

    One day, two versions of it, and a real counterparty's signed commitments
    behind them: the version that starves the day endangers what it was carrying,
    the version that carries it advances the same commitments. Measured when
    written, on the סנו framework: the starving version moved the budget
    projection 1,284,000 -> 1,168,800 and the TRP 127.2 -> 107.04; the carrying
    version moved them to 1,300,000 and 130.0.

    ``scoped`` is load-bearing. Written without it, this test created two
    proposals under the real ``data/day_proposals``. Agreements are READ from the
    real store on purpose; everything written goes to ``tmp_path``.
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

def test_scaling_the_delivery_ledger_does_not_rely_on_deprecated_pandas():
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
