"""Top and Tail: the paired creatives, the validity window, and the measurement.

The trade note is the authority these tests are written against:

    A campaign carries many creatives, up to twenty versions. A common structure
    is a 10 second spot plus a 6 second closer, with the constraint that they air
    in the same break separated by exactly one or two other advertisements. These
    are hard placement constraints and the optimiser has to honour them. Each
    creative also carries a validity window: until when it may be scheduled.

Three things are proved here.

**The rule is load bearing.** ``test_removing_the_rule_stops_the_detection``
takes the authored pair away and shows the same broken ordering reports nothing.
That is the failure the old behaviour had on every day of its life: the product
could not express the constraint, so it could not name a single breach of it.

**The three states are three.** A pair whose closer is not in the traffic file is
unknown, and a pair placed wrongly is violated, and the two are never the same
answer.

**The measurement is reproducible.** The numbers in
``docs/top-and-tail-design.md`` are asserted here against the shipped daily file,
so a change that moves them fails rather than quietly restating them.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from kairos.data.loaders import load_daily_input
from kairos.optimize._frequency_rules import (
    BETWEEN,
    COLUMNS,
    PAIR_SEPARATION,
    FrequencyRule,
    FrequencyRuleSet,
    load_frequency_rules,
    pair_rules,
)
from kairos.optimize._pair_placement import SATISFIED, UNKNOWN, VIOLATED
from kairos.optimize.frequency import SpotView, enforce_spots, pair_verdicts
from kairos_api import campaigns_assets, campaigns_assets_constraints as constraints

ROOT = Path(__file__).resolve().parents[1]
SHIPPED_DAY = ROOT / "data" / "daily_input" / "Wally_Prime_Reshet_Example_2025-04-27.csv"

# The strongest candidate on the shipped file: a 28 second Mercedes lead and its
# 6 second closer, both real house numbers read from the traffic log.
LEAD = "CID179035"
CLOSER = "HID179039"
CAMPAIGN = "2025-04 - כלמוביל - מרצדס — מרצדס STAR LEASE"


def a_pair(lead: str = LEAD, closer: str = CLOSER, low: float = 1.0, high: float = 2.0) -> FrequencyRule:
    return FrequencyRule(
        rule_id="MERCEDES_TOP_AND_TAIL",
        limit_type=PAIR_SEPARATION,
        scope="campaign",
        campaign=CAMPAIGN,
        pair_lead=lead,
        pair_closer=closer,
        value=low,
        value_max=high,
        unit=BETWEEN,
    )


def a_spot(key: int, house: str, break_id: str = "b1", campaign: str = CAMPAIGN) -> SpotView:
    return SpotView(
        key=key,
        advertiser="כלמוביל",
        campaign=campaign,
        ad=f"version of {house}",
        break_id=break_id,
        position=None,
        minute=None,
        house_number=house,
    )


# --------------------------------------------------------------------------
# The vocabulary
# --------------------------------------------------------------------------

def test_the_shipped_rule_file_parses_with_the_wider_vocabulary():
    """Widening the rule file must not have cost the rule already in it."""
    ruleset = load_frequency_rules()
    assert ruleset.skipped == []
    assert [rule.rule_id for rule in ruleset.rules] == ["DEFAULT_ONE_PER_BREAK"]
    assert pair_rules(ruleset.rules) == []


def test_the_rule_file_holds_every_new_column():
    with (ROOT / "data" / "frequency_rules.csv").open(encoding="utf-8-sig") as handle:
        header = next(csv.reader(handle))
    assert header == COLUMNS
    for column in ("pair_lead", "pair_closer", "value_max"):
        assert column in header


@pytest.mark.parametrize(
    "row, fragment",
    [
        ({"pair_lead": "A", "campaign": "C"}, "both pair_lead and pair_closer"),
        ({"pair_lead": "A", "pair_closer": "A", "campaign": "C"}, "same creative twice"),
        ({"pair_lead": "A", "pair_closer": "B"}, "needs the campaign"),
        ({"pair_lead": "A", "pair_closer": "B", "campaign": "C", "value_max": "0"}, "0 <= value <= value_max"),
    ],
)
def test_a_pair_row_that_would_have_to_be_guessed_is_refused_with_a_reason(row, fragment):
    """A malformed pair is skipped with a stated reason, never half honoured."""
    from kairos.optimize._frequency_rules import rule_from_row

    base = {"rule_id": "P", "limit_type": PAIR_SEPARATION, "scope": "campaign", "value": "1"}
    rule, reason = rule_from_row({**base, **row})
    assert rule is None
    assert fragment in reason


# --------------------------------------------------------------------------
# The enforcement, and the proof that the rule is what does the work
# --------------------------------------------------------------------------

def test_one_other_between_satisfies_and_names_the_break():
    spots = [a_spot(0, LEAD), a_spot(1, "OTHER"), a_spot(2, CLOSER)]
    verdict, = pair_verdicts(spots, [a_pair()])
    assert verdict.state == SATISFIED
    assert verdict.others_between == 1
    assert verdict.break_id == "b1"
    assert verdict.matched_on == "house_number"


def test_adjacent_creatives_violate_and_say_why_in_both_languages():
    spots = [a_spot(0, LEAD), a_spot(1, CLOSER)]
    verdict, = pair_verdicts(spots, [a_pair()])
    assert verdict.state == VIOLATED
    assert verdict.others_between == 0
    assert "0 other advertisements between them" in verdict.reason
    assert "תשדירים אחרים" in verdict.reason_he
    assert LEAD in verdict.reason_he and CLOSER in verdict.reason_he


def test_removing_the_rule_stops_the_detection():
    """The proof the rule carries the behaviour, not the surface around it.

    The same broken ordering is judged twice. With the pair authored it is a
    violation with a reason. With no pair authored the product has nothing to say
    about it, which is precisely what it had to say about every Top and Tail
    agreement in the trade before this rule existed.
    """
    broken = [a_spot(0, LEAD), a_spot(1, CLOSER)]

    with_rule = pair_verdicts(broken, [a_pair()])
    assert [item.state for item in with_rule] == [VIOLATED]

    without_rule = pair_verdicts(broken, [])
    assert without_rule == []


def test_a_closer_that_is_not_in_the_file_is_unknown_and_not_a_violation():
    """Honest math: three states, and never a zero for an absent thing."""
    spots = [a_spot(0, LEAD), a_spot(1, "OTHER")]
    verdict, = pair_verdicts(spots, [a_pair()])
    assert verdict.state == UNKNOWN
    assert verdict.others_between is None
    assert CLOSER in verdict.reason
    assert "אינו מופיע בקובץ הטראפיק" in verdict.reason_he


def test_two_creatives_in_different_breaks_violate_the_same_break_half():
    spots = [a_spot(0, LEAD, "b1"), a_spot(1, CLOSER, "b2")]
    verdict, = pair_verdicts(spots, [a_pair()])
    assert verdict.state == VIOLATED
    assert verdict.break_id == ""
    assert "never in the same one" in verdict.reason


def test_a_creative_airing_twice_in_one_break_is_judged_on_its_best_occurrence():
    spots = [a_spot(0, CLOSER), a_spot(1, "X"), a_spot(2, "Y"), a_spot(3, "Z"), a_spot(4, LEAD), a_spot(5, "W"), a_spot(6, CLOSER)]
    verdict, = pair_verdicts(spots, [a_pair()])
    assert verdict.state == SATISFIED
    assert verdict.others_between == 1


def test_a_pair_never_removes_a_spot():
    """A pair judges an ordering. It must not be able to shrink one."""
    spots = [a_spot(0, LEAD), a_spot(1, CLOSER)]
    ruleset = FrequencyRuleSet(rules=[a_pair()])
    outcome = enforce_spots(spots, ruleset)
    assert outcome.kept == [0, 1]
    assert outcome.dropped == []
    assert outcome.pair_states == {SATISFIED: 0, VIOLATED: 1, UNKNOWN: 0}


def test_no_authored_pair_leaves_enforcement_exactly_as_it_was():
    spots = [a_spot(0, LEAD), a_spot(1, CLOSER)]
    outcome = enforce_spots(spots, FrequencyRuleSet())
    assert outcome.kept == [0, 1]
    assert outcome.pairs == []


def test_a_pair_does_not_reach_across_campaigns():
    spots = [a_spot(0, LEAD), a_spot(1, "OTHER"), a_spot(2, CLOSER, campaign="a different campaign")]
    verdict, = pair_verdicts(spots, [a_pair()])
    assert verdict.state == UNKNOWN


# --------------------------------------------------------------------------
# The validity window
# --------------------------------------------------------------------------

def test_a_creative_with_no_recorded_window_is_unknown_with_a_way_out():
    window = constraints.validity_window({"valid_from": "", "valid_until": ""})
    assert window["state"] == "unknown"
    assert window["valid_until"] is None
    assert "Record the last day" in window["path_en"]
    assert window["path_he"]
    assert constraints.schedulable_on(window, "2025-04-27")["state"] == "unknown"


def test_a_recorded_window_answers_all_three_days_around_it():
    window = constraints.validity_window({"valid_from": "2025-04-01", "valid_until": "2025-04-30"})
    assert window["state"] == "real"
    assert constraints.schedulable_on(window, "2025-04-27")["state"] == "within"
    assert constraints.schedulable_on(window, "2025-05-01")["state"] == "expired"
    assert constraints.schedulable_on(window, "2025-03-01")["state"] == "not_yet"
    assert "2025-04-30" in constraints.schedulable_on(window, "2025-05-01")["reason_he"]


def test_every_shipped_asset_reports_its_window_unknown_rather_than_open():
    """The traffic log declares no window, and the ledger says so on every row."""
    records = [item for group in campaigns_assets.assets_by_campaign().values() for item in group]
    assert records
    assert all(item["validity"]["state"] == "unknown" for item in records)
    assert "valid_until" in campaigns_assets.COLUMNS


# --------------------------------------------------------------------------
# The measurement on the shipped daily file
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shipped_day():
    return load_daily_input(SHIPPED_DAY)


def test_the_measurement_on_the_shipped_file(shipped_day):
    """The number this whole piece exists to produce, asserted so it cannot drift.

    Nine campaign creatives on the shipped day look like a Top and Tail pair by
    duration and version name, across seven of the file's fifty one campaigns.
    One of the nine airs the way the trade says it must.
    """
    measured = constraints.measure_file(shipped_day)
    assert measured["campaigns_in_file"] == 51
    assert measured["candidate_count"] == 9
    assert measured["campaigns_with_a_candidate"] == 7
    assert measured["states"] == {SATISFIED: 1, VIOLATED: 8, UNKNOWN: 0}


def test_the_wider_duration_only_reading_is_also_recorded(shipped_day):
    """Duration alone, with no version-name evidence, as the looser upper bound."""
    measured = constraints.measure_file(shipped_day, require_shared_name=False)
    assert measured["candidate_count"] == 17
    assert measured["campaigns_with_a_candidate"] == 14
    assert measured["states"] == {SATISFIED: 2, VIOLATED: 15, UNKNOWN: 0}


def test_the_mercedes_pair_is_measured_broken_on_the_real_file(shipped_day):
    """The named example, judged on the real broadcast order rather than a fixture."""
    views = constraints.spot_views(shipped_day)
    verdicts = pair_verdicts(views, [a_pair()])
    assert verdicts
    assert all(item.state == VIOLATED for item in verdicts)
    assert verdicts[0].break_id == "20:40:09"
    assert verdicts[0].others_between == 11


def test_the_candidates_name_only_the_operators_own_channel(shipped_day):
    """The competitor boundary: no candidate carries a channel at all.

    The traffic file has no channel column, so nothing here can name a rival
    channel from it. This asserts the shape of what the measurement emits rather
    than trusting that.
    """
    for candidate in constraints.candidate_pairs(shipped_day):
        assert set(candidate) == {
            "campaign", "lead_house_number", "lead_version_name", "lead_seconds",
            "closer_house_number", "closer_version_name", "closer_seconds",
            "shared_name_words",
        }


# --------------------------------------------------------------------------
# The verification surface
# --------------------------------------------------------------------------

def test_a_pod_carries_the_three_states_and_says_how_many_pairs_are_authored():
    from kairos_api import break_api_pod

    pods = break_api_pod.pods_for_day("2025-04-27")
    assert pods
    for pod in pods:
        block = pod["creative_pairs"]
        assert set(block) == {"verdicts", "states", "authored", "errors"}
        assert set(block["states"]) == {SATISFIED, VIOLATED, UNKNOWN}
        # No pair is authored on the shipped rule file, so no pod may claim one.
        assert block["authored"] == 0
        assert block["verdicts"] == []
        assert not any(error["kind"] == "pair_separation" for error in pod["verification"]["errors"])


def test_a_violated_pair_becomes_a_verification_entry_and_an_unknown_does_not():
    from kairos.optimize._pair_placement import pair_verdicts as judge

    spots = [a_spot(0, LEAD), a_spot(1, CLOSER)]
    violated = judge(spots, [a_pair()])
    entries = constraints.pod_pair_errors(violated)
    assert [item["kind"] for item in entries] == ["pair_separation"]
    assert entries[0]["detail"] and entries[0]["detail_he"]
    assert entries[0]["spot_key"] == 1

    unknown = judge([a_spot(0, LEAD)], [a_pair()])
    assert [item.state for item in unknown] == [UNKNOWN]
    assert constraints.pod_pair_errors(unknown) == []
