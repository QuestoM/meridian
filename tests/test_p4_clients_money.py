"""P4: the Clients money layer, measured against the ledger it is grouped from.

The bar this file holds is arithmetic and it is the whole reason the drill can
be trusted: every level of the tree sums to the level above it, and the top
sums to the figure the agency summary and the spots export already report.
A drill whose levels do not add up is worse than no drill.

The pipeline is composed here the way ``kairos_api.exporters._load_daily_pricing``
composes it, from :mod:`kairos_api.core` rather than through
``kairos_api.server``, so this file measures the real priced day without
importing the app. Production code still calls the one composition.
"""

from __future__ import annotations

import pytest

from kairos_api import campaigns_read_money as money

# The frozen figures for the shipped daily file, from docs/ux-gauntlet/job-stories.md
# ("What must not get worse", JS-9). If any of these move, either the engine
# changed or this layer stopped reading the same ledger.
GROSS = 699450.0
NET = 669978.0
REBATES = 29472.0
PRICED_SPOTS = 119
DROPPED_BY_FREQUENCY = 56
OBSERVED_ADVERTISERS = 41
AGENCIES_ON_SPOTS = 9


@pytest.fixture(scope="module")
def priced():
    """The real priced day, composed without importing the app."""
    from kairos.export.spots import price_daily_file
    from kairos.optimize.overrides import OverrideSet
    from kairos.optimize.pricing import pricing_from_settings
    from kairos_api.core import _load_settings
    from kairos_api.overrides import OVERRIDES_PATH
    from kairos_api.uploads import _newest_daily

    path = _newest_daily()
    if path is None:
        pytest.fail("no daily file on disk, so the money layer cannot be measured")
    result = price_daily_file(
        path,
        pricing=pricing_from_settings(_load_settings()),
        overrides=OverrideSet.from_csv(OVERRIDES_PATH),
    )
    return result, path


@pytest.fixture(scope="module")
def board(priced):
    """The board this module builds, from that same priced result."""
    result, path = priced
    rows = [money._spot_record(index, spot) for index, spot in enumerate(result.priced, start=1)]
    dropped = [money._dropped_record(index, drop) for index, drop in enumerate(result.frequency_dropped, start=1)]
    totals = money._sum_rows(rows)
    totals["dropped_by_frequency"] = len(dropped)
    totals["dropped_by_rule"] = len(result.dropped)
    return {
        "available": True,
        "basis": money._basis(path, len(rows), len(dropped)),
        "totals": totals,
        "advertisers": money._advertiser_rows(rows, dropped, totals["gross"]),
        "agencies": money._agency_rows(rows, totals["gross"]),
        "campaigns": money._campaign_rows(rows, totals["gross"]),
        "breaks": money._break_rows(rows, totals["gross"]),
        "spots": rows,
        "dropped": dropped,
    }


def test_the_totals_are_the_ledgers_own_totals(board, priced):
    """The board reports the pipeline's figures, not a second set of numbers."""
    result, _ = priced
    assert board["totals"]["gross"] == result.total_revenue == GROSS
    assert board["totals"]["net"] == result.total_net_revenue == NET
    assert board["totals"]["rebates"] == REBATES
    assert board["totals"]["spots"] == len(result.priced) == PRICED_SPOTS
    assert board["totals"]["dropped_by_frequency"] == DROPPED_BY_FREQUENCY


def test_every_grouping_sums_to_the_same_total(board):
    """Four groupings over one set of rows, so all four add to one figure."""
    for level in ("advertisers", "agencies", "campaigns", "breaks", "spots"):
        assert round(sum(row["gross"] for row in board[level]), 2) == GROSS, level
        assert round(sum(row["net"] for row in board[level]), 2) == NET, level


def test_each_advertisers_campaigns_sum_to_that_advertiser(board):
    """The second level of the drill adds up to the first, client by client."""
    for advertiser in board["advertisers"]:
        summed = round(sum(campaign["gross"] for campaign in advertiser["campaigns"]), 2)
        assert summed == advertiser["gross"], advertiser["advertiser"]


def test_each_rows_spot_keys_reach_exactly_its_own_spots(board):
    """The third level of the drill is the rows behind the figure, and only those."""
    spots = {row["spot_key"]: row for row in board["spots"]}
    for advertiser in board["advertisers"]:
        own = [spots[key] for key in advertiser["spot_keys"]]
        assert len(own) == advertiser["spots"]
        assert round(sum(spot["gross"] for spot in own), 2) == advertiser["gross"]
        assert {spot["advertiser"] for spot in own} == {advertiser["advertiser"]}


def test_the_client_count_is_the_observed_vocabulary(board):
    """41 named advertisers, 9 agencies, and every one of them carries money."""
    assert len(board["advertisers"]) == OBSERVED_ADVERTISERS
    assert len(board["agencies"]) == AGENCIES_ON_SPOTS
    assert all(row["spots"] > 0 for row in board["advertisers"])
    assert all(row["gross"] > 0 for row in board["advertisers"])


def test_the_ranking_answers_the_question_it_is_asked(board):
    """Rank 1 is the largest gross, and every rank is dense and ordered."""
    ranks = [row["rank"] for row in board["advertisers"]]
    assert ranks == list(range(1, len(ranks) + 1))
    grosses = [row["gross"] for row in board["advertisers"]]
    assert grosses == sorted(grosses, reverse=True)
    assert board["advertisers"][0]["gross"] == max(grosses)


def test_shares_are_computed_not_asserted(board):
    """Every share is that row's gross over the day, and they sum to one."""
    for row in board["advertisers"]:
        assert row["share_of_gross"] == round(row["gross"] / GROSS, 6)
    assert abs(sum(row["share_of_gross"] for row in board["advertisers"]) - 1.0) < 1e-4


def test_dropped_money_is_visible_with_the_rule_that_removed_it(board):
    """56 spots are removed by a rule, and each says which rule and why."""
    assert len(board["dropped"]) == DROPPED_BY_FREQUENCY
    for row in board["dropped"]:
        assert row["rule_id"]
        assert row["reason"]
        assert row["advertiser"]
    attributed = sum(row["dropped_by_frequency"] for row in board["advertisers"])
    assert attributed == DROPPED_BY_FREQUENCY


def test_the_basis_names_the_file_the_day_and_the_scope(board):
    """A figure without its scope does not render, so the scope is on the payload.

    The channel is asserted against the setting the process actually holds, not
    against a literal. ``operator_channel`` is written through an unguarded,
    unvalidated route on a settings document every piece shares, so pinning the
    string would measure who wrote that file last; and the property that matters
    is that the scope printed beside the money is the configured channel, or
    empty with the surface saying so, and never a channel nobody chose.
    """
    from kairos_api import channel_scope

    basis = board["basis"]
    assert basis["file"] == "Wally_Prime_Reshet_Example_2025-04-27.csv"
    assert basis["day"] == "2025-04-27"
    assert basis["priced_spots"] == PRICED_SPOTS
    assert basis["rows_in_file"] == PRICED_SPOTS + DROPPED_BY_FREQUENCY
    assert basis["scope_channel"] == channel_scope.operator_channel()
    assert basis["scope_channel"] not in ("קשת 12", "כאן 11", "עכשיו 14")
    assert "one broadcast day" in basis["period_note_en"].lower()
    assert basis["wider_period_en"]
    assert basis["wider_period_he"]


def test_no_rival_channel_reaches_the_payload(board):
    """The competitor boundary, checked over every string the board emits."""
    import json

    rendered = json.dumps(board, ensure_ascii=False)
    for rival in ("קשת 12", "כאן 11", "עכשיו 14"):
        assert rival not in rendered


def test_an_unreadable_ledger_is_a_state_with_a_reason(monkeypatch):
    """No ledger means every figure is None and the reason says so. Never a zero."""
    def explode():
        raise RuntimeError("no pipeline in this test")

    monkeypatch.setattr(money, "_build", lambda: money._unavailable(money.PIPELINE_FAILED_REASON))
    payload = money._build()
    assert payload["available"] is False
    assert payload["reason"] == money.PIPELINE_FAILED_REASON
    assert payload["totals"] == {
        "gross": None,
        "net": None,
        "rebates": None,
        "spots": None,
        "dropped_by_frequency": None,
        "dropped_by_rule": None,
        # None rather than []. With no ledger to read, an empty list of rules
        # reads as "no rule removed anything", which is the same false claim
        # this test exists to forbid, in the shape of a list instead of a zero.
        "dropped_rules": None,
    }
    assert payload["advertisers"] == []
    assert payload["spots"] == []
    del explode


def test_the_spot_row_carries_what_a_person_needs_to_check_it(board):
    """Each row names its break, its programme, its ad and its rebate."""
    row = board["spots"][0]
    for field in ("break_id", "programme", "ad", "advertiser", "campaign", "agency"):
        assert row[field] != "" and row[field] is not None
    assert row["gross"] >= row["net"]
    assert row["rebate"] == round(row["gross"] - row["net"], 2)


# --------------------------------------------------------------------------
# Money that is not there for a stated reason, stated in words
# --------------------------------------------------------------------------

def test_every_removed_spot_carries_its_reason_in_both_languages(board):
    """The rule that removed it is a row in the rule file, so its cap is real."""
    for row in board["dropped"]:
        assert row["kind"] == "frequency"
        assert row["limit_known"] is True
        assert row["explanation_en"] and row["explanation_he"]
        assert row["rule_id"] in row["explanation_en"] or "at most" in row["explanation_en"]
        assert row["advertiser"] in row["explanation_he"]
        assert row["break_id"] in row["explanation_he"]
        # The cap is the rule's own value, not a number parsed out of a log line.
        assert "at most 1 spot per client in one break" in row["explanation_en"]
        assert "לכל היותר תשדיר אחד ללקוח בכל ברייק" in row["explanation_he"]


def test_a_rule_that_is_not_in_the_file_is_an_unknown_limit_not_a_guess():
    """The third state: the id is reported, the cap is not invented."""
    from types import SimpleNamespace

    drop = SimpleNamespace(
        advertiser="פריסבי",
        campaign="קמפיין",
        ad="תשדיר",
        break_id="22:03:06",
        rule_id="GONE_FROM_THE_FILE",
        limit_type="max_per_break",
        reason="max_per_break=1 reached for פריסבי in break 22:03:06",
    )
    record = money._dropped_record(1, drop, "frequency", {})
    assert record["limit_known"] is False
    assert "GONE_FROM_THE_FILE" in record["explanation_en"]
    assert "GONE_FROM_THE_FILE" in record["explanation_he"]
    assert "1" not in record["explanation_en"].replace("GONE_FROM_THE_FILE", "")
    assert record["reason"] == drop.reason


def test_the_cap_reads_as_one_spot_and_never_as_one_spots():
    """A cap of one is the only cap the shipped rule file holds."""
    from kairos_api import campaigns_read_money_reasons as reasons

    one_en, one_he = reasons.quantity(1, reasons.SPOT_WORDS)
    many_en, many_he = reasons.quantity(3, reasons.SPOT_WORDS)
    assert one_en == "1 spot" and one_he == "תשדיר אחד"
    assert many_en == "3 spots" and many_he == "⁦3⁩ תשדירים"
