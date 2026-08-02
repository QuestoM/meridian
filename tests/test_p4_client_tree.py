"""P4: the client tree, the containment an account manager and an analyst share.

The tree joins four stores and adds no number of its own, so the property that
matters is that it stays a join: an agency row is exactly the sum of its client
rows, a client that has money and no agency is listed rather than dropped, and
the money on it is the same ledger the money board reports.

This runs against the shipped data, read only. Nothing here writes a store.
"""

from __future__ import annotations

import pytest

from kairos_api.campaigns_read_clients import client_tree

GROSS = 699450.0
NET = 669978.0
OBSERVED_ADVERTISERS = 41
AGENCIES = 9
BOOKED_ONLY_CLIENT = "לקוח שהוזמן בבדיקה"


@pytest.fixture()
def tree_with_a_booked_client(monkeypatch, tmp_path):
    """The tree after one campaign is booked for a client with no spot and no agency.

    The store this fixture builds holds exactly that one campaign and nothing
    else, so the counts below measure the thing they name. It used to start from
    the shipped store because that store was empty; it no longer is, since
    ``scripts/seed_campaigns.py`` writes demo campaigns into it, and a fixture
    that inherits those is measuring the seed instead of its own subject. The
    store is redirected to a temporary file, so nothing here writes the tracked one.
    """
    import pandas as pd

    from kairos_api import campaigns_api_store as store

    frame = pd.DataFrame(columns=store.COLUMNS)
    row = store.blank_row()
    row.update({
        "record_type": store.CAMPAIGN,
        "campaign_id": "CMP_TEST",
        "name": "קמפיין בדיקה",
        "advertiser": BOOKED_ONLY_CLIENT,
        "agency_id": "",
        "status": "active",
        "starts_on": "2026-09-01",
        "ends_on": "2026-09-30",
        "data_source": "manual",
    })
    path = tmp_path / "campaigns.csv"
    store.append(frame, row)[store.COLUMNS].to_csv(path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(store, "CAMPAIGNS_PATH", path)
    return client_tree()


@pytest.fixture(scope="module")
def tree():
    return client_tree()


def test_the_tree_is_the_nine_agencies_and_the_forty_one_clients(tree):
    assert tree["available"] is True
    assert tree["counts"]["agencies"] == AGENCIES
    assert tree["counts"]["clients"] == OBSERVED_ADVERTISERS
    assert tree["counts"]["clients_with_money"] == OBSERVED_ADVERTISERS
    assert sum(agency["client_count"] for agency in tree["agencies"]) + len(tree["unlinked"]) == OBSERVED_ADVERTISERS


def test_an_agency_row_is_the_sum_of_its_client_rows(tree):
    """The containment is a sum, not a second computation."""
    for agency in tree["agencies"]:
        gross = round(sum(client["gross"] or 0.0 for client in agency["clients"]), 2)
        net = round(sum(client["net"] or 0.0 for client in agency["clients"]), 2)
        assert gross == agency["gross"], agency["agency_id"]
        assert net == agency["net"], agency["agency_id"]


def test_the_whole_tree_sums_to_the_ledger(tree):
    """Every shekel in the totals is reachable from exactly one row."""
    gross = round(sum(agency["gross"] for agency in tree["agencies"]), 2)
    gross += round(sum(client["gross"] or 0.0 for client in tree["unlinked"]), 2)
    assert gross == tree["totals"]["gross"] == GROSS
    net = round(sum(agency["net"] for agency in tree["agencies"]), 2)
    net += round(sum(client["net"] or 0.0 for client in tree["unlinked"]), 2)
    assert net == tree["totals"]["net"] == NET


def test_every_agency_keeps_its_commercial_terms_and_both_contacts(tree):
    """Bar 3 for P4, on the payload the new surface reads."""
    for agency in tree["agencies"]:
        terms = agency["terms"]
        assert set(terms) == {
            "payment_terms_days",
            "rebate_percent",
            "commission_percent",
            "credit_limit_ils",
            "vat_id",
        }
        assert len(agency["contacts"]) == 2
        assert agency["status"] in {"active", "suspended"}
    named = {agency["agency_id"]: agency for agency in tree["agencies"]}
    assert named["AGY_01"]["terms"]["rebate_percent"] == 4.0
    assert named["AGY_01"]["terms"]["commission_percent"] == 15.0
    assert named["AGY_01"]["terms"]["payment_terms_days"] == 60
    assert named["AGY_01"]["terms"]["credit_limit_ils"] == 3000000.0
    assert named["AGY_01"]["terms"]["vat_id"]
    assert named["AGY_01"]["contacts"][0]["name"]
    assert named["AGY_01"]["contacts"][1]["name"]


def test_every_client_carries_its_identity_and_its_reason(tree):
    """A client is named, or it says why it is not, in both languages."""
    clients = [client for agency in tree["agencies"] for client in agency["clients"]]
    assert len(clients) == OBSERVED_ADVERTISERS
    for client in clients:
        assert client["shown_name"]
        assert client["link_source"] in {"observed", "manual"}
        assert "money_reason_en" in client and "money_reason_he" in client
        if client["gross"] is None:
            assert client["money_reason_he"]


def test_a_client_reachable_only_through_a_campaign_is_in_the_tree(tree_with_a_booked_client):
    """It is listed, it is counted, and its money is none with the reason said."""
    tree = tree_with_a_booked_client
    booked = tree["clients_booked_without_spots"]
    assert [client["advertiser"] for client in booked] == [BOOKED_ONLY_CLIENT]
    record = booked[0]
    assert record["gross"] is None
    assert record["net"] is None
    assert record["spots"] is None
    assert record["campaign_count"] == 1
    assert record["campaigns"][0]["campaign_id"] == "CMP_TEST"
    assert record["money_reason_en"] and record["money_reason_he"]


def test_the_header_count_covers_every_client_the_tree_renders(tree_with_a_booked_client):
    """The count is over the rows, so the header cannot argue with them."""
    tree = tree_with_a_booked_client
    rendered = (
        sum(agency["client_count"] for agency in tree["agencies"])
        + len(tree["unlinked"])
        + len(tree["clients_booked_without_spots"])
    )
    assert rendered == OBSERVED_ADVERTISERS + 1
    assert tree["counts"]["clients"] == rendered
    assert tree["counts"]["campaigns"] == 1
    assert tree["counts"]["clients_with_money"] == OBSERVED_ADVERTISERS


def test_a_booked_client_adds_no_money_to_the_totals(tree_with_a_booked_client):
    """A booking is not revenue, so the ledger totals are the ones they were."""
    tree = tree_with_a_booked_client
    assert tree["totals"]["gross"] == GROSS
    assert tree["totals"]["net"] == NET


def test_no_rival_channel_reaches_the_tree(tree):
    import json

    rendered = json.dumps(tree, ensure_ascii=False)
    for rival in ("קשת 12", "כאן 11", "עכשיו 14"):
        assert rival not in rendered


def test_the_basis_travels_with_the_tree(tree):
    """The scope is on the payload that prints the figures, not only on the board.

    The channel is asserted against the setting the process actually holds
    rather than against a literal, for two reasons. A literal would pass while
    the surface printed a channel nobody configured, which is the failure this
    assertion exists to catch; and ``operator_channel`` is an unguarded,
    unvalidated write on a settings document every piece shares, so a test that
    pins the string measures who wrote that file last rather than this payload.
    When no channel is set the scope is empty and the surface says so, which is
    the honest state and never a guessed channel.
    """
    from kairos_api import channel_scope

    assert tree["basis"]["file"]
    assert tree["basis"]["day"] == "2025-04-27"
    assert tree["basis"]["scope_channel"] == channel_scope.operator_channel()


# --------------------------------------------------------------------------
# An agency with no priced client has no total, which is not a total of zero
# --------------------------------------------------------------------------

@pytest.fixture()
def tree_with_a_new_agency(monkeypatch, tmp_path):
    """The tree after an agency is created and nothing has aired through it.

    This is the state a new agency is in on the day it is created, and it was
    reported as ₪0 gross while its clients correctly reported a dash with the
    reason. Both stores are redirected, so the tracked files are never written.
    """
    import pandas as pd

    from kairos_api import agencies as store

    frame = store._load_frame()
    row = {column: "" for column in frame.columns}
    row.update({
        "agency_id": "AGY_NEW",
        "name": "סוכנות חדשה",
        "display_name": "סוכנות חדשה",
        "status": "active",
        "data_source": "manual",
        "payment_terms_days": "60",
        "rebate_percent": "0",
        "commission_percent": "0",
        "credit_limit_ils": "0",
    })
    path = tmp_path / "agencies.csv"
    pd.concat([frame, pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False, encoding="utf-8-sig")
    monkeypatch.setattr(store, "AGENCIES_PATH", path)
    return client_tree()


def test_a_new_agency_has_no_total_rather_than_a_total_of_zero(tree_with_a_new_agency):
    """The third state survives the sum: none with the reason, never a figure."""
    tree = tree_with_a_new_agency
    fresh = next(agency for agency in tree["agencies"] if agency["agency_id"] == "AGY_NEW")
    assert fresh["clients"] == []
    assert fresh["gross"] is None
    assert fresh["net"] is None
    assert fresh["rebates"] is None
    assert fresh["spots"] is None
    assert fresh["clients_with_money"] == 0
    assert fresh["money_reason_en"] and fresh["money_reason_he"]
    assert "zero" in fresh["money_reason_en"]


def test_the_nine_shipped_agencies_keep_every_figure_they_had(tree_with_a_new_agency):
    """The tri-state changes what an empty agency reports and nothing else."""
    tree = tree_with_a_new_agency
    priced = [agency for agency in tree["agencies"] if agency["gross"] is not None]
    assert len(priced) == AGENCIES
    assert round(sum(agency["gross"] for agency in priced), 2) == GROSS
    assert round(sum(agency["net"] for agency in priced), 2) == NET
    assert tree["totals"]["gross"] == GROSS
    for agency in priced:
        assert agency["clients_with_money"] == len(agency["clients"])
        assert agency["money_reason_en"] == "" and agency["money_reason_he"] == ""
