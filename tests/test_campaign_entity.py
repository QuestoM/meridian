"""The campaign entity: identity, commitment, flight, creative and delivery.

The bars asserted here are the ones the owner named, stated as measurements
rather than as prose.

**A commitment is both halves.** A campaign carries money in shekels and a
rating-point goal against a named target audience, and the audience is a closed
choice from the market list in ``docs/campaign-rate-card-research.md`` rather
than free text.

**A figure nobody committed is not zero.** A blank budget reads as none through
the store, the record and the payload, all the way to the surface.

**A rating point has a base.** This product's ratings are the all-viewers TVR, so
a goal named against any other audience reports its progress as unavailable with
the reason instead of dividing by a base nobody asked for.

**The operator owns one channel.** It is stamped from settings on every booking
and never taken from the request, so no rival name can enter through a body.

**An unreadable property is unknown with a way out.** The four things only a
video file can answer arrive as unknown carrying the action that would resolve
them, in both languages, never as a plausible default.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

HEBREW = set("אבגדהוזחטיכלמנסעפצקרשתךםןףץ")
RIVALS = ("קשת 12", "כאן 11", "עכשיו 14")


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A client over the campaign routes with every store pointed at tmp_path."""
    from kairos_api import (
        agencies,
        campaigns_api,
        campaigns_api_store,
        campaigns_assets,
        campaigns_delivery,
        version_store,
    )

    monkeypatch.setattr(agencies, "AGENCIES_PATH", tmp_path / "agencies.csv")
    monkeypatch.setattr(agencies, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(campaigns_api_store, "CAMPAIGNS_PATH", tmp_path / "campaigns.csv")
    monkeypatch.setattr(campaigns_api_store, "BACKUP_DIR", tmp_path / "_backups")
    monkeypatch.setattr(campaigns_assets, "ASSETS_PATH", tmp_path / "campaign_assets.csv")
    monkeypatch.setattr(campaigns_delivery, "DELIVERY_PATH", tmp_path / "campaign_delivery.csv")
    monkeypatch.setattr(version_store, "snapshot_manual_edit", lambda request, logical: None)

    app = FastAPI()
    app.include_router(agencies.router)
    app.include_router(campaigns_api.router)
    return TestClient(app)


def _agency(client) -> str:
    response = client.post("/api/agencies", json={"agency_id": "AGY_01", "name": "OMD"})
    assert response.status_code == 201, response.text
    return "AGY_01"


ORDER = {
    "name": "מתחתנים 2026",
    "advertiser": "בנק הפועלים",
    "agency_id": "AGY_01",
    "starts_on": "2026-09-06",
    "ends_on": "2026-09-12",
    "budget_ils": 250000,
    "rating_goal_points": 150,
    "rating_goal_audience": "all_viewers",
    "price_model": "cpp",
    "priority": "guaranteed",
    "pacing_mode": "even",
    "brand": "הפועלים",
}


def _book(client, **overrides):
    _agency(client)
    response = client.post("/api/clients/campaigns", json={**ORDER, **overrides})
    return response


# --------------------------------------------------------------------------
# The commitment
# --------------------------------------------------------------------------

def test_a_campaign_carries_money_and_a_rating_goal_against_a_named_audience(client):
    """Both halves, because the owner named both, and the audience is named too."""
    booked = _book(client)
    assert booked.status_code == 201, booked.text
    terms = booked.json()["commitment"]
    assert terms["budget_ils"] == 250000.0
    assert terms["rating_goal_points"] == 150.0
    assert terms["rating_goal_audience"] == "all_viewers"
    assert terms["rating_goal_audience_label_he"] == "כלל הצופים"
    assert terms["rating_goal_measurable"] is True
    assert terms["price_model"] == "cpp"
    assert terms["priority"] == "guaranteed"


def test_a_commitment_nobody_made_is_none_and_never_zero(client):
    """A blank budget is a blank budget. Zero is a promise to spend nothing."""
    booked = _book(client, budget_ils=None, rating_goal_points=None, bonus_ils=None)
    terms = booked.json()["commitment"]
    assert terms["budget_ils"] is None
    assert terms["rating_goal_points"] is None
    assert terms["bonus_ils"] is None

    payload = client.get("/api/clients/campaigns").json()["campaigns"][0]
    assert payload["commitment"]["budget_ils"] is None
    progress = payload["delivery"]["budget_progress"]
    assert progress["percent"] is None
    assert progress["state"] == "unknown"
    assert progress["reason_en"] and progress["reason_he"]


def _ledger(tmp_path, campaign_id: str) -> None:
    """One real aired day for this campaign, so progress has a source to stand on."""
    import pandas as pd

    from kairos_api import campaigns_delivery

    row = {column: "" for column in campaigns_delivery.COLUMNS}
    row.update({
        "campaign_id": campaign_id,
        "broadcast_date": "2026-09-06",
        "air_state": campaigns_delivery.AIRED,
        "channel": "רשת 13",
        "spots": "4", "seconds": "120", "rating_points_planned": "30", "spend_ils": "50000",
        "spots_dropped_by_rule": "0", "source_file": "test.csv",
        "counted_as_of": "2026-09-06T23:00:00", "is_demo": "false",
    })
    pd.DataFrame([row], columns=campaigns_delivery.COLUMNS).to_csv(
        tmp_path / "campaign_delivery.csv", index=False, encoding="utf-8-sig"
    )


def test_a_goal_against_an_audience_this_product_cannot_count_says_so(client, tmp_path):
    """Two currencies are not compared. The progress is unavailable with the reason."""
    booked = _book(client, rating_goal_audience="women_25_54").json()
    _ledger(tmp_path, booked["campaign_id"])
    payload = client.get("/api/clients/campaigns").json()["campaigns"][0]
    assert payload["commitment"]["rating_goal_measurable"] is False
    assert payload["commitment"]["rating_goal_reason_he"]
    assert payload["delivery"]["available"] is True
    progress = payload["delivery"]["rating_progress"]
    assert progress["percent"] is None
    assert progress["state"] == "unavailable"
    assert "audience" in progress["reason_en"]
    assert set(progress["reason_he"]) & HEBREW
    # The money half is countable in its own currency, so it is counted.
    assert payload["delivery"]["budget_progress"]["percent"] == 20.0


def test_no_source_means_unknown_progress_and_never_a_confident_zero(client):
    """Zero counted out of no source is not zero delivered. It is nobody knowing."""
    _book(client)
    delivery = client.get("/api/clients/campaigns").json()["campaigns"][0]["delivery"]
    assert delivery["available"] is False
    assert delivery["aired"]["spend_ils"] == 0
    for key in ("rating_progress", "budget_progress"):
        assert delivery[key]["percent"] is None, key
        assert delivery[key]["state"] == "unknown", key
        assert set(delivery[key]["reason_he"]) & HEBREW, key


def test_the_audience_list_is_the_market_list_and_says_which_one_counts(client):
    """The vocabulary travels on the payload with the measurable flag on each entry."""
    payload = client.get("/api/clients/campaigns").json()
    audiences = payload["target_audiences"]
    assert [entry["value"] for entry in audiences][0] == "all_viewers"
    assert {entry["value"] for entry in audiences} >= {
        "all_viewers", "adults_18_plus", "women_25_54", "men_25_54", "children_4_14",
    }
    assert sum(1 for entry in audiences if entry["measurable"]) == 1
    for entry in audiences:
        assert entry["label_en"] and entry["label_he"]
        assert set(entry["label_he"] + entry["reason_he"]) & HEBREW
    assert "research" in payload["target_audience_source_en"]


def test_a_commitment_outside_the_vocabulary_is_refused_in_both_languages(client):
    """Nothing is silently coerced, and the refusal is readable in the flow's language."""
    _agency(client)
    attempts = {
        "audience": {"rating_goal_audience": "women_35_to_infinity"},
        "price model": {"price_model": "barter"},
        "priority": {"priority": "urgent"},
        "budget below zero": {"budget_ils": -1},
        "rating goal below zero": {"rating_goal_points": -5},
    }
    for label, override in attempts.items():
        response = client.post("/api/clients/campaigns", json={**ORDER, **override})
        assert response.status_code == 400, f"{label} was not refused"
        detail = response.json()["detail"]
        assert detail["message_en"] and detail["message_he"], label
        assert detail["message_en"] != detail["message_he"], label
        assert set(detail["message_he"]) & HEBREW, label


# --------------------------------------------------------------------------
# The channel, and the boundary it draws
# --------------------------------------------------------------------------

def test_the_channel_is_stamped_from_settings_and_never_from_the_request(client):
    """A booking is the operator's own inventory, so the body cannot name a channel."""
    from kairos_api import campaigns_commitment

    booked = _book(client, channel="קשת 12")
    assert booked.status_code == 201
    assert booked.json()["channel"] == campaigns_commitment.operator_channel()
    assert booked.json()["channel"] != "קשת 12"


def test_no_rival_channel_name_reaches_the_campaign_payload(client):
    """The whole payload is searched, not one field, because one leak is the defect."""
    import json

    _book(client)
    rendered = json.dumps(client.get("/api/clients/campaigns").json(), ensure_ascii=False)
    for rival in RIVALS:
        assert rival not in rendered, rival


# --------------------------------------------------------------------------
# The demo marker
# --------------------------------------------------------------------------

def test_a_campaign_an_operator_books_is_not_a_demo_row(client):
    """The column is written false on a real booking and the payload reads it."""
    booked = _book(client)
    assert booked.json()["is_demo"] is False
    payload = client.get("/api/clients/campaigns").json()
    assert payload["demo_count"] == 0
    assert payload["booked_count"] == 1
    assert payload["campaigns"][0]["demo"] == {"is_demo": False}


def test_a_demo_row_carries_its_marking_in_both_languages(client):
    """A seeded row says it is one, says what is real on it and says how to replace it."""
    from kairos_api import campaigns_api_store as store

    _book(client)
    frame = store.load_frame()
    frame.loc[0, "is_demo"] = "true"
    store.write_frame(frame)
    campaign = client.get("/api/clients/campaigns").json()["campaigns"][0]
    assert campaign["is_demo"] is True
    demo = campaign["demo"]
    assert demo["label_he"] == "הדגמה"
    assert set(demo["meaning_he"]) & HEBREW
    assert set(demo["replace_he"]) & HEBREW
    assert not set(demo["meaning_en"]) & HEBREW


# --------------------------------------------------------------------------
# Delivery and creative, with nothing on disk
# --------------------------------------------------------------------------

def test_with_no_ledger_the_delivery_is_unknown_and_names_the_missing_feed(client):
    """Unavailable with a path, never a zero delivery and never a silent absence."""
    _book(client)
    payload = client.get("/api/clients/campaigns").json()
    delivery = payload["delivery"]
    assert delivery["available"] is False
    assert delivery["campaigns_with_a_source"] == 0
    assert "delivery" in delivery["reason_en"].lower()
    assert set(delivery["reason_he"]) & HEBREW
    # The path forward speaks the trade's language, not the developer's: it
    # names the missing as-run feed and promises unknown-not-guessed, and no
    # script path or code artifact leaks into an operator-facing sentence.
    assert "as-run" in delivery["path_forward_en"]
    assert "unknown" in delivery["path_forward_en"]
    assert "scripts/" not in delivery["path_forward_en"]
    assert "As Run" in delivery["path_forward_he"]
    assert "scripts/" not in delivery["path_forward_he"]

    own = payload["campaigns"][0]["delivery"]
    assert own["available"] is False
    assert own["aired"]["spots"] == 0
    assert own["rating_progress"]["percent"] is None


def test_with_no_creative_the_campaign_says_so_rather_than_showing_nothing(client):
    """An empty creative list is a stated state with a reason in both languages."""
    _book(client)
    campaign = client.get("/api/clients/campaigns").json()["campaigns"][0]
    assert campaign["assets"] == []
    summary = campaign["assets_summary"]
    assert summary["count"] == 0
    assert summary["seconds_total"] is None
    assert summary["reason_en"] and set(summary["reason_he"]) & HEBREW


def test_the_detail_and_the_board_report_the_same_campaign(client):
    """One assembly, two routes, so a drawer cannot disagree with the row that opened it."""
    booked = _book(client).json()
    board = client.get("/api/clients/campaigns").json()["campaigns"][0]
    detail = client.get(f"/api/clients/campaigns/{booked['campaign_id']}/detail")
    assert detail.status_code == 200
    assert detail.json()["campaign"] == board
    assert client.get("/api/clients/campaigns/CMP_404/detail").status_code == 404


def test_an_unreadable_creative_property_is_unknown_with_the_way_out(client):
    """The four things only a video can answer, each carrying its own action."""
    from kairos_api import campaigns_assets

    for kind in ("media", "video_format", "aspect_ratio", "loudness", "clearance"):
        english, hebrew = campaigns_assets.UNKNOWN_PATHS[kind]
        assert english and hebrew, kind
        assert set(hebrew) & HEBREW, kind
        assert not set(english) & HEBREW, kind
    blank = campaigns_assets.unknown_property("media", "")
    assert blank == {"state": "unknown", "value": None,
                     "path_en": campaigns_assets.UNKNOWN_PATHS["media"][0],
                     "path_he": campaigns_assets.UNKNOWN_PATHS["media"][1]}
    real = campaigns_assets.unknown_property("media", "https://example.test/spot.mp4")
    assert real["state"] == "real"
    assert real["value"].endswith(".mp4")


def test_the_flight_still_carries_no_delivered_figure_of_its_own(client):
    """Delivery is a derived ledger beside the booking, never a column inside it."""
    from kairos_api import campaigns_api_store as store

    booked = _book(client).json()
    added = client.post(f"/api/clients/campaigns/{booked['campaign_id']}/flights", json={
        "starts_on": "2026-09-06", "ends_on": "2026-09-12", "goal_kind": "grp", "goal_value": 150,
    })
    assert added.status_code == 201
    assert "delivered" not in " ".join(store.COLUMNS)
    assert "delivered" not in added.json()
    assert added.json()["is_demo"] is False
