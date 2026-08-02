"""P5: the rate-card delta, and the events read that used to lie before the click.

The delta's only real claim is arithmetic: break revenue is
``cpp * rating_points * units * premium``, so it is strictly linear in the
effective rate and a committed plan row re-prices by one multiplication. That
claim is checkable rather than believable, and the check is that re-pricing the
plan under the card as saved reproduces the plan's own revenue and its own yield
per second exactly. If it ever stops doing that, the delta is measuring
something other than this plan and the payload says so.

The second half is the fourth open read section 4.5 puts on this piece: the
calendar's ``model_context`` block on a run surface.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.core as core

ROOT = Path(__file__).resolve().parents[1]

# The configuration the frozen figures in this file were measured under, which
# is the one committed in `data/kairos_settings.json`. Pinned here rather than
# read from the deployed document, because that document is writable by any
# client of `PUT /api/settings` and a file anything can write is not a fixture.
# Measured on 2026-08-01: a settings write emptied both of these in the shared
# tree, which moved the worth of a second from 142.7044 to 142.6719 and put four
# channels inside a figure the operator reads as their own.
OPERATOR_CHANNEL = "רשת 13"
BASELINE_PRICING = {"pricing_activation": {"show": False, "events": True}}


def _settings_at(tmp_path, **overrides):
    """A private copy of the settings document, with the fields this file pins."""
    copy = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", copy)
    document = json.loads(copy.read_text(encoding="utf-8"))
    document.update(overrides)
    copy.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return copy


@pytest.fixture(autouse=True)
def declared_channel(tmp_path, monkeypatch):
    monkeypatch.setattr(core, "SETTINGS_PATH", _settings_at(
        tmp_path, operator_channel=OPERATOR_CHANNEL, pricing_overrides=BASELINE_PRICING,
    ))
    return tmp_path

TRAINING_LEXICON = (
    "gate", "held_out", "tau", "drift", "coefficient", "pooling",
    "p_value", "training_window", "wartime",
)


@pytest.fixture()
def client() -> TestClient:
    import kairos_api.pricing_api_effect as effect_api

    app = FastAPI()
    app.include_router(effect_api.router)
    return TestClient(app)


def _effect(client: TestClient, overrides: dict) -> dict:
    response = client.post("/api/pricing/effect", json={"overrides": overrides})
    assert response.status_code == 200, response.text
    return response.json()


def test_an_empty_edit_reproduces_the_plan_on_record_to_the_last_digit(client):
    body = _effect(client, {})
    assert body["available"] is True
    assert body["reproduces_plan"] is True, (
        "the saved side has to be the plan's own money, or the delta is against a different plan"
    )
    assert body["saved"]["revenue"] == body["plan_revenue_on_record"]
    assert body["saved"]["yield_per_second"] == body["draft"]["yield_per_second"]
    assert body["delta"]["revenue"] == 0.0
    assert body["saved"]["rows_unpriced"] == 0


# The operator channel's revenue in the plan the Bar 3 figure was measured on.
# The yield is that revenue over that plan's ad seconds, so the figure is only
# checkable against the plan it was taken from.
BASELINE_PLAN_REVENUE = 40944759.33


def test_the_saved_side_is_the_same_yield_the_yield_endpoint_serves(client):
    """One worth of a second in the product, not a second opinion beside it."""
    from kairos_api.yield_api import router as yield_router

    app = FastAPI()
    app.include_router(yield_router)
    yield_client = TestClient(app)
    yields = yield_client.get("/api/yield-per-second").json()
    saved = _effect(client, {})
    assert saved["saved"]["yield_per_second"] == yields["totals"]["yield_per_second"], (
        "the rate card and the yield endpoint have to be one figure, not two"
    )
    assert saved["reproduces_plan"] is True

    on_disk = saved["plan_revenue_on_record"]
    if on_disk != BASELINE_PLAN_REVENUE:
        pytest.skip(
            "the Bar 3 figure 142.7044 is a function of the saved plan, and the plan on disk "
            f"now totals {on_disk} for {OPERATOR_CHANNEL} where the baseline totals "
            f"{BASELINE_PLAN_REVENUE}. The agreement above still holds; the literal is "
            "unverifiable until the plan it was measured on is back"
        )
    assert yields["totals"]["yield_per_second"] == 142.7044


@pytest.mark.parametrize("factor", [1.1, 0.5, 2.0])
def test_a_base_price_change_moves_the_plan_exactly_in_proportion(client, factor):
    base = _effect(client, {})
    current = 60.0
    body = _effect(client, {"base_price_per_second_per_tvr_point": current * factor})
    assert body["draft"]["revenue"] == pytest.approx(base["saved"]["revenue"] * factor, rel=1e-9)
    assert body["draft"]["yield_per_second"] == pytest.approx(
        base["saved"]["yield_per_second"] * factor, rel=1e-6,
    )
    assert body["delta"]["percent"] == pytest.approx((factor - 1) * 100, abs=1e-6)


def test_the_edit_is_priced_and_never_saved(client):
    from kairos_api.core import _load_settings

    before = dict(getattr(_load_settings(), "pricing_overrides", None) or {})
    _effect(client, {"base_price_per_second_per_tvr_point": 999})
    after = dict(getattr(_load_settings(), "pricing_overrides", None) or {})
    assert after == before, "a preview must not write the rate card"


def test_a_spot_level_layer_is_named_rather_than_reported_as_no_change(client):
    body = _effect(client, {"pricing_activation": {"position": True}})
    changed = {row["layer"]: row for row in body["changed_layers"]}
    assert "position" in changed
    assert changed["position"]["moves_plan"] is False
    assert "position" in body["spot_level_layers"]
    assert body["delta"]["revenue"] == 0.0


def test_an_unusable_edit_is_refused_with_the_reason_and_no_figure(client):
    body = _effect(client, {"base_price_per_second_per_tvr_point": -5})
    assert body["available"] is False
    assert body["reason"]
    assert "saved" not in body and "delta" not in body


def test_every_money_figure_carries_the_scope_it_was_summed_over(client):
    body = _effect(client, {"base_price_per_second_per_tvr_point": 66})
    scope = body["scope"]
    assert scope["channel"] == OPERATOR_CHANNEL
    assert scope["scoped"] is True and scope["channels_priced"] == 1
    assert scope["date_from"] and scope["date_to"]
    assert scope["rows"] > 0 and scope["days"] > 0
    assert body["basis"]["formula"] and body["basis"]["source"]


def test_with_no_declared_channel_the_figure_says_it_is_the_market_and_not_yours(client, tmp_path, monkeypatch):
    """Measured live: an unvalidated settings write emptied the channel, and the
    rate card then served 221,873,624 across four channels under the operator's
    own heading. The figure is real; what was missing is that it is not theirs."""
    monkeypatch.setattr(core, "SETTINGS_PATH", _settings_at(tmp_path, operator_channel=""))
    body = _effect(client, {"base_price_per_second_per_tvr_point": 66})
    scope = body["scope"]
    assert scope["scoped"] is False, "an undeclared channel has to be reported, not implied"
    assert scope["channels_priced"] > 1
    assert scope["unscoped_reason"], "the reason travels with the payload"
    assert scope["channel"] == ""
    assert scope["rows"] > 0


# ---------------------------------------------------------------------------
# The pricing read, and the events read, both on run surfaces.


def test_the_pricing_read_says_whether_the_events_switch_may_be_thrown():
    from kairos_api.pricing_api import router as pricing_router

    app = FastAPI()
    app.include_router(pricing_router)
    body = TestClient(app).get("/api/pricing").json()
    assert "can_edit_events" in body, (
        "the events activation is company-only on the write, so the read has to say so"
    )
    assert len(body["layers"]) == 6
    assert [layer["name"] for layer in body["layers"]] == [
        "base", "program", "day", "show", "position", "ad_type",
    ]
    for layer in body["layers"][1:]:
        assert "live_today" in layer and "activatable" in layer


# The three lexicon words that are the calendar's disclosure rather than a
# verdict: the window the per-event overlap count is measured against, and the
# wartime sentence, whose own text says "passes the held-out gate". The other six
# are verdict vocabulary and a run surface carries none of them.
DISCLOSURE_LEXICON = ("gate", "training_window", "wartime")
VERDICT_LEXICON = tuple(word for word in TRAINING_LEXICON if word not in DISCLOSURE_LEXICON)


def test_the_calendar_wall_takes_the_verdicts_and_leaves_the_disclosure(monkeypatch):
    """Section 4.2's lexicon check, applied where it bites instead of literally.

    Revision 1 of this piece read test 2 as "withhold ``model_context`` from a
    channel account entirely" and shipped that. Measured on the shipped surface,
    it took no training content off the run screen. Every event still carries
    ``window_overlap_days``; ``CalendarEventsModel.jsx`` still renders it as
    "days inside the training window" and "the training data did not see this
    condition" under a panel titled "Event overlaps: training window and current
    plan"; and ``ModelContextPanel`` fell through to its empty state, which reads
    "the backend did not report the model context", a sentence that was no longer
    true because the backend was withholding it on purpose. So the channel reader
    kept the training-window figure and lost the only sentence on the product
    that says no event retention effect is measured. That also broke
    ``tests/test_qa8_permissions.py``, the shipped permissions contract, which
    pins ``model_context`` in a channel account's payload.

    The wall the rest of this wave built is surgical, and this route is now the
    same. ``/api/parameters`` serves ``model_version`` to every account and walls
    only the verdicts (``scenario_api_parameters.py``), and the assistant's event
    pipeline "keeps every run-side stage for a channel account", replacing only
    the measured verdict (``assistant_model_disclosure.py``, pinned by
    ``tests/test_p9_kai_boundary.py``). Both halves below are measured.
    """
    import kairos_api.events_api as events_api

    app = FastAPI()
    app.include_router(events_api.router)
    client = TestClient(app)

    monkeypatch.setattr(events_api, "requester_is_company", lambda request: False)
    channel_body = client.get("/api/events").json()
    assert channel_body["can_edit"] is False
    assert channel_body["training_visible"] is False, "the copy served is named, not inferred"
    context = channel_body["model_context"]
    assert set(context) == {
        "training_window", "weekday_premiums", "measurement", "wartime_disclosure",
    }, "the disclosure stays, the gate verdict goes"
    assert set(context["measurement"]) == {"available", "computed_at"}, (
        "a run surface gets the model's date and no held-out or drift verdict"
    )
    assert context["wartime_disclosure"]["ceasefire_date"] == "2024-11-27"
    text = json.dumps(channel_body, ensure_ascii=False).lower()
    hits = [word for word in VERDICT_LEXICON if word in text]
    assert hits == [], f"a run surface returned verdict vocabulary: {hits}"

    monkeypatch.setattr(events_api, "requester_is_company", lambda request: True)
    company_body = client.get("/api/events").json()
    assert company_body["can_edit"] is True and company_body["training_visible"] is True
    assert set(company_body["model_context"]) == {
        "training_window", "weekday_premiums", "measurement",
        "wartime_disclosure", "training_gate",
    }, "company staff keep exactly what they had"
    assert set(company_body["model_context"]["measurement"]) == {
        "available", "computed_at", "detrend_baseline_mode", "seasonal_baseline",
        "level_drift",
    }
