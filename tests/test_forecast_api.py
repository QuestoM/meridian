"""The forecast where it changes a decision: the routes, and the TRP projection.

Split from ``tests/test_forecast.py``, which pins the model layer. These are the
WIRING properties: that the planning surface answers for the operator's own
channel and no other, that no payload on any of the four routes carries a rival
channel name, and that the forecast-based line on a delivery guarantee appears
only when a forecast was actually available and never displaces the committed
projection beside it.
"""

import json
from datetime import date

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from kairos.model.audience_frame import build_training_frame
from kairos.model.forecast import ForecastService

# Inside the measured window, so every route answers rather than refusing.
IN_WINDOW_DAY = "2024-11-20"


@pytest.fixture(scope="module")
def frame():
    return build_training_frame()


@pytest.fixture(scope="module")
def owned(frame):
    channel = ForecastService.load(frame=frame).model.owned_channel
    if not channel:
        pytest.skip("no owned channel configured in the shipped artifact")
    return channel


@pytest.fixture(scope="module")
def client():
    from kairos_api.server import app

    return TestClient(app)


def _rivals(frame, owned):
    names = sorted({str(c) for c in frame["channel"].unique()} - {owned})
    if not names:
        pytest.skip("the real history carries no rival channel to leak")
    return names


def test_no_forecast_payload_names_a_rival_channel(client, frame, owned):
    """THE CHANNEL WALL. The model trains on every channel in the file. Their
    names live in the base's per-channel maps and must never reach a payload."""
    rivals = _rivals(frame, owned)
    responses = {
        "programme": client.get("/api/forecast/programme", params={
            "title": "חדשות 13", "date": IN_WINDOW_DAY, "start": "20:00"}),
        "drivers": client.get("/api/forecast/drivers"),
        "accuracy": client.get("/api/forecast/accuracy"),
        "schedule": client.get("/api/forecast/schedule", params={"date": IN_WINDOW_DAY}),
    }
    for name, response in responses.items():
        assert response.status_code == 200, (name, response.text[:300])
        blob = json.dumps(response.json(), ensure_ascii=False)
        for rival in rivals:
            assert rival not in blob, f"{name} leaked the rival channel {rival!r}"

    # The base block ships scalars only, which is what holds that wall.
    base = responses["drivers"].json()["base"]
    assert base and all(
        not isinstance(value, (dict, list)) for value in base.values()
    ), base


def test_a_forecast_for_another_channel_is_refused_without_echoing_it(client, frame, owned):
    rival = _rivals(frame, owned)[0]
    response = client.get("/api/forecast/programme", params={
        "title": "כל תוכנית", "date": IN_WINDOW_DAY, "channel": rival})
    assert response.status_code == 422
    assert rival not in response.text, "a refusal must not echo the rival's name back"


def test_the_schedule_surface_forecasts_the_plan_s_own_programmes(client, owned):
    """The planning surface: a channel-day of real programmes, each with its
    forecast, its band, and the historical mean beside it."""
    response = client.get("/api/forecast/schedule", params={"date": IN_WINDOW_DAY})
    assert response.status_code == 200
    body = response.json()
    assert body["available"] is True
    assert body["channel"] == owned
    day = body["days"][0]
    assert day["summary"]["n_forecast"] > 0
    assert day["summary"]["n_with_band"] == day["summary"]["n_forecast"]
    assert day["summary"]["mean_expected_tvr"] is not None
    assert day["summary"]["mean_historical_tvr"] is not None
    programme = day["programmes"][0]
    assert programme["start_clock"][2] == ":"
    assert programme["interval"]["available"] is True


# ------------------------------------------------- the obligations forward line

def _delivery(rows):
    return pd.DataFrame(rows, columns=[
        "campaign_id", "broadcast_date", "air_state", "channel", "spots",
        "seconds", "rating_points_planned", "spend_ils", "counted_as_of",
    ])


def _trp_obligation():
    return {
        "instance_id": "i-t", "term_id": "trp-delivery-guarantee",
        "params": {"points": 300, "audience": "כלל הצופים", "window": "year"},
        "scope": {}, "window": {},
    }


def _head():
    return {"agreement_id": "agr-f", "counterparty": {"advertiser": "טכנו-קור"},
            "window": {"starts_on": "2026-01-01", "ends_on": "2026-12-31"}}


def _evaluate(forecast_points):
    from datetime import date

    from kairos.trade import obligations

    delivery = _delivery([
        ("C1", "2026-02-01", "aired", "רשת 13", 10, 300, 120.0, 100_000, "t"),
        ("C1", "2026-08-01", "scheduled", "רשת 13", 10, 300, 40.0, 40_000, "t"),
    ])
    inputs = obligations.Inputs(
        delivery=delivery,
        campaigns=pd.DataFrame([("C1", "טכנו-קור")], columns=["campaign_id", "advertiser"]),
        agency_links=pd.DataFrame(columns=["agency_id", "agency_name", "advertiser"]),
        today=date(2026, 6, 30),
        forecast_points=forecast_points,
    )
    (snapshot,) = obligations.evaluate_all(
        {"version_id": "v-f", "agreement_id": "agr-f", "instances": [_trp_obligation()]},
        _head(), inputs,
    )
    return snapshot


def test_the_forecast_forward_line_appears_only_when_a_forecast_is_available():
    """The third projection line rides BESIDE booked-forward and pace-forward and
    replaces neither. With no provider, or a provider that cannot answer, the
    payload is exactly what it was before the forecast stage existed."""
    from kairos.trade import obligations_forecast

    bare = _evaluate(None)
    assert set(bare["projection_method"]) == {"booked_forward", "pace_forward", "note"}
    assert bare["projection"] == bare["projection_method"]["booked_forward"]

    # A provider with no schedule for those days cannot form a ratio, so no line.
    def no_schedule(*, counted, scheduled):
        return obligations_forecast.forward_line(
            counted, scheduled, schedule_fn=None,
            forecast_rows_fn=lambda rows: [],
        )

    refused = _evaluate(no_schedule)
    assert "forecast_forward" not in refused["projection_method"]
    assert refused["projection"] == bare["projection"], (
        "a refused forecast must not move the committed projection"
    )

    # A provider that CAN answer adds the line, and only the line.
    def schedule_fn(channel, day):
        return pd.DataFrame({"program_title": ["תוכנית א", "תוכנית ב"],
                             "start_seconds": [72_000.0, 75_600.0]})

    def forecast_rows_fn(rows):
        # Expected 20 percent above the historical mean the traffic log priced on.
        return [{"available": True, "expected_tvr": 6.0,
                 "history": {"historical_tvr": 5.0}} for _ in range(len(rows))]

    def provider(*, counted, scheduled):
        return obligations_forecast.forward_line(
            counted, scheduled, schedule_fn=schedule_fn,
            forecast_rows_fn=forecast_rows_fn,
        )

    served = _evaluate(provider)
    line = served["projection_method"]["forecast_forward"]
    assert line["available"] is True
    assert served["projection_method"]["booked_forward"] == bare["projection_method"]["booked_forward"]
    assert served["projection_method"]["pace_forward"] == bare["projection_method"]["pace_forward"]
    assert served["projection"] == bare["projection"], (
        "the committed projection stays booked points; a model expectation is not a booking"
    )
    # 120 counted + 40 scheduled re-rated by 6.0/5.0 = 120 + 48 = 168.
    assert line["value"] == pytest.approx(168.0)
    assert line["scheduled_points_rerated"] == pytest.approx(48.0)
    assert line["n_days_rerated"] == 1
    assert line["label_he"] and line["basis_he"]


def test_a_broken_forecast_provider_cannot_break_a_standing():
    def explode(*, counted, scheduled):
        raise RuntimeError("the forecast service is down")

    snapshot = _evaluate(explode)
    assert "forecast_forward" not in snapshot["projection_method"]
    assert snapshot["standing"]["counted"] == 120.0
