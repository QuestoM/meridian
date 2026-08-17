"""The rating forecast: a measured band, an exact decomposition, honest refusals.

Every assertion here runs against the SHIPPED artifact and the REAL observation
history, not a fixture. The properties are the ones a forecast can be wrong about
in a way that costs money: a band that does not widen when the evidence thins, a
driver list that does not multiply back to the number it claims to explain, a
fallback that happens silently, a currency substitution, and a rival channel name
on a payload.
"""

import json
import math

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from kairos.model.audience_frame import build_training_frame
from kairos.model.forecast import ForecastService
from kairos.model.forecast_basis import (
    AUDIENCE_TRADE_CURRENCY,
    AUDIENCE_UNMEASURED,
    classify_audience,
)

# The date every in-window test forecasts on: inside the measured window, so the
# payload is a fit and says so. Out-of-sample accuracy is test_forecast_backtest.
IN_WINDOW_DAY = "2024-11-20"


@pytest.fixture(scope="module")
def frame():
    return build_training_frame()


@pytest.fixture(scope="module")
def service(frame):
    return ForecastService.load(frame=frame)


@pytest.fixture(scope="module")
def owned(service):
    channel = service.model.owned_channel
    if not channel:
        pytest.skip("no owned channel configured in the shipped artifact")
    return channel


def _series_by_count(frame, channel, wanted):
    """A real (title, start_seconds, n) for the series whose count is ``wanted``."""
    own = frame[frame["channel"] == channel]
    counts = own.groupby("series_key").size().sort_values(ascending=False)
    for key, n in counts.items():
        if wanted(int(n)):
            row = own[own["series_key"] == key].iloc[0]
            return str(row["title"]), float(row["start_seconds"]), int(n)
    pytest.skip("the real history carries no series with the required observation count")


def _forecast(service, channel, title, start, **kwargs):
    return service.forecast_programme(
        channel=channel, program_title=title, day=IN_WINDOW_DAY,
        start_seconds=start, duration_seconds=1800.0, **kwargs,
    )


# --------------------------------------------------------------------- the band

def test_the_band_widens_when_the_evidence_thins(service, frame, owned):
    """A thin cell must be visibly less certain than a rich one, in real numbers.

    This is the property the whole interval exists for. The band is computed from
    the shrinkage weight the fit actually applied, so the comparison is between
    two REAL cells of the shipped model rather than two invented variances.
    """
    fat_title, fat_start, fat_n = _series_by_count(frame, owned, lambda n: n >= 40)
    thin_title, thin_start, thin_n = _series_by_count(frame, owned, lambda n: n <= 3)
    assert fat_n > thin_n

    fat = _forecast(service, owned, fat_title, fat_start)
    thin = _forecast(service, owned, thin_title, thin_start)
    for payload in (fat, thin):
        assert payload["available"], payload
        assert payload["interval"]["available"], payload["interval"]

    # The log-space spread is the scale-free statement of the inequality; the
    # ratio of the band's ends is the same statement in rating points.
    assert thin["interval"]["sd_log"] > fat["interval"]["sd_log"]
    fat_ratio = fat["interval"]["high"] / fat["interval"]["low"]
    thin_ratio = thin["interval"]["high"] / thin["interval"]["low"]
    assert thin_ratio > fat_ratio

    # And the reason is the shrinkage weight, not a coincidence: the thin cell
    # sits most of the way toward its parent, the rich cell barely at all.
    assert (thin["interval"]["components"]["weight_on_parent"]
            > fat["interval"]["components"]["weight_on_parent"])
    assert fat["interval"]["components"]["n_observations"] == fat_n
    assert thin["interval"]["components"]["n_observations"] == thin_n


def test_the_band_is_withheld_with_a_reason_when_the_scatter_is_unmeasured(owned):
    """No measurement frame means no band. It must never mean an invented band."""
    service = ForecastService.load(with_frame=False)
    payload = service.forecast_programme(
        channel=owned, program_title="כל תוכנית", day=IN_WINDOW_DAY,
        start_seconds=72_000.0, duration_seconds=1800.0,
    )
    assert payload["available"] is True
    assert payload["expected_tvr"] > 0          # the point forecast still stands
    assert payload["interval"]["available"] is False
    assert payload["interval"]["low"] is None and payload["interval"]["high"] is None
    assert "frame" in payload["interval"]["reason"]


# ------------------------------------------------------------------ the drivers

def test_the_drivers_multiply_back_to_the_forecast(service, frame, owned):
    """The decomposition is the number, not a story told beside it."""
    title, start, _n = _series_by_count(frame, owned, lambda n: n >= 40)
    payload = _forecast(service, owned, title, start)
    assert payload["resolution"]["basis"] == "model"

    product = 1.0
    log_sum = 0.0
    for driver in payload["drivers"]:
        product *= driver["value_tvr"] if driver["kind"] == "base" else driver["multiplier"]
        log_sum += driver["log_term"]
    expected = payload["expected_tvr"]

    # In log space the identity is exact to the terms' own 6-decimal rounding.
    assert log_sum == pytest.approx(math.log(expected), abs=1e-5)
    # In points it is exact to the 4-decimal rounding the display carries.
    assert product == pytest.approx(expected, rel=1e-3)

    # An activated family that applied appears as a multiplier; an off family
    # appears in not_applied with the verdict its own measurement returned.
    applied = {d.get("family") for d in payload["drivers"] if d.get("family")}
    assert "weekday_slot" in applied
    off = {entry["family"]: entry for entry in payload["not_applied"]}
    assert "season" in off
    assert off["season"]["verdict"] == "off"
    assert off["season"]["reason"], "an off family must carry the measured reason"


def test_an_off_family_is_reported_as_uncontrasted_not_as_failed(service, frame, owned):
    """Five of eight families are off for ABSENCE OF CONTRAST in a one-month
    window. That is a different fact from having been tried and beaten, and the
    payload must carry the distinction the gate wrote rather than flatten it."""
    title, start, _n = _series_by_count(frame, owned, lambda n: n >= 40)
    off = {e["family"]: e for e in _forecast(service, owned, title, start)["not_applied"]}
    hanukkah = off["calendar_hanukkah"]
    assert hanukkah["verdict"] == "off"
    assert hanukkah["held_out_delta_pct"] is None, (
        "a family with no contrast was never scored, so it has no held-out delta"
    )
    assert "contrast" in hanukkah["reason"]
    # A family that WAS scored and lost carries its measured number instead.
    blackout = off["calendar_religious_blackout"]
    assert blackout["held_out_delta_pct"] is not None


# ----------------------------------------------------------------- the fallback

def test_an_unknown_series_falls_back_and_names_the_level(service, owned):
    """A programme the model never saw is answered by the level that DID answer,
    and the payload says which one and what contributed exactly 1.0."""
    payload = _forecast(
        service, owned, "תוכנית שלא הייתה מעולם 9999", 75_600.0,
    )
    assert payload["available"] is True
    resolution = payload["resolution"]
    assert resolution["level"] != "series"
    assert resolution["level"] in ("genre", "slot", "channel", "global")
    assert resolution["level_he"]
    assert payload["provenance"]["level_that_answered"] == resolution["level"]
    assert any("never observed in training" in note for note in resolution["fallbacks"])
    series_driver = next(
        (d for d in payload["drivers"] if d.get("family") == "series"), None
    )
    if series_driver is not None:
        assert series_driver["multiplier"] == pytest.approx(1.0, abs=5e-5)


# ----------------------------------------------------------------- the currency

def test_the_trade_settlement_currency_is_refused_with_its_own_reason(service, owned):
    """The trade settles on Jewish households, quarter-hour, overnight +1; the
    model's base is the all-viewers planned break rating. ``docs/trade/domain.md``
    section 3: the model must never conflate the two bases. So the settlement
    currency gets its OWN refusal, distinct from an unmeasured demographic,
    because that substitution is the one the domain document names."""
    assert classify_audience("בתי אב יהודיים") == AUDIENCE_TRADE_CURRENCY
    assert classify_audience("רייטינג רבעי שעה") == AUDIENCE_TRADE_CURRENCY
    assert classify_audience("גברים 25-54") == AUDIENCE_UNMEASURED

    trade = _forecast(service, owned, "כל תוכנית", 72_000.0, audience="בתי אב יהודיים")
    assert trade["available"] is False
    assert trade["audience_state"] == AUDIENCE_TRADE_CURRENCY
    assert "שני מטבעות שונים" in trade["reason_he"]
    assert trade["audience_basis"]["serves_trade_currency"] is False

    demographic = _forecast(service, owned, "כל תוכנית", 72_000.0, audience="גברים 25-54")
    assert demographic["available"] is False
    assert demographic["audience_state"] == AUDIENCE_UNMEASURED
    assert demographic["reason_he"] != trade["reason_he"], (
        "two different mistakes must not share one reason"
    )


def test_a_date_past_the_measured_window_is_refused_with_the_distance(service, owned):
    """The fit carries no trend and no season term, so a date far past the last
    observation is refused rather than extrapolated.

    The window the real artifact was measured on is November 2024, so this
    refusal is LIVE for present-day dates: the shipped observation history is one
    month old in calendar terms, and the forecast says so instead of carrying a
    year-old level forward under a confidence band."""
    far = service.forecast_programme(
        channel=owned, program_title="כל תוכנית", day="2027-06-15",
        start_seconds=72_000.0, duration_seconds=1800.0,
    )
    assert far["available"] is False
    assert "מגמה" in far["reason_he"]
    assert far["horizon"]["days_after_window"] > 365


def test_a_date_outside_the_bundled_calendar_is_refused_separately(service, owned):
    """Outside the calendar table the holiday features read false rather than
    measured, so the forecast would be scored on fabricated context. That is a
    different refusal from the horizon one and carries its own reason."""
    payload = service.forecast_programme(
        channel=owned, program_title="כל תוכנית", day="2030-01-15",
        start_seconds=72_000.0, duration_seconds=1800.0,
    )
    assert payload["available"] is False
    assert "לוח השנה" in payload["reason_he"]
    assert "calendar" in payload["reason_en"]


def test_an_in_window_date_says_it_is_a_fit_and_points_at_the_backtest(service, frame, owned):
    title, start, _n = _series_by_count(frame, owned, lambda n: n >= 40)
    horizon = _forecast(service, owned, title, start)["horizon"]
    assert horizon["inside_measured_window"] is True
    assert "בקטסט" in horizon["note_he"]


def test_history_rides_beside_the_forecast_on_every_payload(service, frame, owned):
    """The plain historical mean is what this product priced on before the model.
    Shipping the forecast without it would be a number with nothing to be
    compared against; the backtest measures which one is actually closer."""
    title, start, _n = _series_by_count(frame, owned, lambda n: n >= 40)
    payload = _forecast(service, owned, title, start)
    assert payload["history"]["historical_tvr"] > 0
    assert payload["history"]["level"] in ("series", "genre", "slot", "channel", "global")


# --------------------------------------------------------------- the API surface

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
