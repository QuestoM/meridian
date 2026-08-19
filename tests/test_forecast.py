"""The rating forecast: a measured band, an exact decomposition, honest refusals.

Every assertion here runs against the SHIPPED artifact and the REAL observation
history, not a fixture. The properties are the ones a forecast can be wrong about
in a way that costs money: a band that does not widen when the evidence thins, a
driver list that does not multiply back to the number it claims to explain, a
fallback that happens silently, and a currency substitution.

The routes and the obligations wiring are pinned in ``tests/test_forecast_api.py``
and the out-of-sample accuracy in ``tests/test_forecast_backtest.py``.
"""

import math

import pandas as pd
import pytest

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

    # In log space the identity is exact to the rounding on BOTH sides, and the
    # binding one is not the terms'. Each log term is rounded to 6 decimals, so
    # they contribute at most 5e-7 each; `expected_tvr` is rounded to 4 decimals
    # for display, and taking a logarithm of that turns half a display step into
    # 0.00005/expected — around 2e-5 to 3e-5 at a typical rating, an order of
    # magnitude larger.
    #
    # This was a flat 1e-5, smaller than the display rounding it compares
    # against. It passed while the driver count and the rating happened to land
    # favourably and failed the first time either changed. Derived from the
    # payload so it stays right at any rating and for any number of families.
    tolerance = 0.00005 / expected + 5e-7 * len(payload["drivers"])
    assert log_sum == pytest.approx(math.log(expected), abs=tolerance)
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


def test_the_forecast_is_the_shipped_prediction_and_not_a_second_one(service, frame, owned):
    """The claim the whole surface rests on: this explains ``predict_tvr``'s
    number, it does not compute a prettier one.

    Scored over 300 real rows of the owned channel against
    :meth:`AudienceModel.predict_tvr` itself, so any future drift in the driver
    decomposition breaks here rather than shipping a payload whose explanation
    and whose figure disagree."""
    from kairos.model.audience_frame import PREDICTION_COLUMNS

    sample = frame[frame["channel"] == owned].head(300)
    rows = pd.DataFrame({
        "date": sample["date"].astype(str).str.slice(0, 10),
        "channel": sample["channel"].astype(str),
        "program_title": sample["title"].astype(str),
        "start_seconds": sample["start_seconds"].astype(float),
        "duration_seconds": 0.0,
    }, columns=list(PREDICTION_COLUMNS)).reset_index(drop=True)

    shipped = service.model.predict_tvr(rows)
    payloads = service.forecast_rows(rows)
    assert len(payloads) == len(rows)
    assert all(p["available"] for p in payloads)
    for position, payload in enumerate(payloads):
        assert payload["expected_tvr"] == round(
            float(shipped["predicted_tvr"].iloc[position]), 4
        ), position
        # And the per-row basis marker agrees, so "the model spoke here" means
        # the same thing on both surfaces.
        assert (payload["resolution"]["basis"] == "model") == (
            shipped["basis"].iloc[position] == "model"
        ), position


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
