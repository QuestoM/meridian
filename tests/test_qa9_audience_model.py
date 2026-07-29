"""Audience (expected TVR) model QA: frame construction, EB pooling, family
gates, the all-gates-off historical identity, the frozen artifact shape, and
the real-data verdicts.

The synthetic fixtures are deterministic (seeded) and use the bundled Israeli
calendar table, so the planted contrasts (summer viewing genuinely lower,
Hanukkah evenings genuinely lifted) align with the real ``cal_*`` features.
The operator events store is pointed at a missing path in every synthetic
test so real stored events can never leak into the fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kairos.data.title_features import canonicalize_series
from kairos.model.audience_factors import (
    FAMILIES,
    cell_key,
    family_cells,
    fit_cell_deltas,
)
from kairos.model.audience_frame import (
    attach_pressure,
    build_training_frame,
    prediction_frame,
    slot_band_of_hour,
)
from kairos.model.audience_model import (
    ARTIFACT_PATH,
    AudienceBase,
    AudienceModel,
    audience_model_activation,
    audience_model_status,
    fit_audience_model,
    load_audience_model,
    predict_tvr,
)

ROOT = Path(__file__).resolve().parents[1]
STAMP = "2026-07-29T00:00:00+00:00"
CHANNEL_A = "ערוץ א"
TITLES_A = ("אלף בית", "גימל דלת", "הא וו", "זין חת", "טת יוד", "כף למד")
HOURS_A = (17, 18, 19, 20, 21, 22)

SPOTS_ON_DISK = (
    (ROOT / "data" / "reference" / "Spots.xlsx").exists()
    or (ROOT / "data" / "Spots.csv").exists()
)


def _spots(records: list[tuple[str, str, float, pd.Timestamp]]) -> pd.DataFrame:
    channels, titles, tvrs, stamps = zip(*records)
    return pd.DataFrame(
        {
            "Channel": list(channels),
            "Title": list(titles),
            "TVR": list(tvrs),
            "air_dt": list(stamps),
        }
    )


def _missing_events(tmp_path: Path) -> Path:
    return tmp_path / "no_events.csv"


def _contrast_free_frame(tmp_path: Path) -> pd.DataFrame:
    """Sun-Thu November 2024, one channel, six level titles, tiny noise.

    No Hanukkah, one season, no shabbat or yom tov, no school holidays, no
    stored events: every calendar family is honestly contrast-free, and the
    titles share one level so series and weekday carry only noise.
    """
    rng = np.random.default_rng(1)
    records = []
    for day in pd.date_range("2024-11-03", "2024-11-28"):
        if day.weekday() not in (6, 0, 1, 2, 3):
            continue
        for hour, title in zip(HOURS_A, TITLES_A):
            tvr = 5.0 * float(np.exp(rng.normal(0.0, 0.05)))
            records.append((CHANNEL_A, title, tvr, day + pd.Timedelta(hours=hour)))
    return build_training_frame(_spots(records), events_path=_missing_events(tmp_path))


def _contrast_free_model(tmp_path: Path) -> AudienceModel:
    return fit_audience_model(
        frame=_contrast_free_frame(tmp_path),
        owned_channel=CHANNEL_A,
        lineup_frame_fn=lambda dates, owned: None,
        computed_at=STAMP,
    )


HANUKKAH_WINDOWS = (
    ("2024-12-26", "2025-01-02"),
    ("2025-12-15", "2025-12-22"),
)


def _planted_frame(tmp_path: Path) -> pd.DataFrame:
    """Two fabricated years where summer TVR genuinely drops and Hanukkah
    evenings genuinely lift, on top of real per-title levels."""
    rng = np.random.default_rng(11)
    hanukkah_days = set()
    for start, end in HANUKKAH_WINDOWS:
        hanukkah_days.update(pd.date_range(start, end))
    records = []
    titles = (("כוכב הצפון", 9.0, 20), ("דרך המלך", 10.0, 21), ("שער הזהב", 11.0, 22))
    for day in pd.date_range("2024-01-01", "2025-12-31"):
        season_mult = 0.75 if day.month in (6, 7, 8, 9) else 1.0
        hanukkah_mult = 1.5 if day in hanukkah_days else 1.0
        for title, level, hour in titles:
            tvr = (
                level * season_mult * hanukkah_mult
                * float(np.exp(rng.normal(0.0, 0.05)))
            )
            records.append(("ערוץ בדיקה", title, tvr, day + pd.Timedelta(hours=hour)))
    return build_training_frame(_spots(records), events_path=_missing_events(tmp_path))


def test_training_frame_construction_multi_channel(tmp_path):
    day = pd.Timestamp("2024-11-05")
    records = [
        ("ערוץ א", "אלף בית", 4.0, day + pd.Timedelta(hours=20, minutes=5)),
        ("ערוץ א", "אלף בית", 6.0, day + pd.Timedelta(hours=20, minutes=40)),
        ("ערוץ א", "אלף בית", 7.0, day + pd.Timedelta(hours=21, minutes=10)),
        ("ערוץ ב", "גימל דלת", 2.0, day + pd.Timedelta(hours=9, minutes=15)),
    ]
    frame = build_training_frame(_spots(records), events_path=_missing_events(tmp_path))

    assert len(frame) == 3
    first = frame[(frame["channel"] == "ערוץ א") & (frame["slot_hour"] == 20)].iloc[0]
    assert first["tvr"] == pytest.approx(5.0)
    assert first["n_spots"] == 2
    assert first["start_seconds"] == pytest.approx(20 * 3600 + 5 * 60)
    assert first["slot_band"] == "prime"
    assert first["series_key"] == canonicalize_series("אלף בית")
    assert isinstance(first["genre"], str) and first["genre"]
    other = frame[frame["channel"] == "ערוץ ב"].iloc[0]
    assert other["slot_band"] == slot_band_of_hour(9) == "morning"
    for column in ("cal_weekday_iso", "cal_season", "cal_is_hanukkah", "event_active"):
        assert column in frame.columns
    assert (frame["event_active"] == 0).all()


def test_thin_series_pulled_toward_genre():
    frame = pd.DataFrame(
        {
            "channel": [CHANNEL_A] * 62,
            "genre": ["Other"] * 62,
            "slot_band": ["prime"] * 62,
            "series_key": ["עשיר"] * 60 + ["דק"] * 2,
            "tvr": [8.0] * 60 + [16.0] * 2,
        }
    )
    base = AudienceBase.fit(frame)
    log_tvr = np.log(np.maximum(frame["tvr"].to_numpy(float), base.tvr_floor))
    residuals = log_tvr - base.log_base(frame)
    table = fit_cell_deltas(residuals, family_cells(frame, "series"), base.shrinkage_k)

    thin_rows = frame[frame["series_key"] == "דק"]
    base_value = float(np.exp(base.log_base(thin_rows)[0]))
    predicted_thin = base_value * float(np.exp(table[cell_key(CHANNEL_A, "דק")]))
    # Two observations against a pseudo-count of twenty: the thin series must
    # end far closer to its genre level than to its own raw mean of 16.
    assert abs(predicted_thin - base_value) < 0.2 * abs(16.0 - base_value)


def test_contrast_free_window_records_every_family_off(tmp_path):
    model = _contrast_free_model(tmp_path)
    assert set(model.gates) == set(FAMILIES)
    for family in FAMILIES:
        gate = model.gates[family]
        assert gate["verdict"] == "off"
        assert gate["reason"]
        assert gate["measured_at"] == STAMP
    assert "Hanukkah" in model.gates["calendar_hanukkah"]["reason"]
    assert "school-holiday" in model.gates["calendar_school_and_chol_hamoed"]["reason"]
    assert "religious-blackout" in model.gates["calendar_religious_blackout"]["reason"]
    assert "operator-event" in model.gates["operator_events"]["reason"]
    assert "season" in model.gates["season"]["reason"]
    # These two had a real contrast and were measured, honestly below the bar.
    assert model.gates["series"]["held_out_delta_pct"] is not None
    assert model.gates["weekday_slot"]["held_out_delta_pct"] is not None
    # Contrast-free families record no number at all, never a fabricated one.
    assert model.gates["calendar_hanukkah"]["held_out_delta_pct"] is None
    assert model.gates["season"]["held_out_delta_pct"] is None
    assert model.factors == {}


def test_all_gates_off_prediction_is_the_historical_mean_path(tmp_path):
    frame = _contrast_free_frame(tmp_path)
    model = fit_audience_model(
        frame=frame,
        owned_channel=CHANNEL_A,
        lineup_frame_fn=lambda dates, owned: None,
        computed_at=STAMP,
    )
    assert all(gate["verdict"] == "off" for gate in model.gates.values())

    rows = pd.DataFrame(
        {
            "date": ["2026-08-05", "2026-08-05", "2026-08-05"],
            "channel": [CHANNEL_A, CHANNEL_A, "ערוץ לא מוכר"],
            "program_title": [TITLES_A[0], "תוכנית חדשה לגמרי", TITLES_A[0]],
            "start_seconds": [72000.0, 72000.0, 72000.0],
            "duration_seconds": [3600.0, 3600.0, 3600.0],
        }
    )
    scored = model.predict_tvr(rows, events_path=_missing_events(tmp_path))
    assert list(scored["basis"]) == ["base", "base", "base"]

    series_key = canonicalize_series(TITLES_A[0])
    stored_mean = model.base.hist_series[CHANNEL_A][series_key]
    # Exact identity: the stored plain historical mean, unmodified.
    assert float(scored["predicted_tvr"].iloc[0]) == stored_mean
    assert stored_mean == pytest.approx(
        float(frame.loc[frame["series_key"] == series_key, "tvr"].mean()), rel=1e-12
    )
    # An unseen title falls to the genre mean, an unseen channel to the global
    # historical mean: the same plain-mean path, never an invention.
    genre = scored["predicted_tvr"].iloc[1]
    assert float(genre) == model.base.hist_genre[CHANNEL_A][
        prediction_frame(rows.iloc[[1]]).iloc[0]["genre"]
    ]
    assert float(scored["predicted_tvr"].iloc[2]) == model.base.hist_global


def test_planted_two_year_contrasts_self_activate(tmp_path):
    frame = _planted_frame(tmp_path)
    model = fit_audience_model(
        frame=frame,
        owned_channel="ערוץ בדיקה",
        lineup_frame_fn=lambda dates, owned: None,
        computed_at=STAMP,
    )
    assert model.gates["season"]["verdict"] == "on"
    assert model.gates["season"]["held_out_delta_pct"] > 2.0
    assert model.gates["calendar_hanukkah"]["verdict"] == "on"
    assert model.gates["calendar_hanukkah"]["held_out_delta_pct"] > 2.0
    assert model.gates["series"]["verdict"] == "on"
    # Nothing was planted for these: the measurement must leave them off.
    assert model.gates["weekday_slot"]["verdict"] == "off"
    assert model.gates["calendar_religious_blackout"]["verdict"] == "off"
    assert model.gates["operator_events"]["verdict"] == "off"
    assert model.gates["competitor_lineup"]["verdict"] == "off"

    rows = pd.DataFrame(
        {
            "date": ["2025-12-18", "2025-12-01", "2025-08-05", "2025-02-05"],
            "channel": ["ערוץ בדיקה"] * 4,
            "program_title": ["דרך המלך"] * 4,
            "start_seconds": [75600.0] * 4,
            "duration_seconds": [3600.0] * 4,
        }
    )
    scored = model.predict_tvr(rows, events_path=_missing_events(tmp_path))
    assert (scored["basis"] == "model").all()
    hanukkah_evening, plain_winter, august, february = scored["predicted_tvr"]
    assert hanukkah_evening > plain_winter
    assert august < february


def test_artifact_shape_and_roundtrip(tmp_path):
    frame = _planted_frame(tmp_path)
    model = fit_audience_model(
        frame=frame,
        owned_channel="ערוץ בדיקה",
        lineup_frame_fn=lambda dates, owned: None,
        computed_at=STAMP,
        source_fingerprints={"data/Spots.csv": "0" * 64},
    )
    target = tmp_path / "audience_model.json"
    model.write_artifact(target)
    payload = json.loads(target.read_text(encoding="utf-8"))

    assert set(payload) == {
        "computed_at",
        "activation_default",
        "base",
        "gates",
        "source_fingerprints",
    }
    assert payload["computed_at"] == STAMP
    assert payload["activation_default"] is False
    assert payload["source_fingerprints"] == {"data/Spots.csv": "0" * 64}
    assert set(payload["gates"]) == set(FAMILIES)
    for gate in payload["gates"].values():
        assert set(gate) == {"verdict", "reason", "held_out_delta_pct", "measured_at"}
        assert gate["verdict"] in ("on", "off")

    reloaded = load_audience_model(target)
    rows = pd.DataFrame(
        {
            "date": ["2025-12-18", "2025-08-05"],
            "channel": ["ערוץ בדיקה"] * 2,
            "program_title": ["שער הזהב"] * 2,
            "start_seconds": [79200.0] * 2,
            "duration_seconds": [3600.0] * 2,
        }
    )
    events = _missing_events(tmp_path)
    direct = model.predict_tvr(rows, events_path=events)
    from_disk = predict_tvr(rows, path=target, events_path=events)
    assert np.allclose(direct["predicted_tvr"], from_disk["predicted_tvr"])
    assert list(direct["basis"]) == list(from_disk["basis"])


def test_activation_settings_flag_defaults_false():
    assert audience_model_activation({}) is False
    assert audience_model_activation({"audience_model_activation": True}) is True
    assert audience_model_activation({"audience_model_activation": False}) is False
    # The shipped settings file does not carry the key yet: absent reads False.
    assert audience_model_activation() is False


def test_status_is_tri_state_honest(tmp_path):
    absent = audience_model_status(tmp_path / "nowhere.json")
    assert absent == {
        "available": False,
        "computed_at": None,
        "activation": False,
        "gates": {},
        "base_summary": None,
    }
    model = _contrast_free_model(tmp_path)
    target = tmp_path / "audience_model.json"
    model.write_artifact(target)
    present = audience_model_status(target)
    assert present["available"] is True
    assert present["computed_at"] == STAMP
    assert set(present["gates"]) == set(FAMILIES)
    assert present["base_summary"]["n_observations"] == 120


def test_null_competitor_pressure_is_family_not_applicable(tmp_path):
    frame = _contrast_free_frame(tmp_path)
    first_day = frame["date"].iloc[0]

    def lineup(dates, owned):
        return pd.DataFrame(
            {
                "date": [first_day, first_day],
                "start_seconds": [0.0, 63000.0],
                "end_seconds": [63000.0, 86400.0],
                "competitor_pressure": [2.5, None],
                "competitor_titles": ["חדשות מול", ""],
            }
        )

    pressure, reason = attach_pressure(frame, CHANNEL_A, lineup_frame_fn=lineup)
    assert reason is None
    # 17:00 sits in the known window; the null window and the uncovered dates
    # are NaN, never a fabricated zero.
    assert pressure[0] == 2.5
    assert np.isnan(pressure[1]) and np.isnan(pressure[2])
    assert np.isnan(pressure[3:]).all()

    base_model = _contrast_free_model(tmp_path)
    forced = AudienceModel(
        base=base_model.base,
        gates={
            **base_model.gates,
            "competitor_lineup": {
                "verdict": "on",
                "reason": "test construction",
                "held_out_delta_pct": 5.0,
                "measured_at": STAMP,
            },
        },
        factors={"competitor_lineup": {"beta": -0.05, "reference": 2.0}},
        computed_at=STAMP,
        owned_channel=CHANNEL_A,
    )
    rows = pd.DataFrame(
        {
            "date": [first_day, first_day + pd.Timedelta(days=1)],
            "channel": [CHANNEL_A] * 2,
            "program_title": [TITLES_A[0]] * 2,
            "start_seconds": [61200.0] * 2,
            "duration_seconds": [3600.0] * 2,
        }
    )
    events = _missing_events(tmp_path)
    scored = forced.predict_tvr(rows, events_path=events, lineup_frame_fn=lineup)
    # Known pressure: the factor applies multiplicatively on the pooled base.
    covered = prediction_frame(rows.iloc[[0]], events_path=events)
    expected = float(np.exp(forced.base.log_base(covered)[0] - 0.05 * (2.5 - 2.0)))
    assert float(scored["predicted_tvr"].iloc[0]) == pytest.approx(expected, rel=1e-12)
    assert scored["basis"].iloc[0] == "model"
    # Missing forward lineup: null pressure, family not applicable, honest base.
    assert scored["basis"].iloc[1] == "base"
    series_key = canonicalize_series(TITLES_A[0])
    assert float(scored["predicted_tvr"].iloc[1]) == forced.base.hist_series[CHANNEL_A][series_key]


def test_empty_history_records_off_never_errors(tmp_path):
    model = fit_audience_model(
        frame=build_training_frame(
            pd.DataFrame(columns=["Channel", "Title", "TVR", "air_dt"]),
            events_path=_missing_events(tmp_path),
        ),
        owned_channel=CHANNEL_A,
        lineup_frame_fn=lambda dates, owned: None,
        computed_at=STAMP,
    )
    for family in FAMILIES:
        assert model.gates[family]["verdict"] == "off"
        assert model.gates[family]["reason"]


@pytest.mark.skipif(not SPOTS_ON_DISK, reason="reference spots history not on disk")
def test_real_data_family_verdicts():
    model = fit_audience_model(computed_at=STAMP)
    assert set(model.gates) == set(FAMILIES)
    for family in FAMILIES:
        gate = model.gates[family]
        assert gate["verdict"] in ("on", "off")
        assert gate["reason"]
        print(f"REAL-DATA VERDICT {family}: {gate['verdict']} :: {gate['reason']}")
    # One November: no Hanukkah days and a single season, so these two cannot
    # honestly activate; the rest are the measurement's call.
    assert model.gates["calendar_hanukkah"]["verdict"] == "off"
    assert model.gates["season"]["verdict"] == "off"


@pytest.mark.skipif(not ARTIFACT_PATH.exists(), reason="audience artifact not built")
def test_shipped_artifact_matches_contract():
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    assert set(payload) == {
        "computed_at",
        "activation_default",
        "base",
        "gates",
        "source_fingerprints",
    }
    assert payload["activation_default"] is False
    assert set(payload["gates"]) == set(FAMILIES)
