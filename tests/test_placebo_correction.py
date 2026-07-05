"""Tests for the placebo-drift correction layer.

Synthetic fixtures prove the arithmetic (per-genre subtraction, content-only
baseline math, determinism, the Law 9 defaults); the realdata tests prove the
gated rebuild end to end: flag OFF reproduces the shipped coefficients
byte-equivalently (also covered by tests/test_rebuild_equivalence.py), and
force-on reproduces the candidate artifact
models/candidates/tv_break_coefficients_placebo_corrected.json with the
review's corrected pooled value.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kairos.model.measure import (
    _baseline_levels,
    _content_only_baseline_levels,
    _dayparts_frame,
    break_effects,
)
from kairos.model.placebo_correction import (
    PlaceboCorrection,
    apply_placebo_correction,
    measure_placebo_drift,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compute_measured_coefficients.py"
SHIPPED = ROOT / "models" / "tv_break_coefficients.json"
CANDIDATE = ROOT / "models" / "candidates" / "tv_break_coefficients_placebo_corrected.json"


def _classifier():
    from kairos.data import ProgramClassifier

    return ProgramClassifier.from_yaml()


# --- per-genre subtraction arithmetic (hand-computed) ------------------------

def _correction(**overrides) -> PlaceboCorrection:
    base = dict(
        per_genre_drift={"News": 0.0131, "Other": 0.0147, "PrimeShow1": 0.0096},
        per_genre_n={"News": 1399, "Other": 4132, "PrimeShow1": 342},
        per_genre_se={"News": 0.006, "Other": 0.005, "PrimeShow1": 0.014},
        pooled_drift=0.0150,
        n_pseudo=5873,
        n_clusters=121,
        se=0.0037,
    )
    base.update(overrides)
    return PlaceboCorrection(**base)


def test_apply_subtracts_each_genre_drift_exactly() -> None:
    effects = pd.DataFrame(
        {
            "channel_name": ["News_first_short", "Other_last_long", "PrimeShow1_middle_standard"],
            "program_type": ["News", "Other", "PrimeShow1"],
            "log_effect": [-0.05, -0.03, 0.01],
        }
    )
    out = apply_placebo_correction(effects, _correction())
    # Hand-computed: raw minus the genre's drift, in log units.
    assert out["log_effect"].tolist() == pytest.approx(
        [-0.05 - 0.0131, -0.03 - 0.0147, 0.01 - 0.0096], abs=1e-15
    )
    # The input frame is not mutated and non-effect columns are untouched.
    assert effects["log_effect"].tolist() == [-0.05, -0.03, 0.01]
    assert out["channel_name"].tolist() == effects["channel_name"].tolist()


def test_apply_falls_back_to_pooled_drift_for_unknown_genre() -> None:
    effects = pd.DataFrame(
        {"program_type": ["PrimeShow2"], "log_effect": [-0.02]}
    )
    out = apply_placebo_correction(effects, _correction())
    assert out["log_effect"].tolist() == pytest.approx([-0.02 - 0.0150], abs=1e-15)


def test_apply_on_empty_frame_returns_empty() -> None:
    effects = pd.DataFrame(columns=["program_type", "log_effect"])
    assert apply_placebo_correction(effects, _correction()).empty


# --- synthetic world: one programme, one break, minute-level audience --------

def _programmes() -> pd.DataFrame:
    rows = [("חדשות הערב", "A", "2024-11-04 20:00:00", "2024-11-04 21:00:00", 3600.0)]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "end", "Duration"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    frame["end_dt"] = pd.to_datetime(frame["end"])
    return frame


def _spot(channel: str, start: str, duration: float) -> dict:
    return {"Channel": channel, "air_dt": pd.Timestamp(start), "Duration": duration}


def _break_spots() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _spot("A", "2024-11-04 20:30:00", 30),
            _spot("A", "2024-11-04 20:30:30", 30),  # two-spot break 20:30 -> 20:31
        ]
    )


def _flat_dayparts(tvr: float = 10.0) -> pd.DataFrame:
    """Four days of a perfectly flat 20:00-21:00 curve on channel A."""
    rows = [
        {"date": pd.Timestamp(date), "timeband": f"20:{m:02d}", "channel": "A", "tvr": tvr}
        for date in ("2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07")
        for m in range(0, 60)
    ]
    rows += [
        {"date": pd.Timestamp(date), "timeband": "21:00", "channel": "A", "tvr": tvr}
        for date in ("2024-11-04", "2024-11-05", "2024-11-06", "2024-11-07")
    ]
    return pd.DataFrame(rows)


def test_drift_is_exactly_zero_on_a_flat_world() -> None:
    # Every observed and baseline window mean is the same constant, so each
    # sampled pseudo-break measures log(1) - log(1) = 0: the drift, its pooled
    # mean and every per-genre mean are exactly 0.0 by hand.
    spots, programmes, dayparts = _break_spots(), _programmes(), _flat_dayparts()
    classifier = _classifier()
    effects = break_effects(spots, programmes, dayparts, classifier)
    assert len(effects) == 1
    correction = measure_placebo_drift(spots, programmes, dayparts, classifier, effects)
    assert correction.n_pseudo == 3  # one source break, k=3 eligible draws
    assert correction.pooled_drift == 0.0
    assert set(correction.per_genre_drift) == {str(effects.iloc[0]["program_type"])}
    assert list(correction.per_genre_drift.values()) == [0.0]


def test_correction_is_deterministic_and_byte_stable() -> None:
    spots, programmes, dayparts = _break_spots(), _programmes(), _flat_dayparts()
    classifier = _classifier()
    effects = break_effects(spots, programmes, dayparts, classifier)
    first = measure_placebo_drift(spots, programmes, dayparts, classifier, effects)
    second = measure_placebo_drift(spots, programmes, dayparts, classifier, effects)
    assert first == second
    assert json.dumps(first.as_metadata(), sort_keys=False) == json.dumps(
        second.as_metadata(), sort_keys=False
    )


# --- content-only baseline math (hand-computed) -------------------------------

def _two_day_dayparts() -> pd.DataFrame:
    """Two days, minutes 20:00-20:05: day 2 dips to 6.0 at 20:02 and 20:03."""
    rows = []
    for m in range(0, 6):
        rows.append({"date": pd.Timestamp("2024-11-04"), "timeband": f"20:0{m}", "channel": "A", "tvr": 10.0})
    for m in range(0, 6):
        tvr = 6.0 if m in (2, 3) else 10.0
        rows.append({"date": pd.Timestamp("2024-11-05"), "timeband": f"20:0{m}", "channel": "A", "tvr": tvr})
    return pd.DataFrame(rows)


def test_content_only_baseline_excludes_ad_minutes_by_wall_clock() -> None:
    # One 90-second single-spot run on day 2 covers wall-clock minutes
    # 20:02-20:03 of that day only. Plain baseline at broadcast minutes
    # 20:02/20:03 averages both days: (10 + 6) / 2 = 8. Content-only drops
    # day 2's ad minutes there, keeping day 1's content minute: 10. Every
    # other minute is untouched (day 1's same clock minutes stay in).
    frame = _dayparts_frame(_two_day_dayparts())
    spots = pd.DataFrame([_spot("A", "2024-11-05 20:02:00", 90)])
    plain = _baseline_levels(frame)
    clean = _content_only_baseline_levels(frame, spots)
    mod_2002 = (20 - 2) * 60 + 2
    mod_2003 = mod_2002 + 1
    assert plain[("A", mod_2002)] == pytest.approx(8.0)
    assert plain[("A", mod_2003)] == pytest.approx(8.0)
    assert clean[("A", mod_2002)] == pytest.approx(10.0)
    assert clean[("A", mod_2003)] == pytest.approx(10.0)
    for mod in (mod_2002 - 2, mod_2002 - 1, mod_2003 + 1, mod_2003 + 2):
        assert clean[("A", mod)] == pytest.approx(plain[("A", mod)])


def test_content_only_baseline_with_no_spots_equals_plain() -> None:
    frame = _dayparts_frame(_two_day_dayparts())
    no_spots = pd.DataFrame(columns=["Channel", "air_dt", "Duration"])
    assert _content_only_baseline_levels(frame, no_spots) == _baseline_levels(frame)


def _contaminated_dayparts() -> pd.DataFrame:
    """Break day 2024-11-04 plus three days whose after-window minutes are ads.

    Before-window (20:27-20:29): 10.0 every day. After-window (20:32-20:34):
    9.0 on the break day; 8.0 on the other three days, where a single-spot ad
    run covers exactly those minutes and depresses the curve.
    """
    rows = []
    for m in range(0, 60):
        rows.append({"date": pd.Timestamp("2024-11-04"), "timeband": f"20:{m:02d}", "channel": "A",
                     "tvr": 9.0 if m in (32, 33, 34) else 10.0})
    for date in ("2024-11-05", "2024-11-06", "2024-11-07"):
        for m in range(0, 60):
            rows.append({"date": pd.Timestamp(date), "timeband": f"20:{m:02d}", "channel": "A",
                         "tvr": 8.0 if m in (32, 33, 34) else 10.0})
    return pd.DataFrame(rows)


def test_break_effects_content_only_baseline_hand_computed() -> None:
    # Other days' ad runs (single spots, so NOT detected breaks) depress the
    # plain baseline at the after-window: base_after = (9 + 8*3) / 4 = 8.25,
    # observed 9/10 vs expected 8.25/10 makes the break look spuriously GOOD
    # (log_effect = log(0.9 / 0.825) > 0). Content-only excludes those ad
    # minutes, base_after = 9 (the break day's own content), and the effect
    # is exactly 0. This is the review's contamination mechanism in miniature.
    programmes, classifier = _programmes(), _classifier()
    spots = pd.concat(
        [
            _break_spots(),
            pd.DataFrame(
                [_spot("A", f"{d} 20:32:00", 150) for d in ("2024-11-05", "2024-11-06", "2024-11-07")]
            ),
        ],
        ignore_index=True,
    )
    dayparts = _contaminated_dayparts()
    plain = break_effects(spots, programmes, dayparts, classifier)
    clean = break_effects(spots, programmes, dayparts, classifier, baseline_content_only=True)
    assert len(plain) == 1 and len(clean) == 1
    assert plain.iloc[0]["log_effect"] == pytest.approx(np.log(0.9 / 0.825), abs=1e-12)
    assert clean.iloc[0]["log_effect"] == pytest.approx(0.0, abs=1e-12)


def test_content_only_rejects_seasonal_baseline_combination() -> None:
    with pytest.raises(ValueError, match="unmeasured configuration"):
        break_effects(
            _break_spots(), _programmes(), _flat_dayparts(), _classifier(),
            baseline_content_only=True, baseline_seasonality="month_minute",
        )


# --- gated rebuild end to end (real reference data) ---------------------------

def _run_rebuild(tmp_path: Path, *extra: str) -> tuple[dict, str]:
    out = tmp_path / "coeffs.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env["PYTHONUTF8"] = "1"
    for key in ("KAIROS_SERIES_LAYER", "KAIROS_COUNTERPROGRAMMING", "KAIROS_PLACEBO_CORRECTION"):
        env.pop(key, None)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--output", str(out), *extra],
        cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(out.read_text(encoding="utf-8")), proc.stdout


@pytest.mark.realdata
def test_force_off_reproduces_the_uncorrected_charge(tmp_path) -> None:
    """The correction ships ON by default, so force-off is the diagnostic path:
    it must reproduce the UNCORRECTED measurement (pooled about -0.0391) and be
    strictly cheaper per break than the shipped corrected coefficients."""
    fresh, _stdout = _run_rebuild(tmp_path, "--placebo-correction", "force-off",
                                  "--interval-calibration", "force-off")
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    meta = fresh["metadata"]
    assert meta["placebo_correction_active"] is False
    assert "force-off" in meta["placebo_correction_reason"]
    assert meta["placebo_correction"]["pooled_drift"] > 0
    # Uncorrected pooled charge, the pre-correction measurement: the
    # n-weighted mean of the EB cells sits at the old pooled value.
    detail = fresh["detail"]
    total_n = sum(d["n"] for d in detail.values())
    pooled_charge = sum(d["coefficient"] * d["n"] for d in detail.values()) / total_n
    assert pooled_charge == pytest.approx(-0.0391, abs=0.003)
    # Every shipped (corrected) coefficient charges at least as much.
    assert set(fresh["coefficients"]) == set(shipped["coefficients"])
    cheaper = sum(
        1
        for name, value in fresh["coefficients"].items()
        if value > shipped["coefficients"][name]
    )
    assert cheaper == len(shipped["coefficients"])


@pytest.mark.realdata
def test_force_on_reproduces_candidate_artifact(tmp_path) -> None:
    fresh, stdout = _run_rebuild(tmp_path, "--placebo-correction", "force-on")
    shipped = json.loads(SHIPPED.read_text(encoding="utf-8"))
    candidate = json.loads(CANDIDATE.read_text(encoding="utf-8"))

    meta = fresh["metadata"]
    assert meta["placebo_correction_active"] is True
    drift = meta["placebo_correction"]
    assert drift["baseline"] == "content_only"
    assert drift["n_pseudo"] > 5000
    assert set(drift["per_genre_drift"]) == {"News", "Other", "PrimeShow1", "PrimeShow2"}
    # The review's combined fix 1 + fix 2 target: about -0.0496 pooled.
    assert drift["pooled_corrected_delta"] == pytest.approx(-0.0496, abs=0.002)
    assert "placebo correction ACTIVE" in stdout

    # force-on now matches the default: the rebuild reproduces the SHIPPED
    # corrected coefficients exactly, and the candidate that predated the
    # default flip agrees on every point coefficient.
    assert fresh["coefficients"] == shipped["coefficients"]
    assert fresh["coefficients"] == candidate["coefficients"]
    assert drift == candidate["metadata"]["placebo_correction"]
