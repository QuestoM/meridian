"""Tests for the counter-programming covariate: feature math, gate, contract.

Hand-computed synthetic fixtures prove: the rival-programme-start feature is
the exact fraction of rivals with an EPG start in the window; the held-out gate
activates on a planted competitor effect and stays off on noise; the future-EPG
file contract parses and computes features honestly, contributing EXACTLY
nothing when the file is absent; and the information boundary fails loudly on
a mislabeled training-only beta. Pure pandas, no Meridian.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kairos.model.competitor_features import (
    ForwardBoundaryError,
    attach_competitor_features,
)
from kairos.model.competitor_gate import counterprogramming_holdout_gate
from kairos.model.future_epg import (
    counterprogramming_features_for_window,
    forward_adjustment,
    load_future_competitor_epg,
)

_OWN = "קשת 12"
_RIVAL = "רשת 13"


def _classifier():
    from kairos.data import ProgramClassifier

    return ProgramClassifier.from_yaml()


def _breaks_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "channel": _OWN,
            "break_start": pd.Timestamp("2024-11-04 20:06:00"),
            "break_end": pd.Timestamp("2024-11-04 20:07:00"),
        }]
    )


def _dayparts() -> pd.DataFrame:
    rows = []
    for tb in ("20:06", "20:07"):
        rows.append({"date": pd.Timestamp("2024-11-04"), "timeband": tb,
                     "channel": _RIVAL, "tvr": 3.0})
        rows.append({"date": pd.Timestamp("2024-11-04"), "timeband": tb,
                     "channel": _OWN, "tvr": 5.0})
    return pd.DataFrame(rows)


def _programmes(rival_start_in_window: bool) -> pd.DataFrame:
    """Own programme spans the break; ONE rival has a programme start at 20:08,
    inside [break_start - 1 min, break_end + 3 min] = [20:05, 20:10], when
    requested. The other two rivals only have programmes starting at 19:00.
    """
    rows = [("תוכנית", _OWN, "2024-11-04 19:00:00", "2024-11-04 21:00:00")]
    if rival_start_in_window:
        rows.append(("סרט ערב", _RIVAL, "2024-11-04 19:00:00", "2024-11-04 20:08:00"))
        rows.append(("חדשות לילה", _RIVAL, "2024-11-04 20:08:00", "2024-11-04 21:00:00"))
    else:
        rows.append(("סרט ערב", _RIVAL, "2024-11-04 19:00:00", "2024-11-04 21:00:00"))
    for other in ("כאן 11", "עכשיו 14"):
        rows.append(("תוכנית אחרת", other, "2024-11-04 19:00:00", "2024-11-04 21:00:00"))
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "end"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    frame["end_dt"] = pd.to_datetime(frame["end"])
    return frame


def _no_spots() -> pd.DataFrame:
    return pd.DataFrame(columns=["Channel", "air_dt", "Duration"])


# --- the rival-programme-start feature ---------------------------------------

def test_prog_start_is_the_fraction_of_rivals_starting_in_window() -> None:
    # Exactly one of the three rivals has a programme start (20:08) inside
    # [20:05, 20:10], so the feature is 1/3 by hand.
    out = attach_competitor_features(
        _breaks_frame(), _programmes(True), _dayparts(), _no_spots(), _classifier(),
    )
    assert out.iloc[0]["competitor_prog_start"] == pytest.approx(1.0 / 3.0)


def test_prog_start_zero_when_no_rival_start_is_near() -> None:
    out = attach_competitor_features(
        _breaks_frame(), _programmes(False), _dayparts(), _no_spots(), _classifier(),
    )
    assert out.iloc[0]["competitor_prog_start"] == 0.0


# --- the held-out gate ---------------------------------------------------------

def _gate_frame(*, planted_beta: float, seed: int = 7, n: int = 300) -> pd.DataFrame:
    """Two cells; strength varies within cells; other features constant.

    log_effect = planted_beta * strength + N(0, 0.01) noise, so with a real
    planted effect the covariate must dominate the cell mean out of sample.
    """
    rng = np.random.default_rng(seed)
    cells = np.where(rng.random(n) < 0.5, "News_first_short", "Other_last_long")
    strength = rng.uniform(0.0, 10.0, size=n)
    noise = rng.normal(0.0, 0.01, size=n)
    return pd.DataFrame({
        "channel_name": cells,
        "log_effect": planted_beta * strength + noise,
        "competitor_strength": strength,
        "competitor_genre_contrast": np.full(n, 0.5),
        "competitor_prog_start": np.zeros(n),
        "competitor_in_break": np.full(n, 0.3),
    })


def test_gate_activates_on_a_planted_competitor_effect() -> None:
    gate = counterprogramming_holdout_gate(_gate_frame(planted_beta=-0.02))
    hold = gate["counterprogramming_holdout"]
    assert gate["counterprogramming_active"] is True
    assert hold["rmse_with"] < hold["rmse_without"] * 0.98
    beta = gate["counterprogramming_betas"]["competitor_strength"]["beta"]
    assert beta == pytest.approx(-0.02, abs=0.005)


def test_gate_stays_off_on_pure_noise() -> None:
    gate = counterprogramming_holdout_gate(_gate_frame(planted_beta=0.0))
    assert gate["counterprogramming_active"] is False
    assert "covariate left off" in gate["counterprogramming_reason"]


def test_gate_reports_honestly_on_empty_effects() -> None:
    gate = counterprogramming_holdout_gate(pd.DataFrame())
    assert gate["counterprogramming_active"] is False
    assert gate["counterprogramming_holdout"]["n_test"] == 0


# --- the future-EPG file contract ----------------------------------------------

def _contract_csv(tmp_path) -> str:
    path = tmp_path / "CompetitorProgrammes.csv"
    frame = pd.DataFrame({
        "Channel": [_OWN, _RIVAL, _RIVAL],
        "Title": ["תוכנית", "סרט ערב", "חדשות לילה"],
        "Date": ["04/11/2024", "04/11/2024", "04/11/2024"],
        "Start time": ["19:00:00", "19:00:00", "20:08:00"],
        "End time": ["21:00:00", "20:08:00", "21:00:00"],
        "Duration": [7200, 4080, 3120],
    })
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


def test_future_epg_contract_parses_and_computes_features(tmp_path) -> None:
    epg, status = load_future_competitor_epg(_contract_csv(tmp_path))
    assert status["present"] is True
    assert status["rows"] == 3
    assert status["window_start"] == "2024-11-04"
    feats = counterprogramming_features_for_window(
        window_start=pd.Timestamp("2024-11-04 20:06:00"),
        window_end=pd.Timestamp("2024-11-04 20:07:00"),
        epg=epg,
        classifier=_classifier(),
        baseline={(_RIVAL, m): 4.0 for m in range(0, 1440)},
        own_channel=_OWN,
        own_category=None,
    )
    # One rival in the file; its 20:08 start is inside the window: 1/1.
    assert feats["competitor_prog_start"] == pytest.approx(1.0)
    # Strength is the historical curve's 4.0 at every window minute.
    assert feats["competitor_strength"] == pytest.approx(4.0)
    # Unknown own genre contributes no contrast, as in training.
    assert feats["competitor_genre_contrast"] == 0.0


def test_future_epg_absent_state_contributes_exactly_nothing(tmp_path) -> None:
    epg, status = load_future_competitor_epg(tmp_path / "does_not_exist.csv")
    assert epg is None
    assert status["present"] is False
    assert "contributes nothing" in status["reason"]
    betas = {"competitor_strength": {"beta": -0.002, "reference": 3.0, "role": "forward"}}
    adj = forward_adjustment(None, betas)
    assert adj["adjustment"] == 0.0
    assert adj["applied"] is False


def test_forward_adjustment_hand_computed() -> None:
    betas = {"competitor_strength": {"beta": -0.002, "reference": 3.0, "role": "forward"}}
    adj = forward_adjustment({"competitor_strength": 4.0}, betas)
    # -0.002 * (4.0 - 3.0) = -0.002 by hand.
    assert adj["adjustment"] == pytest.approx(-0.002)
    assert adj["applied"] is True


def test_mislabelled_training_only_beta_fails_loudly() -> None:
    betas = {"competitor_in_break": {"beta": 1.0, "reference": 0.0, "role": "forward"}}
    with pytest.raises(ForwardBoundaryError):
        forward_adjustment({"competitor_in_break": 1.0}, betas)


def test_epg_with_only_the_own_channel_is_treated_as_absent(tmp_path) -> None:
    path = tmp_path / "CompetitorProgrammes.csv"
    pd.DataFrame({
        "Channel": [_OWN], "Title": ["תוכנית"], "Date": ["04/11/2024"],
        "Start time": ["19:00:00"], "End time": ["21:00:00"], "Duration": [7200],
    }).to_csv(path, index=False, encoding="utf-8-sig")
    epg, status = load_future_competitor_epg(path)
    assert status["present"] is True  # the file exists and parses
    feats = counterprogramming_features_for_window(
        window_start=pd.Timestamp("2024-11-04 20:06:00"),
        window_end=pd.Timestamp("2024-11-04 20:07:00"),
        epg=epg,
        classifier=_classifier(),
        baseline={},
        own_channel=_OWN,
    )
    assert feats is None  # no rivals -> no covariate, never a fabricated one
