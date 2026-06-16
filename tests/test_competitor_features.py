"""Tests for the Stage 3 competitor-context features and the information boundary.

These prove the load-bearing property of Stage 3: the forward features are read
only from the rival EPG and audience curve (available before the week airs), the
training-only feature is read from rival ad logs, and the boundary that keeps the
training-only signal out of any forward decision is enforced in code. All pure
pandas, no Meridian.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.model.competitor_features import (
    ALL_FEATURES,
    FORWARD_FEATURES,
    TRAINING_ONLY_FEATURES,
    ForwardBoundaryError,
    assert_forward_only,
    attach_competitor_features,
)


# --- the information boundary -------------------------------------------------

def test_forward_and_training_features_are_disjoint() -> None:
    assert set(FORWARD_FEATURES).isdisjoint(TRAINING_ONLY_FEATURES)
    assert set(ALL_FEATURES) == set(FORWARD_FEATURES) | set(TRAINING_ONLY_FEATURES)


def test_assert_forward_only_passes_for_forward_features() -> None:
    # The forward features are legitimately available at decision time.
    assert_forward_only(FORWARD_FEATURES)  # must not raise


def test_assert_forward_only_rejects_training_only_feature() -> None:
    # Rival ad placement is known only historically; it must never reach the
    # forward path. The boundary is enforced, not merely documented.
    with pytest.raises(ForwardBoundaryError):
        assert_forward_only(["competitor_strength", "competitor_in_break"])


# --- synthetic fixtures for the extractor ------------------------------------

_OWN = "קשת 12"
_RIVAL = "רשת 13"


def _classifier():
    from kairos.data import ProgramClassifier

    return ProgramClassifier.from_yaml()


def _breaks_frame() -> pd.DataFrame:
    """One break on the own channel from 20:06 to 20:07."""
    return pd.DataFrame(
        [{
            "channel": _OWN,
            "break_start": pd.Timestamp("2024-11-04 20:06:00"),
            "break_end": pd.Timestamp("2024-11-04 20:07:00"),
        }]
    )


def _dayparts(rival_tvr: float) -> pd.DataFrame:
    """Daypart rows giving the rival a known audience at the break minutes."""
    rows = []
    for tb in ("20:06", "20:07"):
        rows.append({"date": pd.Timestamp("2024-11-04"), "timeband": tb,
                     "channel": _RIVAL, "tvr": rival_tvr})
        rows.append({"date": pd.Timestamp("2024-11-04"), "timeband": tb,
                     "channel": _OWN, "tvr": 5.0})
    return pd.DataFrame(rows)


def _programmes(rival_title: str, own_title: str) -> pd.DataFrame:
    rows = [
        (own_title, _OWN, "2024-11-04 20:00:00", "2024-11-04 21:00:00"),
        (rival_title, _RIVAL, "2024-11-04 20:00:00", "2024-11-04 21:00:00"),
    ]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start", "end"])
    frame["start_dt"] = pd.to_datetime(frame["start"])
    frame["end_dt"] = pd.to_datetime(frame["end"])
    return frame


def _spots(rival_in_break: bool) -> pd.DataFrame:
    """Optionally a rival two-spot break overlapping the own break window."""
    if not rival_in_break:
        return pd.DataFrame(columns=["Channel", "air_dt", "Duration"])
    return pd.DataFrame([
        {"Channel": _RIVAL, "air_dt": pd.Timestamp("2024-11-04 20:06:00"), "Duration": 30.0},
        {"Channel": _RIVAL, "air_dt": pd.Timestamp("2024-11-04 20:06:30"), "Duration": 30.0},
    ])


def test_attach_competitor_features_reads_rival_strength() -> None:
    # The rival holds a known audience opposite the break, so competitor_strength
    # is positive and rises with the rival's audience.
    weak = attach_competitor_features(
        _breaks_frame(), _programmes("דרמה", "חדשות"), _dayparts(2.0), _spots(False), _classifier(),
    )
    strong = attach_competitor_features(
        _breaks_frame(), _programmes("דרמה", "חדשות"), _dayparts(8.0), _spots(False), _classifier(),
    )
    assert weak.iloc[0]["competitor_strength"] > 0.0
    assert strong.iloc[0]["competitor_strength"] > weak.iloc[0]["competitor_strength"]


def test_genre_contrast_fires_only_when_a_rival_airs_the_same_genre() -> None:
    # Same title on a rival -> same classifier category -> a substitute is airing,
    # so the contrast is positive. A different genre leaves it at zero.
    same = attach_competitor_features(
        _breaks_frame(), _programmes("חדשות", "חדשות"), _dayparts(3.0), _spots(False), _classifier(),
    )
    diff = attach_competitor_features(
        _breaks_frame(), _programmes("סרט קומדיה", "חדשות"), _dayparts(3.0), _spots(False), _classifier(),
    )
    assert same.iloc[0]["competitor_genre_contrast"] > 0.0
    assert diff.iloc[0]["competitor_genre_contrast"] == 0.0


def test_competitor_in_break_is_a_real_overlap_fraction() -> None:
    # When a rival aired its own break across the same minutes, the training-only
    # feature is a positive overlap fraction; with no rival break it is zero.
    with_rival = attach_competitor_features(
        _breaks_frame(), _programmes("דרמה", "חדשות"), _dayparts(3.0), _spots(True), _classifier(),
    )
    without = attach_competitor_features(
        _breaks_frame(), _programmes("דרמה", "חדשות"), _dayparts(3.0), _spots(False), _classifier(),
    )
    assert with_rival.iloc[0]["competitor_in_break"] > 0.0
    assert without.iloc[0]["competitor_in_break"] == 0.0


def test_attach_competitor_features_empty_frame_keeps_columns() -> None:
    empty = pd.DataFrame(columns=["channel", "break_start", "break_end"])
    out = attach_competitor_features(
        empty, _programmes("דרמה", "חדשות"), _dayparts(3.0), _spots(False), _classifier(),
    )
    assert out.empty
    for name in ALL_FEATURES:
        assert name in out.columns
