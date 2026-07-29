"""Contract tests for the calendar-event training seam and the gated event layer.

Proves the four properties the seam must hold:

  1. Annotation correctness: event_active / event_intensity / event_type joined
     on the break date from the events store, overlaps taking the max-intensity
     event, open-ended events covering forward, inactive rows ignored.
  2. Additivity: the pooled coefficients are byte-identical with the annotation
     columns present, on synthetic frames and on the real measured breaks.
  3. Honest verdict today: on the real 30-day window (entirely wartime) the
     gate says off with a reason naming the missing on/off contrast, a null
     held-out delta, and the frozen four-key metadata shape.
  4. Self-activation: on a synthetic history where an event layer genuinely
     improves held-out RMSE past the +2 percent bar, the verdict flips to on
     with no code change, which is the owner's future-wars requirement.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from kairos.model.event_gate import (
    ANNOTATION_COLUMNS,
    annotate_event_columns,
    event_layer_gate,
    load_training_events,
)
from kairos.model.measure import channel_coefficients


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_STORE_HEADER = (
    "event_id,name,type,start_date,end_date,intensity,notes,active,price_multiplier\n"
)


def _write_store(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "calendar_events.csv"
    path.write_text(_STORE_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _store_with_overlap(tmp_path: Path) -> Path:
    """War A (int 3), sport B overlapping it (int 5), an inactive event, and an
    open-ended war starting 2024-12-01."""
    return _write_store(tmp_path, [
        "ev-a,war a,war,2024-11-10,2024-11-20,3,,True,1.0",
        "ev-b,sport b,sport,2024-11-18,2024-11-25,5,,True,1.0",
        "ev-c,inactive,holiday,2024-11-05,2024-11-05,4,,False,1.0",
        "ev-d,open war,war,2024-12-01,,2,,True,1.0",
    ])


def _effects_frame(dated_rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    """Build an effects frame from (break_start, channel_name, log_effect)."""
    return pd.DataFrame([
        {"channel_name": cell, "log_effect": effect,
         "break_start": pd.Timestamp(stamp)}
        for stamp, cell, effect in dated_rows
    ])


# ---------------------------------------------------------------------------
# 1. Annotation correctness
# ---------------------------------------------------------------------------

def test_annotation_joins_on_break_date(tmp_path) -> None:
    store = _store_with_overlap(tmp_path)
    effects = _effects_frame([
        ("2024-11-05 20:06:00", "News_first_short", -0.10),   # inactive event only
        ("2024-11-12 20:06:00", "News_first_short", -0.10),   # inside war a
        ("2024-11-19 20:06:00", "News_first_short", -0.10),   # overlap: sport b wins on intensity
        ("2024-12-15 20:06:00", "News_first_short", -0.10),   # open-ended war covers forward
    ])
    out = annotate_event_columns(effects, store)
    assert list(out["event_active"]) == [0, 1, 1, 1]
    assert list(out["event_intensity"]) == [0, 3, 5, 2]
    assert list(out["event_type"]) == ["", "war", "sport", "war"]
    # The original columns are carried through untouched.
    assert list(out["log_effect"]) == list(effects["log_effect"])


def test_annotation_defaults_without_store_or_break_start(tmp_path) -> None:
    effects = _effects_frame([("2024-11-12 20:06:00", "News_first_short", -0.10)])
    out = annotate_event_columns(effects, tmp_path / "missing.csv")
    assert list(out["event_active"]) == [0]
    undated = pd.DataFrame({"channel_name": ["News_first_short"], "log_effect": [-0.1]})
    out2 = annotate_event_columns(undated, tmp_path / "missing.csv")
    for column in ANNOTATION_COLUMNS:
        assert column in out2.columns
    assert list(out2["event_active"]) == [0]


def test_reader_skips_inactive_and_malformed_rows(tmp_path) -> None:
    store = _write_store(tmp_path, [
        "ev-a,ok,war,2024-11-10,2024-11-20,3,,True,1.0",
        "ev-b,inactive,war,2024-11-10,2024-11-20,5,,False,1.0",
        "ev-c,bad start,war,not-a-date,2024-11-20,5,,True,1.0",
        "ev-d,end before start,war,2024-11-20,2024-11-10,5,,True,1.0",
        "ev-e,unparseable intensity,special,2024-11-01,2024-11-02,much,,True,1.0",
    ])
    events = load_training_events(store)
    assert len(events) == 2
    assert events[0].intensity == 3
    # An active event with an unreadable intensity is still an event (defaults 1).
    assert events[1].intensity == 1 and events[1].event_type == "special"


# ---------------------------------------------------------------------------
# 2. Additivity: coefficients byte-identical with the seam active
# ---------------------------------------------------------------------------

def test_annotation_leaves_pooled_coefficients_byte_identical(tmp_path) -> None:
    store = _store_with_overlap(tmp_path)
    rng = np.random.default_rng(7)
    effects = _effects_frame([
        (f"2024-11-{d:02d} 20:06:00", cell, float(rng.normal(-0.08, 0.03)))
        for d in range(1, 29)
        for cell in ("News_first_short", "Other_last_long")
    ])
    plain = channel_coefficients(effects)
    annotated = channel_coefficients(annotate_event_columns(effects, store))
    assert plain == annotated  # dataclass equality: every field of every cell


@pytest.mark.realdata
def test_real_window_coefficients_byte_identical_with_annotation() -> None:
    """The real rebuild path: measuring the reference month and pooling with the
    annotation columns present yields the exact same coefficients, field for
    field, as pooling the plain frame. This is the additive-seam proof on the
    shipped data (the default events store covers the whole window)."""
    from kairos.data import ProgramClassifier
    from kairos.data.loaders import load_dayparts, load_programmes, load_spots
    from kairos.model.measure import break_effects

    effects = break_effects(
        load_spots(), load_programmes(), load_dayparts(), ProgramClassifier.from_yaml()
    )
    annotated = annotate_event_columns(effects)
    # The real window is fully covered by the stored wartime events.
    assert int((annotated["event_active"] == 1).sum()) == len(annotated)
    assert channel_coefficients(effects) == channel_coefficients(annotated)


# ---------------------------------------------------------------------------
# 3. Honest verdict today + frozen metadata shape
# ---------------------------------------------------------------------------

def test_gate_result_shape_is_the_frozen_contract(tmp_path) -> None:
    result = event_layer_gate(pd.DataFrame(), tmp_path / "missing.csv")
    assert set(result) == {"verdict", "reason", "held_out_delta_pct", "measured_at"}
    assert result["verdict"] == "off"
    assert result["held_out_delta_pct"] is None
    # measured_at is a parseable ISO timestamp; a fixed value is honored.
    datetime.fromisoformat(str(result["measured_at"]))
    pinned = event_layer_gate(
        pd.DataFrame(), tmp_path / "missing.csv", measured_at="2026-07-29T00:00:00+00:00"
    )
    assert pinned["measured_at"] == "2026-07-29T00:00:00+00:00"


def test_gate_off_without_contrast(tmp_path) -> None:
    """Every break inside an event: no on/off contrast, verdict off, reason says so."""
    store = _write_store(tmp_path, ["ev-a,war,war,2024-11-01,2024-11-30,4,,True,1.0"])
    rng = np.random.default_rng(3)
    effects = _effects_frame([
        (f"2024-11-{d:02d} 20:06:00", "News_first_short", float(rng.normal(-0.1, 0.02)))
        for d in range(1, 29) for _ in range(4)
    ])
    result = event_layer_gate(effects, store)
    assert result["verdict"] == "off"
    assert result["held_out_delta_pct"] is None
    assert "contrast" in str(result["reason"])


@pytest.mark.realdata
def test_gate_verdict_off_on_the_real_window() -> None:
    """The real 30-day window sits entirely inside stored wartime events, so the
    honest verdict is off with the no-contrast reason and a null delta."""
    from kairos.data import ProgramClassifier
    from kairos.data.loaders import load_dayparts, load_programmes, load_spots
    from kairos.model.measure import break_effects

    effects = break_effects(
        load_spots(), load_programmes(), load_dayparts(), ProgramClassifier.from_yaml()
    )
    result = event_layer_gate(effects)
    assert result["verdict"] == "off"
    assert result["held_out_delta_pct"] is None
    assert "no event on/off contrast" in str(result["reason"])


def test_shipped_metadata_carries_the_gate_key() -> None:
    """The shipped coefficients artifact records the gate verdict under the
    frozen key, so the events API's model context can read it tri-state."""
    import json

    root = Path(__file__).resolve().parents[1]
    payload = json.loads(
        (root / "models" / "tv_break_coefficients.json").read_text(encoding="utf-8")
    )
    gate = payload["metadata"]["event_layer_gate"]
    assert set(gate) == {"verdict", "reason", "held_out_delta_pct", "measured_at"}
    assert gate["verdict"] in ("on", "off")


# ---------------------------------------------------------------------------
# 4. Self-activation on data with real contrast
# ---------------------------------------------------------------------------

def _contrastful_effects(store_shift: float, rng_seed: int = 11) -> tuple[pd.DataFrame, list[str]]:
    """100 days, 2 cells, 2 breaks per cell-day; two war spans in different
    fifths of the range so every temporal fold trains on both arms. Breaks
    inside a war shed ``store_shift`` more than the cell baseline."""
    rng = np.random.default_rng(rng_seed)
    wars = ["ev-a,war a,war,2025-01-15,2025-02-05,4,,True,1.0",
            "ev-b,war b,war,2025-03-01,2025-03-20,4,,True,1.0"]
    spans = [(pd.Timestamp("2025-01-15"), pd.Timestamp("2025-02-05")),
             (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-20"))]
    days = pd.date_range("2025-01-01", periods=100, freq="D")
    rows = []
    for day in days:
        inside = any(start <= day <= end for start, end in spans)
        for cell, base in (("News_first_short", -0.05), ("Other_last_long", -0.02)):
            for hour in (20, 21):
                effect = base + (store_shift if inside else 0.0) + float(rng.normal(0, 0.01))
                rows.append((f"{day.date()} {hour}:06:00", cell, effect))
    return _effects_frame(rows), wars


def test_gate_self_activates_on_real_contrast(tmp_path) -> None:
    effects, wars = _contrastful_effects(-0.20)
    store = _write_store(tmp_path, wars)
    result = event_layer_gate(effects, store)
    assert result["verdict"] == "on"
    assert result["held_out_delta_pct"] is not None
    assert float(result["held_out_delta_pct"]) > 2.0
    assert "activated" in str(result["reason"])


def test_gate_stays_off_when_events_carry_no_signal(tmp_path) -> None:
    """Same fold structure and both arms populated, but events shift nothing:
    the layer cannot beat the baseline by the bar, so the verdict stays off."""
    effects, wars = _contrastful_effects(0.0)
    store = _write_store(tmp_path, wars)
    result = event_layer_gate(effects, store)
    assert result["verdict"] == "off"
    assert result["held_out_delta_pct"] is not None
    assert float(result["held_out_delta_pct"]) <= 2.0
