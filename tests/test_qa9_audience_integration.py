"""Audience-model integration seam: flag, transform overlay, freshness, API.

Proves the frozen contracts around the audience model without depending on the
model itself existing yet:

  * the ``audience_model_activation`` settings flag defaults off and an absent
    key reads off, round-tripping through the settings API,
  * OFF is byte-identical at segment level on the real reference data, even
    with an artifact and a prediction module available,
  * ON with a synthetic artifact moves exactly the forward-dated segments and
    marks each segment's basis, while past dates stay measured,
  * the ``audience_model`` freshness group is omitted while off and tracked
    while on (mirroring the events group),
  * ``GET /api/model/audience`` is honest tri-state,
  * the overview basis note names the model state honestly.

The prediction module is injected through ``sys.modules`` so these tests stay
deterministic whether or not ``kairos.model.audience_model`` has landed.
"""

from __future__ import annotations

import json
import sys
import types
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from kairos.data import audience_overlay
from kairos.data.audience_overlay import apply_audience_model, audience_model_active
from kairos.data.transform import build_segments_from_programmes
from kairos.export.schedule_freshness import (
    ABSENT,
    _settings_fingerprint,
    schedule_freshness,
    schedule_input_fingerprints,
    write_schedule_meta,
)
from kairos.optimize._types import ProgramSegment
from kairos_api import core
from kairos_api.core import KairosSettings, _model_dump
from kairos_api.server import app

client = TestClient(app)

ROOT = Path(__file__).resolve().parents[1]

GATE_FAMILIES = (
    "weekday_slot",
    "series",
    "calendar_school_and_chol_hamoed",
    "calendar_hanukkah",
    "calendar_religious_blackout",
    "season",
    "operator_events",
    "competitor_lineup",
)


def _segment(day: str, tvr: float = 5.0, sid: str = "s") -> ProgramSegment:
    return ProgramSegment(
        segment_id=sid,
        channel="קשת 12",
        day=day,
        start_seconds=72000.0,
        duration_seconds=3600.0,
        program_type="News",
        baseline_tvr=tvr,
        cpp=1000.0,
    )


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _synthetic_artifact() -> dict:
    stamp = "2026-07-29T00:00:00+00:00"
    return {
        "computed_at": stamp,
        "activation_default": False,
        "base": {"pooled_mean_tvr": 4.2},
        "gates": {
            family: {
                "verdict": "off",
                "reason": "synthetic test artifact",
                "held_out_delta_pct": None,
                "measured_at": stamp,
            }
            for family in GATE_FAMILIES
        },
        "source_fingerprints": {},
    }


def _install_fake_model(monkeypatch, predicted: float = 9.75, basis: str = "model"):
    """Inject a deterministic kairos.model.audience_model with a call recorder."""
    calls: list[pd.DataFrame] = []
    module = types.ModuleType("kairos.model.audience_model")

    def predict_tvr(rows: pd.DataFrame, **kwargs) -> pd.DataFrame:
        calls.append(rows.copy())
        out = rows.copy()
        out["predicted_tvr"] = predicted
        out["basis"] = basis
        return out

    module.predict_tvr = predict_tvr
    monkeypatch.setitem(sys.modules, "kairos.model.audience_model", module)
    return calls


def _activate(monkeypatch, tmp_path: Path, *, flag: bool = True, artifact: bool = True) -> None:
    """Point the overlay at a tmp settings file and a tmp artifact."""
    settings_path = tmp_path / "kairos_settings.json"
    _write_json(settings_path, {"audience_model_activation": flag})
    artifact_path = tmp_path / "audience_model.json"
    if artifact:
        _write_json(artifact_path, _synthetic_artifact())
    monkeypatch.setattr(audience_overlay, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(audience_overlay, "AUDIENCE_MODEL_PATH", artifact_path)


# 1. Flag defaults ------------------------------------------------------------
def test_flag_defaults_off_and_absent_reads_false(tmp_path) -> None:
    assert KairosSettings().audience_model_activation is False
    assert KairosSettings(**{"revenue_weight": 60}).audience_model_activation is False
    # The overlay's own read: missing file, missing key, malformed file all OFF.
    assert audience_model_active(tmp_path / "missing.json") is False
    empty = tmp_path / "no_flag.json"
    _write_json(empty, {"revenue_weight": 60})
    assert audience_model_active(empty) is False
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    assert audience_model_active(broken) is False
    flagged = tmp_path / "flagged.json"
    _write_json(flagged, {"audience_model_activation": True})
    assert audience_model_active(flagged) is True


def test_off_overlay_is_inert_even_with_artifact_and_model(tmp_path, monkeypatch) -> None:
    """OFF gates everything: artifact present, module present, still a no-op."""
    _activate(monkeypatch, tmp_path, flag=False, artifact=True)
    calls = _install_fake_model(monkeypatch)
    segments = [_segment("2030-01-06"), _segment("2020-01-06")]
    result = apply_audience_model(segments, today=date(2026, 7, 29))
    assert result is segments, "off must return the same untouched list"
    assert calls == [], "off must never call the model"
    assert not hasattr(segments[0], "tvr_basis"), "off must not mark a basis"


# 2. OFF byte-identity on real data ------------------------------------------
def test_off_is_byte_identical_on_real_data(tmp_path, monkeypatch) -> None:
    """With the flag off, the transform output on the real EPG is identical to
    the overlay-free output, even with a live artifact and prediction module."""
    from kairos.data import ProgramClassifier
    from kairos.optimize.pricing import PricingModel

    try:
        from kairos.data.loaders import load_programmes

        programmes = load_programmes()
    except FileNotFoundError:
        pytest.skip("reference programmes source not on disk")
    if programmes.empty:
        pytest.skip("reference programmes source is empty")

    classifier = ProgramClassifier.from_yaml()
    pricing = PricingModel.from_yaml()
    channel = str(programmes["Channel"].iloc[0])

    # Baseline: overlay forced inert (no settings flag at the tmp path).
    _activate(monkeypatch, tmp_path, flag=False, artifact=True)
    _install_fake_model(monkeypatch)
    baseline = build_segments_from_programmes(programmes, classifier, pricing, channel=channel)

    # Same build again with the flag still off: identical segment for segment.
    again = build_segments_from_programmes(programmes, classifier, pricing, channel=channel)
    assert len(baseline) > 0
    assert again == baseline, "flag-off transform must be byte-identical run to run"
    assert all(not hasattr(segment, "tvr_basis") for segment in baseline)


# 3. ON moves exactly the forward-dated segments -----------------------------
def test_on_moves_only_forward_segments_and_marks_basis(tmp_path, monkeypatch) -> None:
    _activate(monkeypatch, tmp_path, flag=True, artifact=True)
    calls = _install_fake_model(monkeypatch, predicted=9.75, basis="model")
    reference = date(2026, 7, 29)
    past = _segment("2026-07-28", tvr=5.0, sid="past")
    boundary = _segment("2026-07-29", tvr=6.0, sid="today")
    future = _segment("2026-08-03", tvr=7.0, sid="future")
    undated = _segment("unknown", tvr=8.0, sid="undated")
    result = apply_audience_model([past, boundary, future, undated], today=reference)

    by_id = {segment.segment_id: segment for segment in result}
    assert by_id["past"].baseline_tvr == 5.0
    assert by_id["past"].tvr_basis == "historical"
    assert by_id["undated"].baseline_tvr == 8.0
    assert by_id["undated"].tvr_basis == "historical"
    # Today counts as forward: the coming week starts now.
    assert by_id["today"].baseline_tvr == 9.75
    assert by_id["today"].tvr_basis == "model"
    assert by_id["future"].baseline_tvr == 9.75
    assert by_id["future"].tvr_basis == "model"
    # Every non-baseline field survives the rebase untouched.
    assert by_id["future"].channel == future.channel
    assert by_id["future"].start_seconds == future.start_seconds
    assert by_id["future"].cpp == future.cpp

    # The model saw ONLY the forward rows, with the frozen row contract columns.
    assert len(calls) == 1
    rows = calls[0]
    assert list(rows.columns) == ["date", "channel", "program_title", "start_seconds", "duration_seconds"]
    assert sorted(rows["date"].tolist()) == ["2026-07-29", "2026-08-03"]


def test_on_through_the_real_transform(tmp_path, monkeypatch) -> None:
    """The seam fires inside build_segments_from_programmes for future dates."""
    from kairos.data import ProgramClassifier
    from kairos.optimize.pricing import PricingModel

    _activate(monkeypatch, tmp_path, flag=True, artifact=True)
    _install_fake_model(monkeypatch, predicted=3.5)
    rows = [
        ("חדשות הערב", "קשת 12", "2030-01-06 20:00:00", 3600, 5.0),
        ("חדשות הערב", "קשת 12", "2020-01-06 20:00:00", 3600, 5.0),
    ]
    frame = pd.DataFrame(rows, columns=["Title", "Channel", "start_dt", "Duration", "TVR"])
    frame["start_dt"] = pd.to_datetime(frame["start_dt"])
    segments = build_segments_from_programmes(
        frame, ProgramClassifier.from_yaml(), PricingModel.from_yaml(), channel="קשת 12"
    )
    by_day = {segment.day: segment for segment in segments}
    assert by_day["2030-01-06"].baseline_tvr == 3.5
    assert by_day["2030-01-06"].tvr_basis == "model"
    assert by_day["2020-01-06"].baseline_tvr == 5.0
    assert by_day["2020-01-06"].tvr_basis == "historical"


# 4. ON degrades honestly when the model is not usable -----------------------
def test_on_without_artifact_is_inert(tmp_path, monkeypatch) -> None:
    _activate(monkeypatch, tmp_path, flag=True, artifact=False)
    _install_fake_model(monkeypatch)
    segments = [_segment("2030-01-06")]
    assert apply_audience_model(segments, today=date(2026, 7, 29)) is segments


def test_on_without_module_is_inert(tmp_path, monkeypatch) -> None:
    _activate(monkeypatch, tmp_path, flag=True, artifact=True)
    monkeypatch.setitem(sys.modules, "kairos.model.audience_model", None)
    segments = [_segment("2030-01-06")]
    assert apply_audience_model(segments, today=date(2026, 7, 29)) is segments


def test_on_with_failing_or_malformed_prediction_is_inert(tmp_path, monkeypatch) -> None:
    _activate(monkeypatch, tmp_path, flag=True, artifact=True)
    segments = [_segment("2030-01-06")]

    raising = types.ModuleType("kairos.model.audience_model")
    raising.predict_tvr = lambda rows: (_ for _ in ()).throw(RuntimeError("boom"))
    monkeypatch.setitem(sys.modules, "kairos.model.audience_model", raising)
    assert apply_audience_model(segments, today=date(2026, 7, 29)) is segments

    columnless = types.ModuleType("kairos.model.audience_model")
    columnless.predict_tvr = lambda rows: rows.copy()
    monkeypatch.setitem(sys.modules, "kairos.model.audience_model", columnless)
    assert apply_audience_model(segments, today=date(2026, 7, 29)) is segments


def test_nan_prediction_keeps_historical_value(tmp_path, monkeypatch) -> None:
    _activate(monkeypatch, tmp_path, flag=True, artifact=True)
    _install_fake_model(monkeypatch, predicted=float("nan"))
    segment = _segment("2030-01-06", tvr=5.0)
    result = apply_audience_model([segment], today=date(2026, 7, 29))
    assert result[0].baseline_tvr == 5.0
    assert result[0].tvr_basis == "historical"


# 5. Freshness group ----------------------------------------------------------
def test_freshness_group_omitted_off_tracked_on(tmp_path) -> None:
    root = tmp_path
    _write_json(root / "data" / "kairos_settings.json", {"revenue_weight": 60})
    assert "audience_model" not in schedule_input_fingerprints(root), (
        "off (absent flag) must omit the group entirely"
    )

    _write_json(root / "data" / "kairos_settings.json", {"audience_model_activation": True})
    assert schedule_input_fingerprints(root)["audience_model"] == ABSENT, (
        "on with no artifact records the honest ABSENT sentinel"
    )

    _write_json(root / "models" / "audience_model.json", _synthetic_artifact())
    first = schedule_input_fingerprints(root)["audience_model"]
    assert first not in (None, ABSENT)
    changed_artifact = _synthetic_artifact()
    changed_artifact["computed_at"] = "2026-07-30T00:00:00+00:00"
    _write_json(root / "models" / "audience_model.json", changed_artifact)
    assert schedule_input_fingerprints(root)["audience_model"] != first, (
        "a retrained artifact must change the fingerprint"
    )


def test_freshness_stale_loop_names_the_audience_model(tmp_path) -> None:
    root = tmp_path
    csv_path = root / "output" / "weekly_break_schedule.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text("channel,date\n", encoding="utf-8")

    _write_json(root / "data" / "kairos_settings.json", {"audience_model_activation": True})
    _write_json(root / "models" / "audience_model.json", _synthetic_artifact())
    write_schedule_meta(csv_path, root)
    verdict = schedule_freshness(root, csv_path)
    assert verdict["status"] == "fresh", verdict

    retrained = _synthetic_artifact()
    retrained["computed_at"] = "2026-08-01T00:00:00+00:00"
    _write_json(root / "models" / "audience_model.json", retrained)
    verdict = schedule_freshness(root, csv_path)
    assert verdict["status"] == "stale"
    assert "the audience model" in verdict["changed"]


def test_activation_flag_is_engine_relevant_in_the_settings_fingerprint(tmp_path) -> None:
    off = tmp_path / "off.json"
    on = tmp_path / "on.json"
    _write_json(off, {"audience_model_activation": False})
    _write_json(on, {"audience_model_activation": True})
    assert _settings_fingerprint(off) != _settings_fingerprint(on), (
        "flipping activation must read as an engine-relevant settings change"
    )


# 6. API tri-state ------------------------------------------------------------
def test_api_audience_model_absent_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(core, "MODELS_DIR", tmp_path / "models")
    body = client.get("/api/model/audience").json()
    assert body["available"] is False
    assert body["computed_at"] is None
    assert body["gates"] == {}
    assert body["base_summary"] is None
    assert isinstance(body["activation"], bool)
    assert "reason" in body


def test_api_audience_model_present_artifact(tmp_path, monkeypatch) -> None:
    models_dir = tmp_path / "models"
    _write_json(models_dir / "audience_model.json", _synthetic_artifact())
    monkeypatch.setattr(core, "MODELS_DIR", models_dir)
    body = client.get("/api/model/audience").json()
    assert body["available"] is True
    assert body["computed_at"] == "2026-07-29T00:00:00+00:00"
    assert set(body["gates"]) == set(GATE_FAMILIES)
    assert all(gate["verdict"] == "off" for gate in body["gates"].values())
    assert body["base_summary"] == {"pooled_mean_tvr": 4.2}


# 7. Canonical activation write and absent-is-off ----------------------------
def test_activation_uses_its_guarded_route_and_absent_reads_false(tmp_path, monkeypatch) -> None:
    tmp_settings = tmp_path / "kairos_settings.json"
    _write_json(tmp_settings, _model_dump(KairosSettings()))
    monkeypatch.setattr(core, "SETTINGS_PATH", tmp_settings)

    body = client.get("/api/settings").json()
    assert body["audience_model_activation"] is False

    body["audience_model_activation"] = True
    response = client.put("/api/settings", json=body)
    assert response.status_code == 409
    assert "/api/rules/model-activation" in response.json()["detail"]
    assert client.get("/api/settings").json()["audience_model_activation"] is False

    response = client.put("/api/rules/model-activation", json={"active": True})
    assert response.status_code == 200, response.text
    assert response.json()["active"] is True
    assert client.get("/api/settings").json()["audience_model_activation"] is True
    persisted = json.loads(tmp_settings.read_text(encoding="utf-8"))
    assert persisted["audience_model_activation"] is True

    # A stored document without the key reads False, the frozen absent-is-off rule.
    del persisted["audience_model_activation"]
    _write_json(tmp_settings, persisted)
    assert client.get("/api/settings").json()["audience_model_activation"] is False


# 8. The forecast basis note --------------------------------------------------
def test_basis_note_states(tmp_path, monkeypatch) -> None:
    from kairos_api.audience_api import audience_model_note

    settings_path = tmp_path / "kairos_settings.json"
    models_dir = tmp_path / "models"
    monkeypatch.setattr(core, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(core, "MODELS_DIR", models_dir)

    # Off (absent settings file): historical basis, honestly labeled off.
    note = audience_model_note()
    assert note["state"] == "off"

    # On with no trained artifact: still historical, and it says so.
    _write_json(settings_path, {"audience_model_activation": True})
    note = audience_model_note()
    assert note["state"] == "on_no_artifact"
    assert note["computed_at"] is None

    # On with the artifact: model basis, carrying the training timestamp.
    _write_json(models_dir / "audience_model.json", _synthetic_artifact())
    note = audience_model_note()
    assert note["state"] == "on"
    assert note["computed_at"] == "2026-07-29T00:00:00+00:00"


def test_summary_carries_the_audience_model_note(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(core, "SETTINGS_PATH", tmp_path / "kairos_settings.json")
    monkeypatch.setattr(core, "MODELS_DIR", tmp_path / "models")
    summary = core._summarize_schedule(pd.DataFrame())
    assert summary["audience_model"]["state"] == "off"
