"""The preview-input extraction moved a function and changed no behaviour.

``_preview_inputs`` was a module-private helper in :mod:`kairos_api.overrides`
that :mod:`kairos_api.constraints` imported twice. It is now
:func:`kairos_api.preview_inputs.preview_inputs`, one module with one owner.

The bar is behaviour identity, so these tests rebuild the pre-extraction body
inline and require the extracted function to agree with it segment by segment
and kwarg by kwarg, exercise both of the call sites that changed, and hold the
optional read cache to the same standard in both of its states.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.constraints as constraints_api
import kairos_api.overrides as overrides_api
from kairos_api import preview_inputs as preview_module
from kairos_api import read_cache
from kairos_api.preview_inputs import preview_inputs

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def clean_cache():
    read_cache.invalidate(preview_module.CACHE_NAMESPACE)
    read_cache.reset_stats(preview_module.CACHE_NAMESPACE)
    yield
    read_cache.invalidate(preview_module.CACHE_NAMESPACE)
    read_cache.reset_stats(preview_module.CACHE_NAMESPACE)


@pytest.fixture(scope="module")
def channel_day():
    """One real channel-day, chosen the way the preview's own callers choose it."""
    from kairos.data.loaders import load_programmes
    from kairos_api.core import _load_settings

    try:
        programmes = load_programmes()
    except Exception as exc:  # pragma: no cover - environment without reference data
        pytest.skip(f"no programmes reference data: {exc}")
    valid = programmes[programmes["start_dt"].notna()]
    if valid.empty:  # pragma: no cover - environment without reference data
        pytest.skip("programmes reference has no parseable rows")
    channel = str(_load_settings().operator_channel or "").strip()
    mine = valid[valid["Channel"].astype(str) == channel]
    if channel == "" or mine.empty:
        channel = str(valid["Channel"].iloc[0])
        mine = valid[valid["Channel"].astype(str) == channel]
    day = mine["start_dt"].dt.strftime("%Y-%m-%d").min()
    return channel, day


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(overrides_api.router)
    app.include_router(constraints_api.router)
    return TestClient(app)


def original_preview_inputs(channel, day, daily_input):
    """The pre-extraction body, verbatim, as the identity reference.

    This is the function as it stood at ``kairos_api/overrides.py:244-302``
    before the move. If the extracted module ever drifts from it, the drift shows
    up here as a changed segment or a changed engine kwarg rather than as a
    changed money number in production.
    """
    from kairos.data.loaders import load_daily_input, load_programmes
    from kairos.data.transform import (
        build_segments_from_daily_input,
        build_segments_from_programmes,
    )
    from kairos.model.impact import load_impact_model
    from kairos.optimize.pricing import OptimizerAssumptions
    from kairos.service import (
        _apply_first_break_multiplier,
        _build_classifier,
        _pacing_knobs_from_settings,
        guardrails_from_settings,
        pricing_from_settings,
    )
    from kairos_api.core import _load_settings, _model_dump, _reference_today

    saved = _load_settings()
    settings_map = _model_dump(saved)
    pricing = pricing_from_settings(settings_map)
    assumptions = OptimizerAssumptions()
    impact = load_impact_model(ROOT / "models" / "tv_break_posterior.pkl", assumptions=assumptions)
    assumptions = _apply_first_break_multiplier(assumptions)
    classifier = _build_classifier()
    if daily_input:
        daily = load_daily_input(daily_input)
        segments = build_segments_from_daily_input(
            daily, classifier, pricing, assumptions=assumptions, impact_model=impact,
        )
    else:
        programmes = load_programmes()
        segments = build_segments_from_programmes(
            programmes, classifier, pricing,
            assumptions=assumptions, impact_model=impact, channel=channel, day=day,
        )
    engine_kwargs = {
        "guardrails": guardrails_from_settings(settings_map),
        "revenue_weight": saved.revenue_weight / 100.0,
        "risk_lambda": saved.risk_lambda,
        "objective_mode": getattr(saved, "objective_mode", "blend"),
        "pacing_today": _reference_today(saved),
        "pacing_knobs": _pacing_knobs_from_settings(settings_map),
        "operator_channel": str(saved.operator_channel or ""),
        "pricing": pricing,
    }
    return segments, engine_kwargs


def test_the_moved_name_is_the_one_every_dependent_imports() -> None:
    assert not hasattr(overrides_api, "_preview_inputs")
    assert preview_inputs.__module__ == "kairos_api.preview_inputs"


def test_extracted_matches_the_pre_extraction_body_exactly(channel_day) -> None:
    channel, day = channel_day
    expected_segments, expected_kwargs = original_preview_inputs(channel, day, None)
    segments, engine_kwargs = preview_inputs(channel, day, None)
    assert len(segments) == len(expected_segments)
    assert segments == expected_segments
    assert set(engine_kwargs) == set(expected_kwargs)
    for key in sorted(expected_kwargs):
        assert engine_kwargs[key] == expected_kwargs[key], key


def test_the_constraints_back_compat_seam_returns_the_same_segments(channel_day) -> None:
    channel, day = channel_day
    assert constraints_api._build_segments(channel, day, None) == preview_inputs(channel, day, None)[0]


def test_each_call_returns_its_own_list(channel_day) -> None:
    channel, day = channel_day
    first, _ = preview_inputs(channel, day, None)
    count = len(first)
    first.clear()
    second, _ = preview_inputs(channel, day, None)
    assert len(second) == count


def test_an_unknown_channel_day_still_builds_nothing(channel_day) -> None:
    channel, _ = channel_day
    segments, _ = preview_inputs(channel, "1999-01-01", None)
    assert segments == []


def test_cache_on_returns_the_same_segments_as_cache_off(channel_day, monkeypatch) -> None:
    channel, day = channel_day
    monkeypatch.setattr(preview_module, "USE_READ_CACHE", False)
    uncached, uncached_kwargs = preview_inputs(channel, day, None)
    monkeypatch.setattr(preview_module, "USE_READ_CACHE", True)
    first, first_kwargs = preview_inputs(channel, day, None)
    second, second_kwargs = preview_inputs(channel, day, None)
    assert first == uncached
    assert second == uncached
    assert read_cache.stats(preview_module.CACHE_NAMESPACE)["hits"] == 1
    for kwargs in (first_kwargs, second_kwargs):
        for key in sorted(uncached_kwargs):
            assert kwargs[key] == uncached_kwargs[key], key


def test_cache_on_never_serves_a_changed_first_break_multiplier(channel_day, monkeypatch) -> None:
    """The measured multiplier is not a file signature and not a seam identity.

    It reaches the segments through the folded assumptions, which is why the
    fingerprint carries the folded value. Warm the cache, change the multiplier,
    and the segments must change with it, exactly as the commit path's would.
    """
    import kairos.service as service

    channel, day = channel_day
    monkeypatch.setattr(preview_module, "USE_READ_CACHE", True)
    before, _ = preview_inputs(channel, day, None)
    assert before, "expected real segments for the fixture channel-day"

    original = service.read_coefficients_metadata

    def folded(path):
        metadata = dict(original(path) or {})
        metadata["first_break_multiplier"] = 3.0
        return metadata

    monkeypatch.setattr(service, "read_coefficients_metadata", folded)
    after, _ = preview_inputs(channel, day, None)
    assert [s.first_break_multiplier for s in after] != [s.first_break_multiplier for s in before]
    assert all(s.first_break_multiplier == 3.0 for s in after)


def test_cache_on_never_serves_segments_built_by_a_replaced_seam(channel_day, monkeypatch) -> None:
    import kairos.data.transform as transform

    channel, day = channel_day
    monkeypatch.setattr(preview_module, "USE_READ_CACHE", True)
    preview_inputs(channel, day, None)
    monkeypatch.setattr(transform, "build_segments_from_programmes", lambda *a, **k: [])
    segments, _ = preview_inputs(channel, day, None)
    assert segments == []


def test_cache_on_never_serves_across_a_settings_change(channel_day, monkeypatch) -> None:
    import kairos_api.core as core

    channel, day = channel_day
    monkeypatch.setattr(preview_module, "USE_READ_CACHE", True)
    _, before_kwargs = preview_inputs(channel, day, None)
    saved = core._load_settings()
    moved = saved.model_copy(update={"revenue_weight": 25})

    monkeypatch.setattr(core, "_load_settings", lambda: moved)
    _, after_kwargs = preview_inputs(channel, day, None)
    assert before_kwargs["revenue_weight"] != after_kwargs["revenue_weight"]
    assert after_kwargs["revenue_weight"] == 0.25
    assert read_cache.stats(preview_module.CACHE_NAMESPACE)["misses"] == 2


def test_both_effect_endpoints_are_identical_with_the_cache_on(client, channel_day, monkeypatch) -> None:
    """The Bar 3 proof at the endpoint: same bytes, cache off and cache on."""
    channel, day = channel_day
    params = {"channel": channel, "day": day}

    monkeypatch.setattr(preview_module, "USE_READ_CACHE", False)
    off = {
        path: client.get(path, params=params).content
        for path in ("/api/constraints/effect", "/api/overrides/effect")
    }
    monkeypatch.setattr(preview_module, "USE_READ_CACHE", True)
    on = {
        path: client.get(path, params=params).content
        for path in ("/api/constraints/effect", "/api/overrides/effect")
    }
    assert on == off
    body = json.loads(off["/api/constraints/effect"])
    assert body["summary"]["before_total_breaks"] == body["summary"]["after_total_breaks"]


@pytest.mark.parametrize("path,extra", [
    ("/api/constraints/effect", {}),
    ("/api/overrides/effect", {}),
    ("/api/overrides/effect", {"kind": "forbid"}),
])
def test_the_routes_return_what_the_pre_extraction_body_returned(
    client, channel_day, monkeypatch, path, extra,
) -> None:
    """The bar, at the route boundary: byte-identical bodies before and after.

    The "before" leg is the pre-extraction function itself, patched in where the
    routes now call the extracted one, so this compares the two versions of the
    code rather than two runs of the same version. The third case drives the
    candidate branch, which is the only one that runs a different pair of legs.
    """
    channel, day = channel_day
    params = {"channel": channel, "day": day}
    if extra:
        params = {"target_id": f"{day}|{channel}|3", **extra}

    after = client.get(path, params=params)
    calls: list[int] = []

    def replica(channel_arg, day_arg, daily_input_arg):
        calls.append(1)
        return original_preview_inputs(channel_arg, day_arg, daily_input_arg)

    monkeypatch.setattr(preview_module, "preview_inputs", replica)
    before = client.get(path, params=params)
    # Without this the comparison could pass by running the same code twice.
    assert calls, "the pre-extraction replica was not the function the route called"
    assert before.status_code == after.status_code == 200
    assert before.content == after.content


def test_the_cache_ships_off(channel_day) -> None:
    """The measured default, asserted so a flip is a deliberate act with a number.

    Turning it on buys 11 ms of an 1,871 ms response, measured in
    docs/ux-gauntlet/contracts/W0-5.md. A future piece may flip it after
    measuring its own path; nobody should flip it by accident.
    """
    assert preview_module.USE_READ_CACHE is False
    channel, day = channel_day
    preview_inputs(channel, day, None)
    assert read_cache.stats(preview_module.CACHE_NAMESPACE)["entries"] == 0
