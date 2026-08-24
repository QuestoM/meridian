"""The frontier's key and its computation must see ONE settings snapshot.

The defect this pins was transient, exact, and reproduced before it was fixed:
the two frontier lru_caches used to read the LIVE settings file inside their
bodies -- at whatever moment the background thread reached that line -- while
their cache key carried only floor/bph/risk/weight/mode plus the data-file
signature. A test that doubles the base CPP for a few milliseconds (and resets
it) could land its window exactly on the bundle's capture, and the bundle then
priced the whole day at exactly 2.000000x -- same plan, same breaks, double the
money -- cached forever under the unchanged key. One full sweep caught it once:
current.gross exactly twice the sweep's own anchor.

The fix pins the settings to the key: ``pinned_pacing_json()`` captures once at
request time, travels as a key component, and the cached bodies parse IT and
read nothing live. These tests hold that seam shut without threads or timing --
they poison the live store and prove the body cannot see the poison.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kairos_api import core
from kairos_api import plan_read_frontier as prf

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def tmp_settings(tmp_path, monkeypatch) -> Path:
    target = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", target)
    monkeypatch.setattr(core, "SETTINGS_PATH", target)
    return target


def _double_base_cpp(settings_path: Path) -> None:
    """The polluter's exact write shape: a FLAT pricing_overrides key, the same
    one PUT /api/pricing persists (test_journey_inspector_pricing asserts
    ``persisted["pricing_overrides"]["base_price_per_second_per_tvr_point"]``).
    Nesting it under a "base" sub-key is silently a no-op nobody reads: the
    doubled pin then prices exactly 1.0x and the test passes while testing
    nothing, so the shape here must stay FLAT."""
    saved = json.loads(settings_path.read_text(encoding="utf-8"))
    overrides = dict(saved.get("pricing_overrides") or {})
    base = float(overrides.get("base_price_per_second_per_tvr_point") or 55.2)
    overrides["base_price_per_second_per_tvr_point"] = base * 2
    saved["pricing_overrides"] = overrides
    settings_path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")


def test_the_cached_body_cannot_see_a_live_settings_write(tmp_settings):
    """THE 2x, HELD SHUT. Capture the pinned settings clean, poison the live
    store with the exact 2x write, compute the bundle with the CLEAN pin -- the
    money must be the clean money. Before the fix this fails at exactly 2x,
    because the body read the live store at compute time."""
    if not (ROOT / "output" / "weekly_break_schedule.csv").exists():
        pytest.skip("no plan of record on this machine")

    settings = core._load_settings()
    signature = prf.frontier_data_signature()
    owned = str(settings.operator_channel or "").strip()
    if not owned:
        pytest.skip("no operator channel configured")
    day = prf.owned_representative_day(signature, owned)
    if day is None:
        pytest.skip("no owned dated programmes")

    clean_pin = prf.pinned_pacing_json()
    args = (
        signature, owned, day,
        float(settings.min_retention_floor), int(settings.max_breaks_per_hour),
        float(settings.risk_lambda), int(settings.revenue_weight),
        str(getattr(settings, "objective_mode", "blend") or "blend"),
    )

    prf.frontier_net_bundle_cached.cache_clear()
    baseline = prf.frontier_net_bundle_cached(*args, clean_pin)
    assert baseline.get("comparison_available"), baseline
    clean_gross = baseline["current"]["gross"]

    # Poison the live store the way the pricing journey test does, then compute
    # again under the SAME clean pin on a cold cache. A live read anywhere in
    # the body doubles this figure.
    _double_base_cpp(tmp_settings)
    prf.frontier_net_bundle_cached.cache_clear()
    repriced = prf.frontier_net_bundle_cached(*args, clean_pin)
    assert repriced.get("comparison_available"), repriced
    assert abs(repriced["current"]["gross"] - clean_gross) < 0.5, (
        f"the cached body saw the live settings write: clean {clean_gross}, "
        f"poisoned-store recompute {repriced['current']['gross']}"
    )


def test_a_different_pinned_capture_is_a_different_cache_entry(tmp_settings):
    """The pin is part of the key, so a settings edit MISSES the cache instead
    of serving pre-edit money. Held via cache_info to keep it cheap: two pins,
    two entries."""
    if not (ROOT / "output" / "weekly_break_schedule.csv").exists():
        pytest.skip("no plan of record on this machine")

    settings = core._load_settings()
    signature = prf.frontier_data_signature()
    owned = str(settings.operator_channel or "").strip()
    if not owned:
        pytest.skip("no operator channel configured")
    day = prf.owned_representative_day(signature, owned)
    if day is None:
        pytest.skip("no owned dated programmes")

    args = (
        signature, owned, day,
        float(settings.min_retention_floor), int(settings.max_breaks_per_hour),
        float(settings.risk_lambda), int(settings.revenue_weight),
        str(getattr(settings, "objective_mode", "blend") or "blend"),
    )
    pin_before = prf.pinned_pacing_json()
    _double_base_cpp(tmp_settings)
    pin_after = prf.pinned_pacing_json()
    assert pin_before != pin_after, "the pin must reflect a pricing-override edit"

    prf.frontier_net_bundle_cached.cache_clear()
    first = prf.frontier_net_bundle_cached(*args, pin_before)
    second = prf.frontier_net_bundle_cached(*args, pin_after)
    info = prf.frontier_net_bundle_cached.cache_info()
    assert info.currsize == 2, info
    # And the doubled pin REPRICES -- it is live pricing, not an inert tag.
    # Deliberately not asserted at exactly 2x: revenue is linear in base CPP
    # only for a FIXED plan, and the pacing steer reads money, so doubling the
    # price can legitimately move the chosen plan (measured here: 1.84x). The
    # defect's own 2.000000x arose in the case where the plan happened to be
    # identical. What this seam guarantees is coherence, not linearity.
    if first.get("comparison_available") and second.get("comparison_available"):
        ratio = second["current"]["gross"] / max(first["current"]["gross"], 1e-9)
        assert ratio > 1.5, (
            f"a 2x base-CPP pin must visibly reprice the bundle; got {ratio}"
        )


def test_no_live_settings_read_remains_inside_the_cached_bodies():
    """Structural pin: the bodies parse the pinned capture; only the capture
    helper itself may touch _pacing_call_kwargs. A live read reintroduced into
    either body is the whole defect back."""
    import ast
    import inspect

    for func in (prf.frontier_points_cached, prf.frontier_net_bundle_cached):
        source = inspect.getsource(func.__wrapped__)
        calls = [node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                 for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Call)]
        assert "_pacing_call_kwargs" not in calls, (
            f"{func.__wrapped__.__name__} reads live settings inside the cached body"
        )
        assert "_pacing_from_pinned" in calls, (
            f"{func.__wrapped__.__name__} no longer parses the pinned capture"
        )
