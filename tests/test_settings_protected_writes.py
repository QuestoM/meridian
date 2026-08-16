"""Protected settings can move only through the route that owns their safety."""

from __future__ import annotations

import json
import threading

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.core as core
import kairos_api.settings_api as settings_api
from kairos_api.settings_api import router


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    path = tmp_path / "kairos_settings.json"
    path.write_text(json.dumps(core._model_dump(core.KairosSettings())), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_PATH", path)
    monkeypatch.setenv("KAIROS_VERSIONS_DIR", str(tmp_path / "versions"))
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.mark.parametrize(
    ("field", "value", "route"),
    [
        ("max_breaks_per_hour", 3, "/api/rules/guardrails"),
        ("audience_model_activation", True, "/api/rules/model-activation"),
        ("pricing_overrides", {"premiums": {"show": {"news": 1.2}}}, "/api/pricing"),
    ],
)
def test_generic_settings_put_rejects_protected_moves_without_writing(
    client: TestClient, field: str, value: object, route: str,
) -> None:
    before = client.get("/api/settings").json()
    candidate = {**before, field: value}

    response = client.put("/api/settings", json=candidate)

    assert response.status_code == 409
    assert route in response.json()["detail"]
    assert client.get("/api/settings").json() == before


def test_ordinary_settings_put_may_carry_unchanged_protected_values(client: TestClient) -> None:
    before = client.get("/api/settings").json()
    response = client.put("/api/settings", json={**before, "revenue_weight": 65})

    assert response.status_code == 200, response.text
    assert response.json()["revenue_weight"] == 65
    assert response.json()["max_breaks_per_hour"] == before["max_breaks_per_hour"]
    assert response.json()["audience_model_activation"] == before["audience_model_activation"]
    assert response.json()["pricing_overrides"] == before["pricing_overrides"]


def test_partial_generic_body_preserves_every_omitted_setting(client: TestClient) -> None:
    before = client.get("/api/settings").json()
    seeded = client.put(
        "/api/settings",
        json={**before, "locale": "en", "direction": "ltr", "objective_mode": "revenue_net"},
    )
    assert seeded.status_code == 200, seeded.text

    response = client.put("/api/settings", json={"revenue_weight": 65})

    assert response.status_code == 200, response.text
    assert response.json()["revenue_weight"] == 65
    assert response.json()["locale"] == "en"
    assert response.json()["direction"] == "ltr"
    assert response.json()["objective_mode"] == "revenue_net"


def test_empty_generic_body_is_a_true_noop(client: TestClient, monkeypatch) -> None:
    from kairos_api import version_store

    before = client.get("/api/settings").json()
    snapshots: list[str] = []
    monkeypatch.setattr(
        version_store,
        "snapshot_manual_edit",
        lambda _request, kind: snapshots.append(kind),
    )

    response = client.put("/api/settings", json={})

    assert response.status_code == 200, response.text
    assert response.json() == before
    assert snapshots == []


def test_generic_compare_and_save_cannot_race_a_canonical_protected_write(
    client: TestClient, monkeypatch,
) -> None:
    """A stale whole-document body cannot put a protected subtree back."""
    before = client.get("/api/settings").json()
    candidate = core.KairosSettings(**{**before, "revenue_weight": 65})
    compared = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []
    original_check = settings_api._require_canonical_protected_writes

    def paused_check(incoming, current=None):
        original_check(incoming, current)
        compared.set()
        assert release.wait(timeout=2)

    monkeypatch.setattr(settings_api, "_require_canonical_protected_writes", paused_check)

    def generic_write():
        try:
            settings_api.update_settings(candidate)
        except BaseException as exc:  # pragma: no cover - reported below
            failures.append(exc)

    def canonical_write():
        try:
            current = core._load_settings()
            current.audience_model_activation = True
            core._save_settings(current)
        except BaseException as exc:  # pragma: no cover - reported below
            failures.append(exc)

    generic = threading.Thread(target=generic_write)
    generic.start()
    assert compared.wait(timeout=2)
    canonical = threading.Thread(target=canonical_write)
    canonical.start()
    release.set()
    generic.join(timeout=2)
    canonical.join(timeout=2)

    assert not generic.is_alive() and not canonical.is_alive()
    assert failures == []
    final = client.get("/api/settings").json()
    assert final["revenue_weight"] == 65
    assert final["audience_model_activation"] is True


def test_two_canonical_writers_preserve_both_changes(client: TestClient, monkeypatch) -> None:
    from kairos_api import model_activation, pricing_api, version_store

    snapshot_started = threading.Event()
    release = threading.Event()
    failures: list[BaseException] = []
    original_snapshot = version_store.snapshot_manual_edit

    def paused_snapshot(*args, **kwargs):
        snapshot_started.set()
        assert release.wait(timeout=2)
        return original_snapshot(*args, **kwargs)

    monkeypatch.setattr(version_store, "snapshot_manual_edit", paused_snapshot)

    def price_write():
        try:
            pricing_api.put_pricing(pricing_api.PricingUpdate(overrides={
                "base_price_per_second_per_tvr_point": 123.0,
            }))
        except BaseException as exc:  # pragma: no cover - reported below
            failures.append(exc)

    def activation_write():
        try:
            model_activation.set_active(True)
        except BaseException as exc:  # pragma: no cover - reported below
            failures.append(exc)

    pricing = threading.Thread(target=price_write)
    pricing.start()
    assert snapshot_started.wait(timeout=2)
    activation = threading.Thread(target=activation_write)
    activation.start()
    release.set()
    pricing.join(timeout=2)
    activation.join(timeout=2)

    assert not pricing.is_alive() and not activation.is_alive()
    assert failures == []
    final = client.get("/api/settings").json()
    assert final["audience_model_activation"] is True
    assert final["pricing_overrides"]["base_price_per_second_per_tvr_point"] == 123.0
