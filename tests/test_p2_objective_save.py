"""P2: saving the objective changes the objective and nothing else.

The settings endpoint takes the whole model and defaults every field a body
omits, so a partial write silently clears fields the writer never mentioned.
Measured on the running app while building this piece: a body carrying one
lever cleared ``operator_channel`` and ``pricing_overrides``, and an empty
operator channel restates every money figure in the product from the operator's
own plan to the whole four-channel market, which is a competitor-boundary
breach as well as a wrong number.

These tests pin the two halves of the defence this piece owns. The surface
sends the full saved model with its four levers on top, and it refuses to
report success on a save that moved the channel scope. The endpoint's own
behaviour is not this piece's to change; it is reported to the piece that owns
the settings path.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "tv-break-dashboard" / "src" / "plan" / "week" / "use-plan-surface.js"
SETTINGS = ROOT / "data" / "kairos_settings.json"

OBJECTIVE_LEVERS = (
    "revenue_weight",
    "min_retention_floor",
    "max_breaks_per_hour",
    "risk_lambda",
    "objective_mode",
)


@pytest.fixture()
def client():
    from kairos_api.server import app

    return TestClient(app)


@pytest.fixture()
def settings_restored(tmp_path):
    """Every write in this file is undone, byte for byte, before the test ends."""
    backup = tmp_path / "kairos_settings.json"
    shutil.copy2(SETTINGS, backup)
    yield
    shutil.copy2(backup, SETTINGS)


def test_the_surface_sends_the_whole_saved_model_not_only_the_levers():
    text = SURFACE.read_text(encoding="utf-8")
    assert "const payload = { ...saved, ...draft };" in text
    assert "api.saveSettings(payload)" in text


def test_the_surface_refuses_a_save_that_moved_the_channel_scope():
    text = SURFACE.read_text(encoding="utf-8")
    assert "result.data?.operator_channel" in text
    assert "saved.operator_channel" in text
    # A refusal is a refusal: the state is error and the settings are re-read.
    body = text.split("operator_channel || '')")[2]
    assert "setSaveState('error')" in body
    assert "loadSettings()" in body


def test_the_full_model_round_trip_keeps_every_field_the_objective_does_not_own(
    client, settings_restored
):
    before = client.get("/api/settings").json()
    draft = {
        "revenue_weight": 85 if before["revenue_weight"] != 85 else 60,
        "min_retention_floor": 0.70,
        "max_breaks_per_hour": before["max_breaks_per_hour"],
        "risk_lambda": before["risk_lambda"],
        "objective_mode": before["objective_mode"],
    }
    response = client.put("/api/settings", json={**before, **draft})
    assert response.status_code == 200
    after = response.json()

    moved = {key for key in set(before) | set(after) if before.get(key) != after.get(key)}
    assert moved <= set(OBJECTIVE_LEVERS), f"the objective save moved {sorted(moved - set(OBJECTIVE_LEVERS))}"
    assert after["operator_channel"] == before["operator_channel"]
    assert after["pricing_overrides"] == before["pricing_overrides"]
    assert after["locale"] == before["locale"]


def test_a_partial_write_can_never_clear_the_declared_channel(client, settings_restored):
    """The defect this pinned is closed, and the invariant is what survives.

    Measured on this tree when the pin was written: a body carrying one lever
    wiped ``operator_channel``, and an empty channel un-scopes every money figure
    in the product from the operator's plan to the whole market. The owning piece
    has since closed it at the route, which now refuses the write with 400 and
    names the two ways to send it. Either answer is correct as long as the
    declared channel is still standing afterwards, and that is what this asserts,
    because that is the only part the objective save depends on.
    """
    before = client.get("/api/settings").json()
    assert before["operator_channel"], "the reference tree configures an operator channel"
    response = client.put("/api/settings", json={"revenue_weight": before["revenue_weight"]})
    if response.status_code == 200:
        assert response.json()["operator_channel"] == before["operator_channel"]
    else:
        assert response.status_code == 400
        assert "operator_channel" in response.json()["detail"]
    stored = json.loads(SETTINGS.read_text(encoding="utf-8"))
    assert stored["operator_channel"] == before["operator_channel"]
