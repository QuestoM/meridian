"""P4 and P11 seam: the empty make-good reason follows the reader's locale."""

from pathlib import Path

from fastapi.testclient import TestClient

from kairos_api.server import app


ROOT = Path(__file__).resolve().parents[1]
SURFACE = ROOT / "tv-break-dashboard/src/clients/MakeGoodAlerts.jsx"


def test_the_empty_projection_carries_one_reason_for_each_reader(monkeypatch) -> None:
    monkeypatch.setenv("KAIROS_AUTH_DISABLED", "1")
    body = TestClient(app).get("/api/make-good-alerts").json()

    assert body["data_available"] is False
    assert body["reason_en"] == body["reason"]
    assert any("א" <= character <= "ת" for character in body["reason_he"])


def test_the_surface_reads_the_reason_pair_in_its_locale() -> None:
    source = SURFACE.read_text(encoding="utf-8")

    assert "localized(payload, 'reason', locale)" in source
    assert "{payload.reason}" not in source
