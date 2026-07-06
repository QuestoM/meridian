"""QA 2026-07-06 regression guards: the three real bugs this wave found, fixed.

Each test reproduces a concrete defect (Law 12: a file+behavior locus and a
reproduced payload, never speculation), then asserts the fixed behavior. They
started as deliberately failing pins; the fixes landed centrally and these are
now standing regression guards.

Run just these:
    python -m pytest -q tests/test_qa_known_bugs_20260706.py -ra

All in-process (TestClient / direct calls); nothing writes the weekly CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos_api.server import app  # noqa: E402


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


# --------------------------------------------------------------------------- #
# BUG A (fixed): /api/compliance and /api/overview used to 500 on a fresh
# deploy. dashboard_api._build_compliance's fallback path (no committed CSV, so
# _plan_guardrail_items() is empty) compared summary["average_retention"] (None
# on an empty schedule) to the floor and raised TypeError. It now reports an
# honest "unknown" retention verdict instead of crashing.
# --------------------------------------------------------------------------- #
def test_compliance_fallback_survives_an_empty_schedule() -> None:
    from kairos_api import dashboard_api as D
    from kairos_api.core import KairosSettings

    original = D._plan_guardrail_items
    D._plan_guardrail_items = lambda: []  # simulate no committed CSV on disk
    try:
        result = D._build_compliance(pd.DataFrame(), KairosSettings())
    finally:
        D._plan_guardrail_items = original
    assert result["status"] in {"compliant", "at_risk", "unknown"}
    retention = next(c for c in result["checks"] if c["id"] == "retention_floor")
    # An unknown observed value is reported honestly, never asserted compliant.
    assert retention["observed"] is None or isinstance(retention["observed"], (int, float))
    if retention["observed"] is None:
        assert retention["status"] == "unknown"


# --------------------------------------------------------------------------- #
# BUG B (fixed): /api/inventory and /api/campaigns fabricated revenue = 0. The
# loaded spots source (data/reference/Spots.xlsx) carries no revenue_ils column,
# so the rollups reported a hard 0 as if measured. They now report revenue as
# None with revenue_available=false (honest unavailable), never a fabricated 0.
# --------------------------------------------------------------------------- #
def test_inventory_reports_honest_unavailable_not_a_fabricated_zero(client) -> None:
    body = client.get("/api/inventory").json()
    summary = body["summary"]
    assert summary["spots"] > 0, "there are real spots"
    assert "revenue_available" in body
    if not body["revenue_available"]:
        assert summary["revenue"] is None
        assert all(row.get("revenue") is None for row in body["by_channel"])
    else:
        assert summary["revenue"] is not None
    # The forbidden state: spots present but revenue a hard fabricated 0.
    assert not (summary["spots"] > 0 and summary["revenue"] == 0)


def test_campaigns_report_honest_unavailable_not_a_fabricated_zero(client) -> None:
    body = client.get("/api/campaigns").json()
    campaigns = body["campaigns"]
    assert campaigns, "there are real campaigns"
    assert "revenue_available" in body
    if not body["revenue_available"]:
        assert all(c.get("revenue") is None for c in campaigns)
    else:
        assert any((c.get("revenue") or 0) != 0 for c in campaigns)


# --------------------------------------------------------------------------- #
# BUG C (fixed): the constraint /effect preview optimized at the bare engine
# defaults (revenue_weight 0.5, default guardrails, blend, from_yaml pricing),
# so its absolute revenue misstated the plan the weekly recompute writes under
# the saved settings. The preview now threads the saved guardrails, revenue
# weight, risk aversion, objective mode, and pricing_from_settings, so its
# baseline equals the saved-settings optimization.
# --------------------------------------------------------------------------- #
@pytest.mark.realdata
def test_effect_preview_runs_under_the_saved_settings(client) -> None:
    from kairos.optimize.optimizer import optimize_breaks
    from kairos.service import guardrails_from_settings
    from kairos_api import constraints as constraints_module
    from kairos_api.core import _load_settings, _model_dump

    channel, day = "קשת 12", "2024-11-06"
    segments = constraints_module._build_segments(channel, day, None)
    assert segments, "reference data must build segments for this channel-day"

    saved = _load_settings()
    settings_map = _model_dump(saved)
    engine_kwargs = {
        "guardrails": guardrails_from_settings(settings_map),
        "revenue_weight": saved.revenue_weight / 100.0,
        "risk_lambda": saved.risk_lambda,
        "objective_mode": getattr(saved, "objective_mode", "blend"),
    }
    saved_plan = optimize_breaks(segments, **engine_kwargs)

    response = client.get(
        "/api/constraints/effect", params={"channel": channel, "day": day}
    )
    assert response.status_code == 200, response.text
    before = response.json()["summary"]["before_revenue"]
    # The preview baseline equals the saved-settings optimization to the cent.
    assert before == round(saved_plan.total_revenue, 2)
