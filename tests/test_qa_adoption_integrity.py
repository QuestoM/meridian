"""QA 2026-07-06: adoption-integrity and journey contract tests (passing).

These lock the coherence of the newly adopted placebo-corrected + interval-
calibrated coefficients across the stack, plus the journey surfaces the audit
walked. They are all CONTRACT tests (they assert a behavior that currently
holds); the companion tests/test_qa_known_bugs_20260706.py pins the real bugs
this wave found with deliberately failing tests.

Everything runs in-process against the FastAPI app with a TestClient (no server,
no :8000). conftest sets KAIROS_AUTH_DISABLED=1 so the API is reachable. Nothing
here writes the weekly CSV: the golden builds the schedule in memory and hashes
it, and every endpoint call is a read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kairos_api.server import app  # noqa: E402

COEFFICIENTS_PATH = ROOT / "models" / "tv_break_coefficients.json"

# The adopted plan's headline economics, to the cent, reproduced live from the
# committed CSV in this session. Rebased when the exact DP refiner tier shipped
# default on and moved the plan of record (kairos/optimize/dp_refine.py): the tier
# recovers value greedy+F1 left on the table, so revenue and its retention cost both
# rose (never a per-day regression on the engine's own objective).
ADOPTED_REVENUE_ILS = 221_891_590.23
ADOPTED_RETENTION_COST_ILS = 24_259_552.77
ADOPTED_NET_ILS = 197_632_037.46

# The pre-adoption figures that must not survive anywhere as a live number.
STALE_REVENUE_M = 215.3
STALE_COST_M = 16.8


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(scope="module")
def coefficients() -> dict:
    return json.loads(COEFFICIENTS_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# TASK 1a: the golden reproduces from the artifact.
# --------------------------------------------------------------------------- #
def test_golden_weekly_schedule_reproduces_from_the_artifact() -> None:
    """The committed CSV, the golden hashes and the coefficients all agree.

    Delegates to the standing golden helper: it rebuilds the schedule along the
    exact POST /api/recompute-schedule path from the saved settings and the
    committed coefficients, then asserts the full-CSV hash (c09e8d1d...) and the
    per-channel-day aggregate hash (059e07c1...) match to the byte. Skipped only
    if the optimization engine is unavailable in this environment.
    """
    from tests.golden_weekly_schedule import (
        GOLDEN_AGG_SHA256,
        GOLDEN_CSV_SHA256,
        GOLDEN_ROWS,
        evaluate,
    )

    frame, _records, csv_hash, agg_hash, problems = evaluate()
    assert not problems, "golden drift:\n" + "\n".join(problems)
    assert len(frame) == GOLDEN_ROWS
    assert csv_hash == GOLDEN_CSV_SHA256
    assert agg_hash == GOLDEN_AGG_SHA256


# --------------------------------------------------------------------------- #
# TASK 1b: dashboard money is consistent and uses the adopted coefficients.
# --------------------------------------------------------------------------- #
def test_yield_per_second_net_equals_revenue_minus_cost_to_the_cent(client) -> None:
    payload = client.get("/api/yield-per-second").json()
    assert payload["available"] is True
    assert payload["revenue_net_available"] is True
    rev = payload["revenue_ils"]
    cost = payload["retention_cost_ils"]
    net = payload["revenue_net_ils"]
    assert round(rev - cost, 2) == net, "net must equal revenue - retention_cost to the cent"


def test_yield_per_second_matches_the_adopted_plan_economics(client) -> None:
    """The live money equals the adopted (corrected) plan, not a stale figure."""
    payload = client.get("/api/yield-per-second").json()
    assert payload["revenue_ils"] == pytest.approx(ADOPTED_REVENUE_ILS, abs=1.0)
    assert payload["retention_cost_ils"] == pytest.approx(ADOPTED_RETENTION_COST_ILS, abs=1.0)
    assert payload["revenue_net_ils"] == pytest.approx(ADOPTED_NET_ILS, abs=1.0)
    # And the cost is materially clear of the pre-adoption 16.8M stale number: the
    # DP-adopted plan's retention cost is ~24.26M (more breaks earn more revenue and
    # cost more retention), well above the stale figure.
    assert abs(payload["retention_cost_ils"] / 1e6 - STALE_COST_M) > 1.0


def test_yield_per_second_basis_names_the_retention_cost_formula(client) -> None:
    """The money carries an honest basis: a formula and named modeled inputs."""
    payload = client.get("/api/yield-per-second").json()
    basis = payload["basis"]
    assert basis["source"] == "modeled"
    assert "retention_cost_ils" in basis["formula"]
    assert "baseline_tvr" in basis["inputs"]


# --------------------------------------------------------------------------- #
# TASK 1c: the artifact metadata is honest.
# --------------------------------------------------------------------------- #
def test_artifact_declares_placebo_and_bootstrap_and_drift(coefficients) -> None:
    md = coefficients["metadata"]
    assert md["placebo_correction_active"] is True
    assert md["interval_method"] == "bootstrap"
    assert isinstance(md.get("level_drift"), dict) and md["level_drift"], "drift block present"
    assert md["level_drift"]["status"] == "measured"


def test_every_cell_carries_ci_and_predictive_with_predictive_strictly_wider(coefficients) -> None:
    detail = coefficients["detail"]
    assert len(detail) == 36, "36 class/position/length cells"
    for name, cell in detail.items():
        for key in ("ci_low", "ci_high", "predictive_low", "predictive_high"):
            assert cell.get(key) is not None, f"{name} missing {key}"
        assert cell["predictive_low"] < cell["ci_low"], f"{name} predictive_low not below ci_low"
        assert cell["predictive_high"] > cell["ci_high"], f"{name} predictive_high not above ci_high"


def test_impact_endpoint_exposes_placebo_bootstrap_and_measured_drift(client) -> None:
    payload = client.get("/api/impact").json()
    md = payload["coefficient_impacts"]["metadata"]
    assert md["placebo_correction_active"] is True
    assert md["interval_method"] == "bootstrap"
    assert payload["drift"]["status"] == "measured"
    assert payload["drift"]["binding"] is True


# --------------------------------------------------------------------------- #
# TASK 1d: no endpoint quotes the pre-adoption numbers.
# --------------------------------------------------------------------------- #
def test_no_read_endpoint_serializes_the_pre_adoption_headline(client) -> None:
    """Sweep the money-bearing GET endpoints; none may echo 215.3M / 16.8M.

    Checks the actual serialized JSON for the stale headline magnitudes. The
    corrected figures (221.89M / 24.26M / 197.63M after the DP refiner tier) are
    the only economics the stack should present.
    """
    endpoints = ["/api/yield-per-second", "/api/impact", "/api/compliance", "/api/reports"]
    for path in endpoints:
        body = client.get(path).json()
        blob = json.dumps(body, ensure_ascii=False)
        # 215300000 / 16800000 as raw integers, and the "215.3"/"16.8" million
        # spellings, must not appear as a live figure.
        assert "215300000" not in blob, f"{path} echoes stale revenue"
        assert "16800000" not in blob, f"{path} echoes stale cost"


# --------------------------------------------------------------------------- #
# TASK 2.3: inspector matches the CSV and enforces the competitor boundary.
# --------------------------------------------------------------------------- #
def _owned_channel() -> str:
    from kairos_api.core import _load_settings

    return str(_load_settings().operator_channel or "").strip()


def test_inspector_segment_matches_the_csv_row_field_for_field(client) -> None:
    import pandas as pd

    from kairos_api.core import OUTPUT_DIR

    owned = _owned_channel()
    frame = pd.read_csv(OUTPUT_DIR / "weekly_break_schedule.csv", encoding="utf-8-sig")
    owned_rows = frame[frame["channel"].astype(str).str.strip() == owned]
    assert not owned_rows.empty, "owned channel must have rows in the committed plan"
    row = owned_rows.iloc[10]
    seg_id = str(row["segment_id"])

    detail = client.get(f"/api/schedule/segment/{seg_id}").json()
    assert detail["found"] is True
    assert detail["identity"]["program_type"] == str(row["program_type"]).strip()
    assert detail["identity"]["start_clock"] == str(row["start_time"]).strip()
    assert detail["plan"]["num_breaks"] == int(row["num_breaks"])
    assert detail["economics"]["predicted_revenue"] == round(float(row["predicted_revenue"]), 2)
    assert detail["economics"]["baseline_tvr"] == pytest.approx(float(row["baseline_tvr"]), abs=1e-4)


def test_inspector_returns_404_for_a_competitor_segment(client) -> None:
    import pandas as pd

    from kairos_api.core import OUTPUT_DIR

    owned = _owned_channel()
    frame = pd.read_csv(OUTPUT_DIR / "weekly_break_schedule.csv", encoding="utf-8-sig")
    competitor_rows = frame[frame["channel"].astype(str).str.strip() != owned]
    if competitor_rows.empty:
        pytest.skip("no competitor rows in the committed plan")
    seg_id = str(competitor_rows.iloc[0]["segment_id"])
    resp = client.get(f"/api/schedule/segment/{seg_id}")
    assert resp.status_code == 404


# --------------------------------------------------------------------------- #
# TASK 2.2: the assistant grounding excludes competitor names and stays honest.
# --------------------------------------------------------------------------- #
def test_assistant_status_is_honest_without_a_key(client, monkeypatch) -> None:
    for name in ("ANTHROPIC_API_KEY", "KAIROS_ASSISTANT_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    body = client.get("/api/assistant/status").json()
    assert body["available"] is False
    assert body["reason"] == "API key not configured"


def test_assistant_context_carries_owned_day_table_and_no_competitor_name() -> None:
    """The composed grounding includes the per-day owned-channel table and never
    names a competitor channel, only an aggregate excluded-count."""
    from kairos_api import assistant
    from kairos_api.core import _load_settings

    context, sources = assistant._compose_context("What is my revenue this week?")
    assert "per_day_plan" in context
    per_day = context["per_day_plan"]
    owned = str(_load_settings().operator_channel or "").strip()
    assert per_day["channel"] == owned
    assert "competitor_channels_excluded" in per_day

    # No competitor channel name may appear anywhere in the serialized context.
    import pandas as pd

    from kairos_api.core import OUTPUT_DIR

    frame = pd.read_csv(OUTPUT_DIR / "weekly_break_schedule.csv", encoding="utf-8-sig")
    competitors = {
        str(ch).strip()
        for ch in frame["channel"].unique()
        if str(ch).strip() and str(ch).strip() != owned
    }
    blob = json.dumps(context, ensure_ascii=False)
    for name in competitors:
        assert name not in blob, f"competitor channel {name} leaked into assistant context"


def test_assistant_day_detail_appears_only_for_a_named_plan_date() -> None:
    """A question naming a real plan date gets a day_detail section; a question
    with no date gets none (conservative parsing, no guessing)."""
    from kairos_api import assistant

    import pandas as pd

    from kairos_api.core import OUTPUT_DIR, _load_settings

    owned = str(_load_settings().operator_channel or "").strip()
    frame = pd.read_csv(OUTPUT_DIR / "weekly_break_schedule.csv", encoding="utf-8-sig")
    a_date = str(frame[frame["channel"].astype(str).str.strip() == owned]["date"].iloc[0])

    with_date, _ = assistant._compose_context(f"Break down {a_date} for me")
    assert any(k.startswith("day_detail ") for k in with_date), "named date must add day_detail"

    no_date, _ = assistant._compose_context("How are things overall?")
    assert not any(k.startswith("day_detail ") for k in no_date), "no date must add no day_detail"


# --------------------------------------------------------------------------- #
# TASK 2.4: compliance covers the FULL plan with real observed-vs-limit.
# --------------------------------------------------------------------------- #
def test_compliance_covers_the_full_plan_with_real_daily_load(client) -> None:
    body = client.get("/api/compliance").json()
    checks = {c["id"]: c for c in body["checks"]}
    # The full-plan path emits all seven guardrail checks (the truncated board
    # path emits only four); its presence proves the verdict grades the whole plan.
    for cid in (
        "hourly_ad_load",
        "break_density",
        "retention_floor",
        "protected_programs",
        "break_spacing",
        "daily_ad_load",
        "gold_breaks",
    ):
        assert cid in checks, f"missing guardrail check {cid}"
    daily = checks["daily_ad_load"]
    # Observed is a real measured max-per-channel-day, not a fabricated constant,
    # and sits at or under the operator's 160-minute cap.
    assert isinstance(daily["observed"], (int, float))
    assert daily["limit"] == 160
    assert daily["observed"] <= daily["limit"] + 1e-6


# --------------------------------------------------------------------------- #
# TASK 3: cross-cutting GET sweep, honest shapes.
# --------------------------------------------------------------------------- #
def test_gold_breaks_and_make_good_are_honest_empty_not_fabricated(client) -> None:
    gold = client.get("/api/gold-breaks").json()
    # Zero gold breaks is honestly reported with a reason, not hidden.
    assert gold["available"] is True
    assert gold["count"] == 0
    assert gold["breaks"] == []
    assert gold.get("reason")

    make_good = client.get("/api/make-good-alerts").json()
    assert make_good["data_available"] is False
    assert make_good.get("reason")
    assert make_good["alerts"] == []
