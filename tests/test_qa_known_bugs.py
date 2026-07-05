"""KNOWN BUGS, pinned as failing tests (QA journeys audit, 2026-07-05).

Every test in this file asserts a PROMISE the product makes (compliance page
reports the plan, displayed data is never fabricated, settings gate the
engine, previews honor saved policy) and currently FAILS because the promise
is broken. Per the QA mandate the failing tests stay in place, clearly
commented, until the lead decides each fix; when a fix lands its test flips
green and becomes the standing regression gate for that promise.

Full evidence and the ranked bug list: docs/qa-journeys-2026-07-05.md.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import kairos_api.core as core

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"
SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"


@pytest.fixture()
def client() -> TestClient:
    from kairos_api.server import app

    return TestClient(app, raise_server_exceptions=False)


def test_bug1_compliance_verdict_must_cover_the_whole_committed_plan(client):
    """KNOWN BUG (HIGH). /api/compliance claims a per-rule verdict for the
    plan, but it evaluates 30-40 SYNTHESIZED breaks from only the first 12
    programmes per channel of the earliest day, not the committed plan.

    Chain: /api/compliance (kairos_api/server.py:1847) -> _build_compliance
    (server.py:1261) -> _build_break_operations, which truncates the EPG with
    .head(12) per channel (server.py:469) and re-synthesizes break times and
    counts (server.py:500 caps counts at min(5, capacity)). The committed plan
    spans 30 days and hits the daily cap exactly (9600s on several
    channel-days), yet the endpoint reports a max observed daily load of about
    24 minutes, so days 2 to 30 are simply not checked: a violation introduced
    there would still read "compliant".

    Promise asserted: the compliance page's observed daily load equals the
    committed plan's real maximum. Suggested fix: evaluate the guardrails on
    breaks reconstructed from the full weekly CSV (the conformance suite in
    tests/test_guardrail_conformance.py shows the reconstruction), or on the
    optimizer's own persisted placements, instead of the truncated display
    board."""
    plan = pd.read_csv(CSV_PATH, encoding="utf-8")
    true_max_daily_minutes = (
        plan.groupby(["channel", "date"])["total_break_time"].sum().max() / 60.0
    )
    body = client.get("/api/compliance").json()
    daily = next(c for c in body["checks"] if c["id"] == "daily_ad_load")
    assert daily["observed"] == pytest.approx(true_max_daily_minutes, abs=1.0), (
        f"compliance reports max daily load {daily['observed']} min but the committed "
        f"plan's true maximum is {true_max_daily_minutes:.1f} min: the verdict does not "
        "cover the plan it stands next to"
    )


def test_bug2_break_operations_must_not_fabricate_gold_flags():
    """KNOWN BUG (HIGH). The break-operations board synthesizes is_gold from
    settings heuristics (prime hour, first break in the programme:
    kairos_api/server.py:549-555) instead of reading the plan's own is_gold.
    The committed plan carries ZERO gold breaks (output CSV, is_gold all
    False), yet on any EPG slice that reaches prime time the board marks
    first-of-programme prime breaks gold, up to 4 on one channel-day, above
    the gold cap of 3. The compliance builder then counts those fabricated
    golds against the gold_breaks guardrail (server.py:1162-1165). Today the
    .head(12) truncation (BUG 1) hides this by only ever showing early-morning
    programmes; the two bugs cancel until an EPG starts mid-day.

    Promise asserted: a break the board displays claims gold only when the
    committed plan marks that segment gold. Suggested fix: source is_gold from
    the joined plan row (already available as schedule_row) and delete the
    prime-time synthesis."""
    from kairos_api.server import _build_break_operations
    from kairos_api.core import _load_break_schedule, _load_programmes

    plan = _load_break_schedule()
    plan_gold_ids = set(plan[plan["is_gold"] == True]["segment_id"])  # noqa: E712
    programmes = _load_programmes()
    programmes = programmes[programmes["start_dt"].notna()]
    prime_slice = programmes[
        (programmes["start_dt"].dt.hour >= 20)
        & (programmes["start_dt"].dt.strftime("%Y-%m-%d") == "2024-11-01")
    ]
    assert not prime_slice.empty, "no prime-time EPG rows to probe"
    operations = _build_break_operations(prime_slice, plan)
    fabricated = [b for b in operations["breaks"] if b["is_gold"]]
    assert not fabricated or plan_gold_ids, (
        f"{len(fabricated)} displayed breaks claim is_gold=True (example: "
        f"{fabricated[0]['channel']} {fabricated[0]['date']} {fabricated[0]['start_time']}) "
        "but the committed plan contains zero gold breaks; the flag is synthesized "
        "from settings, not read from the plan"
    )


def test_bug3_gold_disabled_setting_must_gate_the_engine():
    """KNOWN BUG (MEDIUM). Turning gold breaks OFF in settings does not stop
    the engine from emitting gold. sponsorships_enabled / gold_breaks_enabled
    are consulted only by the display synthesis (kairos_api/server.py:550-551)
    and the gold report (kairos_api/phase_b.py:229-231); the mapping the
    recompute uses, kairos.service.guardrails_from_settings
    (kairos/service.py:128-153), drops both flags, and the optimizer honors a
    gold override regardless (kairos/optimize/optimizer.py:394). So an
    operator who disables sponsorship inventory still gets is_gold=True rows
    in the committed plan from any active gold override.

    Promise asserted: with gold_breaks_enabled=False in the saved settings, an
    optimization run emits no gold placements. Suggested fix: gate gold
    overrides (and gold pins) at the seam where settings become engine inputs,
    for example by rejecting gold overrides in _apply_segment_overrides when
    the settings switch is off, reported via rejected_overrides."""
    from kairos.optimize.optimizer import ProgramSegment, optimize_breaks
    from kairos.optimize.overrides import Override, OverrideSet
    from kairos.service import guardrails_from_settings

    settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    settings["gold_breaks_enabled"] = False
    settings["sponsorships_enabled"] = False
    guardrails = guardrails_from_settings(settings)

    segment = ProgramSegment(
        segment_id="2024-11-01|qa|000", channel="qa", day="2024-11-01",
        start_seconds=20 * 3600.0, duration_seconds=3600.0, program_type="Drama",
        baseline_tvr=5.0, cpp=100.0, impact_coefficient=-0.01,
    )
    overrides = OverrideSet(overrides=[Override(
        override_id="qa-gold", scope="segment",
        target_id=segment.segment_id, kind="gold", value="",
    )])
    result = optimize_breaks(
        [segment], guardrails, revenue_weight=0.6, overrides=overrides,
    )
    gold_placements = [p for p in result.placements if p.is_gold]
    assert not gold_placements, (
        f"settings disable gold breaks, yet the engine emitted {len(gold_placements)} "
        "gold placement(s): the enabled switch never reaches the engine "
        "(guardrails_from_settings drops it)"
    )


def test_bug4_effect_preview_must_honor_saved_guardrails(client, tmp_path, monkeypatch):
    """KNOWN BUG (MEDIUM). The override effect preview, the number the operator
    reads before saving a decision, ignores the saved settings entirely: it
    optimizes with default Guardrails(), the default revenue weight, the YAML
    pricing and the bare YAML classifier (kairos_api/overrides.py:223-226 and
    314-319 call optimize_breaks with no guardrails/settings arguments), while
    the committed plan is built from the saved settings
    (kairos/export/schedule.py:228-244). Today the saved guardrails happen to
    equal the engine defaults, so the drift is invisible; the moment the
    operator tightens a rule the preview quotes before/after numbers from a
    policy world the recompute will not produce.

    Demonstration: with a saved hourly cap of 1 break, a 24-hour channel-day
    can carry at most 24 compliant breaks, yet the preview still reports the
    default-policy baseline (39 breaks on 2024-11-01).

    Promise asserted: the preview's baseline is feasible under the saved
    guardrails. Suggested fix: build the preview through the same seam the
    plan uses (guardrails_from_settings + pricing_from_settings +
    _build_classifier + the saved revenue_weight / risk_lambda), ideally via
    kairos.optimize.day_core._optimize_one_day."""
    tightened = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    tightened["max_breaks_per_hour"] = 1
    tmp_settings = tmp_path / "kairos_settings.json"
    tmp_settings.write_text(json.dumps(tightened, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_PATH", tmp_settings)

    plan = pd.read_csv(CSV_PATH, encoding="utf-8")
    day_rows = plan[(plan["channel"] == "עכשיו 14") & (plan["date"] == "2024-11-01")]
    target = day_rows[day_rows["num_breaks"] >= 1].iloc[0]
    response = client.get("/api/overrides/effect", params={
        "target_id": target.segment_id, "kind": "gold",
    })
    assert response.status_code == 200, response.text
    before_total = response.json()["summary"]["before_total_breaks"]
    hourly_cap_ceiling = 24 * tightened["max_breaks_per_hour"]
    assert before_total <= hourly_cap_ceiling, (
        f"saved settings cap breaks at {tightened['max_breaks_per_hour']}/hour "
        f"(so at most {hourly_cap_ceiling} in the day), but the preview baseline "
        f"reports {before_total} breaks: the preview never reads the saved settings"
    )
