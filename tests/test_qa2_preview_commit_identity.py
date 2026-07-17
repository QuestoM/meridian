"""The effect previews are the commit engine, not a parallel one.

/api/overrides/effect and /api/constraints/effect promise plans "the weekly
recompute would write". These tests hold them to it against the real commit
engine (kairos.export.schedule.build_weekly_schedule) on one real channel-day:

- On the current repo state (empty override store, no constraints file) both
  previews must reproduce the commit plan exactly: identical break totals,
  revenue within per-row CSV rounding, and an empty changed list.
- With the measured first-break multiplier activated (monkeypatched coefficient
  metadata), the commit plan moves; the preview must move WITH it. Before this
  wave the preview skipped the first-break fold entirely, so this leg is the
  regression tripwire for any future seam the commit gains and the preview
  misses.

Read-only end to end: the commit leg builds a frame in memory (nothing under
data/ or output/ is written) and the previews write nothing by contract.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.constraints as constraints_api
import kairos_api.overrides as overrides_api

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def channel_day():
    """One real (channel, day) pair plus the filtered programmes frame."""
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
    filtered = mine[mine["start_dt"].dt.strftime("%Y-%m-%d") == day].copy()
    return channel, day, filtered


@pytest.fixture(scope="module")
def client() -> TestClient:
    app = FastAPI()
    app.include_router(overrides_api.router)
    app.include_router(constraints_api.router)
    return TestClient(app)


def _commit_frame(filtered: pd.DataFrame) -> pd.DataFrame:
    """Run the REAL commit engine on the filtered channel-day, in memory."""
    from kairos.export.schedule import build_weekly_schedule
    from kairos_api.core import _load_settings, _model_dump, _reference_today

    saved = _load_settings()
    return build_weekly_schedule(
        programmes=filtered,
        settings=_model_dump(saved),
        revenue_weight=saved.revenue_weight / 100.0,
        risk_lambda=saved.risk_lambda,
        operator_channel=saved.operator_channel,
        today=_reference_today(saved),
        objective_mode=getattr(saved, "objective_mode", "blend"),
    )


def _commit_totals(frame: pd.DataFrame) -> tuple[int, float]:
    breaks = int(pd.to_numeric(frame["num_breaks"], errors="coerce").fillna(0).sum())
    revenue = float(pd.to_numeric(frame["predicted_revenue"], errors="coerce").fillna(0).sum())
    return breaks, revenue


def _revenue_tolerance(frame: pd.DataFrame) -> float:
    # The commit frame rounds each row's revenue to 2 decimals while the preview
    # rounds the day total once, so allow half a cent per row of rounding drift.
    return 0.005 * max(1, len(frame)) + 0.01


def test_override_preview_reproduces_the_commit_plan(channel_day, client) -> None:
    channel, day, filtered = channel_day
    commit_breaks, commit_revenue = _commit_totals(_commit_frame(filtered))

    response = client.get("/api/overrides/effect", params={"channel": channel, "day": day})
    assert response.status_code == 200, response.text
    body = response.json()
    summary = body["summary"]
    # Empty override store today: baseline, overridden and commit are ONE plan.
    assert summary["before_total_breaks"] == commit_breaks
    assert summary["after_total_breaks"] == commit_breaks
    tolerance = _revenue_tolerance(filtered)
    assert summary["before_revenue"] == pytest.approx(commit_revenue, abs=tolerance)
    assert summary["after_revenue"] == pytest.approx(commit_revenue, abs=tolerance)
    assert body["changed"] == []
    assert body["rejected_overrides"] == []


def test_constraint_preview_reproduces_the_commit_plan(channel_day, client) -> None:
    channel, day, filtered = channel_day
    commit_breaks, commit_revenue = _commit_totals(_commit_frame(filtered))

    response = client.get("/api/constraints/effect", params={"channel": channel, "day": day})
    assert response.status_code == 200, response.text
    body = response.json()
    summary = body["summary"]
    # No constraints file today: both legs equal the commit plan exactly.
    assert summary["before_total_breaks"] == commit_breaks
    assert summary["after_total_breaks"] == commit_breaks
    tolerance = _revenue_tolerance(filtered)
    assert summary["before_revenue"] == pytest.approx(commit_revenue, abs=tolerance)
    assert summary["after_revenue"] == pytest.approx(commit_revenue, abs=tolerance)
    assert body["changed"] == []
    assert body["skipped_constraints"] == []
    assert summary["matched_segments"] == 0


def test_preview_follows_the_commit_when_the_first_break_fold_activates(
    channel_day, client, monkeypatch,
) -> None:
    """Activate the measured first-break multiplier for BOTH engines and require
    the preview to track the commit exactly. The old preview skipped this fold,
    so a regression back to a parallel preview engine fails here loudly."""
    import kairos.service as service

    channel, day, filtered = channel_day
    original = service.read_coefficients_metadata

    def folded(path):
        metadata = dict(original(path) or {})
        metadata["first_break_multiplier"] = 3.0
        return metadata

    monkeypatch.setattr(service, "read_coefficients_metadata", folded)

    frame = _commit_frame(filtered)
    commit_breaks, commit_revenue = _commit_totals(frame)
    response = client.get("/api/overrides/effect", params={"channel": channel, "day": day})
    assert response.status_code == 200, response.text
    summary = response.json()["summary"]
    assert summary["before_total_breaks"] == commit_breaks
    assert summary["after_total_breaks"] == commit_breaks
    tolerance = _revenue_tolerance(filtered)
    assert summary["before_revenue"] == pytest.approx(commit_revenue, abs=tolerance)
    assert summary["after_revenue"] == pytest.approx(commit_revenue, abs=tolerance)
