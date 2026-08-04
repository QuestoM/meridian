"""P5: the two money bases a restriction is priced on, and their starting points.

Split out of ``test_p5_restrictions.py`` under the 450-line law. That file is
about the language and what it compiles to; this one is about the money, which
is the half that has to join two engines and a file on disk.

A restriction is a decision about somebody else's revenue, so its cost is on
screen before the save. It is reported on two named bases that are never blended:
the scored one counts from the plan of record, the re-allocated one from a run
today, and they do not share a starting point. Both are pinned here against their
own source rather than against each other, because two figures that agree only
with one another prove nothing.

Every store write is relocated into tmp, so no test touches ``data/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.constraints as constraints_api
import kairos_api.version_store as vs

ROOT = Path(__file__).resolve().parents[1]

CHANNEL = "רשת 13"

# The plan in `output/` is a shared artifact and a recompute moves it. These
# tests are about the money joins, not about one programme's break count, so the
# case they price is discovered from the plan of record rather than frozen into
# this file. Measured 2026-08-02: after a recompute under a raised retention
# floor and a lowered revenue weight (`data/kairos_settings.json`:
# min_retention_floor 0.72 to 0.78, revenue_weight 60 to 35), the frozen case,
# eight protected minutes on הצינור ש.ח, stopped binding any airing. The product
# said so honestly and five tests failed on a plan that had simply changed under
# them.
_BINDING: dict[str, object] = {}
TAIL_PROGRAMME = "הצינור ש.ח"


@pytest.fixture(autouse=True)
def relocated(tmp_path, monkeypatch):
    import shutil

    import kairos_api.core as core

    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", tmp_path / "kairos_constraints.csv")
    monkeypatch.setattr(constraints_api, "BACKUP_DIR", tmp_path / "_backups")
    # The operator channel is pinned into a private copy of the settings
    # document. Every airing, every title and every figure below is scoped to it.
    settings_copy = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", settings_copy)
    document = json.loads(settings_copy.read_text(encoding="utf-8"))
    document["operator_channel"] = CHANNEL
    settings_copy.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_PATH", settings_copy)
    return tmp_path


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(constraints_api.router)
    return TestClient(app)


def _preview(client, draft: dict) -> dict:
    response = client.post("/api/constraints/restrictions/preview", json=draft)
    assert response.status_code == 200, response.text
    return response.json()


def _tail_conditions(day: str | None = None) -> list[dict]:
    conditions = [{"field": "programme", "operator": "is", "value": TAIL_PROGRAMME}]
    if day:
        conditions.append({"field": "date", "operator": "is", "value": day})
    return conditions


def _binding_tail(client) -> dict:
    """The smallest tail window that binds, and one day of the plan it binds on.

    Both bases have to be available on that day, because the tests below compare
    them. Cached for the session: the plan does not move inside one run.
    """
    if _BINDING:
        return dict(_BINDING)
    for minutes in (8, 10, 12, 15, 20, 25, 30):
        wide = {
            "kind": "clean_tail",
            "params": {"protected_minutes": minutes},
            "where": {"combinator": "and", "conditions": _tail_conditions()},
        }
        body = _preview(client, wide)
        if not body.get("bound_airings"):
            continue
        for day in sorted({str(change.get("day")) for change in body.get("changes", []) if change.get("day")}):
            dated = {
                "kind": "clean_tail",
                "params": {"protected_minutes": minutes},
                "where": {"combinator": "and", "conditions": _tail_conditions(day)},
            }
            probe = _preview(client, dated)
            if probe["scored"].get("available") and probe["exact"].get("available"):
                _BINDING.update({"minutes": minutes, "day": day})
                return dict(_BINDING)
    pytest.skip(
        f"no clean_tail window on {TAIL_PROGRAMME} binds a day of the plan of record as it "
        "stands, so there is no priced case to compare the two bases on"
    )


def _tail_draft(client) -> dict:
    case = _binding_tail(client)
    return {
        "kind": "clean_tail",
        "params": {"protected_minutes": case["minutes"]},
        "where": {"combinator": "and", "conditions": _tail_conditions(str(case["day"]))},
    }


def test_the_saved_record_carries_author_reason_and_expiry(client):
    minutes = _binding_tail(client)["minutes"]
    draft = {
        "kind": "clean_tail",
        "params": {"protected_minutes": minutes},
        "where": {"combinator": "and", "conditions": _tail_conditions()},
        "author": "נציגת תוכן",
        "reason": "גמר עונה",
        "expires_on": "2026-12-31",
    }
    saved = client.post("/api/constraints/restrictions", json=draft).json()
    assert saved["author"] == "נציגת תוכן"
    assert saved["reason"] == "גמר עונה"
    assert saved["expires_on"] == "2026-12-31"
    assert saved["status"] == "active"
    assert str(minutes) in saved["sentence_he"] and TAIL_PROGRAMME in saved["sentence_he"]

    listed = client.get("/api/constraints/restrictions").json()["restrictions"]
    assert [row["restriction_id"] for row in listed] == [saved["restriction_id"]]
    assert listed[0]["sentence_he"] == saved["sentence_he"], (
        "the list has to read the rule back the way its author wrote it, not the way one "
        "compiled row happens to sort"
    )


def test_a_rule_can_be_dated_to_start_later_as_well_as_to_stop(client):
    """The composer offers both ends now, so the store has to hold both.

    The dates bracket a night the rule really binds, discovered from the plan of
    record rather than frozen, because both ends compile into the predicate the
    engine evaluates: a window that contains no breaching airing correctly saves
    nothing, and asserting a hard-coded fortnight would be asserting the plan.
    """
    from datetime import date, timedelta

    case = _binding_tail(client)
    priced = date.fromisoformat(str(case["day"]))
    base = {
        "kind": "clean_tail",
        "params": {"protected_minutes": case["minutes"]},
        "where": {"combinator": "and", "conditions": _tail_conditions()},
        "author": "נציגת תוכן",
        "starts_on": (priced - timedelta(days=1)).isoformat(),
        "expires_on": (priced + timedelta(days=1)).isoformat(),
    }
    saved = client.post("/api/constraints/restrictions", json=base)
    assert saved.status_code == 201, saved.text
    record = saved.json()
    assert record["starts_on"] == base["starts_on"]
    assert record["expires_on"] == base["expires_on"]

    backwards = {**base, "starts_on": base["expires_on"], "expires_on": base["starts_on"]}
    refused = client.post("/api/constraints/restrictions", json=backwards)
    assert refused.status_code == 400
    assert "after the start date" in refused.json()["detail"]


def test_the_preview_reports_both_bases_and_never_blends_them(client):
    body = client.post("/api/constraints/restrictions/preview", json=_tail_draft(client)).json()
    assert body["scored"]["basis"] == "scored"
    assert body["exact"]["basis"] == "reallocated"
    assert body["scored"]["revenue_delta"] <= 0
    assert body["exact"]["revenue_delta"] <= 0
    assert "revenue_total" not in body, "the two bases must never be summed into one figure"
    assert not constraints_api.CONSTRAINTS_PATH.exists(), "a preview must write nothing"


def test_the_exact_basis_is_the_commit_paths_own_before_to_the_cent(client):
    """The re-allocated figure is the optimizer's, or the preview is a second engine."""
    from kairos.optimize.constraints_store import load_constraints
    from kairos.optimize.day_core import _optimize_one_day
    from kairos_api.overrides import _resolved_store_overrides
    from kairos_api.preview_inputs import preview_inputs

    body = client.post("/api/constraints/restrictions/preview", json=_tail_draft(client)).json()
    segments, kwargs = preview_inputs(CHANNEL, str(_binding_tail(client)["day"]), None)
    stored = load_constraints(constraints_api.CONSTRAINTS_PATH)
    active, _stale = _resolved_store_overrides(segments)
    baseline = _optimize_one_day(
        segments, constraints=stored, overrides=active if active.overrides else None, **kwargs,
    )
    assert round(baseline.total_revenue, 2) == body["exact"]["revenue_before"]
    assert body["exact"]["starting_point"] == "optimizer_today"


def test_the_scored_basis_is_the_saved_plans_own_revenue_for_the_day(client):
    """Cross-checked against the plan file itself, not against another engine call.

    The two bases do not share a starting point and cannot be asserted equal:
    the scored one counts from the plan of record, the re-allocated one from a
    run today. This pins the scored one to the file the plan lives in, which is
    the only source outside the code that can confirm it.
    """
    import pandas as pd

    from kairos_api.core import OUTPUT_DIR

    body = client.post("/api/constraints/restrictions/preview", json=_tail_draft(client)).json()
    priced = str(_binding_tail(client)["day"])
    plan = pd.read_csv(OUTPUT_DIR / "weekly_break_schedule.csv")
    day = plan[(plan["channel"].astype(str) == CHANNEL) & (plan["date"].astype(str) == priced)]
    assert not day.empty, "the golden plan has to carry the day this test prices"
    filed = float(day["predicted_revenue"].astype(float).sum())
    # The file holds 82 rows each already rounded to the agora and the basis
    # rounds once at the end, so the two cannot agree closer than the rounding
    # the file itself carries. Measured difference on the golden plan: 0.01.
    assert abs(filed - body["scored"]["revenue_before"]) <= 0.05, (
        "the scored basis has to be the plan of record's own revenue for the day"
    )
    assert body["scored"]["breaks_before"] == int(day["num_breaks"].astype(float).sum())
    assert body["scored"]["starting_point"] == "saved_plan"


def test_the_two_starting_points_are_named_and_their_gap_is_reported(client):
    """Two money figures on one screen never differ without the payload saying so."""
    body = client.post("/api/constraints/restrictions/preview", json=_tail_draft(client)).json()
    points = body["starting_points"]
    assert points["comparable"] is True
    gap = round(body["exact"]["revenue_before"] - body["scored"]["revenue_before"], 2)
    assert points["gap"] == gap
    assert points["same_start"] is (gap == 0.0)
    if gap:
        assert points["note_he"] and points["note_en"]
    for side in ("scored", "exact"):
        assert body[side]["starting_point_he"], f"{side} has to name its starting point in Hebrew"
        assert body[side]["starting_point_en"], f"{side} has to name its starting point in English"
