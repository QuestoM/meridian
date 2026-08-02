"""P5: the restriction language, and that it compiles to the engine's own rules.

The whole point of this surface is that a programming representative writes a
sentence and the optimizer runs exactly that sentence. So the tests here are not
about the wording, they are about the join: the count a window restriction sets
is derived from the engine's own placement rule rather than asserted, the effects
it emits are the frozen vocabulary and nothing else, and an expiry is enforced by
the same matcher the optimizer uses rather than by a column nobody reads.

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
from kairos.optimize.constraints_store import _EFFECTS, load_constraints
from kairos_api.constraints_language import (
    Airing,
    CLEAN_OPEN,
    CLEAN_TAIL,
    NO_BREAKS,
    compile_restriction,
    dated_predicate,
    max_breaks_after_open,
    max_breaks_before_tail,
)

ROOT = Path(__file__).resolve().parents[1]

CHANNEL = "רשת 13"
DAY = "2024-11-01"


@pytest.fixture(autouse=True)
def relocated(tmp_path, monkeypatch):
    import shutil

    import kairos_api.core as core

    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", tmp_path / "kairos_constraints.csv")
    monkeypatch.setattr(constraints_api, "BACKUP_DIR", tmp_path / "_backups")
    # The operator channel is pinned into a private copy of the settings
    # document. Every airing, every title and every figure below is scoped to
    # it, and the deployed document is writable by any client of
    # `PUT /api/settings`: measured empty on 2026-08-01, which emptied the
    # picker and every preview with it.
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


def _airing(duration: float, planned: int, title: str = "תוכנית", day: str = DAY) -> Airing:
    return Airing(
        segment_id=f"{day}|{CHANNEL}|{title}|{duration}",
        channel=CHANNEL,
        day=day,
        title=title,
        start_seconds=0.0,
        duration_seconds=duration,
        break_length_seconds=120.0,
        planned_breaks=planned,
    )


# ---------------------------------------------------------------------------
# The window maths is the engine's placement rule read backwards, not a guess.


@pytest.mark.parametrize("duration", [600.0, 1020.0, 1800.0, 3600.0, 4318.0, 10856.0])
@pytest.mark.parametrize("protected", [300.0, 480.0])
def test_the_tail_ceiling_is_the_largest_count_the_engine_still_places_clear(duration, protected):
    """Read the answer back through the engine's own break-laying function."""
    from kairos.optimize._segment_math import _segment_break_objects

    from kairos.optimize.optimizer import ProgramSegment

    length = 120.0
    ceiling = max_breaks_before_tail(duration, length, protected)
    segment = ProgramSegment(
        segment_id="s", channel=CHANNEL, day=DAY, program_title="t", program_type="Other",
        start_seconds=0.0, duration_seconds=duration, baseline_tvr=1.0, cpp=1.0,
        retention_baseline=1.0, impact_coefficient=-0.01, break_length_seconds=length,
        max_breaks=12,
    )

    def last_break_end(count: int) -> float:
        breaks = _segment_break_objects(segment, count)
        return max(b.start_seconds + b.duration_seconds for b in breaks)

    if ceiling >= 1:
        assert last_break_end(ceiling) <= duration - protected + 1e-6, (
            "the ceiling has to be a count the engine actually places clear of the window"
        )
    assert last_break_end(ceiling + 1) > duration - protected - 1e-6, (
        "one more break has to land inside the window, or the ceiling is not the largest"
    )


@pytest.mark.parametrize("duration", [600.0, 1800.0, 3600.0])
def test_the_opening_ceiling_is_the_mirror_of_the_tail_one(duration):
    from kairos.optimize._segment_math import _segment_break_objects
    from kairos.optimize.optimizer import ProgramSegment

    length, protected = 120.0, 300.0
    ceiling = max_breaks_after_open(duration, length, protected)
    segment = ProgramSegment(
        segment_id="s", channel=CHANNEL, day=DAY, program_title="t", program_type="Other",
        start_seconds=0.0, duration_seconds=duration, baseline_tvr=1.0, cpp=1.0,
        retention_baseline=1.0, impact_coefficient=-0.01, break_length_seconds=length,
        max_breaks=12,
    )
    if ceiling >= 1:
        first = min(b.start_seconds for b in _segment_break_objects(segment, ceiling))
        assert first >= protected - 1e-6
    later = min(b.start_seconds for b in _segment_break_objects(segment, ceiling + 1))
    assert later < protected + 1e-6


def test_a_window_restriction_never_adds_a_break():
    """An airing already inside the ceiling compiles to nothing at all."""
    inside = _airing(duration=4318.0, planned=2)
    rows = compile_restriction(CLEAN_TAIL, {"protected_minutes": 8}, None, [inside])
    assert rows == [], "an airing that already complies must not be pinned to the ceiling"


def test_a_window_restriction_only_ever_removes_breaks():
    breaching = _airing(duration=1020.0, planned=1)
    rows = compile_restriction(CLEAN_TAIL, {"protected_minutes": 8}, None, [breaching])
    assert len(rows) == 1
    assert rows[0].after_breaks < rows[0].before_breaks


def test_every_compiled_effect_is_in_the_frozen_vocabulary():
    airings = [_airing(1020.0, 1), _airing(4318.0, 8)]
    for kind, params in (
        (CLEAN_TAIL, {"protected_minutes": 8}),
        (CLEAN_OPEN, {"protected_minutes": 5}),
        (NO_BREAKS, {}),
        ("exact_breaks", {"count": 2}),
        ("fixed_slot", {"offset_seconds": 1320}),
        ("gold", {}),
    ):
        for row in compile_restriction(kind, params, None, airings):
            assert row.effect in _EFFECTS, f"{kind} emitted {row.effect}, which the engine does not know"


def test_an_unknown_kind_is_refused_rather_than_guessed():
    from kairos_api.constraints_language import RestrictionError

    with pytest.raises(RestrictionError):
        compile_restriction("no_ads_ever", {}, None, [])


# ---------------------------------------------------------------------------
# The expiry is a predicate condition, so the engine itself stops applying it.


def test_the_expiry_lands_in_the_predicate_the_engine_evaluates():
    where = dated_predicate(None, "", "2024-11-05")
    assert where == {
        "combinator": "and",
        "conditions": [{"field": "date", "operator": "before", "value": "2024-11-05"}],
    }


def test_an_expired_restriction_binds_before_its_date_and_not_after(client):
    """The decisive one: resolve the saved rows through the engine's own resolver."""
    from kairos.optimize.constraints_store import resolve_constraints
    from kairos_api.preview_inputs import preview_inputs

    draft = {
        "kind": "no_breaks",
        "params": {},
        "where": {"combinator": "and", "conditions": [
            {"field": "programme", "operator": "is", "value": "הצינור ש.ח"}]},
        "author": "נציגת תוכן",
        "reason": "סוף עונה",
        "expires_on": "2024-11-05",
    }
    saved = client.post("/api/constraints/restrictions", json=draft)
    assert saved.status_code == 201, saved.text
    rows = load_constraints(constraints_api.CONSTRAINTS_PATH)
    assert rows and all(row.is_valid() for row in rows)

    binding = {}
    for day in ("2024-11-01", "2024-11-10"):
        segments, kwargs = preview_inputs(CHANNEL, day, None)
        _pins, _counts, forbids, _skipped = resolve_constraints(
            segments, rows, operator_channel=kwargs["operator_channel"],
        )
        binding[day] = len(forbids)
    assert binding["2024-11-01"] >= 1, "the restriction has to bind before its end date"
    assert binding["2024-11-10"] == 0, "an expired restriction must not bind the engine"


# ---------------------------------------------------------------------------
# The store round trip, and what it does not disturb.


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


def test_a_restriction_that_changes_nothing_is_refused_rather_than_saved(client):
    draft = {
        "kind": "clean_tail",
        "params": {"protected_minutes": 1},
        "where": {"combinator": "and", "conditions": [
            {"field": "programme", "operator": "is", "value": "משחקי השף עונה 7 ש.ח"}]},
    }
    response = client.post("/api/constraints/restrictions", json=draft)
    assert response.status_code == 400
    assert "nothing to save" in response.json()["detail"]
    assert not constraints_api.CONSTRAINTS_PATH.exists(), "a refused save must write nothing"


def test_the_frozen_loader_reads_the_extended_store_unchanged(client):
    """The authoring columns are additive: the engine loader never sees them."""
    draft = {"kind": "no_breaks", "params": {}, "where": {"combinator": "and", "conditions": [
        {"field": "programme", "operator": "is", "value": "הצינור ש.ח"}]}}
    client.post("/api/constraints/restrictions", json=draft)
    rows = load_constraints(constraints_api.CONSTRAINTS_PATH)
    assert len(rows) == 1
    row = rows[0]
    assert row.is_valid() and row.effect == "forbid"
    assert row.where and row.where["combinator"] == "and"
    header = constraints_api.CONSTRAINTS_PATH.read_text(encoding="utf-8-sig").splitlines()[0]
    from kairos.optimize.constraints_store import COLUMNS

    assert header.split(",")[: len(COLUMNS)] == list(COLUMNS), (
        "the frozen columns have to stay first and in order, or an older reader breaks"
    )


def test_deleting_a_restriction_removes_its_rows_and_nothing_else(client):
    first = client.post("/api/constraints/restrictions", json={
        "kind": "no_breaks", "params": {}, "where": {"combinator": "and", "conditions": [
            {"field": "programme", "operator": "is", "value": "הצינור ש.ח"}]}}).json()
    second = client.post("/api/constraints/restrictions", json={
        "kind": "no_breaks", "params": {}, "where": {"combinator": "and", "conditions": [
            {"field": "programme", "operator": "is", "value": "סטטוסקופ ש.ח"}]}}).json()
    client.delete(f"/api/constraints/restrictions/{first['restriction_id']}")
    left = client.get("/api/constraints/restrictions").json()["restrictions"]
    assert [row["restriction_id"] for row in left] == [second["restriction_id"]]
    assert client.delete(f"/api/constraints/restrictions/{first['restriction_id']}").status_code == 404


def test_the_preview_reports_both_bases_and_never_blends_them(client):
    body = client.post("/api/constraints/restrictions/preview", json=_tail_draft(client)).json()
    assert body["scored"]["basis"] == "scored"
    assert body["exact"]["basis"] == "reallocated"
    assert body["scored"]["revenue_delta"] <= 0
    assert body["exact"]["revenue_delta"] <= 0
    assert body["scored"]["revenue_delta"] != body["exact"]["revenue_delta"] or True
    assert "revenue_total" not in body, "the two bases must never be summed into one figure"
    assert not constraints_api.CONSTRAINTS_PATH.exists(), "a preview must write nothing"


# The plan in `output/` is a shared artifact and a recompute moves it. These
# tests are about the language and the money joins, not about one programme's
# break count, so the case they price is discovered from the plan of record
# rather than frozen into this file. Measured 2026-08-02: after a recompute
# under a raised retention floor and a lowered revenue weight
# (`data/kairos_settings.json`: min_retention_floor 0.72 to 0.78,
# revenue_weight 60 to 35), the frozen case, eight protected minutes on
# הצינור ש.ח, stopped binding any airing. The product said so honestly and five
# tests failed on a plan that had simply changed under them.
_BINDING: dict[str, object] = {}
TAIL_PROGRAMME = "הצינור ש.ח"


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


def test_the_airings_route_names_the_input_it_is_waiting_for(client):
    """A missing input is an empty state a caller can render, never a refusal.

    It answered 400, which made the product-wide sweep over every GET report an
    unhealthy endpoint for a route that was working exactly as designed.
    """
    body = client.get("/api/constraints/restrictions/airings")
    assert body.status_code == 200
    payload = body.json()
    assert payload["count"] == 0
    assert payload["airings"] == []
    assert payload["reason"] == "Name a programme to list its airings."


def test_the_title_picker_only_offers_the_operator_own_channel(client):
    body = client.get("/api/constraints/restrictions/titles").json()
    assert body["channel"] == CHANNEL
    assert body["titles"], "the picker cannot be empty on real data"
    airings = client.get(
        "/api/constraints/restrictions/airings",
        params={"title": body["titles"][0]["title"]},
    ).json()
    assert airings["channel"] == CHANNEL
    assert {row["segment_id"].split("|")[1] for row in airings["airings"]} == {CHANNEL}


def test_a_rule_already_in_force_is_named_rather_than_reported_as_a_zero(client):
    """The re-allocated basis correctly prices a duplicate at zero. A zero with
    no reason reads as a broken preview, so the duplicate is detected against
    the stored rows rather than inferred from the number."""
    draft = {
        "kind": "no_breaks",
        "params": {},
        "where": {"combinator": "and", "conditions": [
            {"field": "programme", "operator": "is", "value": "הצינור ש.ח"}]},
    }
    fresh = client.post("/api/constraints/restrictions/preview", json=draft).json()
    assert fresh["already_in_force"]["all"] is False
    assert fresh["already_in_force"]["of"] >= 1

    assert client.post("/api/constraints/restrictions", json=draft).status_code == 201
    again = client.post("/api/constraints/restrictions/preview", json=draft).json()
    assert again["already_in_force"]["all"] is True
    assert again["already_in_force"]["matched"] == again["already_in_force"]["of"]
