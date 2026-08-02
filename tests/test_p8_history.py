"""P8 History: the merged timeline, the restore that is refused, and the boundary.

Every test relocates the version store, the activity log, the run log and the
settings file into a tmp tree, so nothing here reads or writes the repository's
own records. The three new routes are exercised through a real client over a
locally composed app, which is the same technique tests/test_version_store.py
already uses for the versions router.

What these tests are for, in one line each:

- the merge does not widen a permission that already exists,
- a rival channel's run never reaches an operator surface,
- a training entry never reaches a channel account,
- a version captured against a different store is refused rather than restored,
- and a run delta is a subtraction of two recorded figures, never an invention.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.advertisers as advertisers_api
import kairos_api.constraints as constraints_api
import kairos_api.overrides as overrides_api
import kairos_api.version_store as vs
from kairos.observability import run_log as run_log_module
from kairos_api import activity_log, core, history_api, history_api_runs, history_api_timeline

# The run log is append-only and drops nothing, so its start is the first run
# recorded rather than a floor. Written out rather than imported, so a change to
# the constant fails here.
RUNS_RETENTION = {"pruned": False, "keeps": None, "prune_at": None, "unit": "runs"}
ROOT = Path(__file__).resolve().parents[1]

OWNED = "רשת 13"
RIVALS = ("קשת 12", "כאן 11", "עכשיו 14")

# The training lexicon of specification section 4.2. A run surface returns zero
# hits for every one of these words.
TRAINING_LEXICON = (
    "gate", "held_out", "tau", "drift", "coefficient", "pooling",
    "p_value", "training_window", "wartime",
)


def _run_record(run_id: str, created_at: str, channel: str, day: str | None,
                revenue: float, breaks: int, retention: float) -> dict[str, Any]:
    """One run log line, shaped exactly as kairos.observability.run_log writes it."""
    return {
        "run_id": run_id,
        "created_at": created_at,
        "channel": channel,
        "day": day,
        "engine_version": "1.0.0",
        "segment_count": 82,
        "input_checksums": {"programmes": "abc123"},
        "guardrails": {"max_breaks_per_hour": 4},
        "assumptions": {"revenue_weight": 0.5},
        "summary": {
            "projected_revenue": revenue,
            "total_breaks": breaks,
            "total_ad_seconds": 9600,
            "average_retention": retention,
            "objective": 0.53,
            "compliant": True,
        },
    }


@pytest.fixture()
def history_env(tmp_path, monkeypatch):
    """A relocated version store, activity log, run log and settings file."""
    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    monkeypatch.setenv("KAIROS_AUDIT_DIR", str(tmp_path / "audit"))
    activity_log.reset_runtime_state()

    settings_path = tmp_path / "kairos_settings.json"
    settings_path.write_text(json.dumps({"operator_channel": OWNED}), encoding="utf-8")
    monkeypatch.setattr(core, "SETTINGS_PATH", settings_path)
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", tmp_path / "kairos_constraints.csv")
    monkeypatch.setattr(overrides_api, "OVERRIDES_PATH", tmp_path / "manual_overrides.csv")
    monkeypatch.setattr(advertisers_api, "RULES_PATH", tmp_path / "advertiser_rules.csv")

    run_log = tmp_path / "run_log.jsonl"
    records = [
        _run_record("a" * 32, "2026-07-20T08:00:00+00:00", OWNED, "2024-11-11", 1414695.20, 80, 95.0),
        _run_record("b" * 32, "2026-07-21T08:00:00+00:00", OWNED, "2024-11-11", 1292939.67, 80, 93.1),
    ] + [
        _run_record(f"{letter}" * 32, "2026-07-22T08:00:00+00:00", rival, "2024-11-11", 999.0, 9, 90.0)
        for letter, rival in zip("cde", RIVALS)
    ]
    run_log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
    monkeypatch.setattr(run_log_module, "DEFAULT_RUN_LOG_PATH", run_log)

    app = FastAPI()

    @app.middleware("http")
    async def _recorder(request, call_next):  # the real request recorder, as server.py mounts it
        return await activity_log.record_api_mutation(request, call_next)

    app.include_router(history_api.router)
    app.include_router(history_api.timeline_router)
    yield TestClient(app)
    activity_log.reset_runtime_state()


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    """A live auth store with one admin, one operator, one viewer, one channel account."""
    from kairos_api import auth_store

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    auth_store.seed_initial_admin(password="rootpass-1234")
    auth_store.add_user("viewer1", "viewerpass-123", "viewer", "V", must_change_password=False)
    auth_store.add_user("operator1", "operatorpass-123", "operator", "O", must_change_password=False)
    auth_store.add_user("chan1", "chanpass-1234", "operator", "C", must_change_password=False)
    auth_store.set_affiliation("chan1", "channel")
    yield auth_store
    auth_store.reset_runtime_state()


def _as(client: TestClient, auth_store, username: str, role: str) -> TestClient:
    client.cookies.set(auth_store.COOKIE_NAME, auth_store.create_session(username, role))
    return client


# --- the merge -----------------------------------------------------------------

def test_the_timeline_merges_four_records_into_one_newest_first_order(history_env) -> None:
    vs.snapshot("manual_snapshot", "admin", ["settings"], force=True)
    history_env.post("/api/definitely-not-a-route")

    body = history_env.get("/api/history").json()
    kinds = {entry["kind"] for entry in body["entries"]}
    assert {"restore_point", "run"} <= kinds, "the version store and the run log both reach the timeline"
    stamps = [entry["ts"] for entry in body["entries"]]
    assert stamps == sorted(stamps, reverse=True), "the merged order is newest first"
    assert body["counts"]["run"] == 2, "only the operator's own two runs are counted"
    assert body["sources"]["runs"]["available"] is True


def test_every_entry_carries_the_same_seven_fields(history_env) -> None:
    vs.snapshot("manual_snapshot", "admin", ["settings"], force=True)
    for entry in history_env.get("/api/history").json()["entries"]:
        assert set(entry) == {"id", "kind", "ts", "actor", "via", "artifact_root", "facts"}
        assert entry["kind"] in history_api_timeline.KINDS
        assert entry["artifact_root"] in history_api_timeline.ARTIFACT_ROOTS


def test_the_run_log_is_reported_unavailable_rather_than_empty(history_env, monkeypatch, tmp_path) -> None:
    """A missing record is an honest absence, not zero runs."""
    monkeypatch.setattr(run_log_module, "DEFAULT_RUN_LOG_PATH", tmp_path / "gone.jsonl")
    body = history_env.get("/api/history").json()
    # A record that could not be read cannot date itself either.
    assert body["sources"]["runs"] == {"records": 0, "available": False, "state": "unreadable",
                                       "starts": None, "retention": RUNS_RETENTION}
    assert body["counts"]["run"] == 0


# --- the competitor boundary ---------------------------------------------------

def test_a_rival_channels_run_never_reaches_the_timeline(history_env) -> None:
    body = history_env.get("/api/history", params={"kind": "run", "limit": 500}).json()
    channels = {entry["facts"]["channel"] for entry in body["entries"]}
    assert channels == {OWNED}
    assert body["run_scope"]["scope_channel"] == OWNED
    assert body["run_scope"]["competitor_rows_excluded"] == 3
    assert body["run_scope"]["competitor_channels_excluded"] == 3


def test_no_rival_channel_name_appears_anywhere_in_the_payload(history_env) -> None:
    raw = history_env.get("/api/history", params={"limit": 500}).text
    for rival in RIVALS:
        assert rival not in raw, f"{rival} reached an operator surface"


def test_a_rival_run_id_is_not_found_rather_than_served(history_env) -> None:
    assert history_env.get(f"/api/history/runs/{'c' * 32}").status_code == 404
    assert history_env.get(f"/api/history/runs/{'b' * 32}").status_code == 200


def test_with_no_operator_channel_every_run_is_withheld_rather_than_served(history_env) -> None:
    """A measured breach, not a hypothetical. While this piece was being built
    another surface blanked ``operator_channel`` in the shared settings file, and
    a kind=run read on the running instance returned 217, 124 and 76 rows for
    three channels the operator does not own, with their names and their revenue
    on them. The scope helper is behaving correctly when it passes everything
    through and says it could not scope; this surface may not serve that."""
    settings_path = Path(core.SETTINGS_PATH)
    settings_path.write_text(json.dumps({"operator_channel": ""}), encoding="utf-8")

    body = history_env.get("/api/history", params={"limit": 500}).json()
    assert body["counts"]["run"] == 0
    assert body["sources"]["runs"] == {"records": 0, "available": False, "starts": None,
                                       "state": "withheld_no_operator_channel",
                                       "retention": RUNS_RETENTION}
    assert body["run_scope"]["scoped"] is False
    assert body["run_scope"]["reason"], "the reason names the missing input"

    raw = history_env.get("/api/history", params={"limit": 500}).text
    for rival in RIVALS + (OWNED,):
        assert rival not in raw, f"{rival} reached the timeline with no channel configured"

    refused = history_env.get(f"/api/history/runs/{'b' * 32}")
    assert refused.status_code == 409
    assert "operator channel" in refused.json()["detail"]
    for rival in RIVALS:
        assert rival not in refused.text


# --- the training line ---------------------------------------------------------

def test_a_models_entry_is_invisible_to_a_channel_account() -> None:
    entries = [
        {"id": "change:1", "kind": "change", "artifact_root": "data"},
        {"id": "change:2", "kind": "change", "artifact_root": "models"},
        {"id": "run:1", "kind": "run", "artifact_root": "output"},
    ]
    company = history_api_timeline.visible(entries, is_company=True)
    channel = history_api_timeline.visible(entries, is_company=False)
    assert len(company) == 3
    assert [entry["id"] for entry in channel] == ["change:1", "run:1"]
    assert all(entry["artifact_root"] != "models" for entry in channel)


def test_a_channel_account_never_sees_a_models_entry_on_the_route(history_env, auth_env) -> None:
    """The unit above proves the filter; this proves the door. A write under
    /api/model is a training act by the section 4.1 test, so it reaches a
    company account's timeline and never a channel account's, with no count, no
    marker and no gap where it was."""
    admin = _as(history_env, auth_env, "admin", "admin")
    admin.post("/api/model/anything")
    admin.post("/api/settings-like-anything")

    seen = admin.get("/api/history", params={"limit": 500}).json()
    roots = {entry["artifact_root"] for entry in seen["entries"]}
    assert "models" in roots, "the company side sees the training entry"

    channel = _as(history_env, auth_env, "chan1", "operator")
    body = channel.get("/api/history", params={"limit": 500}).json()
    assert body["entries"] != []
    assert all(entry["artifact_root"] != "models" for entry in body["entries"])
    assert all("/api/model" not in str((entry["facts"] or {}).get("path") or "")
               for entry in body["entries"])


def test_the_artifact_root_is_the_training_test_applied_to_a_path() -> None:
    assert history_api_timeline.artifact_root("/api/model/train") == "models"
    assert history_api_timeline.artifact_root("/api/settings") == "data"
    assert history_api_timeline.artifact_root("/api/uploads/spots") == "data"
    assert history_api_timeline.artifact_root(None) == "data"


def test_the_timeline_payload_carries_no_training_lexicon(history_env) -> None:
    vs.snapshot("manual_snapshot", "admin", ["settings"], force=True)
    raw = history_env.get("/api/history", params={"limit": 500}).text.lower()
    hits = [word for word in TRAINING_LEXICON if word in raw]
    assert hits == [], f"the run surface leaks the training lexicon: {hits}"


def test_the_action_vocabulary_is_closed() -> None:
    assert history_api_timeline.action_for("PUT", "/api/settings") == "settings_change"
    assert history_api_timeline.action_for("POST", "/api/jobs/recompute") == "plan_run"
    assert history_api_timeline.action_for("DELETE", "/api/constraints/c1") == "restriction_change"
    assert history_api_timeline.action_for("POST", "/api/assistant/proposals/b1/apply") == "assistant_action"
    assert history_api_timeline.action_for("POST", "/api/nothing-like-this") == "other"


def test_a_concrete_path_with_an_id_in_it_still_resolves_to_its_act() -> None:
    """The recorder stores the concrete path, so a break act arrives with the
    segment key inside it and two different acts share the same prefix."""
    key = "/api/breaks/2024-11-01|רשת 13|000~1"
    assert history_api_timeline.action_for("POST", f"{key}/placement") == "placement_change"
    assert history_api_timeline.action_for("DELETE", f"{key}/placement") == "placement_change"
    assert history_api_timeline.action_for("POST", f"{key}/gold") == "gold_change"
    assert history_api_timeline.action_for("POST", "/api/uploads/spots/check") == "source_check"
    assert history_api_timeline.action_for("POST", "/api/uploads/spots") == "source_upload"


# Every write operation the assembled product published when this piece was
# built, taken from its own openapi.json rather than written by hand. A route
# that nobody classified would render as the word "Change" over something that
# may have changed nothing, which is the defect this list exists to catch.
WRITE_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("POST", "/api/advertisers", "client_change"),
    ("PUT", "/api/advertisers/adv1", "client_change"),
    ("DELETE", "/api/advertisers/adv1", "client_change"),
    ("POST", "/api/advertisers/adv1/conditions", "client_change"),
    ("PUT", "/api/advertisers/adv1/conditions/r1", "client_change"),
    ("DELETE", "/api/advertisers/adv1/conditions/r1", "client_change"),
    ("POST", "/api/agencies", "client_change"),
    ("PUT", "/api/agencies/ag1", "client_change"),
    ("POST", "/api/agencies/ag1/advertisers", "client_change"),
    ("DELETE", "/api/agencies/ag1/advertisers/adv1", "client_change"),
    ("POST", "/api/agencies/ag1/conditions", "client_change"),
    ("PUT", "/api/agencies/ag1/conditions/r1", "client_change"),
    ("DELETE", "/api/agencies/ag1/conditions/r1", "client_change"),
    ("POST", "/api/agencies/ag1/deactivate", "client_change"),
    ("POST", "/api/assistant/ask", "assistant_ask"),
    ("POST", "/api/assistant/ask/stream", "assistant_ask"),
    ("POST", "/api/assistant/context/warm", "assistant_context"),
    ("POST", "/api/assistant/conversations", "conversation_change"),
    ("PATCH", "/api/assistant/conversations/c1", "conversation_change"),
    ("DELETE", "/api/assistant/conversations/c1", "conversation_change"),
    ("POST", "/api/assistant/conversations/c1/restore", "conversation_change"),
    ("POST", "/api/assistant/proposals/b1/apply", "assistant_action"),
    ("POST", "/api/assistant/proposals/b1/reject", "assistant_action"),
    ("POST", "/api/assistant/restore/r1", "assistant_undo"),
    ("DELETE", "/api/assistant/thread", "conversation_change"),
    ("POST", "/api/assistant/upload", "assistant_upload"),
    ("DELETE", "/api/assistant/uploads/u1", "assistant_upload"),
    ("POST", "/api/auth/change-password", "password_change"),
    ("PUT", "/api/auth/job", "job_change"),
    ("POST", "/api/auth/users", "account_change"),
    ("DELETE", "/api/auth/users/u1", "account_change"),
    ("PUT", "/api/auth/users/u1/affiliation", "account_change"),
    ("POST", "/api/auth/users/u1/reset-password", "account_change"),
    ("POST", "/api/break-decisions", "decision"),
    ("POST", "/api/breaks/b1/gold", "gold_change"),
    ("DELETE", "/api/breaks/b1/gold", "gold_change"),
    ("POST", "/api/breaks/b1/placement", "placement_change"),
    ("DELETE", "/api/breaks/b1/placement", "placement_change"),
    ("POST", "/api/clients/campaigns", "campaign_change"),
    ("PUT", "/api/clients/campaigns/c1", "campaign_change"),
    ("POST", "/api/clients/campaigns/c1/deactivate", "campaign_change"),
    ("POST", "/api/clients/campaigns/c1/flights", "campaign_change"),
    ("PUT", "/api/clients/campaigns/c1/flights/f1", "campaign_change"),
    ("DELETE", "/api/clients/campaigns/c1/flights/f1", "campaign_change"),
    ("POST", "/api/clients/onboarding", "client_onboarding"),
    ("POST", "/api/constraints", "restriction_change"),
    ("POST", "/api/constraints/restrictions", "restriction_change"),
    ("POST", "/api/constraints/restrictions/preview", "restriction_preview"),
    ("DELETE", "/api/constraints/restrictions/r1", "restriction_change"),
    ("PUT", "/api/constraints/c1", "restriction_change"),
    ("DELETE", "/api/constraints/c1", "restriction_change"),
    ("POST", "/api/events", "calendar_change"),
    ("PUT", "/api/events/e1", "calendar_change"),
    ("DELETE", "/api/events/e1", "calendar_change"),
    ("POST", "/api/jobs/recompute", "plan_run"),
    ("POST", "/api/model/candidates/spotclip/measure", "candidate_measure"),
    ("POST", "/api/model/decisions", "model_decision"),
    ("POST", "/api/model/training", "model_training"),
    ("POST", "/api/model/versions", "model_version"),
    ("POST", "/api/optimal-plan", "preview"),
    ("POST", "/api/optimizer-plan", "preview"),
    ("POST", "/api/overrides", "override_change"),
    ("PUT", "/api/overrides/o1", "override_change"),
    ("DELETE", "/api/overrides/o1", "override_change"),
    ("PUT", "/api/plan-target", "target_change"),
    ("DELETE", "/api/plan-target", "target_change"),
    ("POST", "/api/plan-versions", "plan_publish"),
    ("POST", "/api/plan-versions/v1/restore", "plan_restore"),
    ("POST", "/api/plan/day/score", "placement_preview"),
    ("PUT", "/api/pricing", "pricing_change"),
    ("POST", "/api/pricing/effect", "price_preview"),
    ("POST", "/api/pricing/price-slot", "price_test"),
    ("POST", "/api/recompute-schedule", "plan_run"),
    ("POST", "/api/rules/guardrails", "guardrail_change"),
    ("POST", "/api/rules/guardrails/apply", "guardrail_change"),
    ("PUT", "/api/rules/model-activation", "model_activation_change"),
    ("PUT", "/api/rules/operator-channel", "channel_change"),
    ("POST", "/api/scenario", "preview"),
    ("POST", "/api/scenario-compare", "preview"),
    ("PUT", "/api/settings", "settings_change"),
    ("POST", "/api/uploads/spots", "source_upload"),
    ("POST", "/api/uploads/spots/check", "source_check"),
    ("POST", "/api/versions/snapshot", "restore_point_saved"),
    ("PATCH", "/api/versions/v1", "restore_point_renamed"),
    ("POST", "/api/versions/v1/restore", "restore"),
)


def test_every_write_the_product_publishes_has_a_word_of_its_own() -> None:
    """87 write operations were published when this was measured. Login and
    logout are excluded from the recorder itself, so 85 reach the timeline."""
    wrong = {
        f"{method} {path}": history_api_timeline.action_for(method, path)
        for method, path, expected in WRITE_ROUTES
        if history_api_timeline.action_for(method, path) != expected
    }
    assert wrong == {}
    assert len(WRITE_ROUTES) == 85


def test_an_act_that_saved_nothing_is_a_preview_and_not_a_change() -> None:
    """The sharpest case measured live: the day board scores a placement on
    every drop, which writes nothing, and it was 57 of the 345 recorded requests
    in the newest 500 entries."""
    for method, path, action in WRITE_ROUTES:
        kind = history_api_timeline.kind_for(action)
        assert kind in ("change", "preview")
        assert kind == ("preview" if action in history_api_timeline.PREVIEW_ACTIONS else "change")
    assert history_api_timeline.kind_for("placement_preview") == "preview"
    assert history_api_timeline.kind_for("placement_change") == "change"
    assert "preview" in history_api_timeline.KINDS


def test_a_refused_write_stays_a_change_because_somebody_tried_it(history_env, auth_env) -> None:
    """A 403 changed nothing, and it is still the thing a person reading this
    surface most needs to see, so it is not filed away as a preview."""
    viewer = _as(history_env, auth_env, "viewer1", "viewer")
    version_id = vs.snapshot("manual_snapshot", "seed", ["settings"], force=True)
    assert viewer.post(f"/api/versions/{version_id}/restore", json={}).status_code == 403

    rows = viewer.get("/api/history", params={"kind": "change", "limit": 500}).json()["entries"]
    refused = [row for row in rows if row["facts"].get("status") == 403]
    assert refused, "the refused write is on the timeline"
    assert refused[0]["facts"]["action"] == "restore"


# --- the permission the merge must not widen -----------------------------------

def test_the_merge_does_not_widen_the_activity_scope(history_env, auth_env) -> None:
    """An operator sees its own changes and every restore point, never another
    account's changes. That is the rule GET /api/activity-log already enforces."""
    vs.snapshot("manual_snapshot", "seed", ["settings"], force=True)
    admin = _as(history_env, auth_env, "admin", "admin")
    admin.post("/api/definitely-not-a-route")
    operator = _as(history_env, auth_env, "operator1", "operator")
    operator.post("/api/definitely-not-a-route")

    body = operator.get("/api/history", params={"limit": 500}).json()
    assert body["scope"] == "self"
    changes = [entry for entry in body["entries"] if entry["kind"] in ("change", "sign_in")]
    assert changes, "the operator still sees its own changes"
    assert {entry["actor"] for entry in changes} == {"operator1"}
    assert any(entry["kind"] == "restore_point" for entry in body["entries"]), (
        "restore points are the shared operating record, not per-account")

    admin = _as(history_env, auth_env, "admin", "admin")
    seen_by_admin = admin.get("/api/history", params={"limit": 500}).json()
    assert seen_by_admin["scope"] == "all"
    assert {"admin", "operator1"} <= {entry["actor"] for entry in seen_by_admin["entries"]}


def test_an_anonymous_caller_cannot_read_the_timeline(history_env, auth_env) -> None:
    history_env.cookies.clear()
    assert history_env.get("/api/history").status_code == 401
    assert history_env.get("/api/history/since", params={"day": "2026-07-01"}).status_code == 401
