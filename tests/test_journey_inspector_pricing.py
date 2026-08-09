"""Journey tests: the manual decision loop, the pricing loop, and uploads.

Journey 3 (manual decision loop): click a programme in the schedule editor,
inspect it (GET /api/schedule/segment/{id} must agree with the committed CSV
row exactly), preview an override's effect (honest with-vs-without delta,
nothing written), and download the plan (the export must be the saved CSV).

Journey 4 (pricing): the rate-card state marks which layers are live, the
price-slot tester's breakdown multiplies out exactly, a saved edit moves the
tester's answer, and reset restores the shipped card. Settings writes are
redirected to a temporary file so the real rate card never moves.

Journey 5 (uploads): the status endpoint reports honest in_use semantics, in
particular the amber stored-but-shadowed state while the reference xlsx files
are the live inputs.
"""

from __future__ import annotations

import io
import json
import math
import shutil
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

import kairos_api.core as core

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"
OVERRIDES_PATH = ROOT / "data" / "manual_overrides.csv"


@pytest.fixture(scope="module")
def client() -> TestClient:
    from kairos_api.server import app

    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def plan() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH, encoding="utf-8")


@pytest.fixture(scope="module")
def owned_channel() -> str:
    settings = json.loads((ROOT / "data" / "kairos_settings.json").read_text(encoding="utf-8"))
    return str(settings["operator_channel"]).strip()


def _percent(value: float) -> float:
    return round(value * 100, 2)


def test_inspector_agrees_with_committed_rows(client, plan, owned_channel):
    """Journey 3, step 2: the inspector payload for a segment must equal the
    committed CSV row field for field (identity, plan, economics, retention).
    Sampled across zero-break, mid, and max-break rows on the owned channel."""
    owned = plan[plan["channel"] == owned_channel]
    sample = pd.concat([
        owned[owned["num_breaks"] == 0].head(3),
        owned[owned["num_breaks"] == 1].head(3),
        owned[owned["num_breaks"] == owned["num_breaks"].max()].head(3),
    ])
    assert len(sample) >= 6, "not enough owned-channel rows to sample"
    for row in sample.itertuples(index=False):
        response = client.get(f"/api/schedule/segment/{row.segment_id}")
        assert response.status_code == 200, f"{row.segment_id}: {response.text[:120]}"
        detail = response.json()
        assert detail["identity"]["channel"] == row.channel
        assert detail["identity"]["date"] == row.date
        assert detail["identity"]["program_type"] == row.program_type
        assert detail["identity"]["start_clock"] == row.start_time
        assert detail["plan"]["num_breaks"] == int(row.num_breaks)
        assert detail["plan"]["break_length_seconds"] == pytest.approx(row.break_length)
        assert detail["plan"]["total_break_seconds"] == pytest.approx(row.total_break_time)
        assert detail["plan"]["is_gold"] == bool(row.is_gold)
        assert detail["economics"]["predicted_revenue"] == pytest.approx(row.predicted_revenue)
        assert detail["economics"]["base_rate"] == pytest.approx(row.base_rate)
        assert detail["economics"]["baseline_tvr"] == pytest.approx(row.baseline_tvr)
        assert detail["retention"]["predicted_retention"] == pytest.approx(
            _percent(row.predicted_retention)
        )
        assert detail["retention"]["retention_used"] == pytest.approx(
            _percent(row.retention_used)
        )
        anchor = detail["anchor"]
        assert anchor == {
            "date": row.date, "start_clock": row.start_time, "program": row.program_type,
        }


def test_inspector_enforces_competitor_boundary(client, plan, owned_channel):
    """Journey 3: a competitor channel's segment returns 404, never a payload."""
    competitor = plan[plan["channel"] != owned_channel].iloc[0]
    response = client.get(f"/api/schedule/segment/{competitor.segment_id}")
    assert response.status_code == 404
    assert client.get("/api/schedule/segment/not-a-real-id").status_code == 404


def test_schedule_segments_lists_owned_channel_only(client, plan, owned_channel):
    """The editor's target list is scoped to the owned channel with anchors."""
    body = client.get("/api/schedule/segments").json()
    assert body["operator_channel"] == owned_channel
    segments = body["segments"]
    owned_rows = plan[plan["channel"] == owned_channel]
    assert len(segments) == len(owned_rows)
    assert {s["segment_id"] for s in segments} == set(owned_rows["segment_id"])
    first = segments[0]
    assert set(first["anchor"]) == {"date", "start_clock", "program"}
    assert set(first["state"]) >= {"num_breaks", "is_gold", "predicted_revenue", "retention"}


def test_effect_preview_forbid_delta_and_no_write(client, plan, owned_channel):
    """Journey 3, step 3: the candidate effect preview isolates one decision.
    Forbidding a k-break segment must lower the day's total by exactly the
    breaks the engine can no longer keep there or place elsewhere is allowed;
    the honest minimum contract asserted: the target segment goes to 0, totals
    change by a real recomputed amount, and NOTHING is written to the store."""
    day_rows = plan[(plan["channel"] == owned_channel) & (plan["date"] == "2024-11-01")]
    target = day_rows[day_rows["num_breaks"] >= 1].iloc[0]
    store_before = OVERRIDES_PATH.read_bytes()

    response = client.get("/api/overrides/effect", params={
        "target_id": target.segment_id, "kind": "forbid",
    })
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["candidate"] == {
        "target_id": target.segment_id, "kind": "forbid", "value": "",
    }
    changed = {c["segment_id"]: c for c in body["changed"]}
    assert target.segment_id in changed, "the forbidden segment must appear in the delta"
    assert changed[target.segment_id]["after"] == 0
    assert changed[target.segment_id]["before"] == int(target.num_breaks)
    summary = body["summary"]
    assert summary["before_total_breaks"] >= summary["after_total_breaks"] >= 0
    assert isinstance(summary["before_revenue"], float)
    assert OVERRIDES_PATH.read_bytes() == store_before, "preview must write nothing"


def test_effect_preview_baseline_matches_committed_day(client, plan, owned_channel):
    """The preview's WITH-stored-overrides baseline must reproduce the committed
    day (same optimizer, same inputs, empty stores): equal break totals and
    revenue within row-rounding tolerance."""
    day_rows = plan[(plan["channel"] == owned_channel) & (plan["date"] == "2024-11-01")]
    target = day_rows[day_rows["num_breaks"] >= 1].iloc[0]
    response = client.get("/api/overrides/effect", params={
        "target_id": target.segment_id, "kind": "gold",
    })
    assert response.status_code == 200, response.text
    summary = response.json()["summary"]
    assert summary["before_total_breaks"] == int(day_rows["num_breaks"].sum())
    assert summary["before_revenue"] == pytest.approx(
        day_rows["predicted_revenue"].sum(), abs=0.05
    )


def test_export_download_is_the_saved_plan(client, plan):
    """Journey 3, step 5: the download equals the operator's slice of the plan.

    It said "equals the saved weekly plan" until ruling 009. The route serves the
    operator's own channel now, because a download of a rival's programme titles
    and revenue is the same breach as printing them on a screen. So the journey's
    step is unchanged in what it is FOR, a planner downloading what they just
    saw, and the comparison is against their own rows rather than the file.
    """
    response = client.get("/api/export/schedule.csv")
    assert response.status_code == 200
    assert response.headers["content-disposition"].startswith("attachment")
    exported = pd.read_csv(io.StringIO(response.text))
    assert list(exported.columns) == list(plan.columns)
    from kairos_api import channel_scope

    owned = str(channel_scope.operator_channel() or "").strip()
    assert owned, "no operator channel is configured, so the route would refuse"
    mine = plan[plan["channel"].astype(str).str.strip() == owned]
    assert len(exported) == len(mine)
    assert exported["segment_id"].tolist() == mine["segment_id"].tolist()
    assert set(exported["channel"].astype(str).str.strip()) == {owned}, (
        "the download carried a channel this operator does not own"
    )
    pd.testing.assert_series_equal(
        exported["num_breaks"].reset_index(drop=True),
        mine["num_breaks"].reset_index(drop=True),
    )
    assert exported["predicted_revenue"].sum() == pytest.approx(
        mine["predicted_revenue"].sum()
    )


@pytest.fixture()
def tmp_settings(tmp_path, monkeypatch) -> Path:
    target = tmp_path / "kairos_settings.json"
    shutil.copy(ROOT / "data" / "kairos_settings.json", target)
    monkeypatch.setattr(core, "SETTINGS_PATH", target)
    return target


def test_pricing_state_declares_live_and_wired_off_layers(client):
    """Journey 4: the rate-card state must mark program and day live and ship
    show, position and ad_type with activation OFF (their multipliers are not
    1.0, so activating them moves real revenue). The zero-multiplier hazard on
    the ad_type layer must be disclosed as a structured warning."""
    body = client.get("/api/pricing").json()
    layers = {layer["name"]: layer for layer in body["layers"]}
    assert layers["program"]["live_today"] is True
    assert layers["day"]["live_today"] is True
    for name in ("show", "position", "ad_type"):
        assert layers[name]["live_today"] is False, f"{name} must ship wired off"
    assert body["activation"] == {"show": False, "position": False, "ad_type": False}
    ad_type_warnings = layers["ad_type"]["warnings"]
    assert any(w["kind"] == "zeroes_on_activation" for w in ad_type_warnings), (
        "the promo=0 activation hazard must be disclosed"
    )


def test_price_slot_breakdown_multiplies_out(client):
    """Journey 4: the tester's final price must equal base times the product of
    the live layers it itself reports (Law 9: every number traces)."""
    response = client.post("/api/pricing/price-slot", json={
        "pricing_class": "News", "weekday_iso": 5,
    })
    assert response.status_code == 200, response.text
    body = response.json()
    product = body["base_cpp"]
    for layer in body["layers"]:
        product *= layer["multiplier"]
    assert body["final_cpp"] == pytest.approx(product, rel=1e-9)
    assert math.isfinite(body["final_cpp"]) and body["final_cpp"] > 0


def test_pricing_edit_moves_the_tester_and_reset_restores(client, tmp_settings):
    """Journey 4, full loop: read the current slot price, save a base-rate edit,
    the tester must move by exactly the ratio, then reset restores the shipped
    card. All writes land in the temporary settings copy."""
    before = client.post("/api/pricing/price-slot", json={
        "pricing_class": "Other", "weekday_iso": 3,
    }).json()
    base_before = before["base_cpp"]

    response = client.put("/api/pricing", json={
        "overrides": {"base_price_per_second_per_tvr_point": base_before * 2},
    })
    assert response.status_code == 200, response.text
    assert response.json()["has_overrides"] is True
    persisted = json.loads(tmp_settings.read_text(encoding="utf-8"))
    assert persisted["pricing_overrides"]["base_price_per_second_per_tvr_point"] == (
        base_before * 2
    )

    after = client.post("/api/pricing/price-slot", json={
        "pricing_class": "Other", "weekday_iso": 3,
    }).json()
    assert after["base_cpp"] == pytest.approx(base_before * 2)
    assert after["final_cpp"] == pytest.approx(before["final_cpp"] * 2, rel=1e-9)

    reset = client.put("/api/pricing", json={"reset": True})
    assert reset.status_code == 200
    assert reset.json()["has_overrides"] is False
    restored = client.post("/api/pricing/price-slot", json={
        "pricing_class": "Other", "weekday_iso": 3,
    }).json()
    assert restored["base_cpp"] == pytest.approx(base_before)


def test_pricing_rejects_invalid_edit(client, tmp_settings):
    """A negative premium must be rejected with 422 and change nothing on disk.

    Compared against the pre-request bytes, not against an empty overrides map:
    the seeded settings mirror the operator's real file, which may legitimately
    carry saved pricing edits already."""
    before = tmp_settings.read_bytes()
    response = client.put("/api/pricing", json={
        "overrides": {"base_price_per_second_per_tvr_point": -5},
    })
    assert response.status_code == 422, response.text
    assert tmp_settings.read_bytes() == before, "a rejected edit must not touch the settings file"


def test_uploads_status_reports_honest_in_use(client):
    """Journey 5: while the reference xlsx files are the live engine inputs,
    stored CSV uploads must read in_use=False with a real reason (the amber
    stored-not-used state), the daily Wally file must read in_use=True, and the
    rate card must disclose that no engine code reads it."""
    body = client.get("/api/uploads/status").json()
    inputs = {item["kind"]: item for item in body["inputs"]}
    for kind in ("programmes", "spots", "dayparts"):
        reference = ROOT / "data" / "reference" / f"{kind.capitalize()}.xlsx"
        if reference.exists() and inputs[kind]["exists"]:
            assert inputs[kind]["in_use"] is False, f"{kind} shadowed upload must not claim in_use"
            assert inputs[kind]["in_use_reason"], f"{kind} needs an honest reason"
    if inputs["daily"]["exists"] and inputs["daily"]["valid"]:
        assert inputs["daily"]["in_use"] is True
    if inputs["rate_card"]["exists"]:
        assert inputs["rate_card"]["in_use"] is False
        assert "optimization_weights.yaml" in str(inputs["rate_card"]["in_use_reason"])
