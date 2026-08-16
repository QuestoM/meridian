"""GET /api/today: the three answers, and the laws they are answered under.

The surface has a five-second bar with zero clicks in it, so the answers arrive
in one round trip. That makes this payload the whole of what a general manager
reads before touching anything, and every law that applies to a screen applies
to it: honest math, the competitor boundary, the training line, and Bar 3's
promise that the five priority decisions still carry the same figures.

The router is mounted alone rather than through the whole server app, so this
suite measures P1's surface and nothing else, and it keeps answering while
other pieces are mid-edit.
"""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import auth_store, target_store

# Section 4.2's lexicon test: a run surface returns zero hits for any of these.
TRAINING_LEXICON = (
    "gate",
    "held_out",
    "tau",
    "drift",
    "coefficient",
    "pooling",
    "p_value",
    "training_window",
    "wartime",
)

VIEWER_PASSWORD = "viewerpass-123"
ADMIN_PASSWORD = "rootpass-1234"


@pytest.fixture(scope="module")
def owned_channel():
    """Pin the operator's channel for the scoped assertions.

    ``data/kairos_settings.json`` is a shared store that other work edits, and a
    channel that is declared on one run and empty on the next would make every
    boundary assertion below pass or fail by accident. It is pinned in the two
    places that read it, the settings the payload composes from and the helper
    the boundary scopes with, so the whole surface agrees on one channel. The
    undeclared state has its own tests further down.
    """
    from kairos_api import channel_scope, core, overview_api, overview_api_target
    from kairos_api.core import _load_break_schedule, _load_settings

    plan_channels = sorted(
        {str(name).strip() for name in _load_break_schedule()["channel"].astype(str).unique() if str(name).strip()}
    )
    assert len(plan_channels) > 1, "the reference plan must carry a lineup for the boundary to be testable"
    chosen = "רשת 13" if "רשת 13" in plan_channels else plan_channels[0]
    pinned = _load_settings().model_copy(update={"operator_channel": chosen})

    saved = (channel_scope.operator_channel, core._load_settings, overview_api._load_settings, overview_api_target._load_settings)
    channel_scope.operator_channel = lambda settings=None: chosen
    core._load_settings = lambda: pinned
    overview_api._load_settings = lambda: pinned
    overview_api_target._load_settings = lambda: pinned
    overview_api._overview_cached.cache_clear()
    yield chosen
    channel_scope.operator_channel, core._load_settings, overview_api._load_settings, overview_api_target._load_settings = saved
    overview_api._overview_cached.cache_clear()


@pytest.fixture(scope="module")
def client(owned_channel) -> TestClient:
    from kairos_api import overview_api

    app = FastAPI()
    app.include_router(overview_api.router)
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def today(client) -> dict:
    response = client.get("/api/today")
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture()
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setenv(target_store.PATH_ENV, str(tmp_path / "plan_targets.csv"))
    return tmp_path


# ---------------------------------------------------------------------------
# The three answers
# ---------------------------------------------------------------------------


def test_the_payload_carries_all_three_answers(today):
    assert set(today).issuperset({"money", "target", "verdict", "health", "decisions", "window"})
    assert today["money"]["available"] is True
    assert today["health"]["state"] in {"clear", "attention"}
    assert today["decisions"]["count"] >= 1


def test_the_money_figure_prints_its_scope_with_it(today):
    scope = today["money"]["scope"]
    assert scope["channel"]
    assert scope["date_from"] and scope["date_to"]
    assert scope["n_dates"] >= 1
    assert scope["source"] == "saved_plan"
    assert today["money"]["currency"] == "ILS"
    assert today["money"]["metric"] == "projected_revenue"


def test_the_window_is_the_same_slice_the_overview_summary_reports(client, today):
    summary = client.get("/api/overview").json()["summary"]
    week = summary["week"]
    assert today["window"]["date_from"] == week["date_from"]
    assert today["window"]["date_to"] == week["date_to"]
    assert today["money"]["amount_ils"] == week["projected_revenue"]


def test_the_days_behind_the_figure_sum_back_to_it(today):
    """Stripe's drill is only honest when the rows add up to the amount."""
    money = today["money"]
    days = money["days"]
    assert len(days) == money["scope"]["n_dates"]
    assert money["days_total_ils"] == pytest.approx(money["amount_ils"], abs=0.5)
    assert money["reconciled"] is True
    assert abs(money["residual_ils"]) < 0.5


def test_every_day_row_carries_its_weekday_and_the_israeli_weekend(today):
    days = today["money"]["days"]
    assert all(day["weekday_en"] and day["weekday_he"] for day in days)
    weekend = {day["weekday_en"] for day in days if day["is_weekend"]}
    assert weekend <= {"Fri", "Sat"}
    assert not any(day["is_weekend"] for day in days if day["weekday_en"] == "Sun")


def test_the_decisions_are_the_same_five_with_the_same_figures(client, today):
    """Bar 3, P1: the priority decisions keep their identity and their money."""
    recommendations = client.get("/api/overview").json()["recommendations"]
    items = today["decisions"]["items"]
    assert len(items) == len(recommendations)
    assert [item["id"] for item in items] == [rec["id"] for rec in recommendations]
    assert [item["impact"] for item in items] == [rec["impact"] for rec in recommendations]
    assert [item["retention"] for item in items] == [rec["retention"] for rec in recommendations]
    assert [item["title"] for item in items] == [rec["title"] for rec in recommendations]
    assert [item["title_he"] for item in items] == [rec["title_he"] for rec in recommendations]


def test_the_decision_list_states_what_it_is_ranked_by(today):
    assert today["decisions"]["ranked_by"] == "projected_revenue"


def test_the_stated_span_is_withheld_when_the_summary_was_scoped_to_somebody_else():
    """A span read off another channel's rows is not this operator's span.

    The same rule the money figure is withheld under: two independent reads
    have to name the same channel before anything derived from them is printed
    as this surface's.
    """
    from kairos_api import overview_api_today

    body = {"summary": {"scope_channel": "A", "date_from": "2024-11-01", "date_to": "2024-11-30", "n_dates": 30}}
    mine = overview_api_today.decisions_scope(body, "A")
    assert (mine["channel"], mine["date_from"], mine["date_to"], mine["n_dates"]) == ("A", "2024-11-01", "2024-11-30", 30)
    theirs = overview_api_today.decisions_scope(body, "B")
    assert (theirs["channel"], theirs["date_from"], theirs["date_to"], theirs["n_dates"]) == (None, None, None, 0)
    none_declared = overview_api_today.decisions_scope(body, "")
    assert (none_declared["channel"], none_declared["date_from"], none_declared["date_to"]) == (None, None, None)


def test_the_decision_list_states_the_span_it_was_drawn_from(client, today):
    """A ranking scans the whole saved plan, so it says which span that is.

    The rows are routinely dated outside the seven-day window the money answer
    above them names, and on the reference data every one of them is. Without
    the span the list came from, a reader meets five dated rows under a window
    that excludes them and reads the product as broken.
    """
    summary = client.get("/api/overview").json()["summary"]
    scope = today["decisions"]["scope"]
    assert scope["channel"] == summary["scope_channel"] == today["money"]["scope"]["channel"]
    assert scope["date_from"] == summary["date_from"]
    assert scope["date_to"] == summary["date_to"]
    assert scope["n_dates"] == summary["n_dates"]
    assert scope["source"] == "saved_plan"
    dates = [str(item["date"]) for item in today["decisions"]["items"] if item.get("date")]
    assert dates, "the ranked rows carry their dates, which is what the span explains"
    assert all(scope["date_from"] <= date <= scope["date_to"] for date in dates)
    outside = [date for date in dates if not (today["window"]["date_from"] <= date <= today["window"]["date_to"])]
    if not outside:
        pytest.skip("every ranked row falls inside the money window on this plan, so the two spans agree")
    assert scope["date_to"] > today["window"]["date_to"]


def test_a_target_on_another_window_is_named_rather_than_silently_dropped(client, isolated_store):
    """A number a person supplied does not vanish because the window moved.

    The store refuses to read one span's target as another's, which is the
    scope rule. Refusing to read it is not licence to stop naming it: the
    payload carries the other windows with their spans and their amounts, so
    the surface can print what exists instead of reporting an absence.
    """
    window = client.get("/api/today").json()["window"]
    saved = client.put(
        "/api/plan-target",
        json={
            "amount_ils": 9_500_000,
            "at_risk_band_percent": 4,
            "period_start": "2024-11-08",
            "period_end": "2024-11-14",
        },
    )
    assert saved.status_code == 200, saved.text
    assert window["date_from"] != "2024-11-08"
    target = client.get("/api/today").json()["target"]
    assert target["state"] == "unset"
    assert target["amount_ils"] is None
    others = target["other_windows"]
    assert [(row["period_start"], row["period_end"], row["amount_ils"]) for row in others] == [
        ("2024-11-08", "2024-11-14", 9_500_000.0)
    ]
    assert others[0]["set_by"]
    # And it stays named once this window has a target of its own, so the two
    # are read side by side rather than one replacing the other.
    client.put("/api/plan-target", json={"amount_ils": 10_500_000, "at_risk_band_percent": 5})
    with_target = client.get("/api/today").json()["target"]
    assert with_target["state"] == "set"
    assert [row["period_start"] for row in with_target["other_windows"]] == ["2024-11-08"]


def test_the_health_answer_is_computed_from_real_fields(client, today):
    body = client.get("/api/overview").json()
    checks = {check["id"]: check for check in today["health"]["checks"]}
    assert "licence" in checks and "inputs" in checks
    assert checks["licence"]["checks_total"] == len(body["compliance"]["checks"])
    assert checks["inputs"]["programmes"] == body["source_counts"]["programmes"]
    assert checks["inputs"]["spots"] == body["source_counts"]["spots"]
    assert checks["inputs"]["planned_break_rows"] == body["source_counts"]["planned_break_rows"]


def test_a_stale_plan_splits_into_the_two_things_it_actually_means(client, today):
    """One banner today fuses a change you made with a model somebody trained."""
    # Earlier tests in this module can move freshness inputs. Read both sides of
    # this invariant now rather than comparing a module-scoped Today snapshot
    # with a later overview verdict.
    current_today = client.get("/api/today").json()
    freshness = client.get("/api/overview").json()["schedule_freshness"]
    ids = {check["id"] for check in current_today["health"]["checks"]}
    if freshness["status"] != "stale":
        pytest.skip("the saved plan is not stale on this tree, so there is nothing to split")
    model_changed = any(
        str(group) in {"coefficients", "the impact model", "the audience model"}
        for group in freshness["changed"]
    )
    operator_changed = any(
        str(group) not in {"coefficients", "the impact model", "the audience model"}
        for group in freshness["changed"]
    )
    assert ("newer_model_version" in ids) is model_changed
    assert ("plan_out_of_date" in ids) is operator_changed


# ---------------------------------------------------------------------------
# The laws
# ---------------------------------------------------------------------------


def test_no_training_word_reaches_this_run_surface(client):
    """Section 4.2's lexicon test, run on the payload Today actually reads."""
    body = client.get("/api/today").text
    hits = {word: len(re.findall(word, body, re.I)) for word in TRAINING_LEXICON}
    assert not {word: count for word, count in hits.items() if count}


def test_the_model_reaches_this_surface_as_a_date_and_nothing_else(today):
    stamp = today["model_trained_at"]
    assert stamp is None or re.match(r"^\d{4}-\d{2}-\d{2}", str(stamp))


def test_no_rival_channel_name_or_row_reaches_the_payload(client, today):
    """The competitor boundary, checked by name against the loaded lineup."""
    from kairos_api import channel_scope
    from kairos_api.core import _load_break_schedule

    owned = today["money"]["scope"]["channel"]
    everyone = {
        str(name).strip()
        for name in _load_break_schedule()["channel"].astype(str).unique()
        if str(name).strip()
    }
    rivals = everyone - {owned}
    assert rivals, "the reference data carries rival channels, so this test can bite"
    body = client.get("/api/today").text
    for rival in rivals:
        assert rival not in body
    note = today["money"]["boundary"]
    assert note["scope_channel"] == owned
    assert note["scoped"] is True
    assert note["competitor_rows_excluded"] > 0
    assert note["competitor_channels_excluded"] == len(rivals)
    assert channel_scope.NO_OPERATOR_CHANNEL_REASON not in body


def test_an_absent_target_is_an_absence_and_never_a_number(client, isolated_store):
    body = client.get("/api/today").json()
    assert body["target"]["state"] == "unset"
    assert body["target"]["amount_ils"] is None
    assert body["verdict"]["state"] == "unavailable"
    assert body["verdict"]["reason"] == "no_target"
    assert body["verdict"]["variance_ils"] is None


def test_the_verdict_becomes_real_only_when_a_person_supplies_the_number(client, isolated_store):
    first = client.get("/api/today").json()
    amount = first["money"]["amount_ils"] * 0.9
    saved = client.put(
        "/api/plan-target",
        json={"amount_ils": amount, "at_risk_band_percent": 5, "note": "set by the test"},
    )
    assert saved.status_code == 200, saved.text
    body = client.get("/api/today").json()
    assert body["target"]["state"] == "set"
    assert body["target"]["amount_ils"] == pytest.approx(round(amount, 2))
    assert body["verdict"]["state"] == "on_plan"
    assert body["verdict"]["variance_ils"] == pytest.approx(round(first["money"]["amount_ils"] - round(amount, 2), 2))
    assert "percent" in body["verdict"]["threshold_en"]
    cleared = client.delete("/api/plan-target")
    assert cleared.status_code == 200
    assert client.get("/api/today").json()["verdict"]["state"] == "unavailable"


def test_the_target_is_keyed_to_the_window_it_measures(client, isolated_store):
    client.put(
        "/api/plan-target",
        json={
            "amount_ils": 1_000_000,
            "at_risk_band_percent": 5,
            "period_start": "2030-01-06",
            "period_end": "2030-01-12",
        },
    )
    body = client.get("/api/today").json()
    assert body["target"]["state"] == "unset"
    assert body["verdict"]["state"] == "unavailable"
    other = client.get("/api/plan-target", params={"period_start": "2030-01-06", "period_end": "2030-01-12"})
    assert other.json()["state"] == "set"


def test_the_target_read_says_before_the_click_whether_it_can_be_changed(client, isolated_store):
    body = client.get("/api/plan-target").json()
    assert "can_edit" in body


def test_a_refused_target_never_lands(client, isolated_store):
    refused = client.put("/api/plan-target", json={"amount_ils": -5, "at_risk_band_percent": 5})
    assert refused.status_code == 400
    assert client.get("/api/today").json()["target"]["state"] == "unset"


# ---------------------------------------------------------------------------
# The wall on the write, on a real resolved session
# ---------------------------------------------------------------------------


@pytest.fixture()
def auth_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.setenv(target_store.PATH_ENV, str(tmp_path / "plan_targets.csv"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    yield tmp_path
    auth_store.reset_runtime_state()


def test_a_viewer_reads_the_target_and_is_refused_the_write(auth_env):
    """Affiliation decides seeing, role decides changing, on this surface too."""
    from kairos_api import overview_api
    from kairos_api.auth import router as auth_router

    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(overview_api.router)

    auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
    admin = TestClient(app)
    signed = admin.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD})
    assert signed.status_code == 200, signed.text
    created = admin.post("/api/auth/users", json={
        "username": "view1", "password": VIEWER_PASSWORD, "role": "viewer",
        "display_name": "view1", "must_change_password": False, "affiliation": "company",
    })
    assert created.status_code == 201, created.text

    viewer = TestClient(app)
    logged_in = viewer.post("/api/auth/login", json={"username": "view1", "password": VIEWER_PASSWORD})
    assert logged_in.status_code == 200, logged_in.text

    read = viewer.get("/api/plan-target")
    assert read.status_code == 200
    assert read.json()["can_edit"] is False
    assert read.json()["can_edit_reason"] == target_store.TARGET_WALL.role_detail

    write = viewer.put("/api/plan-target", json={"amount_ils": 9_500_000, "at_risk_band_percent": 5})
    assert write.status_code == 403
    assert write.json()["detail"] == target_store.TARGET_WALL.role_detail
    assert target_store.read_all() == []

    allowed = admin.put("/api/plan-target", json={"amount_ils": 9_500_000, "at_risk_band_percent": 5})
    assert allowed.status_code == 200, allowed.text
    assert allowed.json()["can_edit"] is True
    assert allowed.json()["set_by"] == "admin"


def test_the_shared_overview_payload_keeps_exactly_the_keys_it_had(client):
    """Bar 3: Today added a surface, it did not reshape the payload others read."""
    body = client.get("/api/overview").json()
    assert set(body) == {
        "brand", "workspace", "data_freshness", "summary", "source_counts",
        "recommendations", "frontier_scope", "settings", "compliance", "frontier",
        "frontier_status", "frontier_net_point", "frontier_basis", "schedule_freshness",
    }
    assert len(body["recommendations"]) == 5
    assert all("candidates" in item for item in body["recommendations"])
