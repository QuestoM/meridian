"""What a control says it does, and what a refusal says, in the reader's language.

Three defects were measured on the shipped Today screen and all three are the
same class: a surface stating something it cannot deliver.

- The day panel offered "Open this day in the plan" while the call site passed a
  zero-argument function, so the date was discarded and the plan opened on its
  default day. The plan surface takes no date today, so the honest control names
  the place it reaches.
- The decision list promised that a row "opens the decision", and the surface it
  handed to no longer selects one. The row now opens the decision here, on the
  surface that has every part of it in hand.
- The target's refusal was printed exactly as the wall sent it, which is Hebrew,
  under an English paragraph, to the English-locale reader this destination is
  built for.

A fourth, found while fixing the third: the same segment read 80.5 percent in
the decision list and 81.0 in the rows behind it, because the drill rounded the
retention fraction to two places before turning it into a percentage.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import auth_store, target_store
from kairos_api.affiliation_wall import READ_ONLY_ROLE_DETAIL

ROOT = Path(__file__).resolve().parents[1]
TODAY = ROOT / "tv-break-dashboard" / "src" / "today"

VIEWER_PASSWORD = "viewerpass-123"
ADMIN_PASSWORD = "rootpass-1234"

HEBREW = re.compile(r"[֐-׿]")


def _source(name: str) -> str:
    return (TODAY / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def owned_channel():
    """Pin the operator's channel, as the sibling suites do, for one reason.

    ``data/kairos_settings.json`` is shared and other work edits it, so a
    scoped assertion that depended on it would pass or fail by accident.
    """
    from kairos_api import channel_scope, core, overview_api, overview_api_target
    from kairos_api.core import _load_break_schedule, _load_settings

    plan_channels = sorted(
        {str(name).strip() for name in _load_break_schedule()["channel"].astype(str).unique() if str(name).strip()}
    )
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


# ---------------------------------------------------------------------------
# The refusal, in the language of the person reading it
# ---------------------------------------------------------------------------


def test_the_refusal_carries_both_languages_and_the_hebrew_half_is_the_walls_own():
    english, hebrew = target_store.refusal_words(READ_ONLY_ROLE_DETAIL)
    assert hebrew == READ_ONLY_ROLE_DETAIL, "the Hebrew half must be the wall's own string, byte for byte"
    assert english and not HEBREW.search(english), "the English half must be English"


def test_a_refusal_this_map_does_not_know_is_repeated_and_never_invented():
    english, hebrew = target_store.refusal_words("סירוב שאיש לא רשם כאן")
    assert english == hebrew == "סירוב שאיש לא רשם כאן"


def test_the_payload_carries_both_halves_only_when_it_refuses(tmp_path, monkeypatch):
    """Measured on a real resolved session, not on a mocked gate."""
    from kairos_api import overview_api
    from kairos_api.auth import router as auth_router

    monkeypatch.setenv("KAIROS_AUTH_DIR", str(tmp_path / "auth"))
    monkeypatch.setenv(target_store.PATH_ENV, str(tmp_path / "plan_targets.csv"))
    monkeypatch.delenv("KAIROS_AUTH_DISABLED", raising=False)
    monkeypatch.delenv("KAIROS_ADMIN_PASSWORD", raising=False)
    auth_store.reset_runtime_state()
    try:
        app = FastAPI()
        app.include_router(auth_router)
        app.include_router(overview_api.router)

        auth_store.seed_initial_admin(password=ADMIN_PASSWORD)
        admin = TestClient(app)
        assert admin.post("/api/auth/login", json={"username": "admin", "password": ADMIN_PASSWORD}).status_code == 200
        created = admin.post("/api/auth/users", json={
            "username": "view2", "password": VIEWER_PASSWORD, "role": "viewer",
            "display_name": "view2", "must_change_password": False, "affiliation": "company",
        })
        assert created.status_code == 201, created.text

        viewer = TestClient(app)
        assert viewer.post("/api/auth/login", json={"username": "view2", "password": VIEWER_PASSWORD}).status_code == 200
        refused = viewer.get("/api/plan-target").json()
        assert refused["can_edit"] is False
        assert refused["can_edit_reason"] == READ_ONLY_ROLE_DETAIL
        assert refused["can_edit_reason_he"] == READ_ONLY_ROLE_DETAIL
        assert refused["can_edit_reason_en"] and not HEBREW.search(refused["can_edit_reason_en"])

        allowed = admin.get("/api/plan-target").json()
        assert allowed["can_edit"] is True
        assert "can_edit_reason" not in allowed
        assert "can_edit_reason_en" not in allowed and "can_edit_reason_he" not in allowed
    finally:
        auth_store.reset_runtime_state()


def test_the_surface_prints_the_half_its_reader_can_read():
    source = _source("TodayMoney.jsx")
    assert "String(target.can_edit_reason || '')" not in source, "the raw wall string is being printed again"
    assert source.count("{refusalText(target, locale)}") == 2, "both refusal sites must go through the locale pair"
    assert "can_edit_reason_en" in source and "can_edit_reason_he" in source


# ---------------------------------------------------------------------------
# One quantity, one reading
# ---------------------------------------------------------------------------


def test_a_segment_reads_the_same_retention_in_the_list_and_in_the_rows_behind_it(client):
    body = client.get("/api/today").json()
    items = [item for item in body["decisions"]["items"] if item.get("segment_id") and item.get("date")]
    assert items, "no decision carries a segment, so this guard would pass vacuously"
    checked = 0
    for item in items:
        day = client.get(f"/api/today/day/{item['date']}").json()
        rows = {row["segment_id"]: row for row in day.get("rows", [])}
        row = rows.get(item["segment_id"])
        assert row is not None, f"the plan rows for {item['date']} carry no row for {item['segment_id']}"
        assert row["projected_revenue"] == item["impact"], "the same segment must carry one revenue"
        assert row["retention_percent"] == item["retention"], "the same segment must carry one retention"
        checked += 1
    assert checked >= 1


def test_the_drill_rounds_the_retention_once_and_not_twice():
    from kairos_api import overview_api_drill

    assert overview_api_drill._fraction(0.8054) == pytest.approx(0.8054)
    assert round(overview_api_drill._fraction(0.8054) * 100, 1) == 80.5


# ---------------------------------------------------------------------------
# A control names what it opens
# ---------------------------------------------------------------------------


def test_the_day_panel_names_the_place_it_reaches_and_not_a_day_it_cannot_open():
    panel = _source("TodayDayDetail.jsx")
    assert "Open this day in the plan" not in panel
    assert "'Open the plan'" in panel
    assert "onOpenInPlan" not in panel, "the discarded-date prop name is gone with the promise"
    page = _source("OverviewPage.jsx")
    assert "onOpenDay={() =>" not in page, "a zero-argument arrow behind a control that named a day"
    assert "onOpenPlan={openPlan}" in page


def test_the_decision_row_opens_the_decision_on_this_screen():
    panel = _source("TodayDecisions.jsx")
    assert "import TodayDecisionDetail from './TodayDecisionDetail';" in panel
    assert "<TodayDecisionDetail" in panel
    assert "and opens the decision." not in panel, "the promise the optimizer no longer keeps"
    assert "opens the plan row behind it here, without leaving this screen" in panel
    detail = _source("TodayDecisionDetail.jsx")
    assert "fetchTodayDay" in detail, "the rows behind a decision are read, not asserted"
    assert "'Open the optimizer'" in detail, "the path the product already had survives the change"
    assert "money === listedMoney && retention === listedRetention" in detail, "the two reads are compared, not assumed"


def test_every_change_the_engine_can_propose_has_words_on_the_surface():
    """A kind added on one side and not the other would open a blank line."""
    from kairos_api import overview_api

    kinds = set()
    for risk in ("High", "Medium", "Low"):
        for breaks in (0, 1, 2, 4):
            for gold in (False, True):
                kind = overview_api._proposed_kind(risk, breaks, gold)
                if kind is not None:
                    kinds.add(kind)
    detail = _source("TodayDecisionDetail.jsx")
    named = set(re.findall(r"^  (\w+): \[", detail, flags=re.M))
    assert kinds == {"lower_count", "forbid", "pin", "gold"}, kinds
    assert named == kinds, f"the surface names {named} and the engine proposes {kinds}"


def test_the_four_figures_sentence_waits_for_the_read_that_makes_it_true():
    page = _source("OverviewPage.jsx")
    assert "summaryWithheld" not in page, "the sentence was gated on the refusal alone"
    assert "const summaryAttributed = attributed(overviewScope(overview));" in page
    assert "{summaryAttributed ? (" in page


# ---------------------------------------------------------------------------
# The words the campaign retired
# ---------------------------------------------------------------------------

RETIRED = ("recompute", "rebuild", "חישוב מחדש", "בנייה מחדש")


def test_no_retired_word_survives_anywhere_on_this_destination():
    """The cross-cutting check is a grep, and a comment fails it like a label.

    One did: a comment in ``TodayMoney.jsx`` explained why a target keyed to one
    span is not read as another's and used the retired word for a plan run.
    """
    offences = []
    for path in sorted(TODAY.rglob("*")):
        if not path.is_file():
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            lowered = line.lower()
            for word in RETIRED:
                if word in lowered:
                    offences.append(f"{path.name}:{number}: {word}")
    assert offences == [], offences
