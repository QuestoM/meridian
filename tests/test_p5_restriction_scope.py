"""P5: a restriction may not remove a break its own sentence did not ask for.

The defect this file exists for was measured on the running product. The rule
"no breaks in the last 8 minutes of משחקי השף עונה 7 ש.ח" matches 43 airings, of
which exactly 7 breach it, and the compiler emitted 7 rows whose predicate named
a programme and a date. A programme and a date do not name one airing: those 7
rows bound 17 airings, 10 of which the compiler itself had judged compliant, and
the engine's resolver emptied every one. 38 breaks went, 31 of them off airings
nobody had asked about, and of the 470,562.01 ILS the panel reported, 404,538.45
was that overreach. The screen never said so.

The fix is one condition. The frozen predicate contract carries an ``hour`` field
read as ``int(start_seconds // 3600) % 24``, so a compiled row pins the
programme, the date and the hour, which is the finest scope the contract has.
Two airings of one programme can still start inside one clock hour, so the
surplus is not assumed away: the preview counts it, prices it on its own, and
marks every change row with whether the sentence asked for it.

These tests pin that, against the real plan of record and the engine's own
resolver rather than against a fixture of what the compiler believes.
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
# The programme the overreach was measured on: many airings, many broadcast days,
# and only a handful of them long enough to breach an eight minute tail.
WIDE_TITLE = "משחקי השף עונה 7 ש.ח"
TAIL_MINUTES = 8

# The one place on the real data where the hour is still not enough. On
# 2024-11-04 three airings of the promo block start inside hour 22, one breaches a
# five minute tail and one is already compliant and carries a break, so a rule
# derived from the first reaches the second. It is the case the surplus report
# exists for, and it is real rather than constructed.
SHARED_HOUR_TITLE = "קובץ פרומו/פרסומות"
SHARED_HOUR_DAY = "2024-11-04"
SHARED_HOUR = 22


@pytest.fixture(autouse=True)
def relocated(tmp_path, monkeypatch):
    import shutil

    import kairos_api.core as core

    monkeypatch.setenv(vs.VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.delenv(vs.ASSISTANT_DIR_ENV, raising=False)
    monkeypatch.setattr(constraints_api, "CONSTRAINTS_PATH", tmp_path / "kairos_constraints.csv")
    monkeypatch.setattr(constraints_api, "BACKUP_DIR", tmp_path / "_backups")
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


def _where(*conditions: dict) -> dict:
    return {"combinator": "and", "conditions": list(conditions)}


def _title(value: str) -> dict:
    return {"field": "programme", "operator": "is", "value": value}


def _preview(client: TestClient, kind: str, params: dict, where: dict) -> dict:
    response = client.post("/api/constraints/restrictions/preview", json={
        "kind": kind, "params": params, "where": where,
    })
    assert response.status_code == 200, response.text
    return response.json()


def _breaching(title: str, minutes: int) -> list:
    """The airings that really do breach the sentence, from the plan of record."""
    from kairos_api.constraints_airings import matching
    from kairos_api.constraints_language import max_breaks_before_tail

    protected = float(minutes) * 60.0
    return [
        airing for airing in matching(_where(_title(title)))
        if airing.planned_breaks is not None
        and airing.planned_breaks > max_breaks_before_tail(
            airing.duration_seconds, airing.break_length_seconds, protected,
        )
    ]


# ---------------------------------------------------------------------------
# 1. The rule binds what its sentence asks for, and the money follows.


def test_a_window_rule_binds_exactly_the_airings_that_breach_its_own_sentence(client):
    """The measured defect, asserted as a number rather than as a wording.

    Before the hour was pinned this rule bound 17 airings and took 38 breaks
    against 7 that breach. The assertion is equality with the breaching count,
    not an inequality, because anything above it is somebody else's revenue.
    """
    breaching = _breaching(WIDE_TITLE, TAIL_MINUTES)
    assert len(breaching) > 1, "this programme has to breach more than once, or it tests nothing"
    body = _preview(client, "clean_tail", {"protected_minutes": TAIL_MINUTES}, _where(_title(WIDE_TITLE)))

    assert body["matched_airings"] > len(breaching), (
        "the scope has to be wider than the breach, or the overreach cannot show"
    )
    assert body["compiled_rows"] == len(breaching)
    assert body["bound_airings"] == len(breaching), (
        f"the rule binds {body['bound_airings']} airings and its sentence asks for "
        f"{len(breaching)}; every airing above that is a break removed unasked"
    )
    assert body["asked_for_airings"] == len(breaching)
    assert {change["segment_id"] for change in body["changes"]} == {
        airing.segment_id for airing in breaching
    }
    assert all(change["asked_for"] is True for change in body["changes"])
    assert body["collateral"] == {
        "applies": True, "bound": 0, "changed": 0, "breaks_removed": 0, "days": [],
    }


def test_the_money_on_screen_is_the_money_the_sentence_asks_for(client):
    """Every break the panel prices belongs to an airing that breaches the rule."""
    breaching = {airing.segment_id: airing for airing in _breaching(WIDE_TITLE, TAIL_MINUTES)}
    body = _preview(client, "clean_tail", {"protected_minutes": TAIL_MINUTES}, _where(_title(WIDE_TITLE)))
    scored = body["scored"]
    assert scored["available"] is True, scored

    removed = scored["breaks_before"] - scored["breaks_after"]
    assert removed == sum(
        change["before_breaks"] - change["after_breaks"] for change in body["changes"]
    )
    assert removed == len(breaching), (
        "one break comes off each breaching airing, and none comes off any other"
    )
    assert scored["revenue_delta"] < 0, "removing breaks cannot be worth more money"


def test_each_compiled_row_pins_the_hour_the_engine_itself_reads(client):
    """The predicate is the engine's own arithmetic, not a second opinion of it."""
    from kairos.optimize.predicate import ALLOWED_OPERATORS, _extract_field
    from kairos_api.constraints_airings import segments_for
    from kairos_api.constraints_language import airing_predicate

    assert "eq" in ALLOWED_OPERATORS["hour"], "the frozen contract has to carry this operator"
    for airing in _breaching(WIDE_TITLE, TAIL_MINUTES):
        where = airing_predicate(_where(_title(WIDE_TITLE)), airing)
        hours = [
            node for node in where["conditions"]
            if node["field"] == "hour" and node["operator"] == "eq"
        ]
        assert len(hours) == 1, where
        [segment] = segments_for([airing])
        assert hours[0]["value"] == _extract_field("hour", segment), (
            "the pinned hour has to be the hour the engine derives from the segment"
        )


def test_the_saved_rule_does_to_the_plan_what_the_preview_promised(client):
    """Resolved through the commit path's own loader, on the saved rows."""
    from kairos.optimize.constraints_store import load_constraints, resolve_constraints
    from kairos_api.preview_inputs import preview_inputs

    body = _preview(client, "clean_tail", {"protected_minutes": TAIL_MINUTES}, _where(_title(WIDE_TITLE)))
    saved = client.post("/api/constraints/restrictions", json={
        "kind": "clean_tail", "params": {"protected_minutes": TAIL_MINUTES},
        "where": _where(_title(WIDE_TITLE)), "author": "נציגת תוכן",
    })
    assert saved.status_code == 201, saved.text
    stored = load_constraints(constraints_api.CONSTRAINTS_PATH)

    day = body["changes"][0]["day"]
    promised = {
        change["segment_id"]: change["after_breaks"]
        for change in body["changes"] if change["day"] == day
    }
    segments, kwargs = preview_inputs(CHANNEL, day, None)
    pins, counts, _forbids, _skipped = resolve_constraints(
        segments, stored, operator_channel=kwargs["operator_channel"],
    )
    assert dict(counts) == promised, (
        "the plan holds exactly the airings the preview named, at the counts it named"
    )
    assert not pins


# ---------------------------------------------------------------------------
# 2. Where the hour is still not enough, the surplus is named and priced.


def test_a_shared_clock_hour_leaves_surplus_that_is_counted_priced_and_marked(client):
    """The residual case, on real data, reported rather than assumed away."""
    where = _where(
        _title(SHARED_HOUR_TITLE),
        {"field": "date", "operator": "is", "value": SHARED_HOUR_DAY},
        {"field": "hour", "operator": "eq", "value": SHARED_HOUR},
    )
    body = _preview(client, "clean_tail", {"protected_minutes": 5}, where)

    assert body["compiled_rows"] == 1, "one airing in this hour breaches a five minute tail"
    assert body["asked_for_airings"] == 1
    assert body["bound_airings"] > body["asked_for_airings"], (
        "this is the case the hour cannot separate, so it has to stay a real one"
    )
    collateral = body["collateral"]
    assert collateral["applies"] is True
    assert collateral["bound"] == body["bound_airings"] - body["asked_for_airings"]
    assert collateral["changed"] >= 1, "an airing already keeping the window clean is emptied"
    assert collateral["breaks_removed"] >= 1
    assert collateral["revenue"]["available"] is True
    assert collateral["revenue"]["revenue_delta"] < 0

    surplus = [change for change in body["changes"] if change["asked_for"] is False]
    assert len(surplus) == collateral["changed"]
    assert all(change["asked_for"] is True or change["asked_for"] is False for change in body["changes"])
    priced = sum(change["before_breaks"] - change["after_breaks"] for change in surplus)
    assert priced == collateral["breaks_removed"]


def test_the_surplus_is_a_share_of_the_total_and_never_larger_than_it(client):
    """The two figures on screen have to be the same arithmetic, not two engines."""
    where = _where(
        _title(SHARED_HOUR_TITLE),
        {"field": "date", "operator": "is", "value": SHARED_HOUR_DAY},
        {"field": "hour", "operator": "eq", "value": SHARED_HOUR},
    )
    body = _preview(client, "clean_tail", {"protected_minutes": 5}, where)
    whole = body["scored"]
    part = body["collateral"]["revenue"]
    assert whole["available"] and part["available"]
    assert whole["revenue_before"] == part["revenue_before"], (
        "both count from the same plan as saved, so the starting point is one number"
    )
    assert part["revenue_delta"] >= whole["revenue_delta"], (
        "the surplus cannot cost more than the whole change it is part of"
    )
    assert whole["breaks_before"] - whole["breaks_after"] >= body["collateral"]["breaks_removed"]


# ---------------------------------------------------------------------------
# 3. A scope-level rule has no surplus, because its sentence names the scope.


@pytest.mark.parametrize("kind,params", [
    ("no_breaks", {}),
    ("exact_breaks", {"count": 1}),
    ("fixed_slot", {"offset_seconds": 1320}),
    ("gold", {}),
])
def test_a_scope_level_rule_reports_no_surplus_because_it_asked_for_all_of_them(client, kind, params):
    body = _preview(client, kind, params, _where(_title(WIDE_TITLE)))
    assert body["collateral"]["applies"] is False, (
        f"{kind} names its whole scope, so 'the sentence did not ask for this' has no meaning"
    )
    assert body["asked_for_airings"] == body["bound_airings"]
    assert all(change["asked_for"] is True for change in body["changes"])
