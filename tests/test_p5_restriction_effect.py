"""P5: what the composer says a restriction does, against what the engine does.

The first round of this surface shipped a composer that refused to save four of
its six sentences. A scope-level rule compiles to one store row rather than one
row per airing, and every screen field was derived from the per-airing rows, so
a rule that binds forty-three nights reported nought of nought airings, both
money panels reported nothing, and the save was disabled while the identical
request posted straight to the API returned 201.

The fix is not a wording change, so neither are these tests. They pin the four
facts the screen now rests on:

1. A scope-level rule reports the airings it matches and the airings the engine
   binds, and the two are measured rather than defaulted to nought.
2. The per-airing effect is the engine's own resolver, proven by resolving the
   saved rows through the commit path and comparing.
3. The plan of record joins on every broadcast day, not only the first. This one
   is a regression test with a measured cause: segment ids carry the row index
   within the built frame, so a whole-window build numbered them across the
   window while the plan was written a day at a time, and 2,458 of 2,540 airings
   reported a planned count of nought that no file held.
4. Every basis that cannot be computed says why in both languages, names its own
   scope, and never sends a reader to a figure that does not exist.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.constraints as constraints_api
import kairos_api.version_store as vs

ROOT = Path(__file__).resolve().parents[1]

CHANNEL = "רשת 13"
# A programme with many airings across many broadcast days, which is the shape
# every scope-level kind failed on and the one the exact optimizer run cannot
# price inside its day budget.
WIDE_TITLE = "משחקי השף עונה 7 ש.ח"

KINDS = (
    ("clean_tail", {"protected_minutes": 8}),
    ("clean_open", {"protected_minutes": 5}),
    ("no_breaks", {}),
    ("exact_breaks", {"count": 1}),
    ("fixed_slot", {"offset_seconds": 1320}),
    ("gold", {}),
)

SCOPE_KINDS = ("no_breaks", "exact_breaks", "fixed_slot", "gold")


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


def _where(title: str = WIDE_TITLE, day: str = "") -> dict:
    conditions = [{"field": "programme", "operator": "is", "value": title}]
    if day:
        conditions.append({"field": "date", "operator": "is", "value": day})
    return {"combinator": "and", "conditions": conditions}


def _preview(client: TestClient, kind: str, params: dict, **extra) -> dict:
    response = client.post("/api/constraints/restrictions/preview", json={
        "kind": kind, "params": params, "where": extra.pop("where", _where()), **extra,
    })
    assert response.status_code == 200, response.text
    return response.json()


# ---------------------------------------------------------------------------
# 1. A scope-level rule reports what it binds, and can be saved.


@pytest.mark.parametrize("kind,params", [(k, p) for k, p in KINDS if k in SCOPE_KINDS])
def test_a_scope_rule_reports_the_airings_it_matches_rather_than_nought(client, kind, params):
    listed = client.get(
        "/api/constraints/restrictions/airings", params={"title": WIDE_TITLE},
    ).json()
    assert listed["count"] > 1, "this programme has to air more than once, or it tests nothing"
    body = _preview(client, kind, params)
    assert body["compiled_rows"] == 1, "a scope-level kind compiles to one store row"
    assert body["matched_airings"] == listed["count"], (
        "the composer has to report every airing the rule matches, not the airings "
        "that happen to carry their own compiled row"
    )


@pytest.mark.parametrize("kind,params", KINDS)
def test_every_kind_the_composer_offers_can_be_saved(client, kind, params):
    """The defect this file exists for: the save was gated on the wrong field.

    ``compiled_rows`` is what the store writes and what the create route accepts,
    so it is what the button has to be gated on. Proven by posting the same draft
    the preview was built from and requiring the record back.
    """
    body = _preview(client, kind, params)
    assert body["compiled_rows"] > 0, f"{kind} compiled to no row, so it can never be saved"
    saved = client.post("/api/constraints/restrictions", json={
        "kind": kind, "params": params, "where": _where(), "author": "נציגת תוכן",
    })
    assert saved.status_code == 201, saved.text
    assert saved.json()["row_count"] == body["compiled_rows"], (
        "the preview has to promise the number of rows the save actually writes"
    )


def test_a_scope_rule_prices_the_whole_scope_it_binds(client):
    body = _preview(client, "no_breaks", {})
    scored = body["scored"]
    assert body["bound_airings"] == body["matched_airings"]
    assert body["changes"], "a rule that removes every break has to list the airings it empties"
    assert scored["available"] is True, scored
    assert scored["days"] == body["bound_days"] > 1, (
        "the scored basis covers every broadcast day the rule binds"
    )
    removed = sum(change["before_breaks"] for change in body["changes"])
    assert all(change["after_breaks"] == 0 for change in body["changes"])
    assert removed > 0
    assert scored["breaks_before"] - scored["breaks_after"] == removed, (
        "the breaks the basis says it removed have to be the breaks it listed"
    )
    assert scored["revenue_delta"] < 0, "removing breaks cannot be worth more money"


def test_the_change_list_is_the_airings_the_rule_binds_not_the_rows_it_writes(client):
    """Seven compiled rows bind seventeen airings, and the screen says seventeen.

    A window rule compiles one row per airing, but the predicate that row carries
    names a programme and a date, which is the finest scope the frozen contract
    has, so it also binds any other airing of that programme that night. The
    preview asks the resolver instead of counting its own rows.
    """
    body = _preview(client, "clean_tail", {"protected_minutes": 8})
    assert body["compiled_rows"] >= 1
    assert body["bound_airings"] >= body["compiled_rows"]
    assert len(body["changes"]) == body["bound_airings"] - body["unchanged_airings"]


# ---------------------------------------------------------------------------
# 2. The effect on screen is the engine's, proven through the saved rows.


def test_the_previewed_effect_is_what_the_saved_rule_does_to_the_plan(client):
    """Resolve the saved rows through the commit path and compare, per segment."""
    from kairos.optimize.constraints_store import load_constraints, resolve_constraints
    from kairos_api.preview_inputs import preview_inputs

    body = _preview(client, "no_breaks", {})
    assert client.post("/api/constraints/restrictions", json={
        "kind": "no_breaks", "params": {}, "where": _where(),
    }).status_code == 201
    stored = load_constraints(constraints_api.CONSTRAINTS_PATH)

    day = body["changes"][0]["day"]
    expected = {c["segment_id"] for c in body["changes"] if c["day"] == day}
    segments, kwargs = preview_inputs(CHANNEL, day, None)
    _pins, counts, forbids, _skipped = resolve_constraints(
        segments, stored, operator_channel=kwargs["operator_channel"],
    )
    assert forbids >= expected, (
        "every airing the composer showed emptied has to be forbidden by the engine"
    )
    assert all(counts[segment_id] == 0 for segment_id in expected)


# ---------------------------------------------------------------------------
# 3. The plan of record joins on every broadcast day, not only the first.


def test_the_planned_break_count_joins_the_plan_on_every_broadcast_day():
    """The regression that made 2,458 of 2,540 airings report a false nought."""
    import pandas as pd

    from kairos_api.constraints_airings import all_airings
    from kairos_api.core import OUTPUT_DIR

    airings, _segments = all_airings()
    assert len({airing.day for airing in airings}) > 1, "the window has to span days"
    plan = pd.read_csv(OUTPUT_DIR / "weekly_break_schedule.csv")
    mine = plan[plan["channel"].astype(str) == CHANNEL]
    known = {str(row.segment_id): int(float(row.num_breaks)) for row in mine.itertuples()}

    unknown = [a for a in airings if a.planned_breaks is None]
    assert not unknown, (
        f"{len(unknown)} airings carry no plan row, first {unknown[0].segment_id if unknown else ''}"
    )
    assert sum(a.planned_breaks for a in airings) == sum(known.values()), (
        "the airings' counts have to be the plan of record's own counts for this channel"
    )
    per_day = {a.day for a in airings if a.planned_breaks}
    assert len(per_day) > 1, "breaks have to be found on more than the first day of the window"


def test_an_airing_with_no_plan_row_is_unknown_and_never_judged():
    """The tri-state, exercised directly: today the real data has none of these."""
    from kairos_api.constraints_language import CLEAN_TAIL, Airing, compile_restriction

    unknown = Airing(
        segment_id="x", channel=CHANNEL, day="2024-11-01", title="t",
        start_seconds=0.0, duration_seconds=1020.0, break_length_seconds=120.0,
        planned_breaks=None,
    )
    assert compile_restriction(CLEAN_TAIL, {"protected_minutes": 8}, None, [unknown]) == [], (
        "an airing whose planned count is unknown cannot be judged against a ceiling"
    )


# ---------------------------------------------------------------------------
# 4. Every empty state names its own scope, in the reader's own language.


@pytest.mark.parametrize("kind,params", KINDS)
def test_no_money_panel_prints_a_bare_reason_or_a_raw_key(client, kind, params):
    body = _preview(client, kind, params)
    for name in ("scored", "exact"):
        side = body[name]
        if side.get("available"):
            continue
        assert "reason" not in side, f"{kind}/{name} still carries a single-language reason"
        assert side["reason_en"] and side["reason_he"], f"{kind}/{name} has an empty reason"
        assert not re.search(r"[A-Za-z]{3,}", side["reason_he"]), (
            f"{kind}/{name} leaks an English word or an internal key onto a Hebrew surface: "
            f"{side['reason_he']}"
        )


def test_an_unavailable_basis_never_sends_a_reader_to_a_figure_that_is_not_there(client):
    """The circle the first round shipped: two empty panels citing each other."""
    body = _preview(client, "gold", {})
    scored, exact = body["scored"], body["exact"]
    assert scored["available"] is False and exact["available"] is False, (
        "this draft is the case where neither basis can be computed"
    )
    for side in (scored, exact):
        for text in (side["reason_en"], side["reason_he"]):
            assert "figure" not in text and "המספר המחושב" not in text, (
                f"an empty panel points at the other empty panel: {text}"
            )
    assert str(body["matched_airings"]) in scored["reason_he"], (
        "the scored panel has to name the scope it looked at"
    )
    assert str(exact["days"]) in exact["reason_he"], (
        "the exact panel has to name the scope it would have run"
    )


def test_a_pin_the_plan_cannot_carry_is_named_in_words_and_never_in_engine_tokens(client):
    """Two figures that disagree, with the reason on screen rather than implied.

    A pinned count above an airing's capacity is priced by the count basis and
    refused by the optimizer, so the two panels differ by construction. The
    engine states why, in its own vocabulary, and that vocabulary is translated
    rather than printed: JS-4's target is nought engine words on the path.
    """
    body = _preview(client, "exact_breaks", {"count": 6}, where=_where(day="2024-11-01"))
    assert body["scored"]["available"] and body["exact"]["available"]
    assert body["scored"]["revenue_delta"] > 0, "the count basis prices the arrangement asked for"
    refusals = body["exact"]["refusals"]
    assert refusals, "the optimizer refused these pins and the payload has to carry that"
    assert len(refusals) == len(body["exact"]["rejected_overrides"])
    for refusal in refusals:
        assert refusal["reason_en"] and refusal["reason_he"]
        assert not re.search(r"[A-Za-z_]{3,}", refusal["reason_he"]), (
            f"an engine token reached a Hebrew surface: {refusal['reason_he']}"
        )
    raw = " ".join(item["reason"] for item in body["exact"]["rejected_overrides"])
    assert "max_breaks" in raw, "the engine's own wording is kept, and kept off the screen"
    assert all("max_breaks" not in r["reason_he"] for r in refusals)


@pytest.mark.parametrize("raw,fragment", [
    ("pinned count 6 exceeds max_breaks 4", "4"),
    ("pinned breaks breach a guardrail (spacing/load) for the segment", "עומס"),
    ("conflicting pin_count 2 (already pinned at 3)", "כלל אחר"),
    ("fix_offset needs an offset (offset_seconds or both window bounds)", "דקה"),
    ("gold needs a break to gild", "זהב"),
    ("something nobody has written yet", "מנוע התוכנית"),
])
def test_every_engine_refusal_reaches_the_screen_in_hebrew_with_no_token(raw, fragment):
    """Both closed sets, including the fallback, carry no internal token."""
    from kairos_api.constraints_cost import refusals

    [record] = refusals([{"segment_id": "s", "reason": raw}])
    assert fragment in record["reason_he"], record["reason_he"]
    assert record["reason_en"]
    assert not re.search(r"[A-Za-z_]{3,}", record["reason_he"]), record["reason_he"]


def test_the_exact_basis_cites_the_scored_figure_only_when_that_figure_exists(client):
    wide = _preview(client, "no_breaks", {})
    assert wide["scored"]["available"] is True
    assert wide["exact"]["available"] is False, "this scope is wider than the day budget"
    assert "המספר המחושב" in wide["exact"]["reason_he"], (
        "with a real scored figure beside it, the exact panel should say so"
    )
    assert str(wide["scored"]["days"]) in wide["exact"]["reason_he"]
