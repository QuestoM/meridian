"""Competing server-side versions of one broadcast day, and the single decision.

The operation builds a daily plan and then people change it by hand. Until now a
disagreement about one Tuesday had nowhere to live: a browser-local draft is
private to the browser that holds it and a weekly freeze moves seven days to
settle one. These tests hold the object in between - a named, authored, readable
proposal for ONE channel-day, several of them at once, exactly one adopted.

Every test runs against a relocated proposal store and a relocated plan, so
nothing in the repository is written and the assertions are about the store's own
behaviour rather than about whatever plan happens to be on disk.
"""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from kairos_api import channel_scope, day_proposal_store as store

CHANNEL = "רשת 13"
DAY = "2024-11-05"

COLUMNS = ["channel", "date", "day", "program_type", "start_time", "num_breaks",
           "break_length", "total_break_time", "predicted_revenue",
           "predicted_retention", "base_rate", "baseline_tvr", "segment_id"]

# One operator channel-day across five dayparts. The money is deliberately
# lopsided toward prime, which is what makes an attribution readable.
BASE_ROWS = [
    ("08:00", "Morning", 2, 240.0, 100_000.0, "S1"),
    ("13:00", "Talk", 2, 240.0, 120_000.0, "S2"),
    ("18:00", "News", 1, 120.0, 90_000.0, "S3"),
    ("20:00", "Drama", 3, 360.0, 400_000.0, "S4"),
    ("21:00", "Entertainment", 3, 360.0, 500_000.0, "S5"),
    ("23:30", "Late", 1, 120.0, 40_000.0, "S6"),
]


def frame(rows, channel: str = CHANNEL, day: str = DAY) -> pd.DataFrame:
    """A weekly-schema day frame. ``base_rate`` prices a real retention loss.

    ``base_rate * baseline_tvr * ad_seconds`` is the gross potential the
    net-of-retention reader (:func:`kairos.optimize.revenue_net.frame_revenue_net`)
    values, so setting it to ``revenue / (0.9 * ad_seconds)`` gives every row a
    genuine, non-zero retention cost instead of one that clips to nothing.
    """
    records = []
    for start_time, program_type, breaks, seconds, revenue, segment_id in rows:
        records.append({
            "channel": channel, "date": day, "day": "Tue",
            "program_type": program_type, "start_time": start_time,
            "num_breaks": breaks, "break_length": 120.0,
            "total_break_time": seconds, "predicted_revenue": revenue,
            "predicted_retention": 0.9,
            "base_rate": round(revenue / (0.9 * seconds), 6) if seconds else 0.0,
            "baseline_tvr": 1.0,
            "segment_id": f"{day}|{channel}|{segment_id}",
        })
    return pd.DataFrame(records, columns=COLUMNS)


def ref_for(rows: pd.DataFrame, *, computed_at: str = "2026-08-16T09:00:00+00:00",
            settings: dict | None = None) -> dict:
    return {
        "basis": "engine-day-plan",
        "day_sha256": hashlib.sha256(store.canonical_bytes(rows)).hexdigest(),
        "plan_sha256": "plan-sha-1",
        "computed_at": computed_at,
        "segments": int(len(rows)),
        "settings_basis": settings or SETTINGS_BASIS,
        "captured_at": "2026-08-16T09:00:00+00:00",
    }


SETTINGS_BASIS = {
    "revenue_weight": 60, "min_retention_floor": 0.72, "max_breaks_per_hour": 4,
    "risk_lambda": 0.0, "objective_mode": "blend", "operator_channel": CHANNEL,
}


@pytest.fixture()
def relocated(tmp_path, monkeypatch):
    """A relocated proposal store and a fixed operator channel."""
    monkeypatch.setenv(store.PROPOSALS_DIR_ENV, str(tmp_path / "day_proposals"))
    monkeypatch.setattr(store, "_settings_basis", lambda: dict(SETTINGS_BASIS))
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: CHANNEL)
    return tmp_path


def make(name: str, rows: pd.DataFrame, *, author: str = "dana",
         ref: dict | None = None, edits: dict | None = None) -> dict:
    return store.create_proposal(
        channel=CHANNEL, date=DAY, name=name, author=author, rows=rows,
        baseline_ref=ref or ref_for(frame(BASE_ROWS)), edits=edits or {},
        rows_source="engine-day-plan-with-edits",
    )


# ------------------------------------------------------------------- the object

def test_a_proposal_freezes_the_rows_the_author_and_the_baseline(relocated):
    rows = frame(BASE_ROWS)
    manifest = make("שמירה על הצפייה", rows, author="dana")

    assert manifest["name"] == "שמירה על הצפייה"
    assert manifest["author"] == "dana"
    assert manifest["status"] == store.PROPOSED
    assert manifest["channel"] == CHANNEL
    assert manifest["date"] == DAY
    assert manifest["seq"] == 1
    assert manifest["decision"] is None
    # The frozen bytes are the proposal: the sha is over the canonical form and
    # the file on disk is byte-identical to it.
    directory = store.day_root(CHANNEL, DAY) / manifest["proposal_id"]
    payload = (directory / store.PLAN_FILENAME).read_bytes()
    assert payload == store.canonical_bytes(rows)
    assert manifest["rows_sha256"] == hashlib.sha256(payload).hexdigest()
    assert manifest["settings_basis"]["operator_channel"] == CHANNEL
    assert manifest["baseline_ref"]["basis"] == "engine-day-plan"
    # The rows read back identical, which is what makes adoption a publish.
    read_back = store.rows_for(CHANNEL, DAY, manifest["proposal_id"])
    assert int(read_back["num_breaks"].sum()) == 12
    assert round(float(read_back["predicted_revenue"].sum()), 2) == 1_250_000.0


def test_the_money_is_the_operators_and_the_scope_note_travels_with_it(relocated):
    mixed = pd.concat([frame(BASE_ROWS), frame(BASE_ROWS[:1], channel="קשת 12")],
                      ignore_index=True)
    manifest = make("עם שורה של מתחרה", mixed)

    owned = manifest["summary"]["owned"]
    every = manifest["summary"]["all_channels"]
    assert owned["revenue"] == 1_250_000.0
    assert owned["breaks"] == 12
    assert every["revenue"] == 1_350_000.0
    note = manifest["summary"]["scope"]
    assert note["scope_channel"] == CHANNEL
    assert note["scoped"] is True
    assert note["competitor_rows_excluded"] == 1


def test_a_proposal_nobody_can_name_is_refused(relocated):
    with pytest.raises(store.ProposalRefused) as caught:
        make("   ", frame(BASE_ROWS))
    assert caught.value.code == "no_name"
    assert caught.value.reason_he


def test_three_competing_versions_sit_on_the_table_at_once(relocated):
    make("העמסת פריים", frame(BASE_ROWS), author="dana")
    make("שמירה על הצפייה", frame(BASE_ROWS), author="yossi")
    make("מיזוג יום", frame(BASE_ROWS), author="rina")

    listed = store.list_for_day(CHANNEL, DAY)
    assert [item["seq"] for item in listed] == [3, 2, 1]
    assert {item["author"] for item in listed} == {"dana", "yossi", "rina"}
    assert {item["status"] for item in listed} == {store.PROPOSED}
    assert store.adopted_for_day(CHANNEL, DAY) is None


# --------------------------------------------------------------- the decision

def test_adopting_one_closes_the_rivals_with_lineage_and_keeps_them_readable(relocated):
    rows = frame(BASE_ROWS)
    ref = ref_for(rows)
    winner = make("העמסת פריים", rows, ref=ref)
    loser_one = make("שמירה על הצפייה", rows, ref=ref)
    loser_two = make("מיזוג יום", rows, ref=ref)

    adopted = store.update_status(CHANNEL, DAY, winner["proposal_id"], store.ADOPTED,
                                 actor="miri", note="פריים נמכר, ההפסד בבוקר מקובל",
                                 current_ref=ref)
    assert adopted["status"] == store.ADOPTED
    assert adopted["decision"] == {
        "verdict": store.ADOPTED, "by": "miri",
        "at": adopted["decision"]["at"], "note": "פריים נמכר, ההפסד בבוקר מקובל",
    }
    closed = store.reject_rivals(CHANNEL, DAY, winner["proposal_id"], actor="miri",
                                note="נעקפה על ידי ההצעה שאומצה")
    assert {item["proposal_id"] for item in closed} == {
        loser_one["proposal_id"], loser_two["proposal_id"]}
    for item in closed:
        assert item["status"] == store.REJECTED
        assert item["lineage"]["superseded_by"] == winner["proposal_id"]
    # Rejection is not deletion: the losing versions are still fully readable.
    for item in closed:
        rows_back = store.rows_for(CHANNEL, DAY, item["proposal_id"])
        assert rows_back is not None and len(rows_back) == 6
        assert store.get(CHANNEL, DAY, item["proposal_id"])["decision"]["note"]
    assert len(store.list_for_day(CHANNEL, DAY)) == 3


def test_a_second_adoption_is_refused_by_name(relocated):
    rows = frame(BASE_ROWS)
    ref = ref_for(rows)
    first = make("העמסת פריים", rows, ref=ref)
    second = make("מיזוג יום", rows, ref=ref)
    store.update_status(CHANNEL, DAY, first["proposal_id"], store.ADOPTED,
                       actor="miri", note="נבחרה", current_ref=ref)

    with pytest.raises(store.ProposalRefused) as caught:
        store.update_status(CHANNEL, DAY, second["proposal_id"], store.ADOPTED,
                            actor="miri", note="גם זו", current_ref=ref)
    assert caught.value.code == "already_adopted"
    assert "העמסת פריים" in caught.value.reason
    assert "העמסת פריים" in caught.value.reason_he


def test_a_withdrawn_proposal_cannot_be_adopted_even_on_an_open_day(relocated):
    """The terminal guard, isolated from the one-per-day guard that precedes it."""
    rows = frame(BASE_ROWS)
    ref = ref_for(rows)
    manifest = make("ניסוי", rows, ref=ref)
    store.withdraw(CHANNEL, DAY, manifest["proposal_id"], actor="dana", note="חוזר בי")
    assert store.adopted_for_day(CHANNEL, DAY) is None
    with pytest.raises(store.ProposalRefused) as caught:
        store.update_status(CHANNEL, DAY, manifest["proposal_id"], store.ADOPTED,
                            actor="miri", note="בכל זאת", current_ref=ref)
    assert caught.value.code == "already_decided"
    assert "ניסוי" in caught.value.reason_he


def test_an_adopted_proposal_is_terminal(relocated):
    rows = frame(BASE_ROWS)
    ref = ref_for(rows)
    manifest = make("העמסת פריים", rows, ref=ref)
    store.update_status(CHANNEL, DAY, manifest["proposal_id"], store.ADOPTED,
                       actor="miri", note="נבחרה", current_ref=ref)
    with pytest.raises(store.ProposalRefused) as caught:
        store.update_status(CHANNEL, DAY, manifest["proposal_id"], store.REJECTED,
                            actor="miri", note="בעצם לא")
    assert caught.value.code == "already_decided"


# ----------------------------------------------------------------- staleness

def test_a_stale_proposal_refuses_adoption_and_names_the_baseline_move(relocated):
    authored_against = frame(BASE_ROWS)
    manifest = make("העמסת פריים", authored_against, ref=ref_for(authored_against))

    moved = list(BASE_ROWS)
    moved[3] = ("20:00", "Drama", 4, 480.0, 520_000.0, "S4")
    current = ref_for(frame(moved))

    state = store.staleness(manifest, current)
    assert state["known"] is True
    assert state["stale"] is True
    assert [item["field"] for item in state["moved"]] == ["day_sha256"]
    assert state["moved"][0]["reason_he"]

    with pytest.raises(store.ProposalRefused) as caught:
        store.update_status(CHANNEL, DAY, manifest["proposal_id"], store.ADOPTED,
                            actor="miri", note="בכל זאת", current_ref=current)
    assert caught.value.code == "stale"
    assert "day_sha256" in caught.value.reason


def test_a_settings_move_is_a_baseline_move_too(relocated):
    rows = frame(BASE_ROWS)
    manifest = make("העמסת פריים", rows)
    current = ref_for(rows, settings={**SETTINGS_BASIS, "revenue_weight": 80})
    state = store.staleness(manifest, current)
    assert state["stale"] is True
    assert [item["field"] for item in state["moved"]] == ["settings.revenue_weight"]


def test_an_unreadable_baseline_is_unknown_and_still_refuses_adoption(relocated):
    manifest = make("העמסת פריים", frame(BASE_ROWS))
    state = store.staleness(manifest, None)
    assert state["known"] is False
    assert state["stale"] is False
    assert state["reason_he"]
    with pytest.raises(store.ProposalRefused) as caught:
        store.update_status(CHANNEL, DAY, manifest["proposal_id"], store.ADOPTED,
                            actor="miri", note="בלי לדעת", current_ref=None)
    assert caught.value.code == "stale"


def test_an_explicit_rebase_records_the_move_and_unblocks_adoption(relocated):
    authored_against = frame(BASE_ROWS)
    manifest = make("העמסת פריים", authored_against, ref=ref_for(authored_against))
    moved = list(BASE_ROWS)
    moved[3] = ("20:00", "Drama", 4, 480.0, 520_000.0, "S4")
    current = ref_for(frame(moved))

    rebased = store.rebase(CHANNEL, DAY, manifest["proposal_id"], actor="dana",
                           new_ref=current, note="בדקתי, הגרסה עומדת")
    assert rebased["baseline_ref"]["day_sha256"] == current["day_sha256"]
    assert rebased["lineage"]["rebased_from"]["day_sha256"] == \
        manifest["baseline_ref"]["day_sha256"]
    assert rebased["lineage"]["rebased_by"] == "dana"
    assert store.staleness(rebased, current)["stale"] is False
    # The author's rows did not move: a re-base is a statement, not an edit.
    assert rebased["rows_sha256"] == manifest["rows_sha256"]

    adopted = store.update_status(CHANNEL, DAY, manifest["proposal_id"], store.ADOPTED,
                                 actor="miri", note="מאומצת", current_ref=current)
    assert adopted["status"] == store.ADOPTED


def test_withdrawing_takes_a_version_off_the_table_and_leaves_it_readable(relocated):
    manifest = make("ניסוי", frame(BASE_ROWS), author="dana")
    withdrawn = store.withdraw(CHANNEL, DAY, manifest["proposal_id"], actor="dana",
                               note="חוזר בי")
    assert withdrawn["status"] == store.WITHDRAWN
    assert store.rows_for(CHANNEL, DAY, manifest["proposal_id"]) is not None
    # A withdrawn version is not a rival, so adopting another is unobstructed.
    other = make("החלופה", frame(BASE_ROWS))
    ref = ref_for(frame(BASE_ROWS))
    assert store.update_status(CHANNEL, DAY, other["proposal_id"], store.ADOPTED,
                              actor="miri", note="נבחרה",
                              current_ref=ref)["status"] == store.ADOPTED


def test_a_channel_with_hebrew_letters_gets_a_legible_directory(relocated):
    make("העמסת פריים", frame(BASE_ROWS))
    assert store.channel_slug(CHANNEL) == "רשת-13"
    assert (store.proposals_root() / "רשת-13" / DAY).is_dir()
    with pytest.raises(store.ProposalRefused):
        store.channel_slug("///")


def test_a_day_that_is_not_a_day_is_refused(relocated):
    with pytest.raises(store.ProposalRefused) as caught:
        store.create_proposal(channel=CHANNEL, date="week 45", name="x", author="d",
                              rows=frame(BASE_ROWS), baseline_ref={})
    assert caught.value.code == "bad_date"


# ------------------------------------------------- adoption publishes the plan

@pytest.fixture()
def live_plan(tmp_path, monkeypatch):
    """A relocated committed plan plus a relocated week-freeze store."""
    from kairos_api import day_proposal_rows as rows_api
    from kairos_api import plan_version_store

    path = tmp_path / "weekly_break_schedule.csv"
    other_day = frame([("09:00", "Morning", 1, 120.0, 10_000.0, "T1")], day="2024-11-06")
    rival = frame(BASE_ROWS[:2], channel="קשת 12")
    pd.concat([frame(BASE_ROWS), other_day, rival], ignore_index=True).to_csv(
        path, index=False, encoding="utf-8")
    path.with_name(path.name + ".meta.json").write_text(
        json.dumps({"computed_at": "2026-08-16T09:00:00+00:00", "fingerprints": {}}),
        encoding="utf-8")
    monkeypatch.setenv(plan_version_store.PLAN_VERSIONS_DIR_ENV, str(tmp_path / "versions"))
    monkeypatch.setattr(plan_version_store, "plan_path", lambda: path)
    monkeypatch.setattr(plan_version_store, "meta_path",
                        lambda: path.with_name(path.name + ".meta.json"))
    monkeypatch.setattr(plan_version_store, "_settings_basis", lambda: dict(SETTINGS_BASIS))
    monkeypatch.setattr(rows_api, "plan_path", lambda: path)
    return path


def test_adoption_publishes_the_day_and_freezes_what_it_replaced(relocated, live_plan):
    from kairos_api import day_proposal_rows as rows_api
    from kairos_api import plan_version_store

    proposed = list(BASE_ROWS)
    proposed[4] = ("21:00", "Entertainment", 4, 480.0, 630_000.0, "S5")
    rows = frame(proposed)
    manifest = make("העמסת פריים", rows)

    published = rows_api.publish_day(CHANNEL, DAY, rows, actor="miri",
                                     proposal_name=manifest["name"])
    assert published["ok"] is True
    assert published["rows_written"] == 6
    assert published["rows_replaced"] == 6
    assert published["revenue_published"] == 1_380_000.0
    # The plan as it stood is a named version before a byte moved.
    safety = plan_version_store.get(published["safety_version_id"])
    assert safety["name"].startswith("before adopting")
    # The freeze is the whole plan as it stood, which is both operator days:
    # this Tuesday at 1,250,000 and the Wednesday beside it at 10,000.
    assert safety["summary"]["owned"]["revenue"] == 1_260_000.0
    assert safety["summary"]["owned"]["days"] == 2

    after = pd.read_csv(live_plan)
    mine = after[(after["channel"] == CHANNEL) & (after["date"] == DAY)]
    assert round(float(mine["predicted_revenue"].sum()), 2) == 1_380_000.0
    # Only this channel-day moved: the other day and the competitor rows survive.
    assert len(after[after["date"] == "2024-11-06"]) == 1
    assert len(after[after["channel"] == "קשת 12"]) == 2
    assert len(after) == 9


def test_publishing_into_a_missing_plan_refuses(relocated, tmp_path, monkeypatch):
    from kairos_api import day_proposal_rows as rows_api

    monkeypatch.setattr(rows_api, "plan_path", lambda: tmp_path / "absent.csv")
    with pytest.raises(FileNotFoundError):
        rows_api.publish_day(CHANNEL, DAY, frame(BASE_ROWS), actor="miri",
                             proposal_name="x")


def test_the_shipped_plan_guard_refuses_a_publish_on_a_read_only_tree(relocated, monkeypatch):
    """The wall the whole test session runs behind is the one adoption goes through."""
    from kairos.export.plan_guard import PlanArtifactProtected
    from kairos_api import day_proposal_rows as rows_api
    from kairos_api import plan_version_store

    shipped = rows_api.shipped_plan_path()
    if not shipped.exists():
        pytest.skip("this tree carries no shipped plan to protect")
    monkeypatch.setattr(rows_api, "plan_path", lambda: shipped)
    monkeypatch.setattr(plan_version_store, "plan_path", lambda: shipped)
    monkeypatch.setenv("KAIROS_PLAN_READONLY", "1")
    with pytest.raises(PlanArtifactProtected):
        rows_api.publish_day(CHANNEL, DAY, frame(BASE_ROWS), actor="miri",
                             proposal_name="x")


# --------------------------------------------------- the surface, over HTTP

@pytest.fixture()
def api(relocated, live_plan, monkeypatch):
    """The routes with the engine seam pointed at fixture rows.

    Building a real day plan runs the optimizer against the operator's own data
    and takes about a second per call; this fixture substitutes the two engine
    entry points so the ROUTES are what these tests measure. Everything below
    the seam - the store, the state machine, the comparison, the publish - is
    the real thing.
    """
    from fastapi.testclient import TestClient

    from kairos_api import day_proposal_api as api_module
    from kairos_api import day_proposal_rows as rows_api
    from kairos_api.server import app

    baseline = frame(BASE_ROWS)
    caps = {"max_daily_ad_seconds": 1500.0, "max_ad_seconds_per_hour": 720.0,
            "max_breaks_per_hour": 4}
    monkeypatch.setattr(rows_api, "baseline_for_day", lambda day: {
        "channel": CHANNEL, "day": DAY, "plan": None, "rows": baseline,
        "ref": ref_for(baseline), "caps": caps,
        "engine": {"compliance": {"available": True, "compliant": True,
                                  "checks_run": 6, "violations": []}},
    })

    def _proposal_rows(day, moves=None):
        rows = frame(BASE_ROWS if not moves else _moved_rows(moves))
        return {
            "channel": CHANNEL, "day": DAY, "rows": rows,
            "engine": {"compliance": {"available": True, "compliant": True,
                                      "checks_run": 6, "violations": []},
                       "engine_ms": 812.5},
            "rows_source": "engine-day-plan-with-edits" if moves else "engine-day-plan",
            "baseline_ref": ref_for(baseline),
        }

    monkeypatch.setattr(rows_api, "proposal_rows", _proposal_rows)
    monkeypatch.setattr(api_module, "trade_context", lambda: None)
    return TestClient(app)


def _moved_rows(moves):
    """One edit per move, so each proposal created over HTTP is a distinct day."""
    rows = list(BASE_ROWS)
    for index, _move in enumerate(moves):
        start_time, program_type, breaks, seconds, revenue, segment_id = rows[3 + index]
        rows[3 + index] = (start_time, program_type, breaks + 1, seconds + 120.0,
                           revenue + 120_000.0, segment_id)
    return rows


def _create(client, name, moves):
    response = client.post("/api/plan/day-proposals",
                           json={"day": DAY, "name": name, "moves": moves})
    assert response.status_code == 201, response.text
    return response.json()["proposal"]["proposal_id"]


def test_the_routes_carry_a_day_from_three_proposals_to_one_decision(api):
    first = _create(api, "העמסת פריים", [{"break_id": f"{DAY}|{CHANNEL}|S4~1",
                                          "duration_seconds": 240.0}])
    second = _create(api, "שמירה על הצפייה", [
        {"break_id": f"{DAY}|{CHANNEL}|S4~1", "duration_seconds": 240.0},
        {"break_id": f"{DAY}|{CHANNEL}|S5~1", "duration_seconds": 240.0},
    ])
    third = _create(api, "מיזוג יום", [])

    listed = api.get("/api/plan/day-proposals", params={"day": DAY}).json()
    assert listed["available"] is True
    assert listed["count"] == 3
    assert listed["status_counts"] == {"proposed": 3}
    assert listed["adopted"] is None
    assert all(item["staleness"]["known"] for item in listed["proposals"])

    compared = api.post("/api/plan/day-proposals/compare", json={
        "day": DAY, "proposal_ids": [first, second, third]}).json()
    assert compared["available"] is True
    assert compared["scored_sides"] == 3
    deltas = {side["label"]: side["delta"]["revenue"] for side in compared["sides"]}
    assert deltas == {"העמסת פריים": 120_000.0, "שמירה על הצפייה": 240_000.0,
                      "מיזוג יום": 0.0}
    assert compared["highest_revenue_side"] == second

    # A decision with no reason in writing is refused before anything moves.
    refused = api.post(f"/api/plan/day-proposals/{second}/decide",
                       json={"day": DAY, "verdict": "adopt", "note": "   "})
    assert refused.status_code == 422
    assert refused.json()["detail"]["code"] == "no_annotation"
    assert refused.json()["detail"]["reason_he"]

    adopted = api.post(f"/api/plan/day-proposals/{second}/decide", json={
        "day": DAY, "verdict": "adopt", "note": "הפריים נמכר; מקבלים את אורך הברייק"})
    assert adopted.status_code == 200, adopted.text
    body = adopted.json()
    assert body["verdict"] == "adopted"
    assert body["published"]["rows_written"] == 6
    assert body["published"]["safety_version_id"]
    assert {item["proposal_id"] for item in body["rejected"]} == {first, third}
    assert all(item["lineage"]["superseded_by"] == second for item in body["rejected"])

    # The plan of record now carries the adopted day and nothing else moved.
    after = pd.read_csv(live_plan_path(api))
    mine = after[(after["channel"] == CHANNEL) & (after["date"] == DAY)]
    assert round(float(mine["predicted_revenue"].sum()), 2) == 1_490_000.0

    # A second adoption on the same day is refused by name, over HTTP.
    second_attempt = api.post(f"/api/plan/day-proposals/{first}/decide",
                              json={"day": DAY, "verdict": "adopt", "note": "גם זו"})
    assert second_attempt.status_code == 409
    assert second_attempt.json()["detail"]["code"] == "already_adopted"
    assert "שמירה על הצפייה" in second_attempt.json()["detail"]["reason_he"]

    history = api.get("/api/plan/day-proposals/history", params={"day": DAY}).json()
    assert history["count"] == 3
    statuses = {entry["name"]: entry["status"] for entry in history["entries"]}
    assert statuses == {"שמירה על הצפייה": "adopted", "העמסת פריים": "rejected",
                        "מיזוג יום": "rejected"}
    winner = next(e for e in history["entries"] if e["status"] == "adopted")
    assert winner["decision"]["note"] == "הפריים נמכר; מקבלים את אורך הברייק"
    assert winner["revenue"] == 1_490_000.0
    assert winner["scope"]["scope_channel"] == CHANNEL
    # Every rejected alternative is still fully readable with its own figures.
    for entry in history["entries"]:
        detail = api.get(f"/api/plan/day-proposals/{entry['proposal_id']}",
                         params={"day": DAY})
        assert detail.status_code == 200
        assert detail.json()["proposal"]["summary"]["owned"]["revenue"] > 0


def live_plan_path(client):
    from kairos_api import day_proposal_rows as rows_api

    return rows_api.plan_path()


def test_a_proposal_with_no_name_is_refused_over_http(api):
    response = api.post("/api/plan/day-proposals",
                        json={"day": DAY, "name": "  ", "moves": []})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "no_name"
    assert response.json()["detail"]["reason_he"]


def test_rejecting_a_proposal_needs_a_reason_and_keeps_it_readable(api):
    proposal_id = _create(api, "ניסוי", [])
    assert api.post(f"/api/plan/day-proposals/{proposal_id}/decide",
                    json={"day": DAY, "verdict": "reject"}).status_code == 422
    rejected = api.post(f"/api/plan/day-proposals/{proposal_id}/decide", json={
        "day": DAY, "verdict": "reject", "note": "פוגע ברייטינג בלי תמורה"})
    assert rejected.status_code == 200
    assert rejected.json()["proposal"]["status"] == "rejected"
    assert rejected.json()["proposal"]["decision"]["note"] == "פוגע ברייטינג בלי תמורה"
    assert api.get(f"/api/plan/day-proposals/{proposal_id}",
                   params={"day": DAY}).status_code == 200


def test_an_unknown_verdict_is_refused(api):
    proposal_id = _create(api, "ניסוי", [])
    response = api.post(f"/api/plan/day-proposals/{proposal_id}/decide",
                        json={"day": DAY, "verdict": "maybe", "note": "?"})
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "bad_verdict"
