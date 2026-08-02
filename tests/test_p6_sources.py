"""P6 Sources: the state of every input, the rows behind a count, and the audit.

Every test runs against an app carrying only the routers this piece owns, so a
failure here is this piece's and every writable path is relocated to a
temporary directory before anything is written. The door a bad file cannot pass
is the one subject that is not here: it is in ``test_p6_door.py``, because this
file had grown past the 450-line law and a refusal is its own subject.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import kairos_api.downloads_api as downloads_api
import kairos_api.uploads as uploads
import kairos_api.uploads_preview as uploads_preview
import kairos_api.uploads_status as uploads_status
from kairos_api import read_cache

ROOT = Path(__file__).resolve().parents[1]

# The nine words section 4.2 of the specification greps a run surface for. A
# Sources payload returns zero of them: which model version the numbers rest on
# is an operator fact, and every verdict behind it is the company side's.
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


@pytest.fixture()
def client() -> TestClient:
    app = FastAPI()
    app.include_router(uploads.router)
    app.include_router(downloads_api.router)
    return TestClient(app)


@pytest.fixture()
def owned(monkeypatch) -> str:
    """A configured operator channel, pinned rather than read off disk.

    The settings file is shared writable state and another process blanking it
    would turn every boundary assertion below into a skip, which is the one
    outcome a boundary test may never have.
    """
    from kairos.data.loaders import CHANNELS
    from kairos_api import channel_scope, read_cache

    channel = CHANNELS[-1]
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: channel)
    read_cache.invalidate(uploads_preview.PREVIEW_NAMESPACE)
    return channel


@pytest.fixture()
def isolated(tmp_path, monkeypatch) -> TestClient:
    """Every writable path relocated, so no repository input is touched."""
    data_dir = tmp_path / "data"
    monkeypatch.setattr(uploads, "DATA_DIR", data_dir)
    monkeypatch.setattr(uploads, "DAILY_DIR", data_dir / "daily_input")
    monkeypatch.setattr(uploads, "BACKUP_DIR", data_dir / "_backups")
    monkeypatch.setattr(
        uploads, "VALIDATION_REPORTS_PATH", tmp_path / "output" / "upload_validation_reports.json"
    )
    app = FastAPI()
    app.include_router(uploads.router)
    return TestClient(app)


# --- the closed state vocabulary, and a remedy on every one of them ------------
def test_every_input_carries_one_of_six_states_and_a_remedy(client: TestClient) -> None:
    body = client.get("/api/uploads/status").json()
    assert body["inputs"], "the status must report the inputs the optimizer depends on"
    for entry in body["inputs"]:
        assert entry["state"] in uploads_status.STATES, f"{entry['kind']} invented a state"
        assert entry["remedy"]["he"], f"{entry['kind']} states a problem with no remedy in Hebrew"
        assert entry["remedy"]["en"], f"{entry['kind']} states a problem with no remedy in English"


def test_no_two_states_share_a_word_in_either_language() -> None:
    """A closed vocabulary whose words collide is not a vocabulary."""
    from json import loads

    source = (ROOT / "tv-break-dashboard" / "src" / "sources" / "sources-copy.js").read_text(encoding="utf-8")
    block = source[source.index("export const STATE_LABELS"):]
    block = block[block.index("{"): block.index("};") + 1]
    pairs = dict(
        (key, (en, he))
        for key, en, he in __import__("re").findall(r"(\w+):\s*\{ en: '([^']*)', he: '([^']*)' \}", block)
    )
    assert set(pairs) == set(uploads_status.STATES), "the surface and the server disagree on the state set"
    assert len({en for en, _ in pairs.values()}) == len(pairs), "two states share an English word"
    assert len({he for _, he in pairs.values()}) == len(pairs), "two states share a Hebrew word"


def test_the_state_summary_counts_what_the_entries_say(client: TestClient) -> None:
    body = client.get("/api/uploads/status").json()
    for state in uploads_status.STATES:
        counted = sum(1 for entry in body["inputs"] if entry["state"] == state)
        assert body["summary"][state] == counted, f"the {state} count disagrees with the entries"
    assert body["summary"]["total"] == len(body["inputs"])


# --- the consequence of an upload, stated before it happens --------------------
def test_every_input_states_what_an_upload_would_do(client: TestClient) -> None:
    body = client.get("/api/uploads/status").json()
    for entry in body["inputs"]:
        assert entry["consequence"]["code"] in uploads_status.CONSEQUENCES
        assert entry["consequence"]["he"] and entry["consequence"]["en"]


def test_a_shadowed_upload_says_it_changes_no_number(client: TestClient) -> None:
    """Measured today: the three channel kinds are read from the reference
    workbooks, so uploading a CSV of those kinds changes nothing."""
    body = client.get("/api/uploads/status").json()
    for entry in body["inputs"]:
        if entry["in_use"]:
            continue
        assert entry["consequence"]["code"] == "stored_not_read"


def test_the_live_input_that_the_model_was_measured_on_says_so() -> None:
    """The one condition under which an operator upload moves the model's basis."""
    assert uploads_status.consequence_for(in_use=True, model_input=True) == "changes_model_basis"
    assert uploads_status.consequence_for(in_use=True, model_input=False) == "replaces_live_input"
    assert uploads_status.consequence_for(in_use=False, model_input=True) == "stored_not_read"


# --- the model version, and nothing from the other side of the line ------------
def test_the_model_block_is_a_version_a_state_and_its_sources(client: TestClient) -> None:
    model = client.get("/api/uploads/status").json()["model"]
    assert model["status"] in {"fresh", "stale", "unknown"}, "the model state must stay tri-state"
    if model["available"]:
        assert model["version"], "a model version must be named by the day it was trained"
        assert model["measured_on"], "a version must name the sources it was measured on"
    assert model["note_he"] and model["note_en"]


def test_no_sources_payload_carries_a_training_word(client: TestClient) -> None:
    """Section 4.2's lexicon test, with the one exception measured, not hidden.

    The file audit names ``models/tv_break_coefficients.json``, because a file
    audit that hides the artifact every retention figure rests on is worse than
    a filename, and section 5.6 of the specification asks for it by name. What
    crosses is a path, a role and a sentence about which screen carries its
    state. The test strips that one path and then requires zero hits, so a
    verdict, a coverage note or a p-value arriving later still fails.
    """
    allowed = downloads_api.MODEL_ARTIFACT.lower()
    for path in ("/api/uploads/status", "/api/files", "/api/reports", "/api/uploads/spots/preview?limit=1"):
        body = json.dumps(client.get(path).json(), ensure_ascii=False).lower()
        hits = {word for word in TRAINING_LEXICON if word in body.replace(allowed, "")}
        assert hits == set(), f"{path} leaked the training lexicon: {sorted(hits)}"


def test_the_model_artifact_is_a_path_and_a_role_and_nothing_from_inside_it(client: TestClient) -> None:
    """The artifact that drives every retention number is on the audit, and it
    is the only file on it whose name is also a word the run side never uses."""
    audit = client.get("/api/files").json()
    record = next(
        (row for row in audit["also_read"] if row["path"].replace("\\", "/") == downloads_api.MODEL_ARTIFACT),
        None,
    )
    assert record is not None, "the file audit omits the artifact every retention number rests on"
    assert record["role"] == "model"
    assert record["in_use"] is True
    assert record["note"]["code"] == "model_live"
    assert record["note"]["he"] and record["note"]["en"]
    assert set(record) == {"path", "exists", "size", "modified", "role", "in_use", "note"}
    paths = [row["path"] for row in audit["files"]] + [row["path"] for row in audit["also_read"]]
    named = [path for path in paths if any(word in path.lower() for word in TRAINING_LEXICON)]
    assert named == [downloads_api.MODEL_ARTIFACT], f"a second training-named path reached the audit: {named}"


# The door itself, which is the largest subject here and the one this
# destination's reference bar is set on, is in ``test_p6_door.py``: the file was
# past the 450-line law and a refusal is its own subject.


# --- the rows behind a count, and the boundary that guards them ----------------
def test_the_preview_shows_only_the_operators_own_channel(client: TestClient, owned: str) -> None:
    body = client.get("/api/uploads/spots/preview?limit=25").json()
    if not body["available"] or not body["rows"]:
        pytest.skip("no historical spots file on disk to preview")
    index = body["columns"].index("Channel")
    assert {row[index] for row in body["rows"]} == {owned}, "a rival channel reached an operator surface"
    assert body["scope"]["scope_channel"] == owned
    assert body["scoped_rows"] <= body["total_rows"]


def test_the_preview_discloses_what_it_excluded_without_naming_it(client: TestClient, owned: str) -> None:
    body = client.get("/api/uploads/spots/preview?limit=5").json()
    if not body["available"]:
        pytest.skip("no historical spots file on disk to preview")
    from kairos.data.loaders import CHANNELS

    payload = json.dumps(body, ensure_ascii=False)
    for rival in [channel for channel in CHANNELS if channel != owned]:
        assert rival not in payload, "the preview named a channel the operator does not own"
    assert body["scope"]["competitor_rows_excluded"] >= 0
    assert body["scope"]["competitor_channels_excluded"] >= 0


def test_a_channel_file_shows_nothing_at_all_until_a_channel_is_configured(client: TestClient, monkeypatch) -> None:
    """Without a configured channel the boundary cannot be applied, so the rows
    are withheld and the note names where to set it. Passing them through would
    serve every channel's rows on an operator surface."""
    from kairos_api import channel_scope, read_cache

    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: "")
    read_cache.invalidate(uploads_preview.PREVIEW_NAMESPACE)
    for kind in sorted(uploads_preview.CHANNEL_KINDS):
        body = client.get(f"/api/uploads/{kind}/preview?limit=5").json()
        assert body["available"] is False, f"the {kind} preview rendered rows with no channel configured"
        assert body["rows"] == []
        assert body["notes"][0]["code"] == "no_channel"
        assert body["notes"][0]["he"] and body["notes"][0]["en"]


def test_the_dayparts_preview_hides_rival_channel_columns(client: TestClient, owned: str) -> None:
    body = client.get("/api/uploads/dayparts/preview?limit=2").json()
    if not body["available"]:
        pytest.skip("no dayparts file on disk to preview")
    from kairos.data.loaders import CHANNELS

    for rival in [channel for channel in CHANNELS if channel != owned]:
        assert rival not in body["columns"], "the dayparts preview kept a rival channel column"
    assert body["columns_hidden"] == len([c for c in CHANNELS if c != owned])


def test_the_preview_withholds_the_three_inverted_columns(client: TestClient, owned: str) -> None:
    body = client.get("/api/uploads/spots/preview?limit=2").json()
    if not body["available"]:
        pytest.skip("no historical spots file on disk to preview")
    for column in uploads_preview.INVERTED_COLUMNS:
        assert column not in body["columns"], f"{column} flags the wrong channel and must not render"
    assert body["columns_withheld"] == 3
    assert body["notes"], "withholding a column must say so"


def test_every_preview_note_is_bilingual(client: TestClient, owned: str) -> None:
    for kind in ("programmes", "spots", "dayparts", "daily", "rate_card"):
        body = client.get(f"/api/uploads/{kind}/preview?limit=1").json()
        for entry in body.get("notes") or []:
            assert entry["en"] and entry["he"], f"{kind} note {entry['code']} is not bilingual"


def test_the_preview_is_capped_and_honest_about_the_total(client: TestClient, owned: str) -> None:
    body = client.get("/api/uploads/spots/preview?limit=5000").json()
    if not body["available"]:
        pytest.skip("no historical spots file on disk to preview")
    assert body["shown_rows"] <= uploads_preview.MAX_PREVIEW_ROWS
    assert body["total_rows"] >= body["scoped_rows"] >= body["shown_rows"]


def test_a_missing_file_says_so_rather_than_showing_an_empty_table(isolated: TestClient) -> None:
    body = isolated.get("/api/uploads/campaign_flights/preview").json()
    assert body["available"] is False
    assert body["notes"] and body["notes"][0]["he"], "an absent file must say so in Hebrew"
    assert body["rows"] == []


# --- the refusal is legible before the click ----------------------------------
def test_the_status_says_whether_this_account_may_change_anything(client: TestClient) -> None:
    body = client.get("/api/uploads/status").json()
    assert isinstance(body["can_edit"], bool)
    assert uploads.UPLOAD_WALL.company_only is False, "uploading is not a company-only act"


def test_a_viewer_reads_every_state_and_is_told_it_may_change_none(monkeypatch) -> None:
    """The refusal is legible before the click and it is the server's own words."""
    from kairos_api import affiliation_wall

    monkeypatch.setattr(
        affiliation_wall,
        "session_for",
        lambda request: {"username": "v", "role": "viewer", "affiliation": "channel"},
    )
    stamped = uploads.UPLOAD_WALL.stamp({"inputs": []}, object())
    assert stamped["can_edit"] is False
    assert stamped["can_edit_reason"] == affiliation_wall.READ_ONLY_ROLE_DETAIL
    # A channel affiliation is not what closed it: uploading is not company-only.
    monkeypatch.setattr(
        affiliation_wall,
        "session_for",
        lambda request: {"username": "o", "role": "operator", "affiliation": "channel"},
    )
    assert uploads.UPLOAD_WALL.stamp({"inputs": []}, object())["can_edit"] is True


# --- the file audit: on disk and read by the engine are two facts --------------
def test_every_audited_file_carries_a_role_and_a_read_verdict(client: TestClient) -> None:
    body = client.get("/api/files").json()
    assert len(body["files"]) == 8, "the audited file set is the one the report counts"
    for record in body["files"]:
        assert record["role"] in {"input", "plan", "model"}
        assert isinstance(record["in_use"], bool)
        if not record["in_use"]:
            assert record["note"]["he"], f"{record['path']} is unread with no reason in Hebrew"
            assert record["note"]["en"], f"{record['path']} is unread with no reason in English"


def test_every_path_a_source_card_prints_resolves_to_a_row(client: TestClient) -> None:
    """The no-dead-end invariant, on the one name this surface prints most."""
    audit = client.get("/api/files").json()
    known = {record["path"] for record in audit["files"]} | {record["path"] for record in audit["also_read"]}
    for entry in client.get("/api/uploads/status").json()["inputs"]:
        reads = entry["engine_reads"]
        if reads is None:
            continue
        assert reads in known, f"{entry['kind']} names {reads}, which opens nothing"


def test_the_second_list_never_grows_the_audited_one(client: TestClient) -> None:
    audit = client.get("/api/files").json()
    audited = {record["path"] for record in audit["files"]}
    assert len(audited) == 8
    assert audited.isdisjoint({record["path"] for record in audit["also_read"]})
    for record in audit["also_read"]:
        assert record["exists"] is True, "a file that is not there is not being read"
        assert record["in_use"] is True


def test_the_posterior_is_marked_a_fallback_that_is_not_read(client: TestClient) -> None:
    """Measured: the measured model version resolves first, so the pkl is not in use."""
    record = next(r for r in client.get("/api/files").json()["files"] if r["path"].endswith("tv_break_posterior.pkl"))
    assert record["role"] == "model"
    assert record["in_use"] is False
    assert record["note"]["code"] == "model_fallback"


def test_the_file_audit_and_the_inputs_view_agree_on_every_shared_file(client: TestClient) -> None:
    """One verdict, two surfaces. Two derivations is how they drift."""
    files = {record["path"]: record for record in client.get("/api/files").json()["files"]}
    status = {entry["kind"]: entry for entry in client.get("/api/uploads/status").json()["inputs"]}
    for path, kind in downloads_api._FILE_KINDS.items():
        assert files[path]["in_use"] == status[kind]["in_use"], f"{path} disagrees with the {kind} input"


# --- speed: the status is read from a fingerprint, not from twenty megabytes ---
def test_the_row_counts_are_served_from_the_files_own_signature(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    read_cache.invalidate(uploads_status.SHAPE_NAMESPACE)
    reads: list[Path] = []

    def counting_reader(target: Path):
        reads.append(target)
        return (["a", "b"], 2, [])

    for _ in range(5):
        columns, rows, _ = uploads_status.file_shape(path, counting_reader)
    assert (columns, rows) == (["a", "b"], 2)
    assert len(reads) == 1, "the file was re-read while its signature had not moved"

    path.write_text("a,b\n1,2\n3,4\n5,6\n", encoding="utf-8")
    uploads_status.file_shape(path, counting_reader)
    assert len(reads) == 2, "a changed file must be re-read"


def test_the_cached_shape_cannot_be_edited_through_a_caller(tmp_path) -> None:
    path = tmp_path / "sample.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    read_cache.invalidate(uploads_status.SHAPE_NAMESPACE)
    columns, _, _ = uploads_status.file_shape(path, lambda target: (["a", "b"], 1, []))
    columns.append("c")
    again, _, _ = uploads_status.file_shape(path, lambda target: (["a", "b"], 1, []))
    assert again == ["a", "b"], "a caller edited the cached value"
