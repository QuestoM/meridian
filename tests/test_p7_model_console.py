"""The model console tells the truth about what the gates decided, and changes nothing.

Seven properties, each of which a screenshot could fake and a test cannot:

1. **The three off-states are three states.** Today every one of them renders as
   a grey "Off" chip on an operator's calendar page, and they are three
   different pieces of news. The ledger separates them from the artifacts' own
   records, and this asserts the separation on the real files.
2. **A verdict carries its basis, and the bar is not asserted anywhere.** Every
   bar is read from the engine constant that applies it, and where the artifact
   records its own bar the two must agree.
3. **Nothing regresses.** Every family the calendar renders today appears in the
   ledger with the same verdict the audience route reports.
4. **Recording a version is an act.** It never happens as a side effect of
   a read, and it is idempotent. That a read writes nothing at all is measured
   in ``test_p7_model_surface.py``, over every route on the surface.
5. **A verdict is recorded and nothing is adopted.** Recording a ship decision
   leaves both artifacts byte-identical and marks the adoption escalated.
6. **A release note cannot carry a gate verdict.** It is the only training text
   that crosses to the operator side, so the rule is enforced, not trusted.
7. **The competitor boundary** is measured over the whole surface in
   ``test_p7_model_surface.py``, which also proves every read has a screen.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from kairos_api import model_console_artifacts as artifacts
from kairos_api import model_console_coverage as coverage_module
from kairos_api import model_console_gates as gates
from kairos_api import model_console_training as training
from kairos_api import model_version_store as store

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def client(tmp_path, monkeypatch) -> TestClient:
    """The console's own routes on a throwaway store, so no test writes models/."""
    monkeypatch.setenv("KAIROS_MODEL_RELEASES_DIR", str(tmp_path / "releases"))
    from kairos_api.model_console_api import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_hashes() -> dict[str, str]:
    return {name: _sha(artifacts.MODELS_DIR / name)
            for name in (artifacts.RETENTION_ARTIFACT, artifacts.AUDIENCE_ARTIFACT)
            if (artifacts.MODELS_DIR / name).is_file()}


# ---------------------------------------------------------------------------
# 1 and 2: the states, and the basis behind each verdict
# ---------------------------------------------------------------------------


def test_the_three_off_states_are_three_states_on_the_real_artifacts() -> None:
    ledger = gates.ledger()
    counts = ledger["counts"]
    assert counts[gates.ACTIVE] > 0, "no active gate: the artifacts changed shape"
    assert counts[gates.TESTED_AND_LOST] > 0, "the tested-and-lost state has nothing in it"
    assert counts[gates.NO_CONTRAST] > 0, "the no-contrast state has nothing in it"
    assert sum(counts.values()) == ledger["total"] == len(ledger["gates"])
    # The separation must be derived, not labelled: a gate with a measured
    # figure lost, a gate without one could not run.
    for row in ledger["gates"]:
        measured = row["basis"]["value"]
        if row["state"] == gates.TESTED_AND_LOST:
            assert measured is not None, f"{row['id']} says it was tested and carries no figure"
        if row["state"] == gates.NO_CONTRAST:
            assert measured is None, f"{row['id']} says it could not run and carries a figure"


def test_every_gate_reads_its_bar_from_the_engine_constant_that_applies_it() -> None:
    for row in gates.ledger()["gates"]:
        basis = row["basis"]
        assert basis["bar"] is not None, f"{row['id']} has no bar"
        assert "." in basis["bar_source"], f"{row['id']} does not name where its bar came from"
        module, _, constant = basis["bar_source"].rpartition(".")
        imported = __import__(module, fromlist=[constant])
        assert hasattr(imported, constant), f"{basis['bar_source']} does not exist"


def test_the_artifact_and_the_engine_agree_about_the_bar_where_both_record_it() -> None:
    """Two independent sources of the same threshold, checked against each other."""
    checked = 0
    for row in gates.ledger()["gates"]:
        detail = row["basis"].get("detail")
        if not isinstance(detail, dict) or "min_relative_improvement" not in detail:
            continue
        recorded = round(float(detail["min_relative_improvement"]) * 100, 6)
        assert recorded == row["basis"]["bar"], (
            f"{row['id']}: artifact says {recorded}, engine constant says {row['basis']['bar']}")
        checked += 1
    assert checked >= 2, "no gate carried its own bar, so nothing was cross-checked"


def test_a_gate_with_no_record_reads_as_not_yet_measured() -> None:
    """The fourth state is reachable, which the shipped artifacts do not exercise."""
    rows = gates.audience_rows({}, 0)
    assert {row["state"] for row in rows} == {gates.NOT_MEASURED}
    assert all(row["basis"]["value"] is None for row in rows)


def test_layers_are_not_filed_as_gates() -> None:
    """An owner's choice must never read as a measurement."""
    ledger = gates.ledger()
    assert len(ledger["layers"]) == 3
    families = {row["family"] for row in ledger["gates"]}
    for layer in ledger["layers"]:
        assert layer["family"] not in families
        assert layer["decided_by_en"] == "an owner decision, not a gate"


# ---------------------------------------------------------------------------
# 3: nothing regresses
# ---------------------------------------------------------------------------


def test_the_ledger_shows_every_verdict_the_calendar_shows_today(client) -> None:
    """Bar 3's row for this piece, checked family by family.

    ``CalendarAudienceModel`` renders eight families from the audience gates.
    The console must carry all eight with the same verdict before the calendar
    stops showing them.
    """
    from kairos_api.audience_api import build_audience_model_payload

    calendar = build_audience_model_payload()["gates"]
    ledger = {row["family"]: row for row in client.get("/api/model/gates").json()["gates"]}
    for family, record in calendar.items():
        row = ledger.get(family)
        assert row is not None, f"the console lost the {family} verdict the calendar shows"
        expected = gates.ACTIVE if record["verdict"] == "on" else (
            gates.TESTED_AND_LOST if record.get("held_out_delta_pct") is not None
            else gates.NO_CONTRAST)
        assert row["state"] == expected, f"{family}: console says {row['state']}"
        assert row["reason"] == record["reason"], f"{family}: the reason was rewritten"


def test_the_console_carries_the_retention_gates_the_calendar_never_had() -> None:
    """The console is strictly wider than the surface it replaces."""
    ledger = gates.ledger()
    retention = {row["id"] for row in ledger["gates"] if row["model"] == "retention"}
    assert retention == {
        "retention.first_break", "retention.series", "retention.counterprogramming",
        "retention.event_layer", "retention.detrend_seasonality"}


# ---------------------------------------------------------------------------
# 4: recording a version is an act, not a side effect of a read
# ---------------------------------------------------------------------------


def test_recording_a_version_is_an_act_and_is_idempotent(client) -> None:
    assert client.get("/api/model/versions").json()["model_version"]["recorded"] is False
    first = client.post("/api/model/versions")
    assert first.status_code == 200, first.text
    assert first.json()["model_version"]["recorded"] is True
    again = client.post("/api/model/versions")
    assert again.json()["recorded"]["first_seen_at"] == first.json()["recorded"]["first_seen_at"]
    assert len(client.get("/api/model/versions").json()["observed"]) == 1


# ---------------------------------------------------------------------------
# 5 and 6: a verdict is recorded, nothing is adopted, and the note is clean
# ---------------------------------------------------------------------------


def test_recording_a_ship_verdict_moves_no_artifact_byte(client) -> None:
    before = _artifact_hashes()
    response = client.post("/api/model/decisions", json={
        "decision": "shipped", "subject": "candidate", "candidate_id": "spotclip",
        "reason": "it beats the shipped artifact on the measured plan",
        "release_note_he": "עלות השימור לברייק עלתה מעט. תוכניות שירוצו מעכשיו ימקמו פחות ברייקים ראשונים.",
        "money_direction": "up",
    })
    assert response.status_code == 200, response.text
    record = response.json()
    assert record["adoption"]["performed"] is False
    assert record["adoption"]["state"] in ("escalated", "recorded")
    assert _artifact_hashes() == before, "recording a verdict moved an artifact"


def test_a_ship_verdict_on_a_measured_candidate_escalates_with_the_figure(client, monkeypatch) -> None:
    store.save_measurement({
        "candidate_id": "spotclip",
        "fingerprint": __import__("kairos_api.model_console_candidates", fromlist=["x"])
        .measurement_fingerprint(Path(ROOT / "models/candidates/tv_break_coefficients_spotclip.json")),
        "measured_at": "2026-08-01T00:00:00+00:00",
        "operator_channel_delta": {"revenue_delta": 873395.8, "revenue_delta_pct": 2.1331},
        "scope": {"operator_channel": {"rows": 2540, "basis": "the operator's own channel"}},
    })
    record = client.post("/api/model/decisions", json={
        "decision": "shipped", "subject": "candidate", "candidate_id": "spotclip",
        "reason": "measured on the plan",
        "release_note_he": "הכנסה צפויה עולה מעט בעקבות עדכון המודל.",
    }).json()
    assert record["adoption"]["state"] == "escalated"
    assert record["adoption"]["measured_revenue_delta"] == 873395.8
    assert record["evidence"]["money_state"] == "measured"


def test_a_stale_measurement_names_the_input_that_moved(tmp_path, monkeypatch) -> None:
    """A stale figure that does not say what moved is a shrug, not a state."""
    from kairos_api import model_console_candidates as mcc

    path = ROOT / "models" / "candidates" / "tv_break_coefficients_spotclip.json"
    stored = {"inputs": {**mcc.measurement_inputs(path), "settings": "something-else"}}
    assert mcc.changed_inputs(path, stored) == ["settings"]
    stored_two = {"inputs": {**mcc.measurement_inputs(path), "settings": "a", "shipped": "b"}}
    assert mcc.changed_inputs(path, stored_two) == ["settings", "shipped"]
    assert mcc.changed_inputs(path, {"inputs": mcc.measurement_inputs(path)}) == []


def test_a_stale_row_keeps_the_measurement_it_had(tmp_path, monkeypatch) -> None:
    """A superseded figure is real, so it survives labelled rather than vanishing."""
    from kairos_api import model_console_candidates as mcc

    path = ROOT / "models" / "candidates" / "tv_break_coefficients_spotclip.json"
    stale = {
        "fingerprint": "a-fingerprint-that-does-not-match",
        "inputs": {**mcc.measurement_inputs(path), "settings": "moved"},
        "measured_at": "2026-08-01T00:00:00+00:00",
        "operator_channel_delta": {"revenue_delta": 873395.8, "revenue_delta_pct": 2.1331},
    }
    row = mcc.summary_row(path, {}, stale)
    assert row["money"]["state"] == "stale"
    assert row["money"]["changed"] == ["settings"]
    assert "ההגדרות השמורות" in row["money"]["reason_he"]
    assert row["money"]["operator_channel_delta"]["revenue_delta"] == 873395.8


def test_a_stale_row_with_no_recorded_inputs_says_so_rather_than_blaming_one(tmp_path) -> None:
    """The honest form of "I cannot tell you what moved".

    A measurement stored before the per-input digests existed cannot name what
    changed. Saying "an input changed" would read as a fact somebody checked, so
    the row says the record does not carry its inputs and offers the remedy.
    """
    from kairos_api import model_console_candidates as mcc

    path = ROOT / "models" / "candidates" / "tv_break_coefficients_spotclip.json"
    row = mcc.summary_row(path, {}, {"fingerprint": "does-not-match",
                                     "measured_at": "2026-08-01T00:00:00+00:00"})
    assert row["money"]["state"] == "stale"
    assert row["money"]["changed"] == []
    assert "אינה רושמת את הקלטים" in row["money"]["reason_he"]
    assert "does not record its inputs" in row["money"]["reason_en"]


def test_a_release_note_carrying_a_gate_verdict_is_refused(client) -> None:
    response = client.post("/api/model/decisions", json={
        "decision": "shipped", "subject": "current", "reason": "shipping",
        "release_note_he": "השער החדש עבר עם p=0.004 ולכן המקדם עודכן.",
    })
    assert response.status_code == 400
    assert "שער" in response.json()["detail"]


def test_a_decision_with_no_reason_is_refused(client) -> None:
    response = client.post("/api/model/decisions",
                           json={"decision": "not_shipped", "subject": "current", "reason": "  "})
    assert response.status_code == 400


def test_a_ship_decision_without_a_release_note_is_refused(client) -> None:
    response = client.post("/api/model/decisions", json={
        "decision": "shipped", "subject": "current", "reason": "shipping with no note"})
    assert response.status_code == 400


# ---------------------------------------------------------------------------
# The training runs: they cannot write the shipped artifact
# ---------------------------------------------------------------------------


def test_a_training_run_cannot_be_pointed_at_the_shipped_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAIROS_MODEL_RELEASES_DIR", str(tmp_path / "releases"))
    for artifact in ("retention", "audience"):
        command = training.build_command(artifact, "tr-probe")
        output = Path(command[command.index("--output") + 1])
        assert str(tmp_path) in str(output), f"{artifact} writes outside the releases store"
        assert output != artifacts.MODELS_DIR / artifacts.RETENTION_ARTIFACT
        assert output != artifacts.MODELS_DIR / artifacts.AUDIENCE_ARTIFACT


def test_a_training_run_refuses_a_flag_the_script_does_not_accept(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("KAIROS_MODEL_RELEASES_DIR", str(tmp_path / "releases"))
    with pytest.raises(training.TrainingError):
        training.build_command("retention", "tr-probe", {"--invent": "force-on"})
    with pytest.raises(training.TrainingError):
        training.build_command("retention", "tr-probe", {"--series": "maybe"})
    command = training.build_command("retention", "tr-probe", {"--series": "force-on"})
    assert "--series" in command and "force-on" in command


def test_the_training_flags_are_the_scripts_own_five() -> None:
    flags = {row["flag"] for row in training.available_flags("retention")}
    assert flags == {"--series", "--counterprogramming", "--placebo-correction",
                     "--interval-calibration", "--moderated-variances"}
    assert "--output" not in flags


# ---------------------------------------------------------------------------
# Coverage, the activation mirror, and the competitor boundary
# ---------------------------------------------------------------------------


def test_every_blocked_gate_names_its_condition_and_its_source(client) -> None:
    blocked = client.get("/api/model/coverage").json()["blocked"]
    assert blocked, "the register is empty while gates report no contrast"
    for row in blocked:
        assert row["condition_he"].strip()
        assert row["condition_en"].strip()
        assert row["source"].strip()
        assert row["earliest_state"] in ("dated", "unknown")
        if row["earliest_state"] == "dated":
            assert row["earliest"]["start"], row["gate_id"]


def test_the_console_mirrors_the_activation_switch_and_carries_no_control(client) -> None:
    activation = client.get("/api/model/console").json()["activation"]
    assert activation["control_lives_on"] == "rules"
    assert "can_edit" not in activation
    assert activation["state"] in ("off", "on", "on_no_artifact")


def test_the_wartime_headline_is_computed_from_the_artifact(client) -> None:
    window = client.get("/api/model/coverage").json()["window"]
    assert window["available"] is True
    metadata = artifacts.retention_metadata()
    assert window["total_breaks_measured"] == metadata["total_breaks_measured"]
    assert str(window["post_ceasefire_breaks"]) in window["headline_he"]
    assert window["post_ceasefire_pct"] == round(
        100.0 * window["post_ceasefire_breaks"] / window["total_breaks_measured"], 2)


def test_the_retention_contrast_is_read_from_the_artifacts_own_cells() -> None:
    payload = artifacts.read_artifact(artifacts.RETENTION_ARTIFACT) or {}
    contrast = coverage_module.retention_contrast()
    assert contrast["cells"] == len(payload["detail"])
    assert contrast["observations"] == payload["metadata"]["total_breaks_measured"]
    assert contrast["contrast_ratio"] == round(
        payload["metadata"]["between_cell_variance_tau2"]
        / payload["metadata"]["pooled_within_variance"], 6)
