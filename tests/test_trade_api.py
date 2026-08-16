"""The trade engine over HTTP, end to end without a provider.

The extraction stage is exercised through the runner's own tests; here the
proposal is injected directly so the route layer's real contract is what is
measured: the gate refuses approval and names why, approval creates a version
AND makes rules bind, a superseding status takes them away again, and any
live rule id resolves back to the clause that created it.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from kairos.optimize import _frequency_rules
    from kairos_api import advertiser_conditions, agency_conditions, trade_store

    monkeypatch.setenv(trade_store.AGREEMENTS_DIR_ENV, str(tmp_path / "agreements"))
    monkeypatch.setenv("KAIROS_VERSIONS_DIR", str(tmp_path / "versions"))
    adv = tmp_path / "advertiser_conditions.csv"
    adv.write_text(
        "advertiser_id,rule_id,scope_positions,scope_genres,scope_dayparts,"
        "scope_programmes,scope_weekdays,effect,value,mode,notes\n",
        encoding="utf-8-sig",
    )
    agc = tmp_path / "agency_conditions.csv"
    agc.write_text(
        "agency_id,rule_id,scope_positions,scope_genres,scope_dayparts,"
        "scope_programmes,scope_weekdays,effect,value,mode,notes\n",
        encoding="utf-8-sig",
    )
    freq = tmp_path / "frequency_rules.csv"
    freq.write_text(
        "rule_id,limit_type,scope,advertiser_id,campaign,ad,pair_lead,pair_closer,"
        "competing_group,members,value,value_max,unit,enabled,notes\n",
        encoding="utf-8-sig",
    )
    monkeypatch.setattr(advertiser_conditions, "CONDITIONS_PATH", adv)
    monkeypatch.setattr(agency_conditions, "CONDITIONS_PATH", agc)
    monkeypatch.setattr(_frequency_rules, "DEFAULT_FREQUENCY_PATH", freq)

    from kairos_api.server import app

    with TestClient(app) as test_client:
        test_client._paths = {"adv": adv, "freq": freq}  # type: ignore[attr-defined]
        yield test_client


def _proposal(document_id: str) -> dict:
    return {
        "document_id": document_id,
        "clauses": [
            {"clause_id": "5.4", "text": "תשדירי המפרסם לא ישובצו במקבץ הצמוד לסיקור אסונות.", "pages": [1]},
            {"clause_id": "5.6", "text": "לא ישודר אותו תשדיר יותר מארבע פעמים ביממת שידור.", "pages": [1]},
            {"clause_id": "9.9", "text": "ולראיה באו הצדדים על החתום", "pages": [2]},
            {"clause_id": "7.7", "text": "הסוכנות רשאית למכור מלאי לצד שלישי.", "pages": [2]},
        ],
        "instances": [
            {
                "instance_id": "i-adj", "term_id": "content-adjacency-exclusion",
                "params": {"excluded_content": ["סיקור אסונות"], "radius": "same_break", "hard": True},
                "citations": [{"document_id": document_id, "page": 1, "clause_id": "5.4",
                               "quote": "לא ישובצו במקבץ הצמוד לסיקור אסונות"}],
                "confidence": "high", "scope": {}, "window": {}, "missing": [], "notes": "",
            },
            {
                "instance_id": "i-freq", "term_id": "frequency-caps",
                "params": {"unit": "day", "cap": 4},
                "citations": [{"document_id": document_id, "page": 1, "clause_id": "5.6",
                               "quote": "יותר מארבע פעמים ביממת שידור"}],
                "confidence": "high", "scope": {}, "window": {}, "missing": [], "notes": "",
            },
        ],
        "dispositions": [
            {"clause_id": "5.4", "disposition": "mapped", "instance_ids": ["i-adj"]},
            {"clause_id": "5.6", "disposition": "mapped", "instance_ids": ["i-freq"]},
            {"clause_id": "9.9", "disposition": "irrelevant",
             "irrelevant_class": "signature-block", "reason": "חתימות"},
            {"clause_id": "7.7", "disposition": "unmapped",
             "reason": "זכות מכירה משנית שאין לה מונח נתמך"},
        ],
    }


def _agreement_with_proposal(client):
    created = client.post("/api/trade/agreements", json={
        "title": "הסכם טכנו-קור 2026", "level": "advertiser",
        "counterparty": {"advertiser": "טכנו-קור"},
        "window": {"starts_on": "2026-01-01", "ends_on": "2026-12-31"},
    })
    assert created.status_code == 200, created.text
    aid = created.json()["agreement_id"]
    uploaded = client.post(
        f"/api/trade/agreements/{aid}/documents",
        files={"file": ("deal.pdf", b"%PDF-1.4 test", "application/pdf")},
    )
    assert uploaded.status_code == 200, uploaded.text
    doc_id = uploaded.json()["document_id"]
    from kairos_api import trade_store

    trade_store.save_extraction(aid, doc_id, _proposal(doc_id), "tester")
    trade_store.set_status(aid, trade_store.IN_REVIEW, "tester")
    return aid, doc_id


def test_gate_blocks_approval_and_names_the_blockers(client):
    aid, doc_id = _agreement_with_proposal(client)
    gate = client.get(f"/api/trade/agreements/{aid}/gate").json()
    assert gate["ready"] is False
    kinds = {b["kind"] for b in gate["blockers"]}
    assert kinds == {"clauses_unseen", "instances_undecided", "unmapped_unacknowledged"}
    refused = client.post(f"/api/trade/agreements/{aid}/approve", json={})
    assert refused.status_code == 422
    assert "completeness gate" in refused.json()["detail"]


def test_approval_binds_rules_and_supersession_removes_them(client):
    aid, doc_id = _agreement_with_proposal(client)
    client.post(f"/api/trade/agreements/{aid}/documents/{doc_id}/seen",
                json={"clause_ids": ["5.4", "5.6", "9.9", "7.7"]})
    for iid in ("i-adj", "i-freq"):
        r = client.post(
            f"/api/trade/agreements/{aid}/documents/{doc_id}/instances/{iid}/decide",
            json={"verdict": "confirmed"})
        assert r.status_code == 200, r.text
    ack = client.post(
        f"/api/trade/agreements/{aid}/documents/{doc_id}/clauses/7.7/acknowledge",
        json={"note": "מכירה משנית מנוהלת מחוץ למערכת"})
    assert ack.status_code == 200

    approved = client.post(f"/api/trade/agreements/{aid}/approve",
                           json={"note": "אישור ראשון"})
    assert approved.status_code == 200, approved.text
    body = approved.json()
    assert body["compiled"]["summary"]["conditions"] == 1
    assert body["bound"]["written"] == {"advertiser_conditions": 1, "frequency_rules": 1}

    adv_text = Path(client._paths["adv"]).read_text(encoding="utf-8-sig")
    assert "טכנו-קור" in adv_text and "forbid" in adv_text
    freq_text = Path(client._paths["freq"]).read_text(encoding="utf-8-sig")
    assert "max_per_day" in freq_text

    detail = client.get(f"/api/trade/agreements/{aid}").json()
    assert detail["agreement"]["status"] == "approved"
    assert set(detail["bound_rules"]) == {"advertiser_conditions", "frequency_rules"}

    superseded = client.post(f"/api/trade/agreements/{aid}/status",
                             json={"target": "superseded", "note": "הוחלף"})
    assert superseded.status_code == 200
    assert superseded.json()["unbound"]["removed"] == {
        "advertiser_conditions": 1, "frequency_rules": 1}
    assert "טכנו-קור" not in Path(client._paths["adv"]).read_text(encoding="utf-8-sig")


def test_any_bound_rule_resolves_back_to_its_clause(client):
    aid, doc_id = _agreement_with_proposal(client)
    client.post(f"/api/trade/agreements/{aid}/documents/{doc_id}/seen",
                json={"clause_ids": ["5.4", "5.6", "9.9", "7.7"]})
    for iid in ("i-adj", "i-freq"):
        client.post(
            f"/api/trade/agreements/{aid}/documents/{doc_id}/instances/{iid}/decide",
            json={"verdict": "confirmed"})
    client.post(f"/api/trade/agreements/{aid}/documents/{doc_id}/clauses/7.7/acknowledge",
                json={"note": "מנוהל ידנית"})
    client.post(f"/api/trade/agreements/{aid}/approve", json={})

    freq_text = Path(client._paths["freq"]).read_text(encoding="utf-8-sig")
    rule_id = next(line.split(",")[0] for line in freq_text.splitlines()
                   if line.startswith("TRD:"))
    resolved = client.get(f"/api/trade/attribution/{rule_id}").json()
    assert resolved["trade_rule"] is True and resolved["resolved"] is True
    assert resolved["agreement_title"] == "הסכם טכנו-קור 2026"
    assert resolved["term"]["term_id"] == "frequency-caps"
    assert resolved["term"]["citations"][0]["clause_id"] == "5.6"
    assert "ארבע פעמים" in resolved["term"]["citations"][0]["quote"]

    plain = client.get("/api/trade/attribution/R_MANUAL_ROW").json()
    assert plain == {"trade_rule": False}


def test_obligations_need_an_approved_version_and_report_honestly(client):
    aid, doc_id = _agreement_with_proposal(client)
    before = client.get(f"/api/trade/agreements/{aid}/obligations").json()
    assert before["available"] is False
    assert "מאושר" in before["reason"]


def test_a_reviewer_edit_is_recorded_beside_the_extraction(client):
    aid, doc_id = _agreement_with_proposal(client)
    edited = client.post(
        f"/api/trade/agreements/{aid}/documents/{doc_id}/instances/i-freq/decide",
        json={"verdict": "edited", "edited_params": {"unit": "day", "cap": 3}})
    assert edited.status_code == 200
    proposal = client.get(
        f"/api/trade/agreements/{aid}/documents/{doc_id}/proposal").json()
    entry = proposal["review"]["instances"]["i-freq"]
    assert entry["state"] == "edited"
    assert entry["edited_params"]["cap"] == 3
    original = next(i for i in proposal["extraction"]["instances"]
                    if i["instance_id"] == "i-freq")
    assert original["params"]["cap"] == 4, "the extraction's own value is untouched"


def test_unknown_agreement_is_a_404_not_a_500(client):
    assert client.get("/api/trade/agreements/agr-nope").status_code == 404
    assert client.get("/api/trade/agreements/agr-nope/gate").status_code == 404
