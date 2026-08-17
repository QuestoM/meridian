"""Seed the agreement store from the measured trade corpus.

Development fixture, not product data. Every clause, instance, citation and
disposition below comes from tests/trade_corpus (the authored ground truth for
the extraction harness), loaded through the same corpus loader the accuracy
tests use, and written through the same store the API writes through. Nothing
here is invented: confidence, missing-field gaps, dispositions and conflicts
are whatever the corpus says they are.

Run from the repository root:

    ~/.venvs/meridian/bin/python scripts/seed_trade_agreements.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kairos.trade import corpus  # noqa: E402
from kairos_api import trade_review, trade_store  # noqa: E402

ACTOR = "seed-script"
CORPUS = ROOT / "tests" / "trade_corpus" / "agreements"


def _payload_for(corpus_id: str, document_id: str) -> dict:
    """Load one corpus document and re-key it onto the stored document id."""
    doc = corpus.load_corpus_document(CORPUS / corpus_id)
    payload = doc.to_payload()
    payload["document_id"] = document_id
    for instance in payload.get("instances", []):
        for citation in instance.get("citations", []):
            citation["document_id"] = document_id
    conflicts = []
    for index, raw in enumerate(payload.get("stats", {}).get("expected_conflicts", [])):
        conflicts.append({
            "conflict_id": f"cf-{index + 1}",
            "instances": raw.get("instances", []),
            "contested": raw.get("contested", ""),
            "resolution": raw.get("resolution"),
            "winner": raw.get("winner"),
            "rule": raw.get("via", ""),
            "explanation_he": raw.get("via", ""),
        })
    payload.setdefault("stats", {})["conflicts"] = conflicts
    return payload


def _seed_conflicts(agreement_id: str, document_id: str, payload: dict) -> None:
    conflicts = payload.get("stats", {}).get("conflicts", [])
    if not conflicts:
        return
    review = trade_store.load_review(agreement_id, document_id)
    for conflict in conflicts:
        review.setdefault("conflicts", {})[conflict["conflict_id"]] = {
            "instances": conflict.get("instances", []),
            "contested": conflict.get("contested", ""),
            "resolution": conflict.get("resolution"),
            "winner": conflict.get("winner"),
            "rule": conflict.get("rule"),
            "explanation_he": conflict.get("explanation_he", ""),
        }
    trade_store.save_review(agreement_id, document_id, review)


def seed(*, corpus_id: str, title: str, level: str, counterparty: dict,
         window: dict, extract: bool = True, review_all: bool = False,
         approve: bool = False) -> str:
    head = trade_store.create(
        title=title, level=level, actor=ACTOR,
        counterparty=counterparty, window=window,
        note="נטען מתיק הבדיקה המדוד לצורך פיתוח הממשק",
    )
    agreement_id = head["agreement_id"]
    entry = trade_store.attach_document(
        agreement_id,
        filename=f"{corpus_id}.pdf",
        payload=(CORPUS / corpus_id / "document.pdf").read_bytes(),
        actor=ACTOR,
    )
    document_id = entry["document_id"]
    if not extract:
        print(f"  {agreement_id}: draft, document attached, no proposal")
        return agreement_id

    payload = _payload_for(corpus_id, document_id)
    trade_store.save_extraction(agreement_id, document_id, payload, ACTOR)
    _seed_conflicts(agreement_id, document_id, payload)
    trade_store.set_status(agreement_id, trade_store.IN_REVIEW, ACTOR,
                           note="הצעת מיפוי מוכנה לסקירה")

    if review_all:
        _review_everything(agreement_id, document_id, payload)
    gate = trade_review.agreement_gate(agreement_id)
    if approve and gate["ready"]:
        manifest = trade_review.approve(agreement_id, ACTOR, note="אושר בסקירת הבדיקה")
        _compile_and_bind(agreement_id, manifest)
    gate = trade_review.agreement_gate(agreement_id)
    print(f"  {agreement_id}: {trade_store.load_head(agreement_id)['status']}, "
          f"{len(payload['clauses'])} clauses, {len(payload['instances'])} instances, "
          f"gate ready={gate['ready']} blockers={[b['kind'] for b in gate['blockers']]}")
    return agreement_id


def _review_everything(agreement_id: str, document_id: str, payload: dict) -> None:
    """Walk the whole review the way a reviewer would, through the real API."""
    trade_review.mark_clauses_seen(
        agreement_id, document_id,
        [c["clause_id"] for c in payload["clauses"]], ACTOR)
    for instance in payload["instances"]:
        trade_review.decide_instance(
            agreement_id, document_id, instance["instance_id"], "confirmed", ACTOR)
    for disposition in payload["dispositions"]:
        if disposition["disposition"] == "unmapped":
            trade_review.acknowledge_unmapped(
                agreement_id, document_id, disposition["clause_id"], ACTOR,
                "נקרא ואומת: אין מונח בטקסונומיה שמייצג את הסעיף, והוא נשאר מעקב אנושי")
    review = trade_store.load_review(agreement_id, document_id)
    for conflict_id, entry in review.get("conflicts", {}).items():
        if entry.get("resolution") in ("resolved_by_rule", "resolved_by_human"):
            continue
        trade_review.resolve_conflict(
            agreement_id, document_id, conflict_id, entry["instances"][0], ACTOR,
            note="הוכרע בסקירה: הגרסה המוקדמת גוברת בהיעדר לשון עדיפות")


def _compile_and_bind(agreement_id: str, manifest: dict) -> None:
    import json

    from kairos.trade.compile import compile_termset
    from kairos_api import trade_bind

    head = trade_store.load_head(agreement_id)
    termset = trade_store.load_termset(agreement_id, manifest["version_id"])
    artifacts = compile_termset(termset, head)
    bound = trade_bind.bind(artifacts, ACTOR)
    directory = trade_store.versions_dir(agreement_id) / manifest["version_id"]
    (directory / "compiled.json").write_text(
        json.dumps({
            "conditions": artifacts.conditions,
            "frequency_rules": artifacts.frequency_rules,
            "settlement": artifacts.settlement,
            "skipped": artifacts.skipped,
            "bound": bound,
        }, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    print(f"    approved {manifest['version_id']}: bound {bound}")


def main() -> None:
    print(f"Seeding into {trade_store.agreements_root()}")
    # Windows use starts_on/ends_on because that is what the store normalises
    # and what every agreement created through the API carries. One agreement
    # below is deliberately open-ended, so the surface is exercised against the
    # FOREVER marker rather than only against closed windows.
    seed(
        corpus_id="heb-annual-framework-2026",
        title="הסכם מסגרת שנתי — אופק מדיה 2026",
        level="agency_framework",
        counterparty={"counterparty_type": "agency", "agency": "אופק מדיה בע\"מ"},
        window={"starts_on": "2026-01-01", "ends_on": "2026-12-31"},
    )
    seed(
        corpus_id="heb-contradictory-2026",
        title="הסכם מסגרת — קבוצת ריטייל 2026",
        level="agency_framework",
        counterparty={"counterparty_type": "agency", "agency": "קבוצת ריטייל מדיה"},
        window={"starts_on": "2026-01-01", "ends_on": "2026-12-31"},
    )
    seed(
        corpus_id="heb-edge-stress-2026",
        title="הסכם מפרסם — נובה פארם 2026",
        level="advertiser",
        counterparty={"counterparty_type": "advertiser", "advertiser": "Nova Pharm בע\"מ"},
        window={"starts_on": "2026-03-01", "ends_on": "2027-02-28"},
    )
    seed(
        corpus_id="heb-scanned-smallbiz-2026",
        title="הסכם מפרסם — מאפיית שדות 2026",
        level="advertiser",
        counterparty={"counterparty_type": "advertiser", "advertiser": "מאפיית שדות"},
        window={"starts_on": "2026-04-01", "ends_on": "2026-09-30"},
        review_all=True,
        approve=True,
    )
    seed(
        corpus_id="heb-sponsorship-bundle-2026",
        title="חבילת חסויות — בנק הבירה 2026",
        level="advertiser",
        counterparty={"counterparty_type": "advertiser", "advertiser": "בנק הבירה בע\"מ"},
        window={"starts_on": "2026-05-01", "ends_on": None},
        extract=False,
    )


if __name__ == "__main__":
    main()
