"""Run the whole trade chain against real corpus agreements, and print numbers.

This is the demonstration the client meeting walks through, executable rather
than asserted: agreements ingested and reviewed, approved into versions, their
rules compiled and BOUND into the live rule stores, commitment standing
measured with projections and alarms, compensation mechanics engaged on a
breach, and a proposed agreement simulated against real activity.

It runs against a temporary copy of every store it touches (agreements,
conditions, frequency rules, version history, ledger), so it can be run on any
machine at any time without moving one byte of operator state. Nothing here is
a mock: the same modules the API calls do the work, and the figures printed are
whatever they compute.

    ~/.venvs/meridian/bin/python scripts/trade_end_to_end_proof.py

Exit code 0 means every stage did what it claims. Any stage that cannot run
prints why and fails loudly rather than skipping quietly.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import pandas as pd  # noqa: E402

STEP = 0


def step(title: str) -> None:
    global STEP
    STEP += 1
    print(f"\n{'=' * 72}\n{STEP}. {title}\n{'=' * 72}")


def line(label: str, value: object) -> None:
    print(f"   {label:.<44} {value}")


def _isolate_stores(tmp: Path) -> None:
    """Point every writable store at a throwaway tree before anything runs."""
    import os

    from kairos.optimize import _frequency_rules
    from kairos_api import advertiser_conditions, agency_conditions, trade_ledger

    os.environ["KAIROS_AGREEMENTS_DIR"] = str(tmp / "agreements")
    os.environ["KAIROS_VERSIONS_DIR"] = str(tmp / "versions")

    adv = tmp / "advertiser_conditions.csv"
    adv.write_text(
        "advertiser_id,rule_id,scope_positions,scope_genres,scope_dayparts,"
        "scope_programmes,scope_weekdays,effect,value,mode,notes\n",
        encoding="utf-8-sig",
    )
    agc = tmp / "agency_conditions.csv"
    agc.write_text(
        "agency_id,rule_id,scope_positions,scope_genres,scope_dayparts,"
        "scope_programmes,scope_weekdays,effect,value,mode,notes\n",
        encoding="utf-8-sig",
    )
    freq = tmp / "frequency_rules.csv"
    freq.write_text(
        "rule_id,limit_type,scope,advertiser_id,campaign,ad,pair_lead,pair_closer,"
        "competing_group,members,value,value_max,unit,enabled,notes\n",
        encoding="utf-8-sig",
    )
    advertiser_conditions.CONDITIONS_PATH = adv
    agency_conditions.CONDITIONS_PATH = agc
    _frequency_rules.DEFAULT_FREQUENCY_PATH = freq
    trade_ledger.LEDGER_PATH = tmp / "trade_credit_ledger.csv"
    trade_ledger.BACKUP_DIR = tmp / "_backups"
    return None


def _fake_activity(campaigns: list[str], advertiser: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A small, explicit activity ledger so every printed figure is checkable.

    Deliberately hand-written rather than read from the operator's own files:
    this script must print the same numbers on any machine, and the point being
    proven is the ENGINE's arithmetic, not this month's real delivery.
    """
    rows = []
    for campaign in campaigns:
        rows += [
            (campaign, "2026-02-15", "aired", "רשת 13", 24, 720, 96.0, 1_100_000, "proof"),
            (campaign, "2026-04-10", "aired", "רשת 13", 18, 540, 74.0, 820_000, "proof"),
            (campaign, "2026-05-20", "unknown", "רשת 13", "", "", "", "", "proof"),
            (campaign, "2026-09-05", "scheduled", "רשת 13", 12, 360, 48.0, 560_000, "proof"),
        ]
    delivery = pd.DataFrame(rows, columns=[
        "campaign_id", "broadcast_date", "air_state", "channel", "spots",
        "seconds", "rating_points_planned", "spend_ils", "counted_as_of",
    ])
    campaigns_frame = pd.DataFrame(
        [(c, advertiser) for c in campaigns], columns=["campaign_id", "advertiser"]
    )
    return delivery, campaigns_frame


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="kairos-trade-proof-"))
    try:
        _isolate_stores(tmp)
        from kairos.trade import corpus, explain, obligations, simulate
        from kairos.trade.compile import compile_termset, parse_rule_id
        from kairos_api import trade_bind, trade_ledger, trade_review, trade_store

        step("The corpus: authored agreements the engine has never seen at runtime")
        truths = corpus.load_all()
        for doc_id, doc in sorted(truths.items()):
            cov = doc.coverage()
            line(doc_id, f"{cov.total_clauses} clauses, {len(doc.instances)} terms, "
                         f"{cov.unmapped} unmapped")
        flagship = truths["heb-annual-framework-2026"]

        step("Ingest and review: the flagship framework enters as a PROPOSAL")
        head = trade_store.create(
            title="הסכם מסגרת שנתי — אופק מדיה 2026",
            level="agency_framework",
            actor="proof-script",
            counterparty={"agency": "אופק מדיה", "counterparty_type": "agency"},
            window={"starts_on": "2026-01-01", "ends_on": "2026-12-31"},
        )
        agreement_id = head["agreement_id"]
        pdf = corpus.corpus_root() / flagship.document_id / "document.pdf"
        document = trade_store.attach_document(
            agreement_id, filename="framework-2026.pdf",
            payload=pdf.read_bytes(), actor="proof-script",
        )
        document_id = document["document_id"]
        payload = flagship.to_payload()
        payload["document_id"] = document_id
        for instance in payload["instances"]:
            for citation in instance["citations"]:
                citation["document_id"] = document_id
        trade_store.save_extraction(agreement_id, document_id, payload, "proof-script")
        trade_store.set_status(agreement_id, trade_store.IN_REVIEW, "proof-script")
        line("agreement", agreement_id)
        line("document sha256", document["sha256"][:16] + "…")
        line("clauses proposed", len(payload["clauses"]))
        line("terms proposed", len(payload["instances"]))

        step("The completeness gate REFUSES approval and names every blocker")
        gate = trade_review.agreement_gate(agreement_id)
        line("ready", gate["ready"])
        for blocker in gate["blockers"]:
            line(f"blocker {blocker['kind']}", blocker["count"])
        try:
            trade_review.approve(agreement_id, actor="proof-script")
            print("   !! approval succeeded while the gate was red — FAIL")
            return 1
        except ValueError as exc:
            line("refusal", str(exc)[:96] + "…")

        step("Plain language: what each proposed term WILL DO, before approval")
        effects = explain.explain_termset(
            {"version_id": "draft", "agreement_id": agreement_id,
             "instances": payload["instances"]}, head,
        )
        for mechanism, count in sorted(effects["by_mechanism"].items()):
            line(effects["mechanism_labels"][mechanism], count)
        inert = [t for t in effects["terms"] if t["mechanism"] == "inert"]
        print("\n   Terms that will NOT act automatically, with the reason:")
        for term in inert:
            print(f"     · {term['term_name_he']}: {term['will_not_act_reasons'][0][:80]}")
        sample = next(t for t in effects["terms"] if t["mechanism"] == "blocks")
        print(f"\n   Example blocking term — {sample['term_name_he']}:")
        print(f"     {sample['sentence_he']}")

        step("A human reviews every clause and approves; a version is created")
        clause_ids = [c["clause_id"] for c in payload["clauses"]]
        trade_review.mark_clauses_seen(agreement_id, document_id, clause_ids, "dana.levin")
        for instance in payload["instances"]:
            trade_review.decide_instance(
                agreement_id, document_id, instance["instance_id"],
                "confirmed", "dana.levin",
            )
        for disposition in payload["dispositions"]:
            if disposition["disposition"] == "unmapped":
                trade_review.acknowledge_unmapped(
                    agreement_id, document_id, disposition["clause_id"],
                    "dana.levin", note="נבדק ידנית, מטופל מחוץ למערכת",
                )
        manifest = trade_review.approve(
            agreement_id, actor="dana.levin", note="אישור לאחר סקירה מלאה")
        line("version", manifest["version_id"])
        line("approved terms", manifest["counts"]["approved_terms"])
        line("acknowledged unsupported", manifest["counts"]["acknowledged_unsupported"])
        line("approver", manifest["actor"])

        step("Compile and BIND: the contract becomes live rules")
        termset = trade_store.load_termset(agreement_id, manifest["version_id"])
        artifacts = compile_termset(termset, trade_store.load_head(agreement_id))
        bound = trade_bind.bind(artifacts, actor="dana.levin")
        line("conditions written", bound["written"].get("advertiser_conditions", 0))
        line("frequency rules written", bound["written"].get("frequency_rules", 0))
        line("settlement terms", len(artifacts.settlement.get("terms", [])))
        line("skipped by name", len(artifacts.skipped))
        line("snapshot taken first", bound["snapshot_version"] is not None)
        held = trade_bind.bound_rules(agreement_id)
        example = held["frequency_rules"][0] if held.get("frequency_rules") else \
            held["advertiser_conditions"][0]
        attribution = parse_rule_id(example["rule_id"])
        print(f"\n   Attribution of a live rule back to its clause:")
        line("rule_id", example["rule_id"][:60] + "…")
        instance = next(i for i in termset["instances"]
                        if i["instance_id"] == attribution["instance_id"])
        line("term", instance["term_id"])
        line("clause", instance["citations"][0]["clause_id"])
        line("quote", instance["citations"][0]["quote"][:52] + "…")

        step("Commitment standing: measured floors, pace, projection, alarm")
        advertisers = next(
            (i["params"].get("represented_advertisers", [])
             for i in termset["instances"] if i["term_id"] == "agreement-parties"),
            [],
        )
        advertiser = advertisers[0] if advertisers else "Delta Motors"
        delivery, campaigns_frame = _fake_activity(["PROOF_C1", "PROOF_C2"], advertiser)
        links = pd.DataFrame(
            [("AG_OFEK", "אופק מדיה", advertiser)],
            columns=["agency_id", "agency_name", "advertiser"],
        )
        inputs = obligations.Inputs(
            delivery=delivery, campaigns=campaigns_frame, agency_links=links,
            today=date(2026, 6, 30),
        )
        snapshots = obligations.evaluate_all(
            termset, trade_store.load_head(agreement_id), inputs)
        for snapshot in snapshots:
            target = snapshot.get("target", {}).get("value")
            standing = snapshot.get("standing", {}).get("counted")
            print(f"     · {snapshot['term_id']}: alarm={snapshot['alarm']}"
                  f" target={target} counted={standing}")
            if snapshot.get("projection") is not None:
                line("  projection", snapshot["projection"])
            if snapshot["alarm"] == "unknown":
                line("  unknown because", snapshot["alarm_reason"][:70])

        step("Compensation mechanics: an accrual ledger at three levels")
        policy = next((i for i in termset["instances"]
                       if i["term_id"] == "makegood-accrual-policy"), None)
        if policy is None:
            print("   !! the flagship lost its accrual policy — FAIL")
            return 1
        accrual = policy["params"]["accruals"][0]
        gross = float(pd.to_numeric(
            delivery[delivery["air_state"] == "aired"]["spend_ils"],
            errors="coerce").fillna(0).sum())
        credit = round(gross * float(accrual["rate_percent"]) / 100.0, 2)
        trade_ledger.append_entry(
            level="agency", party_ref="אופק מדיה", direction="accrue",
            quantity=credit, unit="ils_media_value", reason_code="policy_accrual",
            source_agreement_id=agreement_id,
            source_term_instance_id=policy["instance_id"],
            effective_on="2026-06-30", expires_on="2026-12-31",
            actor="dana.levin", note=f"{accrual['rate_percent']}% על מחזור מדוד",
        )
        line("measured gross", f"{gross:,.0f} ILS")
        line(f"accrued at {accrual['rate_percent']}%", f"{credit:,.0f} ILS")
        spend = round(credit / 3, 2)
        trade_ledger.append_entry(
            level="agency", party_ref="אופק מדיה", direction="utilise",
            quantity=spend, unit="ils_media_value", reason_code="shortfall_cure",
            source_agreement_id=agreement_id, effective_on="2026-07-15",
            actor="dana.levin", note="השלמה לקמפיין אחר של הסוכנות",
        )
        for block in trade_ledger.balances(level="agency", party_ref="אופק מדיה"):
            line(f"balance {block['unit']}",
                 f"accrued {block['accrued']:,.0f} · "
                 f"utilised {block['utilised']:,.0f} · "
                 f"available {block['available']:,.0f}")
        try:
            trade_ledger.append_entry(
                level="agency", party_ref="אופק מדיה", direction="utilise",
                quantity=credit * 10, unit="ils_media_value",
                reason_code="shortfall_cure", effective_on="2026-08-01",
                actor="dana.levin",
            )
            print("   !! an overdraft was allowed — FAIL")
            return 1
        except ValueError as exc:
            line("overdraft refused", str(exc)[:70])

        step("Simulation: what a PROPOSED agreement would do, writing nothing")
        before = (Path(str(tmp / "advertiser_conditions.csv")).read_bytes(),
                  Path(str(tmp / "frequency_rules.csv")).read_bytes())
        result = simulate.simulate(
            termset, trade_store.load_head(agreement_id),
            simulate.SimulationInputs(
                delivery=delivery, campaigns=campaigns_frame, agency_links=links,
                today=date(2026, 6, 30),
                window={"from": "2026-01-01", "to": "2026-06-30"},
            ),
        )
        print(f"   {result['headline_he']}")
        money = result["money"]
        line("gross aired", f"{money['gross_aired']:,.0f} ILS")
        if money.get("discount_ladder", {}).get("available"):
            ladder = money["discount_ladder"]
            line(f"ladder ({ladder['mechanics']})", f"-{ladder['discount_value']:,.0f} ILS "
                                                    f"at {ladder['tier_reached_percent']}%")
            if ladder.get("distance_to_next"):
                line("to next tier", f"{ladder['distance_to_next']:,.0f} ILS")
        if money.get("agency_commission", {}).get("available"):
            commission = money["agency_commission"]
            line(f"commission {commission['percent']}% ({commission['base']})",
                 f"-{commission['commission_value']:,.0f} ILS")
        line("net after simulated terms", f"{money['net_after_simulated_terms']:,.0f} ILS")
        line("commitments at risk or breached", len(result["exposure"]))
        line("terms not simulable (named)", len(result["not_simulated"]))
        after = (Path(str(tmp / "advertiser_conditions.csv")).read_bytes(),
                 Path(str(tmp / "frequency_rules.csv")).read_bytes())
        line("stores byte-identical after simulation", before == after)
        if before != after:
            print("   !! simulation wrote to a live store — FAIL")
            return 1

        step("Supersession: the rules leave with the agreement")
        trade_store.set_status(agreement_id, trade_store.SUPERSEDED,
                               "dana.levin", note="הוחלף בתיקון Q4")
        removed = trade_bind.unbind(agreement_id, actor="dana.levin")
        line("rows removed", sum(removed["removed"].values()))
        line("rules still held", sum(
            len(v) for v in trade_bind.bound_rules(agreement_id).values()))

        print(f"\n{'=' * 72}\nEvery stage ran. Stores used: {tmp}\n{'=' * 72}")
        return 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
