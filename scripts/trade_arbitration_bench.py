"""Three readings of the same corpus, scored against the same truth.

The question this answers is not rhetorical and was asked directly: would
handing the whole agreement to one large model, under a strict schema, beat the
clause-by-clause pipeline? And when the two disagree, can a third model holding
everything simply rule, instead of handing the disagreement to a person?

So all three are measured on the SAME eight documents against the SAME
independently authored ground truth the accuracy report uses:

  A  pipeline     clause-by-clause, the shipped reader
  B  whole        one call, the entire document, reasoning tier
  C  arbitrated   A and B aligned, every disagreement ruled by a third call
                  that holds the document, the taxonomy and both candidates

Nothing here is a simulation: every number comes from real provider calls
against the real corpus PDFs, and the same scorer as the shipped report.

    ~/.venvs/meridian/bin/python scripts/trade_arbitration_bench.py [DOC_ID ...]
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kairos.trade import (arbitrate, corpus, extract_provider, extract_run,  # noqa: E402
                          extract_wholedoc)
from kairos.trade.documents import DocumentExtraction  # noqa: E402
from scripts.trade_extraction_accuracy import score_document  # noqa: E402

OUT_MD = ROOT / "docs" / "trade" / "arbitration-accuracy.md"
OUT_JSON = ROOT / "docs" / "trade" / "arbitration-accuracy.json"


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.1f}%"


def _as_extraction(base: DocumentExtraction, instances: list[Any]) -> DocumentExtraction:
    """The same document and clause ledger, carrying a different reading.

    Clauses and dispositions are the segmenter's either way - that is the point
    of the constraint - so only the instances move. Dispositions are re-derived
    from the instances so coverage stays honest for each reading.
    """
    from kairos.trade.documents import ClauseDisposition

    cited: dict[str, list[str]] = {}
    for inst in instances:
        for cite in inst.citations:
            cited.setdefault(cite.clause_id, []).append(inst.instance_id)
    base_by_clause = {d.clause_id: d for d in base.dispositions}
    dispositions = []
    for clause in base.clauses:
        ids = cited.get(clause.clause_id, [])
        if ids:
            dispositions.append(ClauseDisposition(
                clause_id=clause.clause_id, disposition="mapped",
                instance_ids=tuple(ids)))
            continue
        # A clause this reading produced nothing for is UNMAPPED for this
        # reading, even if the pipeline mapped it. Inheriting "mapped" from the
        # pipeline would let readings B and C borrow the pipeline's disposition
        # score for clauses they never placed - a bias in favour of exactly the
        # architecture being tested. "irrelevant" is inherited, because a
        # signature block is not commercial no matter who reads it.
        old = base_by_clause.get(clause.clause_id)
        if old is not None and old.disposition == "irrelevant":
            dispositions.append(old)
        else:
            dispositions.append(ClauseDisposition(
                clause_id=clause.clause_id, disposition="unmapped",
                reason="הקריאה הזאת לא הפיקה מסעיף זה מונח"))
    return DocumentExtraction(
        document_id=base.document_id,
        clauses=base.clauses, instances=list(instances),
        dispositions=dispositions,
        source_language=base.source_language, ingest_route=base.ingest_route,
        stats=dict(base.stats),
    )


def main() -> None:
    wanted = [a for a in sys.argv[1:] if not a.startswith("--")]
    truths = corpus.load_all()
    doc_ids = wanted or sorted(truths)
    client, auth_mode = extract_provider.build_client()
    records: dict[str, Any] = {}

    for doc_id in doc_ids:
        directory = corpus.corpus_root() / doc_id
        pdf = directory / "document.pdf"
        if not pdf.exists():
            print(f"skip {doc_id}: no rendered PDF")
            continue
        declared = json.loads((directory / "render.json").read_text(encoding="utf-8"))
        route = str(declared.get("ingest_route", "digital"))
        images = sorted((directory / "pages").glob("page-*.png")) if route == "scanned-vision" else []
        truth = truths[doc_id]

        stats = extract_provider.RunStats()
        # The whole-document reader and the arbiter answer for an entire
        # agreement in one response, so they get room for it. 4000 - fine for a
        # clause-level answer - truncated the first run before its instances
        # array and returned a reading of nothing.
        caller = extract_provider.StageCaller(
            client=client, stats=stats, auth_mode=auth_mode,
            max_tokens_by_stage={"wholedoc": 16000, "arbitrate": 16000},
        )
        print(f"\n=== {doc_id} ({route})", flush=True)

        # ---- A: the shipped clause-by-clause pipeline -----------------------
        started = time.monotonic()
        pipeline = extract_run.run_pdf(
            pdf, caller, document_id=doc_id, agreement_id=doc_id,
            force_route=route if route != "digital" else None,
            page_images=images or None,
        )
        a_seconds = time.monotonic() - started
        a_score = score_document(truth, pipeline)
        print(f"  A pipeline    recall {_pct(a_score['recall'])}  "
              f"precision {_pct(a_score['precision'])}  params {_pct(a_score['param_accuracy'])}  "
              f"{a_seconds:.0f}s", flush=True)

        # ---- B: one reading of the whole document ---------------------------
        started = time.monotonic()
        page_blocks = extract_wholedoc.vision_blocks(images) if images else None
        whole = extract_wholedoc.read_whole_document(
            pipeline.clauses, caller.call, page_images=page_blocks)
        b_instances = extract_wholedoc.instances_from_records(
            whole["instances"], pipeline.clauses, document_id=doc_id)
        b_seconds = time.monotonic() - started
        b_score = score_document(truth, _as_extraction(pipeline, b_instances))
        print(f"  B whole-doc   recall {_pct(b_score['recall'])}  "
              f"precision {_pct(b_score['precision'])}  params {_pct(b_score['param_accuracy'])}  "
              f"{b_seconds:.0f}s  (dropped {len(whole['dropped'])})", flush=True)

        # ---- C: the arbiter over both ---------------------------------------
        started = time.monotonic()
        alignment = extract_wholedoc.align(pipeline.instances, whole["instances"])
        ruled = arbitrate.arbitrate(pipeline.clauses, alignment, caller.call, document_id=doc_id)
        c_seconds = time.monotonic() - started
        c_score = score_document(truth, _as_extraction(pipeline, ruled["instances"]))
        print(f"  C arbitrated  recall {_pct(c_score['recall'])}  "
              f"precision {_pct(c_score['precision'])}  params {_pct(c_score['param_accuracy'])}  "
              f"{c_seconds:.0f}s  (agreed {ruled['agreed_count']}, "
              f"ruled {len(ruled['rulings'])})", flush=True)

        records[doc_id] = {
            "route": route,
            "alignment": {k: len(alignment[k]) for k in
                          ("agreed", "params_differ", "pipeline_only", "whole_only")},
            "whole_dropped": whole["dropped"],
            "rulings": ruled["rulings"],
            "scores": {"pipeline": a_score, "whole": b_score, "arbitrated": c_score},
            "seconds": {"pipeline": a_seconds, "whole": b_seconds, "arbitrated": c_seconds},
            "provider": stats.to_payload(),
        }
        _write(records)

    if not records:
        raise SystemExit("nothing measured")
    _write(records)
    print(f"\nwrote {OUT_MD}")


def _totals(records: dict[str, Any], reading: str) -> dict[str, Any]:
    matched = sum(r["scores"][reading]["matched"] for r in records.values())
    truth_n = sum(r["scores"][reading]["instances_truth"] for r in records.values())
    found = sum(r["scores"][reading]["instances_found"] for r in records.values())
    hits = sum(r["scores"][reading]["param_hits"] for r in records.values())
    leaves = sum(r["scores"][reading]["param_total"] for r in records.values())
    seconds = sum(r["seconds"][reading] for r in records.values())
    return {
        "recall": matched / truth_n if truth_n else None,
        "precision": matched / found if found else None,
        "params": hits / leaves if leaves else None,
        "matched": matched, "truth": truth_n, "found": found,
        "hits": hits, "leaves": leaves, "seconds": seconds,
    }


def _write(records: dict[str, Any]) -> None:
    OUT_JSON.write_text(json.dumps(
        {"run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "documents": records}, ensure_ascii=False, indent=1), encoding="utf-8")

    rows = [
        "# Three readings of the same corpus, measured",
        "",
        "Produced by `scripts/trade_arbitration_bench.py` against the corpus in",
        "`tests/trade_corpus/agreements` and the same independently authored ground",
        "truth as `extraction-accuracy.md`. Every number is a real provider run.",
        "",
        "| reading | what it is | recall | precision | parameters | seconds |",
        "|---|---|---|---|---|---|",
    ]
    labels = {
        "pipeline": "A — clause by clause (shipped)",
        "whole": "B — one call, whole document",
        "arbitrated": "C — A and B, disagreements ruled by a third call",
    }
    for reading in ("pipeline", "whole", "arbitrated"):
        t = _totals(records, reading)
        rows.append(
            f"| {reading} | {labels[reading]} | {_pct(t['recall'])} ({t['matched']}/{t['truth']}) "
            f"| {_pct(t['precision'])} ({t['matched']}/{t['found']}) "
            f"| {_pct(t['params'])} ({t['hits']}/{t['leaves']}) | {t['seconds']:.0f} |"
        )

    rows += ["", "## Where the two readers disagreed", "",
             "| document | agreed | different parameters | only A saw | only B saw | rulings |",
             "|---|---|---|---|---|---|"]
    for doc_id, rec in sorted(records.items()):
        a = rec["alignment"]
        rows.append(f"| `{doc_id}` | {a['agreed']} | {a['params_differ']} | "
                    f"{a['pipeline_only']} | {a['whole_only']} | {len(rec['rulings'])} |")

    rows += ["", "## How the arbiter ruled", "",
             "| verdict | meaning | count |", "|---|---|---|"]
    verdicts: dict[str, int] = {}
    for rec in records.values():
        for ruling in rec["rulings"]:
            verdicts[ruling["verdict"]] = verdicts.get(ruling["verdict"], 0) + 1
    meaning = {"a": "the clause reader governs", "b": "the whole-document reader governs",
               "revised": "neither; the arbiter wrote the term itself",
               "neither": "no commercial term here at all"}
    for verdict, count in sorted(verdicts.items(), key=lambda kv: -kv[1]):
        rows.append(f"| `{verdict}` | {meaning.get(verdict, verdict)} | {count} |")

    rows += ["", "## Per document", "",
             "| document | A recall | B recall | C recall | A params | B params | C params |",
             "|---|---|---|---|---|---|---|"]
    for doc_id, rec in sorted(records.items()):
        s = rec["scores"]
        rows.append(
            f"| `{doc_id}` | {_pct(s['pipeline']['recall'])} | {_pct(s['whole']['recall'])} "
            f"| {_pct(s['arbitrated']['recall'])} | {_pct(s['pipeline']['param_accuracy'])} "
            f"| {_pct(s['whole']['param_accuracy'])} | {_pct(s['arbitrated']['param_accuracy'])} |"
        )
    OUT_MD.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
