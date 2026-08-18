"""Measure the extraction pipeline against the corpus's own ground truth.

Runs the REAL pipeline (ingest → segment → classify → parameterise → assemble)
over each corpus agreement's rendered PDF and scores it against truth.json,
then writes docs/trade/extraction-accuracy.md and a machine record beside it.

Scored, per document and aggregated:

- **clause coverage** — every clause carries a disposition (a structural
  guarantee, so this must be 100% or the pipeline is broken), and how many
  dispositions match truth's CLASS (mapped / irrelevant / unmapped).
- **term recall / precision** — expected term instances found, and found
  instances that exist in truth. Matching is by term_id + citation overlap:
  an instance counts as found when some extracted instance of the same term
  cites at least one clause the truth instance cites.
- **parameter accuracy** — per matched instance, the share of truth's
  parameter leaves reproduced (numbers within 0.5%, enums exact, free text by
  normalised containment). Reported per term family too, because a product
  that nails prices and fumbles cures is not the same product as the reverse.
- **citation fidelity** — every quote must appear verbatim in the clause it
  cites. This is a hard property of the pipeline (it verifies quotes itself),
  so a non-zero failure here is a defect, not a score.
- **conflict detection** — planted contradictions found and resolved as truth
  says they should be.

Usage:
  python scripts/trade_extraction_accuracy.py                 # every document
  python scripts/trade_extraction_accuracy.py <doc-id> ...    # a subset
  python scripts/trade_extraction_accuracy.py --dry-run       # no provider

The run costs real tokens; the stats block records what it spent.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kairos.trade import corpus, extract_provider, extract_run  # noqa: E402
from kairos.trade import taxonomy  # noqa: E402
from kairos.trade.documents import DocumentExtraction  # noqa: E402

OUT_DOC = REPO / "docs" / "trade" / "extraction-accuracy.md"
OUT_JSON = REPO / "docs" / "trade" / "extraction-accuracy.json"


def routing() -> dict[str, str]:
    """Which model answered which tier on this run."""
    return {tier: extract_provider.model_for(tier) for tier in ("small", "mid", "reason")}


def _routing_slug() -> str:
    """A filename for one routing: the family word of each tier, in order.

    Short on purpose — the archive's own meta carries the full model ids, and a
    file called extraction-haiku-sonnet-opus.json says at a glance which run it
    is without anyone having to open it.
    """
    words = []
    for tier in ("small", "mid", "reason"):
        name = extract_provider.model_for(tier)
        family = next((word for word in ("haiku", "sonnet", "opus") if word in name), None)
        words.append(family or name.replace("/", "-"))
    return "-".join(words)


# ----------------------------------------------------------------- scoring

def _leaves(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a params dict to comparable leaves."""
    out: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            out.update(_leaves(item, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            out.update(_leaves(item, f"{prefix}[{i}]"))
    else:
        out[prefix] = value
    return out


def _leaf_matches(expected: Any, found: Any) -> bool:
    if isinstance(expected, bool) or isinstance(found, bool):
        return bool(expected) == bool(found)
    if isinstance(expected, (int, float)) and isinstance(found, (int, float)):
        if expected == 0:
            return abs(found) < 1e-9
        return abs(float(found) - float(expected)) / abs(float(expected)) <= 0.005
    exp_text = str(expected or "").strip()
    got_text = str(found or "").strip()
    if not exp_text:
        return not got_text
    if exp_text == got_text:
        return True
    # Free text: normalised containment either way (the model may phrase a
    # summary more or less fully than the truth author did).
    norm_exp = " ".join(exp_text.split())
    norm_got = " ".join(got_text.split())
    return norm_exp in norm_got or norm_got in norm_exp


def _score_params(expected: Mapping[str, Any], found: Mapping[str, Any]) -> tuple[int, int]:
    exp_leaves = _leaves(dict(expected))
    got_leaves = _leaves(dict(found))
    if not exp_leaves:
        return (0, 0)
    hits = sum(1 for key, value in exp_leaves.items()
               if key in got_leaves and _leaf_matches(value, got_leaves[key]))
    return (hits, len(exp_leaves))


def score_document(truth: DocumentExtraction, got: DocumentExtraction) -> dict[str, Any]:
    truth_disp = {d.clause_id: d for d in truth.dispositions}
    got_disp = {d.clause_id: d for d in got.dispositions}

    # The completeness guarantee is about the PIPELINE's own reading: every
    # clause it segmented carries a disposition. Its denominator is therefore
    # the pipeline's clause count, not the truth's — a segmenter that finds 13
    # where the author wrote 12 (a split table, a glued wrap) must read as
    # 13/13 accounted, not as 13/12 "104%", which is how the report once
    # printed a coverage above one hundred percent.
    accounted = len(got_disp)
    clauses_pipeline = len(got.clauses)
    class_hits = sum(
        1 for cid, d in truth_disp.items()
        if cid in got_disp and got_disp[cid].disposition == d.disposition
    )

    truth_cites = {
        i.instance_id: {c.clause_id for c in i.citations} for i in truth.instances
    }
    got_by_term: dict[str, list[Any]] = {}
    for inst in got.instances:
        got_by_term.setdefault(inst.term_id, []).append(inst)

    matched: list[tuple[Any, Any]] = []
    used: set[str] = set()
    missed: list[dict[str, str]] = []
    for truth_inst in truth.instances:
        candidates = [
            g for g in got_by_term.get(truth_inst.term_id, [])
            if g.instance_id not in used
            and ({c.clause_id for c in g.citations} & truth_cites[truth_inst.instance_id])
        ]
        if not candidates:
            missed.append({"instance_id": truth_inst.instance_id,
                           "term_id": truth_inst.term_id})
            continue
        best = max(
            candidates,
            key=lambda g: _score_params(truth_inst.params, g.params)[0],
        )
        used.add(best.instance_id)
        matched.append((truth_inst, best))

    spurious = [
        {"instance_id": g.instance_id, "term_id": g.term_id}
        for g in got.instances if g.instance_id not in used
    ]

    param_hits = param_total = 0
    by_family: dict[str, dict[str, int]] = {}
    for truth_inst, found_inst in matched:
        hits, total = _score_params(truth_inst.params, found_inst.params)
        param_hits += hits
        param_total += total
        family = taxonomy.get(truth_inst.term_id).family
        bucket = by_family.setdefault(family, {"hits": 0, "total": 0, "instances": 0})
        bucket["hits"] += hits
        bucket["total"] += total
        bucket["instances"] += 1

    clause_text = {c.clause_id: c.text for c in got.clauses}
    bad_citations = [
        {"instance_id": i.instance_id, "clause_id": c.clause_id, "quote": c.quote}
        for i in got.instances for c in i.citations
        if c.quote not in clause_text.get(c.clause_id, "")
    ]

    expected_conflicts = truth.stats.get("expected_conflicts", [])
    found_conflicts = got.stats.get("conflicts", [])
    conflict_hits = 0
    for expected in expected_conflicts:
        want = set(expected.get("instances", []))
        for found in found_conflicts:
            pair = set(found.get("instances", []))
            truth_pair = {
                m[1].instance_id for m in matched if m[0].instance_id in want
            }
            if pair and pair == truth_pair:
                conflict_hits += 1
                break

    return {
        "clauses_truth": len(truth.clauses),
        "clauses_pipeline": clauses_pipeline,
        "clauses_accounted": accounted,
        "disposition_class_hits": class_hits,
        "instances_truth": len(truth.instances),
        "instances_found": len(got.instances),
        "matched": len(matched),
        "missed": missed,
        "spurious": spurious,
        "recall": round(len(matched) / len(truth.instances), 4) if truth.instances else None,
        "precision": round(len(matched) / len(got.instances), 4) if got.instances else None,
        "param_hits": param_hits,
        "param_total": param_total,
        "param_accuracy": round(param_hits / param_total, 4) if param_total else None,
        "by_family": {
            f: {**b, "accuracy": round(b["hits"] / b["total"], 4) if b["total"] else None}
            for f, b in sorted(by_family.items())
        },
        "citation_failures": bad_citations,
        "conflicts_expected": len(expected_conflicts),
        "conflicts_found": conflict_hits,
        "unmapped_found": sum(1 for d in got.dispositions if d.disposition == "unmapped"),
        "unmapped_truth": sum(1 for d in truth.dispositions if d.disposition == "unmapped"),
    }


# -------------------------------------------------------------------- report

def _pct(value: Any) -> str:
    return "—" if value is None else f"{float(value) * 100:.1f}%"


def write_report(records: dict[str, dict[str, Any]], meta: dict[str, Any]) -> None:
    total = {
        "clauses": sum(r["score"]["clauses_pipeline"] for r in records.values()),
        "accounted": sum(r["score"]["clauses_accounted"] for r in records.values()),
        "class_hits": sum(r["score"]["disposition_class_hits"] for r in records.values()),
        "instances_truth": sum(r["score"]["instances_truth"] for r in records.values()),
        "matched": sum(r["score"]["matched"] for r in records.values()),
        "found": sum(r["score"]["instances_found"] for r in records.values()),
        "param_hits": sum(r["score"]["param_hits"] for r in records.values()),
        "param_total": sum(r["score"]["param_total"] for r in records.values()),
        "citation_failures": sum(len(r["score"]["citation_failures"]) for r in records.values()),
        "conflicts_expected": sum(r["score"]["conflicts_expected"] for r in records.values()),
        "conflicts_found": sum(r["score"]["conflicts_found"] for r in records.values()),
    }
    lines = [
        "# Extraction accuracy, measured",
        "",
        f"Run {meta['run_at']} against the corpus in `tests/trade_corpus/agreements`.",
        "Models: "
        + ", ".join(f"{tier}={extract_provider.model_for(tier)}" for tier in ("small", "mid", "reason"))
        + ".",
        "",
        "Every number here is produced by `scripts/trade_extraction_accuracy.py`",
        "running the real pipeline against ground truth authored independently of",
        "it. Nothing on this page is asserted by hand.",
        "",
        "## Aggregate",
        "",
        "| measure | value |",
        "|---|---|",
        f"| documents | {len(records)} |",
        f"| clauses accounted for | {total['accounted']}/{total['clauses']} "
        f"({_pct(total['accounted'] / total['clauses'] if total['clauses'] else None)}) |",
        f"| disposition class correct | {_pct(total['class_hits'] / total['clauses'] if total['clauses'] else None)} |",
        f"| term recall | {total['matched']}/{total['instances_truth']} "
        f"({_pct(total['matched'] / total['instances_truth'] if total['instances_truth'] else None)}) |",
        f"| term precision | {total['matched']}/{total['found']} "
        f"({_pct(total['matched'] / total['found'] if total['found'] else None)}) |",
        f"| parameter accuracy | {total['param_hits']}/{total['param_total']} "
        f"({_pct(total['param_hits'] / total['param_total'] if total['param_total'] else None)}) |",
        f"| citation fidelity failures | {total['citation_failures']} |",
        f"| planted conflicts detected | {total['conflicts_found']}/{total['conflicts_expected']} |",
        "",
        "## Per document",
        "",
        "| document | clauses | class | recall | precision | params | conflicts |",
        "|---|---|---|---|---|---|---|",
    ]
    for doc_id, record in sorted(records.items()):
        s = record["score"]
        lines.append(
            f"| `{doc_id}` | {s['clauses_accounted']}/{s.get('clauses_pipeline', s['clauses_truth'])} | "
            f"{_pct(s['disposition_class_hits'] / s['clauses_truth'] if s['clauses_truth'] else None)} | "
            f"{_pct(s['recall'])} | {_pct(s['precision'])} | {_pct(s['param_accuracy'])} | "
            f"{s['conflicts_found']}/{s['conflicts_expected']} |"
        )
    lines += ["", "## Parameter accuracy by term family", "",
              "| family | instances | leaves | accuracy |", "|---|---|---|---|"]
    families: dict[str, dict[str, int]] = {}
    for record in records.values():
        for family, bucket in record["score"]["by_family"].items():
            acc = families.setdefault(family, {"hits": 0, "total": 0, "instances": 0})
            acc["hits"] += bucket["hits"]
            acc["total"] += bucket["total"]
            acc["instances"] += bucket["instances"]
    for family, bucket in sorted(families.items()):
        lines.append(
            f"| {family} — {taxonomy.FAMILIES.get(family, family)} | {bucket['instances']} | "
            f"{bucket['total']} | "
            f"{_pct(bucket['hits'] / bucket['total'] if bucket['total'] else None)} |"
        )
    misses: list[str] = []
    for doc_id, record in sorted(records.items()):
        for miss in record["score"]["missed"]:
            misses.append(f"- `{doc_id}` — {miss['term_id']} ({miss['instance_id']})")
    lines += ["", "## What the pipeline missed", ""]
    lines += misses or ["Nothing: every ground-truth term was found."]
    lines += ["", "## Cost and latency", "",
              "| document | seconds | calls | input tokens | output tokens |",
              "|---|---|---|---|---|"]
    for doc_id, record in sorted(records.items()):
        provider = record.get("provider", {})
        calls = sum(b["calls"] for b in provider.values())
        tin = sum(b["input_tokens"] for b in provider.values())
        tout = sum(b["output_tokens"] for b in provider.values())
        lines.append(f"| `{doc_id}` | {record.get('elapsed', 0):.1f} | {calls} | "
                     f"{tin:,} | {tout:,} |")
    # Model routing is a VARIABLE of this measurement, not a constant of it, so
    # every run is archived under the routing that produced it and the table
    # below is generated from those archives. The alternative — one report that
    # the newest run overwrites — makes "did the bigger model help?" a question
    # answered from memory, which is how a number nobody re-measured becomes a
    # fact. Same discipline the arbiter's prompt gets in
    # scripts/trade_arbitration_bench.py.
    archive = OUT_JSON.parent / f"extraction-{_routing_slug()}.json"
    body = json.dumps({"meta": meta, "documents": records}, ensure_ascii=False, indent=1)
    archive.write_text(body, encoding="utf-8")

    # Every archive EXCEPT the headline file, which the glob also matches and
    # which is a copy of whichever run wrote it last: leaving it in put a
    # duplicate row in the table under a routing of "? / ? / ?", because the
    # headline predates the field. A comparison whose rows are not distinct runs
    # is not a comparison.
    archives = sorted(path for path in OUT_JSON.parent.glob("extraction-*.json")
                      if path.name != OUT_JSON.name)
    if len(archives) > 1:
        lines += ["", "## Model routing is a variable, so it was measured", "",
                  "| routing | documents | recall | precision | parameters | seconds |",
                  "|---|---|---|---|---|---|"]
        for path in archives:
            other = json.loads(path.read_text(encoding="utf-8"))
            docs = other.get("documents", {})
            if not docs:
                continue
            matched = sum(r["score"]["matched"] for r in docs.values())
            truth = sum(r["score"]["instances_truth"] for r in docs.values())
            found = sum(r["score"]["instances_found"] for r in docs.values())
            hits = sum(r["score"]["param_hits"] for r in docs.values())
            leaves = sum(r["score"]["param_total"] for r in docs.values())
            seconds = sum(r.get("elapsed", 0.0) for r in docs.values())
            routing = other.get("meta", {}).get("routing") or {}
            shown = " / ".join(str(routing.get(tier, "?")) for tier in ("small", "mid", "reason"))
            lines.append(
                f"| {shown} | {len(docs)} | {_pct(matched / truth if truth else None)} "
                f"| {_pct(matched / found if found else None)} "
                f"| {_pct(hits / leaves if leaves else None)} | {seconds:.0f} |")
        lines += ["",
                  "Read as small / mid / reason. The rows are whole-corpus runs of the",
                  "same pipeline against the same ground truth, so the only thing that",
                  "moved between them is which model answered which stage.", ""]

    OUT_DOC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    OUT_JSON.write_text(
        json.dumps({"meta": meta, "documents": records}, ensure_ascii=False, indent=1),
        encoding="utf-8",
    )


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry_run = "--dry-run" in sys.argv
    truths = corpus.load_all()
    wanted = args or sorted(truths)
    if dry_run:
        print("corpus loaded:", ", ".join(sorted(truths)))
        return
    client, auth_mode = extract_provider.build_client()
    records: dict[str, dict[str, Any]] = {}
    for doc_id in wanted:
        truth = truths[doc_id]
        directory = corpus.corpus_root() / doc_id
        pdf = directory / "document.pdf"
        if not pdf.exists():
            print(f"skip {doc_id}: no rendered PDF")
            continue
        stats = extract_provider.RunStats()
        caller = extract_provider.StageCaller(client=client, stats=stats,
                                             auth_mode=auth_mode)
        started = time.monotonic()
        # The corpus DECLARES each document's route. Detection cannot be trusted
        # to rediscover it: the scanned agreement's PDF carries a text layer, so
        # detection called it digital, read a layer nobody should trust, and
        # scored zero. A declared scan is fed its page images.
        declared = json.loads((directory / "render.json").read_text(encoding="utf-8"))
        route = str(declared.get("ingest_route", "digital"))
        images = sorted((directory / "pages").glob("page-*.png")) \
            if route == "scanned-vision" else []
        print(f"running {doc_id} ({route}) ...", flush=True)
        got = extract_run.run_pdf(
            pdf, caller, document_id=doc_id, agreement_id=doc_id,
            force_route=route if route != "digital" else None,
            page_images=images or None,
        )
        elapsed = time.monotonic() - started
        records[doc_id] = {
            "score": score_document(truth, got),
            "provider": stats.to_payload(),
            "elapsed": elapsed,
        }
        (directory / "measured-extraction.json").write_text(
            json.dumps(got.to_payload(), ensure_ascii=False, indent=1), encoding="utf-8")
        score = records[doc_id]["score"]
        print(f"  recall {_pct(score['recall'])}  precision {_pct(score['precision'])}  "
              f"params {_pct(score['param_accuracy'])}  {elapsed:.0f}s")
    if not records:
        raise SystemExit("nothing measured")
    write_report(records, {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "documents": sorted(records),
        "routing": routing(),
    })
    print(f"\nwrote {OUT_DOC.relative_to(REPO)}")


if __name__ == "__main__":
    main()
