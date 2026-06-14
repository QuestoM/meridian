"""Validate the program classifier against the real Programmes.xlsx titles.

Runs the classifier over every distinct programme title in the reference data
and writes a UTF-8 coverage report to output/classifier_coverage.md. Console
output is ASCII-only (Windows consoles default to cp1252 and cannot print
Hebrew). Read the markdown report for the Hebrew detail.

Usage:
    python scripts/validate_classifier.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from kairos.data.classifier import ProgramClassifier  # noqa: E402

PROGRAMMES = ROOT / "data" / "reference" / "Programmes.xlsx"
REPORT = ROOT / "output" / "classifier_coverage.md"


def main() -> int:
    if not PROGRAMMES.exists():
        print(f"ERROR: {PROGRAMMES} not found")
        return 1

    frame = pd.read_excel(PROGRAMMES)
    titles = frame["Title"].astype(str).str.strip()
    distinct = sorted(titles.unique())

    clf = ProgramClassifier.from_yaml()

    # Coverage weighted by airing frequency (what the model actually sees).
    rows_report = clf.coverage_report(titles.tolist())
    # Coverage over distinct titles (taxonomy breadth).
    distinct_report = clf.coverage_report(distinct)

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Program classifier coverage report")
    lines.append("")
    lines.append(f"Source: `{PROGRAMMES.relative_to(ROOT)}`")
    lines.append(f"Total programme rows: {rows_report['total']}")
    lines.append(f"Distinct titles: {distinct_report['total']}")
    lines.append("")
    lines.append("## Coverage")
    lines.append("")
    lines.append(f"- By airing rows: {rows_report['coverage_pct']}% categorised "
                 f"({rows_report['covered']}/{rows_report['total']})")
    lines.append(f"- By distinct title: {distinct_report['coverage_pct']}% categorised "
                 f"({distinct_report['covered']}/{distinct_report['total']})")
    lines.append(f"- Reruns flagged (rows): {rows_report['rerun_count']}")
    lines.append("")
    lines.append("## Category distribution (by airing rows)")
    lines.append("")
    lines.append("| Category | Rows | Share |")
    lines.append("|---|---:|---:|")
    for category, count in rows_report["by_category"].items():
        share = round(100 * count / rows_report["total"], 1)
        lines.append(f"| {category} | {count} | {share}% |")
    lines.append("")
    lines.append("## Match rule distribution (by airing rows)")
    lines.append("")
    for rule, count in sorted(rows_report["by_rule"].items(), key=lambda kv: kv[1], reverse=True):
        lines.append(f"- {rule}: {count}")
    lines.append("")
    lines.append(f"## Uncovered distinct titles ({len(distinct_report['uncovered'])})")
    lines.append("")
    lines.append("These returned `Other`. Add a keyword or specific-program override to cover them.")
    lines.append("")
    for title in distinct_report["uncovered"]:
        lines.append(f"- {title}")
    lines.append("")
    lines.append(f"## Low-confidence distinct titles ({len(distinct_report['low_confidence'])})")
    lines.append("")
    lines.append("Categorised but with weak evidence (confidence < 0.6). Worth a review.")
    lines.append("")
    for title, conf in distinct_report["low_confidence"]:
        lines.append(f"- {title} ({conf})")
    lines.append("")

    REPORT.write_text("\n".join(lines), encoding="utf-8")

    # ASCII-only console summary.
    print("=== classifier coverage ===")
    print(f"rows: {rows_report['covered']}/{rows_report['total']} "
          f"({rows_report['coverage_pct']}%) categorised")
    print(f"distinct: {distinct_report['covered']}/{distinct_report['total']} "
          f"({distinct_report['coverage_pct']}%) categorised")
    print(f"uncovered distinct titles: {len(distinct_report['uncovered'])}")
    print(f"report written to: {REPORT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
