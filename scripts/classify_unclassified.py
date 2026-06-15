"""Find and (when an LLM is configured) resolve unclassified programme titles.

Scans the reference Programmes and any daily-input files for titles the
rule-based classifier leaves as "Other" (genuinely unclassified, not merely
priced as the Other class), routes them to the grounded AI classifier when one
is available, and writes the verdicts to a cache the engine can fold back in via
:class:`~kairos.data.ai_classifier.CachedClassifier`.

With no key or client library present it still reports the unclassified titles
and records them as ``unavailable`` -- it never invents a genre.

Run from the repo root:  python scripts/classify_unclassified.py
"""

from __future__ import annotations

from pathlib import Path

from kairos.data import ProgramClassifier
from kairos.data.ai_classifier import (
    GeminiGroundedClassifier,
    resolve_unclassified,
    unclassified_titles,
)
from kairos.data.loaders import DAILY_DIR, load_daily_input, load_programmes

ROOT = Path(__file__).resolve().parents[1]
CACHE_PATH = ROOT / "models" / "ai_program_classifications.json"


def _collect_titles() -> list[str]:
    titles: list[str] = []
    try:
        programmes = load_programmes()
        if "Title" in programmes.columns:
            titles.extend(programmes["Title"].dropna().astype(str).tolist())
    except Exception as error:  # reference file optional in some checkouts
        print(f"programmes not loaded: {error}")
    if DAILY_DIR.exists():
        for csv in sorted(DAILY_DIR.glob("*.csv")):
            daily = load_daily_input(csv)
            if "program" in daily.columns:
                titles.extend(daily["program"].dropna().astype(str).tolist())
    return titles


def main() -> None:
    classifier = ProgramClassifier.from_yaml()
    titles = _collect_titles()
    pending = unclassified_titles(titles, classifier)
    print(f"scanned {len(titles)} titles | {len(pending)} distinct unclassified")
    for title in pending:
        print(f"  - {title}")

    ai = GeminiGroundedClassifier.from_env()
    if ai is None:
        print("\nNo AI classifier configured (set GEMINI_API_KEY and install google-genai).")
        print("Recording the unclassified titles as unavailable; no genre is invented.")
    resolved = resolve_unclassified(titles, classifier, ai, cache_path=CACHE_PATH)
    placed = sum(1 for v in resolved.values() if v.source == "ai" and v.category != "Other")
    print(f"\nwrote {CACHE_PATH} | {placed}/{len(resolved)} placed by AI")


if __name__ == "__main__":
    main()
