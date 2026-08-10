"""The gate a wave cannot start without.

This exists because of a question worth more than the document that prompted it:
what makes a written lesson actually get used?

Nothing does, on its own. `campaign-plan.md` says a builder should be handed a
dossier instead of discovering its piece cold, and that sentence is worth exactly
zero until something refuses to launch a wave that has no dossier. A document
whose failure mode is being unread is the same shape as a check whose failure
mode is silence: it reports nothing, so it looks like it is working.

So this is the enforcement, and it does two jobs.

  1. It REFUSES to let a wave start when a piece has no dossier, or when the
     dossier is missing a section, or when it still carries placeholder text.
  2. It refuses when the dossier has ROTTED. Every file inventory row carries a
     line count, this re-counts the file, and a count that has drifted fails.
     That is the part that matters. A stale dossier is worse than no dossier,
     because it is believed. Making the dossier self-invalidating is what stops
     it becoming decoration six weeks from now.

It also prints the exact dossier paths, so the wave script names files rather
than pasting content, which is what keeps a resume cache key stable.

Usage:

    python3 scripts/gauntlet/wave_preflight.py --pieces P10,P11,P12,P13

Exit 0 means the wave may launch. Any other exit means it may not, and the
output says precisely which line of which file is wrong.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOSSIERS = ROOT / "docs" / "ux-gauntlet" / "dossiers"
SETTINGS = ROOT / "data" / "kairos_settings.json"

# Every section a builder was measured to need before it could write a line. Each
# one is here because a round was spent rediscovering it.
REQUIRED_SECTIONS = (
    "## Job stories and their done conditions",
    "## Baseline numbers",
    "## File inventory",
    "## The API surface this piece owns",
    "## Reference product, and what to compare",
    "## Trade facts that bind this piece",
    "## What is already built",
    "## Exact commands",
)

# Text that means the dossier was started and not finished. A half-written
# dossier passes a human skim and fails a builder.
PLACEHOLDERS = ("TODO", "TBD", "FIXME", "<fill", "XXX", "...tbd")

# A file inventory row: | `path/to/file.py` | 312 lines | note |
INVENTORY_ROW = re.compile(r"^\|\s*`([^`]+)`\s*\|\s*(\d+)\s+lines?\s*\|")

# The operational cap was raised by the owner on 2026-08-10. Existing files in
# the 451-499 band are no longer wave work by themselves; 500 remains the point
# where a dossier must force an explicit split decision.
SIZE_CAP = 500

PINNED = {"locale": "he", "direction": "rtl"}


class Failure(Exception):
    """A reason the wave may not start, phrased for the person reading stdout."""


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise Failure(f"{path.relative_to(ROOT)} does not exist") from exc


def check_settings() -> list[str]:
    """The shared writable store, checked before a wave rather than after it.

    Every wave so far has had at least one agent walk the product in a browser,
    and a browser walk writes this whole file back. Catching it here costs
    nothing; catching it at commit time has twice cost a restore.
    """
    problems: list[str] = []
    try:
        live = json.loads(SETTINGS.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"data/kairos_settings.json is unreadable: {exc}"]
    for key, want in PINNED.items():
        if live.get(key) != want:
            problems.append(
                f"data/kairos_settings.json has {key}={live.get(key)!r}, expected {want!r}. "
                "An agent measured in the other locale and did not restore it. Fix it "
                "before launching, not after."
            )
    if not live.get("operator_channel"):
        problems.append(
            "data/kairos_settings.json has no operator_channel, so every scoped surface "
            "will refuse to draw and every critic will report the refusal as a defect."
        )
    return problems


def check_dossier(piece: str) -> list[str]:
    """Everything wrong with one piece's dossier, in the order a reader fixes it."""
    problems: list[str] = []
    path = DOSSIERS / f"{piece}.md"
    if not path.exists():
        return [
            f"{piece} has no dossier at docs/ux-gauntlet/dossiers/{piece}.md. "
            "A builder without one spends its first round rediscovering what the last "
            "round already knew. Write it, then launch."
        ]
    text = _read(path)

    for section in REQUIRED_SECTIONS:
        if section not in text:
            problems.append(f"{piece}: the dossier has no section {section!r}")

    for token in PLACEHOLDERS:
        if token in text:
            problems.append(
                f"{piece}: the dossier still carries {token!r}, so it was started and "
                "not finished"
            )

    problems.extend(check_inventory(piece, text))
    return problems


def check_inventory(piece: str, text: str) -> list[str]:
    """The self-invalidating part. A count that has drifted fails the gate.

    This is the whole reason the gate is worth writing. Any document can be
    written once and believed forever; this one cannot, because the repository
    moves underneath it and the movement is measured here.
    """
    problems: list[str] = []
    rows = 0
    for line in text.splitlines():
        match = INVENTORY_ROW.match(line.strip())
        if not match:
            continue
        rows += 1
        rel, claimed = match.group(1), int(match.group(2))
        target = ROOT / rel
        if not target.exists():
            problems.append(
                f"{piece}: the inventory lists `{rel}`, which does not exist. Either the "
                "path is wrong or the file was moved since the dossier was written."
            )
            continue
        actual = len(target.read_text(encoding="utf-8", errors="replace").splitlines())
        if actual != claimed:
            problems.append(
                f"{piece}: `{rel}` is {actual} lines, the dossier says {claimed}. "
                "The dossier has rotted. Update the count, and while you are there check "
                "whether what the dossier says about the file is still true."
            )
        elif actual > SIZE_CAP:
            problems.append(
                f"{piece}: `{rel}` is {actual} lines, over the {SIZE_CAP} cap, and the "
                "dossier must say so out loud or a builder will open it and add to it."
            )
    if rows == 0:
        problems.append(
            f"{piece}: the file inventory has no rows in the form "
            "| `path` | N lines | note |, so nothing about it can be verified"
        )
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pieces",
        required=True,
        help="comma separated piece ids, for example P10,P11,P12,P13",
    )
    parser.add_argument(
        "--skip-settings",
        action="store_true",
        help="do not check the shared settings store (for unit tests only)",
    )
    args = parser.parse_args(argv)

    pieces = [p.strip() for p in args.pieces.split(",") if p.strip()]
    if not pieces:
        print("no pieces named, so there is nothing to launch")
        return 2

    problems: list[str] = []
    if not args.skip_settings:
        problems.extend(check_settings())
    for piece in pieces:
        problems.extend(check_dossier(piece))

    if problems:
        print("THE WAVE MAY NOT START. " f"{len(problems)} reason(s):\n")
        for index, problem in enumerate(problems, start=1):
            print(f"  {index}. {problem}")
        print(
            "\nEvery one of these is something a builder would otherwise discover with "
            "its own context, one round late, and report as a finding."
        )
        return 1

    print(f"Preflight clear for {', '.join(pieces)}. Name these paths in the prompts:\n")
    for piece in pieces:
        print(f"  {piece}: docs/ux-gauntlet/dossiers/{piece}.md")
    print(
        "\nName the path, do not paste the content: interpolated content changes the "
        "resume cache key and re-runs every cached agent in the wave."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
