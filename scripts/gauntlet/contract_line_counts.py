"""Keep the line counts a contract publishes true, or fail loudly.

Every contract under docs/ux-gauntlet/contracts/ publishes a table of the files
its piece owns with a line count beside each one. Those counts are the first
thing a builder reads to learn what exists and what is near the 450 cap, and
they rot the moment anybody edits a file.

They rot silently, which is the whole problem. Measured on 2026-08-09:

    P12.md   39 rows   27 stale
    P7.md    20 rows   13 stale
    P8.md     6 rows    1 stale
    W0-2.md  14 rows   14 stale

P12's own round-10 critic had already caught this once and marked it closed. It
re-rotted, because nothing re-counted. A number nobody re-derives is a number
nobody checks, which is the same sentence the native-control budget earned
earlier the same day.

A count is a MEASUREMENT, not a decision, so re-taking it is not editing the
contract's meaning and ``--fix`` may do it in bulk. What ``--fix`` must never do
is invent a row or delete one: a path that no longer exists is a real finding
about the contract and is reported rather than quietly dropped.

    python3 scripts/gauntlet/contract_line_counts.py          check, exit 1 on drift
    python3 scripts/gauntlet/contract_line_counts.py --fix    re-count and rewrite
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACTS = ROOT / "docs" / "ux-gauntlet" / "contracts"

# | `path/to/file.py` | 312 | whatever the row says next |
ROW = re.compile(
    r"^(?P<head>\|\s*`?)(?P<path>[\w./\-]+\.(?:py|jsx|js|css|json|csv))(?P<mid>`?\s*\|\s*)(?P<count>\d+)(?P<tail>\s*\|)"
)


# Where a contract's first column may be rooted. W0-2's table names a bare file
# in column one and its path under the dashboard's src in column three, so a
# resolver that only tries the repository root calls half that table missing and
# the real finding drowns in the noise.
SEARCH_ROOTS = (Path("."), Path("tv-break-dashboard/src"), Path("kairos_api"), Path("kairos"))


def resolve(rel: str) -> Path | None:
    for root in SEARCH_ROOTS:
        candidate = ROOT / root / rel
        if candidate.exists():
            return candidate
    # A bare basename, which W0-2's table uses. One match is the file; several
    # mean the contract is ambiguous about which one it owns, and that is a
    # finding rather than something to guess at.
    if "/" not in rel:
        found = [
            path
            for path in (ROOT / "tv-break-dashboard" / "src").rglob(rel)
            if path.is_file()
        ]
        if len(found) == 1:
            return found[0]
    return None


def line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8", errors="replace").splitlines())


def audit(fix: bool = False) -> tuple[list[str], list[str]]:
    """Returns (drifted, missing). With fix, drifted rows are rewritten in place."""
    drifted: list[str] = []
    missing: list[str] = []
    for contract in sorted(CONTRACTS.glob("*.md")):
        text = contract.read_text(encoding="utf-8")
        out: list[str] = []
        changed = False
        for line in text.splitlines(keepends=True):
            match = ROW.match(line.strip())
            if not match:
                out.append(line)
                continue
            target = resolve(match.group("path"))
            if target is None:
                missing.append(f"{contract.name}: `{match.group('path')}` does not exist")
                out.append(line)
                continue
            actual = line_count(target)
            claimed = int(match.group("count"))
            if actual == claimed:
                out.append(line)
                continue
            drifted.append(f"{contract.name}: `{match.group('path')}` is {actual}, says {claimed}")
            if fix:
                stripped = line.strip()
                rebuilt = ROW.sub(
                    lambda m: f"{m.group('head')}{m.group('path')}{m.group('mid')}{actual}{m.group('tail')}",
                    stripped,
                    count=1,
                )
                out.append(line.replace(stripped, rebuilt))
                changed = True
            else:
                out.append(line)
        if fix and changed:
            contract.write_text("".join(out), encoding="utf-8")
    return drifted, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fix", action="store_true", help="re-count and rewrite the tables")
    args = parser.parse_args(argv)

    drifted, missing = audit(fix=args.fix)

    if missing:
        print(f"{len(missing)} row(s) name a file that is gone. This is NOT auto-fixable:\n")
        for problem in missing:
            print(f"  {problem}")
        print("\nEither the path moved and the contract must say where, or the row is dead.")

    if args.fix:
        print(f"re-counted {len(drifted)} row(s)")
        return 1 if missing else 0

    if drifted:
        print(f"{len(drifted)} contract row(s) publish a line count that is no longer true:\n")
        for problem in drifted:
            print(f"  {problem}")
        print(
            "\nA builder reads these to learn what is near the 450 cap before it opens a file. "
            "Run with --fix; a count is a measurement, not a decision."
        )
    if drifted or missing:
        return 1
    print("Every contract's line counts match the files on disk.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
