"""Give the advertiser stores the identity the agency stores already have.

The rules store is keyed on ADV_01..ADV_45 and the real advertisers are 41
Hebrew names, and the two key spaces do not intersect at all, so every rule
lookup has always missed in silence. This migration does the two things that
make an advertiser addressable, and neither of them changes a price:

  1. ``data/advertiser_rules.csv`` gains ``name``, ``display_name`` and
     ``aliases``, appended after the existing columns and blank on every
     existing row. Every cell that was there stays exactly where it was, so the
     45 rows keep their premiums and stay bound to nothing. Which of the two
     honest dispositions those rows get is the owner's decision and is not made
     here.
  2. ``data/advertiser_names.csv`` is written from the advertisers actually
     observed in the real data: the daily files' advertiser column and the
     agency link store. Only names that appear in one of those sources are
     written, so nothing is invented, and an advertiser already in the file
     keeps its display name and its aliases.

Usage:
    python scripts/migrate_advertiser_identity.py            apply
    python scripts/migrate_advertiser_identity.py --dry-run  report only
    python scripts/migrate_advertiser_identity.py --check    verify, exit 1 if unmet

The check verifies the bar this migration exists for: every advertiser in the
newest daily file resolves to a named record. Console output is ASCII-only,
because Windows consoles default to cp1252 and cannot print Hebrew.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kairos.optimize.advertiser_rules_identity import (  # noqa: E402
    IDENTITY_COLUMNS,
    NAMES_COLUMNS,
    join_aliases,
    load_advertiser_names,
    normalize_name,
    read_csv_rows,
    split_aliases,
)

DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
RULES_PATH = DATA_DIR / "advertiser_rules.csv"
NAMES_PATH = DATA_DIR / "advertiser_names.csv"
LINKS_PATH = DATA_DIR / "agency_advertisers.csv"
DAILY_DIR = DATA_DIR / "daily_input"

# The daily file's advertiser and date columns, as the Hebrew source writes them
# (kairos.data.loaders.DAILY_COLUMN_MAP renames these downstream).
DAILY_ADVERTISER_COLUMN = "מפרסם"
DAILY_DATE_COLUMN = "תאריך"

OBSERVED = "observed"


def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
    """Write a store the way the running product writes them.

    Temp file plus ``os.replace`` so a reader never sees a torn csv, utf-8 with
    a BOM and newline-only line endings so the file matches the ones already on
    disk and the diff stays about content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in header})
    os.replace(tmp, path)


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    target = BACKUP_DIR / f"{path.stem}_{stamp}{path.suffix}"
    shutil.copy2(path, target)
    return target


def _rules_header(path: Path) -> list[str]:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8-sig", newline="") as handle:
        return next(csv.reader(handle), [])


def add_identity_columns(*, dry_run: bool = False) -> dict[str, object]:
    """Append the identity columns to the rules store, blank on every row."""
    header = _rules_header(RULES_PATH)
    if not header:
        return {"changed": False, "reason": "no rules store on disk", "added": []}
    missing = [column for column in IDENTITY_COLUMNS if column not in header]
    if not missing:
        return {"changed": False, "reason": "already migrated", "added": []}
    rows = read_csv_rows(RULES_PATH)
    if not dry_run:
        _backup(RULES_PATH)
        _write_csv(RULES_PATH, header + missing, rows)
    return {"changed": True, "reason": "columns appended", "added": missing, "rows": len(rows)}


def observed_advertisers() -> dict[str, dict[str, str]]:
    """Every advertiser observed in the real data, keyed by normalized name.

    Two sources, both real: the agency link store (which records the date each
    advertiser was observed) and the advertiser column of every daily file on
    disk. A name present in neither is not written, so the roster can only ever
    hold advertisers that something actually saw.
    """
    found: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(LINKS_PATH):
        name = str(row.get("advertiser", "") or "").strip()
        if not name:
            continue
        found.setdefault(normalize_name(name), {
            "name": name,
            "source": str(row.get("source", "") or OBSERVED).strip() or OBSERVED,
            "first_seen": str(row.get("observed_date", "") or "").strip(),
        })
    for path in sorted(DAILY_DIR.glob("Wally_*.csv")) if DAILY_DIR.exists() else []:
        for row in read_csv_rows(path):
            name = str(row.get(DAILY_ADVERTISER_COLUMN, "") or "").strip()
            if not name:
                continue
            record = found.setdefault(normalize_name(name), {
                "name": name, "source": OBSERVED, "first_seen": "",
            })
            if not record["first_seen"]:
                record["first_seen"] = str(row.get(DAILY_DATE_COLUMN, "") or "").strip()[:10]
    return found


def build_names(*, dry_run: bool = False) -> dict[str, object]:
    """Write the observed name space, keeping every hand-written cell."""
    existing = load_advertiser_names(NAMES_PATH)
    observed = observed_advertisers()
    rows: list[dict[str, str]] = []
    for key in sorted(observed, key=lambda item: observed[item]["name"]):
        seen = observed[key]
        held = existing.get(key)
        rows.append({
            "name": seen["name"],
            "display_name": held.display_name if held else "",
            "aliases": join_aliases(held.aliases) if held else "",
            "source": held.source if held and held.source else seen["source"],
            "first_seen": held.first_seen if held and held.first_seen else seen["first_seen"],
            "notes": held.notes if held else "",
        })
    # A name the operator added by hand that no source has observed yet is kept,
    # marked with the source it already carries, never dropped.
    for key, held in existing.items():
        if key not in observed:
            rows.append({
                "name": held.name, "display_name": held.display_name,
                "aliases": join_aliases(held.aliases), "source": held.source or "operator",
                "first_seen": held.first_seen, "notes": held.notes,
            })
    if not dry_run:
        _backup(NAMES_PATH)
        _write_csv(NAMES_PATH, list(NAMES_COLUMNS), rows)
    return {"rows": len(rows), "observed": len(observed), "kept": len(existing)}


def check() -> dict[str, object]:
    """Resolve every advertiser in the newest daily file against the roster."""
    from kairos.optimize.advertiser_rules_identity import (
        _names_token_index,
        load_name_index,
        resolve_advertiser,
    )

    names = load_advertiser_names(NAMES_PATH)
    tokens = _names_token_index(names)
    rules_index = load_name_index(RULES_PATH)
    daily = sorted(DAILY_DIR.glob("Wally_*.csv")) if DAILY_DIR.exists() else []
    if not daily:
        return {"daily_file": None, "total": 0, "resolved": 0, "unresolved": []}
    path = daily[-1]
    seen = sorted({
        str(row.get(DAILY_ADVERTISER_COLUMN, "") or "").strip()
        for row in read_csv_rows(path)
        if str(row.get(DAILY_ADVERTISER_COLUMN, "") or "").strip()
    })
    unresolved = [
        name for name in seen
        if resolve_advertiser(name, names=names, rules_index=rules_index, names_tokens=tokens) is None
    ]
    bound = sum(
        1 for name in seen
        if (found := resolve_advertiser(name, names=names, rules_index=rules_index, names_tokens=tokens))
        is not None and found.has_rules_row
    )
    return {
        "daily_file": path.name,
        "total": len(seen),
        "resolved": len(seen) - len(unresolved),
        "bound_to_a_rules_row": bound,
        "unresolved": unresolved,
    }


def _report(result: dict[str, object]) -> None:
    for key, value in result.items():
        print(f"  {key}: {value}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate the advertiser identity stores")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--check", action="store_true", help="verify resolution only, write nothing")
    args = parser.parse_args(argv)

    if args.check:
        result = check()
        print("advertiser identity check")
        _report(result)
        return 0 if result["total"] and not result["unresolved"] else 1

    print("rules store identity columns")
    _report(add_identity_columns(dry_run=args.dry_run))
    print("observed name space")
    _report(build_names(dry_run=args.dry_run))
    print("resolution")
    result = check()
    _report(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
