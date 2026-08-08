"""No shared operator store may carry a row an agent left behind.

Three of these have now been found, all written by an agent walking the product
in a browser and all committed without anybody noticing:

  data/kairos_settings.json   revenue_weight and min_retention_floor, then
                              locale and direction. Moved 15,844,833 ILS and put
                              the operator's own front page into a declared
                              licence breach.
  data/manual_overrides.csv   one gold mark on 2024-11-03. Moved 131,878.70 ILS
                              and survived the settings restore because nothing
                              guarded this file.
  data/agencies.csv           four agencies named "סוכנות ביקורת", with contacts
                              at critic.example, plus their four links in
                              agency_advertisers.csv and four campaigns in
                              campaigns.csv marked is_demo false, so seeded rows
                              presented as real bookings. One pollution, five
                              failing tests across three test files.

The first two are guarded by tests/test_plan_artifact_fingerprint.py, each
learned the hard way and each too narrow at the time it was written. This is the
next size up and it guards the class rather than the file.

THE RULE. A row in a shared store is one of two things and never a third:

  a LABELLED SYNTHETIC SEED, which says so in data_source and carries example
  contact details on purpose, so nobody mistakes it for a booking; or
  a REAL OPERATOR ACTION, which may carry anything except pretend contacts.

A row claiming to be a real operator action while carrying a reserved example
address is neither. It is a test fixture wearing a booking's clothes, and that is
exactly what an agent leaves behind. RFC 2606 reserves .example for this, which
is why it is a reliable marker rather than a guess.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Reserved for documentation and tests, never routable. RFC 2606 and RFC 6761.
PRETEND_DOMAINS = (".example", "@example.com", "@example.org", "@test", ".invalid")

# A row that says it came from a person, rather than from a labelled seed.
CLAIMS_REAL = ("manual", "operator", "observed")


def _stores() -> list[Path]:
    """Every committed CSV store, excluding the backup folder that holds removals."""
    return sorted(
        path
        for path in DATA.glob("*.csv")
        if path.is_file() and "_backups" not in path.parts
    )


def _rows(path: Path) -> list[dict[str, str]]:
    try:
        with open(path, newline="", encoding="utf-8-sig") as handle:
            return list(csv.DictReader(handle))
    except Exception:
        return []


def test_no_store_claims_a_real_row_while_carrying_a_pretend_contact():
    offenders: list[str] = []
    for path in _stores():
        for index, row in enumerate(_rows(path), start=2):
            source = str(row.get("data_source") or row.get("source") or "").strip().lower()
            if source not in CLAIMS_REAL:
                continue
            contacts = " ".join(
                str(value or "") for key, value in row.items() if key and "email" in key.lower()
            ).lower()
            if any(domain in contacts for domain in PRETEND_DOMAINS):
                name = row.get("name") or row.get("display_name") or row.get("agency_id") or "?"
                offenders.append(f"{path.name}:{index} {name} says {source!r} and mails {contacts}")
    assert offenders == [], (
        "a shared store carries a row that claims to be a real operator action and has a "
        "contact address reserved for tests:\n  "
        + "\n  ".join(offenders)
        + "\nThis is what an agent leaves behind after walking the product in a browser. "
        "Move it to data/_backups/ rather than deleting it, and find out which run wrote it."
    )


@pytest.mark.parametrize("path", _stores(), ids=lambda p: p.name)
def test_every_row_says_where_it_came_from(path: Path):
    """A row with no provenance cannot be told from an agent's leftovers at all.

    Only stores that already publish the column are held to it, because adding
    the column elsewhere is a schema decision and not this guard's business. What
    this catches is a store that HAS the column and rows that leave it empty,
    which is the state in which nothing can be attributed to anybody.
    """
    rows = _rows(path)
    if not rows:
        pytest.skip("empty store")
    column = next((c for c in ("data_source", "source") if c in rows[0]), None)
    if column is None:
        pytest.skip(f"{path.name} does not publish a provenance column")

    # Grouped by record type, because a store may hold more than one.
    # campaigns.csv holds 52 campaign rows and 55 flight rows in one table, and a
    # flight legitimately carries no data_source: provenance lives on the
    # campaign it belongs to. A flat check called all 55 a defect, which was the
    # check being wrong rather than the data.
    #
    # So the rule is self-describing: within a record type, if ANY row fills the
    # column then every row of that type must. That still catches an agent's
    # leftover sitting among rows that all fill it, which is the case this guard
    # exists for, and it stops inventing a schema the store never had.
    kind_column = next(iter(rows[0]))
    groups: dict[str, list[tuple[int, dict[str, str]]]] = {}
    for index, row in enumerate(rows, start=2):
        groups.setdefault(str(row.get(kind_column) or "").strip(), []).append((index, row))

    problems: list[str] = []
    for kind, members in groups.items():
        filled = [i for i, row in members if str(row.get(column) or "").strip()]
        blank = [i for i, row in members if not str(row.get(column) or "").strip()]
        if filled and blank:
            problems.append(
                f"{kind or 'untyped'}: {len(blank)} of {len(members)} rows leave {column} "
                f"empty while the rest fill it (first at line {blank[0]})"
            )
    assert problems == [], (
        f"{path.name} is inconsistent about provenance:\n  "
        + "\n  ".join(problems)
        + "\nA row with no provenance among rows that have it cannot be told from an "
        "agent's leftover, which is the whole reason this column exists."
    )
