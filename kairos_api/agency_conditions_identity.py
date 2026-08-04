"""Naming a client the operator created, in the observed name space.

Split out of :mod:`kairos_api.agency_conditions` to keep that module under the
project line limit. It does one thing: when the product creates an advertiser,
it writes that advertiser into ``data/advertiser_names.csv`` so the client is a
named record from the moment it exists.

**The measured defect this closes.** ``data/agency_advertisers.csv`` said who
buys through which agency and ``data/advertiser_names.csv`` said who each
advertiser is, and only the first of the two had a creation path. Measured on
the working tree before this module existed: the link store held 45 advertisers
and the name space held 41, and the four in the difference were every client the
onboarding flow had ever created. Two things followed. The name space and the
link store are asserted equal by the identity suite, so the tree shipped red;
and a created client resolved to no record, so its own screen read "source
unknown" for ever and no rule could ever be bound to it by name.

**Honesty rules.** Nothing here invents a name, a display name or an alias: the
row carries the string the operator typed and nothing else. ``source`` is
``manual``, never ``observed``, because a client that has not aired was not
observed, and ``first_seen`` is the date the record was created, which is the
first date this product can honestly say it knew the name. Registration is
idempotent under the same fold resolution uses, so a name already in the space
is left exactly as it is and reported as already named.

**Where the file lives.** The name space is a sibling of the link store in the
same data directory, so the path is derived from the link store's own path
rather than from a second constant. A caller that redirects its stores, which is
every test in this suite, redirects this one with them and never writes a name
into the operator's real data.
"""

from __future__ import annotations

import csv
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from kairos.optimize.advertiser_rules_identity import (
    DEFAULT_NAMES_PATH,
    NAMES_COLUMNS,
    normalize_name,
    read_csv_rows,
)

NAMES_FILENAME = "advertiser_names.csv"
BACKUP_DIRNAME = "_backups"

# The source a created client carries. It is not "observed", because nothing
# observed it, and the difference is what tells a reader which figures can exist
# for this client at all.
MANUAL = "manual"

REGISTERED_REASON = (
    "Created in this product, so the name space holds it as a named record before it has aired."
)
REGISTERED_REASON_HE = (
    "נוצר במוצר הזה, ולכן מרחב השמות מחזיק אותו ככרטיס בשם עוד לפני ששודר."
)
ALREADY_NAMED_REASON = "This name is already in the name space, so nothing was written."
ALREADY_NAMED_REASON_HE = "השם הזה כבר במרחב השמות, ולכן לא נכתב דבר."

_STORE_LOCK = threading.Lock()


def names_path_for(store_path: Any) -> Path:
    """The name space that sits beside a given store, real or redirected."""
    if store_path is None:
        return Path(DEFAULT_NAMES_PATH)
    return Path(store_path).parent / NAMES_FILENAME


def _backup(path: Path) -> None:
    if not path.exists():
        return
    target = path.parent / BACKUP_DIRNAME
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(path, target / f"{path.stem}_{stamp}{path.suffix}")


def _write_names(path: Path, rows: list[dict[str, str]]) -> None:
    """Write the name space the way every other store here is written.

    Temp file plus ``os.replace`` so a reader never sees a torn csv, utf-8 with
    a BOM and newline-only line endings so the file matches the one already on
    disk and the diff stays about content.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(NAMES_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in NAMES_COLUMNS})
    os.replace(tmp, path)


def _held(rows: list[dict[str, str]], wanted: str) -> Optional[dict[str, str]]:
    """The row that already holds this name, folded as resolution folds it."""
    key = normalize_name(wanted)
    for row in rows:
        if normalize_name(row.get("name", "")) == key:
            return row
    return None


def register_advertiser_name(
    name: str,
    *,
    store_path: Any = None,
    names_path: Any = None,
    first_seen: str = "",
    notes: str = "",
) -> dict[str, Any]:
    """Put a created client in the name space, or report that it is already there.

    Returns the outcome rather than raising on a name that is already held,
    because creating a client twice is a thing an account manager really does
    and the second time is not an error.
    """
    wanted = str(name or "").strip()
    path = Path(names_path) if names_path is not None else names_path_for(store_path)
    if not wanted:
        return {"advertiser": "", "outcome": "not_named", "source": "", "first_seen": ""}
    with _STORE_LOCK:
        rows = read_csv_rows(path)
        already = _held(rows, wanted)
        if already is not None:
            return {
                "advertiser": str(already.get("name", "") or wanted),
                "outcome": "already_named",
                "source": str(already.get("source", "") or ""),
                "first_seen": str(already.get("first_seen", "") or ""),
                "reason_en": ALREADY_NAMED_REASON,
                "reason_he": ALREADY_NAMED_REASON_HE,
            }
        seen = str(first_seen or "").strip() or datetime.now(timezone.utc).date().isoformat()
        rows.append({
            "name": wanted,
            "display_name": "",
            "aliases": "",
            "source": MANUAL,
            "first_seen": seen,
            "notes": str(notes or ""),
        })
        _backup(path)
        _write_names(path, rows)
    return {
        "advertiser": wanted,
        "outcome": "registered",
        "source": MANUAL,
        "first_seen": seen,
        "reason_en": REGISTERED_REASON,
        "reason_he": REGISTERED_REASON_HE,
    }
