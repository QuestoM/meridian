"""A store writer must not be the authority on which columns its file has.

Five stores wrote their CSV as ``frame[COLUMNS].to_csv(...)``. That projection
drops any column present in the file but absent from the module's own hardcoded
list, and a dashboard PUT is enough to trigger it.

MEASURED 2026-08-09 across all five: one live erasure and four latent.
``data_source`` is on all 45 rows of ``advertiser_rules.csv`` and appears in no
COLUMNS list, so one edit of one advertiser would have erased the provenance of
every row in the file. That field is what the shared-store provenance guard reads
to tell a labelled synthetic seed from a real operator action, so the erasure
would have disarmed a guard rather than only losing data.

The reason it is a class: a column is added to a file by a migration, a seed
script or a sweep, and none of those edit the writer's constant. The writer is
written once and the file keeps growing.

TWO TESTS, and they check different things on purpose.

The first is a UNIT test of the helper, so the rule itself is proven rather than
assumed. The second reads the SHIPPED FILES and fails on a store nobody has
thought about yet, including one added after this was written. A guard that only
knew the five stores of 2026-08-09 would go quiet exactly when a sixth appeared.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from kairos_api.store_columns import projected, write_order

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "kairos_api"
DATA = ROOT / "data"


def test_the_helper_keeps_what_it_was_not_told_about_and_keeps_the_order():
    frame = pd.DataFrame([{"b": 2, "a": 1, "surprise": 3, "later": 4}])
    order = write_order(frame, ["a", "b", "absent_from_file"])
    # Declared order first, and a declared column the file does not carry is
    # skipped rather than raising: a store whose file predates a newly declared
    # column has to stay writable.
    assert order[:2] == ["a", "b"]
    # Everything else survives, in the order the frame carried it.
    assert order[2:] == ["surprise", "later"]
    assert list(projected(frame, ["a", "b"]).columns) == ["a", "b", "surprise", "later"]


def _stores() -> list[tuple[str, list[str], Path]]:
    """Every module that writes a CSV through a COLUMNS list, found by reading.

    Discovered rather than listed, so a sixth store is covered on the day it is
    written. The path is resolved from the module's own ``DATA_DIR / "name.csv"``
    expression, which is how all five spell it.
    """
    found: list[tuple[str, list[str], Path]] = []
    for module in sorted(API.glob("*.py")):
        source = module.read_text(encoding="utf-8")
        if ".to_csv(" not in source or "COLUMNS" not in source:
            continue
        block = re.search(r"^COLUMNS\s*=\s*\[(.*?)^\]", source, re.S | re.M)
        if not block:
            continue
        names = re.findall(r'"([^"]+)"', block.group(1))
        for raw in re.findall(r'DATA_DIR\s*/\s*"([^"]+\.csv)"', source):
            found.append((module.name, names, DATA / raw))
    return found


def test_no_store_would_erase_a_column_its_own_file_carries():
    """The class guard, read from the shipped files rather than from a list."""
    stores = _stores()
    assert stores, "found no COLUMNS-projecting store, so this guard is checking nothing"
    erasures: dict[str, list[str]] = {}
    for module, columns, path in stores:
        if not path.exists():
            continue
        on_disk = list(pd.read_csv(path, nrows=0).columns)
        lost = [name for name in on_disk if name not in columns]
        if lost:
            # Not a failure by itself. It IS a failure if the module still
            # projects through the bare list, because then the column is gone on
            # the next write.
            source = (API / module).read_text(encoding="utf-8")
            if "frame[COLUMNS].to_csv(" in source:
                erasures[f"{module} -> {path.name}"] = lost
    assert not erasures, (
        "a store writes through its own COLUMNS list while its file carries columns "
        f"that list does not name, so the next save erases them: {erasures}. "
        "Write through kairos_api.store_columns.projected instead."
    )


def test_the_advertiser_provenance_column_is_the_measured_case_and_survives_a_write():
    """The one live erasure, driven end to end rather than argued.

    Reads the shipped file, writes it back through the same projection the API
    uses, and asserts the provenance column is still there with the same values.
    """
    path = DATA / "advertiser_rules.csv"
    if not path.exists():
        pytest.skip("advertiser rules are not on this tree")
    frame = pd.read_csv(path)
    if "data_source" not in frame.columns:
        pytest.skip("the file no longer carries data_source, so there is nothing to erase")
    from kairos_api.advertisers import COLUMNS

    assert "data_source" not in COLUMNS, (
        "data_source joined COLUMNS, which fixes this one file and leaves the "
        "mechanism in place for the next column and the next store"
    )
    written = projected(frame, COLUMNS)
    assert "data_source" in written.columns
    assert written["data_source"].tolist() == frame["data_source"].tolist()
    assert len(written.columns) == len(frame.columns)
