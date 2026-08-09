"""The column order a store writes, with every column it did not know about kept.

Five stores write their CSV as ``frame[COLUMNS].to_csv(...)``. That projection is
a silent eraser: any column present in the file but absent from the module's own
hardcoded list is dropped the first time anything saves, and a save is one PUT
from the dashboard.

MEASURED on 2026-08-09, all five stores against their shipped files:

    advertiser_rules.csv    12 on disk, 11 in COLUMNS   -> data_source ERASED
    agencies.csv            24 on disk, 24 in COLUMNS   -> none
    campaigns.csv           30 on disk, 30 in COLUMNS   -> none
    make_goods.csv          32 on disk, 32 in COLUMNS   -> none
    advertiser_conditions   8 on disk, 11 in COLUMNS    -> none

So it is live at one site today and latent at four. ``data_source`` is on all 45
advertiser rows and is the field the shared-store provenance guard reads to tell
a labelled synthetic seed from a real operator action. One edit of one
advertiser's notes would have erased the provenance of every row in the file, and
nothing would have said so.

The reason this is a class and not one bug is the direction of the dependency: a
column gets added to a file by a migration, a seed script, or a sweep, and none of
those edit the writer's constant. The writer is written once and the file keeps
growing. So the writer must not be the authority on which columns exist.

What this does instead: the module's own COLUMNS still fix the ORDER and still
guarantee its own fields are present, and anything else the frame carries is
appended after them rather than dropped. Order stays stable for a reader
diffing the file; unknown columns survive.

The narrow alternative, adding ``data_source`` to one list, was rejected. It
fixes the one site that has already been measured and leaves the mechanism that
produced it in place at four more.
"""

from __future__ import annotations

from typing import Any, Iterable


def write_order(frame: Any, columns: Iterable[str]) -> list[str]:
    """The declared columns in their declared order, then everything else kept.

    A declared column the frame does not carry is skipped rather than raising:
    the callers already fill their own fields before writing, and a store whose
    file predates a newly declared column should still be writable.
    """
    declared = list(columns)
    present = list(frame.columns)
    ordered = [name for name in declared if name in present]
    extra = [name for name in present if name not in declared]
    return ordered + extra


def projected(frame: Any, columns: Iterable[str]) -> Any:
    """``frame`` ready to write: declared order first, unknown columns preserved."""
    return frame[write_order(frame, columns)]
