"""CSV persistence and count-to-override conversion for the constraint store.

Extracted from constraints_store to keep that module under the size limit. This
module has no dependency on constraints_store (the import is one-directional), so
the path constants and the CSV column order live here and constraints_store
re-exports them.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
DEFAULT_CONSTRAINTS_PATH = DATA_DIR / "kairos_constraints.csv"

# CSV columns, in the order they are written.
# ``where_json`` is the predicate column (JSON string). It is appended so existing
# CSVs without it stay valid; _load_frame adds a blank column when absent.
COLUMNS = (
    "constraint_id",
    "scope_type",
    "scope_value",
    "channel",
    "effect",
    "offset_seconds",
    "offset_min_seconds",
    "offset_max_seconds",
    "count",
    "duration_seconds",
    "duration_min_seconds",
    "duration_max_seconds",
    "order_index",
    "notes",
    "where_json",
)


def count_pins_to_overrides(count_pins: dict[str, int], forbids: set[str]):
    """Build an OverrideSet that pins / forbids segment break counts.

    The optimizer consumes count constraints through an
    :class:`~kairos.optimize.overrides.OverrideSet` (segment scope). A forbid maps
    to a FORBID override; any other pinned count maps to a PIN override at that
    count. Segments with an explicit placement pin are NOT included here, because
    a placement pin already forces the count.
    """
    from kairos.optimize.overrides import FORBID as O_FORBID
    from kairos.optimize.overrides import PIN as O_PIN
    from kairos.optimize.overrides import SEGMENT, Override, OverrideSet

    overrides: list[Override] = []
    for sid in sorted(forbids):
        overrides.append(Override(
            override_id=f"forbid|{sid}", scope=SEGMENT, target_id=sid, kind=O_FORBID,
        ))
    for sid, count in sorted(count_pins.items()):
        if sid in forbids:
            continue
        overrides.append(Override(
            override_id=f"pin|{sid}", scope=SEGMENT, target_id=sid, kind=O_PIN, value=str(count),
        ))
    return OverrideSet(overrides=overrides)


def _load_frame(path: Path = DEFAULT_CONSTRAINTS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    frame = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _backup(path: Path = DEFAULT_CONSTRAINTS_PATH) -> None:
    if not path.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(path, BACKUP_DIR / f"kairos_constraints_{stamp}.csv")


def _write_frame(frame: pd.DataFrame, path: Path = DEFAULT_CONSTRAINTS_PATH) -> None:
    _backup(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame[list(COLUMNS)].to_csv(path, index=False, encoding="utf-8-sig")
