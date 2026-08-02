"""Which broadcast dates the daily spot files on disk actually cover.

Split out of :mod:`kairos.export.spots` rather than added to it, because that
module is at 449 lines against the 450-line cap and because this asks a different
question: not what a spot is worth, but whether there is a ledger behind a day at
all. Nothing here prices anything, so no engine figure can move through it.

It exists so a surface can tell the truth about delivered money instead of
guessing. Measured on the shipped data this returns exactly ``{'2025-04-27'}``
while the saved weekly plan covers 2024-11-01 to 2024-11-30, so the honest answer
for every planned break today is that delivered is unavailable, with both dates
named and the path to supply a delivery feed stated beside it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from kairos.data.loaders import load_daily_input

DAILY_INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "daily_input"


def daily_input_days(directory: str | Path | None = None) -> set[str]:
    """The ISO broadcast dates covered by every readable daily file in a folder.

    An absent or unreadable folder yields an empty set, which reads as no
    coverage rather than as an error, because the caller reports coverage as a
    tri-state and a missing ledger is a state of the data, not a fault of the
    page. One unreadable file is skipped rather than failing the whole answer.
    """
    root = Path(directory) if directory is not None else DAILY_INPUT_DIR
    if not root.exists():
        return set()
    days: set[str] = set()
    for path in sorted(root.glob("*.csv")):
        try:
            frame = load_daily_input(path)
        except Exception:  # noqa: BLE001 - one unreadable file is not the answer
            continue
        if "date" not in frame.columns:
            continue
        stamps = pd.to_datetime(frame["date"], errors="coerce").dropna()
        days.update(stamps.dt.strftime("%Y-%m-%d").tolist())
    return days
