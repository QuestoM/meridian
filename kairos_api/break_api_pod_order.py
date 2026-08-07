"""The order an operator wants a pod's spots to air in, on ``data/break_pod_order.csv``.

The traffic file declares an order and a traffic operator changes it. The file on
disk is the source and stays untouched, so this register is the operator's own
half of that decision: which pod, which spots in which order, who decided, when,
and what the pod looked like at the time.

That last field is the one that earns its place. A pod is read live from a file
another team replaces every day, so a saved order can outlive the spots it
ordered. The record therefore carries a fingerprint of the pod as it was read
when the order was saved, and a read compares it. When the two disagree the saved
order is reported as stale and **the file's own order is what the surface shows**,
because applying half an order to a pod that has changed underneath it would put
an advertiser in a position nobody chose.

Written with the same discipline as every other operator store in this package:
one lock over load, mutate and write, a backup before the write, and a temp file
plus ``os.replace`` so a reader never sees a torn CSV. Nothing here prices
anything and no figure moves through it.
"""

from __future__ import annotations

import logging
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
ORDER_PATH = DATA_DIR / "break_pod_order.csv"

COLUMNS = ("pod_id", "spot_keys", "fingerprint", "actor", "saved_at", "note")
KEY_SEPARATOR = "|"

STALE = "The traffic file changed after this order was saved, so the pod is shown in the order the file declares."
STALE_HE = "קובץ הטראפיק השתנה לאחר שמירת הסדר הזה, ולכן התוכן מוצג בסדר שהקובץ מצהיר עליו."
APPLIED = "This pod is shown in the order an operator saved, not the order the traffic file declares."
APPLIED_HE = "התוכן הזה מוצג בסדר ששמר מפעיל, ולא בסדר שקובץ הטראפיק מצהיר עליו."
FILE_ORDER = "This pod is shown in the order the traffic file declares."
FILE_ORDER_HE = "התוכן הזה מוצג בסדר שקובץ הטראפיק מצהיר עליו."

_STORE_LOCK = threading.Lock()


def _load_frame() -> pd.DataFrame:
    if not ORDER_PATH.exists():
        return pd.DataFrame(columns=list(COLUMNS))
    try:
        frame = pd.read_csv(ORDER_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    except Exception:  # noqa: BLE001 - an unreadable register is a state, not a crash
        logger.exception("pod order register read failed")
        return pd.DataFrame(columns=list(COLUMNS))
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _write_frame(frame: pd.DataFrame) -> None:
    if ORDER_PATH.exists():
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        shutil.copy2(ORDER_PATH, BACKUP_DIR / f"break_pod_order_{stamp}.csv")
    ORDER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = ORDER_PATH.with_name(ORDER_PATH.name + ".tmp")
    frame[list(COLUMNS)].to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, ORDER_PATH)


def _record(row: "pd.Series[Any]") -> dict[str, Any]:
    return {column: str(row.get(column, "")) for column in COLUMNS}


def stored(pod_id: str) -> Optional[dict[str, Any]]:
    """The saved order for one pod, exactly as stored, or None."""
    wanted = str(pod_id or "").strip()
    frame = _load_frame()
    if frame.empty or not wanted:
        return None
    mask = frame["pod_id"].astype(str) == wanted
    if not mask.any():
        return None
    return _record(frame[mask].iloc[0])


def save(pod_id: str, spot_keys: list[str], fingerprint: str, actor: str = "", note: str = "") -> dict[str, Any]:
    """Record the order one pod should air in, replacing any earlier order of it.

    One pod carries at most one saved order, because a second save of the same pod
    is the same decision restated rather than a second decision.
    """
    row = {
        "pod_id": str(pod_id).strip(),
        "spot_keys": KEY_SEPARATOR.join(str(key).strip() for key in spot_keys),
        "fingerprint": str(fingerprint or ""),
        "actor": str(actor or ""),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "note": str(note or ""),
    }
    replaced: Optional[dict[str, Any]] = None
    with _STORE_LOCK:
        frame = _load_frame()
        if not frame.empty:
            mask = frame["pod_id"].astype(str) == row["pod_id"]
            if mask.any():
                replaced = _record(frame[mask].iloc[0])
            frame = frame[~mask].reset_index(drop=True)
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        _write_frame(frame)
    return {**row, "replaced": replaced}


def forget(pod_id: str) -> Optional[dict[str, Any]]:
    """Drop one pod's saved order and return the record that was dropped."""
    wanted = str(pod_id or "").strip()
    with _STORE_LOCK:
        frame = _load_frame()
        if frame.empty:
            return None
        mask = frame["pod_id"].astype(str) == wanted
        if not mask.any():
            return None
        dropped = _record(frame[mask].iloc[0])
        frame = frame[~mask].reset_index(drop=True)
        _write_frame(frame)
    return dropped


def applied(pod_id: str, spots: list[dict[str, Any]], fingerprint: str) -> dict[str, Any]:
    """The pod's spots in the order they should be shown, and why that order.

    Three outcomes and each says which it is. No saved order leaves the file's own
    order alone. A saved order whose fingerprint still matches is applied and the
    sequence numbers are restated so the surface never prints a position from one
    order beside a sequence from another. A saved order whose fingerprint has
    moved is reported stale and not applied.
    """
    record = stored(pod_id)
    if record is None:
        return {"spots": spots, "order": {"state": "file", "reason": FILE_ORDER, "reason_he": FILE_ORDER_HE}}
    keys = [key for key in str(record.get("spot_keys", "")).split(KEY_SEPARATOR) if key]
    present = {spot["spot_key"]: spot for spot in spots}
    moved = record.get("fingerprint", "") != fingerprint
    if moved or sorted(keys) != sorted(present):
        return {
            "spots": spots,
            "order": {
                "state": "stale",
                "reason": STALE,
                "reason_he": STALE_HE,
                "saved_at": record.get("saved_at", ""),
                "actor": record.get("actor", ""),
                "saved_fingerprint": record.get("fingerprint", ""),
                "pod_fingerprint": fingerprint,
            },
        }
    reordered = [dict(present[key]) for key in keys]
    for sequence, spot in enumerate(reordered, start=1):
        spot["sequence"] = sequence
    return {
        "spots": reordered,
        "order": {
            "state": "operator",
            "reason": APPLIED,
            "reason_he": APPLIED_HE,
            "saved_at": record.get("saved_at", ""),
            "actor": record.get("actor", ""),
            "note": record.get("note", ""),
        },
    }
