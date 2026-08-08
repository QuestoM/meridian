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

**Locking is the same row, one flag further, and locking must never invent an
order.** The trade's own step after verification is finalising, so a pod can
be locked: whichever order is on screen is frozen and a further order write is
refused until it is unlocked. Locking reuses this register rather than opening
a second one, because the frozen order and the saved order are the same fact
from two moments.

But most locks happen on a pod nobody has reordered, and this register exists
to attribute a traffic decision to the operator who made it. So every row
carries ``order_source``, ``operator`` when an operator's own reorder produced
it and ``file`` when locking simply froze the file's own order with no operator
decision behind it. Locking a pod whose row carries ``order_source``
``operator`` touches only the three lock columns and leaves ``spot_keys``,
``saved_at`` and ``note`` exactly as the operator wrote them, **whether or not
that row's fingerprint still matches the pod.** A stale row stays reported
stale after the lock, now carrying the lock too: locking never applies an
order over a pod that changed underneath it, but it also never destroys the
record that an operator made one, which is the mirror mistake and just as
real a loss. Locking a pod that carries no row at all, or that carried only a
prior file-sourced lock, writes a fresh row with an empty ``spot_keys`` and
``order_source`` of ``file``, so :func:`applied` reports ``state`` ``file``
rather than ``operator`` or ``stale`` even while the lock is real. Unlocking
then leaves behind only what was really there: clearing the lock columns on
an operator row regardless of its staleness, or dropping a file-sourced row
entirely, so a lock and an unlock on a pod nobody ever reordered leaves the
register holding no trace of it.

Written with the same discipline as every other operator store in this package:
one lock over load, mutate and write, a backup before the write, and a temp file
plus ``os.replace`` so a reader never sees a torn CSV. Nothing here prices
anything and no figure moves through it.

**Where this file lives.** ``data/break_pod_order.csv`` is not a row in section
8.2's table; it is the runtime output of this module, which is itself a
declared helper module the 450-line law forced this piece to create. A helper
module's own data file is the module's to place, the way ``break_store.py``
places its CSVs without a table row naming each one. Stated here once so a
later reader does not need to re-derive it.
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

COLUMNS = ("pod_id", "spot_keys", "fingerprint", "actor", "saved_at", "note", "locked", "locked_at", "locked_by", "order_source")
KEY_SEPARATOR = "|"
SOURCE_OPERATOR = "operator"
SOURCE_FILE = "file"

STALE = "The traffic file changed after this order was saved, so the pod is shown in the order the file declares."
STALE_HE = "קובץ הטראפיק השתנה לאחר שמירת הסדר הזה, ולכן התוכן מוצג בסדר שהקובץ מצהיר עליו."
APPLIED = "This pod is shown in the order an operator saved, not the order the traffic file declares."
APPLIED_HE = "התוכן הזה מוצג בסדר ששמר מפעיל, ולא בסדר שקובץ הטראפיק מצהיר עליו."
FILE_ORDER = "This pod is shown in the order the traffic file declares."
FILE_ORDER_HE = "התוכן הזה מוצג בסדר שקובץ הטראפיק מצהיר עליו."
LOCKED_TRUE = "1"

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


def _order_source(record: dict[str, Any]) -> str:
    """Which decision this row records, tolerating a row written before this column.

    A row this module wrote carries ``order_source`` outright. A row an earlier
    version of this module wrote carries none, and every one of those rows was
    written by :func:`save` with real ``spot_keys``, so the presence of a key
    list is the honest fallback for it.
    """
    value = str(record.get("order_source", "") or "")
    if value:
        return value
    return SOURCE_OPERATOR if record.get("spot_keys") else SOURCE_FILE


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
        "locked": "",
        "locked_at": "",
        "locked_by": "",
        "order_source": SOURCE_OPERATOR,
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


def lock(pod_id: str, fingerprint: str, actor: str = "") -> dict[str, Any]:
    """Freeze this pod's current order and mark it finalised.

    Whether an operator decision exists to preserve is read from the register's
    own ``order_source`` column on the pod's current row, never from a
    comparison of saved keys against the file's own order, because a saved
    order that happens to equal the file's own order is indistinguishable from
    no order at all by keys alone. The row's own recorded intent has no such
    ambiguity, and reading it here means locking never has to be told the
    answer by a caller that computed it from a snapshot already a moment
    stale.

    A row whose own ``order_source`` is ``operator`` is left almost entirely
    alone, whether or not its fingerprint currently matches the pod: only the
    three lock columns change, so the note, the save time and the saved keys
    an operator actually wrote survive the lock untouched, and a stale row
    stays reported stale, now carrying the lock too, rather than having its
    order thrown away the moment it is finalised. Locking never applies a
    stale order over a pod that changed underneath it; it only ever refuses
    to destroy the record that an operator made one.

    No row at all, or a row whose own ``order_source`` is ``file`` (a previous
    lock that froze the file's own order with nothing recorded against it),
    means no operator decision exists to preserve. Any such row is dropped and
    a fresh one is written with an empty ``spot_keys`` and ``order_source`` of
    ``file``, which is what tells :func:`applied` to keep reporting ``state``
    ``file`` even though the pod is locked.
    """
    now = datetime.now(timezone.utc).isoformat()
    wanted = str(pod_id).strip()
    with _STORE_LOCK:
        frame = _load_frame()
        mask = frame["pod_id"].astype(str) == wanted if not frame.empty else None
        existing_idx = frame.index[mask][0] if mask is not None and mask.any() else None
        existing_record = _record(frame.loc[existing_idx]) if existing_idx is not None else None
        has_operator_order = existing_record is not None and _order_source(existing_record) == SOURCE_OPERATOR
        if existing_idx is not None and has_operator_order:
            frame.loc[existing_idx, "locked"] = LOCKED_TRUE
            frame.loc[existing_idx, "locked_at"] = now
            frame.loc[existing_idx, "locked_by"] = str(actor or "")
            _write_frame(frame)
            record = _record(frame.loc[existing_idx])
        else:
            if existing_idx is not None:
                frame = frame[~mask].reset_index(drop=True)
            record = {
                "pod_id": wanted,
                "spot_keys": "",
                "fingerprint": str(fingerprint or ""),
                "actor": "",
                "saved_at": "",
                "note": "",
                "locked": LOCKED_TRUE,
                "locked_at": now,
                "locked_by": str(actor or ""),
                "order_source": SOURCE_FILE,
            }
            frame = pd.concat([frame, pd.DataFrame([record])], ignore_index=True)
            _write_frame(frame)
    return record


def unlock(pod_id: str) -> Optional[dict[str, Any]]:
    """Clear a pod's lock, leaving behind only what was really there.

    A row that carries a real operator order (``order_source`` ``operator``)
    keeps that order; only the three lock columns clear. A row that carries no
    operator order at all, because locking only ever froze the traffic file's
    own order, is dropped outright, so unlocking a pod nobody ever reordered
    leaves the register holding no trace that anybody decided anything.
    """
    wanted = str(pod_id or "").strip()
    with _STORE_LOCK:
        frame = _load_frame()
        if frame.empty:
            return None
        mask = frame["pod_id"].astype(str) == wanted
        if not mask.any() or str(frame.loc[mask, "locked"].iloc[0]) != LOCKED_TRUE:
            return None
        idx = frame.index[mask][0]
        record = _record(frame.loc[idx])
        if _order_source(record) == SOURCE_FILE:
            frame = frame[~mask].reset_index(drop=True)
            _write_frame(frame)
            return {**record, "locked": "", "locked_at": "", "locked_by": ""}
        frame.loc[idx, ["locked", "locked_at", "locked_by"]] = ""
        _write_frame(frame)
        record = _record(frame.loc[idx])
    return record


def applied(pod_id: str, spots: list[dict[str, Any]], fingerprint: str) -> dict[str, Any]:
    """The pod's spots in the order they should be shown, and why that order.

    Four outcomes and each says which it is. No saved order leaves the file's
    own order alone. A row that carries a lock but no operator order, because
    locking only ever froze the file's own order, reports the same file state
    with the lock facts attached, rather than manufacturing an operator
    decision nobody made. A saved order whose fingerprint still matches is
    applied and the sequence numbers are restated so the surface never prints
    a position from one order beside a sequence from another. A saved order
    whose fingerprint has moved is reported stale and not applied. ``locked``
    travels on every outcome, never missing, so a caller never has to guess it
    as False by absence, and so do ``locked_at`` and ``locked_by``: a lock the
    register timed is a lock the audit line can name, and a stale row is the
    one outcome where the finalising moment matters most, because the order
    that was frozen is not the order on screen.
    """
    record = stored(pod_id)
    if record is None:
        return {"spots": spots, "order": {"state": "file", "reason": FILE_ORDER, "reason_he": FILE_ORDER_HE, "locked": False}}
    locked = str(record.get("locked", "")) == LOCKED_TRUE
    if _order_source(record) == SOURCE_FILE:
        return {
            "spots": spots,
            "order": {
                "state": "file",
                "reason": FILE_ORDER,
                "reason_he": FILE_ORDER_HE,
                "locked": locked,
                "locked_at": record.get("locked_at", "") if locked else "",
                "locked_by": record.get("locked_by", "") if locked else "",
            },
        }
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
                "note": record.get("note", ""),
                "saved_fingerprint": record.get("fingerprint", ""),
                "pod_fingerprint": fingerprint,
                "locked": locked,
                "locked_at": record.get("locked_at", "") if locked else "",
                "locked_by": record.get("locked_by", "") if locked else "",
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
            "locked": locked,
            "locked_at": record.get("locked_at", "") if locked else "",
            "locked_by": record.get("locked_by", "") if locked else "",
        },
    }
