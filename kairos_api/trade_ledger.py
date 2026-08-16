"""The credit ledger: who holds make-good credit, where it came from, where it went.

The sibling ``makegood_store`` is the decision ledger — what was decided about
one campaign's shortfall — and holds no balance. The trade this product serves
(docs/trade/domain.md section 8) manages compensation as an
accrual-and-utilisation ledger at three levels AT ONCE: a campaign, an
advertiser and an agency each hold credit, an agency accrues under its framework
— a share of net spend per quarter, say — and spends the credit on a DIFFERENT
campaign later. That object is deliberately larger than the foreign shape that
binds each cure to its own deal, and this module is it. The two ledgers link and
never merge: a settlement spending credit stamps ``source_makegood_id``.

Five rules hold this store honest, each enforced here rather than explained on a
surface. **Credit is lots, and spending is FIFO**: every entry that adds credit
opens a lot with its own expiry; a utilisation consumes the oldest lots still in
force first and stamps which lots it took how much from in ``consumes``, so
"where did it go" is answered by the row itself, never by re-running an
allocator. **Nothing takes a balance below zero**: the consuming directions are
refused as an overdraft, by name; an adjustment only adds, and the correction
that removes credit is a utilisation carrying ``manual_adjust``, so it stands
under the same refusal. **The unit is named on every entry and never assumed**:
shekel media value, airtime seconds and rating points share this file and share
nothing else — every balance, lot walk and refusal is computed within one unit.
**No rate is invented here**: what percentage accrues, on what spend, is the
agreement's term (``makegood-accrual-policy``, ``added-value-media``); the
obligation engine reads the approved termset and calls this ledger with a
quantity. **Nothing is deleted**: the file is append-only, an expiry is an entry
naming the lots it kills, and a wrong entry is corrected by a counter-entry.
"""

from __future__ import annotations

import os
import shutil
import threading
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

# The controlled values and bilingual labels live beside this module, split out
# under the section 8.2 helper naming rule for the 450-line law.
from kairos_api.trade_ledger_words import (  # noqa: F401
    ACCRUE, ADDED_VALUE_GRANT, ADJUST, ADVERTISER, AGENCY, CAMPAIGN, CONSUMING_DIRECTIONS,
    DIRECTION_VOCABULARY, DIRECTIONS, EXPIRE, EXPIRY, ILS_MEDIA_VALUE, LEVEL_VOCABULARY, LEVELS,
    LOT_DIRECTIONS, MANUAL_ADJUST, NOTE_REQUIRED, POLICY_ACCRUAL, PREEMPTION_CREDIT,
    RATING_POINTS, REASON_VOCABULARY, REASONS, REASONS_FOR_DIRECTION, SECONDS, SHORTFALL_CURE,
    UNIT_VOCABULARY, UNITS, UTILISE, reason_allowed,
)
from kairos_api.store_columns import projected

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
BACKUP_DIR = DATA_DIR / "_backups"
LEDGER_PATH = DATA_DIR / "trade_credit_ledger.csv"

# How far ahead a balance block warns of credit about to die.
EXPIRING_SOON_DAYS = 60

COLUMNS = [
    "entry_id",
    # Which of the three balances this entry moves, and whose.
    "level", "party_ref",
    "direction", "quantity", "unit", "reason_code",
    # Where the movement came from: the agreement and term that set the policy,
    # and the decision-ledger row when a settlement spends credit.
    "source_agreement_id", "source_term_instance_id", "source_makegood_id",
    # Which lots a consuming entry took, as ``TC_0001:100|TC_0002:20``. Stamped
    # at write time so the allocation is a recorded fact, not a re-derivation.
    "consumes",
    "effective_on", "expires_on", "note",
    "created_at", "created_by", "is_demo",
]

_STORE_LOCK = threading.Lock()


def now_stamp() -> str:
    """The instant an act is recorded at, in UTC, to the second."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def lock() -> threading.Lock:
    """The store lock. The writers take it themselves; do not hold it around them."""
    return _STORE_LOCK


def load_frame() -> pd.DataFrame:
    """Every entry, or an empty frame when the ledger has never been written."""
    if not LEDGER_PATH.exists():
        return pd.DataFrame(columns=COLUMNS)
    frame = pd.read_csv(LEDGER_PATH, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    for column in COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    return frame


def _backup() -> None:
    if not LEDGER_PATH.exists():
        return
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    shutil.copy2(LEDGER_PATH, BACKUP_DIR / f"trade_credit_ledger_{stamp}.csv")


def write_frame(frame: pd.DataFrame) -> None:
    """Back up, then write atomically, exactly as the sibling stores do."""
    _backup()
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER_PATH.with_name(LEDGER_PATH.name + ".tmp")
    projected(frame, COLUMNS).to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, LEDGER_PATH)


def blank_row() -> dict[str, str]:
    return {column: "" for column in COLUMNS}


def next_id(frame: pd.DataFrame) -> str:
    """The next TC_ id, one above the highest already used."""
    highest = 0
    for value in frame.get("entry_id", []):
        text = str(value or "").strip()
        if text.startswith("TC_") and text[3:].isdigit():
            highest = max(highest, int(text[3:]))
    return f"TC_{highest + 1:04d}"


def _text(row: Any, column: str) -> str:
    return str(row.get(column, "") or "").strip()


def _truth(raw: Any) -> bool:
    return str(raw or "").strip().lower() in {"true", "yes", "1", "y"}


def _amount(raw: Any, field: str) -> float:
    try:
        return round(float(str(raw).strip()), 4)
    except (TypeError, ValueError):
        raise ValueError(f"{field} must be a number, got {raw!r}") from None


def _amount_text(value: float) -> str:
    """A number as the file writes it: no trailing zeros, never scientific."""
    return f"{round(value, 4):.4f}".rstrip("0").rstrip(".")


def _day(raw: Any, field: str) -> date:
    if isinstance(raw, date):
        return raw
    try:
        return date.fromisoformat(str(raw or "").strip())
    except ValueError:
        raise ValueError(f"{field} must be an ISO date (YYYY-MM-DD), got {raw!r}") from None


def _consumption(row: Any) -> list[dict[str, Any]]:
    """The lots one consuming entry took, parsed from its ``consumes`` stamp."""
    out: list[dict[str, Any]] = []
    for piece in filter(str.strip, _text(row, "consumes").split("|")):
        lot_id, _, taken = piece.rpartition(":")
        out.append({"entry_id": lot_id.strip(), "quantity": _amount(taken, "consumes")})
    return out


def record(row: Any) -> dict[str, Any]:
    """One entry as the API reports it, with every figure a number."""
    entry: dict[str, Any] = {column: _text(row, column) for column in COLUMNS}
    entry["quantity"] = _amount(row.get("quantity"), "quantity")
    entry["consumes"] = _consumption(row)
    entry["is_demo"] = _truth(row.get("is_demo"))
    return entry


# ------------------------------------------------------------------- lot math

def _scope_rows(frame: pd.DataFrame, level: str, party_ref: str, unit: str) -> list[dict]:
    rows = [dict(row) for _, row in frame.iterrows()
            if _text(row, "level") == level and _text(row, "party_ref") == party_ref
            and _text(row, "unit") == unit]
    return sorted(rows, key=lambda row: (_text(row, "effective_on"), _text(row, "entry_id")))


def _lot_state(rows: list[dict], up_to: Optional[date] = None) -> list[dict[str, Any]]:
    """Every lot in one scope, oldest first, with its remaining quantity.

    ``remaining`` is the lot's quantity minus every stamped consumption of it —
    from all rows, or only rows effective on or before ``up_to``. Consumption is
    read from the stamps, never re-allocated, so the file remains the record.
    """
    consumed: dict[str, float] = {}
    for row in rows:
        if up_to is not None and _day(_text(row, "effective_on"), "effective_on") > up_to:
            continue
        for take in _consumption(row):
            consumed[take["entry_id"]] = consumed.get(take["entry_id"], 0.0) + take["quantity"]
    lots: list[dict[str, Any]] = []
    for row in rows:
        if _text(row, "direction") not in LOT_DIRECTIONS:
            continue
        on = _day(_text(row, "effective_on"), "effective_on")
        if up_to is not None and on > up_to:
            continue
        dies = _text(row, "expires_on")
        entry_id = _text(row, "entry_id")
        quantity = _amount(row.get("quantity"), "quantity")
        lots.append({
            "entry_id": entry_id,
            "effective_on": on,
            "expires_on": _day(dies, "expires_on") if dies else None,
            "quantity": quantity,
            "remaining": round(quantity - consumed.get(entry_id, 0.0), 4),
        })
    return lots


def _live(lot: dict[str, Any], on: date) -> bool:
    """A lot is spendable through its expiry date and lapses the day after."""
    return lot["expires_on"] is None or lot["expires_on"] >= on


def _allocate(frame: pd.DataFrame, level: str, party_ref: str, unit: str,
              quantity: float, on: date, direction: str) -> list[tuple[str, float]]:
    """The FIFO allocation for one consuming entry, or the overdraft refusal.

    A utilisation draws on lots still in force on its effective day; an expiry on
    lots already past theirs. Both walk oldest-first and both are refused, with
    the numbers, when the pool cannot cover the quantity.
    """
    lots = _lot_state(_scope_rows(frame, level, party_ref, unit))
    eligible = [lot for lot in lots if lot["remaining"] > 0
                and (_live(lot, on) if direction == UTILISE else not _live(lot, on))]
    pool = round(sum(lot["remaining"] for lot in eligible), 4)
    if quantity > pool + 1e-9:
        lapsed = round(sum(lot["remaining"] for lot in lots
                           if lot["remaining"] > 0 and not _live(lot, on)), 4)
        beside = (f"; a further {_amount_text(lapsed)} lapsed past expiry and waits for expire_due"
                  if direction == UTILISE and lapsed > 0 else "")
        raise ValueError(
            f"{direction} of {_amount_text(quantity)} {unit} refused: an overdraft — "
            f"{party_ref} holds {_amount_text(pool)} {unit} at the {level} level on "
            f"{on.isoformat()}{beside}"
        )
    takes: list[tuple[str, float]] = []
    need = quantity
    for lot in eligible:
        if need <= 1e-9:
            break
        take = round(min(lot["remaining"], need), 4)
        takes.append((lot["entry_id"], take))
        need = round(need - take, 4)
    return takes


# ------------------------------------------------------------------- writing

def append_entry(
    *, level: str, party_ref: str, direction: str, quantity: Any, unit: str,
    reason_code: str, actor: str, effective_on: Any, expires_on: Any = "",
    source_agreement_id: str = "", source_term_instance_id: str = "",
    source_makegood_id: str = "", note: str = "", is_demo: bool = False,
) -> dict[str, Any]:
    """Validate one movement, allocate its lots when it consumes, and append it.

    Raises ``ValueError`` naming the exact refusal: an unknown level, direction,
    unit or reason, a reason that does not belong to the direction, a
    non-positive quantity, an expiry on an entry that opens no lot, a manual
    movement without a note, or an overdraft.
    """
    party = str(party_ref or "").strip()
    if not party:
        raise ValueError("an entry needs a party_ref: the campaign id, advertiser name or agency id whose balance it moves")
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if unit not in UNITS:
        raise ValueError(f"the unit is named per entry, never assumed: it must be one of {UNITS}, got {unit!r}")
    if reason_code not in REASONS:
        raise ValueError(f"reason_code must be one of {REASONS}, got {reason_code!r}")
    if not reason_allowed(direction, reason_code):
        raise ValueError(f"reason {reason_code!r} does not belong to direction {direction!r}; "
                         f"allowed: {sorted(REASONS_FOR_DIRECTION[direction])}")
    if not str(actor or "").strip():
        raise ValueError("an entry records who wrote it; actor is required")
    amount = _amount(quantity, "quantity")
    if amount <= 0:
        raise ValueError(f"quantity must be positive — the direction carries the sign — got {_amount_text(amount)}")
    on = _day(effective_on, "effective_on")
    dies: Optional[date] = None
    if str(expires_on or "").strip():
        if direction not in LOT_DIRECTIONS:
            raise ValueError(f"expires_on belongs to credit entering the ledger "
                             f"({sorted(LOT_DIRECTIONS)}), not to {direction!r}")
        dies = _day(expires_on, "expires_on")
        if dies < on:
            raise ValueError(f"expires_on {dies.isoformat()} is before effective_on {on.isoformat()}")
    if reason_code in NOTE_REQUIRED and not str(note or "").strip():
        raise ValueError(f"a {reason_code} movement requires a note saying why")
    with _STORE_LOCK:
        frame = load_frame()
        consumes = ""
        if direction in CONSUMING_DIRECTIONS:
            takes = _allocate(frame, level, party, unit, amount, on, direction)
            consumes = "|".join(f"{lot_id}:{_amount_text(take)}" for lot_id, take in takes)
        row = blank_row()
        row.update({
            "entry_id": next_id(frame), "level": level, "party_ref": party,
            "direction": direction, "quantity": _amount_text(amount), "unit": unit,
            "reason_code": reason_code, "consumes": consumes,
            "source_agreement_id": str(source_agreement_id or "").strip(),
            "source_term_instance_id": str(source_term_instance_id or "").strip(),
            "source_makegood_id": str(source_makegood_id or "").strip(),
            "effective_on": on.isoformat(), "expires_on": dies.isoformat() if dies else "",
            "note": str(note or "").strip(), "created_at": now_stamp(),
            "created_by": str(actor).strip(), "is_demo": "true" if is_demo else "",
        })
        frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
        write_frame(frame)
    return record(row)


def expire_due(as_of: Any, actor: str) -> list[dict[str, Any]]:
    """Write off every lot past its expiry, one entry per balance, naming the lots.

    Idempotent: a lot already fully consumed or expired contributes nothing, so
    running the sweep twice writes nothing the second time. Returns what it wrote.
    """
    day = _day(as_of, "as_of")
    if not str(actor or "").strip():
        raise ValueError("an entry records who wrote it; actor is required")
    written: list[dict[str, Any]] = []
    with _STORE_LOCK:
        frame = load_frame()
        scopes = sorted({(_text(row, "level"), _text(row, "party_ref"), _text(row, "unit"))
                         for _, row in frame.iterrows()
                         if _text(row, "direction") in LOT_DIRECTIONS})
        for level, party, unit in scopes:
            lots = _lot_state(_scope_rows(frame, level, party, unit))
            due = [lot for lot in lots if lot["remaining"] > 0 and not _live(lot, day)]
            if not due:
                continue
            row = blank_row()
            row.update({
                "entry_id": next_id(frame), "level": level, "party_ref": party,
                "direction": EXPIRE, "reason_code": EXPIRY, "unit": unit,
                "quantity": _amount_text(round(sum(lot["remaining"] for lot in due), 4)),
                "consumes": "|".join(f"{lot['entry_id']}:{_amount_text(lot['remaining'])}"
                                     for lot in due),
                "effective_on": day.isoformat(),
                "note": f"written off by expire_due; past expiry on {day.isoformat()}",
                "created_at": now_stamp(), "created_by": str(actor).strip(),
            })
            frame = pd.concat([frame, pd.DataFrame([row])], ignore_index=True)
            written.append(record(row))
        if written:
            write_frame(frame)
    return written


# ------------------------------------------------------------------- reading

def balances(level: Optional[str] = None, party_ref: Optional[str] = None,
             unit: Optional[str] = None, as_of: Any = None) -> list[dict[str, Any]]:
    """Every balance the filters keep, one block per (level, party_ref, unit).

    A block never sums across units, and its figures obey one identity, every
    quantity accounted for: accrued + adjusted == utilised + expired + lapsed +
    available. ``available`` is what a utilisation may draw on the as-of day;
    ``lapsed`` is credit past its expiry with no expiry entry written yet — not
    spendable, not yet ``expired``, waiting for :func:`expire_due`;
    ``expiring_soon`` lists live lots dying within ``EXPIRING_SOON_DAYS``.
    """
    day = _day(as_of, "as_of") if as_of else date.today()
    frame = load_frame()
    scopes = sorted({(_text(row, "level"), _text(row, "party_ref"), _text(row, "unit"))
                     for _, row in frame.iterrows()
                     if _day(_text(row, "effective_on"), "effective_on") <= day
                     and (level is None or _text(row, "level") == level)
                     and (party_ref is None or _text(row, "party_ref") == str(party_ref).strip())
                     and (unit is None or _text(row, "unit") == unit)})
    blocks: list[dict[str, Any]] = []
    for scope_level, scope_party, scope_unit in scopes:
        rows = [row for row in _scope_rows(frame, scope_level, scope_party, scope_unit)
                if _day(_text(row, "effective_on"), "effective_on") <= day]
        sums = {ACCRUE: 0.0, ADJUST: 0.0, UTILISE: 0.0, EXPIRE: 0.0}
        for row in rows:
            direction = _text(row, "direction")
            if direction in sums:
                sums[direction] += _amount(row.get("quantity"), "quantity")
        available = lapsed = 0.0
        soon: list[dict[str, Any]] = []
        for lot in _lot_state(rows, up_to=day):
            if lot["remaining"] <= 0:
                continue
            if not _live(lot, day):
                lapsed += lot["remaining"]
                continue
            available += lot["remaining"]
            if lot["expires_on"] is not None and lot["expires_on"] <= day + timedelta(days=EXPIRING_SOON_DAYS):
                soon.append({"entry_id": lot["entry_id"], "expires_on": lot["expires_on"].isoformat(),
                             "remaining": lot["remaining"], "unit": scope_unit})
        blocks.append({
            "level": scope_level, "party_ref": scope_party, "unit": scope_unit,
            "as_of": day.isoformat(),
            "accrued": round(sums[ACCRUE], 4), "adjusted": round(sums[ADJUST], 4),
            "utilised": round(sums[UTILISE], 4), "expired": round(sums[EXPIRE], 4),
            "lapsed": round(lapsed, 4), "available": round(available, 4),
            "expiring_soon": soon,
        })
    return blocks


def statement(party_ref: str, level: str) -> list[dict[str, Any]]:
    """One party's entries in order, each stamped with the balance after it.

    The running balance is per unit — the entry's own — and is the signed sum of
    written entries, so a lapsed lot does not move it until :func:`expire_due`
    writes the expiry. The written record is the statement.
    """
    frame = load_frame()
    rows = sorted([dict(row) for _, row in frame.iterrows()
                   if _text(row, "level") == level
                   and _text(row, "party_ref") == str(party_ref).strip()],
                  key=lambda row: (_text(row, "effective_on"), _text(row, "entry_id")))
    running: dict[str, float] = {}
    out: list[dict[str, Any]] = []
    for row in rows:
        entry = record(row)
        sign = 1.0 if entry["direction"] in LOT_DIRECTIONS else -1.0
        running[entry["unit"]] = round(running.get(entry["unit"], 0.0) + sign * entry["quantity"], 4)
        entry["running_balance"] = running[entry["unit"]]
        out.append(entry)
    return out


def vocabularies() -> dict[str, Any]:
    """Every controlled word the ledger writes, both languages."""
    return {
        "levels": [dict(entry) for entry in LEVEL_VOCABULARY],
        "directions": [dict(entry) for entry in DIRECTION_VOCABULARY],
        "units": [dict(entry) for entry in UNIT_VOCABULARY],
        "reasons": [dict(entry) for entry in REASON_VOCABULARY],
        "reasons_for_direction": {d: sorted(r) for d, r in REASONS_FOR_DIRECTION.items()},
        "note_required": sorted(NOTE_REQUIRED),
        "expiring_soon_days": EXPIRING_SOON_DAYS,
    }
