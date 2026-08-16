"""Rebuild the booked-demand inventory from the canonical aired-spots log.

The optimizer consumes one CSV row per observed booked spot and groups those
rows by ``(channel, Date_dt, hour_of_day)``.  The checked-in inventory extract
lost all temporal fields and was scoped to a different broadcaster.  This
script creates the small, explicit contract the optimizer actually needs from
the canonical reference workbook and the saved operator channel.

This is a historical replay signal.  It does not claim remaining capacity,
future availability, observed revenue, or advertiser identity.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from kairos.data.loaders import load_spots

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "reference" / "Spots.xlsx"
DEFAULT_SETTINGS = ROOT / "data" / "kairos_settings.json"
DEFAULT_OUTPUT = ROOT / "data" / "Spots - inventory.csv"

OUTPUT_COLUMNS = (
    "source_row_id",
    "channel",
    "Date_dt",
    "Start_dt",
    "hour_of_day",
)


class InventoryBuildError(ValueError):
    """The source cannot prove a complete, operator-scoped inventory."""


def _operator_channel(settings_path: Path) -> str:
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise InventoryBuildError(f"Cannot read settings from {settings_path}") from exc
    channel = str(payload.get("operator_channel") or "").strip()
    if not channel:
        raise InventoryBuildError("operator_channel is required before rebuilding inventory")
    return channel


def build_inventory(source_path: Path, settings_path: Path) -> pd.DataFrame:
    """Return the canonical operator-scoped row-per-booked-spot inventory."""
    try:
        raw = pd.read_excel(source_path)
        parsed = load_spots(source_path)
    except (OSError, ValueError) as exc:
        raise InventoryBuildError(f"Cannot read the canonical source {source_path}") from exc

    required = {"Unnamed: 0", "Channel", "Date", "Start time"}
    missing = required.difference(raw.columns)
    if missing:
        raise InventoryBuildError(f"Canonical source is missing columns: {sorted(missing)}")
    if len(raw) != len(parsed):
        raise InventoryBuildError("Raw and parsed canonical sources have different row counts")
    if not raw["Unnamed: 0"].is_unique:
        raise InventoryBuildError("Canonical source row ids are not unique")

    raw_channels = raw["Channel"].fillna("").astype(str).str.strip()
    parsed_channels = parsed["Channel"].fillna("").astype(str).str.strip()
    if not raw_channels.equals(parsed_channels):
        raise InventoryBuildError("Canonical source row order changed during date parsing")

    channel = _operator_channel(settings_path)
    owned_mask = parsed_channels.eq(channel)
    owned = parsed.loc[owned_mask].copy()
    owned_ids = raw.loc[owned_mask, "Unnamed: 0"].copy()
    if owned.empty:
        raise InventoryBuildError(f"Canonical source has no rows for operator channel {channel!r}")
    if owned["air_dt"].isna().any():
        count = int(owned["air_dt"].isna().sum())
        raise InventoryBuildError(
            f"Canonical source has {count} operator rows without a provable air time"
        )

    numeric_ids = pd.to_numeric(owned_ids, errors="coerce")
    if numeric_ids.isna().any() or not numeric_ids.is_unique:
        raise InventoryBuildError("Operator source row ids are missing, invalid, or duplicated")
    integer_ids = numeric_ids.astype("int64")
    if not numeric_ids.eq(integer_ids).all():
        raise InventoryBuildError("Operator source row ids are not integers")

    moments = pd.to_datetime(owned["air_dt"], errors="coerce")
    result = pd.DataFrame(
        {
            "source_row_id": integer_ids.to_numpy(),
            "channel": channel,
            "Date_dt": moments.dt.strftime("%Y-%m-%d").to_numpy(),
            "Start_dt": moments.dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
            "hour_of_day": moments.dt.hour.astype("int64").to_numpy(),
        },
        columns=OUTPUT_COLUMNS,
    )
    if result.isna().any().any() or (result.astype(str).apply(lambda c: c.str.strip()) == "").any().any():
        raise InventoryBuildError("Rebuilt inventory contains an empty required field")
    if set(result["channel"]) != {channel}:
        raise InventoryBuildError("Rebuilt inventory crossed the operator channel boundary")
    return result


def _csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")


def _write_atomic(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Atomically replace the output. Without this flag, verify only.",
    )
    args = parser.parse_args()

    rebuilt = build_inventory(args.source, args.settings)
    payload = _csv_bytes(rebuilt)
    if args.write:
        _write_atomic(args.output, payload)
    elif not args.output.exists() or args.output.read_bytes() != payload:
        raise InventoryBuildError(
            f"{args.output} does not match the canonical operator-scoped rebuild"
        )

    slots = rebuilt.groupby(["channel", "Date_dt", "hour_of_day"], sort=True).size()
    print(
        f"inventory {'wrote' if args.write else 'verified'}: "
        f"rows={len(rebuilt)} slots={len(slots)} days={rebuilt['Date_dt'].nunique()} "
        f"channel={rebuilt['channel'].iloc[0]} output={args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
