"""One raw, all-days advertiser activity read for the in-product assistant.

The newest-file money ranking and one-break pod tools cannot safely answer
"what has advertiser X advertised in the data we have?" This module reads every
daily traffic file, resolves the advertiser conservatively, and aggregates all
matching raw rows before limiting result lists.

Daily uploads are replaceable snapshots, not an append-only event log.  When two
files cover the same broadcast day, the latest modified file is authoritative
for that day and the older version is reported as shadowed.  Re-uploads therefore
cannot double-count delivery.  Coverage always names the exact sourced days; a
date range between the first and last file is never presented as continuous
history, and no money is computed here.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from kairos_api import read_cache

ROOT = Path(__file__).resolve().parents[1]
DAILY_INPUT_DIR = ROOT / "data" / "daily_input"
CACHE_NAMESPACE = "assistant_advertiser_airings"
MAX_AIRINGS = 100
DEFAULT_AIRINGS = 50
MAX_GROUPS = 50
read_cache.configure(CACHE_NAMESPACE, capacity=4)


ADVERTISER_ACTIVITY_READ_TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_advertiser_airings",
        "description": (
            "Read what one advertiser has advertised across every covered raw daily "
            "traffic file, in one query. Returns exact source coverage, all-days totals, "
            "campaign/creative/break aggregates and paginated individual spots. Raw traffic "
            "rows are used, not the priced ledger, so pricing or frequency rules cannot hide "
            "a sourced spot. Coverage is only the dates actually on disk, never an assertion "
            "of complete history through today. Use this for questions such as what an "
            "advertiser advertised, how many spots it has, or where its campaigns appeared."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string",
                         "description": "Advertiser name or a stored exact alias."},
                "date_from": {"type": "string",
                              "description": "Optional inclusive first broadcast day, YYYY-MM-DD."},
                "date_to": {"type": "string",
                            "description": "Optional inclusive last broadcast day, YYYY-MM-DD."},
                "limit": {"type": "integer",
                    "description": "Detailed spots to return, 1-100 (default 50).",
                },
                "offset": {"type": "integer",
                           "description": "Zero-based offset into the ordered detailed spots."},
            },
            "required": ["name"],
        },
    }
]

def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()

def _number(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(parsed) else parsed

def _integer(value: Any) -> int | None:
    parsed = _number(value)
    return None if parsed is None else int(parsed)

def _clock(value: Any) -> str:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.strftime("%H:%M:%S")
    text = _text(value)
    parsed = pd.to_datetime(text, format="%H:%M:%S", errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(text, format="%H:%M", errors="coerce")
    return "" if pd.isna(parsed) else parsed.strftime("%H:%M:%S")

def _ranges(days: list[str]) -> list[dict[str, Any]]:
    parsed = sorted({date.fromisoformat(day) for day in days})
    if not parsed:
        return []
    ranges: list[dict[str, Any]] = []
    start = previous = parsed[0]
    for current in parsed[1:]:
        if current != previous + timedelta(days=1):
            ranges.append({"date_from": start.isoformat(), "date_to": previous.isoformat(),
                           "days": (previous - start).days + 1})
            start = current
        previous = current
    ranges.append({"date_from": start.isoformat(), "date_to": previous.isoformat(),
                   "days": (previous - start).days + 1})
    return ranges

def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date", "advertiser", "campaign", "creative", "house_number", "agency",
        "spot_type", "duration_sec", "program", "break_start", "spot_time",
        "break_type", "pricing_type", "position_in_break", "status", "_day",
        "_source_file", "_source_row",
    ])

def _build_corpus() -> dict[str, Any]:
    """Load every file and select one latest source version per actual day."""
    from kairos.data.loaders import load_daily_input

    paths = sorted(DAILY_INPUT_DIR.glob("Wally_*.csv")) if DAILY_INPUT_DIR.exists() else []
    candidates: dict[str, list[dict[str, Any]]] = {}
    files: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    rows_read = rows_without_day = 0

    for path in paths:
        try:
            frame = load_daily_input(path).copy()
        except Exception:  # noqa: BLE001 - the coverage block carries the failed file
            failed.append({
                "source_file": path.name,
                "reason": "the daily traffic loader could not parse this file",
            })
            continue
        if "date" not in frame.columns or "advertiser" not in frame.columns:
            failed.append({
                "source_file": path.name,
                "reason": "the parsed file lacks the date or advertiser column",
            })
            continue
        stat = path.stat()
        frame["_day"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        frame["_source_file"] = path.name
        frame["_source_row"] = list(range(2, len(frame) + 2))
        rows_read += len(frame)
        missing = int(frame["_day"].isna().sum())
        rows_without_day += missing
        days = sorted(str(day) for day in frame["_day"].dropna().unique())
        files.append({"source_file": path.name, "rows": len(frame), "covered_days": days})
        for day in days:
            candidates.setdefault(day, []).append({
                "source_file": path.name,
                "mtime_ns": stat.st_mtime_ns,
                "rows": frame[frame["_day"] == day].copy(),
            })

    selected: list[pd.DataFrame] = []
    shadowed: list[dict[str, Any]] = []
    day_sources: dict[str, str] = {}
    for day, versions in sorted(candidates.items()):
        winner = max(versions, key=lambda item: (item["mtime_ns"], item["source_file"]))
        selected.append(winner["rows"])
        day_sources[day] = winner["source_file"]
        if len(versions) > 1:
            shadowed.append({
                "day": day,
                "authoritative_file": winner["source_file"],
                "shadowed_files": sorted(
                    item["source_file"] for item in versions if item is not winner
                ),
            })
    corpus = pd.concat(selected, ignore_index=True) if selected else _empty_frame()
    return {
        "frame": corpus,
        "files_discovered": len(paths),
        "files": files,
        "failed": failed,
        "rows_read": rows_read,
        "rows_without_day": rows_without_day,
        "day_sources": day_sources,
        "shadowed": shadowed,
    }


def _corpus() -> dict[str, Any]:
    fingerprint = read_cache.directory_signatures(DAILY_INPUT_DIR, "Wally_*.csv")
    return read_cache.cached(
        CACHE_NAMESPACE, str(DAILY_INPUT_DIR.resolve()), fingerprint, _build_corpus
    )


def _parse_day(value: Any, field: str) -> tuple[date | None, str | None]:
    text = _text(value)
    if not text:
        return None, None
    try:
        return date.fromisoformat(text), None
    except ValueError:
        return None, f"{field} must be an ISO date (YYYY-MM-DD), got {text!r}"


def _pagination(args: dict[str, Any]) -> tuple[int, int]:
    try:
        limit = int(args.get("limit") or DEFAULT_AIRINGS)
    except (TypeError, ValueError):
        limit = DEFAULT_AIRINGS
    try:
        offset = int(args.get("offset") or 0)
    except (TypeError, ValueError):
        offset = 0
    return max(1, min(limit, MAX_AIRINGS)), max(0, offset)


def _identity_matcher(query: str, raw_names: list[str]) -> tuple[dict[str, Any], Callable[[str], bool]]:
    """Resolve aliases exactly; an unknown name may still match its raw spelling."""
    from kairos.optimize.advertiser_rules_identity import (
        _names_token_index,
        load_advertiser_names,
        load_name_index,
        normalize_name,
        resolve_advertiser,
    )
    from kairos_api.advertisers import RULES_PATH

    names = load_advertiser_names()
    tokens = _names_token_index(names)
    rules = load_name_index(RULES_PATH)
    resolved = resolve_advertiser(
        query, names=names, rules_index=rules, names_tokens=tokens
    )
    if resolved is not None:
        wanted = normalize_name(resolved.key)

        def matches(raw: str) -> bool:
            row = resolve_advertiser(raw, names=names, rules_index=rules, names_tokens=tokens)
            return row is not None and normalize_name(row.key) == wanted

        matched = sorted({raw for raw in raw_names if matches(raw)})
        return ({
            "resolved": True,
            "canonical_name": resolved.name,
            "shown_name": resolved.shown_name,
            "aliases": list(resolved.aliases),
            "matched_on": resolved.matched_on,
            "source": resolved.source,
            "rules_bound": resolved.has_rules_row,
            "raw_names_matched": matched,
        }, matches)

    wanted = normalize_name(query)
    matched = sorted({raw for raw in raw_names if normalize_name(raw) == wanted})
    return ({
        "resolved": bool(matched),
        "canonical_name": matched[0] if matched else query,
        "shown_name": matched[0] if matched else query,
        "aliases": [],
        "matched_on": "raw_exact" if matched else "",
        "source": "daily_traffic" if matched else "",
        "rules_bound": False,
        "raw_names_matched": matched,
    }, lambda raw: bool(wanted) and normalize_name(raw) == wanted)


def _airing(row: Any) -> dict[str, Any]:
    duration = _number(row.get("duration_sec"))
    day = _text(row.get("_day"))
    clock = _clock(row.get("spot_time"))
    return {
        "day": day,
        "spot_time": clock or None,
        "airing_at": f"{day}T{clock}" if day and clock else None,
        "break_start": _clock(row.get("break_start")) or None,
        "programme": _text(row.get("program")) or None,
        "break_type": _text(row.get("break_type")) or None,
        "advertiser": _text(row.get("advertiser")),
        "campaign": _text(row.get("campaign")) or None,
        "creative": _text(row.get("creative")) or None,
        "house_number": _text(row.get("house_number")) or None,
        "agency": _text(row.get("agency")) or None,
        "spot_type": _text(row.get("spot_type")) or None,
        "duration_seconds": None if duration is None else round(duration, 3),
        "position_in_break": _integer(row.get("position_in_break")),
        "pricing_type": _text(row.get("pricing_type")) or None,
        "status": _text(row.get("status")) or None,
        "source_file": _text(row.get("_source_file")),
        "source_row": _integer(row.get("_source_row")),
    }


def _group(rows: list[dict[str, Any]], keys: tuple[str, ...], shown: tuple[str, ...]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row.get(key) for key in keys), []).append(row)
    result = []
    for key, members in groups.items():
        times = sorted(item["airing_at"] for item in members if item.get("airing_at"))
        entry = {name: value for name, value in zip(shown, key)}
        entry.update({
            "airings": len(members),
            "seconds": round(sum(item.get("duration_seconds") or 0 for item in members), 3),
            "first_airing_at": times[0] if times else None,
            "last_airing_at": times[-1] if times else None,
        })
        result.append(entry)
    return sorted(result, key=lambda item: (-item["airings"], str(tuple(item.get(k) for k in shown))))


def _cap_groups(payload: dict[str, Any]) -> None:
    for key in ("days", "campaigns", "creatives", "breaks"):
        rows = payload[key]
        payload[f"{key}_total"] = len(rows)
        if len(rows) > MAX_GROUPS:
            payload[f"{key}_omitted"] = len(rows) - MAX_GROUPS
            payload[key] = rows[:MAX_GROUPS]

def _coverage(data: dict[str, Any], frame: pd.DataFrame, selected: pd.DataFrame) -> dict[str, Any]:
    from kairos_api import channel_scope

    available_days = sorted(data["day_sources"])
    selected_days = sorted(str(day) for day in selected.get("_day", pd.Series(dtype=str)).dropna().unique())
    sources_used = sorted({data["day_sources"][day] for day in selected_days})
    channel = channel_scope.operator_channel() or None
    payload = {
        "source_pattern": "data/daily_input/Wally_*.csv",
        "files_discovered": data["files_discovered"],
        "files_read": len(data["files"]),
        "files_failed": data["failed"],
        "file_rows": data["files"],
        "rows_read": data["rows_read"],
        "rows_without_broadcast_day": data["rows_without_day"],
        "authoritative_rows": len(frame),
        "available_days": available_days,
        "available_ranges": _ranges(available_days),
        "selected_days": selected_days,
        "selected_ranges": _ranges(selected_days),
        "selected_rows": len(selected),
        "latest_covered_day": available_days[-1] if available_days else None,
        "source_files_used": sources_used,
        "shadowed_day_versions": data["shadowed"],
        "channel": {
            "value": channel,
            "basis": (
                "the operator channel from settings; daily traffic files carry no channel "
                "column, so no competitor channel can enter this result"
            ),
        },
        "complete_for_available_files": not data["failed"] and not data["rows_without_day"],
        "complete_through_today": False,
        "completeness_note": (
            "Coverage is exactly the listed broadcast days on disk, not continuous history "
            "between them and not a claim of coverage through today."
        ),
    }
    return payload


def _read_get_advertiser_airings(
    args: dict[str, Any], user: str | None = None
) -> dict[str, Any]:
    del user
    query = _text(args.get("name"))
    if not query:
        return {"error": "provide an advertiser name"}
    low, low_error = _parse_day(args.get("date_from"), "date_from")
    high, high_error = _parse_day(args.get("date_to"), "date_to")
    if low_error or high_error:
        return {"error": low_error or high_error}
    if low is not None and high is not None and low > high:
        return {"error": "date_from must be on or before date_to"}

    data = _corpus()
    frame = data["frame"]
    selected = frame
    if low is not None:
        selected = selected[selected["_day"] >= low.isoformat()]
    if high is not None:
        selected = selected[selected["_day"] <= high.isoformat()]
    raw_names = sorted({_text(value) for value in frame.get("advertiser", []) if _text(value)})
    identity, matches = _identity_matcher(query, raw_names)
    mask = selected["advertiser"].map(lambda value: matches(_text(value))).astype(bool)
    matched = selected[mask].copy()
    matched["_clock"] = matched.get("spot_time", pd.Series("", index=matched.index)).map(_clock)
    matched = matched.sort_values(["_day", "_clock", "_source_file", "_source_row"])
    rows = [_airing(row) for _, row in matched.iterrows()]
    limit, offset = _pagination(args)
    shown = rows[offset:offset + limit]
    times = [row["airing_at"] for row in rows if row["airing_at"]]
    days = _group(rows, ("day",), ("day",))
    campaigns = _group(rows, ("campaign",), ("campaign",))
    creatives = _group(rows, ("campaign", "creative", "house_number"),
                       ("campaign", "creative", "house_number"))
    breaks = _group(rows, ("day", "break_start", "programme"),
                    ("day", "break_start", "programme"))
    total = len(rows)
    status = "ok" if identity["resolved"] else "not_found"
    payload = {
        "status": status,
        "query": {"name": query, "date_from": low.isoformat() if low else None,
                  "date_to": high.isoformat() if high else None},
        "identity": identity,
        "coverage": _coverage(data, frame, selected),
        "summary": {
            "airings": total,
            "seconds": round(sum(row.get("duration_seconds") or 0 for row in rows), 3),
            "broadcast_days": len({row["day"] for row in rows}),
            "campaigns": len(campaigns),
            "creatives": len(creatives),
            "breaks": len(breaks),
            "agencies": sorted({row["agency"] for row in rows if row.get("agency")}),
            "first_airing_at": min(times) if times else None,
            "last_airing_at": max(times) if times else None,
        },
        "days": days,
        "campaigns": campaigns,
        "creatives": creatives,
        "breaks": breaks,
        "airings": shown,
        "pagination": {
            "offset": offset, "limit": limit, "returned": len(shown), "total": total,
            "has_more": offset + len(shown) < total,
            "next_offset": offset + len(shown) if offset + len(shown) < total else None,
        },
        "basis": (
            "Raw rows in the authoritative daily traffic file version for each covered "
            "broadcast day. Counts are not priced, invoiced, or reduced by advertiser, "
            "agency, frequency or separation rules."
        ),
    }
    _cap_groups(payload)
    return payload


_ADVERTISER_ACTIVITY_EXECUTORS = {
    "get_advertiser_airings": _read_get_advertiser_airings,
}

ADVERTISER_ACTIVITY_SOURCE_BY_TOOL = {
    "get_advertiser_airings": (
        "all authoritative raw daily traffic files on disk (data/daily_input/Wally_*.csv)"
    ),
}


def register(executors: dict[str, Any], sources: dict[str, str]) -> None:
    executors.update(_ADVERTISER_ACTIVITY_EXECUTORS)
    sources.update(ADVERTISER_ACTIVITY_SOURCE_BY_TOOL)
