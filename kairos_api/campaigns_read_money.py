"""The Clients money layer: one priced ledger, grouped the four ways a client asks for it.

Split out of :mod:`kairos_api.campaigns_read` to keep that module under the
project line limit. It is a read. It runs no pricing of its own: it calls the
one composition the product already has for the daily per-spot path, the same
one the spots export, the agency summary and :mod:`kairos_api.spot_ledger` all
call, so a figure read here can never disagree with the same figure read there.

What it produces is one tree with the levels a money question actually walks:

    total -> advertiser -> campaign -> spot

and two more groupings over the identical rows, by agency and by break, because
"what did this agency deliver" and "what was in that break" are the same
question asked along a different edge. Every level sums to the level above it,
exactly, and a test asserts it against the ledger's own totals.

The honest limits, stated here rather than discovered on the surface:

  * **One broadcast day, not a month.** The product prices the newest daily
    file, and there is one on disk. Every figure below is that day. A month is
    not a period this data has, and the payload says so in ``basis`` rather
    than summing a month that does not exist.
  * **Gross and net are the same two quantities the ledger defines.** Gross is
    the priced revenue; net is gross after the agency rebate, reporting only.
    Nothing here invoices and nothing here is projected.
  * **Money that was dropped is money that is not there for a stated reason.**
    The shipped frequency rule drops 56 of the day's 175 spots, so the dropped
    rows travel with the board, carrying the rule that removed them.
  * **No channel column exists on the daily file**, so no rival row can enter
    this payload. The scope names the operator's own channel from settings and
    says how it was established, instead of claiming a filter that ran.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from kairos_api import read_cache

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily_input"

_NAMESPACE = "clients_money"

NO_LEDGER_REASON = "No daily spot file on disk, so there is no priced ledger to read."
NO_LEDGER_REASON_HE = "אין קובץ תשדירים יומי בדיסק, ולכן אין ספר מתומחר לקרוא."
PIPELINE_FAILED_REASON = "The daily pricing pipeline could not run, so no figure is reported."
PIPELINE_FAILED_REASON_HE = "נתיב התמחור היומי לא הצליח לרוץ, ולכן לא מדווח שום סכום."
PERIOD_NOTE_EN = "One broadcast day. The product prices the newest daily file, and this is it."
PERIOD_NOTE_HE = "יום שידור אחד. המערכת מתמחרת את קובץ היומי העדכני ביותר, וזהו הוא."
SCOPE_NOTE_EN = "The daily file carries no channel column, so the scope is the operator channel from settings."
SCOPE_NOTE_HE = "בקובץ היומי אין עמודת ערוץ, ולכן ההיקף הוא ערוץ המפעיל מתוך ההגדרות."
MONTH_PATH_EN = "Upload more daily files to widen the period. One day is priced today."
MONTH_PATH_HE = "העלו קבצים יומיים נוספים כדי להרחיב את התקופה. היום מתומחר יום אחד."

# Every input the priced ledger reads. A change to any one of them changes the
# money, so all of them are in the fingerprint rather than the daily file alone.
_INPUT_PATHS = (
    DATA_DIR / "advertiser_rules.csv",
    DATA_DIR / "advertiser_conditions.csv",
    DATA_DIR / "advertiser_names.csv",
    DATA_DIR / "agencies.csv",
    DATA_DIR / "agency_conditions.csv",
    DATA_DIR / "agency_advertisers.csv",
    DATA_DIR / "frequency_rules.csv",
    DATA_DIR / "rate_card_premiums.csv",
    DATA_DIR / "manual_overrides.csv",
    DATA_DIR / "kairos_settings.json",
)


def _daily_files() -> list[Path]:
    """Every daily file on disk, so the period statement can be counted."""
    if not DAILY_DIR.exists():
        return []
    return sorted(DAILY_DIR.glob("Wally_*.csv"))


def _fingerprint() -> tuple[Any, ...]:
    return read_cache.file_signatures([*_daily_files(), *_INPUT_PATHS])


def _round(value: float) -> float:
    return round(float(value or 0.0), 2)


def _operator_channel() -> str:
    try:
        from kairos_api import channel_scope

        return channel_scope.operator_channel()
    except Exception:  # noqa: BLE001 - a missing setting is a state, not a crash
        return ""


def _basis(path: Optional[Path], priced: int, dropped: int) -> dict[str, Any]:
    """What these figures are of: the file, the day, the scope, the coverage."""
    files = _daily_files()
    day = ""
    if path is not None:
        stem = path.stem
        day = stem[-10:] if len(stem) >= 10 and stem[-10:].count("-") == 2 else ""
    return {
        "file": path.name if path is not None else None,
        "day": day,
        "daily_files_on_disk": len(files),
        "priced_spots": priced,
        "rows_in_file": priced + dropped,
        "scope_channel": _operator_channel(),
        "scope_note_en": SCOPE_NOTE_EN,
        "scope_note_he": SCOPE_NOTE_HE,
        "period_note_en": PERIOD_NOTE_EN,
        "period_note_he": PERIOD_NOTE_HE,
        "wider_period_en": MONTH_PATH_EN,
        "wider_period_he": MONTH_PATH_HE,
    }


def _spot_record(ordinal: int, spot: Any) -> dict[str, Any]:
    """One priced spot as a row a person can read, with its own address."""
    gross = _round(spot.revenue)
    net = _round(spot.net_revenue)
    return {
        "spot_key": f"S{ordinal:03d}",
        "advertiser": spot.advertiser,
        "campaign": spot.campaign,
        "agency": spot.agency,
        "programme": spot.program,
        "genre": spot.genre,
        "daypart": spot.daypart,
        "break_id": spot.break_id,
        "position": spot.position,
        "ad": spot.ad,
        "duration_seconds": _round(spot.duration_seconds),
        "planned_tvr": round(float(spot.planned_tvr or 0.0), 4),
        "pricing_type": spot.pricing_type,
        "premium": round(float(spot.premium or 1.0), 6),
        "gross": gross,
        "rebate_percent": round(float(spot.rebate_percent or 0.0), 4),
        "rebate": _round(gross - net),
        "net": net,
    }


def _dropped_record(ordinal: int, drop: Any, kind: str = "frequency",
                    rules: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """One spot a rule removed, carrying why it went in words a reader can act on.

    The machine reason the engine writes stays on the payload, because an export
    and an audit both want it verbatim. It is not what the surface prints: a rule
    id and ``max_per_break=1 reached for ... in break ...`` are a log line, and a
    person reading a money board is owed the sentence instead.
    """
    from kairos_api import campaigns_read_money_reasons as reasons

    if kind == "frequency":
        index = rules if rules is not None else reasons.frequency_rules()
        english, hebrew, known = reasons.explain_drop(drop, index)
    else:
        english, hebrew, known = reasons.RULE_DROP_EN, reasons.RULE_DROP_HE, True
    return {
        "spot_key": f"D{ordinal:03d}",
        "advertiser": drop.advertiser,
        "campaign": drop.campaign,
        "ad": getattr(drop, "ad", ""),
        "break_id": getattr(drop, "break_id", ""),
        "rule_id": getattr(drop, "rule_id", ""),
        "limit_type": getattr(drop, "limit_type", ""),
        "reason": getattr(drop, "reason", ""),
        "kind": kind,
        "limit_known": known,
        "explanation_en": english,
        "explanation_he": hebrew,
    }


def _rules_behind(drops: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Which rules removed these spots, each named once, most first.

    Every removed spot already carries the sentence for the rule that removed
    it. What the summaries above them carried was a COUNT and the bare word
    "rule": a reader was told that money had been left out of a total and not
    told what to change to get it back. The rule was reachable — one drawer, one
    campaign, one spot down — which is not the same as being told.

    A level can involve more than one rule, so this is a list. It is never
    collapsed to "several rules": a surface that cannot fit them all should say
    how many there are and show the largest, and that is a decision for the
    surface, not for this reader.
    """
    counts: dict[str, int] = {}
    limits: dict[str, str] = {}
    for drop in drops:
        rule_id = str(drop.get("rule_id") or "")
        counts[rule_id] = counts.get(rule_id, 0) + 1
        limits.setdefault(rule_id, str(drop.get("limit_type") or ""))

    # The CAP, not one spot's story. Each removed spot carries a sentence naming
    # the advertiser and the break it lost — right for that row, wrong the
    # moment it stands over a total: "the cap was already reached for Factory 54
    # in the 20:40 break" is a true sentence about one of fifty-six spots and a
    # false summary of all of them. The pacing vocabulary already composes the
    # rule itself — "at most one spot per client per break" — and that is what a
    # summary is owed.
    try:
        from kairos_api import pacing_alerts_api_words as words

        sentences = words.booking_rules(counts)
    except Exception:  # noqa: BLE001 - an unreadable vocabulary is unknown, not a crash
        sentences = {}

    out: list[dict[str, Any]] = []
    for rule_id, spots in counts.items():
        block = sentences.get(rule_id) or {}
        out.append({
            "rule_id": rule_id,
            "limit_type": limits.get(rule_id, ""),
            "known": bool(block.get("known")),
            "sentence_en": block.get("rule_en", ""),
            "sentence_he": block.get("rule_he", ""),
            "spots": spots,
        })
    return sorted(out, key=lambda r: (-r["spots"], r["rule_id"]))


def _blank_totals() -> dict[str, Any]:
    return {
        "gross": None,
        "net": None,
        "rebates": None,
        "spots": None,
        "dropped_by_frequency": None,
        "dropped_by_rule": None,
        # None, not an empty list. With no ledger to read, "no rule removed
        # anything" is a claim and not a blank, exactly as a zero would be.
        "dropped_rules": None,
    }


def _sum_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gross = _round(sum(row["gross"] for row in rows))
    net = _round(sum(row["net"] for row in rows))
    return {
        "gross": gross,
        "net": net,
        "rebates": _round(gross - net),
        "spots": len(rows),
    }


def _group_by(rows: list[dict[str, Any]], field: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(field) or ""), []).append(row)
    return grouped


def _ranked(groups: dict[str, list[dict[str, Any]]], field: str, total_gross: float) -> list[dict[str, Any]]:
    """One row per group, ranked by gross, each carrying its share and its keys."""
    records = []
    for key, rows in groups.items():
        summed = _sum_rows(rows)
        share = summed["gross"] / total_gross if total_gross else 0.0
        records.append({
            field: key,
            **summed,
            "share_of_gross": round(share, 6),
            "spot_keys": [row["spot_key"] for row in rows],
        })
    records.sort(key=lambda record: (-record["gross"], record[field]))
    for index, record in enumerate(records, start=1):
        record["rank"] = index
    return records


def _campaign_rows(rows: list[dict[str, Any]], total_gross: float) -> list[dict[str, Any]]:
    campaigns = _ranked(_group_by(rows, "campaign"), "campaign", total_gross)
    for campaign in campaigns:
        own = [row for row in rows if row["campaign"] == campaign["campaign"]]
        campaign["advertisers"] = sorted({row["advertiser"] for row in own})
        campaign["agencies"] = sorted({row["agency"] for row in own if row["agency"]})
        campaign["breaks"] = sorted({row["break_id"] for row in own if row["break_id"]})
    return campaigns


def _advertiser_rows(
    rows: list[dict[str, Any]],
    dropped: list[dict[str, Any]],
    total_gross: float,
) -> list[dict[str, Any]]:
    """Each advertiser with its money, its agencies and its own campaign rows."""
    advertisers = _ranked(_group_by(rows, "advertiser"), "advertiser", total_gross)
    dropped_by_advertiser = _group_by(dropped, "advertiser")
    for advertiser in advertisers:
        own = [row for row in rows if row["advertiser"] == advertiser["advertiser"]]
        advertiser["agencies"] = sorted({row["agency"] for row in own if row["agency"]})
        advertiser["campaigns"] = _campaign_rows(own, advertiser["gross"])
        advertiser["breaks"] = sorted({row["break_id"] for row in own if row["break_id"]})
        removed = dropped_by_advertiser.get(advertiser["advertiser"], [])
        advertiser["dropped_by_frequency"] = len(removed)
        advertiser["dropped_keys"] = [row["spot_key"] for row in removed]
        advertiser["dropped_rules"] = _rules_behind(removed)
    return advertisers


def _break_rows(rows: list[dict[str, Any]], total_gross: float) -> list[dict[str, Any]]:
    breaks = _ranked(_group_by(rows, "break_id"), "break_id", total_gross)
    for record in breaks:
        own = [row for row in rows if row["break_id"] == record["break_id"]]
        record["programmes"] = sorted({row["programme"] for row in own if row["programme"]})
        record["advertisers"] = sorted({row["advertiser"] for row in own})
        record["seconds"] = _round(sum(row["duration_seconds"] for row in own))
    breaks.sort(key=lambda record: record["break_id"])
    return breaks


def _agency_rows(rows: list[dict[str, Any]], total_gross: float) -> list[dict[str, Any]]:
    agencies = _ranked(_group_by(rows, "agency"), "agency", total_gross)
    for record in agencies:
        own = [row for row in rows if row["agency"] == record["agency"]]
        record["advertisers"] = sorted({row["advertiser"] for row in own})
        rebates = {row["rebate_percent"] for row in own}
        record["rebate_percent"] = sorted(rebates)[0] if len(rebates) == 1 else None
        record["rebate_percent_varies"] = len(rebates) > 1
    return agencies


def _unavailable(reason: str) -> dict[str, Any]:
    hebrew = {
        NO_LEDGER_REASON: NO_LEDGER_REASON_HE,
        PIPELINE_FAILED_REASON: PIPELINE_FAILED_REASON_HE,
    }
    return {
        "available": False,
        "reason": reason,
        "reason_he": hebrew.get(reason, ""),
        "basis": _basis(None, 0, 0),
        "totals": _blank_totals(),
        "advertisers": [],
        "agencies": [],
        "campaigns": [],
        "breaks": [],
        "spots": [],
        "dropped": [],
    }


def _build() -> dict[str, Any]:
    """Price the day once and group it. One composition, one set of numbers."""
    try:
        from kairos_api.exporters import _load_daily_pricing
        from kairos_api.uploads import _newest_daily

        result = _load_daily_pricing()
        path = _newest_daily()
    except Exception:  # noqa: BLE001 - an unreadable ledger is a state, not a crash
        return _unavailable(PIPELINE_FAILED_REASON)
    if result is None:
        return _unavailable(NO_LEDGER_REASON)

    from kairos_api.campaigns_read_money_reasons import frequency_rules

    rules = frequency_rules()
    rows = [_spot_record(index, spot) for index, spot in enumerate(result.priced, start=1)]
    dropped = [
        _dropped_record(index, drop, "frequency", rules)
        for index, drop in enumerate(result.frequency_dropped, start=1)
    ]
    rule_dropped = [
        _dropped_record(index, drop, "rule", rules)
        for index, drop in enumerate(result.dropped, start=1)
    ]
    totals = _sum_rows(rows)
    totals["dropped_by_frequency"] = len(dropped)
    totals["dropped_by_rule"] = len(rule_dropped)
    totals["dropped_rules"] = _rules_behind(dropped + rule_dropped)
    return {
        "available": True,
        "reason": "",
        "reason_he": "",
        "basis": _basis(path, len(rows), len(dropped) + len(rule_dropped)),
        "totals": totals,
        "advertisers": _advertiser_rows(rows, dropped, totals["gross"]),
        "agencies": _agency_rows(rows, totals["gross"]),
        "campaigns": _campaign_rows(rows, totals["gross"]),
        "breaks": _break_rows(rows, totals["gross"]),
        "spots": rows,
        "dropped": dropped + rule_dropped,
    }


def board() -> dict[str, Any]:
    """The whole money tree, rebuilt whenever any priced input changes.

    Cached on the fingerprint of every file the pricing path reads, so an
    operator edit to a rebate, a rule or the daily file itself is visible on the
    next read and a stale figure cannot be served. The build is the expensive
    part (the pipeline parses the day and prices it); the grouping is arithmetic
    over 175 rows.
    """
    return read_cache.cached(_NAMESPACE, "board", _fingerprint(), _build)


def money_for_advertiser(name: str) -> dict[str, Any]:
    """One advertiser's row, its campaigns and the spots behind them."""
    data = board()
    if not data["available"]:
        return {"available": False, "reason": data["reason"], "advertiser": name}
    wanted = str(name or "").strip()
    record = next(
        (row for row in data["advertisers"] if row["advertiser"] == wanted),
        None,
    )
    if record is None:
        return {
            "available": True,
            "found": False,
            "advertiser": wanted,
            "basis": data["basis"],
            "reason": "This advertiser has no priced spot in the daily file being read.",
        }
    keys = set(record["spot_keys"]) | set(record["dropped_keys"])
    return {
        "available": True,
        "found": True,
        "basis": data["basis"],
        "advertiser": record,
        "spots": [row for row in data["spots"] if row["spot_key"] in keys],
        "dropped": [row for row in data["dropped"] if row["spot_key"] in keys],
        "rank_of": len(data["advertisers"]),
    }
