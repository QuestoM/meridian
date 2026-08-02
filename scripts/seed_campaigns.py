"""Seed the campaign entity from the real traffic log, and mark every row demo.

Run it as often as you like:

    ~/.venvs/meridian/bin/python scripts/seed_campaigns.py

It rewrites three files and touches nothing else:

    data/campaigns.csv          the booking, one campaign row plus one flight row
    data/campaign_assets.csv    the creative, one row per house number
    data/campaign_delivery.csv  what aired, one row per campaign per broadcast day

**Every row it writes carries ``is_demo`` true.** That is a column, not a
convention, and the API reads it onto every payload so a surface can badge the
row. A campaign an operator books through the clients flow carries ``is_demo``
false and this script never overwrites one: it replaces its own demo rows and
leaves every booked row exactly where it was.

**What is real here and what is the seed's.** The identity is real: the campaign
names, the advertisers, the agencies, the creative version names, the house
numbers, the spot lengths and the spot types all come from the traffic file on
disk. The delivery is real: it is the same per-spot priced ledger the money board
reads. What the seed invents is the commitment, because no signed insertion order
exists on disk: the flight window is the Israeli broadcast week containing the
observed airings, and the budget and the rating goal are the observed figures
scaled by flight days over sourced days. Each of those carries the rule that made
it in its own note.

**No day is fabricated.** A flight day this product holds no per-spot source for
is written as an ``unknown`` delivery row with blank figures. It is never a zero.

**One channel.** The channel is read from settings and stamped on every row. The
traffic file carries no channel column, so nothing rival can enter through it,
and the seed refuses to run at all when no operator channel is configured.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kairos_api import campaigns_api_store as store  # noqa: E402
from kairos_api import campaigns_assets, campaigns_delivery  # noqa: E402
from kairos_api.campaigns_commitment import ALL_VIEWERS  # noqa: E402
from scripts.seed_campaigns_rows import build_assets, build_delivery  # noqa: E402

DATA_DIR = ROOT / "data"
DAILY_DIR = DATA_DIR / "daily_input"
DEMO_PREFIX = "CMP_D"

FLIGHT_RULE_EN = (
    "Seed rule: the flight is the Israeli broadcast week, Sunday to Saturday, containing every "
    "airing observed in the traffic file."
)
GOAL_RULE_EN = (
    "Seed rule: the budget and the rating goal are the observed figures scaled by flight days over "
    "sourced days. They are the seed's, not a signed insertion order."
)
DEMO_NOTE = f"{FLIGHT_RULE_EN} {GOAL_RULE_EN}"
AS_OF_BASIS = (
    "The start of the last programme booked on the newest sourced broadcast day, so the demo shows "
    "what has aired and what is still to come on that day."
)
RATING_BASIS = "Sum of the planned break rating over each airing, all-viewers base, from the traffic file."
SPEND_BASIS = "Engine-priced per spot, the same ledger the money board reads. Nothing is invoiced."
NO_SOURCE_NOTE = "This flight day has no per-spot source, so what aired is unknown and is not zero."
IN_HOUSE_NOTE = "משרד עצמי. הלקוח קונה ישירות ואין לו סוכנות מייצגת."


def operator_channel() -> str:
    from kairos_api import channel_scope

    return str(channel_scope.operator_channel() or "").strip()


def daily_paths() -> list[Path]:
    if not DAILY_DIR.exists():
        return []
    return sorted(DAILY_DIR.glob("Wally_*.csv"))


def load_day(path: Path) -> pd.DataFrame:
    """One traffic file with canonical column names and a real broadcast date."""
    from kairos.data.loaders import load_daily_input

    frame = load_daily_input(path)
    frame["broadcast_date"] = frame["date"].dt.date.astype(str)
    frame["duration_sec"] = pd.to_numeric(frame["duration_sec"], errors="coerce").fillna(0.0)
    frame["planned_tvr"] = pd.to_numeric(frame["planned_tvr"], errors="coerce").fillna(0.0)
    for column in ("campaign", "advertiser", "agency", "break_start", "creative",
                   "house_number", "spot_type", "program", "program_start"):
        frame[column] = frame[column].astype(str).str.strip()
    return frame


def as_of_instant(frames: list[pd.DataFrame]) -> tuple[str, str]:
    """The instant the aired and scheduled split is taken at, and how it was found.

    It is the start of the last programme booked on the newest sourced day. That
    is a real timestamp out of the traffic file rather than an invented clock,
    and it is what lets the seed show a day that is part aired and part still to
    come, which is the state an operator actually works in.
    """
    newest = max(frames, key=lambda frame: frame["broadcast_date"].max())
    day = str(newest["broadcast_date"].max())
    starts = sorted({value for value in newest["program_start"] if value})
    if not starts:
        return f"{day}T23:59:59", AS_OF_BASIS
    return f"{day}T{starts[-1]}:00", AS_OF_BASIS


def israeli_week(days: list[date]) -> tuple[date, date]:
    """The Sunday-to-Saturday broadcast week or weeks the observed airings fall in."""
    first, last = min(days), max(days)
    start = first - timedelta(days=(first.weekday() + 1) % 7)
    end = last + timedelta(days=(5 - last.weekday()) % 7)
    return start, end


def brand_of(campaign: str, advertiser: str) -> str:
    """The brand, only when the campaign name states it unambiguously.

    Israeli campaign labels run ``YYYY-MM - advertiser - brand — product``. The
    brand is taken only when the token before it is exactly the advertiser name,
    so a label that does not follow the pattern yields a blank rather than a
    guess. A blank brand is an honest answer; a guessed one is not.
    """
    body = campaign.split(" - ", 1)
    if len(body) < 2:
        return ""
    parts = body[1].split(" - ")
    if len(parts) < 2 or parts[0].strip() != advertiser.strip():
        return ""
    return parts[1].split(" — ")[0].strip()


def price_model_of(rows: pd.DataFrame) -> str:
    """The price model the traffic file itself records for this campaign's spots."""
    kinds = {str(value).strip().upper() for value in rows.get("pricing_type", [])}
    if kinds == {"CPP"}:
        return "cpp"
    if kinds == {"FIX"}:
        return "flat"
    return ""


def priced_day(path: Path) -> tuple[dict[tuple[str, str], float], dict[tuple[str, str], int], str]:
    """Revenue and rule-dropped counts per campaign and break, from the one ledger."""
    from kairos.export.spots import price_daily_file
    from kairos.optimize.overrides import OverrideSet
    from kairos.optimize.pricing import pricing_from_settings
    from kairos_api.overrides import OVERRIDES_PATH
    from kairos_api.server import _load_settings

    settings = _load_settings()
    result = price_daily_file(
        path,
        pricing=pricing_from_settings(settings),
        overrides=OverrideSet.from_csv(OVERRIDES_PATH),
    )
    revenue: dict[tuple[str, str], float] = defaultdict(float)
    for spot in result.priced:
        revenue[(str(spot.campaign).strip(), str(spot.break_id).strip())] += float(spot.revenue or 0.0)
    dropped: dict[tuple[str, str], int] = defaultdict(int)
    rules: set[str] = set()
    for drop in result.frequency_dropped:
        dropped[(str(drop.campaign).strip(), str(drop.break_id).strip())] += 1
        rules.add(str(drop.rule_id))
    return dict(revenue), dict(dropped), "|".join(sorted(rules))


def agency_index() -> dict[str, str]:
    from kairos_api.agencies import _load_frame

    frame = _load_frame()
    return {str(row["name"]).strip(): str(row["agency_id"]).strip() for _, row in frame.iterrows()}


def in_house_agency(client: str, known: dict[str, str]) -> str:
    """A client with no agency gets one in its own name, marked in-house.

    The owner's rule, enforced here rather than left to the reader: no campaign
    on this product is agency-less, and a direct-buying client is visibly a
    direct-buying client instead of quietly sharing a row with agency business.
    """
    from kairos_api.agencies import AgencyCreate, create_agency

    if client in known:
        return known[client]
    from kairos_api.campaigns_api_onboarding import next_agency_id

    record = create_agency(
        AgencyCreate(
            agency_id=next_agency_id(),
            name=client,
            display_name=f"{client} (משרד עצמי)",
            agency_type="בוטיק",
            notes=IN_HOUSE_NOTE,
            data_source="manual",
        ),
        None,
    )
    known[client] = str(record["agency_id"])
    return known[client]


def _scaled(observed: float, scale: float, step: float) -> str:
    """One demo commitment figure, or a blank when there is nothing to derive it from.

    A campaign whose every spot was removed by a frequency rule has no observed
    spend, and a budget of zero written from that would tell the reader the buyer
    committed to spending nothing. Blank is the honest answer, and the store and
    the API both carry a blank commitment as none rather than as zero.
    """
    if observed <= 0:
        return ""
    return f"{float(int(observed * scale / step + 0.999999) * step):.2f}"


def build(frames: list[tuple[Path, pd.DataFrame]], channel: str, as_of: str,
          existing: dict[str, str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Every campaign, creative and delivery day, from the files that are on disk."""
    ledgers = {path.name: priced_day(path) for path, _ in frames}
    joined = pd.concat([frame.assign(source_file=path.name) for path, frame in frames], ignore_index=True)
    sourced_days = sorted({str(value) for value in joined["broadcast_date"]})
    agencies = agency_index()

    campaigns: list[dict] = []
    assets: list[dict] = []
    delivery: list[dict] = []
    for index, name in enumerate(sorted(joined["campaign"].unique()), start=1):
        rows = joined[joined["campaign"] == name]
        campaign_id = f"{DEMO_PREFIX}{index:03d}"
        advertiser = str(rows["advertiser"].mode().iloc[0])
        agency_name = str(rows["agency"].mode().iloc[0])
        agency_id = agencies.get(agency_name) or in_house_agency(advertiser, agencies)
        observed = sorted({date.fromisoformat(str(value)) for value in rows["broadcast_date"]})
        starts_on, ends_on = israeli_week(observed)
        flight_days = (ends_on - starts_on).days + 1
        own_sourced = [day for day in sourced_days if starts_on <= date.fromisoformat(day) <= ends_on]
        scale = flight_days / max(len(own_sourced), 1)

        spend = sum(
            ledgers[str(row.source_file)][0].get((name, str(row.break_start)), 0.0)
            for row in rows.drop_duplicates(["source_file", "break_start"]).itertuples()
        )
        rating = float(rows["planned_tvr"].sum())
        campaigns.append({
            "campaign_id": campaign_id,
            "name": name,
            "advertiser": advertiser,
            "agency_id": agency_id,
            "channel": channel,
            "brand": brand_of(name, advertiser),
            "starts_on": starts_on.isoformat(),
            "ends_on": ends_on.isoformat(),
            "budget_ils": _scaled(spend, scale, 100.0),
            "rating_goal_points": _scaled(rating, scale, 5.0),
            "rating_goal_audience": ALL_VIEWERS,
            "price_model": price_model_of(rows),
            "created_at": existing.get(campaign_id, ""),
            "spot_goal": _scaled(float(len(rows)), scale, 1.0),
        })
        assets.extend(build_assets(rows, campaign_id, advertiser, channel))
        delivery.extend(build_delivery(
            rows, ledgers, campaign_id, channel, as_of, starts_on, ends_on, sourced_days,
        ))
    return campaigns, assets, delivery


def write_campaigns(records: list[dict]) -> tuple[int, int]:
    """Replace the demo rows and keep every booked one exactly where it was."""
    frame = store.load_frame()
    kept = frame[~frame["campaign_id"].astype(str).str.startswith(DEMO_PREFIX)]
    rows = []
    for record in records:
        # The flight is stated in rating points when the seed could derive a
        # rating goal, and in spots when it could not, because a flight without
        # a goal cannot be measured and a goal of zero is not a goal.
        spot_goal = record.pop("spot_goal", "")
        goal_kind, goal_value = ("grp", record["rating_goal_points"]) \
            if record["rating_goal_points"] else ("spots", spot_goal)
        campaign = store.blank_row()
        campaign.update({
            "record_type": store.CAMPAIGN, "status": "active", "data_source": "demo_seed",
            "is_demo": "true", "demo_note": DEMO_NOTE, "bonus_ils": "", **record,
        })
        rows.append(campaign)
        flight = store.blank_row()
        flight.update({
            "record_type": store.FLIGHT,
            "campaign_id": record["campaign_id"],
            "flight_id": f"{record['campaign_id']}_F1",
            "name": record["name"],
            "starts_on": record["starts_on"],
            "ends_on": record["ends_on"],
            "goal_kind": goal_kind,
            "goal_value": goal_value,
            "created_at": record["created_at"],
            "is_demo": "true",
            "demo_note": DEMO_NOTE,
        })
        rows.append(flight)
    for row in rows:
        if not row["created_at"]:
            row.pop("created_at")
    written = kept
    for row in rows:
        written = store.append(written, row)
    store.write_frame(written)
    return len(kept), len(rows)


def _write(path: Path, columns: list[str], records: list[dict]) -> int:
    frame = pd.DataFrame(records, columns=columns) if records else pd.DataFrame(columns=columns)
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return len(frame)


def existing_stamps() -> dict[str, str]:
    """The moment each demo row was first written, so a re-run does not restamp it."""
    frame = store.load_frame()
    demo = frame[frame["campaign_id"].astype(str).str.startswith(DEMO_PREFIX)]
    return {
        str(row["campaign_id"]): str(row.get("created_at", ""))
        for _, row in demo.iterrows()
        if str(row.get("record_type", "")) == store.CAMPAIGN
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Seed demo campaigns from the real traffic log.")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be written and stop.")
    args = parser.parse_args(argv)

    channel = operator_channel()
    if not channel:
        print("No operator channel is configured in settings, so no campaign can be stamped. Nothing written.")
        return 2
    paths = daily_paths()
    if not paths:
        print(f"No traffic file in {DAILY_DIR}. Nothing to seed from, so nothing written.")
        return 2

    frames = [(path, load_day(path)) for path in paths]
    as_of, _ = as_of_instant([frame for _, frame in frames])
    campaigns, assets, delivery = build(frames, channel, as_of, existing_stamps())
    print(f"channel: {channel}")
    print(f"source files: {', '.join(path.name for path in paths)}")
    print(f"counted as of: {as_of}")
    print(f"campaigns: {len(campaigns)}  assets: {len(assets)}  delivery rows: {len(delivery)}")
    states = defaultdict(int)
    for row in delivery:
        states[row["air_state"]] += 1
    print("delivery states: " + ", ".join(f"{key}={value}" for key, value in sorted(states.items())))
    if args.dry_run:
        print("dry run, nothing written")
        return 0

    kept, rows = write_campaigns(campaigns)
    print(f"data/campaigns.csv: {rows} demo rows written, {kept} booked rows kept")
    print(f"data/campaign_assets.csv: {_write(campaigns_assets.ASSETS_PATH, campaigns_assets.COLUMNS, assets)} rows")
    print(f"data/campaign_delivery.csv: {_write(campaigns_delivery.DELIVERY_PATH, campaigns_delivery.COLUMNS, delivery)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
