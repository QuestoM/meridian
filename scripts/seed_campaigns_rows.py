"""The three row builders the campaign seed writes, kept out of it for the line limit.

Nothing here decides anything. :mod:`scripts.seed_campaigns` resolves the
channel, the flight and the counted-as-of instant; this module turns one
campaign's traffic rows into the creative rows and the delivery rows that go on
disk. It is split out only so both files stay under the project line limit, and
it is imported by exactly one caller.

The one rule it enforces on its own is the honest blank. A flight day with no
per-spot source produces a row in the ``unknown`` state whose figure cells are
empty strings, never zeros, so no reader and no sum can mistake a gap for a
delivery of nothing.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from kairos_api import campaigns_delivery

RATING_BASIS = "Sum of the planned break rating over each airing, all-viewers base, from the traffic file."
SPEND_BASIS = "Engine-priced per spot, the same ledger the money board reads. Nothing is invoiced."
NO_SOURCE_NOTE = "This flight day has no per-spot source, so what aired is unknown and is not zero."
AS_OF_BASIS = (
    "The start of the last programme booked on the newest sourced broadcast day, so the demo shows "
    "what has aired and what is still to come on that day."
)


def build_assets(rows: pd.DataFrame, campaign_id: str, advertiser: str, channel: str) -> list[dict]:
    """One creative per house number, real where the log says so and unknown elsewhere."""
    records = []
    for order, (house, group) in enumerate(sorted(rows.groupby("house_number")), start=1):
        spot_type = str(group["spot_type"].mode().iloc[0])
        records.append({
            "asset_id": f"{campaign_id}_A{order:02d}",
            "campaign_id": campaign_id,
            "advertiser": advertiser,
            "channel": channel,
            "house_number": str(house),
            "version_name": str(group["creative"].mode().iloc[0]),
            "spot_type": spot_type,
            "length_class": "sponsorship" if spot_type == "חסות" else "commercial",
            "duration_seconds": f"{float(group['duration_sec'].max()):.0f}",
            "media_url": "",
            "media_state": "unknown",
            "video_format": "",
            "aspect_ratio": "",
            "loudness_lufs": "",
            "loudness_standard": "",
            "clearance_verdict": "unknown",
            "clearance_authority": "",
            "clearance_checked_at": "",
            "first_observed_on": str(group["broadcast_date"].min()),
            "last_observed_on": str(group["broadcast_date"].max()),
            "airings_observed": str(len(group)),
            "identity_source": "traffic_log",
            "source_file": "|".join(sorted({str(value) for value in group["source_file"]})),
            "is_demo": "true",
            "notes": "",
        })
    return records


def build_delivery(rows: pd.DataFrame, ledgers: dict, campaign_id: str, channel: str, as_of: str,
              starts_on: date, ends_on: date, sourced_days: list[str]) -> list[dict]:
    """One row per broadcast day of the flight, in one of the three honest states."""
    cutoff = as_of.split("T", 1)[1]
    campaign = str(rows["campaign"].iloc[0])
    records = []
    day = starts_on
    while day <= ends_on:
        key = day.isoformat()
        if key not in sourced_days:
            records.append(_blank_day(campaign_id, key, channel, as_of))
            day += timedelta(days=1)
            continue
        same = rows[rows["broadcast_date"] == key]
        for state in (campaigns_delivery.AIRED, campaigns_delivery.SCHEDULED):
            part = same[same["break_start"] < cutoff] if state == campaigns_delivery.AIRED \
                else same[same["break_start"] >= cutoff]
            if part.empty:
                continue
            records.append(_sourced_day(part, ledgers, campaign, campaign_id, key, channel, as_of, state))
        day += timedelta(days=1)
    return records


def _blank_day(campaign_id: str, key: str, channel: str, as_of: str) -> dict:
    return {
        "campaign_id": campaign_id,
        "broadcast_date": key,
        "air_state": campaigns_delivery.UNKNOWN,
        "channel": channel,
        "spots": "", "seconds": "", "rating_points_planned": "", "spend_ils": "",
        "spots_dropped_by_rule": "", "dropped_rule_id": "",
        "figures_basis": "", "source_file": "",
        "counted_as_of": as_of, "counted_as_of_basis": AS_OF_BASIS,
        "is_demo": "true", "note": NO_SOURCE_NOTE,
    }


def _sourced_day(part: pd.DataFrame, ledgers: dict, campaign: str, campaign_id: str, key: str,
                 channel: str, as_of: str, state: str) -> dict:
    files = sorted({str(value) for value in part["source_file"]})
    breaks = part.drop_duplicates(["source_file", "break_start"])
    spend = sum(ledgers[str(row.source_file)][0].get((campaign, str(row.break_start)), 0.0)
                for row in breaks.itertuples())
    dropped = sum(ledgers[str(row.source_file)][1].get((campaign, str(row.break_start)), 0)
                  for row in breaks.itertuples())
    # A rule id is written only when the rule actually removed a spot from this
    # day. Naming a rule beside a count of zero says a rule bit when none did.
    rules = {ledgers[name][2] for name in files if ledgers[name][2]} if dropped else set()
    return {
        "campaign_id": campaign_id,
        "broadcast_date": key,
        "air_state": state,
        "channel": channel,
        "spots": str(len(part)),
        "seconds": f"{float(part['duration_sec'].sum()):.0f}",
        "rating_points_planned": f"{float(part['planned_tvr'].sum()):.4f}",
        "spend_ils": f"{spend:.2f}",
        "spots_dropped_by_rule": str(dropped),
        "dropped_rule_id": "|".join(sorted(rules)),
        "figures_basis": f"{RATING_BASIS} {SPEND_BASIS}",
        "source_file": "|".join(files),
        "counted_as_of": as_of,
        "counted_as_of_basis": AS_OF_BASIS,
        "is_demo": "true",
        "note": "",
    }
