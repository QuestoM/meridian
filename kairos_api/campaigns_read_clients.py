"""The client tree: agency, then the clients under it, then their campaigns.

Split out of :mod:`kairos_api.campaigns_read` to keep that module under the
project line limit. It joins the four stores that already exist into the one
containment an account manager and an analyst both work in, and it adds no
number of its own:

  * ``data/agencies.csv`` for the agency and its commercial terms,
  * ``data/agency_advertisers.csv`` plus the newest daily file for which clients
    buy through which agency,
  * ``data/advertiser_names.csv`` through the identity join for who each client
    is, and
  * ``data/campaigns.csv`` for what has been booked under each of them.

Money comes from :mod:`kairos_api.campaigns_read_money`, which is the daily
priced ledger and the only money on this surface, so an agency total here is the
sum of its clients' totals and both are the same ledger the agency summary and
the spots export report.

Two states are reported rather than hidden. A client that has money and no
agency link is listed under ``unlinked`` instead of being dropped, so every
shekel in the totals is reachable from a row. A client with a campaign and no
priced spot is listed under ``clients_booked_without_spots``, with its money
reported as none and the reason, because a campaign booked for next month is not
the same thing as a client that delivered nothing.

Both of those groups are inside ``counts.clients``. A header that counts fewer
clients than the rows beneath it argues with itself, and the reader believes the
header, so the count is over every client this payload carries.
"""

from __future__ import annotations

from typing import Any, Optional

NO_MONEY_REASON = "This client has no priced spot in the daily file being read."
NO_MONEY_REASON_HE = "ללקוח הזה אין תשדיר מתומחר בקובץ היומי הנקרא."
UNLINKED_REASON = "This client has priced spots and no agency link. Link it on the agency record."
UNLINKED_REASON_HE = "ללקוח הזה יש תשדירים מתומחרים ואין שיוך לסוכנות. שייכו אותו בכרטיס הסוכנות."
BOOKED_ONLY_REASON = "This client has a booked campaign, no priced spot in the daily file being read, and no agency link."
BOOKED_ONLY_REASON_HE = "ללקוח הזה יש קמפיין מוזמן, אין תשדיר מתומחר בקובץ היומי הנקרא, ואין שיוך לסוכנות."
NO_AGENCY_MONEY_REASON = "No client of this agency has a priced spot in the daily file being read, so this agency has no total, which is not the same as a total of zero."
NO_AGENCY_MONEY_REASON_HE = "לאף לקוח של הסוכנות הזו אין תשדיר מתומחר בקובץ היומי הנקרא, ולכן אין לסוכנות סכום, וזה אינו סכום אפס."


def _agencies() -> list[dict[str, Any]]:
    from kairos_api.agencies import _load_frame, _row_to_record

    return [_row_to_record(row) for _, row in _load_frame().iterrows()]


def _identity_index() -> dict[str, dict[str, Any]]:
    """Every named client, keyed by the name the ledger and the stores share."""
    try:
        from kairos_api.advertisers_identity import identity_report

        report = identity_report()
    except Exception:  # noqa: BLE001 - a missing name space is an empty index, not a crash
        return {}
    return {str(record.get("advertiser", "")): record for record in report.get("advertisers", [])}


def _campaign_index() -> dict[str, list[dict[str, Any]]]:
    from kairos_api import campaigns_api_store as store

    grouped: dict[str, list[dict[str, Any]]] = {}
    for campaign in store.campaigns_with_flights(store.load_frame()):
        grouped.setdefault(campaign["advertiser"], []).append(campaign)
    return grouped


def _contacts(record: dict[str, Any]) -> list[dict[str, str]]:
    """The two contacts as records, so an empty one reads as an empty one."""
    contacts = []
    for prefix in ("contact", "contact2"):
        contacts.append({
            "name": str(record.get(f"{prefix}_name", "")),
            "role": str(record.get(f"{prefix}_role", "")),
            "phone": str(record.get(f"{prefix}_phone", "")),
            "email": str(record.get(f"{prefix}_email", "")),
        })
    return contacts


def _client_record(
    name: str,
    money_row: Optional[dict[str, Any]],
    identity: dict[str, dict[str, Any]],
    campaigns: dict[str, list[dict[str, Any]]],
    link_source: str,
) -> dict[str, Any]:
    record = identity.get(name)
    own_campaigns = campaigns.get(name, [])
    return {
        "advertiser": name,
        "shown_name": (record or {}).get("shown_name", name),
        "resolved": bool((record or {}).get("resolved", False)),
        "source": (record or {}).get("source", ""),
        "aliases": (record or {}).get("aliases", []),
        "bound_to_rules_row": bool(((record or {}).get("rules") or {}).get("bound", False)),
        # The row the SERVER resolved this client to, by name, display name and
        # every alias. It is sent because the surface used to re-derive it and
        # the two matchers drifted: the shipped rows carry their identity in
        # advertiser_id and leave the name column empty, the surface treated an
        # empty name as "binds nothing", found no row, and printed a data-
        # integrity warning about a client whose row was sitting right there.
        # An identity resolved twice is an identity that will disagree with
        # itself.
        "rules_row_id": ((record or {}).get("rules") or {}).get("advertiser_id"),
        "effective_premium": ((record or {}).get("rules") or {}).get("effective_premium"),
        "link_source": link_source,
        "gross": (money_row or {}).get("gross"),
        "net": (money_row or {}).get("net"),
        "rebates": (money_row or {}).get("rebates"),
        "spots": (money_row or {}).get("spots"),
        "dropped_by_frequency": (money_row or {}).get("dropped_by_frequency"),
        "dropped_rules": (money_row or {}).get("dropped_rules") or [],
        "rank": (money_row or {}).get("rank"),
        "share_of_gross": (money_row or {}).get("share_of_gross"),
        "money_reason_en": "" if money_row else NO_MONEY_REASON,
        "money_reason_he": "" if money_row else NO_MONEY_REASON_HE,
        "campaigns": own_campaigns,
        "campaign_count": len(own_campaigns),
        "observed_campaigns": [row["campaign"] for row in (money_row or {}).get("campaigns", [])],
    }


def _summed(clients: list[dict[str, Any]]) -> dict[str, Any]:
    """An agency's money: the sum of its priced clients, or none with the reason.

    A client with no priced spot reports none, and its record prints the reason.
    Coercing that none to zero and adding it up gave a new agency, correctly
    empty, a total of zero shekels, which is a figure nobody computed from
    anything. So the third state survives the sum: no priced client means no
    total, stated, and one priced client means the total of what is priced with
    the count of how many clients that was.
    """
    priced = [client for client in clients if client["gross"] is not None]
    if not priced:
        return {
            "gross": None,
            "net": None,
            "rebates": None,
            "spots": None,
            "clients_with_money": 0,
            "money_reason_en": NO_AGENCY_MONEY_REASON,
            "money_reason_he": NO_AGENCY_MONEY_REASON_HE,
        }
    gross = round(sum(client["gross"] for client in priced), 2)
    net = round(sum(client["net"] or 0.0 for client in priced), 2)
    return {
        "gross": gross,
        "net": net,
        "rebates": round(gross - net, 2),
        "spots": sum(client["spots"] or 0 for client in priced),
        "clients_with_money": len(priced),
        "money_reason_en": "",
        "money_reason_he": "",
    }


def client_tree() -> dict[str, Any]:
    """Agencies with their clients and campaigns, and every figure's basis."""
    from kairos_api.agency_conditions import links_for
    from kairos_api.campaigns_read_money import board

    money = board()
    money_index = {row["advertiser"]: row for row in money["advertisers"]}
    identity = _identity_index()
    campaigns = _campaign_index()

    claimed: set[str] = set()
    agencies: list[dict[str, Any]] = []
    for record in _agencies():
        links = links_for(record["agency_id"])
        clients = []
        for name in links["effective"]:
            claimed.add(name)
            clients.append(_client_record(
                name,
                money_index.get(name),
                identity,
                campaigns,
                "manual" if name in links["manual"] else "observed",
            ))
        clients.sort(key=lambda client: (-(client["gross"] or 0.0), client["advertiser"]))
        agencies.append({
            "agency_id": record["agency_id"],
            "name": record["name"],
            "display_name": record["display_name"],
            "status": record["status"],
            "agency_type": record["agency_type"],
            "data_source": record["data_source"],
            "terms": {
                "payment_terms_days": record["payment_terms_days"],
                "rebate_percent": record["rebate_percent"],
                "commission_percent": record["commission_percent"],
                "credit_limit_ils": record["credit_limit_ils"],
                "vat_id": record["vat_id"],
            },
            "contacts": _contacts(record),
            "clients": clients,
            "client_count": len(clients),
            "campaign_count": sum(client["campaign_count"] for client in clients),
            **_summed(clients),
        })

    unlinked = [
        _client_record(name, money_index[name], identity, campaigns, "")
        for name in sorted(money_index)
        if name not in claimed
    ]
    for client in unlinked:
        client["money_reason_en"] = UNLINKED_REASON
        client["money_reason_he"] = UNLINKED_REASON_HE
    agencies.sort(key=lambda agency: (-(agency["gross"] or 0.0), agency["name"]))
    for index, agency in enumerate(agencies, start=1):
        agency["rank"] = index

    booked_only = [
        _client_record(name, None, identity, campaigns, "")
        for name in sorted(set(campaigns) - set(money_index) - claimed)
    ]
    for client in booked_only:
        client["money_reason_en"] = BOOKED_ONLY_REASON
        client["money_reason_he"] = BOOKED_ONLY_REASON_HE
    return {
        "available": money["available"],
        "reason": money["reason"],
        "basis": money["basis"],
        "totals": money["totals"],
        "agencies": agencies,
        "unlinked": unlinked,
        "clients_booked_without_spots": booked_only,
        "counts": {
            "agencies": len(agencies),
            "clients": len(claimed) + len(unlinked) + len(booked_only),
            "clients_with_money": len(money_index),
            "campaigns": sum(len(rows) for rows in campaigns.values()),
        },
    }
