"""Who each advertiser is, which rules row it is bound to, and what it earned.

Split out of :mod:`kairos_api.advertisers` to keep that module under the project
line limit. It is a read: it joins the three stores that already exist and adds
no number of its own.

  * the observed name space, ``data/advertiser_names.csv``, which says who the
    advertisers are,
  * the rules store, ``data/advertiser_rules.csv``, whose ``name``,
    ``display_name`` and ``aliases`` columns say which row, if any, is bound to
    a given advertiser, and its conditions store beside it,
  * the daily ledger through :mod:`kairos_api.spot_ledger`, which says what that
    advertiser actually earned on the one daily file the product prices.

The honest state on the shipped data. All 41 observed advertisers resolve to a
named record and every one of them is bound to no rules row, because the 45 rows
in the rules store carry no name and the disposition of those rows is the
owner's decision. So each record reports its real money, its real premium of
1.0, and ``bound: false`` with the reason, rather than a blank or a guess. The
moment a name is written onto a rules row, that row's premium and conditions
appear here and start pricing that advertiser's spots, and nothing else changes.

**The money join folds exactly as resolution folds, and every ledger key lands
on exactly one record.** The ledger is keyed on the raw strings the daily file
carries, so joining it on the stored name by string equality meant an advertiser
could resolve perfectly and still report no money, which is the failure this
layer exists to end. :func:`ledger_attribution` groups every ledger key under the
record :func:`resolve_advertiser` says owns it, and the money record names the
spellings that fed it in ``ledger_keys``, so a merge of two spellings is visible
rather than silent. A key no name-space record claims still gets a record of its
own, whether a rules row names it or nothing does, so the sum of the records is
the ledger total and no money is attributable to nobody.
"""

from __future__ import annotations

from typing import Any

from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.advertiser_rules_identity import (
    DEFAULT_NAMES_PATH,
    AdvertiserName,
    ResolvedAdvertiser,
    _names_token_index,
    join_aliases,
    load_advertiser_names,
    normalize_name,
    resolve_advertiser,
)
from kairos_api import spot_ledger

# Stated once, so every unbound advertiser gives the same reason rather than an
# empty field the reader has to interpret.
UNBOUND_REASON = "No rules row is bound to this advertiser, so no rule prices its spots and the premium is 1.0."
BOUND_REASON = "Bound to a rules row, so that row's premium and conditions price this advertiser's spots."
UNRESOLVED_REASON = "This advertiser appears in the daily file and is in no name store, so it has no record yet."
NO_LEDGER_REASON = "This advertiser has no priced spot in the daily file being read."


def _names_path() -> Any:
    return DEFAULT_NAMES_PATH


def _rules_record(engine: AdvertiserRuleEngine, resolved: ResolvedAdvertiser, name: str) -> dict[str, Any]:
    """The rules half of one identity record, bound or honestly unbound."""
    advertiser_id = resolved.advertiser_id if resolved is not None else None
    baseline = engine.baselines.get(advertiser_id) if advertiser_id else None
    conditions = engine.conditions.get(advertiser_id, []) if advertiser_id else []
    return {
        "bound": advertiser_id is not None,
        "advertiser_id": advertiser_id,
        "baseline_premium": round(baseline.default_premium, 6) if baseline is not None else None,
        "effective_premium": round(engine.effective_premium(name), 6),
        "rule_count": len(conditions),
        "reason": BOUND_REASON if advertiser_id is not None else UNBOUND_REASON,
    }


def ledger_attribution(
    ledger: spot_ledger.LedgerRead,
    *,
    names: dict[str, AdvertiserName],
    engine: AdvertiserRuleEngine,
    tokens: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """Every ledger key, grouped under the record resolution says it belongs to.

    The ledger is keyed on the strings the daily file itself carries, unresolved
    and unfolded, because :mod:`kairos_api.spot_ledger` deliberately leaves
    resolution to the caller that knows the name space. This is that caller, so
    the join has to fold exactly as :func:`resolve_advertiser` folds. Joining on
    the raw string instead meant an advertiser could resolve perfectly and still
    report no money, because the daily file spelled it with a Hebrew gershayim
    and the name store with an ASCII quote, or with a doubled space, or in
    another case. That is precisely the failure the name space exists to end.

    A key that resolves to nothing is grouped under itself, which is the key
    :func:`_unresolved_record` reports it under.
    """
    grouped: dict[str, list[str]] = {}
    for key in ledger.by_advertiser:
        resolved = resolve_advertiser(
            key, names=names, rules_index=engine.names, names_tokens=tokens,
        )
        grouped.setdefault(resolved.key if resolved is not None else key, []).append(key)
    return grouped


def _money_record(
    ledger: spot_ledger.LedgerRead, name: str, keys: list[str] | tuple[str, ...] = (),
) -> dict[str, Any] | None:
    """The money half: the advertiser's own share of the one daily ledger.

    ``keys`` are the ledger's own spellings that resolve to this advertiser,
    from :func:`ledger_attribution`. Two spellings of one advertiser are one
    advertiser, so their money adds up, and ``ledger_keys`` names every spelling
    that fed the figure rather than leaving a merge invisible.
    """
    if not ledger.available:
        return None
    monies = [ledger.by_advertiser[key] for key in keys if key in ledger.by_advertiser]
    if not monies:
        return {
            "advertiser": name, "gross": 0.0, "net": 0.0, "spots": 0,
            "dropped_by_rule": 0, "dropped_by_frequency": 0, "ledger_keys": [],
            "basis": ledger.basis, "reason": NO_LEDGER_REASON,
        }
    return {
        "advertiser": name,
        "gross": round(sum(money.gross for money in monies), 2),
        "net": round(sum(money.net for money in monies), 2),
        "spots": sum(money.spots for money in monies),
        "dropped_by_rule": sum(money.dropped_by_rule for money in monies),
        "dropped_by_frequency": sum(money.dropped_by_frequency for money in monies),
        "ledger_keys": [money.advertiser for money in monies],
        "basis": ledger.basis,
        "reason": "",
    }


def _identity_record(
    record: AdvertiserName,
    *,
    engine: AdvertiserRuleEngine,
    ledger: spot_ledger.LedgerRead,
    names: dict[str, AdvertiserName],
    tokens: dict[str, str],
    attribution: dict[str, list[str]],
) -> dict[str, Any]:
    resolved = resolve_advertiser(
        record.name, names=names, rules_index=engine.names, names_tokens=tokens,
    )
    key = resolved.key if resolved is not None else record.name
    return {
        "advertiser": record.name,
        "display_name": record.display_name,
        "shown_name": record.shown_name,
        "aliases": list(record.aliases),
        "source": record.source,
        "first_seen": record.first_seen,
        "resolved": True,
        "matched_on": resolved.matched_on if resolved is not None else "",
        "rules": _rules_record(engine, resolved, record.name),
        "money": _money_record(ledger, record.name, attribution.get(key, [])),
    }


def _rules_only_record(
    resolved: ResolvedAdvertiser,
    *,
    engine: AdvertiserRuleEngine,
    ledger: spot_ledger.LedgerRead,
    keys: list[str],
) -> dict[str, Any]:
    """A daily-file advertiser named by a rules row and not by the name space.

    It is the workflow this store documents, arriving in the other order: the
    operator writes a name onto a rules row before the observed name space has
    caught up with the daily file that carries it. Such an advertiser resolves,
    so it was never in ``unresolved``, and it had no row in the name space, so it
    was in no record either. Measured before this was added: on an empty name
    space with one bound rules row, the report listed 0 advertisers, reported
    ``unresolved`` empty, and 5,000 of ledger gross was reachable from no record
    at all. Money that exists must be attributable to somebody.
    """
    return {
        "advertiser": resolved.name,
        "display_name": resolved.display_name,
        "shown_name": resolved.shown_name,
        "aliases": list(resolved.aliases),
        "source": resolved.source,
        "first_seen": "",
        "resolved": True,
        "matched_on": resolved.matched_on,
        "rules": _rules_record(engine, resolved, resolved.name),
        "money": _money_record(ledger, resolved.name, keys),
    }


def _unresolved_record(name: str, ledger: spot_ledger.LedgerRead) -> dict[str, Any]:
    """A daily-file advertiser the name space does not hold, stated as such."""
    return {
        "advertiser": name,
        "display_name": "",
        "shown_name": name,
        "aliases": [],
        "source": "",
        "first_seen": "",
        "resolved": False,
        "matched_on": "",
        "rules": {
            "bound": False, "advertiser_id": None, "baseline_premium": None,
            "effective_premium": 1.0, "rule_count": 0, "reason": UNRESOLVED_REASON,
        },
        "money": _money_record(ledger, name, [name]),
    }


def identity_report() -> dict[str, Any]:
    """Every advertiser as a named record, with its rules and its money.

    The list is the observed name space plus every advertiser the daily ledger
    carries that the name space does not hold, whether a rules row names it or
    nothing does. Both are listed and marked, never dropped, so the count on this
    payload is the real coverage rather than a filtered one, and every shekel in
    ``ledger`` is reachable from exactly one record.
    """
    names = load_advertiser_names(_names_path())
    tokens = _names_token_index(names)
    engine = AdvertiserRuleEngine.from_files()
    ledger = spot_ledger.read_ledger()
    attribution = ledger_attribution(ledger, names=names, engine=engine, tokens=tokens)

    records = [
        _identity_record(
            record, engine=engine, ledger=ledger, names=names, tokens=tokens,
            attribution=attribution,
        )
        for record in sorted(names.values(), key=lambda item: item.name)
    ]
    # Every ledger key that no name-space record claims still needs a record, or
    # its money is in the totals and in nobody's row. Two kinds arrive here: a
    # key nothing names, and a key only a rules row names.
    listed = {record.name for record in names.values()}
    unresolved: list[str] = []
    for key in sorted(set(attribution) - listed):
        # Resolve the daily file's own spelling, not the group key: a group only
        # a rules row names is keyed on the advertiser id, and resolving that
        # would name the record ADV_01 instead of the advertiser.
        resolved = resolve_advertiser(
            attribution[key][0], names=names, rules_index=engine.names, names_tokens=tokens,
        )
        if resolved is None:
            unresolved.append(key)
            records.append(_unresolved_record(key, ledger))
        else:
            records.append(_rules_only_record(
                resolved, engine=engine, ledger=ledger, keys=attribution[key],
            ))

    in_ledger = [record for record in records if record["money"] and record["money"]["spots"]]
    return {
        "advertisers": records,
        "count": len(records),
        "name_space_rows": len(names),
        "in_ledger": len(in_ledger),
        "resolved": sum(1 for record in records if record["resolved"]),
        "unresolved": sorted(unresolved),
        "bound_to_a_rules_row": sum(1 for record in records if record["rules"]["bound"]),
        "rules_rows": len(engine.baselines),
        "ledger": ledger.totals_dict(),
        "note": (
            "A record is named as soon as an advertiser is observed. It is bound only when a rules row "
            "carries its name or one of its aliases, and only a bound row prices its spots."
        ),
    }


def resolve_one(token: str) -> dict[str, Any]:
    """Resolve a single advertiser string, for a caller holding one name."""
    names = load_advertiser_names(_names_path())
    engine = AdvertiserRuleEngine.from_files()
    resolved = resolve_advertiser(token, names=names, rules_index=engine.names)
    if resolved is None:
        return {"query": token, "resolved": False, "reason": UNRESOLVED_REASON}
    ledger = spot_ledger.read_ledger()
    attribution = ledger_attribution(ledger, names=names, engine=engine)
    return {
        "query": token,
        "resolved": True,
        "advertiser": resolved.name,
        "display_name": resolved.display_name,
        "shown_name": resolved.shown_name,
        "aliases": list(resolved.aliases),
        "source": resolved.source,
        "matched_on": resolved.matched_on,
        "rules": _rules_record(engine, resolved, token),
        "money": _money_record(ledger, resolved.name, attribution.get(resolved.key, [])),
    }


def normalized_aliases(raw: object) -> str:
    """Serialize an aliases payload to the pipe-joined cell the store holds."""
    if isinstance(raw, (list, tuple)):
        return join_aliases(str(item) for item in raw)
    return join_aliases(str(raw or "").split("|"))


def name_is_taken(rows: list[dict[str, Any]], candidate: str, advertiser_id: str) -> bool:
    """Whether another rules row already claims this name or alias."""
    wanted = normalize_name(candidate)
    if not wanted:
        return False
    for row in rows:
        held = str(row.get("advertiser_id", "") or "")
        if held == advertiser_id:
            continue
        for column in ("advertiser_id", "name", "display_name"):
            if normalize_name(row.get(column, "")) == wanted:
                return True
        if wanted in {normalize_name(alias) for alias in str(row.get("aliases", "") or "").split("|")}:
            return True
    return False
