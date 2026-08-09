"""The advertiser rules store must bind to the advertisers that actually air.

THE DEFECT THIS CLOSES. ``data/advertiser_rules.csv`` held 45 rows keyed
``ADV_01``..``ADV_45``, each carrying a commercial rule, and its ``name``,
``display_name`` and ``aliases`` columns were empty in all 45. The resolver in
:mod:`kairos.optimize.advertiser_rules_identity` binds a spot to a rule BY NAME,
so the two key spaces did not intersect at all:

    advertisers airing on the shipped daily file : 41, all real Israeli names
    rules rows                                   : 45
    names any rules row claimed                  : 0
    MATCHED                                      : 0

Every advertiser premium in the product was therefore dead. Not "usually 1.0" -
dead: the layer bound to nothing, so no premium anyone set could ever price a
spot.

THE TWO THINGS THIS FILE HOLDS THE STORE TO.

  1. BINDING. Every advertiser on the shipped daily file resolves to a rules
     row, and every rules row either names an advertiser the operator's own
     traffic file was observed to carry, or is honestly unbound. No row names a
     company that is not in the observed space, because an invented advertiser
     is a fabricated counterparty.

  2. PROVENANCE. The premiums in that store are SYNTHETIC seed values. Attaching
     one to a real bank without saying so fabricates a commercial term about a
     real company. So every row declares ``data_source=synthetic`` and carries a
     Hebrew note saying the commercial detail is a seed and not something agreed
     with the advertiser, exactly the way ``data/agencies.csv`` already does it.

Both bars fail on the pre-fix file, and ``test_a_row_returned_to_the_old_shape_
fails_the_bar`` proves it by rebuilding an unbound row and running the same
check over it, so the guard cannot quietly stop guarding.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from kairos.export.spots import price_daily_spots
from kairos.data.loaders import load_daily_input
from kairos.optimize.advertiser_rules import AdvertiserRuleEngine
from kairos.optimize.advertiser_rules_identity import (
    IDENTITY_COLUMNS,
    load_advertiser_names,
    normalize_name,
    read_csv_rows,
    split_aliases,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RULES_CSV = DATA / "advertiser_rules.csv"
NAMES_CSV = DATA / "advertiser_names.csv"
DAILY_CSV = DATA / "daily_input" / "Wally_Prime_Reshet_Example_2025-04-27.csv"

# The measured baseline of the shipped daily file, unchanged by the binding
# because every bound row carries the identity premium 1.0 and allows ANY.
BASELINE_GROSS = 699450.0
BASELINE_SPOTS = 119
OBSERVED_ADVERTISERS = 41

SYNTHETIC = "synthetic"

# The word that has to appear in a row's Hebrew note for the note to say what it
# is for. "סינתטי" is the same stem data/agencies.csv uses on its seeded rows.
SYNTHETIC_STEM = "סינתטי"

requires_real_data = pytest.mark.skipif(
    not (RULES_CSV.exists() and NAMES_CSV.exists() and DAILY_CSV.exists()),
    reason="the shipped advertiser stores and daily file are required",
)


def _observed_space() -> set[str]:
    """Every advertiser name the operator's own data was observed to carry."""
    return set(load_advertiser_names(NAMES_CSV))


def _row_tokens(row: dict[str, str]) -> list[str]:
    """Every name a rules row claims, in the resolver's comparison form.

    The same four places the resolver reads: the key itself, plus the three
    identity columns. A row claiming a name in any of them is bound by it.
    """
    tokens = [str(row.get("advertiser_id", "") or "")]
    for column in IDENTITY_COLUMNS:
        value = row.get(column, "")
        if column == "aliases":
            tokens.extend(split_aliases(value))
        else:
            tokens.append(str(value or ""))
    return [normalized for normalized in map(normalize_name, tokens) if normalized]


def unbound_and_invented(
    rows: list[dict[str, str]], observed: set[str]
) -> tuple[list[str], list[str]]:
    """Split rows into the honestly unbound and the ones naming a stranger.

    Kept as a plain function rather than inlined into a test so the proof case
    below can run the identical check over a store it builds itself. A row is
    UNBOUND when none of its tokens is an observed advertiser, which is a legal
    state and must be visible. A row is INVENTED when it claims a name the
    observed space does not have, which is a fabricated counterparty and never
    legal.
    """
    unbound: list[str] = []
    invented: list[str] = []
    for row in rows:
        key = str(row.get("advertiser_id", "") or "")
        if any(token in observed for token in _row_tokens(row)):
            continue
        unbound.append(key)
        # A blank identity claims nobody; a filled one that misses the observed
        # space is a name from somewhere the data cannot account for.
        claimed = [str(row.get(column, "") or "").strip() for column in IDENTITY_COLUMNS]
        if any(claimed) or (key and not key.startswith("ADV_")):
            invented.append(key)
    return unbound, invented


# --- bar 1: the store binds ---------------------------------------------------


@requires_real_data
def test_every_advertiser_on_the_daily_file_binds_to_a_rules_row():
    """The headline. This was 0 of 41 before the fix and is 41 of 41 after."""
    daily = load_daily_input(DAILY_CSV)
    airing = sorted({str(value).strip() for value in daily["advertiser"] if str(value).strip()})
    assert len(airing) == OBSERVED_ADVERTISERS

    engine = AdvertiserRuleEngine.from_files()
    missed = [name for name in airing if engine.key_for(name) not in engine.baselines]
    assert missed == [], (
        f"{len(missed)} of {len(airing)} advertisers airing on {DAILY_CSV.name} resolve to no "
        "rules row, so their premium cannot price a spot:\n  " + "\n  ".join(missed)
    )


@requires_real_data
def test_no_two_rules_rows_claim_the_same_advertiser():
    """Two rows on one name would make the price depend on file order."""
    engine = AdvertiserRuleEngine.from_files()
    assert engine.names.collisions == []


@requires_real_data
def test_every_rules_row_names_an_observed_advertiser_or_is_honestly_unbound():
    rows = read_csv_rows(RULES_CSV)
    unbound, invented = unbound_and_invented(rows, _observed_space())
    assert invented == [], (
        "a rules row names an advertiser that appears nowhere in the operator's observed "
        f"data, which is a fabricated counterparty: {invented}"
    )
    # Unbound rows are legal and expected: there are more seed rows than there
    # are observed advertisers. What is not legal is an unbound row that does
    # not say it is unbound, so each one has to carry its reason.
    for key in unbound:
        row = next(r for r in rows if str(r.get("advertiser_id", "")) == key)
        assert str(row.get("notes", "") or "").strip(), (
            f"rules row {key} is bound to no advertiser and its notes column is empty, so "
            "nothing on the row says it prices nothing"
        )


# --- bar 2: the store says its terms are synthetic ----------------------------


@requires_real_data
def test_every_row_declares_its_commercial_terms_synthetic():
    rows = read_csv_rows(RULES_CSV)
    assert rows
    assert "data_source" in rows[0], (
        "data/advertiser_rules.csv lost its data_source column. The CRUD in "
        "kairos_api/advertisers.py writes frame[COLUMNS] and COLUMNS does not list "
        "data_source, so one advertiser edit through the API erases the provenance of all "
        "45 rows. Add data_source to that COLUMNS list; do not drop this assertion."
    )
    wrong = [
        str(row.get("advertiser_id", ""))
        for row in rows
        if str(row.get("data_source", "") or "").strip().lower() != SYNTHETIC
    ]
    assert wrong == [], (
        "these rules rows do not declare their premium synthetic, so a surface could show "
        f"an invented commercial term as a negotiated one: {wrong}"
    )


@requires_real_data
def test_every_row_carries_a_readable_note_about_its_seed_terms():
    """The durable half of the provenance.

    ``notes`` is in the CRUD's COLUMNS list and ``data_source`` is not, so the
    note is what survives an operator edit today. It has to stand on its own.
    """
    missing = [
        str(row.get("advertiser_id", ""))
        for row in read_csv_rows(RULES_CSV)
        if SYNTHETIC_STEM not in str(row.get("notes", "") or "")
    ]
    assert missing == [], (
        f"these rules rows carry no note saying their commercial terms are seed values: {missing}"
    )


@requires_real_data
def test_the_name_a_row_is_bound_to_came_from_the_observed_space_not_this_file():
    """Nothing here is a source of advertiser names; it only points at them.

    Every name the rules store binds has to already exist in
    data/advertiser_names.csv with source=observed, so the store can never
    become the origin of a company nobody saw air.
    """
    names = load_advertiser_names(NAMES_CSV)
    assert names
    assert all(record.source == "observed" for record in names.values())
    bound = {
        token
        for row in read_csv_rows(RULES_CSV)
        for token in _row_tokens(row)
        if token in set(names)
    }
    assert len(bound) == OBSERVED_ADVERTISERS


# --- bar 3: binding did not move the money ------------------------------------


@requires_real_data
def test_binding_the_names_did_not_move_the_daily_money():
    """Binding changes WHICH rows are consulted, so it can change money.

    Here it does not, and that is a property of the data rather than an
    accident: every row that took a name carries default_premium 1.0 and allows
    ANY position and genre, so consulting it is arithmetically the same as the
    unknown-advertiser fallback it replaced. The one seed row with real teeth
    (a 1.27 premium and a first/last restriction) was deliberately left
    unbound, which is why the total below is unchanged to the shekel.
    """
    result = price_daily_spots(load_daily_input(DAILY_CSV))
    assert result.total_revenue == pytest.approx(BASELINE_GROSS)
    assert len(result.priced) == BASELINE_SPOTS
    assert len(result.dropped) == 0
    assert all(spot.premium == pytest.approx(1.0) for spot in result.priced)


@requires_real_data
def test_no_bound_row_carries_a_premium_or_a_restriction_nobody_agreed():
    """A synthetic term may sit in the store; it may not sit on a real company.

    This is the rule that decides which rows stay unbound. A seed premium other
    than 1.0, or an allow list narrower than ANY, is an invented commercial
    term, and binding it to a named advertiser would assert that term about that
    company. Such a row must stay unbound until a real one replaces it.
    """
    observed = _observed_space()
    offenders: list[str] = []
    for row in read_csv_rows(RULES_CSV):
        if not any(token in observed for token in _row_tokens(row)):
            continue
        premium = float(str(row.get("default_premium", "1") or "1"))
        positions = str(row.get("allow_positions", "ANY") or "ANY").strip().upper()
        genres = str(row.get("allow_genres", "ANY") or "ANY").strip().upper()
        prime_only = str(row.get("prime_time_only", "") or "").strip().lower() == "true"
        if premium != 1.0 or positions != "ANY" or genres != "ANY" or prime_only:
            offenders.append(
                f"{row.get('advertiser_id')} premium={premium} positions={positions} "
                f"genres={genres} prime_time_only={prime_only}"
            )
    assert offenders == [], (
        "a named advertiser is bound to a seed row carrying terms nobody agreed with them:\n  "
        + "\n  ".join(offenders)
    )


# --- the proof: this guard fails on the shape the store used to have ----------


@requires_real_data
def test_a_row_returned_to_the_old_shape_fails_the_bar(tmp_path):
    """Restore one row to the pre-fix shape and watch the same checks reject it.

    Two rebuilds, both from the shipped store so nothing here is a hand-written
    fixture: one row is re-keyed back to an ADV_## token with blank identity
    columns, which is exactly how all 45 rows looked before this work.
    """
    rows = read_csv_rows(RULES_CSV)
    observed = _observed_space()
    unbound_now, _ = unbound_and_invented(rows, observed)

    bound = [row for row in rows if any(t in observed for t in _row_tokens(row))]
    assert bound, (
        "no rules row is bound to any observed advertiser, which is the original defect "
        "itself, so there is nothing to revert and the proof below cannot run"
    )
    victim_key = str(bound[0].get("advertiser_id"))
    reverted = []
    for row in rows:
        row = dict(row)
        if str(row.get("advertiser_id")) == victim_key:
            row["advertiser_id"] = "ADV_99"
            for column in IDENTITY_COLUMNS:
                row[column] = ""
        reverted.append(row)

    unbound_after, _ = unbound_and_invented(reverted, observed)
    assert len(unbound_after) == len(unbound_now) + 1, (
        "reverting a bound row to the ADV_## shape did not register as unbound, so this "
        "guard would not have caught the original defect"
    )

    # And the headline check fails too: that advertiser no longer has a rule.
    store = tmp_path / "advertiser_rules.csv"
    with open(store, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(reverted)
    engine = AdvertiserRuleEngine.from_files(
        rules_path=store, conditions_path=tmp_path / "no_conditions.csv"
    )
    assert engine.key_for(victim_key) not in engine.baselines
