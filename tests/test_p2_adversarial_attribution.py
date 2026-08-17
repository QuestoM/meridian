"""Adversarial verification of the money identity, by a second pair of eyes.

The committed suite proves the four-level attribution identity on clean fixtures:
cells sum to buckets, buckets to the attributed total, and the total to the scoped
headline a decision-maker reads. These tests attack that identity with the rows a
fixture would never contain - two rows claiming one programme, a row carrying no
programme id at all, money that runs negative, revenue carrying fractions below
one agora, and a clock hour past midnight - because an identity that holds only on
round numbers is not an identity.

Two of the attacks originally landed and were recorded as strict xfails. Both have
since been fixed and the markers came off, so what stands here now are the locks:
the same measurements, asserting the repaired behaviour. The history is kept in the
docstrings on purpose, because the number a defect used to produce is the only
cheap way to check that a fix actually moved it.

The decision moment, the contractual-standing dimension and the trade chain are
attacked next door in ``test_p2_adversarial_commitments.py``, which imports the
fixtures below rather than restating them.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos_api import channel_scope, day_compare
from kairos_api import day_compare_attribution as attribution
from kairos_api import day_proposal_store as store
from tests.test_day_proposals import (BASE_ROWS, CHANNEL, DAY, SETTINGS_BASIS,
                                      frame, ref_for)

CAPS = {"max_daily_ad_seconds": 1500.0, "max_ad_seconds_per_hour": 720.0,
        "max_breaks_per_hour": 4}


@pytest.fixture()
def scoped(tmp_path, monkeypatch):
    """A relocated store and a fixed operator channel, as the committed suite uses.

    Every test that creates a proposal needs this. Without it the store writes to
    ``data/day_proposals`` on the real tree, one directory away from the seeded
    demo versions - measured once, during this work, and cleaned up.
    """
    monkeypatch.setenv(store.PROPOSALS_DIR_ENV, str(tmp_path / "day_proposals"))
    monkeypatch.setattr(store, "_settings_basis", lambda: dict(SETTINGS_BASIS))
    monkeypatch.setattr(channel_scope, "operator_channel", lambda settings=None: CHANNEL)
    return tmp_path


def identity(baseline: pd.DataFrame, side: pd.DataFrame) -> dict:
    """The four levels the surface stakes its credibility on, in integer agorot.

    Computed the way the payload's own reader would: the money block and the
    attribution come from the same two functions the comparison calls, so this
    measures the shipped arithmetic rather than a re-implementation of it.
    """
    money_base, owned_base = day_compare._money(baseline)
    money_side, owned_side = day_compare._money(side)
    delta = day_compare._money_delta(money_side, money_base)
    attributed = attribution.attribute(owned_base, owned_side)
    return {
        "cells": sum(round(cell["revenue_delta"] * 100)
                     for cell in attributed.get("cells", [])),
        "buckets": sum(round(bucket["revenue_delta"] * 100)
                       for bucket in attributed.get("buckets", [])),
        "total": round(attributed["revenue_delta"] * 100),
        "headline": round(delta["revenue"] * 100),
        "exact": attributed["reconciliation"]["exact"],
        "residual": attributed["reconciliation"]["difference"],
        "attributed": attributed,
    }


def rows_with(mapper) -> pd.DataFrame:
    return frame([mapper(row) for row in BASE_ROWS])


# ------------------------------------------------- sub-agora money (was defect 1)

def test_sub_agora_fractions_must_not_break_the_identity_or_the_exact_flag(scoped):
    """Revenue below one agora per row: the identity the surface promises.

    The engine's ``predicted_revenue`` is a float product of rate, rating and
    seconds, so fractions finer than an agora are the normal case rather than a
    contrived one. Whichever rounding basis wins, the two figures must be the
    same figure, and ``exact`` must never assert an agreement that does not hold.
    """
    baseline = frame(BASE_ROWS)
    fractional = rows_with(lambda row: (row[0], row[1], row[2], row[3],
                                        row[4] + 0.001, row[5]))
    measured = identity(baseline, fractional)
    assert measured["total"] == measured["headline"], (
        f"attributed total {measured['total']} agorot disagrees with the headline "
        f"{measured['headline']} agorot while exact={measured['exact']}")
    assert measured["cells"] == measured["buckets"] == measured["total"]


def test_the_divergence_is_zero_at_any_row_count(scoped):
    """The historical defect's own measurements, inverted into the lock.

    Before the fix, 6 rows carrying +0.001 ILS diverged by 1 agora and 60 rows
    by 6, with exact:True beside both. One rounding basis (the headline's: sum
    once, round once) makes the divergence structurally zero; these are the
    same two measurements asserting the repaired identity.
    """
    # The identity must CLOSE against the headline, and the sub-agora drift
    # the per-row cells cannot carry must surface as an explicit unattributed
    # remainder with exact:False - declared, not denied. exact:True beside a
    # divergence was the defect; exact:False beside a printed one-agora
    # remainder is the honesty mechanism working.
    six = identity(frame(BASE_ROWS),
                   rows_with(lambda r: (r[0], r[1], r[2], r[3], r[4] + 0.001, r[5])))
    assert six["headline"] - six["total"] == 0
    assert six["exact"] is False

    base_many = [("20:00", "Drama", 3, 360.0, 10_000.0, f"X{index}")
                 for index in range(60)]
    side_many = [("20:00", "Drama", 3, 360.0, 10_000.001, f"X{index}")
                 for index in range(60)]
    many = identity(frame(base_many), frame(side_many))
    assert many["headline"] - many["total"] == 0
    assert many["exact"] is False


@pytest.mark.realdata
def test_a_real_plan_day_carries_no_sub_agora_drift_at_all():
    """Why exact:False on rounding dust costs the operator nothing in practice.

    The choice above - surfacing sub-agora drift as an unattributed remainder
    with exact:False - is only safe if it stays rare. If every ordinary
    comparison printed a one-agora remainder, readers would learn to ignore
    exact:False, and a genuinely unattributable row (the duplicate segment_id
    below, worth 500,000 ILS) would then look exactly like dust.

    So it was measured rather than assumed, over every day of the committed plan
    on the operator's own channel: 2,540 rows across 30 days, and the per-row
    agorot sum equalled the sum-once-round-once headline basis on 30 of 30. The
    remainder fires on crafted fractions and not on the engine's real output,
    which is what makes exact:False meaningful when it does appear.
    """
    from kairos_api import plan_version_store

    path = plan_version_store.plan_path()
    if not path.exists():
        pytest.skip("no committed weekly plan on this tree")
    plan = pd.read_csv(path)
    owned, _note = channel_scope.scope_frame(plan)
    if owned.empty or "date" not in owned.columns:
        pytest.skip("the committed plan carries no operator rows")

    drifted = []
    for day, part in owned.groupby(owned["date"].astype(str)):
        values = pd.to_numeric(part["predicted_revenue"], errors="coerce").fillna(0)
        sum_once = int(round(round(float(values.sum()), 2) * 100))
        per_row = sum(int(round(float(value) * 100)) for value in values)
        if sum_once != per_row:
            drifted.append((day, sum_once - per_row, len(part)))
    assert not drifted, (
        "the engine now emits revenue with sub-agora fractions, so ordinary "
        f"comparisons will start reporting exact:False: {drifted}")


# ------------------------------------------- regression locks that already hold

def test_two_rows_claiming_one_programme_surface_as_unattributed(scoped):
    """A duplicate segment_id: the money is real, the diff cannot key it.

    ``_row_index`` keeps one row per segment id, so the second row's money would
    vanish from the explanation. It does not vanish from the frame, and the
    reconciliation is measured against the frame, so the difference lands in the
    ``unattributed`` bucket with its reason and the identity still closes.
    """
    baseline = frame(BASE_ROWS)
    doubled = frame(list(BASE_ROWS)
                    + [("21:00", "Entertainment", 3, 360.0, 250_000.0, "S5")])
    measured = identity(baseline, doubled)
    assert measured["cells"] == measured["buckets"] == measured["total"] == \
        measured["headline"]
    assert measured["exact"] is False, "an unkeyable row must be reported, not hidden"
    assert measured["residual"] == 500_000.0
    unattributed = [cell for cell in measured["attributed"]["cells"]
                    if cell["bucket"] == attribution.UNATTRIBUTED]
    assert len(unattributed) == 1
    assert unattributed[0]["sentence_he"]
    assert unattributed[0]["revenue_delta"] == 500_000.0


def test_a_row_with_no_programme_id_is_reported_and_never_absorbed(scoped):
    """A blank segment_id: the row's money must not be folded into a neighbour."""
    baseline = frame(BASE_ROWS)
    blanked = frame(BASE_ROWS)
    blanked.loc[3, "segment_id"] = ""
    measured = identity(baseline, blanked)
    assert measured["cells"] == measured["buckets"] == measured["total"] == \
        measured["headline"]
    assert measured["exact"] is False
    assert measured["residual"] == 400_000.0
    buckets = {bucket["bucket"] for bucket in measured["attributed"]["buckets"]}
    assert attribution.UNATTRIBUTED in buckets


def test_neither_side_carrying_a_programme_id_refuses_rather_than_returning_zero(scoped):
    """No keyable row at all is a refusal with a reason, not an empty attribution."""
    blank_base = frame(BASE_ROWS)
    blank_side = frame(BASE_ROWS)
    blank_base["segment_id"] = ""
    blank_side["segment_id"] = ""
    attributed = attribution.attribute(blank_base, blank_side)
    assert attributed["available"] is False
    assert attributed["reason_he"]
    assert "cells" not in attributed or not attributed.get("cells")


def test_money_that_runs_negative_keeps_the_identity(scoped):
    """A day whose every row is a loss. Integer agorot must stay symmetric."""
    baseline = frame(BASE_ROWS)
    negative = rows_with(lambda row: (row[0], row[1], row[2], row[3],
                                      -abs(row[4]), row[5]))
    measured = identity(baseline, negative)
    assert measured["cells"] == measured["buckets"] == measured["total"] == \
        measured["headline"]
    assert measured["exact"] is True
    # The baseline day carries 1,250,000 ILS; every row flipping sign moves it to
    # -1,250,000, so the whole day swings by -2,500,000 ILS.
    assert measured["total"] == -250_000_000


def test_an_hour_past_midnight_is_classified_and_never_dropped(scoped):
    """Hour 24 and 25 are real in a broadcast day and must land in a daypart.

    The engine's broadcast day runs past midnight, so ``24:30`` is a legitimate
    clock cell. A row whose daypart could not be resolved would still carry money,
    so silently dropping it would break the identity.
    """
    baseline = frame([("24:30", "Overnight", 1, 120.0, 30_000.0, "S9")])
    side = frame([("24:30", "Overnight", 2, 240.0, 61_000.0, "S9")])
    measured = identity(baseline, side)
    assert measured["total"] == measured["headline"] == 31_000_00
    cell = measured["attributed"]["cells"][0]
    assert cell["daypart"] is not None, "hour 24 fell into no daypart"
    assert cell["bucket"] == attribution.BREAKS_ADDED
