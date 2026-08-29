"""A plan's revenue, beside the advertising load it assumes.

A revenue figure reads as money on the table. It is only that if the airtime
behind it is airtime the channel would really sell, and measured on the real
month the shipped plan carries 1.70 times the seconds the operator channel aired
and 2.70 times the pods. Nothing said so, because the number to compare against
was never computed until the broadcaster's own pod numbering was read.

This is the disclosure-at-the-number rule: a caveat one screen away from a
figure is not disclosure. What these tests pin is mostly what the comparison
must REFUSE to do -- an absent as-run source, a source covering other days, and
a source with no commercial pods all have to read as unknown, because the one
failure that matters here is a missing baseline quietly narrated as agreement.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.model.plan_against_aired import compare_plan_to_aired, disclosure_sentence


def _aired(day: str, pods: int, seconds_each: float = 200.0, channel: str = "A") -> pd.DataFrame:
    rows = []
    for pod in range(pods):
        rows.append({
            "Channel": channel,
            "air_dt": pd.Timestamp(day) + pd.Timedelta(hours=pod),
            "Duration": seconds_each,
            "Pos. Block 1": 1,
            "Spots Block 1": 1,
            "Spot type": "פרסומת",
            "TVR": 1.0,
        })
    return pd.DataFrame(rows)


def test_the_comparison_is_the_ratio_and_nothing_more():
    """It states the two loads and their ratio. It does not adjust revenue and
    does not call the plan wrong; the operator draws that conclusion."""
    spots = _aired("2024-11-01", pods=10, seconds_each=200.0)  # 2000 aired seconds
    out = compare_plan_to_aired(
        plan_pods=20, plan_ad_seconds=4000.0, spots=spots, channel="A", days=["2024-11-01"])
    assert out["comparable"] is True
    assert out["aired_pods"] == 10
    assert out["aired_ad_seconds"] == pytest.approx(2000.0)
    assert out["pod_ratio"] == pytest.approx(2.0)
    assert out["ad_seconds_ratio"] == pytest.approx(2.0)
    assert out["aired_days"] == 1


def test_the_as_run_side_is_restricted_to_the_days_the_plan_covers():
    """A month of plan must never be compared against a year of broadcast."""
    spots = pd.concat([_aired("2024-11-01", 10), _aired("2024-11-02", 10)], ignore_index=True)
    scoped = compare_plan_to_aired(
        plan_pods=20, plan_ad_seconds=4000.0, spots=spots, channel="A", days=["2024-11-01"])
    assert scoped["aired_pods"] == 10 and scoped["aired_days"] == 1
    everything = compare_plan_to_aired(
        plan_pods=20, plan_ad_seconds=4000.0, spots=spots, channel="A", days=None)
    assert everything["aired_pods"] == 20 and everything["aired_days"] == 2


@pytest.mark.parametrize("spots,days,fragment", [
    (None, None, "no as-run source"),
    (pd.DataFrame(), None, "no as-run source"),
    (_aired("2024-11-01", 10), ["2099-01-01"], "none of the days"),
])
def test_a_missing_baseline_reads_unknown_and_never_as_parity(spots, days, fragment):
    out = compare_plan_to_aired(
        plan_pods=20, plan_ad_seconds=4000.0, spots=spots, channel="A", days=days)
    assert out["comparable"] is False
    assert out["ad_seconds_ratio"] is None
    assert out["aired_ad_seconds"] is None
    assert fragment in out["reason"]
    assert disclosure_sentence(out, "he") is None, "an unknown baseline gets no sentence at all"
    assert disclosure_sentence(out, "en") is None


def test_a_channel_with_no_commercial_pods_is_not_comparable():
    """Promos and sponsorships carry no block number, so a channel that aired
    only those has no commercial baseline to compare against."""
    promos = _aired("2024-11-01", 5)
    promos["Spots Block 1"] = 0
    promos["Pos. Block 1"] = 0
    out = compare_plan_to_aired(
        plan_pods=20, plan_ad_seconds=4000.0, spots=promos, channel="A", days=None)
    assert out["comparable"] is False
    assert "no commercial pods" in out["reason"]


def test_the_comparison_answers_only_for_the_channel_asked_for():
    """The operator's own channel. What a competitor airs is out of bounds here
    as everywhere else, so a competitor's pods must not leak into the baseline."""
    spots = pd.concat([
        _aired("2024-11-01", 10, channel="A"),
        _aired("2024-11-01", 40, channel="B"),
    ], ignore_index=True)
    out = compare_plan_to_aired(
        plan_pods=20, plan_ad_seconds=4000.0, spots=spots, channel="A", days=None)
    assert out["aired_pods"] == 10, "a competitor's airtime must not enter the operator's baseline"
    assert out["channel"] == "A"


def test_the_sentence_states_the_volume_and_refuses_a_verdict():
    spots = _aired("2024-11-01", pods=10, seconds_each=200.0)
    out = compare_plan_to_aired(
        plan_pods=20, plan_ad_seconds=4000.0, spots=spots, channel="A", days=None)
    hebrew = disclosure_sentence(out, "he")
    english = disclosure_sentence(out, "en")
    for sentence in (hebrew, english):
        assert sentence and "2.00" in sentence
    assert "לא תוספת על מה שמשודר היום" in hebrew
    assert "not an increment on what airs today" in english


# --- the wiring: one chokepoint, both windows ----------------------------------

def test_the_overview_carries_the_comparison_for_both_windows():
    """The whole-plan totals and the planning-week slice are built by one
    function, so they cannot disagree about how much airtime the revenue
    assumes. Each is scoped to its OWN days: a week's revenue compared against a
    month of broadcast would understate the ratio and quietly flatter the plan.
    """
    from fastapi.testclient import TestClient

    from kairos_api import auth_store
    from kairos_api.server import app

    users = auth_store.load_users()
    if not users:  # a checkout with no seeded account cannot exercise the wall
        return
    client = TestClient(app)
    client.cookies.set(
        auth_store.COOKIE_NAME,
        auth_store.create_session(users[0]["username"], users[0]["role"]),
    )
    summary = client.get("/api/overview").json().get("summary") or {}
    whole = summary.get("ad_load_against_aired")
    assert whole is not None, "the revenue figure must carry the load it assumes"

    week = (summary.get("week") or {}).get("ad_load_against_aired")
    if week and whole.get("comparable") and week.get("comparable"):
        assert week["aired_days"] <= whole["aired_days"], (
            "the week slice must compare against the week's own broadcast, not the month's"
        )
        assert week["plan_ad_seconds"] <= whole["plan_ad_seconds"]


def test_the_ad_minutes_tile_states_the_volume_and_hides_it_when_unknown():
    """Disclosure at the number, not one screen away.

    The projected-revenue tile reads as money on the table. The tile beside it
    now says how that airtime compares with what the channel actually aired, so
    the two are read together. A comparison the backend could not make must
    render as nothing at all -- an unknown baseline shown as a bare total is the
    same silence this replaced.
    """
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "tv-break-dashboard" / "src"
              / "today" / "SummaryMetrics.jsx").read_text(encoding="utf-8")
    assert "ad_load_against_aired" in source, "the tile must read the comparison"
    assert "adLoad.comparable" in source, "and must respect the not-comparable state"
    assert "sub={minutesSub}" in source, "the ratio belongs on the tile, not only in a tooltip"
    # The honest-absence path: the ratio is null unless the backend said comparable.
    assert "loadRatio !== null" in source
