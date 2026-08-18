"""Two keys, two jobs, and the measurement that says they must stay two.

A programme has to be findable across seasons — a new season of מאסטר שף must
reach the old one's history, or every audience signal it has is lost at the
moment it is needed. And the model's cells have to keep predicting as well as
they do, which they manage partly by NOT collapsing seasons and repeats.

Those are opposite requirements, so they get opposite functions. This suite
pins both, and pins the reason they are not one.
"""

from __future__ import annotations

import pytest

from kairos.data.title_features import canonicalize_series, series_join_key


# ------------------------------------------------- the join key: one series

@pytest.mark.parametrize("title", [
    "מאסטר שף",
    "מאסטר שף 12",
    "מאסטר שף - הבחירה הבלתי אפשרית",
    "מאסטר שף – הקינוחים המהפכניים",
    "מאסטר שף - השווארמה המהפכנית של עדן הראל (ש.ח.)",
    "[מאסטר שף עונה 7 ש.ח]",
])
def test_every_way_a_broadcaster_writes_one_series_reaches_one_key(title):
    """The owner's own example. Eleven titles for this series arrive in a single
    fortnight's schedule, because a broadcaster names every episode; without
    this they are eleven programmes with no history between them."""
    assert series_join_key(title) == series_join_key("מאסטר שף")


def test_the_same_title_bracketed_and_not_is_the_same_programme():
    """The aired-spots log wraps its titles and the schedules do not. They are
    the two ends of the join, so a bracket cannot be an identity."""
    assert series_join_key("[זהו זה עונה 7 ש.ח]") == series_join_key("זהו זה עונה 7 ש.ח")


def test_a_break_between_two_programmes_belongs_to_the_one_it_followed():
    """19.2% of spot rows are written "[A] * [B]". Joined, they make a series
    that is neither and fragment both."""
    assert series_join_key("[אבודים עונה 17 ש.ח] * [משחקי השף עונה 7 ש.ח]") \
        == series_join_key("אבודים")


def test_the_episode_name_after_a_dash_is_not_part_of_the_series():
    assert series_join_key("שום פלפל ושמן זית - אלי יצפאן (ש.ח.)") \
        == series_join_key("שום פלפל ושמן זית")


def test_a_hyphen_inside_a_name_is_not_a_separator():
    """The dash rule needs its spaces, or a hyphenated name loses its second
    half and two unrelated programmes meet under one key."""
    assert series_join_key("ד\"ר-פנדל") != series_join_key("ד\"ר")


def test_two_different_programmes_keep_two_keys():
    assert series_join_key("הכוורת") != series_join_key("הצינור")


def test_a_title_that_is_only_markers_still_gets_a_key():
    """Never empty: an empty key is a bucket every unreadable title falls into."""
    assert series_join_key("[עונה 3]")
    assert series_join_key("ש.ח")


def test_nothing_at_all_is_the_only_empty_key():
    assert series_join_key("") == ""
    assert series_join_key(None) == ""


# ------------------------------------ the fit key: deliberately NOT the same

def test_the_model_s_key_still_separates_a_repeat_from_its_premiere():
    """THE REASON THERE ARE TWO KEYS, pinned so it is not "fixed" by accident.

    canonicalize_series leaves the repeat marker inside the key for a wrapped
    title, which looks like a bug and is one. It was fixed, verified not to
    over-collapse, and MEASURED WORSE out of sample on every metric — log RMSE
    0.6834 to 0.7008, points MAE 1.1883 to 1.5028, and the bias flipping sign —
    because a repeat draws a third of the audience of a first broadcast (5.77
    against 1.93 mean TVR over 50,386 rows) and the fit was using that marker to
    know. So it stays until the effect it carries has a name of its own.

    docs/programme-identity.md holds the measurement and the design that
    separates them. If this assertion is ever changed, the backtest is the thing
    to run, not the reasoning.
    """
    assert canonicalize_series("[זהו זה עונה 7 ש.ח]") != canonicalize_series("[זהו זה]")


def test_the_two_keys_are_not_the_same_function():
    wrapped = "[מאסטר שף עונה 7 ש.ח]"
    assert series_join_key(wrapped) != canonicalize_series(wrapped)
    assert series_join_key(wrapped) == "מאסטר שפ"


# --------------------------------------------- what the join key is worth

def test_the_join_key_finds_more_history_than_the_fit_key(tmp_path):
    """Measured on the real files: 120 of 704 future rows find a historical key
    under the fit key, 202 under the join key. The number is not asserted — the
    corpora change — but the ordering is the whole point of the second key."""
    import pandas as pd

    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    future = root / "data" / "reference" / "CompetitorProgrammes.csv"
    history = root / "data" / "Spots.csv"
    if not future.exists() or not history.exists():
        pytest.skip("the corpora are not on this machine")

    fut = pd.read_csv(future, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    hist = pd.read_csv(history, encoding="utf-8-sig", dtype=str, keep_default_na=False)
    hist_titles = {t for t in hist["Title"] if str(t).strip()}

    def covered(key):
        known = {key(t) for t in hist_titles} - {""}
        return sum(1 for t in fut["Title"] if key(t) in known)

    assert covered(series_join_key) > covered(canonicalize_series)


def test_the_contract_carries_the_key_rather_than_leaving_it_to_be_recomputed():
    """Three places need this identity. Written into the file, they cannot
    disagree about it; recomputed, they can and eventually will."""
    from kairos.model import freetv_epg, keshet_epg

    assert "SeriesKey" in keshet_epg.CARRIED_COLUMNS
    rows, _ = freetv_epg.to_contract_rows([{
        "id": 1, "title": "מאסטר שף - הפפנש מגיע", "since": "2026-08-19T18:00:00Z",
        "till": "2026-08-19T19:00:00Z", "repeat": False, "liveBroadcast": True,
    }], channel="קשת 12")
    assert rows[0]["SeriesKey"] == "מאסטר שפ"
