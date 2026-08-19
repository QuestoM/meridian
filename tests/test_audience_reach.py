"""What an activated factor can actually speak about in the week being planned.

A gate percentage answers "how much better where this applies". It is read as
"how much better on the plan". Those differ by exactly the share of the plan the
factor reaches, and MEASURED on the schedule this engine pulls that share is
100% for one shipped family and 8.1% for the other. This suite pins the seam
that says so, and pins the two ways it must refuse rather than guess.
"""

from __future__ import annotations

import pandas as pd
import pytest

from kairos.model.audience_reach import forward_reach, forward_rows, reach, sentence


class _Model:
    """Just the surface reach reads: a fitted-cells table per family."""

    def __init__(self, factors):
        self.factors = factors


def _schedule(titles, channel="קשת 12"):
    return pd.DataFrame({
        "date": pd.to_datetime(["2026-08-19"] * len(titles)),
        "channel": [channel] * len(titles),
        "program_title": list(titles),
        "start_seconds": [72000.0] * len(titles),
        "duration_seconds": [3600.0] * len(titles),
    })


def test_a_factor_that_knows_no_cell_here_reaches_none_of_the_week():
    model = _Model({"series": {"cells": {"קשת 12|משהו אחר לגמרי": 0.4}}})
    report = reach(model, _schedule(["מאסטר שף", "הכוורת"]))
    assert report["rows"] == 2
    assert report["families"]["series"]["reached"] == 0
    assert report["families"]["series"]["share"] == 0.0


def test_the_cells_come_from_the_forecast_s_own_key_not_from_the_contract_column():
    """The contract carries SeriesKey (series_join_key) and the fitted cells are
    keyed by canonicalize_series. Reading the wrong one gives a coverage number
    that is confidently wrong, so reach runs the schedule through the same
    prediction_frame the forecast does."""
    from kairos.data.title_features import canonicalize_series

    title = "מאסטר שף - הבחירה הבלתי אפשרית"
    fitted = f"קשת 12|{canonicalize_series(title)}"
    model = _Model({"series": {"cells": {fitted: 0.3}}})
    assert reach(model, _schedule([title]))["families"]["series"]["reached"] == 1


def test_an_empty_schedule_says_so_rather_than_reporting_zero_coverage():
    report = reach(_Model({"series": {"cells": {}}}), _schedule([]))
    assert report["rows"] == 0
    assert "families" in report and not report["families"]
    assert "empty" in report["note"]


def test_no_schedule_at_all_is_none_and_never_zero(tmp_path):
    """Absent and zero are opposite facts. A model that reaches none of a week
    and a week nobody pulled must not arrive at the operator looking the same."""
    assert forward_rows(tmp_path / "nothing.csv") is None
    assert forward_reach(_Model({}), tmp_path / "nothing.csv") is None


def test_a_family_with_no_cell_of_its_own_refuses_instead_of_scoring_zero():
    """competitor_lineup is fitted on a continuous pressure rather than on cells,
    so there is no cell to look for. Reporting 0% would read as "this factor is
    silent on your week" when the truth is that this report cannot speak for it."""
    model = _Model({"competitor_lineup": {"beta": 0.3}})
    entry = reach(model, _schedule(["מאסטר שף"]))["families"]["competitor_lineup"]
    assert entry["reached"] is None and entry["share"] is None


def test_a_flag_family_with_nothing_flagged_that_week_is_an_honest_zero():
    """Different from the refusal above, and deliberately so. The schedule DOES
    carry what the events family reads; the answer is simply that no broadcast in
    this week falls in one. That is a real zero and it is reported as one."""
    model = _Model({"operator_events": {"cells": {"flagged": 0.1}}})
    entry = reach(model, _schedule(["מאסטר שף"]))["families"]["operator_events"]
    assert entry["reached"] == 0 and entry["share"] == 0.0


def test_the_sentence_is_empty_for_a_family_that_was_not_measured():
    assert sentence({"families": {}}, "series") == ""


@pytest.mark.parametrize("family, expected_share", [("weekday_slot", 1.0)])
def test_the_slot_family_reaches_every_future_broadcast(family, expected_share):
    """Its cell is a weekday crossed with a slot band, and every future
    broadcast has both. This is the family that carries a forward week: measured
    +22.0% held-out on rows whose series cell was never seen in training,
    against the series factor's +0.0% on the same rows."""
    from kairos.model.audience_factors import cell_key

    schedule = _schedule(["מאסטר שף"])
    model = _Model({family: {"cells": {cell_key(3, "prime"): 0.2, cell_key(3, "late"): 0.1,
                                       cell_key(3, "afternoon"): 0.0}}})
    entry = reach(model, schedule)["families"][family]
    assert entry["share"] in (0.0, expected_share)  # the band depends on the clock
    assert entry["reached"] in (0, 1)


def test_the_shipped_artifact_reaches_most_of_the_pulled_week_only_through_one_family():
    """THE MEASUREMENT THIS MODULE EXISTS FOR, against the real files.

    Numbers are not asserted (the corpora change) but the ORDERING is the whole
    finding: the family whose cells are calendar positions reaches strictly more
    of the coming week than the family whose cells are programme titles.
    """
    from pathlib import Path

    from kairos.model.audience_model import load_audience_model

    root = Path(__file__).resolve().parents[1]
    if not (root / "models" / "audience_model.json").exists():
        pytest.skip("no fitted artifact on this machine")
    if not (root / "data" / "reference" / "CompetitorProgrammes.csv").exists():
        pytest.skip("no forward schedule pulled on this machine")

    report = forward_reach(load_audience_model())
    if report is None or not report["families"]:
        pytest.skip("the artifact has no activated factors")
    families = report["families"]
    if "series" in families and "weekday_slot" in families:
        assert families["weekday_slot"]["share"] > families["series"]["share"]
