"""Sanity and feature tests for the deterministic Israeli broadcast calendar.

Covers the bundled table's internal consistency (chol hamoed ranges bounded by
their yom tov days, Hanukkah spanning eight days, the yom tov enumeration
frozen per year), the pure feature function on known dates (shabbat, yom tov,
chol hamoed, school summer, Israeli season bands, erev flags), tolerant
missing-file behavior, and the purely additive frame annotation seam.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from kairos.data.israel_calendar import (
    CALENDAR_PATH,
    FEATURE_KEYS,
    KIND_PRIORITY,
    annotate_calendar,
    calendar_features,
    load_calendar,
    season_of,
)

YEARS = (2024, 2025, 2026, 2027)

# The frozen yom tov enumeration: both Rosh Hashana days, Yom Kippur, first
# day of Sukkot, Shmini Atzeret/Simchat Torah, first and seventh Pesach days,
# Shavuot. Eight dates per calendar year. Changing the table must consciously
# change this list too.
EXPECTED_YOM_TOV = {
    2024: ["04-23", "04-29", "06-12", "10-03", "10-04", "10-12", "10-17", "10-24"],
    2025: ["04-13", "04-19", "06-02", "09-23", "09-24", "10-02", "10-07", "10-14"],
    2026: ["04-02", "04-08", "05-22", "09-12", "09-13", "09-21", "09-26", "10-03"],
    2027: ["04-22", "04-28", "06-11", "10-02", "10-03", "10-11", "10-16", "10-23"],
}

NAME_PESACH_1 = "פסח"
NAME_PESACH_7 = "שביעי של פסח"
NAME_CH_PESACH = "חול המועד פסח"
NAME_SUKKOT_1 = "סוכות"
NAME_CH_SUKKOT = "חול המועד סוכות"
NAME_SHMINI = "שמיני עצרת ושמחת תורה"


@pytest.fixture(scope="module")
def table():
    rows = load_calendar()
    assert rows, f"bundled calendar table missing or empty at {CALENDAR_PATH}"
    return rows


def _one(table, name, year):
    matches = [
        row for row in table
        if row.name_he == name and row.start_date.year == year
    ]
    assert len(matches) == 1, f"expected exactly one {name!r} row in {year}, got {len(matches)}"
    return matches[0]


def _expand_yom_tov(table, year):
    dates = set()
    for row in table:
        if row.is_yom_tov:
            day = row.start_date
            while day <= row.end_date:
                if day.year == year:
                    dates.add(day)
                day += dt.timedelta(days=1)
    return dates


class TestTableSanity:
    def test_ranges_well_formed_and_kinds_known(self, table):
        for row in table:
            assert row.start_date <= row.end_date, row
            assert row.kind in KIND_PRIORITY, row
            assert row.name_he, row

    def test_every_year_covered(self, table):
        for year in YEARS:
            years_touched = {
                row for row in table
                if row.start_date.year == year or row.end_date.year == year
            }
            assert years_touched, f"no calendar rows touch {year}"

    def test_chol_hamoed_adjacent_to_bounding_yom_tov(self, table):
        for year in YEARS:
            pesach_1 = _one(table, NAME_PESACH_1, year)
            pesach_7 = _one(table, NAME_PESACH_7, year)
            ch_pesach = _one(table, NAME_CH_PESACH, year)
            assert ch_pesach.start_date == pesach_1.end_date + dt.timedelta(days=1)
            assert ch_pesach.end_date == pesach_7.start_date - dt.timedelta(days=1)

            sukkot_1 = _one(table, NAME_SUKKOT_1, year)
            shmini = _one(table, NAME_SHMINI, year)
            ch_sukkot = _one(table, NAME_CH_SUKKOT, year)
            assert ch_sukkot.start_date == sukkot_1.end_date + dt.timedelta(days=1)
            assert ch_sukkot.end_date == shmini.start_date - dt.timedelta(days=1)

    def test_chol_hamoed_rows_are_not_yom_tov_and_are_school_holidays(self, table):
        ch_rows = [row for row in table if row.kind == "chol_hamoed"]
        assert len(ch_rows) == 2 * len(YEARS)
        for row in ch_rows:
            assert not row.is_yom_tov, row
            assert row.is_school_holiday, row

    def test_hanukkah_is_eight_days_each_year(self, table):
        hanukkah = [row for row in table if row.kind == "hanukkah"]
        assert sorted(row.start_date.year for row in hanukkah) == list(YEARS)
        for row in hanukkah:
            assert (row.end_date - row.start_date).days == 7, row
            assert not row.is_yom_tov, row

    def test_yom_tov_exactly_on_enumerated_days(self, table):
        for year, month_days in EXPECTED_YOM_TOV.items():
            expected = {
                dt.date.fromisoformat(f"{year}-{month_day}") for month_day in month_days
            }
            assert _expand_yom_tov(table, year) == expected, f"yom tov mismatch in {year}"

    def test_yom_tov_flag_only_on_yom_tov_kind(self, table):
        for row in table:
            assert row.is_yom_tov == (row.kind == "yom_tov"), row

    def test_school_summer_each_year(self, table):
        for year in YEARS:
            rows = [
                row for row in table
                if row.kind == "school_summer" and row.start_date.year == year
            ]
            assert len(rows) == 1
            assert rows[0].start_date == dt.date(year, 7, 1)
            assert rows[0].end_date == dt.date(year, 8, 31)
            assert rows[0].is_school_holiday

    def test_election_rows_present(self, table):
        elections = [row for row in table if row.kind == "election"]
        assert dt.date(2024, 2, 27) in {row.start_date for row in elections}
        provisional = [row for row in elections if row.start_date.year == 2026]
        assert provisional and "PROVISIONAL" in provisional[0].notes


class TestCalendarFeatures:
    def test_feature_keys_exact(self):
        features = calendar_features(dt.date(2025, 7, 15))
        assert tuple(features.keys()) == FEATURE_KEYS

    def test_saturday_is_shabbat_and_blackout(self):
        features = calendar_features(dt.date(2026, 7, 25))  # a plain Saturday
        assert features["weekday_iso"] == 6
        assert features["is_shabbat"] and features["religious_blackout"]
        assert not features["is_yom_tov"] and not features["is_erev_shabbat"]

    def test_friday_is_erev_shabbat_not_blackout(self):
        features = calendar_features(dt.date(2026, 7, 24))  # a plain Friday
        assert features["is_erev_shabbat"] and not features["is_shabbat"]
        assert not features["religious_blackout"]

    def test_yom_kippur_is_yom_tov_and_blackout(self):
        features = calendar_features(dt.date(2025, 10, 2))  # Yom Kippur, a Thursday
        assert features["is_yom_tov"] and features["religious_blackout"]
        assert not features["is_shabbat"]
        assert features["holiday_kind"] == "yom_tov"

    def test_chol_hamoed_sukkot_flags(self):
        features = calendar_features(dt.date(2024, 10, 20))
        assert features["is_chol_hamoed"] and features["is_school_holiday"]
        assert features["holiday_kind"] == "chol_hamoed"
        assert not features["is_yom_tov"]

    def test_july_15_is_school_summer(self):
        features = calendar_features(dt.date(2025, 7, 15))
        assert features["is_school_holiday"]
        assert features["season"] == "summer"
        assert features["holiday_kind"] == "school_summer"

    def test_hanukkah_and_election_flags(self):
        assert calendar_features(dt.date(2025, 12, 18))["is_hanukkah"]
        assert calendar_features(dt.date(2024, 2, 27))["is_election_day"]

    def test_erev_yom_tov(self):
        # Day before first Pesach day 2025.
        assert calendar_features(dt.date(2025, 4, 12))["is_erev_yom_tov"]
        # Day before Rosh Hashana 2025.
        assert calendar_features(dt.date(2025, 9, 22))["is_erev_yom_tov"]
        # Rosh Hashana day one precedes day two but reads as yom tov, not erev.
        first_day = calendar_features(dt.date(2025, 9, 23))
        assert first_day["is_yom_tov"] and not first_day["is_erev_yom_tov"]
        # Hoshana Rabbah is both chol hamoed and erev yom tov.
        hoshana = calendar_features(dt.date(2025, 10, 13))
        assert hoshana["is_chol_hamoed"] and hoshana["is_erev_yom_tov"]

    def test_season_band_edges(self):
        expected = {
            dt.date(2026, 5, 31): "spring",
            dt.date(2026, 6, 1): "summer",
            dt.date(2026, 9, 30): "summer",
            dt.date(2026, 10, 1): "autumn",
            dt.date(2026, 11, 30): "autumn",
            dt.date(2026, 12, 1): "winter",
            dt.date(2026, 2, 28): "winter",
            dt.date(2026, 3, 1): "spring",
        }
        for day, season in expected.items():
            assert season_of(day) == season, day
            assert calendar_features(day)["season"] == season, day


class TestTolerantLoading:
    def test_missing_file_yields_empty_calendar(self, tmp_path):
        missing = tmp_path / "nope.csv"
        assert load_calendar(missing) == ()
        # Weekday and season still work; table-derived flags stay off.
        features = calendar_features(dt.date(2024, 10, 12), path=missing)
        assert features["is_shabbat"]  # Yom Kippur 2024 is also a Saturday
        assert not features["is_yom_tov"] and features["holiday_kind"] == ""
        assert features["season"] == "autumn"

    def test_bad_rows_skipped(self, tmp_path):
        bad = tmp_path / "bad.csv"
        bad.write_text(
            "# comment line\n"
            "start_date,end_date,name_he,kind,is_yom_tov,is_school_holiday,notes\n"
            "not-a-date,2024-01-02,broken,other,0,0,\n"
            "2024-01-05,2024-01-01,inverted,other,0,0,\n"
            "2024-01-01,2024-01-02,good,other,0,1,\n",
            encoding="utf-8",
        )
        rows = load_calendar(bad)
        assert [row.name_he for row in rows] == ["good"]
        assert rows[0].is_school_holiday


class TestAnnotateCalendar:
    def test_purely_additive(self):
        frame = pd.DataFrame(
            {
                "break_date": ["2024-10-12", "2024-10-20", "2025-07-15", "junk"],
                "revenue": [1.0, 2.0, 3.0, 4.0],
            }
        )
        original = frame.copy(deep=True)
        result = annotate_calendar(frame, "break_date")

        # Input untouched, row count preserved, existing columns byte-identical.
        pd.testing.assert_frame_equal(frame, original)
        assert len(result) == len(frame)
        pd.testing.assert_frame_equal(result[["break_date", "revenue"]], original)

        for key in FEATURE_KEYS:
            assert f"cal_{key}" in result.columns

        assert result.loc[0, "cal_religious_blackout"]  # Yom Kippur 2024
        assert result.loc[1, "cal_is_chol_hamoed"]
        assert result.loc[2, "cal_season"] == "summer"
        # Unparseable date degrades to neutral values, never raises.
        assert result.loc[3, "cal_weekday_iso"] == 0
        assert not result.loc[3, "cal_religious_blackout"]
        assert result.loc[3, "cal_season"] == ""

    def test_missing_date_column_raises(self):
        with pytest.raises(KeyError):
            annotate_calendar(pd.DataFrame({"x": [1]}), "break_date")
