"""Tests for the rich predicate evaluator and predicate-bearing constraints.

Covers:
  - AND vs OR nesting and mixing
  - programme field: regex, exact 'is', 'contains', 'in'
  - daypart 'in'
  - weekday 'is_not'
  - date 'between'
  - hour 'gte'
  - implicit channel filter (operator_channel)
  - back-compat: legacy flat scope_type/scope_value constraint still resolves
  - where_json JSON round-trip through PlacementConstraint
  - unknown field/operator -> False (not crash, not match-all)
  - malformed regex -> False (not crash, not match-all)
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from kairos.optimize.constraints_store import (
    COLUMNS,
    PlacementConstraint,
    _parse_where_json,
    load_constraints,
    resolve_constraints,
)
from kairos.optimize.optimizer import ProgramSegment
from kairos.optimize.predicate import evaluate_predicate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def seg(**overrides) -> ProgramSegment:
    base = dict(
        segment_id="s1",
        channel="קשת 12",
        day="2026-06-15",           # Monday
        start_seconds=20 * 3600.0,  # 20:00 -> hour=20, daypart=prime
        duration_seconds=3600.0,
        program_type="Drama",
        program_title="חתונה ממבט ראשון",
        baseline_tvr=10.0,
        cpp=1000.0,
        impact_coefficient=0.0,
        retention_baseline=1.0,
        premium=1.0,
        is_gold=False,
        max_breaks=4,
        break_length_seconds=120.0,
        unit_seconds=1.0,
    )
    base.update(overrides)
    return ProgramSegment(**base)


def pred(combinator: str, *conditions) -> dict:
    return {"combinator": combinator, "conditions": list(conditions)}


def cond(field: str, operator: str, value) -> dict:
    return {"field": field, "operator": operator, "value": value}


SEGMENT = seg()


# ---------------------------------------------------------------------------
# evaluate_predicate: basic combinators
# ---------------------------------------------------------------------------

class TestCombinators:
    def test_and_all_true(self):
        group = pred("and",
            cond("programme", "is", "חתונה ממבט ראשון"),
            cond("daypart", "is", "prime"),
        )
        assert evaluate_predicate(group, SEGMENT) is True

    def test_and_one_false(self):
        group = pred("and",
            cond("programme", "is", "חתונה ממבט ראשון"),
            cond("daypart", "is", "morning"),    # wrong daypart
        )
        assert evaluate_predicate(group, SEGMENT) is False

    def test_or_one_true(self):
        group = pred("or",
            cond("genre", "is", "News"),         # wrong genre
            cond("daypart", "is", "prime"),       # right
        )
        assert evaluate_predicate(group, SEGMENT) is True

    def test_or_all_false(self):
        group = pred("or",
            cond("genre", "is", "News"),
            cond("daypart", "is", "morning"),
        )
        assert evaluate_predicate(group, SEGMENT) is False

    def test_nested_and_or(self):
        # (genre==Drama OR genre==Crime) AND (daypart==prime OR daypart==evening)
        group = pred("and",
            pred("or",
                cond("genre", "is", "Drama"),
                cond("genre", "is", "Crime"),
            ),
            pred("or",
                cond("daypart", "is", "prime"),
                cond("daypart", "is", "evening"),
            ),
        )
        assert evaluate_predicate(group, SEGMENT) is True

    def test_empty_conditions_yields_false_for_and(self):
        assert evaluate_predicate({"combinator": "and", "conditions": []}, SEGMENT) is False

    def test_empty_conditions_yields_false_for_or(self):
        assert evaluate_predicate({"combinator": "or", "conditions": []}, SEGMENT) is False

    def test_unknown_combinator_yields_false(self):
        assert evaluate_predicate({"combinator": "xor", "conditions": [
            cond("genre", "is", "Drama"),
        ]}, SEGMENT) is False


# ---------------------------------------------------------------------------
# programme field
# ---------------------------------------------------------------------------

class TestProgrammeField:
    def test_is_case_insensitive(self):
        group = pred("and", cond("programme", "is", "חתונה ממבט ראשון"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_is_not_matches_other_title(self):
        group = pred("and", cond("programme", "is_not", "חדשות הערב"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_contains_substring(self):
        group = pred("and", cond("programme", "contains", "ממבט"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_not_contains_absent_substring(self):
        group = pred("and", cond("programme", "not_contains", "חדשות"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_starts_with(self):
        group = pred("and", cond("programme", "starts_with", "חתונה"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_ends_with(self):
        group = pred("and", cond("programme", "ends_with", "ראשון"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_regex_match(self):
        group = pred("and", cond("programme", "regex", "^חתונה"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_regex_no_match(self):
        group = pred("and", cond("programme", "regex", "^חדשות"))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_regex_case_insensitive(self):
        # Hebrew doesn't have case, but ASCII part should be case-insensitive
        latin_seg = seg(program_title="Drama Special")
        group = pred("and", cond("programme", "regex", "^drama"))
        assert evaluate_predicate(group, latin_seg) is True

    def test_in_list(self):
        group = pred("and", cond("programme", "in", ["חדשות הערב", "חתונה ממבט ראשון"]))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_in_list_not_present(self):
        group = pred("and", cond("programme", "in", ["חדשות הערב", "תוכנית אחרת"]))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_in_requires_list(self):
        group = pred("and", cond("programme", "in", "חתונה ממבט ראשון"))
        assert evaluate_predicate(group, SEGMENT) is False


# ---------------------------------------------------------------------------
# daypart 'in'
# ---------------------------------------------------------------------------

class TestDaypartField:
    def test_is_prime(self):
        group = pred("and", cond("daypart", "is", "prime"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_is_morning_false(self):
        group = pred("and", cond("daypart", "is", "morning"))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_in_list_hit(self):
        group = pred("and", cond("daypart", "in", ["prime", "evening"]))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_in_list_miss(self):
        group = pred("and", cond("daypart", "in", ["morning", "noon"]))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_is_not(self):
        group = pred("and", cond("daypart", "is_not", "morning"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_noon_segment(self):
        noon_seg = seg(start_seconds=13 * 3600)
        group = pred("and", cond("daypart", "is", "noon"))
        assert evaluate_predicate(group, noon_seg) is True


# ---------------------------------------------------------------------------
# weekday 'is_not'
# ---------------------------------------------------------------------------

class TestWeekdayField:
    # 2026-06-15 is a Monday
    def test_is_monday(self):
        group = pred("and", cond("weekday", "is", "Mon"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_is_not_tuesday(self):
        group = pred("and", cond("weekday", "is_not", "Tue"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_is_not_monday_false(self):
        group = pred("and", cond("weekday", "is_not", "Mon"))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_in_weekdays(self):
        group = pred("and", cond("weekday", "in", ["Mon", "Wed", "Fri"]))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_in_weekdays_miss(self):
        group = pred("and", cond("weekday", "in", ["Tue", "Thu", "Sat"]))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_wednesday_segment(self):
        # 2026-06-17 is a Wednesday
        wed_seg = seg(day="2026-06-17")
        group = pred("and", cond("weekday", "is", "Wed"))
        assert evaluate_predicate(group, wed_seg) is True


# ---------------------------------------------------------------------------
# date 'between'
# ---------------------------------------------------------------------------

class TestDateField:
    def test_is_exact(self):
        group = pred("and", cond("date", "is", "2026-06-15"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_before(self):
        group = pred("and", cond("date", "before", "2026-06-16"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_before_equal_false(self):
        group = pred("and", cond("date", "before", "2026-06-15"))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_after(self):
        group = pred("and", cond("date", "after", "2026-06-14"))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_between_hit(self):
        group = pred("and", cond("date", "between", {"min": "2026-06-14", "max": "2026-06-16"}))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_between_miss(self):
        group = pred("and", cond("date", "between", {"min": "2026-06-16", "max": "2026-06-18"}))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_between_missing_min_false(self):
        group = pred("and", cond("date", "between", {"max": "2026-06-18"}))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_in_list(self):
        group = pred("and", cond("date", "in", ["2026-06-14", "2026-06-15"]))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_in_list_miss(self):
        group = pred("and", cond("date", "in", ["2026-06-16", "2026-06-17"]))
        assert evaluate_predicate(group, SEGMENT) is False


# ---------------------------------------------------------------------------
# hour 'gte' and other numeric operators
# ---------------------------------------------------------------------------

class TestHourField:
    # SEGMENT has start_seconds=20*3600 -> hour=20
    def test_eq(self):
        group = pred("and", cond("hour", "eq", 20))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_eq_false(self):
        group = pred("and", cond("hour", "eq", 21))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_lt(self):
        group = pred("and", cond("hour", "lt", 21))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_lt_equal_false(self):
        group = pred("and", cond("hour", "lt", 20))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_lte(self):
        group = pred("and", cond("hour", "lte", 20))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_gt(self):
        group = pred("and", cond("hour", "gt", 19))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_gte(self):
        group = pred("and", cond("hour", "gte", 20))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_gte_false(self):
        group = pred("and", cond("hour", "gte", 21))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_between_hit(self):
        group = pred("and", cond("hour", "between", {"min": 19, "max": 22}))
        assert evaluate_predicate(group, SEGMENT) is True

    def test_between_miss(self):
        group = pred("and", cond("hour", "between", {"min": 21, "max": 23}))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_hour_wraps_at_24(self):
        # start_seconds = 24.5 * 3600 -> hour = int(24.5 * 3600 // 3600) % 24 = 0
        midnight_seg = seg(start_seconds=24.5 * 3600)
        group = pred("and", cond("hour", "eq", 0))
        assert evaluate_predicate(group, midnight_seg) is True


# ---------------------------------------------------------------------------
# Implicit channel filter
# ---------------------------------------------------------------------------

class TestChannelFilter:
    def test_matching_channel_allows_evaluation(self):
        group = pred("and", cond("genre", "is", "Drama"))
        assert evaluate_predicate(group, SEGMENT, operator_channel="קשת 12") is True

    def test_non_matching_channel_blocks_evaluation(self):
        # A constraint authored for the operator's channel never matches a
        # segment on a different channel.
        group = pred("and", cond("genre", "is", "Drama"))
        other = seg(channel="רשת 13")
        assert evaluate_predicate(group, other, operator_channel="קשת 12") is False

    def test_empty_operator_channel_matches_any_channel(self):
        # operator_channel empty = honest no-op before the operator picks one
        group = pred("and", cond("genre", "is", "Drama"))
        other = seg(channel="רשת 13")
        assert evaluate_predicate(group, other, operator_channel="") is True

    def test_none_operator_channel_matches_any_channel(self):
        group = pred("and", cond("genre", "is", "Drama"))
        other = seg(channel="רשת 13")
        assert evaluate_predicate(group, other, operator_channel=None) is True


# ---------------------------------------------------------------------------
# Defensive: unknown field / operator / malformed regex
# ---------------------------------------------------------------------------

class TestDefensive:
    def test_unknown_field_yields_false(self):
        group = pred("and", cond("unknown_field", "is", "anything"))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_unknown_operator_yields_false(self):
        group = pred("and", cond("genre", "contains_all", "Drama"))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_malformed_regex_yields_false_not_crash(self):
        group = pred("and", cond("programme", "regex", "[invalid regex"))
        # Must not raise; must return False
        result = evaluate_predicate(group, SEGMENT)
        assert result is False

    def test_malformed_regex_does_not_match_everything(self):
        # A broken regex must not accidentally match-all.
        group = pred("and", cond("programme", "regex", "(unclosed"))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_wrong_operator_for_field_is_false(self):
        # 'between' is not a valid operator for programme
        group = pred("and", cond("programme", "between", "abc"))
        assert evaluate_predicate(group, SEGMENT) is False

    def test_non_list_in_operator_is_false(self):
        # 'in' expects a list; passing a string is a client error -> False
        group = pred("and", cond("genre", "in", "Drama"))
        assert evaluate_predicate(group, SEGMENT) is False


# ---------------------------------------------------------------------------
# where_json round-trip through PlacementConstraint
# ---------------------------------------------------------------------------

class TestWhereJsonRoundTrip:
    def test_parse_valid_json(self):
        tree = {"combinator": "and", "conditions": [
            {"field": "genre", "operator": "is", "value": "Drama"},
        ]}
        raw = json.dumps(tree)
        parsed = _parse_where_json(raw)
        assert parsed == tree

    def test_parse_empty_string_yields_none(self):
        assert _parse_where_json("") is None
        assert _parse_where_json("   ") is None

    def test_parse_invalid_json_yields_none(self):
        assert _parse_where_json("{not valid json}") is None

    def test_parse_non_group_json_yields_none(self):
        # A JSON object without "combinator" is not a valid Group
        assert _parse_where_json('{"field": "genre"}') is None

    def test_round_trip_through_csv(self, tmp_path):
        tree = {
            "combinator": "and",
            "conditions": [
                {"field": "daypart", "operator": "in", "value": ["prime", "evening"]},
                {"field": "hour", "operator": "gte", "value": 20},
            ],
        }
        raw_json = json.dumps(tree, ensure_ascii=False)
        row = {column: "" for column in COLUMNS}
        row.update({
            "constraint_id": "c-test",
            "scope_type": "always",
            "effect": "forbid",
            "where_json": raw_json,
        })
        path = tmp_path / "kairos_constraints.csv"
        pd.DataFrame([row])[list(COLUMNS)].to_csv(path, index=False, encoding="utf-8-sig")
        loaded = load_constraints(path)
        assert len(loaded) == 1
        assert loaded[0].where == tree

    def test_legacy_constraint_loads_where_as_none(self, tmp_path):
        """A CSV without where_json col -> constraint.where is None (back-compat)."""
        # Write only the original COLUMNS minus where_json
        old_columns = [c for c in COLUMNS if c != "where_json"]
        row = {c: "" for c in old_columns}
        row.update({
            "constraint_id": "legacy1",
            "scope_type": "channel",
            "scope_value": "קשת 12",
            "effect": "pin_count",
            "count": "2",
        })
        path = tmp_path / "legacy_constraints.csv"
        pd.DataFrame([row])[old_columns].to_csv(path, index=False, encoding="utf-8-sig")
        loaded = load_constraints(path)
        assert len(loaded) == 1
        assert loaded[0].where is None
        assert loaded[0].count == 2


# ---------------------------------------------------------------------------
# Back-compat: legacy flat scope_type/scope_value still resolves
# ---------------------------------------------------------------------------

class TestLegacyBackcompat:
    def test_legacy_always_scope_still_matches(self):
        constraint = PlacementConstraint(
            constraint_id="legacy1",
            scope_type="always",
            scope_value="",
            effect="forbid",
            where=None,
        )
        _, counts, forbids, _ = resolve_constraints([SEGMENT], [constraint])
        assert "s1" in forbids

    def test_legacy_programme_scope_still_matches(self):
        constraint = PlacementConstraint(
            constraint_id="legacy2",
            scope_type="programme",
            scope_value="חתונה ממבט ראשון",
            effect="pin_count",
            count=2,
            where=None,
        )
        _, counts, _, _ = resolve_constraints([SEGMENT], [constraint])
        assert counts.get("s1") == 2

    def test_legacy_programme_scope_wrong_title_no_match(self):
        constraint = PlacementConstraint(
            constraint_id="legacy3",
            scope_type="programme",
            scope_value="חדשות הערב",
            effect="pin_count",
            count=2,
            where=None,
        )
        _, counts, _, _ = resolve_constraints([SEGMENT], [constraint])
        assert "s1" not in counts

    def test_legacy_channel_scope_still_matches(self):
        constraint = PlacementConstraint(
            constraint_id="legacy4",
            scope_type="channel",
            scope_value="קשת 12",
            effect="pin_count",
            count=1,
            where=None,
        )
        _, counts, _, _ = resolve_constraints([SEGMENT], [constraint])
        assert counts.get("s1") == 1

    def test_legacy_weekday_scope_still_matches(self):
        # 2026-06-15 is a Monday (isoweekday=1)
        constraint = PlacementConstraint(
            constraint_id="legacy5",
            scope_type="weekday",
            scope_value="1",
            effect="pin_count",
            count=3,
            where=None,
        )
        _, counts, _, _ = resolve_constraints([SEGMENT], [constraint])
        assert counts.get("s1") == 3


# ---------------------------------------------------------------------------
# operator_channel in resolve_constraints
# ---------------------------------------------------------------------------

class TestOperatorChannelInResolver:
    def test_predicate_constraint_matches_own_channel(self):
        tree = pred("and", cond("genre", "is", "Drama"))
        constraint = PlacementConstraint(
            constraint_id="p1",
            scope_type="always",
            effect="forbid",
            where=tree,
        )
        _, _, forbids, _ = resolve_constraints(
            [SEGMENT], [constraint], operator_channel="קשת 12",
        )
        assert "s1" in forbids

    def test_predicate_constraint_does_not_match_other_channel(self):
        tree = pred("and", cond("genre", "is", "Drama"))
        constraint = PlacementConstraint(
            constraint_id="p2",
            scope_type="always",
            effect="forbid",
            where=tree,
        )
        other = seg(segment_id="s_other", channel="רשת 13")
        _, _, forbids, _ = resolve_constraints(
            [other], [constraint], operator_channel="קשת 12",
        )
        assert "s_other" not in forbids

    def test_empty_operator_channel_matches_any_channel_in_resolver(self):
        tree = pred("and", cond("genre", "is", "Drama"))
        constraint = PlacementConstraint(
            constraint_id="p3",
            scope_type="always",
            effect="forbid",
            where=tree,
        )
        other = seg(segment_id="s_other", channel="רשת 13")
        _, _, forbids, _ = resolve_constraints(
            [other], [constraint], operator_channel="",
        )
        assert "s_other" in forbids

    def test_legacy_constraint_with_operator_channel_filters(self):
        # Legacy flat constraint: operator_channel set -> only own channel matches
        constraint = PlacementConstraint(
            constraint_id="flat1",
            scope_type="always",
            effect="forbid",
            where=None,
        )
        own = seg(segment_id="own", channel="קשת 12")
        other = seg(segment_id="other", channel="רשת 13")
        _, _, forbids, _ = resolve_constraints(
            [own, other], [constraint], operator_channel="קשת 12",
        )
        assert "own" in forbids
        assert "other" not in forbids
