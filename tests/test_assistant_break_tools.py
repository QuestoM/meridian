"""The assistant's two break read tools, against the real saved plan on disk.

Rule 24 of Kai's prompt names the break as one of this product's nine objects and
the scheduler persona is told to answer in days, segments, breaks and pins. Until
these tools existed no read tool reached the break: ``get_day_detail`` returns
segments carrying a break count, and the pod tools read the spots inside a break.
These tests hold the two new tools to the surface they report:

* every figure is the day board's own, never re-rounded and never re-derived here,
* the delivered-money state stays a state and never becomes a zero,
* a day or a break the plan does not carry is an honest error naming the days
  that can be opened,
* and no rival channel name reaches either tool's output.

Nothing is mocked: the executors run their real code against the saved weekly
plan and the operator channel in settings.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kairos_api import assistant_tools as tools
from kairos_api import break_store, channel_scope
from kairos_api.assistant_read_tools import SOURCE_BY_TOOL, _READ_EXECUTORS, execute_read_tool

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"
BREAK_TOOLS = ("get_day_breaks", "get_break")
UNPLANNED_DAY = "1999-01-01"


def _planned_day() -> str:
    days = break_store.plan_days()
    if not days:
        pytest.skip("no saved plan on the operator channel, so there is no day to open")
    return days[0]


def _rival_channels() -> set[str]:
    """Every channel in the saved weekly plan that the operator does not own.

    Derived rather than hardcoded, so the scan below tests the boundary against
    the names actually present in this product's data.
    """
    owned = channel_scope.operator_channel()
    frame = pd.read_csv(PLAN_PATH)
    channels = {str(value).strip() for value in frame["channel"].dropna().unique()}
    return {name for name in channels if name and name != owned}


def _names_found(payload: object, names: set[str]) -> set[str]:
    """Which of these names appear anywhere in a payload, serialized as the model
    would see it. Substring, not equality: a rival hidden inside a programme
    title or a reason would still be a rival reaching the assistant's context."""
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    return {name for name in names if name in blob}


# --- the registry -----------------------------------------------------------------
def test_both_break_tools_are_registered_with_a_named_source() -> None:
    for name in BREAK_TOOLS:
        assert name in tools.READ_TOOL_NAMES, name
        assert name in _READ_EXECUTORS, name
        # A tool whose source is not named is a tool whose answer cannot be
        # checked, so the catalogue entry must name the data behind it.
        assert "day plan" in SOURCE_BY_TOOL[name], name
    assert not (set(BREAK_TOOLS) & set(tools.PROPOSE_TOOL_NAMES))


def test_the_break_tool_schemas_describe_the_break_and_take_its_id() -> None:
    by_name = {schema["name"]: schema for schema in tools.READ_TOOL_SCHEMAS}
    listing = by_name["get_day_breaks"]
    assert set(listing["input_schema"]["properties"]) == {"day", "hour"}
    assert not listing["input_schema"].get("required")
    detail = by_name["get_break"]
    assert detail["input_schema"]["required"] == ["break_id"]
    for schema in (listing, detail):
        text = schema["description"]
        assert "break" in text
        assert "get_day_breaks" in text or "get_break" in text


# --- a planned day: the plan's own figures arrive intact ---------------------------
def test_a_planned_day_returns_every_break_with_the_plans_own_figures() -> None:
    from kairos_api.break_api import plan_day

    day = _planned_day()
    payload = execute_read_tool("get_day_breaks", {"day": day}, None)
    truth = plan_day(day=day)
    assert payload["available"] is True
    assert payload["day"] == day
    assert payload["breaks_on_the_day"] == len(truth["breaks"])
    assert payload["count"] == len(payload["breaks"])
    assert payload["breaks"], "a planned day must return the breaks it places"
    by_id = {record["break_id"]: record for record in truth["breaks"]}
    for row in payload["breaks"]:
        source = by_id[row["break_id"]]
        for key in ("ordinal", "breaks_in_segment", "hour", "start_seconds",
                    "duration_seconds", "offset_seconds", "is_gold",
                    "projected_revenue", "segment_retention", "programme"):
            # Identity, not approximate equality: the tool may not move a break
            # by a tenth of a second or a shekel on its way to the model.
            assert row[key] == source[key], (row["break_id"], key)
    # The day's own totals and its compliance verdict travel unchanged.
    assert payload["totals"] == truth["totals"]
    assert payload["compliance"] == truth["compliance"]
    # The revenue the plan credits the day is the sum of the breaks it credits,
    # recomputed from the rows the tool actually returned. The day board rounds
    # each break to the agora before serving it, so the sum of 80 rounded credits
    # sits within half an agora per break of the day's own total: measured 0.02
    # on the shipped day. Nothing here re-rounds anything on top of that.
    summed = sum(row["projected_revenue"] for row in payload["breaks"])
    assert abs(summed - float(truth["totals"]["revenue"])) <= 0.005 * len(payload["breaks"])


def test_the_basis_carries_both_the_live_plan_and_the_saved_weekly_one() -> None:
    """A day board is re-planned live and the weekly CSV is a different basis.

    Handing one of them over as though it were the other is how a reader ends up
    quoting a figure the saved plan does not hold, so both travel.
    """
    payload = execute_read_tool("get_day_breaks", {"day": _planned_day()}, None)
    basis = payload["basis"]
    assert basis["source"] and basis["channel"] == channel_scope.operator_channel()
    assert basis["committed"]["source"].endswith("the saved weekly plan")
    assert basis["committed"]["breaks"] is not None


def test_the_clock_beside_a_break_is_the_plans_own_rendering() -> None:
    """The scheduler is told to give times in the plan's own clock, and the day
    board serves seconds only. The inspector's own renderer is called rather than
    a second one, and the seconds travel beside it unrounded."""
    from kairos_api.break_api_detail import _clock

    payload = execute_read_tool("get_day_breaks", {"day": _planned_day()}, None)
    for row in payload["breaks"]:
        assert row["start_clock"] == _clock(row["start_seconds"])
        assert row["end_clock"] == _clock(row["start_seconds"] + row["duration_seconds"])


def test_the_hour_filter_narrows_to_that_hour_and_still_states_the_day() -> None:
    day = _planned_day()
    whole = execute_read_tool("get_day_breaks", {"day": day}, None)
    hour = whole["breaks"][0]["hour"]
    narrowed = execute_read_tool("get_day_breaks", {"day": day, "hour": hour}, None)
    assert narrowed["filtered_by"] == {"hour": hour}
    assert {row["hour"] for row in narrowed["breaks"]} == {hour}
    # The count of what was returned never hides how many the day carries.
    assert narrowed["breaks_on_the_day"] == whole["breaks_on_the_day"]
    assert narrowed["count"] == len(narrowed["breaks"]) < whole["count"]
    assert "error" in execute_read_tool("get_day_breaks", {"day": day, "hour": "noon"}, None)


def test_one_break_arrives_whole_with_its_money_reproducing_from_its_own_inputs() -> None:
    day = _planned_day()
    listing = execute_read_tool("get_day_breaks", {"day": day}, None)
    row = listing["breaks"][0]
    payload = execute_read_tool("get_break", {"break_id": row["break_id"]}, None)
    assert "error" not in payload
    assert payload["break_id"] == row["break_id"]
    assert payload["money"]["projected"]["amount"] == row["projected_revenue"]
    programme = payload["programme"]
    money = payload["money"]["projected"]
    # The stated basis reproduces the stated amount: rate per point times the
    # rating this break is priced at, times its length over the rate unit, times
    # the premium. If the tool ever passed a different rating through, this fails.
    amount = (programme["rate_per_point"] * money["rating_at_this_break"]
              * payload["placement"]["duration_seconds"] / programme["rate_unit_seconds"]
              * programme["premium"])
    assert abs(amount - money["amount"]) < 1.0
    assert payload["retention"]["sample_breaks"] >= 0
    assert payload["guardrails"]["hour_breaks"], "the hour a break sits in must open"


# --- honest absence: no plan, no break, no zero -----------------------------------
def test_delivered_money_stays_a_state_and_never_reads_as_zero() -> None:
    """The plan covers November 2024 and the one spot file covers 2025-04-27, so
    no planned break has a ledger behind it. That is a state of the data and an
    amount of 0.0 there would be a measurement nobody made."""
    payload = execute_read_tool("get_day_breaks", {"day": _planned_day()}, None)
    delivered = payload["delivered_money"]
    assert delivered["state"] in ("unavailable", "unknown", "real")
    if delivered["state"] != "real":
        assert delivered["amount"] is None
        assert delivered["reason"].strip() and delivered["reason_he"].strip()
    # Hoisted because it was byte-identical on every break of the day, not
    # dropped: no break carries a second, different delivered block.
    assert all("delivered" not in row for row in payload["breaks"])
    detail = execute_read_tool("get_break", {"break_id": payload["breaks"][0]["break_id"]}, None)
    assert detail["money"]["delivered"] == delivered


def test_a_day_the_plan_does_not_cover_is_an_honest_error_not_an_empty_day() -> None:
    payload = execute_read_tool("get_day_breaks", {"day": UNPLANNED_DAY}, None)
    assert payload["error"]
    assert "breaks" not in payload
    assert UNPLANNED_DAY not in payload["plan_days"]
    assert payload["plan_days"] == break_store.plan_days()[:20]


def test_an_unknown_break_id_is_an_honest_error_not_an_empty_break() -> None:
    payload = execute_read_tool("get_break", {"break_id": f"{UNPLANNED_DAY}|x|000~1"}, None)
    assert payload["error"]
    assert "money" not in payload and "placement" not in payload
    missing = execute_read_tool("get_break", {}, None)
    assert "get_day_breaks" in missing["error"]
    assert "money" not in missing
    malformed = execute_read_tool("get_break", {"break_id": "not-a-break"}, None)
    assert malformed["error"]


def test_a_break_whose_day_has_no_traffic_file_says_so_about_its_contents() -> None:
    day = _planned_day()
    listing = execute_read_tool("get_day_breaks", {"day": day}, None)
    contents = execute_read_tool("get_break", {"break_id": listing["breaks"][0]["break_id"]}, None)["contents"]
    assert contents["state"] in ("real", "unavailable")
    if contents["state"] != "real":
        assert contents["reason"].strip() and contents["reason_he"].strip()
        assert contents["path_forward"].strip()
    # The pod itself is never carried twice: get_pod reads it whole.
    assert "pod" not in contents


# --- the competitor boundary ------------------------------------------------------
def test_no_rival_channel_name_reaches_either_break_tool() -> None:
    rivals = _rival_channels()
    assert rivals, "the scan is vacuous unless the product's data really carries rivals"
    day = _planned_day()
    listing = execute_read_tool("get_day_breaks", {"day": day}, None)
    assert _names_found(listing, rivals) == set()
    assert listing["operator_channel"] == channel_scope.operator_channel()
    for row in listing["breaks"][:12]:
        detail = execute_read_tool("get_break", {"break_id": row["break_id"]}, None)
        assert _names_found(detail, rivals) == set(), row["break_id"]
        assert detail["identity"]["channel"] == channel_scope.operator_channel()
    # And the refusals carry no rival either, since they print the plan's days.
    assert _names_found(execute_read_tool("get_day_breaks", {"day": UNPLANNED_DAY}, None), rivals) == set()


def test_the_boundary_scan_bites_on_an_unscoped_source() -> None:
    """The positive control for the test above.

    A scan that can never fail proves nothing, so the same scan is pointed at the
    saved weekly plan read WITHOUT the channel scope, which is the very file the
    day plan is built from. It must find rivals there. If this ever passes empty,
    the scan above is asleep.
    """
    rivals = _rival_channels()
    unscoped = pd.read_csv(PLAN_PATH).head(500).to_dict("records")
    assert _names_found(unscoped, rivals), "the scan failed to flag an unscoped plan read"


def test_a_rival_channels_break_id_resolves_to_no_day_at_all() -> None:
    """Why the boundary holds by construction rather than by a filter.

    A day plan is built only for the operator's own channel, so a break id naming
    a rival is not filtered out of a result: it never resolves to a day. This
    builds such an id out of a real rival row in the plan file and asserts the
    tool refuses it, which is the premise the docstring states.
    """
    rivals = _rival_channels()
    frame = pd.read_csv(PLAN_PATH)
    rival = sorted(rivals)[0]
    rows = frame[frame["channel"].astype(str).str.strip() == rival]
    if rows.empty:
        pytest.skip("the plan file carries no rival row to build an id from")
    row = rows.iloc[0]
    break_id = f"{str(row['date']).strip()}|{rival}|{str(row['segment_id']).strip()}~1"
    payload = execute_read_tool("get_break", {"break_id": break_id}, None)
    assert payload["error"], "a rival's break id must not resolve"
    assert "money" not in payload and "placement" not in payload
