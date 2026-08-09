"""The assistant's two pod read tools, against the real traffic file on disk.

Kai is told, for the traffic-operator persona, to answer in breaks, spots and
durations and to be exact about seconds. Until these tools existed it had no way
to read a pod at all, so the only thing it could be exact from was the prompt's
own words. These tests hold the two tools to the surface they report:

* the day's arithmetic passes through per-second, never re-rounded and never
  re-derived here,
* a day with no traffic file behind it answers available false with the reason
  and an empty list rather than a fabricated pod,
* an absent length stays absent instead of reading as zero,
* and no rival channel name reaches either tool's output.

Nothing is mocked: the executors run their real code against
``data/daily_input`` and the saved weekly plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from kairos_api import assistant_tools as tools
from kairos_api import break_api_pod, channel_scope
from kairos_api.assistant_read_tools import SOURCE_BY_TOOL, _READ_EXECUTORS, execute_read_tool
from kairos_api.assistant_read_tools_pod import _figure

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "output" / "weekly_break_schedule.csv"
POD_TOOLS = ("get_break_pods", "get_pod")
UNCOVERED_DAY = "1999-01-01"


def _covered_day() -> str:
    days = break_api_pod.covered_days()
    if not days:
        pytest.skip("no traffic file on disk, so there is no covered day to read")
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
    title or a note would still be a rival reaching the assistant's context."""
    blob = json.dumps(payload, ensure_ascii=False, default=str)
    return {name for name in names if name in blob}


# --- the registry -----------------------------------------------------------------
def test_both_pod_tools_are_registered_with_a_named_source() -> None:
    for name in POD_TOOLS:
        assert name in tools.READ_TOOL_NAMES, name
        assert name in _READ_EXECUTORS, name
        # A tool whose source is not named is a tool whose answer cannot be
        # checked, so the catalogue entry must name the data behind it.
        assert "data/daily_input" in SOURCE_BY_TOOL[name], name
    assert not (set(POD_TOOLS) & set(tools.PROPOSE_TOOL_NAMES))


def test_the_pod_tool_schemas_describe_the_pod_and_take_its_id() -> None:
    by_name = {schema["name"]: schema for schema in tools.READ_TOOL_SCHEMAS}
    listing = by_name["get_break_pods"]
    assert "day" in listing["input_schema"]["properties"]
    assert not listing["input_schema"].get("required")
    detail = by_name["get_pod"]
    assert detail["input_schema"]["required"] == ["pod_id"]
    for schema in (listing, detail):
        text = schema["description"]
        assert "spot" in text and "second" in text
        assert "get_break_pods" in text or "get_pod" in text


# --- a covered day: the arithmetic arrives intact ---------------------------------
def test_a_covered_day_returns_its_pods_with_the_per_second_arithmetic() -> None:
    day = _covered_day()
    payload = execute_read_tool("get_break_pods", {"day": day}, None)
    truth = break_api_pod.list_pods(day=day)
    assert payload["available"] is True
    assert payload["day"] == day
    assert payload["count"] == truth["count"] == len(truth["pods"])
    assert payload["pods"], "a covered day must return the pods it declares"
    by_id = {pod["pod_id"]: pod for pod in truth["pods"]}
    for row in payload["pods"]:
        source = by_id[row["pod_id"]]
        arithmetic = source["arithmetic"]
        assert row["break_start_seconds"] == source["break_start_seconds"]
        assert row["spot_count"] == arithmetic["spot_count"]
        assert row["spots_missing_a_length"] == arithmetic["spots_missing_a_length"]
        for key in ("declared_load", "span", "unfilled", "gap_before_first_spot"):
            # Identity, not approximate equality: the tool may not move a
            # duration by a tenth of a second on its way to the model.
            assert row[key]["seconds"] == arithmetic[key]["seconds"], key
            assert row[key]["state"] == arithmetic[key]["state"], key
        assert row["gaps_between_spots"] == arithmetic["gaps_between_spots"]
        assert row["overlaps_between_spots"] == arithmetic["overlaps_between_spots"]
        assert row["against_declared"] == source["against_declared"]
        assert row["verification_error_count"] == source["verification"]["count"]
    # The prose each figure means travels once for the day rather than per pod,
    # and it is the pod math module's own wording.
    from kairos_api.break_api_pod_math import LOAD_BASIS

    assert payload["arithmetic_basis"]["declared_load"] == LOAD_BASIS


def test_one_pod_arrives_whole_with_its_spots_and_seconds() -> None:
    day = _covered_day()
    truth = max(break_api_pod.pods_for_day(day), key=lambda pod: pod["arithmetic"]["spot_count"])
    payload = execute_read_tool("get_pod", {"pod_id": truth["pod_id"]}, None)
    assert "error" not in payload
    assert payload["pod_id"] == truth["pod_id"]
    assert payload["break_start_clock"] == truth["break_start_clock"]
    assert len(payload["spots"]) == truth["arithmetic"]["spot_count"]
    lengths = [spot["duration"]["seconds"] for spot in payload["spots"]]
    assert all(value is None or value > 0 for value in lengths)
    # The pod's own sum, recomputed from the spots the tool actually returned.
    assert round(sum(value for value in lengths if value is not None), 1) == \
        payload["arithmetic"]["declared_load"]["seconds"]
    positions = {spot["position"]["kind"] for spot in payload["spots"]}
    assert positions <= {"ordinal", "last", "unpositioned", "unknown"}
    assert payload["verification"]["count"] == len(truth["verification"]["errors"])


# --- honest absence: no coverage, no pod, no zero ---------------------------------
def test_a_day_without_coverage_says_so_and_invents_nothing() -> None:
    payload = execute_read_tool("get_break_pods", {"day": UNCOVERED_DAY}, None)
    assert payload["available"] is False
    assert payload["pods"] == []
    assert payload["count"] == 0
    assert payload["reason"].strip() and payload["reason_he"].strip()
    assert payload["path_forward"].strip()
    assert UNCOVERED_DAY not in payload["covered_days"]
    # Nothing invented a duration or a spot count to fill the empty day.
    assert "arithmetic" not in payload
    assert all(key not in payload for key in ("spot_count", "declared_load"))


def test_an_unknown_pod_is_an_honest_error_not_an_empty_pod() -> None:
    payload = execute_read_tool("get_pod", {"pod_id": f"{UNCOVERED_DAY}~10:00:00"}, None)
    assert payload["error"]
    assert "spots" not in payload
    assert payload["covered_days"] == break_api_pod.covered_days()[:20]
    missing_id = execute_read_tool("get_pod", {}, None)
    assert "get_break_pods" in missing_id["error"]
    assert "spots" not in missing_id


def test_an_absent_figure_stays_absent_and_never_reads_as_zero() -> None:
    unknown = _figure({"state": "unknown", "seconds": None, "basis": "x", "basis_he": "y"})
    # None, not 0.0: a length nobody declared is missing, and a zero there would
    # understate a pod by exactly the seconds nobody declared.
    assert unknown == {"state": "unknown", "seconds": None}
    assert unknown["seconds"] is None
    assert _figure({"state": "real", "seconds": 34.0, "basis": "x"})["seconds"] == 34.0


# --- the competitor boundary ------------------------------------------------------
def test_no_rival_channel_name_reaches_either_pod_tool() -> None:
    rivals = _rival_channels()
    assert rivals, "the scan is vacuous unless the product's data really carries rivals"
    day = _covered_day()
    listing = execute_read_tool("get_break_pods", {"day": day}, None)
    assert _names_found(listing, rivals) == set()
    assert listing["channel"]["value"] == channel_scope.operator_channel()
    for pod in listing["pods"]:
        detail = execute_read_tool("get_pod", {"pod_id": pod["pod_id"]}, None)
        assert _names_found(detail, rivals) == set(), pod["pod_id"]
        assert detail["channel"]["value"] == channel_scope.operator_channel()
    # And the empty state carries no rival either, since it still prints a channel.
    assert _names_found(execute_read_tool("get_break_pods", {"day": UNCOVERED_DAY}, None), rivals) == set()


def test_the_boundary_scan_bites_on_an_unscoped_source() -> None:
    """The positive control for the test above.

    A scan that can never fail proves nothing, so the same scan is pointed at
    the saved weekly plan read WITHOUT the channel scope, which is exactly the
    unscoped source every operator surface is forbidden to serve. It must find
    rivals there. If this ever passes empty, the scan above is asleep.
    """
    rivals = _rival_channels()
    unscoped = pd.read_csv(PLAN_PATH).head(500).to_dict("records")
    assert _names_found(unscoped, rivals), "the scan failed to flag an unscoped plan read"


def test_the_traffic_file_carries_no_channel_column_at_all() -> None:
    """Why the pods are scoped by construction rather than by a filter.

    The channel beside a pod is the operator's own from settings; the file the
    pod is read from has no channel of any kind, so no rival can enter through
    it. This asserts that premise instead of trusting the docstring that states
    it.
    """
    paths = sorted(break_api_pod.DAILY_INPUT_DIR.glob("*.csv"))
    if not paths:
        pytest.skip("no traffic file on disk")
    for path in paths:
        columns = [str(column) for column in pd.read_csv(path, nrows=1).columns]
        assert not [name for name in columns if "channel" in name.lower() or "ערוץ" in name], path.name


def test_the_compliance_tool_gives_the_date_the_prompt_tells_kai_to_name():
    """A persona instructed to state a fact it is not given states it anyway.

    assistant_prompt.py tells the compliance owner's persona to name the profile
    AND its effective date. The read tool returned the profile and dropped the
    date, while the licence envelope has carried the profile, the date and the
    regulatory source URL in one dict the whole time.

    That gap is worse than a missing tool. A model with no tool refuses; a model
    with a tool that answers most of the question fills the rest in, and a
    confident wrong regulatory date is the kind of wrong that reaches a
    regulator.
    """
    from kairos_api.assistant_read_tools import _read_get_compliance

    payload = _read_get_compliance({})
    assert payload.get("profile"), "the licence profile is not named"
    assert payload.get("effective_date"), (
        "the compliance tool does not carry the effective date the prompt tells the "
        "assistant to name"
    )
    assert payload.get("source_url"), (
        "a regulatory claim a reader cannot follow to its source is worth less than one "
        "they can"
    )


def test_the_pricing_tool_says_whether_the_settlement_restatement_is_on():
    """Rule 17 tells Kai to state a flag the payload did not carry.

    The prompt says: "say whether the settlement restatement flag is on or off."
    The activation block carried show, position and ad_type and never
    qh_settlement, while PricingModel.enable_qh_settlement and
    qh_billing.qh_settlement_enabled both exist.

    Third instance of one shape in a day, after the compliance effective date and
    the pod itself: a persona instructed to state a fact it is not given states it
    anyway. This one is about money, because the restatement moves measured
    revenue by 7.45 percent when it is activated.
    """
    from kairos_api.assistant_read_tools import _read_get_pricing

    activation = _read_get_pricing({}).get("activation") or {}
    assert "qh_settlement" in activation, (
        "the pricing tool does not say whether the settlement restatement is on, and the "
        "prompt tells the assistant to state it"
    )
    assert isinstance(activation["qh_settlement"], bool), "the flag is a state, not a guess"
