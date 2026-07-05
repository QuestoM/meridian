"""Standing rule-conformance gate for the COMMITTED weekly plan.

Loads the real committed plan (output/weekly_break_schedule.csv) and the real
saved settings (data/kairos_settings.json) and proves, rule by rule, that the
plan conforms to every guardrail the settings promise.

Method (mirrors the ENGINE, not a guess): the CSV persists one row per
programme segment with the optimizer's decided num_breaks, but not the break
geometry. The engine lays k breaks through a segment evenly at
duration / (k + 1), clamped to the segment start, and maps each break to its
clock hour via int(start_seconds // 3600)
(kairos/optimize/_segment_math.py:_segment_break_objects, lines 117-169). So
this suite rebuilds the segments from the same source path the exporter uses
(kairos.data.transform.build_segments_from_programmes over the reference EPG),
joins them to the CSV rows by segment_id, reconstructs the exact Break objects
with the CSV's committed num_breaks / is_gold, and runs the engine's own
guardrail checks (kairos/optimize/guardrails.py) with guardrails mapped from
the saved settings by kairos.service.guardrails_from_settings.

Retention floor semantics (careful, per the engine): the floor binds on BREAK
objects only (guardrails.py:check_retention_floor iterates breaks), and a
0-break segment emits no Break, so it is exempt; its CSV row honestly reads the
full retention_baseline (1.0). The floor is therefore asserted on rows with
num_breaks >= 1, against predicted_retention and retention_used (equal when
risk_lambda is 0, the saved state).

A FAILING assertion here is a real conformance bug in the committed plan; per
QA mandate the failing test stays in place, clearly commented, until the lead
decides the fix. As of 2026-07-05 every rule passes (120 channel-days, zero
violations).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest

from kairos.data.loaders import load_programmes
from kairos.data.transform import build_segments_from_programmes
from kairos.model.impact import load_impact_model
from kairos.optimize._segment_math import _segment_break_objects, _segment_retention
from kairos.optimize.guardrails import (
    check_break_spacing,
    check_breaks_per_hour,
    check_daily_ad_load,
    check_gold_breaks,
    check_hourly_ad_load,
    check_retention_floor,
    _is_protected,
)
from kairos.optimize.overrides import GOLD, STATUS_ACTIVE, OverrideSet
from kairos.optimize.pricing import OptimizerAssumptions, pricing_from_settings
from kairos.service import (
    DEFAULT_IMPACT_MODEL_PATH,
    _apply_first_break_multiplier,
    _build_classifier,
    guardrails_from_settings,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "weekly_break_schedule.csv"
SETTINGS_PATH = ROOT / "data" / "kairos_settings.json"

SECONDS_PER_HOUR = 3600.0


@pytest.fixture(scope="module")
def settings() -> dict:
    assert SETTINGS_PATH.exists(), "saved settings file is missing"
    return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def guardrails(settings):
    return guardrails_from_settings(settings)


@pytest.fixture(scope="module")
def plan() -> pd.DataFrame:
    assert CSV_PATH.exists(), "committed weekly plan CSV is missing"
    frame = pd.read_csv(CSV_PATH, encoding="utf-8")
    assert not frame.empty, "committed weekly plan CSV is empty"
    return frame


@pytest.fixture(scope="module")
def reconstruction(plan, settings):
    """Rebuild engine segments and the committed plan's Break objects.

    Returns (breaks, joined) where breaks is the full list of engine Break
    objects reconstructed from the CSV decisions, and joined is a list of
    (csv_row, segment) pairs for per-row assertions. Fails loudly when the CSV
    and the current source data disagree on segment identity, because then the
    committed plan can no longer be checked against the engine's geometry.
    """
    pricing = pricing_from_settings(settings, None)
    assumptions = _apply_first_break_multiplier(OptimizerAssumptions())
    classifier = _build_classifier()
    impact_model = load_impact_model(DEFAULT_IMPACT_MODEL_PATH, assumptions=assumptions)
    programmes = load_programmes()

    breaks = []
    joined = []
    mismatches = []
    pairs = plan[["channel", "date"]].drop_duplicates().itertuples(index=False, name=None)
    for channel, day in pairs:
        segments = build_segments_from_programmes(
            programmes, classifier, pricing,
            assumptions=assumptions, impact_model=impact_model,
            channel=channel, day=day,
        )
        by_id = {s.segment_id: s for s in segments}
        rows = plan[(plan["channel"] == channel) & (plan["date"] == day)]
        if set(rows["segment_id"]) != set(by_id):
            mismatches.append((channel, day))
            continue
        for row in rows.itertuples(index=False):
            segment = by_id[row.segment_id]
            joined.append((row, segment))
            breaks.extend(
                _segment_break_objects(segment, int(row.num_breaks), is_gold=bool(row.is_gold))
            )
    if mismatches:
        pytest.fail(
            "committed CSV and current source data disagree on segment identity "
            f"for channel-days {mismatches[:5]} (plan is stale vs data); "
            "recompute before running the conformance gate"
        )
    return breaks, joined


def _fmt(violations) -> str:
    return "; ".join(
        f"{v.code} {v.scope} observed={v.observed} limit={v.limit}" for v in violations[:10]
    )


def test_csv_matches_current_engine_inputs(reconstruction):
    """The committed rows still reproduce from today's inputs.

    program_type must match the classifier's verdict and predicted_retention
    must recompute exactly from the segment's coefficients at the committed
    break count, proving the plan was built from the inputs on disk now
    (a drift here means the CSV predates a data/model/settings change).
    """
    _, joined = reconstruction
    type_mismatch = [
        (r.segment_id, r.program_type, s.program_type)
        for r, s in joined if str(r.program_type) != str(s.program_type)
    ]
    assert not type_mismatch, f"program_type drift on {type_mismatch[:5]}"
    retention_mismatch = [
        (r.segment_id, r.predicted_retention, round(_segment_retention(s, int(r.num_breaks)), 4))
        for r, s in joined
        if int(r.num_breaks) > 0
        and abs(round(_segment_retention(s, int(r.num_breaks)), 4) - r.predicted_retention) > 1e-9
    ]
    assert not retention_mismatch, f"retention drift on {retention_mismatch[:5]}"


def test_rule_a_breaks_per_broadcast_hour(reconstruction, guardrails, settings):
    """Rule (a): never more than max_breaks_per_hour breaks in any clock hour.

    Engine semantics: a break's hour is int(break_start // 3600) with the break
    start derived from even spacing (guardrails.py:89-103). Checked for every
    channel in the plan, which is stronger than the owned channel alone.
    """
    breaks, _ = reconstruction
    violations = check_breaks_per_hour(breaks, guardrails)
    assert not violations, _fmt(violations)
    counts: dict = defaultdict(int)
    for b in breaks:
        counts[(b.channel, b.day, b.hour)] += 1
    worst = max(counts.values())
    assert worst <= int(settings["max_breaks_per_hour"]), (
        f"an hour carries {worst} breaks, cap is {settings['max_breaks_per_hour']}"
    )


def test_rule_b_ad_seconds_per_hour(reconstruction, guardrails, settings):
    """Rule (b): ad seconds in any hour never exceed max_ad_minutes_per_hour * 60."""
    breaks, _ = reconstruction
    violations = check_hourly_ad_load(breaks, guardrails)
    assert not violations, _fmt(violations)
    seconds: dict = defaultdict(float)
    for b in breaks:
        seconds[(b.channel, b.day, b.hour)] += b.duration_seconds
    worst = max(seconds.values())
    assert worst <= float(settings["max_ad_minutes_per_hour"]) * 60 + 1e-9


def test_rule_c_protected_program_hours(reconstruction, guardrails):
    """Rule (c): an hour containing any protected-programme break respects the
    tighter protected cap. Mirrors the engine exactly: the protected flag is
    OR-ed per hour (guardrails.py:106-127), so a mixed hour is held to the
    protected limit."""
    breaks, _ = reconstruction
    seconds: dict = defaultdict(float)
    protected: dict = defaultdict(bool)
    for b in breaks:
        key = (b.channel, b.day, b.hour)
        seconds[key] += b.duration_seconds
        protected[key] = protected[key] or _is_protected(b.program_type, guardrails)
    protected_hours = [key for key in seconds if protected[key]]
    assert protected_hours, "no protected-programme hours found; protected rule untested"
    over = [
        (key, seconds[key]) for key in protected_hours
        if seconds[key] > guardrails.protected_max_ad_seconds_per_hour + 1e-9
    ]
    assert not over, f"protected hours over the {guardrails.protected_max_ad_seconds_per_hour}s cap: {over[:5]}"


def test_rule_d_daily_ad_load(reconstruction, guardrails, plan, settings):
    """Rule (d): total ad seconds per channel-day never exceed the daily cap.

    Checked twice: on the reconstructed engine breaks and directly on the CSV's
    own total_break_time column, so the persisted totals and the geometry agree.
    """
    breaks, _ = reconstruction
    violations = check_daily_ad_load(breaks, guardrails)
    assert not violations, _fmt(violations)
    limit = float(settings["max_daily_ad_minutes"]) * 60
    daily = plan.groupby(["channel", "date"])["total_break_time"].sum()
    over = daily[daily > limit + 1e-9]
    assert over.empty, f"channel-days over the daily cap: {over.head().to_dict()}"


def test_rule_e_retention_floor(reconstruction, guardrails, plan, settings):
    """Rule (e): every segment that carries breaks keeps retention at or above
    the floor. The floor binds per Break on the segment's realised retention
    (guardrails.py:75-86); 0-break segments emit no Break and are exempt,
    reading the full baseline (1.0) in the CSV."""
    breaks, _ = reconstruction
    violations = check_retention_floor(breaks, guardrails)
    assert not violations, _fmt(violations)
    floor = float(settings["min_retention_floor"])
    with_breaks = plan[plan["num_breaks"] >= 1]
    below_predicted = with_breaks[with_breaks["predicted_retention"] < floor - 1e-9]
    assert below_predicted.empty, (
        f"rows below the floor on predicted_retention: "
        f"{below_predicted[['segment_id', 'num_breaks', 'predicted_retention']].head().to_dict('records')}"
    )
    below_used = with_breaks[with_breaks["retention_used"] < floor - 1e-9]
    assert below_used.empty, (
        f"rows below the floor on retention_used: "
        f"{below_used[['segment_id', 'num_breaks', 'retention_used']].head().to_dict('records')}"
    )
    zero_break = plan[plan["num_breaks"] == 0]
    assert (zero_break["predicted_retention"] == 1.0).all(), (
        "0-break rows must read the honest full baseline retention (1.0)"
    )


def test_rule_engine_bonus_break_spacing(reconstruction, guardrails):
    """The engine also enforces min_break_spacing_seconds end-to-start within a
    channel-day (guardrails.py:130-147); the committed plan must respect it."""
    breaks, _ = reconstruction
    violations = check_break_spacing(breaks, guardrails)
    assert not violations, _fmt(violations)


def test_rule_f_gold_breaks_per_day(reconstruction, guardrails, plan, settings):
    """Rule (f): gold breaks per channel-day never exceed gold_breaks_max_per_day."""
    breaks, _ = reconstruction
    violations = check_gold_breaks(breaks, guardrails)
    assert not violations, _fmt(violations)
    gold = plan[plan["is_gold"] == True]  # noqa: E712 - CSV bool column
    if not gold.empty:
        per_day = gold.groupby(["channel", "date"]).size()
        cap = int(settings["gold_breaks_max_per_day"])
        over = per_day[per_day > cap]
        assert over.empty, f"channel-days over the gold cap: {over.to_dict()}"


def test_rule_g_num_breaks_bounds(plan):
    """Rule (g): num_breaks is never negative and never above the engine cap.

    The cap the exporter builds segments with is
    OptimizerAssumptions().default_max_breaks (kairos/data/transform.py:386);
    the totals column must equal num_breaks * break_length exactly.
    """
    cap = OptimizerAssumptions().default_max_breaks
    assert (plan["num_breaks"] >= 0).all(), "negative num_breaks in the committed plan"
    over = plan[plan["num_breaks"] > cap]
    assert over.empty, (
        f"rows above the engine max_breaks={cap}: "
        f"{over[['segment_id', 'num_breaks']].head().to_dict('records')}"
    )
    assert (plan["break_length"] > 0).all(), "non-positive break_length"
    mismatch = plan[
        (plan["num_breaks"] * plan["break_length"] - plan["total_break_time"]).abs() > 0.05
    ]
    assert mismatch.empty, (
        f"total_break_time disagrees with num_breaks * break_length: "
        f"{mismatch[['segment_id', 'num_breaks', 'break_length', 'total_break_time']].head().to_dict('records')}"
    )


def test_rule_h_segment_id_unique_and_well_formed(plan):
    """Rule (h): segment_id is unique and formatted date|channel|index, with the
    date and channel fields agreeing with the row's own columns (the exporter
    writes f"{date}|{channel}|{index:03d}", kairos/data/transform.py:369)."""
    dupes = plan[plan["segment_id"].duplicated(keep=False)]
    assert dupes.empty, f"duplicate segment_ids: {sorted(set(dupes['segment_id']))[:5]}"
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2})\|(.+)\|(\d{3,})$")
    for row in plan.itertuples(index=False):
        match = pattern.match(str(row.segment_id))
        assert match, f"malformed segment_id {row.segment_id!r}"
        assert match.group(1) == str(row.date), (
            f"segment_id date part disagrees with row date: {row.segment_id!r} vs {row.date!r}"
        )
        assert match.group(2) == str(row.channel), (
            f"segment_id channel part disagrees with row channel: {row.segment_id!r}"
        )


def test_rule_i_gold_only_where_engine_could_place_it(plan):
    """Rule (i): is_gold appears only where the engine could actually have
    placed gold. The optimizer emits gold placements only from a segment's own
    is_gold flag (never set by the transform), an ACTIVE segment-scope gold
    override, or a gold placement pin (kairos/optimize/optimizer.py:394,
    kairos/optimize/_segment_math.py:139). With both stores empty (the current
    committed state) the plan must carry zero gold rows; with stored gold
    overrides the gold rows must be a subset of their targets."""
    overrides = OverrideSet.from_csv()
    gold_targets = {
        o.target_id
        for o in overrides.overrides
        if o.scope == "segment"
        and (o.kind == GOLD or o.gold)
        and o.status == STATUS_ACTIVE
        and o.is_valid()
    }
    constraints_path = ROOT / "data" / "kairos_constraints.csv"
    has_constraint_rows = False
    if constraints_path.exists():
        constraint_frame = pd.read_csv(constraints_path, encoding="utf-8-sig")
        has_constraint_rows = not constraint_frame.empty
    gold_rows = plan[plan["is_gold"] == True]  # noqa: E712
    if not gold_targets and not has_constraint_rows:
        assert gold_rows.empty, (
            "plan marks segments gold with no gold override and no constraints on disk: "
            f"{gold_rows['segment_id'].head().tolist()}"
        )
    else:
        stray = gold_rows[~gold_rows["segment_id"].isin(gold_targets)]
        # Constraint-pinned gold cannot be attributed row-by-row without a full
        # resolve; only fail when there are no constraint rows to explain a stray.
        if not has_constraint_rows:
            assert stray.empty, (
                f"gold rows with no gold override behind them: {stray['segment_id'].head().tolist()}"
            )


def test_plan_covers_owned_channel(plan, settings):
    """The committed plan must include the operator's own channel, otherwise
    every owned-channel surface (segments, recommendations, inspector) is empty."""
    owned = str(settings.get("operator_channel", "")).strip()
    assert owned, "operator_channel is not configured in the saved settings"
    assert owned in set(plan["channel"].astype(str)), (
        f"committed plan has no rows for the owned channel {owned!r}"
    )
