"""P2: the comparison can be acted on, and the day it turns on can be opened.

The comparison used to end in prose. A planner who read that scenario B is worth
1.25M less in net had two dead ends in front of them: no control anywhere turned
the winning leg into the plan's objective, so the five lever values had to be
carried across two steps by eye and set again by hand, and the per-day table
that names the day the choice turns on rendered each date as plain text, so the
most actionable sentence on the destination opened nothing.

Both are closed here and both halves are checked: the act on the surface, and the
route that makes the second one true rather than approximate. A day row that
opened a board showing a different day would be worse than no control at all, so
``/api/schedule`` takes the date and answers with that date or says why it
cannot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
WEEK = ROOT / "tv-break-dashboard" / "src" / "plan" / "week"

# The four objective levers a comparison leg runs under. The licence cap still
# reaches both engine legs, but is deliberately not adopted or generically saved.
LEVER_PAIRS = (
    ("revenue_weight", "revenue_weight"),
    ("retention_floor", "min_retention_floor"),
    ("risk_lambda", "risk_lambda"),
    ("objective_mode", "objective_mode"),
)


@pytest.fixture()
def client():
    from kairos_api.server import app

    return TestClient(app)


def _text(name: str) -> str:
    return (WEEK / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------- the route


def test_the_week_payload_says_which_broadcast_day_its_board_stands_on(client):
    """The day zoom has always drawn one day and never named it."""
    body = client.get("/api/schedule").json()
    board = body["board"]
    assert board["requested"] is None
    dates = {str(item["date"]) for item in body["break_operations"]["programs"]}
    if not dates:
        pytest.skip("no programme source on this tree, so there is no board to name")
    assert len(dates) == 1, dates
    assert board["date"] == dates.pop()
    assert board["available"] is True
    assert board["programmes"] == len(body["break_operations"]["programs"])
    assert board["breaks"] == len(body["break_operations"]["breaks"])
    # And where that day sits in the source, so the day is placed rather than
    # floating: measured 30 broadcast days, 2024-11-01 to 2024-11-30.
    covers = board["covers"]
    assert covers["n_dates"] >= 1
    assert covers["date_from"] <= board["date"] <= covers["date_to"]


def test_a_day_that_was_asked_for_is_the_day_that_comes_back(client):
    """The whole point of the control: the day the comparison names is the day
    the board draws, not the day the source happens to start on."""
    week = client.get("/api/schedule").json()
    default_day = week["board"]["date"]
    if not default_day:
        pytest.skip("no programme source on this tree")
    covers = week["board"]["covers"]
    wanted = covers["date_to"]
    assert wanted != default_day, "the source carries more than one broadcast day"

    body = client.get("/api/schedule", params={"date": wanted}).json()
    board = body["board"]
    assert board["requested"] == wanted
    assert board["date"] == wanted
    assert board["available"] is True
    programmes = body["break_operations"]["programs"]
    assert programmes, "a day that is in the source draws something"
    assert {str(item["date"]) for item in programmes} == {wanted}
    assert {str(item["date"]) for item in body["break_operations"]["breaks"]} <= {wanted}


def test_the_competitor_boundary_holds_on_a_day_board_too(client):
    """Scoping by date must not re-widen the channel scope."""
    settings = client.get("/api/settings").json()
    owned = str(settings.get("operator_channel") or "").strip()
    if not owned:
        pytest.skip("no operator channel is configured on this tree")
    week = client.get("/api/schedule").json()
    wanted = week["board"]["covers"]["date_to"]
    body = client.get("/api/schedule", params={"date": wanted}).json()
    channels = {str(item["channel"]) for item in body["break_operations"]["programs"]}
    channels |= {str(item["channel"]) for item in body["break_operations"]["breaks"]}
    assert channels <= {owned}, channels


def test_every_day_the_comparison_can_name_is_a_day_that_opens(client):
    """The comparison runs the plan's own week and prints one row per date, so
    every one of those dates has to open. Measured over the seven."""
    progress = client.get("/api/plan-progress").json()
    window = progress.get("window") or {}
    if not progress.get("available") or not window.get("date_from"):
        pytest.skip("no plan week on this tree, so the comparison has no dates to name")
    from datetime import date, timedelta

    start = date.fromisoformat(window["date_from"])
    end = date.fromisoformat(window["date_to"])
    day = start
    opened = []
    while day <= end:
        body = client.get("/api/schedule", params={"date": day.isoformat()}).json()
        board = body["board"]
        assert board["requested"] == day.isoformat()
        if board["available"]:
            assert board["date"] == day.isoformat()
            assert board["programmes"] > 0
            opened.append(day.isoformat())
        else:
            # An honest refusal is allowed; a wrong day is not.
            assert board["date"] is None
            assert board["reason_code"]
            assert body["break_operations"]["programs"] == []
        day += timedelta(days=1)
    assert len(opened) == window["n_dates"], f"only {len(opened)} of the week's days open"


def test_a_day_the_source_does_not_carry_says_so_and_borrows_nothing(client):
    body = client.get("/api/schedule", params={"date": "2099-12-31"}).json()
    board = body["board"]
    assert board["available"] is False
    assert board["date"] is None
    assert board["reason_code"] == "date_not_in_programme_source"
    assert "2099-12-31" in board["reason"]
    assert board["programmes"] == 0 and board["breaks"] == 0
    assert body["break_operations"]["programs"] == []
    assert body["break_operations"]["breaks"] == []


def test_an_unreadable_date_is_refused_rather_than_guessed(client):
    body = client.get("/api/schedule", params={"date": "tuesday"}).json()
    assert body["board"]["reason_code"] == "unreadable_date"
    assert body["board"]["available"] is False
    assert body["break_operations"]["programs"] == []


def test_the_route_keeps_its_shape_for_a_caller_that_sends_no_date(client):
    """Section 9 item 6: the address, the method and the response shape stand."""
    body = client.get("/api/schedule").json()
    for key in ("rows", "break_operations", "break_schedule", "break_schedule_total_rows", "scope"):
        assert key in body, key
    dated = client.get("/api/schedule", params={"date": body["board"]["covers"]["date_to"]}).json()
    assert set(dated) == set(body), set(dated) ^ set(body)
    # The week canvas and the plan slice are the week's on both, because only
    # the embedded day board is scoped by the date.
    assert dated["break_schedule_total_rows"] == body["break_schedule_total_rows"]
    assert dated["scope"] == body["scope"]


# ------------------------------------------------------------- the adoption


def test_the_objective_levers_map_without_adopting_the_licence_guardrail():
    model = _text("plan-week-model.js")
    block = model.split("export const ADOPT_FIELDS")[1].split("];")[0]
    for source, target in LEVER_PAIRS:
        assert f"['{source}', '{target}']" in block, (source, target)
    assert len(re.findall(r"\['", block)) == len(LEVER_PAIRS)
    assert "max_breaks_per_hour" not in block


def test_a_partial_lever_set_is_refused_rather_than_half_adopted():
    """An objective that holds four of the five values the card was priced on
    is a different objective wearing the card's number."""
    model = _text("plan-week-model.js")
    body = model.split("export function objectiveFromLevers(levers)")[1].split("\n}")[0]
    assert "if (value === null || value === undefined || value === '') return null;" in body


def test_each_scenario_card_offers_to_become_the_objective_in_both_languages():
    adopt = _text("ScenarioAdopt.jsx")
    assert "Use scenario ${letter} as the objective" in adopt
    assert "קביעת תרחיש ${letter} כמטרה" in adopt
    # The card prints the four objective values the control would write, so the act names
    # its own consequence before it happens.
    assert "ADOPT_FIELDS.map(([from, to])" in adopt
    assert "leverValueText(to, summary.levers[from], locale)" in adopt
    # A run that did not report its levers offers no control and says why.
    assert "objectiveFromLevers(summary?.levers)" in adopt
    assert "This run did not report the full objective lever set" in adopt

    panel = _text("ComparePanel.jsx")
    assert '<ScenarioAdopt leg="a" summary={a} locale={locale} onAdopt={onAdopt} />' in panel
    assert '<ScenarioAdopt leg="b" summary={b} locale={locale} onAdopt={onAdopt} />' in panel


def test_the_values_written_are_the_ones_the_run_reported_not_the_form_s():
    """The form can be moved after a run, and the money on the card belongs to
    the levers the run used."""
    surface = _text("use-plan-surface.js")
    body = surface.split("const adoptLeg = useCallback")[1].split("}, [comparePayload")[0]
    assert "comparePayload?.b : comparePayload?.a" in body
    assert "objectiveFromLevers(summary?.levers)" in body
    assert "if (!values) return false;" in body
    # It moves the draft and nothing else. No save, no run, no write.
    assert "setDraft((current) => ({ ...current, ...values }));" in body
    for forbidden in ("saveSettings", "api.", "runWeeklyPlan"):
        assert forbidden not in body, forbidden


def test_adopting_lands_on_step_one_and_both_doors_run_the_same_function():
    week = _text("PlanWeek.jsx")
    assert "if (surface.adoptLeg(leg)) go('objective');" in week
    assert "onAdopt={adoptLeg}" in week
    assert "adoptLeg," in _text("plan-week-commands.js")


def test_the_palette_carries_the_act_with_the_chord_that_fires_it():
    commands = _text("plan-week-commands.js")
    assert "id: `adopt-${leg}`" in commands
    assert "shortcut: ['u', leg]" in commands
    assert "disabled: !surface.adoptable(leg)" in commands
    assert "no finished comparison has reported its levers" in commands
    assert "אין השוואה שהסתיימה ודיווחה את הידיות שלה" in commands

    keyboard = _text("use-plan-keyboard.js")
    # The lead is read off the command list, so a chord printed in the palette
    # is a chord the keyboard starts. A hard-coded lead let a new chord print
    # and never fire.
    assert "export function chordLeads(commands)" in keyboard
    assert "chordLeads(latest.current).has(key)" in keyboard
    assert "key === 'g'" not in keyboard


def test_no_single_key_shortcut_collides_with_a_chord_lead():
    """A lead swallows the key it leads with, so a one-key command on the same
    letter would be unreachable."""
    commands = _text("plan-week-commands.js")
    shortcuts = re.findall(r"shortcut: \[([^\]]+)\]", commands)
    leads, singles = set(), set()
    for entry in shortcuts:
        parts = [part.strip().strip("'\"") for part in entry.split(",")]
        if len(parts) == 2 and parts[0] != "mod":
            leads.add(parts[0])
        if len(parts) == 1:
            singles.add(parts[0])
    assert leads, shortcuts
    assert leads & singles == set(), leads & singles


def test_the_banner_prints_every_old_value_beside_its_new_one():
    panel = _text("ObjectivePanel.jsx")
    assert "export function objectiveChanges(draft, saved, locale)" in panel
    assert "was: leverValueText(field, saved[field], locale)" in panel
    assert "next: leverValueText(field, draft[field], locale)" in panel
    assert "Saved now" in panel and "שמור כרגע" in panel
    assert "After this change" in panel and "אחרי השינוי" in panel
    # All four objective fields, never the licence cap.
    fields = panel.split("const OBJECTIVE_FIELDS = [")[1].split("]")[0]
    for _, target in LEVER_PAIRS:
        assert f"'{target}'" in fields, target
    assert "max_breaks_per_hour" not in fields
    # Where the values came from, when they came from the comparison.
    assert "These values are scenario ${adoptedLetter} of the comparison, exactly as it ran." in panel


def test_the_licence_cap_is_read_only_and_identical_in_both_scenario_legs():
    form = _text("ScenarioLegForm.jsx")
    cap = form.split("label={leverLabel('max_breaks_per_hour'", 1)[1].split("</Row>", 1)[0]
    assert "Licence guardrail · identical in both scenarios" in cap
    assert "aria-readonly=\"true\"" in cap
    assert "onChange" not in cap and "<Slider" not in cap

    surface = _text("use-plan-surface.js")
    assert "guardedLegA = legA && saved ? { ...legA, max_breaks_per_hour: saved.max_breaks_per_hour }" in surface
    assert "guardedLegB = legB && saved ? { ...legB, max_breaks_per_hour: saved.max_breaks_per_hour }" in surface
    assert "if (field === 'max_breaks_per_hour') return;" in surface


def test_the_adoption_can_be_put_back():
    panel = _text("ObjectivePanel.jsx")
    assert "Put the saved values back" in panel
    assert "החזרת הערכים השמורים" in panel
    surface = _text("use-plan-surface.js")
    revert = surface.split("const revertDraft = useCallback")[1].split("}, [saved]")[0]
    assert "setDraft(objectiveOf(saved));" in revert
    assert "setAdopted(null);" in revert


# --------------------------------------------------------------- the day row


def test_every_day_row_and_the_sentence_that_names_the_day_open_it():
    table = _text("CompareWeekTable.jsx")
    assert "function DayOpener(" in table
    assert "onClick={() => onOpenDay(date)}" in table
    assert "<DayOpener date={row.date} weekday={row.weekday} locale={locale} onOpenDay={onOpenDay}>" in table
    # The sentence the destination is judged on is itself the control.
    assert "<DayOpener date={turnRow.date} weekday={turnRow.weekday} locale={locale} onOpenDay={onOpenDay}>" in table
    # The label names the day through formatDay, the single home for reading a
    # calendar day, so an Israeli operator hears 03/11/2024 rather than the
    # machine's own ISO string. It said ${date} until the dates law landed.
    assert "Open ${formatDay(date)} on the week board" in table
    assert "פתיחת ${formatDay(date)} בלוח השבוע" in table
    assert "${date}" not in table, "a raw payload date is still interpolated somewhere here"

    panel = _text("ComparePanel.jsx")
    assert panel.count("onOpenDay={onOpenDay}") == 2, "both the live table and the finished one"


def test_opening_a_day_sets_the_day_zoom_and_the_date_together():
    week = _text("PlanWeek.jsx")
    opener = week.split("const openBoardDay = useCallback")[1].split("}, [go]")[0]
    assert "setBoardDate(String(date));" in opener
    assert "setBoardView('day');" in opener
    assert "go('board');" in opener
    assert "onOpenDay={openBoardDay}" in week
    # The canonical day workbench owns subsequent day selection and keeps the
    # addressed day coupled to the full editor rather than a second mini-board.
    panel = _text("BoardPanel.jsx")
    assert "<PlanBoardWorkbench" in panel
    assert "focusDate={focusDate}" in panel
    assert "onFocusDateChange={onFocusDateChange}" in panel
    workbench = _text("PlanBoardWorkbench.jsx")
    assert "onFocusDateChange?.(nextDay);" in workbench
    assert "<DayPicker" in workbench and "onChange={selectDay}" in workbench


def test_the_board_draws_the_day_it_was_asked_for_and_never_a_nearby_one():
    board = _text("BoardPanel.jsx")
    assert "<PlanBoardWorkbench" in board
    assert "focusDate={focusDate}" in board

    workbench = _text("PlanBoardWorkbench.jsx")
    assert "const preferredDay = focusDate" in workbench
    assert "day={day}" in workbench

    day_board = (ROOT / "tv-break-dashboard" / "src" / "plan" / "day" / "DayBoard.jsx").read_text()
    assert "const payload = await fetchDay(targetDay);" in day_board
    assert "if (wantedDayRef.current !== targetDay) return null;" in day_board

    actions = (ROOT / "tv-break-dashboard" / "src" / "plan" / "day" / "day-board-actions.js").read_text()
    assert "/api/plan/day?day=${encodeURIComponent(day)}" in actions


def test_the_reason_a_day_cannot_be_drawn_reaches_the_operator_in_both_languages():
    model = _text("plan-week-model.js")
    block = model.split("const BOARD_REASONS = {")[1].split("\n};")[0]
    for code in ("date_not_in_programme_source", "no_programme_in_source", "unreadable_date"):
        assert code in block, code
    assert "מקור התוכניות אינו כולל" in block
    # An unknown code falls back to the server's own prose rather than silence.
    fallback = model.split("export function boardReason(board, locale)")[1].split("\n}")[0]
    assert "return board.reason || null;" in fallback
