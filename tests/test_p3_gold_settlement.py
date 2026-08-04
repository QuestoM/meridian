"""What a gold change costs, held on the surface that performs it.

The day board has four acts that write, and three of them settled: they hold the
totals the day had, read the day they got, and print the difference beside the
prediction that was on screen. The gold act did not, on the written reasoning
that it changes which breaks are premium rather than where they sit, so the day
it re-reads is the whole answer.

Measured through the routes on ``רשת 13 / 2024-11-01``, that reasoning does not
survive contact. One gold mark on ``001~1`` moves the day from 1,062,669.88 to
1,028,205.58, which is 34,464.30 ILS and 3.24 per cent of the day, and takes it
from 80 breaks to 79. The re-read day is the whole answer only to a person who
memorised the number it replaced, and the change tile beside it re-bases itself
to zero. So the money the act spent was on no surface, and the only trace was a
toast counting how many breaks turned gold.

The inverse was in the same state. It exists and it is exact, measured both
directions on the same day: mark then clear returns the day to 1,062,669.88 and
to 80 breaks, to the cent and to the break. It was simply not offered, because
the panel that offers an inverse never opened.

This file holds both halves. The money is measured on the engine through its own
routes, the settlement is computed by the shipped classifier, and the shipped
panel is rendered in node and read for the figure and the control, so what is
asserted here is what an operator sees rather than a description of it. The last
test holds the invariant the drift came from: every act in the writes module goes
through the one settlement seam.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from urllib.parse import quote

import pytest

# The same fixtures the save settles under, so both acts are measured one way:
# every store below writes into a temporary directory, and the operator channel
# is declared when the shared settings file has lost it.
from test_p3_save_settlement import (  # noqa: F401
    client,
    isolated,
    opened_day,
    owned_channel,
)

pytestmark = pytest.mark.realdata

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "tv-break-dashboard"
SRC = DASHBOARD / "src"


def read(relative: str) -> str:
    return (SRC / relative).read_text(encoding="utf-8")


def render_panel(payload: dict, locale: str, can_undo: bool) -> dict:
    """Run the shipped classifier and render the shipped panel, in node.

    Both modules the operator's browser loads are executed here: the settlement
    is built by ``day-board-settlement.js`` and handed to ``DayBoardSettlement``
    itself, rendered through React. A test that read the source for a heading
    would prove a string exists, not that the panel prints it, and the defect
    this file closes was exactly a panel that never opened.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH, so the shipped panel cannot be rendered")
    if not (DASHBOARD / "node_modules" / "vite").exists():
        pytest.skip("the dashboard's dependencies are not installed, so the panel cannot be rendered")
    script = """
      import { runnerImport } from 'vite';
      import React from 'react';
      import { renderToStaticMarkup } from 'react-dom/server';
      const model = await runnerImport('./src/plan/day/day-board-settlement.js');
      const panel = await runnerImport('./src/plan/day/DayBoardSettlement.jsx');
      const input = JSON.parse(process.env.P3_PANEL_INPUT);
      const settlement = model.module.settlementOf(input.settlement);
      const html = renderToStaticMarkup(React.createElement(panel.module.default, {
        settlement,
        locale: input.locale,
        canUndo: input.canUndo,
        onUndo: () => {},
        onDismiss: () => {},
      }));
      process.stdout.write(JSON.stringify({ settlement, html }));
    """
    environment = dict(os.environ)
    environment["P3_PANEL_INPUT"] = json.dumps(
        {"settlement": payload, "locale": locale, "canUndo": can_undo}
    )
    result = subprocess.run(
        ["node", "--input-type=module", "-e", script],
        capture_output=True, text=True, timeout=300, cwd=DASHBOARD, env=environment,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def shekels(value: float) -> str:
    """The whole-shekel group the panel's own formatter prints for this figure."""
    whole = Decimal(str(abs(float(value)))).quantize(Decimal(1), rounding=ROUND_HALF_UP)
    return f"{int(whole):,}"


def crowded_break(board: dict) -> dict:
    """A break in the programme carrying most of them.

    Gold is carried on the programme segment, so a programme with several breaks
    is where a mark has something to reach. This is the same choice the save
    settlement test makes, for the same reason.
    """
    counts: dict[str, int] = {}
    for row in board["breaks"]:
        counts[row["segment_id"]] = counts.get(row["segment_id"], 0) + 1
    segment_id = max(counts, key=lambda key: counts[key])
    return next(row for row in board["breaks"] if row["segment_id"] == segment_id)


def test_a_gold_act_puts_the_money_it_moved_on_screen_with_a_way_back(client, opened_day):
    """The act, the figure it moved, and the control that reverses it.

    Driven through the routes the board calls, classified by the module the board
    imports, and rendered by the panel the board mounts. Measured on
    ``רשת 13 / 2024-11-01`` while writing this: 34,464.30 ILS and one break, both
    of them on screen afterwards, with the inverse enabled beside them.
    """
    from kairos_api.break_api import _gold_enabled

    if not _gold_enabled():
        pytest.skip("gold breaks are switched off in settings, so the act is refused by design")
    day = opened_day["day"]
    before = opened_day["totals"]
    target = crowded_break(opened_day)

    marked = client.post(f"/api/breaks/{quote(target['break_id'], safe='')}/gold")
    assert marked.status_code == 201
    if not marked.json()["bound"]:
        pytest.skip(f"the mark reached no break in the plan: {marked.json()['reason']}")

    after_board = client.get("/api/plan/day", params={"day": day}).json()
    after = after_board["totals"]
    realised = round(after["revenue"] - before["revenue"], 2)

    rendered = render_panel(
        {
            "act": "gold",
            "basis": after_board["basis"],
            "before": before,
            "after": after,
            "beforeBreaks": opened_day["breaks"],
            "afterBreaks": after_board["breaks"],
            "predictedRevenue": None,
        },
        locale="he",
        can_undo=True,
    )
    settlement = rendered["settlement"]
    html = rendered["html"]

    # The classifier: a gold change makes no prediction, so it is settled against
    # none rather than against an invented zero.
    assert settlement["act"] == "gold"
    assert settlement["predicted"] is None
    assert settlement["verdict"] == "unknown"
    assert settlement["realised"]["revenue"] == pytest.approx(realised, abs=0.01)

    # The panel: the act is named, the figure is a currency figure, and the two
    # break counts the day moved between are both printed.
    assert "מה שינוי הזהב עשה לתוכנית" in html, "the heading has to name the act that spent the money"
    assert "₪" in html, "a money figure without its currency is not a money figure"
    assert shekels(realised) in html, f"the panel does not carry the {realised} the engine moved"
    assert f">{before['breaks']}<" in html and f">{after['breaks']}<" in html
    if abs(realised) > 0.005:
        assert "is-loss" in html if realised < 0 else "is-gain" in html
    else:
        assert "is-flat" in html, "an unchanged day is flat, not a gain"

    # The control: present, named, and enabled.
    actions = html.split("day-readout-actions")[1]
    assert "ביטול שינוי הזהב" in actions
    assert "disabled" not in actions, "the way back is offered or the money is one way"

    # And the same panel with nothing to reverse leaves it disabled, so the
    # control above is bound to a real condition rather than always drawn on.
    off = render_panel(
        {
            "act": "gold",
            "basis": after_board["basis"],
            "before": before,
            "after": after,
            "beforeBreaks": opened_day["breaks"],
            "afterBreaks": after_board["breaks"],
            "predictedRevenue": None,
        },
        locale="he",
        can_undo=False,
    )
    assert "disabled" in off["html"].split("day-readout-actions")[1]

    # The inverse itself, taken by the break id the act named, on the engine.
    cleared = client.delete(f"/api/breaks/{quote(target['break_id'], safe='')}/gold")
    assert cleared.status_code == 200
    restored = client.get("/api/plan/day", params={"day": day}).json()["totals"]
    assert restored["revenue"] == pytest.approx(before["revenue"], abs=0.005)
    assert restored["breaks"] == before["breaks"]
    assert restored["gold_breaks"] == before["gold_breaks"]


def test_the_inverse_holds_when_the_gold_act_removes_the_break_it_was_performed_on(client, opened_day):
    """A gold mark can take away the chip that would have offered the way back.

    Marking makes the engine plan the day again with the mark in force, and the
    second plan chooses how many breaks each programme gets. Measured on
    ``רשת 13 / 2024-11-01``: marking ``001~4`` gold takes the day from 80 breaks
    to 79 and that id is one of the two that stop existing.

    So the inverse is held by the break id the act named rather than read off a
    chip, and this drives the case rather than hoping for it.
    """
    from kairos_api.break_api import _gold_enabled

    if not _gold_enabled():
        pytest.skip("gold breaks are switched off in settings, so the act is refused by design")
    day = opened_day["day"]
    before = opened_day["totals"]
    counts: dict[str, int] = {}
    for row in opened_day["breaks"]:
        counts[row["segment_id"]] = counts.get(row["segment_id"], 0) + 1

    stranded = None
    for segment_id in sorted(counts, key=lambda key: (-counts[key], key))[:3]:
        if counts[segment_id] < 2:
            continue
        target = [row for row in opened_day["breaks"] if row["segment_id"] == segment_id][-1]
        marked = client.post(f"/api/breaks/{quote(target['break_id'], safe='')}/gold")
        assert marked.status_code == 201
        after_board = client.get("/api/plan/day", params={"day": day}).json()
        if target["break_id"] not in {row["break_id"] for row in after_board["breaks"]}:
            stranded = (target, after_board)
            break
        client.delete(f"/api/breaks/{quote(target['break_id'], safe='')}/gold")
    if stranded is None:
        pytest.skip("no gold mark on this day removed the break it was performed on")

    target, after_board = stranded
    assert after_board["totals"]["breaks"] < before["breaks"]
    assert after_board["totals"]["revenue"] != pytest.approx(before["revenue"], abs=0.005), (
        "a mark that changed nothing would not be evidence of anything"
    )
    # The route parses the programme out of the break id and never looks that id
    # up in the plan, which is what lets the inverse survive its own act.
    cleared = client.delete(f"/api/breaks/{quote(target['break_id'], safe='')}/gold")
    assert cleared.status_code == 200
    restored = client.get("/api/plan/day", params={"day": day}).json()["totals"]
    assert restored["revenue"] == pytest.approx(before["revenue"], abs=0.005)
    assert restored["breaks"] == before["breaks"]


def test_the_panel_names_whichever_of_the_three_acts_it_is_reporting():
    """One panel, three acts, and a heading that says which one this was."""
    figures = {
        "before": {"revenue": 1062669.88, "retention": 0.947698, "breaks": 80, "ad_seconds": 9600.0, "gold_breaks": 0},
        "after": {"revenue": 1028205.58, "retention": 0.950056, "breaks": 79, "ad_seconds": 9480.0, "gold_breaks": 3},
        "basis": {"channel": "רשת 13", "day": "2024-11-01"},
        "beforeBreaks": [],
        "afterBreaks": [],
    }
    gold = render_panel({**figures, "act": "gold", "predictedRevenue": None}, "he", True)["html"]
    save = render_panel({**figures, "act": "save", "predictedRevenue": 0.0}, "he", True)["html"]
    undo = render_panel({**figures, "act": "undo", "predictedRevenue": None}, "he", True)["html"]
    assert "מה שינוי הזהב עשה לתוכנית" in gold
    assert "מה השמירה שינתה בתוכנית" in save
    assert "מה הביטול החזיר" in undo
    assert "34,464" in gold and "₪" in gold, "the act's cost is a figure on the panel, in every act"
    assert "ביטול שינוי הזהב" in gold and "ביטול השמירה הזו" in save
    # An undo has already put the day back, so it is the one act with nothing
    # left to reverse, and it offers no control rather than a dead one.
    assert "day-readout-actions" in undo and "ביטול" not in undo.split("day-readout-actions")[1]


def test_every_act_that_writes_goes_through_the_one_settlement_seam():
    """The invariant the gold act drifted out of, asserted on all of them.

    Three acts settled and the fourth did not, and nothing held it. So each
    exported act in the writes module is required to reach ``settleAfter``, and
    the board is required to route the panel's control to the gold inverse when
    the settlement on screen is a gold one.
    """
    acts = read("plan/day/day-board-writes.js")
    exported = [chunk for chunk in acts.split("export async function ")[1:]]
    assert len(exported) >= 4, "the acts are all in one module, so they are all counted here"
    for chunk in exported:
        name = chunk.split("(")[0].strip()
        assert "settleAfter(" in chunk, f"{name} writes to a store without settling what it cost"
    assert "await settleAfter('gold', null, async () => {" in acts, (
        "there is no cheap preview of a gold change, so it settles against no prediction"
    )
    assert "rememberGold({ breakId: item.break_id, wasGold: Boolean(live.isGold) });" in acts
    assert "export async function undoGold({ lastGold, settleAfter, forgetGold, notify }) {" in acts

    board = read("plan/day/DayBoard.jsx")
    assert "return writes.applyGold({ item, live, notify, settleAfter, rememberGold: setLastGold });" in board
    assert "const goldSettled = Boolean(settlement) && settlement.act === 'gold';" in board
    assert "canUndo={(goldSettled ? Boolean(lastGold) : Boolean(lastSave)) && !saving}" in board
    assert "onUndo={goldSettled ? undoLastGold : undoLastSave}" in board
    assert "setLastGold(null);" in board, "opening another day forgets an act that belonged to the old one"

    panel = read("plan/day/DayBoardSettlement.jsx")
    assert "export function headingOf(act, label) {" in panel
    assert "export function failureText(act, message) {" in panel
    assert "notify(...failureText(act, error.message));" in board, (
        "a failed gold change used to report that the save had failed"
    )
