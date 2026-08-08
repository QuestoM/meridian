"""P2, Bar 3: the three things this piece may not make worse.

Section 8.5 of the specification names them for this piece. The frontier point
is still clickable and still applies as a saved retention floor, the plan CSV
still downloads 8,704 rows, and the four objective templates survive. Each is
checked against the artifact rather than against a description of it.
"""

from __future__ import annotations

import csv
import io
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tv-break-dashboard" / "src"
WEEK = SRC / "plan" / "week"

# The four values the settings surface has always applied, carried by value so a
# rewrite cannot quietly move one.
TEMPLATES = {
    "balanced": {"revenue_weight": 60, "risk_lambda": 0, "min_retention_floor": 0.72},
    "revenue": {"revenue_weight": 85, "risk_lambda": 0, "min_retention_floor": 0.70},
    "retention": {"revenue_weight": 35, "risk_lambda": 0, "min_retention_floor": 0.78},
    "conservative": {"revenue_weight": 60, "risk_lambda": 1, "min_retention_floor": 0.74},
}


@pytest.fixture()
def client():
    from kairos_api.server import app

    return TestClient(app)


def _model() -> str:
    return (WEEK / "plan-week-model.js").read_text(encoding="utf-8")


def test_the_four_objective_templates_survive_with_their_own_values():
    text = _model()
    block = text.split("OBJECTIVE_TEMPLATES")[1].split("OBJECTIVE_FOCUS")[0]
    assert len(re.findall(r"key: '", block)) == 4, "there are four templates, no more and no fewer"
    for key, values in TEMPLATES.items():
        assert f"key: '{key}'" in block, key
    for key, values in TEMPLATES.items():
        for field, value in values.items():
            rendered = f"{field}: {value}" if not isinstance(value, float) else f"{field}: {value:g}"
            assert rendered in block or f"{field}: {value}" in block, f"{key}.{field}"


def test_the_templates_are_offered_on_the_surface_and_apply_in_one_click():
    panel = (WEEK / "ObjectivePanel.jsx").read_text(encoding="utf-8")
    assert "OBJECTIVE_TEMPLATES" in panel
    assert "onApplyTemplate(template.values)" in panel
    assert "templateMatches" in panel, "an already-applied template reads as applied"


def test_the_frontier_point_is_still_clickable_and_still_applies_a_saved_floor():
    """Unique to Overview today, so the chart is this piece's and the handler is
    the shell's, and the row holds only if both halves are still wired."""
    chart = (WEEK / "FrontierScopeChart.jsx").read_text(encoding="utf-8")
    assert "onApplyFloor" in chart
    assert "onClick={() => canApply && onApplyFloor(focusPoint.floor)}" in chart
    # The chart refuses to offer the control when the point is already the saved
    # floor, which is the state it had before this wave.
    assert "isSavedFloorSelected" in chart

    actions = (SRC / "shell" / "plan-actions.js").read_text(encoding="utf-8")
    assert "async function handleApplyFrontierFloor(floor)" in actions
    assert "min_retention_floor: nextFloor" in actions

    page = (SRC / "today" / "OverviewPage.jsx").read_text(encoding="utf-8")
    assert "onApplyFloor={onApplyFrontierFloor}" in page


def test_the_plan_csv_downloads_every_row_of_the_operators_own_channel(client):
    """Ruling 009 moved the floor, and it moved it for a reason worth keeping.

    This asserted 8,704 rows, the whole plan, off the route. The route now serves
    the operator's own channel only, because it is an operator surface. So the
    floor here is the operator's own row count, and the whole-file figure is
    asserted on the file below, where it belongs.
    """
    response = client.get("/api/export/schedule.csv")
    assert response.status_code == 200
    rows = list(csv.reader(io.StringIO(response.text)))
    assert rows, "the export is not empty"
    data_rows = len(rows) - 1
    if data_rows == 0:
        pytest.skip("no saved plan on this tree, so there is nothing to export")
    plan = ROOT / "output" / "weekly_break_schedule.csv"
    whole = list(csv.reader(io.StringIO(plan.read_text(encoding="utf-8"))))
    header = [cell.lstrip("﻿") for cell in whole[0]]
    channels = [row[header.index("channel")] for row in whole[1:] if row]
    assert len(channels) == 8704, f"the plan file carries {len(channels)} rows, the floor is 8,704"
    # Through the same seam the route reads, so the test and the product cannot
    # disagree about which channel is the operator's.
    from kairos_api import channel_scope

    owned_rows = channels.count(str(channel_scope.operator_channel() or "").strip())
    assert owned_rows > 0, "the operator's own channel has no rows in the plan file"
    assert data_rows == owned_rows, (
        f"the route served {data_rows} rows and the operator's own channel has {owned_rows}. "
        "It must serve all of its own and none of anybody else's."
    )
    # The export ships a byte-order mark so Excel opens the Hebrew correctly,
    # which is a property of the file worth keeping rather than an accident.
    assert response.text.startswith("﻿")
    assert [cell.lstrip("﻿") for cell in rows[0][:3]] == ["channel", "date", "day"]


def _rendered_week_board() -> dict[str, str]:
    """Every module the week board actually renders, resolved transitively.

    The first version of this test read one module and asserted the export path
    was absent from it. That was true and it proved nothing: the download was
    two hops away through the shell helper, so the assertion passed while the
    board shipped a button that fetched every channel's plan. What renders is
    the import closure, so the closure is what gets checked.
    """
    seen: dict[str, str] = {}
    queue = ["PlanWeek.jsx"]
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        path = WEEK / name
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        seen[name] = text
        for target in re.findall(r"""from\s+['"]\./([^'"]+)['"]""", text):
            stem = target.split("/")[-1]
            if stem.endswith((".js", ".jsx", ".css")):
                queue.append(stem)
            else:
                queue.extend([f"{stem}.jsx", f"{stem}.js"])
    return seen


def test_the_week_board_offers_no_download_of_the_export_that_carries_every_channel():
    """The export is the whole saved plan, competitors included, and it is
    another piece's route serving another piece's file. So no surface in this
    destination downloads it: the board names what that file holds and points at
    the destination that owns it.

    Checked against everything the board renders rather than against one module,
    because the defect this replaces was an indirection through the shell.
    """
    rendered = _rendered_week_board()
    assert "BoardPanel.jsx" in rendered, sorted(rendered)
    assert len(rendered) >= 20, sorted(rendered)

    offenders = {
        name: [line.strip() for line in text.splitlines() if "downloadScheduleCsv" in line or "/api/export/" in line]
        for name, text in rendered.items()
        if "downloadScheduleCsv" in text or "/api/export/" in text
    }
    assert offenders == {}, offenders

    # And nothing in the tree reaches the shell's downloader at all, which is the
    # rule in one line: this destination performs no file download.
    reaching = [path.name for path in WEEK.glob("*.js*") if "shell/downloads" in path.read_text(encoding="utf-8")]
    assert reaching == [], reaching


def test_the_board_names_what_the_plan_file_holds_and_points_at_the_door():
    """Removing the button is only half the fix. The operator still has to be
    able to get the file, and to know before clicking that it is not their
    channel alone."""
    board = (WEEK / "BoardPanel.jsx").read_text(encoding="utf-8")
    assert "exportScopeNote(schedule?.scope?.plan, locale)" in board
    assert "window.location.hash = 'Reports'" in board
    assert "Open the plan file on Sources" in board
    assert "מעבר לקובץ התוכנית במסך המקורות" in board

    model = (WEEK / "plan-week-model.js").read_text(encoding="utf-8")
    note = model.split("export function exportScopeNote")[1]
    # Both counts come from the payload's own scope note, so the sentence cannot
    # outlive the figure it describes.
    assert "note.rows_in" in note
    assert "note.rows_out" in note
    for constant in ("8,704", "8704", "2,540", "2540"):
        assert constant not in note, constant
    assert "every channel in the source" in note
    assert "מכל הערוצים שבמקור" in note


def test_the_plan_file_this_board_refuses_to_serve_really_does_carry_every_channel():
    """The reason the control is gone, measured on the FILE rather than the route.

    This read the route until ruling 009, and the route now serves the operator's
    own channel only, because it is an operator surface and a download of a
    rival's titles and revenue is the same breach as printing them on a screen.

    Reading the route was always the weaker measurement anyway. The claim in this
    test's name is about the artifact: the saved plan carries every channel, which
    is why the board names the file and points at the destination that owns it
    rather than offering a button that would hand three broadcasters' plans over
    in one click. The file is what that sentence is about, so the file is what is
    measured.
    """
    plan = ROOT / "output" / "weekly_break_schedule.csv"
    if not plan.exists():
        pytest.skip("no saved plan on this tree, so there is nothing to measure")
    rows = list(csv.reader(io.StringIO(plan.read_text(encoding="utf-8"))))
    if len(rows) <= 1:
        pytest.skip("the saved plan is empty")
    header = [cell.lstrip("﻿") for cell in rows[0]]
    channel_column = header.index("channel")
    channels = {row[channel_column] for row in rows[1:] if row}
    # Counted, never named: a rival's name does not belong in this file either.
    assert len(channels) > 1, "the plan file is single-channel, so the refusal would be pointless"
