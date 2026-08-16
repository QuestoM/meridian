"""Day-scoped optimization and editor writes cannot bypass their review gates."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "tv-break-dashboard"
DAY = FRONTEND / "src" / "plan" / "day"
WEEK = FRONTEND / "src" / "plan" / "week"


def _source(name: str) -> str:
    return (DAY / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def reviewed_input_comparisons() -> dict[str, bool]:
    script = r"""
      import { createServer } from 'vite';
      const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
      try {
        const { sameDayRunInputs: same } = await server.ssrLoadModule('/src/plan/day/use-scoped-day-run.js');
        const base = {
          settingsSignature: 'saved-v1',
          inventory: { mode: 'inventory', slots: 9, path: 'data/Spots - inventory.csv', signature: 'mtime|size|read_yielding|9' },
        };
        const changed = (key, value) => ({ ...base, inventory: { ...base.inventory, [key]: value } });
        process.stdout.write(JSON.stringify({
          exact: same(base, JSON.parse(JSON.stringify(base))),
          settings: same(base, { ...base, settingsSignature: 'saved-v2' }),
          mode: same(base, changed('mode', 'identity')),
          slots: same(base, changed('slots', 10)),
          path: same(base, changed('path', 'data/other.csv')),
          signature: same(base, changed('signature', 'mtime-2|size|read_yielding|9')),
        }));
      } finally { await server.close(); }
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-"],
        cwd=FRONTEND,
        input=script,
        text=True,
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def test_review_confirmation_requires_the_exact_saved_inputs(reviewed_input_comparisons):
    assert reviewed_input_comparisons == {
        "exact": True,
        "settings": False,
        "mode": False,
        "slots": False,
        "path": False,
        "signature": False,
    }


def test_day_run_checks_saved_settings_and_files_before_review_and_again_before_write():
    source = _source("use-scoped-day-run.js")
    assert "Promise.all([readSettings(), readFiles()])" in source
    assert "inventoryReadinessFromFiles(filesResult.data)" in source
    assert "REQUIRED_SETTINGS.every" in source

    request = source.split("const requestReview", 1)[1].split("const confirmReview", 1)[0]
    assert "const checked = await safety.verify();" in request
    assert "if (checked.status !== 'ready') return false;" in request
    assert "setReview(" in request
    assert request.index("await safety.verify()") < request.index("setReview(")

    confirm = source.split("const confirmReview", 1)[1].split("return {", 1)[0]
    assert "const checked = await safety.verify();" in confirm
    assert "sameDayRunInputs(action.inputs, checked)" in confirm
    assert "current.current.runner(API_BASE, action.scope)" in confirm
    assert confirm.index("sameDayRunInputs(action.inputs, checked)") < confirm.index("current.current.runner")
    assert "if (activeReview.current !== action) return false;" in confirm
    assert "activeReview.current = null;" in source.split("const cancelReview", 1)[1]


def test_both_day_run_surfaces_use_the_one_fail_closed_coordinator():
    for name in ("ScheduleInspector.jsx", "OverrideDecisions.jsx"):
        source = _source(name)
        assert "useScopedDayRun({" in source
        assert "runner: runDayPlanJob" in source
        assert "onClick={dayRun.requestReview}" in source
        assert "onConfirm={dayRun.confirmReview}" in source or "onConfirmDayRun={dayRun.confirmReview}" in source
        assert "DayRunSafetyNotice" in source
        assert "runDayPlanJob(" not in source

    direct_posts = []
    for path in DAY.glob("*.js*"):
        if path.name != "override-console-lib.js" and "/api/jobs/recompute" in path.read_text(encoding="utf-8"):
            direct_posts.append(path.name)
    assert direct_posts == []


def test_day_run_failure_has_retry_and_source_recovery_without_opening_review():
    notice = _source("DayRunSafetyNotice.jsx")
    dialog = _source("DayRunReviewDialog.jsx")
    assert "onClick={safety.retry}" in notice
    assert "openOptimizerSources" in notice
    assert "requestNavigation('Sources', { sources: 'files' })" in notice
    assert "role=\"alert\"" in notice
    assert "initialFocusRef={cancelRef}" in dialog
    assert "ref={cancelRef}" in dialog
    assert "dismissOnBackdrop={false}" in dialog


def test_schedule_editor_enter_and_row_save_open_the_same_exact_review():
    editor = _source("ScheduleEditor.jsx")
    keyboard = editor.split("function handleKeyDown", 1)[1].split("function pinTargetFor", 1)[0]
    assert "event.key === 'Enter'" in keyboard
    assert "requestPinReview(item);" in keyboard
    assert "savePin(item)" not in keyboard
    assert "onSave={requestPinReview}" in editor
    assert "<ScheduleEditorPinReviewDialog" in editor

    dialog = _source("ScheduleEditorPinReviewDialog.jsx")
    for field in ("'Break', 'ברייק'", "'Scope', 'היקף'", "'Time', 'זמן'", "'Consequence', 'השפעה'"):
        assert field in dialog
    assert "review?.target?.item?.break_id" in dialog
    assert "initialFocusRef={cancelRef}" in dialog and "ref={cancelRef}" in dialog
    assert "dismissOnBackdrop={false}" in dialog


def test_day_board_keyboard_toolbar_and_save_share_the_review_coordinator():
    board = _source("DayBoard.jsx")
    keyboard = board.split("function handleKeyDown", 1)[1].split("async function settleAfter", 1)[0]
    assert "event.key === 'g' || event.key === 'G'" in keyboard
    assert "writeReview.requestGold(item);" in keyboard
    assert "toggleGold(item)" not in keyboard
    assert "onGold={() => selectedItem && writeReview.requestGold(selectedItem)}" in board
    assert "onSave={writeReview.requestSave}" in board
    assert "onGold: toggleGold, onSave: saveAll" in board
    assert "<DayBoardWriteDialog" in board

    coordinator = _source("use-day-board-write-review.js")
    assert "setReview({" in coordinator
    assert "if (action?.kind === 'gold') current.current.onGold(action.item);" in coordinator
    assert "else if (action?.kind === 'save') current.current.onSave();" in coordinator

    dialog = _source("DayBoardWriteDialog.jsx")
    for field in ("'Object', 'אובייקט'", "'Scope', 'היקף'", "'Breaks', 'ברייקים'", "'Consequence', 'השפעה'"):
        assert field in dialog
    assert "initialFocusRef={cancelRef}" in dialog and "ref={cancelRef}" in dialog
    assert "dismissOnBackdrop={false}" in dialog


def test_shared_dialog_really_closes_on_escape_and_returns_focus():
    modal = (FRONTEND / "src" / "shell" / "modal-primitives.jsx").read_text(encoding="utf-8")
    assert "onCancel={(event) => { event.preventDefault(); onClose?.('escape'); }}" in modal
    assert "if (initialFocusRef?.current) initialFocusRef.current.focus" in modal
    assert "useFocusReturn(open);" in modal


def test_every_day_module_touched_by_this_safety_pass_stays_under_450_lines():
    for name in (
        "ScheduleInspector.jsx",
        "OverrideDecisions.jsx",
        "ScheduleEditor.jsx",
        "DayBoard.jsx",
        "use-scoped-day-run.js",
        "use-day-board-write-review.js",
        "DayRunSafetyNotice.jsx",
        "DayRunReviewDialog.jsx",
        "ScheduleEditorPinReviewDialog.jsx",
        "DayBoardWriteDialog.jsx",
    ):
        path = DAY / name
        assert len(path.read_text(encoding="utf-8").splitlines()) < 450, path


@pytest.fixture(scope="module")
def prepared_inventory_comparisons() -> dict[str, object]:
    script = r"""
      import { createServer } from 'vite';
      const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
      try {
        const module = await server.ssrLoadModule('/src/plan/week/use-compare-prepare.js');
        const leg = { revenue_weight: 60, retention_floor: 0.72, max_breaks_per_hour: 4, risk_lambda: 0, objective_mode: 'blend' };
        const base = { status: 'ready', mode: 'inventory', slots: 9, path: 'data/Spots - inventory.csv', signature: 'source-v1' };
        const expected = module.inventorySnapshot(base);
        const changed = { ...base, signature: 'source-v2' };
        let fallbackRuns = 0;
        const exactFallback = await module.verifiedCompareFallback(
          { verify: async () => ({ ...base }) }, base, async () => { fallbackRuns += 1; return { ok: true }; },
        );
        const racedFallback = await module.verifiedCompareFallback(
          { verify: async () => changed }, base, async () => { fallbackRuns += 1; return { ok: true }; },
        );
        const failedFallback = await module.verifiedCompareFallback(
          { verify: async () => { throw new Error('source offline'); } },
          base,
          async () => { fallbackRuns += 1; return { ok: true }; },
        );
        process.stdout.write(JSON.stringify({
          exact: module.samePreparedInventory(expected, { ...base }),
          changed: module.samePreparedInventory(expected, changed),
          blocked: module.samePreparedInventory(expected, { ...base, status: 'blocked' }),
          key: module.comparePreparationKey(leg, leg, base),
          changedKey: module.comparePreparationKey(leg, leg, changed),
          exactFallback: exactFallback.ok,
          racedFallback: racedFallback.ok,
          failedFallback: failedFallback.ok,
          fallbackRuns,
        }));
      } finally { await server.close(); }
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-"],
        cwd=FRONTEND,
        input=script,
        text=True,
        check=True,
        capture_output=True,
    )
    return json.loads(result.stdout)


def test_auto_compare_rechecks_the_source_and_refuses_a_signature_race(prepared_inventory_comparisons):
    assert prepared_inventory_comparisons["exact"] is True
    assert prepared_inventory_comparisons["changed"] is False
    assert prepared_inventory_comparisons["blocked"] is False
    assert prepared_inventory_comparisons["key"] != prepared_inventory_comparisons["changedKey"]
    assert prepared_inventory_comparisons["exactFallback"] is True
    assert prepared_inventory_comparisons["racedFallback"] is False
    assert prepared_inventory_comparisons["failedFallback"] is False
    assert prepared_inventory_comparisons["fallbackRuns"] == 1

    prepare = (WEEK / "use-compare-prepare.js").read_text(encoding="utf-8")
    verify = prepare.index("Promise.resolve().then(() => inventory.verify({ announce: false }))")
    signature = prepare.index("samePreparedInventory(expectedInventory, checked)", verify)
    stream = prepare.index("return streamCompare(", signature)
    assert verify < signature < stream
    assert "const key = comparePreparationKey(legA, legB, inventory);" in prepare
    assert "abort.current?.abort()" in prepare

    surface = (WEEK / "use-plan-surface.js").read_text(encoding="utf-8")
    fallback = surface.index("verifiedCompareFallback(inventoryReadiness, checkedInventory")
    plain = surface.index("api.compareScenarios(body)", fallback)
    assert fallback < plain

    actions = (WEEK / "use-plan-optimization-actions.js").read_text(encoding="utf-8")
    week = (WEEK / "PlanWeek.jsx").read_text(encoding="utf-8")
    assert "return surface.compare(checked);" in actions
    assert "checkedInventory?.status !== 'ready'" in surface
    assert "comparePreparationKey(guardedLegA, guardedLegB, checkedInventory)" in surface
    assert "inventory: inventoryReadiness" in surface
    assert "inventoryReadiness," in week
    readiness = (WEEK / "use-plan-inventory-readiness.js").read_text(encoding="utf-8")
    assert "if (options?.announce !== false)" in readiness


def test_retired_direct_compare_surface_has_no_live_inbound_path():
    retired = WEEK / "ScenarioCompare.jsx"
    assert not retired.exists()
    inbound = []
    for path in (FRONTEND / "src").rglob("*.js*"):
        source = path.read_text(encoding="utf-8")
        if "ScenarioCompare" in source and (
            re.search(r"\bfrom\s+['\"][^'\"]*ScenarioCompare", source)
            or re.search(r"\bimport\s*\(\s*['\"][^'\"]*ScenarioCompare", source)
        ):
            inbound.append(path.relative_to(FRONTEND).as_posix())
    assert inbound == []
