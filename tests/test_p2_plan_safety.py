"""Plan optimization is fail-closed on saved settings and its real inventory input."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "tv-break-dashboard"
WEEK = FRONTEND / "src" / "plan" / "week"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def readiness_cases() -> dict[str, dict]:
    script = r"""
      import { createServer } from 'vite';
      const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
      try {
        const { inventoryReadinessFromFiles: read } = await server.ssrLoadModule('/src/plan/week/use-plan-inventory-readiness.js');
        const path = 'data/Spots - inventory.csv';
        const cases = {
          absent: read({ also_read: [] }),
          empty: read({ also_read: [{ path, read_state: 'read_yielding_nothing', yielded_items: 0, note: { en: 'empty', he: 'ריק' } }] }),
          ready: read({ also_read: [{ path, read_state: 'read_yielding', yielded_items: 17, note: { en: 'live', he: 'פעיל' } }] }),
          malformed: read({}),
          ambiguous: read({ also_read: [{ path, read_state: 'unknown', yielded_items: 17 }] }),
        };
        process.stdout.write(JSON.stringify(cases));
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


def test_missing_optional_inventory_is_verified_identity_mode(readiness_cases):
    assert readiness_cases["absent"] | {"path": "ignored"} == {
        "status": "ready", "code": "absent", "slots": 0, "mode": "identity",
        "path": "ignored", "note": None, "error": None, "signature": "absent",
    }


def test_present_but_empty_inventory_is_blocked_with_its_localized_note(readiness_cases):
    state = readiness_cases["empty"]
    assert state["status"] == "blocked" and state["code"] == "empty"
    assert state["slots"] == 0 and state["note"] == {"en": "empty", "he": "ריק"}


def test_only_a_positive_reported_slot_count_is_inventory_ready(readiness_cases):
    assert readiness_cases["ready"]["status"] == "ready"
    assert readiness_cases["ready"]["slots"] == 17
    assert readiness_cases["malformed"]["status"] == "error"
    assert readiness_cases["ambiguous"]["status"] == "error"


def test_settings_failure_clears_values_and_exposes_retry_instead_of_defaults():
    surface = _text(WEEK / "use-plan-surface.js")
    gate = _text(WEEK / "PlanStateGate.jsx")
    for contract in (
        "setSettingsState('loading')", "setSettingsState('error')", "setSaved(null)",
        "setDraft({})", "settingsReady: settingsState === 'ready'", "retrySettings: loadSettings",
    ):
        assert contract in surface
    assert "Nothing is inferred from factory values" in gate
    assert "onClick={retry}" in gate


def test_run_compare_palette_keyboard_and_prepare_share_the_same_lock():
    week = _text(WEEK / "PlanWeek.jsx")
    actions = _text(WEEK / "use-plan-optimization-actions.js")
    commands = _text(WEEK / "plan-week-commands.js")
    surface = _text(WEEK / "use-plan-surface.js")
    assert "inventory.status === 'ready'" in actions
    assert "const checked = await inventory.verify();" in actions
    assert "inventorySlots: checked.slots" in actions
    assert "inventorySignature: checked.signature" in actions
    assert "disabled: surface.runState === 'running' || !optimizationAllowed" in commands
    assert "disabled: surface.compareState === 'running' || !optimizationAllowed" in commands
    assert "run: compareNow" in commands
    assert "enabled: prepareCompare && settingsState === 'ready' && optimizationAllowed" in surface
    assert "runAllowed={optimization.optimizationAllowed}" in week


def test_shell_fallbacks_are_unavailable_and_sections_retry_before_empty_states():
    fallbacks = _text(FRONTEND / "src" / "shell" / "fallbacks.js")
    hook = _text(WEEK / "use-section-data.js")
    for name in ("fallbackSettings", "fallbackOverview", "fallbackSchedule", "fallbackInventory"):
        block = fallbacks.split(f"export const {name} = {{", 1)[1].split("};", 1)[0]
        assert "_unavailable: true" in block
    assert "value._unavailable !== true" in hook
    assert "data: null, state: 'error'" in hook
    assert "retry," in hook


def test_fallback_settings_never_surface_or_write_as_saved_truth():
    script = r"""
      import { createServer } from 'vite';
      const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
      try {
        const { savedSettingsFromOverview: read } = await server.ssrLoadModule('/src/shell/use-kairos-data.js');
        process.stdout.write(JSON.stringify({
          missing: read(null),
          fallback: read({ _unavailable: true, settings: { revenue_weight: 60 } }),
          nested: read({ settings: { _unavailable: true, revenue_weight: 60 } }),
          ready: read({ settings: { revenue_weight: 60 } }),
        }));
      } finally { await server.close(); }
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-"], cwd=FRONTEND, input=script,
        text=True, check=True, capture_output=True,
    )
    assert json.loads(result.stdout) == {
        "missing": None,
        "fallback": None,
        "nested": None,
        "ready": {"revenue_weight": 60},
    }

    data = _text(FRONTEND / "src" / "shell" / "use-kairos-data.js")
    dashboard = _text(FRONTEND / "src" / "shell" / "TVBreakDashboard.jsx")
    actions = _text(FRONTEND / "src" / "shell" / "plan-actions.js")
    router = _text(FRONTEND / "src" / "shell" / "workspace-router.jsx")
    rules = _text(FRONTEND / "src" / "rules" / "RulesWorkspace.jsx")
    topbar = _text(FRONTEND / "src" / "shell" / "top-bar.jsx")

    assert "if (!overview || overview._unavailable === true) return null;" in data
    assert "if (!settings || settings._unavailable === true) return null;" in data
    assert "const settingsAvailable = Boolean(confirmedOverviewSettings) && settings._unavailable !== true;" in dashboard
    assert "if (!confirmedOverviewSettings) return;" in dashboard
    assert "settingsAvailable," in dashboard
    guard = actions.index("if (!settingsAvailable)")
    write = actions.index("fetch(`${API_BASE}/api/settings`", guard)
    assert guard < write
    assert "settingsAvailable={settingsAvailable}" in router
    assert "No fallback values are shown or writable." in rules
    assert "props.settingsAvailable" in rules
    assert "disabled={!settingsAvailable}" in topbar


def test_embedded_day_editor_disables_and_explains_its_run_control():
    toolbar = _text(FRONTEND / "src" / "plan" / "day" / "ScheduleEditorToolbar.jsx")
    assert "disabled={recomputeState === 'running' || recomputeDisabled}" in toolbar
    assert 'id="schedule-editor-run-lock"' in toolbar
    assert "recomputeDisabledReason" in toolbar


def test_every_touched_frontend_module_stays_inside_the_size_law():
    names = [
        "PlanWeek.jsx", "ComparePanel.jsx", "use-plan-surface.js", "PlanActionSafety.jsx",
        "PlanStateGate.jsx", "use-plan-inventory-readiness.js", "use-plan-optimization-actions.js",
    ]
    paths = [WEEK / name for name in names]
    paths += [
        FRONTEND / "src" / "plan" / "day" / "ScheduleEditor.jsx",
        FRONTEND / "src" / "plan" / "day" / "ScheduleEditorToolbar.jsx",
    ]
    for path in paths:
        assert len(_text(path).splitlines()) < 450, path
