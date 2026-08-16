"""Browser-local manual plan variants stay exact, scoped, and non-authoritative."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FRONTEND = ROOT / "tv-break-dashboard"
WEEK = FRONTEND / "src" / "plan" / "week"
DAY = FRONTEND / "src" / "plan" / "day"


def test_local_variant_store_locks_baseline_and_blocks_changed_plan_identity():
    script = r"""
      import { createServer } from 'vite';
      const values = new Map();
      globalThis.window = { localStorage: {
        get length() { return values.size; },
        key(index) { return Array.from(values.keys())[index] ?? null; },
        getItem(key) { return values.get(key) ?? null; },
        setItem(key, value) { values.set(key, String(value)); },
      }};
      const server = await createServer({ server: { middlewareMode: true }, appType: 'custom' });
      try {
        const store = await server.ssrLoadModule('/src/plan/week/local-plan-variants.js');
        const draft = await server.ssrLoadModule('/src/plan/day/day-board-variant-draft.js');
        const board = {
          available: true, operator_channel: 'Owned TV', day: '2026-08-16',
          basis: { channel: 'Owned TV', day: '2026-08-16', revenue_weight: 60, risk_lambda: 0, objective_mode: 'blend', segments: 1 },
          programmes: [{ segment_id: 's1', start_seconds: 3600, duration_seconds: 1800, breaks: 1 }],
          breaks: [{ break_id: 'b1', segment_id: 's1', offset_seconds: 60, duration_seconds: 120, is_gold: false }],
          totals: { revenue: 1000, retention: .8, breaks: 1, ad_seconds: 120 },
        };
        const live = { sha256: 'abc123', computed_at: '2026-08-16T06:00:00Z' };
        const scope = store.createVariantScope(board, live, {});
        const first = store.ensureLocalBaseline(scope, board);
        const second = store.ensureLocalBaseline(scope, { ...board, totals: { ...board.totals, revenue: 1 } });
        const editsA = { b1: { offset_seconds: 120, duration_seconds: 120 } };
        const editsB = { b1: { offset_seconds: 120, duration_seconds: 180 } };
        store.storeLocalVariant(scope, board, { name: 'A', edits: editsA, totals: { revenue: 1000 }, delta: { revenue: 0 }, compliance: { compliant: true } });
        store.storeLocalVariant(scope, board, { name: 'B', edits: editsB, totals: { revenue: 1100 }, delta: { revenue: 100 }, compliance: { compliant: true } });
        const loaded = store.loadDayDrafts(scope);
        const staleScope = store.createVariantScope(board, { ...live, sha256: 'changed' }, {});
        const safe = draft.normalizeVariantDraft(editsA, board.breaks, new Map([['s1', board.programmes[0]]]));
        const unsafe = draft.normalizeVariantDraft({ missing: { offset_seconds: 0, duration_seconds: 30 } }, board.breaks, new Map([['s1', board.programmes[0]]]));
        process.stdout.write(JSON.stringify({
          verifiable: scope.verifiable,
          firstCreated: first.created,
          secondCreated: second.created,
          immutableRevenue: second.baseline.totals.revenue,
          baselineCount: loaded.baselines.length,
          variantCount: loaded.variants.length,
          current: loaded.variants.every((item) => store.variantIsCurrent(item, scope)),
          stale: loaded.variants.some((item) => store.variantIsCurrent(item, staleScope)),
          changed: store.changedEditsBetween(editsA, editsB),
          safe: safe.ok,
          unsafe: unsafe.ok,
        }));
      } finally { await server.close(); }
    """
    result = subprocess.run(
        ["node", "--input-type=module", "-"], cwd=FRONTEND,
        input=script, text=True, check=True, capture_output=True,
    )
    assert json.loads(result.stdout) == {
        "verifiable": True,
        "firstCreated": True,
        "secondCreated": False,
        "immutableRevenue": 1000,
        "baselineCount": 1,
        "variantCount": 2,
        "current": True,
        "stale": False,
        "changed": 1,
        "safe": True,
        "unsafe": False,
    }


def test_daily_workbench_names_the_two_different_persistence_paths():
    panel = (WEEK / "LocalPlanVariants.jsx").read_text(encoding="utf-8")
    readout = (DAY / "DayBoardReadout.jsx").read_text(encoding="utf-8")
    board = (DAY / "DayBoard.jsx").read_text(encoding="utf-8")
    assert "Keep the exact arrangement" in panel
    assert "No API call writes it" in panel
    assert "writes placement constraints, then re-runs the optimizer" in panel
    assert 'id="day-board-server-replan"' in readout
    assert "Review ${editCount}" in readout
    assert "draftEdits={edits}" in board
    assert "draftCommand={draftCommand}" in (WEEK / "PlanBoardWorkbench.jsx").read_text(encoding="utf-8")


def test_stale_drafts_are_visible_but_not_actionable_and_grid_scrolls():
    panel = (WEEK / "LocalPlanVariants.jsx").read_text(encoding="utf-8")
    css = (WEEK / "plan-week-board-v2.css").read_text(encoding="utf-8")
    assert "Visible for audit only" in panel
    assert "disabled={!current}" in panel
    assert "variantIsCurrent(variant, scope)" in panel
    grid = css.split(".plan-board-stage > .plan-board-instrument > .planning-canvas:not(.planning-canvas-timeline) {", 1)[1].split("}", 1)[0]
    assert "overflow-x: auto;" in grid
    assert "max-inline-size: 100%;" in grid


def test_variant_modules_stay_under_the_file_cap_and_actions_keep_44px_targets():
    for path in (
        WEEK / "LocalPlanVariants.jsx",
        WEEK / "local-plan-variants.js",
        WEEK / "local-plan-variants.css",
        DAY / "day-board-variant-draft.js",
        DAY / "DayBoard.jsx",
    ):
        assert len(path.read_text(encoding="utf-8").splitlines()) < 450, path
    css = (WEEK / "local-plan-variants.css").read_text(encoding="utf-8")
    assert "min-block-size: 38px" not in css
    assert "min-block-size: 40px" not in css
    assert "min-block-size: 42px" not in css
    assert css.count("min-block-size: var(--control-height)") >= 4
