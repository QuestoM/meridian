import { API_BASE } from './api';
import { finiteNumber, formatNumber } from './format';

// The run, save and rebuild handlers. A factory rather than a hook, for the
// same reason as decision-actions: no React hook is called and the closures
// are rebuilt each render exactly as the inline declarations were.
export function createPlanActions({
  settings,
  setSettings,
  scenario,
  riskLambda,
  locale,
  notify,
  setRefreshKey,
  setSaveState,
  setRecomputeState,
  setRecomputeProgress,
  setApplyWeightState,
  setOptimizationState,
  setOptimizationPlan,
  setActiveView,
  setOptimizerView,
  setInspectorOpen,
}) {
  function scenarioControls() {
    // The "Balanced" scenario follows the operator's saved revenue_weight so the
    // simulation opens on their real choice, not a hardcoded default.
    const savedWeight = finiteNumber(settings.revenue_weight);
    const balanced = Number.isFinite(savedWeight) ? savedWeight : 60;
    const revenueWeight = scenario === 'Revenue priority' ? 85 : scenario === 'Retention guardrail' ? 35 : balanced;
    return {
      revenue_weight: revenueWeight,
      retention_floor: settings.min_retention_floor,
      max_breaks_per_hour: settings.max_breaks_per_hour,
      risk_lambda: Math.min(1, Math.max(0, riskLambda / 100)),
    };
  }

  async function handleRunOptimization() {
    setActiveView('Optimizer');
    setOptimizerView('grid');
    setInspectorOpen(true);
    setOptimizationState('running');
    try {
      const response = await fetch(`${API_BASE}/api/optimizer-plan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scenarioControls()),
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      const plan = await response.json();
      setOptimizationPlan(plan);
      notify(
        `Optimization produced ${formatNumber(plan.summary?.total_breaks || 0, locale)} compliant breaks.`,
        `האופטימיזציה יצרה ${formatNumber(plan.summary?.total_breaks || 0, locale)} ברייקים תקינים.`,
      );
    } catch {
      notify('Optimizer API is unavailable. Keeping the current working plan.', 'מנוע האופטימיזציה לא זמין. התוכנית הנוכחית נשמרת.');
    } finally {
      setOptimizationState('idle');
    }
  }

  async function persistSettings(nextSettings) {
    setSettings(nextSettings);
    setSaveState('saving');
    try {
      const response = await fetch(`${API_BASE}/api/settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(nextSettings),
      });
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      setSettings(await response.json());
      setSaveState('saved');
      // Bump the refresh key so dependent views refetch against the saved state
      // instead of leaving stale numbers behind a success toast.
      setRefreshKey((k) => k + 1);
      window.setTimeout(() => setSaveState('idle'), 1800);
    } catch {
      setSaveState('error');
    }
  }

  async function handleApplyFrontierFloor(floor) {
    const nextFloor = finiteNumber(floor);
    if (nextFloor === null) return;
    setApplyWeightState('saving');
    try {
      await persistSettings({ ...settings, min_retention_floor: nextFloor });
      const pct = Math.round(nextFloor * 100);
      notify(
        `Saved retention floor set to ${pct} percent.`,
        `רף השימור השמור עודכן ל־${pct} אחוז.`,
      );
    } finally {
      setApplyWeightState('idle');
    }
  }

  async function handleRecomputeSchedule(scope = null) {
    setRecomputeState('running');
    setRecomputeProgress(null);
    const finishOk = (result) => {
      setRecomputeState('done');
      // Refetch so the schedule and overview reflect the freshly computed plan.
      setRefreshKey((k) => k + 1);
      notify(
        `Weekly schedule recomputed: ${formatNumber(result.total_breaks || 0, locale)} breaks, ${formatNumber(Math.round(result.total_revenue || 0), locale)} ILS.`,
        `הלוח השבועי חושב מחדש: ${formatNumber(result.total_breaks || 0, locale)} ברייקים, ${formatNumber(Math.round(result.total_revenue || 0), locale)} ש"ח.`,
      );
      window.setTimeout(() => setRecomputeState('idle'), 2400);
    };
    const finishFail = () => {
      setRecomputeState('error');
      notify('Recompute failed. The saved schedule is unchanged.', 'החישוב מחדש נכשל. הלוח השמור לא השתנה.');
      window.setTimeout(() => setRecomputeState('idle'), 2400);
    };
    const sleep = (ms) => new Promise((resolve) => window.setTimeout(resolve, ms));
    try {
      // Preferred path: the async job endpoint with honest tri-state status and
      // real per-day progress. Falls back to the synchronous endpoint when the
      // job API is absent (older backend).
      const startResponse = await fetch(`${API_BASE}/api/jobs/recompute`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(scope ? { scope } : {}),
      });
      if (startResponse.status === 404) {
        const response = await fetch(`${API_BASE}/api/recompute-schedule`, { method: 'POST' });
        if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
        finishOk(await response.json());
        return;
      }
      if (!startResponse.ok) throw new Error(`${startResponse.status} ${startResponse.statusText}`);
      const { job_id: jobId } = await startResponse.json();
      // Poll to a terminal state; ~10 minute ceiling so a dead backend cannot
      // leave the button spinning forever.
      for (let attempt = 0; attempt < 400; attempt += 1) {
        await sleep(1500);
        const statusResponse = await fetch(`${API_BASE}/api/jobs/${jobId}`);
        if (!statusResponse.ok) throw new Error(`${statusResponse.status} ${statusResponse.statusText}`);
        const record = await statusResponse.json();
        if (record.progress && Number.isFinite(record.progress.done) && Number.isFinite(record.progress.total)) {
          setRecomputeProgress({ done: record.progress.done, total: record.progress.total });
        }
        if (record.status === 'done') {
          finishOk(record.result || {});
          return;
        }
        if (record.status === 'failed') {
          setRecomputeState('error');
          notify(
            `Recompute failed: ${record.error || 'unknown error'}. The saved schedule is unchanged.`,
            `החישוב מחדש נכשל: ${record.error || 'שגיאה לא ידועה'}. הלוח השמור לא השתנה.`,
          );
          window.setTimeout(() => setRecomputeState('idle'), 2400);
          return;
        }
      }
      throw new Error('job polling timed out');
    } catch {
      finishFail();
    } finally {
      setRecomputeProgress(null);
    }
  }

  async function handleApplyOptimization() {
    // The optimizer preview runs on the operator's chosen levers but never saves
    // them, so the weekly schedule (saved CSV) never moves. Apply persists those
    // levers into settings, then runs the legitimate full-week recompute that
    // reads them, so the Schedule, Reports and Overview pages all catch up.
    // The preview result itself is never written to the CSV, which would corrupt
    // the rest of the week; only settings plus a full recompute are persisted.
    const controls = scenarioControls();
    // Map the scenario control fields onto their settings field names. Most match
    // by name; retention_floor lands on min_retention_floor (the setting key).
    const nextSettings = {
      ...settings,
      revenue_weight: Math.round(finiteNumber(controls.revenue_weight) ?? settings.revenue_weight),
      risk_lambda: finiteNumber(controls.risk_lambda) ?? settings.risk_lambda,
      min_retention_floor: finiteNumber(controls.retention_floor) ?? settings.min_retention_floor,
      max_breaks_per_hour: finiteNumber(controls.max_breaks_per_hour) ?? settings.max_breaks_per_hour,
    };
    await persistSettings(nextSettings);
    notify(
      'Saved these levers and rebuilding the whole weekly schedule.',
      'ההגדרות נשמרו והלוח השבועי כולו נבנה מחדש.',
    );
    await handleRecomputeSchedule();
  }

  return {
    scenarioControls,
    handleRunOptimization,
    persistSettings,
    handleApplyFrontierFloor,
    handleRecomputeSchedule,
    handleApplyOptimization,
  };
}
