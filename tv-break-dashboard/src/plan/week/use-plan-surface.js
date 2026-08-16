import { useCallback, useEffect, useRef, useState } from 'react';
import * as api from './plan-week-api';
import { objectiveFromLevers } from './plan-week-model';
import { streamCompare } from './plan-week-compare-stream';
import { comparePreparationKey, compareRequestBody, useComparePrepare, verifiedCompareFallback } from './use-compare-prepare';
import { usePlanFreshness } from './use-plan-freshness';
import { announceRunResult } from './plan-run-result';
const OBJECTIVE_FIELDS = ['revenue_weight', 'min_retention_floor', 'risk_lambda', 'objective_mode'];
const REQUIRED_SETTINGS = [...OBJECTIVE_FIELDS, 'max_breaks_per_hour'];
const NUMERIC_SETTINGS = ['revenue_weight', 'min_retention_floor', 'risk_lambda', 'max_breaks_per_hour'];

function settingsComplete(value) {
  return value && REQUIRED_SETTINGS.every((field) => value[field] !== null && value[field] !== undefined && value[field] !== '')
    && NUMERIC_SETTINGS.every((field) => Number.isFinite(Number(value[field])));
}

function objectiveOf(settings) {
  const out = {};
  OBJECTIVE_FIELDS.forEach((field) => { out[field] = settings?.[field]; });
  return out;
}

function sameObjective(a, b) {
  return OBJECTIVE_FIELDS.every((field) => String(a?.[field]) === String(b?.[field]));
}

export function usePlanSurface({ locale, notify, onGlobalRefresh, prepareCompare = false, optimizationAllowed = false, inventoryReadiness = null, initialFreshness = null, initialFreshnessPending = false }) {
  const [saved, setSaved] = useState(null);
  const [draft, setDraft] = useState({});
  const [saveState, setSaveState] = useState('idle');
  const [settingsState, setSettingsState] = useState('loading');
  const [settingsError, setSettingsError] = useState(null);
  const [adopted, setAdopted] = useState(null);

  const [runState, setRunState] = useState('idle');
  const [runProgress, setRunProgress] = useState(null);
  const [runResult, setRunResult] = useState(null);
  const [runError, setRunError] = useState(null);
  const [elapsed, setElapsed] = useState(0);

  const [legA, setLegA] = useState(null);
  const [legB, setLegB] = useState(null);
  const [compareState, setCompareState] = useState('idle');
  const [comparePayload, setComparePayload] = useState(null);
  const [compareError, setCompareError] = useState(null);
  const [compareWindow, setCompareWindow] = useState(null);
  const [compareDays, setCompareDays] = useState([]);
  // Which lever pair the last finished comparison ran, so the preparation never
  // re-runs a week the planner has already been shown.
  const [comparedKey, setComparedKey] = useState(null);
  const compareAbort = useRef(null);

  const [versions, setVersions] = useState([]);
  const [live, setLive] = useState(null);
  const [canPublish, setCanPublish] = useState(true);
  const [canPublishReason, setCanPublishReason] = useState(null);
  const [versionName, setVersionName] = useState('');
  const [versionNote, setVersionNote] = useState('');
  const [publishState, setPublishState] = useState('idle');
  const [publishError, setPublishError] = useState(null);
  const [selectedVersion, setSelectedVersion] = useState(null);
  const [diff, setDiff] = useState(null);

  const [progress, setProgress] = useState(null);
  const [yieldPerSecond, setYieldPerSecond] = useState(null);

  const {
    verdict: freshness, state: freshnessState, error: freshnessError, reload: reloadFreshness,
  } = usePlanFreshness(initialFreshness, initialFreshnessPending);

  const legsSeeded = useRef(false);

  const loadSettings = useCallback(async () => {
    setSettingsState('loading');
    setSettingsError(null);
    const result = await api.readSettings();
    const complete = result.ok && settingsComplete(result.data);
    if (!complete) {
      setSaved(null);
      setDraft({});
      setLegA(null);
      setLegB(null);
      legsSeeded.current = false;
      setSettingsState('error');
      setSettingsError(result.error || 'the server returned incomplete saved settings');
      return false;
    }
    setSaved(result.data);
    setDraft(objectiveOf(result.data));
    if (!legsSeeded.current) {
      legsSeeded.current = true;
      const base = {
        revenue_weight: result.data.revenue_weight,
        retention_floor: result.data.min_retention_floor,
        max_breaks_per_hour: result.data.max_breaks_per_hour,
        risk_lambda: result.data.risk_lambda,
        objective_mode: result.data.objective_mode,
      };
      setLegA({ ...base });
      setLegB({ ...base, retention_floor: Math.min(0.99, Number(base.retention_floor) + 0.08) });
    }
    setSettingsState('ready');
    return true;
  }, []);

  const loadVersions = useCallback(async () => {
    const result = await api.readPlanVersions();
    if (!result.ok || !result.data) return;
    setVersions(Array.isArray(result.data.versions) ? result.data.versions : []);
    setLive(result.data.live || null);
    setCanPublish(result.data.can_edit !== false);
    setCanPublishReason(result.data.can_edit_reason || null);
  }, []);

  const loadProgress = useCallback(async () => {
    const result = await api.readPlanProgress();
    setProgress(result.ok ? result.data : null);
  }, []);

  const loadYield = useCallback(async () => {
    const result = await api.readYieldPerSecond();
    setYieldPerSecond(result.ok ? result.data : null);
  }, []);

  useEffect(() => { loadSettings(); }, [loadSettings]);
  useEffect(() => { loadVersions(); }, [loadVersions]);
  useEffect(() => { loadProgress(); }, [loadProgress]);
  useEffect(() => { loadYield(); }, [loadYield]);

  useEffect(() => {
    if (runState !== 'running' && compareState !== 'running') {
      setElapsed(0);
      return undefined;
    }
    const started = Date.now();
    setElapsed(0);
    const id = window.setInterval(() => setElapsed(Math.floor((Date.now() - started) / 1000)), 1000);
    return () => window.clearInterval(id);
  }, [runState, compareState]);

  const changeDraft = useCallback((field, value) => {
    if (!OBJECTIVE_FIELDS.includes(field)) return;
    setDraft((current) => ({ ...current, [field]: value }));
    setSaveState('idle');
    setAdopted(null);
  }, []);

  const applyTemplate = useCallback((values) => {
    const patch = {};
    OBJECTIVE_FIELDS.forEach((field) => { if (field in values) patch[field] = values[field]; });
    setDraft((current) => ({ ...current, ...patch }));
    setSaveState('idle');
    setAdopted(null);
  }, []);

  // Take the winning leg of the comparison and make it the objective.
  //
  // The values written are the ones the server says that leg actually ran under,
  // not the ones the form happens to be showing now, so the objective that
  // arrives on step 1 is the objective the money on that card was computed on.
  // Nothing is saved: the draft moves, the unsaved-changes banner prints each
  // old value beside its new one, and the planner still saves and runs.
  const adoptLeg = useCallback((leg) => {
    const summary = leg === 'b' ? comparePayload?.b : comparePayload?.a;
    const values = objectiveFromLevers(summary?.levers);
    if (!values) return false;
    setDraft((current) => ({ ...current, ...values }));
    setSaveState('idle');
    setAdopted(String(leg).toLowerCase());
    notify?.(
      'The objective now holds that scenario. It is not saved, and it is not in the plan until the plan is run.',
      'המטרה מחזיקה כעת את התרחיש הזה. היא אינה שמורה, ואינה בתוכנית עד שהתוכנית רצה.',
    );
    return true;
  }, [comparePayload, notify]);

  // Which legs can be adopted right now, so the palette refuses with a reason
  // rather than offering a control that does nothing.
  const adoptable = useCallback(
    (leg) => Boolean(objectiveFromLevers((leg === 'b' ? comparePayload?.b : comparePayload?.a)?.levers)),
    [comparePayload],
  );

  // The inverse of every draft change, adoption included. A change that cannot
  // be put back is not a safe one to offer in one keystroke.
  const revertDraft = useCallback(() => {
    if (!saved) return;
    setDraft(objectiveOf(saved));
    setSaveState('idle');
    setAdopted(null);
  }, [saved]);

  const saveObjective = useCallback(async () => {
    if (!saved || settingsState !== 'ready') return;
    setSaveState('saving');
    // The settings endpoint takes the whole model and defaults anything the
    // body omits, so a partial write silently clears fields this surface does
    // not own. Measured on the running app: a body carrying one lever wiped
    // operator_channel and pricing_overrides, and an empty operator channel
    // un-scopes every money figure in the product from the operator's plan to
    // the whole market. So the payload is always the saved model with the four
    // objective levers on top, and the round trip is checked below rather than
    // assumed.
    const payload = { ...saved, ...draft };
    const result = await api.saveSettings(payload);
    if (!result.ok) {
      setSaveState('error');
      return;
    }
    if (String(result.data?.operator_channel || '') !== String(saved.operator_channel || '')) {
      // Refuse to report success on a save that moved the channel scope. The
      // planner is told, the surface reloads what is really on disk, and no
      // figure is redrawn against a scope nobody chose.
      setSaveState('error');
      notify?.(
        'The objective was not saved: the write would have changed which channel the plan is scoped to.',
        'המטרה לא נשמרה: הכתיבה הייתה משנה לאיזה ערוץ התוכנית משויכת.',
      );
      loadSettings();
      return;
    }
    setSaved(result.data);
    setDraft(objectiveOf(result.data));
    setSaveState('saved');
    setAdopted(null);
    notify?.(
      'The plan objective was saved. It is not in the plan until the plan is run.',
      'מטרת התוכנית נשמרה. היא אינה בתוכנית עד שהתוכנית רצה.',
    );
    onGlobalRefresh?.();
    // The saved objective is now an input the plan was not built from.
    reloadFreshness();
    window.setTimeout(() => setSaveState('idle'), 2000);
  }, [saved, settingsState, draft, notify, onGlobalRefresh, loadSettings, reloadFreshness]);

  const runPlan = useCallback(async () => {
    if (settingsState !== 'ready' || !saved || !optimizationAllowed) {
      setRunState('error');
      setRunError('saved settings and optimizer inventory must be verified before a run');
      return false;
    }
    setRunState('running');
    setRunError(null);
    setRunProgress(null);
    const result = await api.runWeeklyPlan({ onProgress: setRunProgress });
    setRunProgress(null);
    if (!result.ok) {
      setRunState('error');
      setRunError(result.error);
      notify?.('The run failed and the saved plan is unchanged.', 'ההרצה נכשלה והתוכנית השמורה לא השתנתה.');
      return false;
    }
    setRunResult(result.data);
    const owned = result.data?.owned;
    const zeroBreaks = announceRunResult(owned, notify);
    setRunState(zeroBreaks ? 'warning' : 'done');
    loadVersions();
    // The plan moved, so the figures that stand on it and its state moved too.
    loadProgress();
    loadYield();
    reloadFreshness();
    onGlobalRefresh?.();
    window.setTimeout(() => setRunState('idle'), 2400);
    return true;
  }, [settingsState, saved, optimizationAllowed, notify, onGlobalRefresh, loadVersions, loadProgress, loadYield, reloadFreshness]);

  const changeLeg = useCallback((leg, field, value) => {
    if (field === 'max_breaks_per_hour') return;
    const setter = leg === 'a' ? setLegA : setLegB;
    setter((current) => ({ ...(current || {}), [field]: value }));
  }, []);

  const settle = useCallback((data, key) => {
    if (!data || data.available === false) {
      setCompareState('unavailable');
      setComparePayload(data || null);
      return;
    }
    setComparePayload(data);
    setCompareDays(Array.isArray(data.by_day) ? data.by_day : []);
    setCompareState('ready');
    setComparedKey(key || null);
  }, []);

  const guardedLegA = legA && saved ? { ...legA, max_breaks_per_hour: saved.max_breaks_per_hour } : null;
  const guardedLegB = legB && saved ? { ...legB, max_breaks_per_hour: saved.max_breaks_per_hour } : null;

  const compare = useCallback(async (checkedInventory) => {
    if (!guardedLegA || !guardedLegB || settingsState !== 'ready' || !optimizationAllowed || checkedInventory?.status !== 'ready') return false;
    setCompareState('running');
    setCompareError(null);
    setCompareDays([]);
    setCompareWindow(null);
    setComparedKey(null);
    // The same body the preparation sends, built by the same function, so a
    // prepared week is the week this asks for rather than a near miss.
    const body = compareRequestBody(guardedLegA, guardedLegB);
    const key = comparePreparationKey(guardedLegA, guardedLegB, checkedInventory);
    const controller = new AbortController();
    compareAbort.current = controller;
    try {
      // Fourteen real optimizations over the plan's own week, streamed: each day
      // is on screen the moment both its legs are decided.
      const data = await streamCompare(body, {
        signal: controller.signal,
        onWindow: setCompareWindow,
        onDay: (event) => setCompareDays((current) => [...current, { ...event.day, elapsed_ms: event.elapsed_ms }]),
      });
      settle(data, key);
    } catch (error) {
      if (controller.signal.aborted) {
        setCompareState('idle');
        return;
      }
      // The stream is the fast path, not the only path. A transport that cannot
      // carry it falls back to the plain route, which returns the same week.
      const result = await verifiedCompareFallback(inventoryReadiness, checkedInventory, () => api.compareScenarios(body));
      if (!result.ok) {
        setCompareState('error');
        setCompareError(result.error || error.message);
        return;
      }
      settle(result.data, key);
    } finally {
      compareAbort.current = null;
    }
    return true;
  }, [guardedLegA, guardedLegB, settingsState, optimizationAllowed, settle]);

  const cancelCompare = useCallback(() => {
    compareAbort.current?.abort();
  }, []);

  // The wait, taken before the planner asks for it. It runs only while step
  // three is open, only once the levers have settled, and it is abandoned the
  // moment they change again.
  const prepare = useComparePrepare({
    legA: guardedLegA,
    legB: guardedLegB,
    enabled: prepareCompare && settingsState === 'ready' && optimizationAllowed,
    busy: compareState === 'running',
    settledKey: comparedKey,
    inventory: inventoryReadiness,
  });

  const publish = useCallback(async (confirmCollapse = false) => {
    if (!versionName.trim()) return;
    setPublishState('running');
    setPublishError(null);
    const result = await api.publishPlanVersion(versionName.trim(), versionNote.trim(), confirmCollapse);
    if (!result.ok) {
      setPublishState('error');
      setPublishError(result.error);
      return;
    }
    setPublishState('done');
    setVersionName('');
    setVersionNote('');
    notify?.(
      `Plan version frozen: ${result.data?.name}.`,
      `גרסת התוכנית הוקפאה: ${result.data?.name}.`,
    );
    await loadVersions();
    window.setTimeout(() => setPublishState('idle'), 2000);
  }, [versionName, versionNote, notify, loadVersions]);

  const loadDiff = useCallback(async (versionId, against) => {
    setSelectedVersion(versionId);
    const result = await api.readPlanVersionDiff(versionId, against);
    setDiff(result.data || { available: false, reason: result.error });
  }, []);

  const restore = useCallback(async (version) => {
    const result = await api.restorePlanVersion(version.version_id);
    if (!result.ok) {
      setPublishError(result.error);
      return;
    }
    notify?.(
      `The live plan is now "${version.name}".`,
      `התוכנית החיה היא כעת "${version.name}".`,
    );
    await loadVersions();
    loadProgress();
    loadYield();
    reloadFreshness();
    onGlobalRefresh?.();
  }, [notify, loadVersions, loadProgress, loadYield, reloadFreshness, onGlobalRefresh]);

  return {
    saved,
    draft,
    settingsState,
    settingsError,
    settingsReady: settingsState === 'ready' && Boolean(saved),
    retrySettings: loadSettings,
    dirty: Boolean(saved) && !sameObjective(draft, saved),
    saveState,
    adopted,
    changeDraft,
    applyTemplate,
    adoptLeg,
    adoptable,
    revertDraft,
    saveObjective,
    runState,
    runProgress,
    runResult,
    runError,
    elapsed,
    runPlan,
    legA: guardedLegA || {},
    legB: guardedLegB || {},
    compareState,
    comparePayload,
    compareError,
    compareWindow,
    compareDays,
    comparePrepared: prepare.phase,
    changeLeg,
    compare,
    cancelCompare,
    versions,
    live,
    canPublish,
    canPublishReason,
    versionName,
    setVersionName,
    versionNote,
    setVersionNote,
    publishState,
    publishError,
    publish,
    selectedVersion,
    setSelectedVersion,
    diff,
    loadDiff,
    restore,
    progress,
    reloadProgress: loadProgress,
    yieldPerSecond,
    freshness,
    freshnessState,
    freshnessError,
    reloadFreshness,
  };
}

export default usePlanSurface;
