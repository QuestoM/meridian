import { useCallback, useEffect, useRef, useState } from 'react';
import { pageText } from '../../shell/format';
import * as api from './plan-week-api';
import { objectiveFromLevers } from './plan-week-model';
import { streamCompare } from './plan-week-compare-stream';
import { compareKey, compareRequestBody, useComparePrepare } from './use-compare-prepare';
import { usePlanFreshness } from './use-plan-freshness';

// Every piece of state Plan, the week owns, and every call it makes.
//
// The destination fetches its own settings, plan versions, progress, yield, plan
// state and comparisons rather than taking any of them from the shell, because
// the frozen shell router hands each of the four entrances a different prop set
// and a destination that behaved differently depending on the door somebody came
// through would not be one destination. After every write it refreshes its own
// reads and asks the shell to refresh too, when the entrance gave it a way to.
//
// Nothing here invents a value. A failed call sets an error string that the
// panel renders as an honest state, and every figure a panel prints came back
// from the server on this session.

const OBJECTIVE_FIELDS = ['revenue_weight', 'min_retention_floor', 'max_breaks_per_hour', 'risk_lambda', 'objective_mode'];

function objectiveOf(settings) {
  const out = {};
  OBJECTIVE_FIELDS.forEach((field) => { out[field] = settings?.[field]; });
  return out;
}

function sameObjective(a, b) {
  return OBJECTIVE_FIELDS.every((field) => String(a?.[field]) === String(b?.[field]));
}

export function usePlanSurface({ locale, notify, onGlobalRefresh, prepareCompare = false }) {
  const [saved, setSaved] = useState(null);
  const [draft, setDraft] = useState({});
  const [saveState, setSaveState] = useState('idle');
  // Which comparison leg the draft was taken from, when it was taken from one.
  // Provenance, never a figure: it lets the banner name where the values came
  // from and is cleared the moment anything else touches the draft.
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
  // The comparison arrives one broadcast day at a time, so the window it is
  // running over and the days already decided are their own state: the panel
  // shows a real partial week rather than a spinner over an unknown wait.
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

  // The plan's own state, read here rather than taken from the entrance, and
  // read again after every act below that can move it.
  const {
    verdict: freshness, state: freshnessState, error: freshnessError, reload: reloadFreshness,
  } = usePlanFreshness();

  const legsSeeded = useRef(false);

  const loadSettings = useCallback(async () => {
    const result = await api.readSettings();
    if (!result.ok || !result.data) return;
    setSaved(result.data);
    setDraft((current) => (Object.keys(current).length ? current : objectiveOf(result.data)));
    if (!legsSeeded.current) {
      legsSeeded.current = true;
      // Both legs open on the saved decision, so a comparison starts from what
      // the plan actually is and the planner changes one thing.
      const base = {
        revenue_weight: result.data.revenue_weight,
        retention_floor: result.data.min_retention_floor,
        max_breaks_per_hour: result.data.max_breaks_per_hour,
        risk_lambda: result.data.risk_lambda,
        objective_mode: result.data.objective_mode || 'blend',
      };
      setLegA({ ...base });
      // The floor is the lever measurement says moves the plan, so leg B opens
      // one step above it rather than on a revenue weight that would return the
      // identical plan.
      setLegB({ ...base, retention_floor: Math.min(0.99, Number(base.retention_floor || 0.72) + 0.08) });
    }
  }, []);

  const loadVersions = useCallback(async () => {
    const result = await api.readPlanVersions();
    if (!result.ok || !result.data) return;
    setVersions(Array.isArray(result.data.versions) ? result.data.versions : []);
    setLive(result.data.live || null);
    setCanPublish(result.data.can_edit !== false);
    setCanPublishReason(result.data.can_edit_reason || null);
  }, []);

  // The goal strip and the worth of a second are read on entry, because both
  // are answers the planner reads before touching anything. A failed read
  // leaves them null, which the panels render as an honest absence.
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
    setDraft((current) => ({ ...current, [field]: value }));
    setSaveState('idle');
    setAdopted(null);
  }, []);

  const applyTemplate = useCallback((values) => {
    setDraft((current) => ({ ...current, ...values }));
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
    if (!saved) return;
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
  }, [saved, draft, notify, onGlobalRefresh, loadSettings, reloadFreshness]);

  const runPlan = useCallback(async () => {
    setRunState('running');
    setRunError(null);
    setRunProgress(null);
    const result = await api.runWeeklyPlan({ onProgress: setRunProgress });
    setRunProgress(null);
    if (!result.ok) {
      setRunState('error');
      setRunError(result.error);
      notify?.('The run failed and the saved plan is unchanged.', 'ההרצה נכשלה והתוכנית השמורה לא השתנתה.');
      return;
    }
    setRunResult(result.data);
    setRunState('done');
    const owned = result.data?.owned;
    notify?.(
      `The weekly plan was run: ${owned?.total_breaks ?? '-'} breaks on your channel.`,
      `התוכנית השבועית רצה: ${owned?.total_breaks ?? '-'} ברייקים בערוץ שלכם.`,
    );
    loadVersions();
    // The plan moved, so the figures that stand on it and its state moved too.
    loadProgress();
    loadYield();
    reloadFreshness();
    onGlobalRefresh?.();
    window.setTimeout(() => setRunState('idle'), 2400);
  }, [notify, onGlobalRefresh, loadVersions, loadProgress, loadYield, reloadFreshness]);

  const changeLeg = useCallback((leg, field, value) => {
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

  const compare = useCallback(async () => {
    if (!legA || !legB) return;
    setCompareState('running');
    setCompareError(null);
    setCompareDays([]);
    setCompareWindow(null);
    setComparedKey(null);
    // The same body the preparation sends, built by the same function, so a
    // prepared week is the week this asks for rather than a near miss.
    const body = compareRequestBody(legA, legB);
    const key = compareKey(legA, legB);
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
      const result = await api.compareScenarios(body);
      if (!result.ok) {
        setCompareState('error');
        setCompareError(result.error || error.message);
        return;
      }
      settle(result.data, key);
    } finally {
      compareAbort.current = null;
    }
  }, [legA, legB, settle]);

  const cancelCompare = useCallback(() => {
    compareAbort.current?.abort();
  }, []);

  // The wait, taken before the planner asks for it. It runs only while step
  // three is open, only once the levers have settled, and it is abandoned the
  // moment they change again.
  const prepare = useComparePrepare({
    legA,
    legB,
    enabled: prepareCompare,
    busy: compareState === 'running',
    settledKey: comparedKey,
  });

  const publish = useCallback(async () => {
    if (!versionName.trim()) return;
    setPublishState('running');
    setPublishError(null);
    const result = await api.publishPlanVersion(versionName.trim(), versionNote.trim());
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

  const loadDiff = useCallback(async (versionId) => {
    setSelectedVersion(versionId);
    const result = await api.readPlanVersionDiff(versionId);
    setDiff(result.data || { available: false, reason: result.error });
  }, []);

  const restore = useCallback(async (version) => {
    const question = pageText(
      locale,
      `Roll the live plan back to "${version.name}". The plan on disk now is frozen first, so this is reversible. Continue.`,
      `החזרת התוכנית החיה ל"${version.name}". התוכנית שעל הדיסק כרגע תוקפא קודם, כך שאפשר לחזור אחורה. להמשיך.`,
    );
    if (!window.confirm(question)) return;
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
  }, [locale, notify, loadVersions, loadProgress, loadYield, reloadFreshness, onGlobalRefresh]);

  return {
    saved,
    draft,
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
    legA: legA || {},
    legB: legB || {},
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
