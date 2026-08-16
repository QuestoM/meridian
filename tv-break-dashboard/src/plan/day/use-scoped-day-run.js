import { useCallback, useRef, useState } from 'react';
import { API_BASE } from '../../shell/api';
import { readFiles, readSettings } from '../week/plan-week-api';
import { inventoryReadinessFromFiles } from '../week/use-plan-inventory-readiness';

const REQUIRED_SETTINGS = [
  'revenue_weight',
  'min_retention_floor',
  'risk_lambda',
  'objective_mode',
  'max_breaks_per_hour',
];

const INITIAL_SAFETY = {
  status: 'idle',
  code: 'unchecked',
  error: null,
  settingsSignature: null,
  inventory: null,
};

function settingsCheck(result) {
  const value = result?.data;
  const complete = result?.ok
    && value
    && REQUIRED_SETTINGS.every((field) => value[field] !== null && value[field] !== undefined && value[field] !== '')
    && ['revenue_weight', 'min_retention_floor', 'risk_lambda', 'max_breaks_per_hour']
      .every((field) => Number.isFinite(Number(value[field])));
  if (!complete) {
    return {
      ok: false,
      error: result?.error || 'the server returned incomplete saved settings',
      signature: null,
    };
  }
  return {
    ok: true,
    error: null,
    signature: REQUIRED_SETTINGS.map((field) => `${field}:${JSON.stringify(value[field])}`).join('|'),
  };
}

export function sameDayRunInputs(reviewed, checked) {
  const before = reviewed?.inventory;
  const after = checked?.inventory;
  return reviewed?.settingsSignature === checked?.settingsSignature
    && before?.mode === after?.mode
    && Number(before?.slots) === Number(after?.slots)
    && String(before?.path) === String(after?.path)
    && String(before?.signature) === String(after?.signature);
}

export function useDayRunPreflight() {
  const [state, setState] = useState(INITIAL_SAFETY);
  const requestId = useRef(0);

  const verify = useCallback(async () => {
    const id = requestId.current + 1;
    requestId.current = id;
    setState((current) => ({ ...current, status: 'checking', code: 'checking', error: null }));
    let next;
    try {
      const [settingsResult, filesResult] = await Promise.all([readSettings(), readFiles()]);
      const settings = settingsCheck(settingsResult);
      if (!settings.ok) {
        next = { ...INITIAL_SAFETY, status: 'error', code: 'settings', error: settings.error };
      } else if (!filesResult.ok) {
        next = {
          ...INITIAL_SAFETY,
          status: 'error',
          code: 'inventory',
          error: filesResult.error || 'the optimizer inventory audit could not be read',
        };
      } else {
        const inventory = inventoryReadinessFromFiles(filesResult.data);
        next = inventory.status === 'ready'
          ? {
              status: 'ready',
              code: 'ready',
              error: null,
              settingsSignature: settings.signature,
              inventory,
            }
          : {
              ...INITIAL_SAFETY,
              status: inventory.status === 'blocked' ? 'blocked' : 'error',
              code: inventory.code,
              error: inventory.error,
              settingsSignature: settings.signature,
              inventory,
            };
      }
    } catch (error) {
      next = { ...INITIAL_SAFETY, status: 'error', code: 'unverified', error: error?.message || 'the run inputs could not be verified' };
    }
    if (requestId.current === id) setState(next);
    return requestId.current === id ? next : { ...INITIAL_SAFETY, status: 'error', code: 'superseded', error: 'a newer input check replaced this one' };
  }, []);

  const markChanged = useCallback(() => {
    requestId.current += 1;
    setState({
      ...INITIAL_SAFETY,
      status: 'error',
      code: 'changed',
      error: 'saved settings or optimizer inventory changed after the review',
    });
  }, []);

  return { ...state, verify, retry: verify, markChanged };
}

export function useScopedDayRun({ scope, runner, locale, notify, onDone, success }) {
  const safety = useDayRunPreflight();
  const [review, setReview] = useState(null);
  const activeReview = useRef(null);
  const [jobState, setJobState] = useState('idle');
  const current = useRef({ scope, runner, locale, notify, onDone, success });
  current.current = { scope, runner, locale, notify, onDone, success };

  const requestReview = useCallback(async () => {
    const wanted = current.current.scope;
    if (!Array.isArray(wanted) || wanted.length === 0 || jobState === 'running') return false;
    const checked = await safety.verify();
    if (checked.status !== 'ready') return false;
    const action = { scope: wanted.map((entry) => ({ ...entry })), inputs: checked };
    activeReview.current = action;
    setReview(action);
    return true;
  }, [jobState, safety.verify]);

  const confirmReview = useCallback(async () => {
    const action = activeReview.current;
    if (!action || jobState === 'running') return false;
    const checked = await safety.verify();
    // Escape and Cancel are real cancellation, including while this authority
    // check is in flight; closing the dialog cannot leave a detached write.
    if (activeReview.current !== action) return false;
    if (checked.status !== 'ready') {
      activeReview.current = null;
      setReview(null);
      return false;
    }
    if (!sameDayRunInputs(action.inputs, checked)) {
      activeReview.current = null;
      setReview(null);
      safety.markChanged();
      return false;
    }
    activeReview.current = null;
    setReview(null);
    setJobState('running');
    try {
      const result = await current.current.runner(API_BASE, action.scope);
      if (result.status === 'done') {
        await current.current.onDone?.();
        current.current.notify?.(current.current.success?.en, current.current.success?.he);
        return true;
      }
      if (result.status === 'missing') {
        current.current.notify?.(
          'Running one day needs the updated backend. Run the whole plan instead.',
          'הרצת יום בודד דורשת שרת מעודכן. הריצו את התוכנית כולה במקום.',
        );
        return false;
      }
      const reason = result.error || (result.status === 'timeout' ? 'timed out' : 'unknown error');
      current.current.notify?.(`The day run failed: ${reason}.`, `הרצת היום נכשלה: ${reason}.`);
      return false;
    } catch (error) {
      current.current.notify?.(`The day run failed (${error.message}).`, `הרצת היום נכשלה (${error.message}).`);
      return false;
    } finally {
      setJobState('idle');
    }
  }, [jobState, safety.verify, safety.markChanged]);

  const cancelReview = useCallback(() => {
    activeReview.current = null;
    setReview(null);
  }, []);

  return {
    safety,
    review,
    jobState,
    requestReview,
    confirmReview,
    cancelReview,
  };
}

export default useScopedDayRun;
