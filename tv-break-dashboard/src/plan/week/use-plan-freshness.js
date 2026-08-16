import { useCallback, useEffect, useState } from 'react';
import { readPlanFreshness } from './plan-week-api';

// Is the saved plan still in step with the inputs it was built from.
//
// The shell's overview verdict seeds the first load, avoiding a duplicate read.
// After a write, or when no trustworthy seed exists, Plan owns the revalidation
// beside its settings, versions, progress and yield. Before that split, a run
// could finish while the header kept the shell's pre-run verdict for the rest of
// the visit. A planner reading that would start a two-minute job again for
// nothing.
// Shell ownership stops at that first honest seed.
//
// Three states, and each is the honest name of what is known. Ready carries the
// server's own fresh, stale or unknown verdict. Unavailable means the read
// failed and it carries the reason. Loading means nobody knows yet, and it is
// never drawn as unknown, because "no run stamp was found" is a claim about the
// plan and during a read in flight it would be a false one.

function isFreshnessVerdict(value) {
  return Boolean(value && typeof value === 'object' && typeof value.status === 'string');
}

export function usePlanFreshness(initialVerdict = null, initialPending = false) {
  const initialReady = isFreshnessVerdict(initialVerdict);
  const [verdict, setVerdict] = useState(initialReady ? initialVerdict : null);
  const [state, setState] = useState(initialReady ? 'ready' : 'loading');
  const [error, setError] = useState(null);

  const reload = useCallback(async () => {
    const result = await readPlanFreshness();
    if (!result.ok || !result.data) {
      setVerdict(null);
      setError(result.error || null);
      setState('unavailable');
      return;
    }
    setVerdict(result.data);
    setError(null);
    setState('ready');
  }, []);

  useEffect(() => {
    if (isFreshnessVerdict(initialVerdict)) {
      setVerdict(initialVerdict);
      setError(null);
      setState('ready');
      return;
    }
    if (!initialPending) reload();
  }, [initialVerdict, initialPending, reload]);

  return { verdict, state, error, reload };
}

export default usePlanFreshness;
