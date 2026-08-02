import { useCallback, useEffect, useState } from 'react';
import { readPlanFreshness } from './plan-week-api';

// Is the saved plan still in step with the inputs it was built from.
//
// The destination reads this for itself, beside its settings, its plan versions,
// its progress and its yield, rather than taking it from the shell. Measured
// before it did: a run finished from the Optimizer entrance and the header still
// read out of date for as long as the page stayed open, because only one of the
// four entrances is handed a refresh handler and the shell's copy of the verdict
// had been taken before the run. A planner reading that would start a two-minute
// job again for nothing.
//
// Three states, and each is the honest name of what is known. Ready carries the
// server's own fresh, stale or unknown verdict. Unavailable means the read
// failed and it carries the reason. Loading means nobody knows yet, and it is
// never drawn as unknown, because "no run stamp was found" is a claim about the
// plan and during a read in flight it would be a false one.

export function usePlanFreshness() {
  const [verdict, setVerdict] = useState(null);
  const [state, setState] = useState('loading');
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

  useEffect(() => { reload(); }, [reload]);

  return { verdict, state, error, reload };
}

export default usePlanFreshness;
