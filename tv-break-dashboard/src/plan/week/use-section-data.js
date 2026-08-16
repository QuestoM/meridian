import { useCallback, useEffect, useState } from 'react';

function isUsable(value) {
  return Boolean(value) && value._unavailable !== true;
}

// Lazy section data remains route-local, but a shell fallback is evidence that
// the source was unavailable, not an empty business result. Entering the section
// makes one fresh attempt; an error keeps no stale/fallback data and exposes the
// same retry to its canonical error state.
export function useSectionData(provided, load, wanted) {
  const providedReady = isUsable(provided);
  const [snapshot, setSnapshot] = useState({ data: null, state: providedReady ? 'ready' : 'idle', error: null });
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    if (providedReady) return undefined;
    if (!wanted) return undefined;
    let active = true;
    setSnapshot({ data: null, state: 'loading', error: null });
    Promise.resolve(load()).then((result) => {
      if (!active) return;
      if (result?.ok && isUsable(result.data)) {
        setSnapshot({ data: result.data, state: 'ready', error: null });
      } else {
        setSnapshot({
          data: null,
          state: 'error',
          error: result?.error || 'the source returned no usable data',
        });
      }
    }).catch((error) => {
      if (active) setSnapshot({ data: null, state: 'error', error: error?.message || 'the source could not be read' });
    });
    return () => { active = false; };
  }, [providedReady, wanted, load, attempt]);

  const retry = useCallback(() => setAttempt((current) => current + 1), []);
  const fetchedState = snapshot.state === 'ready' && !isUsable(snapshot.data) ? 'idle' : snapshot.state;
  return {
    data: providedReady ? provided : snapshot.data,
    state: providedReady ? 'ready' : fetchedState,
    error: providedReady ? null : snapshot.error,
    retry,
  };
}

export default useSectionData;
