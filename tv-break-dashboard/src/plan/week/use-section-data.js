import { useEffect, useRef, useState } from 'react';

// Data a section needs, taken from the entrance when it was handed one and
// fetched once when it was not.
//
// The frozen shell router hands each of the four entrances a different prop set:
// Inventory receives the supply payload but not the plan, Forecasts receives
// neither, and only Schedule receives the calendar overlay. So a section asks
// for what it needs and this hook answers from the prop when there is one.
//
// The fetch is deliberately lazy, and that is a latency decision rather than a
// convenience. The plan payload measured 516,470 bytes; a planner who opened
// this destination to compare two scenarios should never wait for it, and with
// this hook they do not, because the board section is the only thing that asks.

export function useSectionData(provided, load, wanted) {
  const [fetched, setFetched] = useState(null);
  const [state, setState] = useState(provided ? 'ready' : 'idle');
  const started = useRef(false);

  useEffect(() => {
    if (provided || !wanted || started.current) return;
    started.current = true;
    setState('loading');
    load().then((result) => {
      if (result.ok) {
        setFetched(result.data);
        setState('ready');
      } else {
        setState('error');
      }
    });
  }, [provided, wanted, load]);

  return { data: provided || fetched, state: provided ? 'ready' : state };
}

export default useSectionData;
