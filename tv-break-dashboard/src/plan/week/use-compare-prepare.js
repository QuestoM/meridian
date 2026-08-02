import { useEffect, useRef, useState } from 'react';
import { streamCompare } from './plan-week-compare-stream';

// Step three's wait, spent before the planner asks rather than after.
//
// The comparison is fourteen real optimizations, two legs over the plan's own
// seven broadcast days, and measured on the reference data that is 12.6 s cold:
// 0.9 s of refined optimizer per day per leg, which is the engine and not the
// route. Two cheaper routes were measured and refused. Threads do not help,
// because the work is GIL bound: six day runs took 4.18 s sequentially and 4.20 s
// across six workers. Reading leg A off the committed plan instead of running it
// would halve the work, but nothing on disk records which levers produced the
// saved plan, so the figure would be a guess about its own provenance.
//
// What is left is to do the work while the planner is still setting the
// scenarios up. A run is deterministic and cached on (day, levers) against the
// plan's own file signature, so a comparison prepared here is the same body the
// comparison returns: measured, the same request twice gave a byte-identical
// payload and the second took 22 ms. Nothing is displayed from a preparation,
// and the panel says plainly that it is happening, because an interface that
// spends a machine's time silently is not being honest about what it costs.
//
// It runs only while step three is open and only after the levers have settled,
// and it is abandoned the moment they change again.

const SETTLE_MS = 900;

const LEVERS = ['revenue_weight', 'retention_floor', 'max_breaks_per_hour', 'risk_lambda', 'objective_mode'];

// The body both the preparation and the comparison send. One builder, because
// two bodies that drift by one field would prepare a week the comparison then
// runs again from scratch, and the planner would wait exactly as long while the
// panel claimed otherwise.
export function compareRequestBody(legA, legB) {
  return {
    weight_a: Math.round(Number(legA?.revenue_weight) || 0),
    weight_b: Math.round(Number(legB?.revenue_weight) || 0),
    a: legA,
    b: legB,
  };
}

export function compareKey(legA, legB) {
  if (!legA || !legB) return null;
  return [legA, legB].map((leg) => LEVERS.map((field) => String(leg?.[field] ?? '')).join('|')).join('||');
}

export function useComparePrepare({ legA, legB, enabled, busy, settledKey }) {
  const [phase, setPhase] = useState('idle');
  const abort = useRef(null);
  const ready = useRef(new Set());
  const legs = useRef({ legA, legB });
  legs.current = { legA, legB };
  const key = compareKey(legA, legB);

  // A comparison the planner ran is a comparison already computed, so it counts
  // as prepared and no second run is started for it.
  useEffect(() => {
    if (settledKey) ready.current.add(settledKey);
  }, [settledKey]);

  useEffect(() => {
    if (!enabled || busy || !key) {
      setPhase('idle');
      return undefined;
    }
    if (ready.current.has(key)) {
      setPhase('ready');
      return undefined;
    }
    setPhase('idle');
    let live = true;
    const timer = window.setTimeout(() => {
      const controller = new AbortController();
      abort.current = controller;
      setPhase('preparing');
      streamCompare(compareRequestBody(legs.current.legA, legs.current.legB), { signal: controller.signal })
        .then((payload) => {
          if (!live) return;
          if (payload && payload.available !== false) {
            ready.current.add(key);
            setPhase('ready');
          } else {
            setPhase('idle');
          }
        })
        .catch(() => {
          // A preparation that fails costs the planner nothing and says nothing.
          // The comparison itself reports any real failure when it is asked for.
          if (live) setPhase('idle');
        })
        .finally(() => {
          abort.current = null;
        });
    }, SETTLE_MS);

    return () => {
      live = false;
      window.clearTimeout(timer);
      abort.current?.abort();
      abort.current = null;
    };
  }, [key, enabled, busy]);

  return { phase, key };
}

export default useComparePrepare;
