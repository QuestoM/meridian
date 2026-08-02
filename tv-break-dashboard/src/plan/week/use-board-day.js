import { useEffect, useRef, useState } from 'react';
import { readSchedule } from './plan-week-api';

// One broadcast day of the week board, fetched when a day is asked for.
//
// The comparison names the day the choice turns on, so that day has to open,
// and the day it opens on has to be that day. The embedded board the week
// payload carries is one day of the operator's own programmes, chosen by the
// source order rather than by anybody, so a day is asked for by date and the
// route answers with that date or says why it cannot.
//
// Nothing here interpolates. While a day is in flight the state is loading and
// the surface says so; a day the programme source does not carry comes back
// unavailable with its reason, and the surface prints the reason instead of a
// neighbouring day's programmes.

export function useBoardDay(date) {
  const [payload, setPayload] = useState(null);
  const [state, setState] = useState('idle');
  const [error, setError] = useState(null);
  const wanted = useRef(null);

  useEffect(() => {
    const key = String(date || '').trim();
    wanted.current = key;
    if (!key) {
      setPayload(null);
      setState('idle');
      setError(null);
      return;
    }
    setState('loading');
    setError(null);
    readSchedule(key).then((result) => {
      // A slower answer for a day nobody is looking at any more never lands.
      if (wanted.current !== key) return;
      if (!result.ok) {
        setPayload(null);
        setState('error');
        setError(result.error);
        return;
      }
      setPayload(result.data);
      setState('ready');
    });
  }, [date]);

  return { payload, state, error };
}

export default useBoardDay;
